# 权限与沙箱

VeriForge 让 Agent 直接在你的仓库里执行命令，所以 Shell 的安全边界是运行时里最重要的部分。本文说明一条命令从模型输出到真正执行，要经过哪些检查。

## 四层边界

```text
命令文本 → ① 语法解析 → ② 提取事实 → ③ 权限决定 → ④ 沙箱执行
```

四层互相补充，不能相互替代：语法分析只知道命令长什么样，审批只判断意图是否可接受，只有沙箱能兜住静态分析看不见的行为（任意脚本、未建模的写入参数、时间竞态）。

### ① 语法解析

`runtime/shell_parser.py` 把命令解析为结构化结果（命令列表、重定向、是否含动态内容、是否完整解析），绝不执行命令。

- Windows 使用 PowerShell 7 自带的 `Parser.ParseInput`，由一个常驻子进程通过 JSONL 提供服务；
- 其他平台使用 tree-sitter-bash。

每个参数都带有「字面量 / 运行时求值」标记，后续分析据此判断，而不是靠字符串里有没有 `$` 来猜。解析器不可用、协议出错或超出解析预算时，该命令直接被拒绝。

### ② 提取事实

`runtime/shell_classification.py` 只产出事实，不做决定。命令按规则表分派给少数几个专用分析器：

- 文件的创建、复制、移动、删除；
- Git（全局参数和目标仓库）；
- 包管理器（查询、安装、自定义脚本）；
- PowerShell Cmdlet（按 `-Path`、`-ItemType`、`-Value` 等参数角色分析）；
- 解释器和网络命令。

产出的事实包括：读写删的目标路径及其作用域（工作区、临时目录、系统目录…），以及 `UNKNOWN_EFFECT`、`NETWORK_UPLOAD`、`DESTRUCTIVE`、`CATASTROPHIC` 等特征。`npm run deploy`、`python script.py` 这类命令被视为「执行未知代码」，而不是只读。

### ③ 权限决定

`runtime/permissions.py` 为每个事实生成一个候选动作（allow / ask / deny），**最严格的胜出**：

| 情况 | 决定 |
| --- | --- |
| 只读命令、可证明的工作区内写入、验证命令（测试、lint 等） | allow |
| 未知程序 | allow，由沙箱兜底 |
| 网络访问、破坏性 Git（如 `reset --hard`）、大范围删除、申请额外权限 | ask |
| 写入工作区外 | deny，除非同一次调用显式申请并获批（见下文） |
| 系统 / 磁盘级破坏（`rm -rf /`、`mkfs`、`dd` 写设备…） | 永远 deny，不进入审批 |

### ④ 沙箱执行

| 平台 | 后端 | 限制 |
| --- | --- | --- |
| Windows | DSH Windows ACL 沙箱（随 Bun 依赖安装） | 限制文件写入；读取和网络由会话策略管理 |
| Linux | bubblewrap（需要安装 `bwrap`） | 只读根文件系统 + 授权目录可写；未授予网络时使用 `--unshare-net` 隔离网络；`/tmp` 是私有的 |
| macOS | 暂无沙箱后端 | 受限模式下 Shell 无法运行 |

没有可用沙箱后端时，`workspace-write` 和 `llm-auto` 失败关闭（拒绝执行），不会退化为无沙箱运行。

默认授权：

- 前台和后台命令默认可写整个工作区；`bash` 的 `write_paths` 参数只能进一步收窄到子目录。
- 工作区内的 `.git`、`.env` 等文件都在授权范围内。沙箱只守住工作区边界，工作区内部改什么由权限策略决定。
- `read-only` 模式不授予工作区写权限：Windows ACL 以只读模式启动，Linux 不添加可写挂载。
- `danger-full-access` 完全跳过沙箱，也忽略 `write_paths`。

Windows 的 ACL 授权会在 NTFS 上留下 ACL 条目和完整性标签；每个目录使用固定 SID，重复授权会复用已有条目。多目录授权由 `frontend/opentui/patches/` 中的补丁实现。

## 权限模式

| 模式 | 直接放行 | 谁来回答 `ask` |
| --- | --- | --- |
| `read-only` | 只读命令（包括普通网络读取） | 没有人：需要审批的操作一律拒绝 |
| `workspace-write` | 可证明的工作区写入、验证命令、沙箱内的未知程序 | 用户 |
| `llm-auto` | 与 `workspace-write` 相同 | 命令审查模型 |
| `danger-full-access` | 除系统 / 磁盘级破坏以外的全部 | 没有：只有 allow / deny |
| `terminal-eval` | Terminal-Bench 独立 runner 的预设 | — |

`llm-auto` 的审查模型自身出错（超时、返回非 JSON、字段非法、服务异常）时按拒绝处理，不会转交给用户；审查模型明确否定时，也不会提示 Agent 改写命令去绕过。唯一跳过审查模型的情况是请求中含有无法脱敏的凭据，这类请求只能由人审批，没有人工入口时直接失败。

计划模式（`/plan`）期间，运行时把权限收紧到 `read-only`，委派出去的 worker 子 Agent 也不会因为换了沙箱而获得写权限。

需要强调：永久 deny 的含义是「这条策略不能被审批覆盖」。静态规则只能识别它认识的命令，在关闭沙箱的 `danger-full-access` 模式下，无法在操作系统层面保证任意脚本不会破坏数据。

## 额外权限

需要写工作区外的目录或访问网络时，工具调用要显式申请：

```json
{
  "command": "python report.py",
  "sandbox_permissions": "with_additional_permissions",
  "additional_permissions": {"file_system": {"write": ["D:\\output"]}, "network": true},
  "justification": "需要把报告写到用户指定的目录"
}
```

- `use_default`（默认）使用会话的沙箱设置；`with_additional_permissions` 在审批通过后，只为这一次调用增加指定的目录写权限或网络授权。
- 审批对象包含命令本身和申请的能力，执行器只使用审批时生成的那份授权快照。改参数、改权限或切换模式都不会复用旧的批准。
- 审批时记录路径解析后的真实位置，执行前再次校验；如果符号链接或 Junction 在审批后被改动，旧授权失效。
- 不支持的权限类型、系统保护路径、只读模式下的申请、子 Agent 发起的提权，都会直接拒绝。
- Windows 上的网络授权只影响审批，不做 OS 级网络隔离。

## 持久授权

审批时选择「本项目后续允许」，会生成一条持久规则，按「命令前缀 + 写入目录 + 沙箱权限 + 网络授权」整体匹配。写入、删除、Git 和网络读取可以持久化；上传、未知解释器和标记为必须人工审批的请求每次都要审批。

每次调用都先重新解析命令、计算权限决定，只有结果是 `ask` 时才去匹配持久规则。

## 结构化文件工具

`write_file`、`apply_patch` 等结构化编辑工具同样只受工作区边界约束，修改前做快照和冲突检测。`.env` 的改动在事件和远程审查载荷中替换为 `[redacted .env diff]`，其他参数中常见的凭据形态也会在落盘前脱敏。

## MCP 服务

MCP 服务按项目登记在状态根的 `mcp.json` 中，每个服务声明一个权限级别（`read`、`network_read`、`dangerous`）。`dangerous` 工具在 `read-only` 和计划模式下被拒绝。配置中的密钥可以用 `${ENV_NAME}` 引用环境变量。

旧版本放在工作区 `.harness/mcp.json` 的配置可能包含可执行的启动命令，因此不会自动迁移；核对后用 `/mcp add` 或 `veriforge mcp add` 重新登记。
