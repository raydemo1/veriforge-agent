# 架构

本文只讲模块边界与运行时契约。安装、用户可见行为和命令见 [README](../README.md)；
上下文版本、压缩与请求诊断见 [deepseek-context-runtime.md](deepseek-context-runtime.md)；
多会话资源的实测数据见 [multi-session-resources.md](multi-session-resources.md)。

## 分层与依赖方向

| 包 | 职责 | 可以依赖 |
| --- | --- | --- |
| `support/` | 叶子原语：取消令牌、token 计数、LLM 客户端适配 | 仅 `config` |
| `runtime/` | 工具 registry、参数校验、权限策略、调度、middleware、MCP、LSP | `support`、`workspace`、`sessions` |
| `workspace/` | 路径规则、文件保护、Shell 与后台任务、write lease、worktree | `support`、`sessions` |
| `memory/` | SQLite 事实与修订、主题视图、向量＋全文检索、后台整理 | `support`、`sessions`、`workspace` |
| `sessions/` | session 元数据、事件总线、逻辑手记、观测与报告 | `support` |
| `agent/` | 对话循环、上下文生命周期、工具执行编排、子代理 | 以上全部 |
| `core/`、`tui/`、`tui_bridge.py`、`cli.py`、`headless.py` | 入口与界面 | 以上全部 |
| `skills/` | 按需加载的 Markdown 知识目录 | 无代码依赖 |

`memory`、`sessions`、`workspace` 不允许 import `agent`；共享原语只能下沉到 `support`。
`tests/test_layering.py` 用 AST 守着这条规则，顺手 import 会让它直接失败。

## 一次工具调用经过的位置

1. `agent/tool_executor.py` 决定哪些调用可以同波并行（`runtime/execution_planner.py` 的资源声明）。
2. `runtime/tool_call_validation.py` 校验模型给出的参数；Schema 未声明 `additionalProperties` 时按闭集处理。
3. `runtime/permissions.py` 把「工具 + 参数 + 命令事实」映射为 allow / ask / deny，
   `runtime/shell_classification.py` 只产事实，不做决定。
4. `runtime/permission_middleware.py` 执行准备阶段算好的决定，并在需要时驱动审批。
5. `runtime/tool_runner.py` 调用处理函数，把结果与失败分类落成事件。
6. `agent/observations.py` 负责内联预算、超限落盘和 `read_file` 回读引用。

模型的可见面与运行时保障是两件事：Schema 只描述接口，权限、隔离和预算由 runtime 实际执行。

## 资源生命周期不变量

Shell、MCP、子代理和会话都遵循同一组不变量；违反其中一条就会出现「看起来没事、实际泄漏」的状态。

1. **终态必须对应事实。** 只有拿到确定的完成或失败证据才进入 `exited` / `stopped` / `failed`。
2. **释放资源只有一个责任方。** 谁申请谁释放，调用方不替它兜底。
3. **取消请求与真实退出分离。** `stop` 只是请求；确认退出前状态仍是 `stopping`。
   **终态只代表执行结果，不保证资源已经释放**：`exited` 的任务仍可能持有写许可，
   这种任务必须继续可停、可导出拒绝、可阻塞休眠。
4. **未确认退出不得假定资源已释放。** 写入许可、临时目录、连接都保持占用并可被观察到；
   释放动作本身失败时，句柄必须保留以便重试。
5. **关闭失败要可诊断、可重试。** 不能静默吞掉，也不能伪报成功。

`workspace/shell_jobs.py` 是这套不变量的完整示例，它把事实分开记录：

- `launcher_exited` + `exit_code`：启动的直接进程是否退出；
- `processes_drained`：受管进程树是否排空；
- `readers_drained`：输出读取线程是否结束（`drained` 是两者的与，读取线程卡住时不会报“已排空”）；
- `resources_released`：write lease 与私有临时目录是否**真的**释放。

由此得到的行为：

| 场景 | 行为 |
| --- | --- |
| 前台命令直接进程退出 | 记录 exit code，进入有上限的排空窗口 |
| 前台命令仍有存活后代 | 宽限期内等待，超过后终止并记录 warning |
| 后台 Job 的启动进程退出、后代仍运行 | 继续显示 `running`，同时报告 `launcher_exited` |
| 主动 stop / 取消 | 终止受管进程树，确认后才释放 lease |
| 终态 Job 仍持有 lease（后代未死） | 仍可用 `stop_shell_job` 清理；`close()`/休眠把它算作阻塞 |
| 监控线程自身异常 | 尽力终止、进入 `failed`、设置完成事件；未确认则保留 lease |
| lease 释放或临时目录清理抛错 | 句柄保留、返回未完成，可重试；不伪报已释放 |
| 进程状态读不到（`AccessDenied` 等） | 按存活处理，lease 保留；只有从**仍存活的启动进程**重新枚举整棵树才算恢复证据 |

后台任务的排空没有上限——它的后代就是任务本体；前台命令的排空有上限，否则一次调用会一直等到自己的超时。
读取线程的 `join` 同样有上限，超时只记 warning，不伪造“输出已完整”。
`export_jobs()` 拒绝导出仍持有资源的任务：休眠必须等在真的释放之后。

## Shell 安全边界

一条 Shell 命令经过四层，任一层都不能被前一层的结论替代：

1. **语法**（`runtime/shell_parser.py`）：Windows 用 PowerShell 7 的 `Parser.ParseInput`（常驻
   JSONL 子进程），其它平台用 tree-sitter-bash。只产出 `commands / redirections / dynamic /
   complete / error`，绝不执行。每个参数同时携带“字面量 / 运行时求值”标志，分类器不再靠字符串
   里是否出现 `$` 猜测。Bash 的 tree-sitter parser 实例有锁，并发解析稳定；语法结果缓存，
   依赖文件系统状态的路径作用域不缓存。
2. **事实**（`runtime/shell_classification.py`）：命令按规则表分派到少量专用分析器——文件
   创建/复制/移动/删除、Git（全局参数 + 仓库目标）、包管理器（查询 / 安装 / 自定义脚本）、
   PowerShell Cmdlet（`-Path` / `-ItemType` / `-Value` 等参数角色）、解释器、网络。
   `ShellTarget` 区分读取、写入、删除；`UNKNOWN_EFFECT`、`NETWORK_UPLOAD`、`DESTRUCTIVE`
   仍是事实，不含决定。`CATASTROPHIC` 只由“拥有系统/全盘目标的那条命令”产生，另一条命令读取
   系统文件不会把工作区删除升级成系统删除。`npm run deploy`、`python script.py` 是“执行未知
   代码”，不是只读。
3. **决定**（`runtime/permissions.py`）：每个事实产生一个候选动作，**最严者胜**。普通只读、
   可证明的工作区写、验证任务和未知程序直接放行（未知程序由沙箱兜底）；网络访问、破坏性 Git、
   大范围删除和一次性额外权限才是 `ask`；系统/磁盘级破坏仍是无条件 `deny`。
   工作区外写入只有在同一次调用显式申请并获批后才放行。解析器不可用、协议失败或超出解析预算时，
   拒绝执行该命令。
4. **隔离**（`workspace/native_shell.py` / `workspace/posix_sandbox.py`）：Windows ACL 目录沙箱；
   Linux bubblewrap（只读根 + 授权目录可写 + 未授予网络时 `--unshare-net`）。
   Windows ACL 限制文件写入，读取与网络访问依赖会话策略。需要 OS 级网络隔离时使用 Linux
   bubblewrap。macOS 的受限 Shell 暂无沙箱后端。
   前台与后台命令默认都获得整个工作区的写授权，`write_paths` 只用于收窄到子树；
   `read-only` 的执行授权不授予工作区写权限：Windows ACL 以 `read-only` 模式启动，
   Linux 不添加工作区可写 `--bind`。
   `.git` 等目录都在授权范围内：沙箱只守住工作区边界，工作区内部改什么由 Agent
   在审批策略下决定。运行时状态不在工作区内，因此工作区写入授权无法触达审批规则和会话记录。
   前台命令按实际授权申请写租约；后台任务只保留显式声明的目录级租约，
   不长期独占整个项目，因此 Watch/开发服务器可以并发运行，仍可通过 `stop_shell_job` 停止。
   `bash` 可用 `background: true` 显式启动后台任务；省略时沿用长任务启发式。
   `danger-full-access` 完全跳过平台沙箱，Windows 与 POSIX 都以非受限方式启动；
   `workspace-write` / `llm-auto` 在没有沙箱后端时失败关闭。

沙箱内的额外权限由工具调用申请：

```json
{
  "command": "python report.py",
  "sandbox_permissions": "with_additional_permissions",
  "additional_permissions": {"file_system": {"write": ["D:\\output"]}, "network": true},
  "justification": "需要把报告写到用户指定的目录"
}
```

`use_default` 使用会话的沙箱设置。`with_additional_permissions` 在审批后为该次调用增加明确的
目录写权限或网络授权。Windows 上网络授权仅用于命令审批。只有用户可以显式切换到
`danger-full-access`。策略校验请求、审批对象包含命令与申请的能力、执行器只使用
`PermissionDecision.authorization` 这一份快照，因此改参数、改权限或切换会话模式都不会复用旧批准。
发起请求的目录不存在时由执行器创建。审批时保存解析后的真实路径，执行前再次校验；若符号链接或
Junction 在审批后改变，旧授权会被拒绝。不支持的权限类型、系统保护路径、只读模式或子 Agent 发起的
升级都会直接 `deny`，不会“批准后假装生效”。

旧工作区的 `.harness/mcp.json` 可能包含可执行的 stdio 启动命令，因此保留原文件。用户核对配置后，
可通过 `/mcp add` 重新登记可信服务。

`danger-full-access` 忽略 `write_paths` 的工作区约束：它没有沙箱可收窄，填不填都不影响执行。
结构化文件工具同样只受工作区边界约束：`.git`、`.env` 等文件可以正常修改，快照和冲突检测照常
生效；`.env` 的变更 diff 在事件和远程审查载荷中替换为 `[redacted .env diff]`，工具事件里针对
`.env` 的 `search` / `replace` 只保留长度，其他参数中的常见凭据形态也会在落盘前脱敏。

持久授权按“命令前缀 + 写入目录 + 沙箱权限 + 网络授权”整体匹配，用户选择“本项目后续允许”时才会
写入。写、删除、Git 和网络读可以持久化；上传、未知解释器和 `requires_human` 请求仍然每次
审批，不会生成可复用规则。Worker 与主 Agent 共享同一工作区，`allowed_paths` 同时约束结构化编辑
工具和 Shell 的实际写授权：Worker 未显式传入 `write_paths` 时，`bash` 默认只申请
`allowed_paths` 这些目录。并发 Worker 的写入范围重叠会在 `spawn_agent` 时被拒绝，需要重新划分
任务或顺序执行；共享工作区的写入租约仍然生效。

下表列出工作台可切换的四个模式。`workspace-write` 与 `llm-auto` 共用沙箱和审批策略，
前者把 `ask` 交给用户，后者交给命令审查模型。Terminal-Bench 独立 runner 还有 `terminal-eval`
预设。Harbor 适配器在评测容器内显式选择 `danger-full-access`。

| 模式 | 直接放行 | `ask` 的答复者 |
| --- | --- | --- |
| `read-only` | 只读命令（含普通网络读） | 没有（需要 `ask` 的一律 deny） |
| `workspace-write` | 可证明的工作区写、验证任务、未知程序（沙箱内）；网络访问需审批 | 用户 |
| `llm-auto` | 与 `workspace-write` 完全相同 | 命令审查模型 |
| `danger-full-access` | 除系统/磁盘级破坏外全部 | 没有：只产出 allow / deny |

审查器自身的失败（超时、非 JSON、字段非法、服务异常）是安全拒绝，不会转交给用户；审查器给出
明确否定也一样，且不会提示 Agent 改写命令绕过同一结论。唯一绕过审查模型的是“连提示都无法安全
构造”的请求（无法脱敏的凭据），这类请求直接找人，没有人工入口就失败关闭。

系统目录写入和系统/磁盘级破坏仍是 `DENY`，不进入审批；工作区外写入默认 `DENY`，只有
`additional_permissions.file_system.write` 明确申请并通过审批的那次调用可以写。静态黑名单只
覆盖它能识别的命令；完全访问关闭沙箱后，无法在操作系统层保证任意脚本绝对不破坏数据，
永久 `DENY` 表示“策略不可被审批覆盖”，不是无限制执行环境中的系统级保证。

四层的关系是**互补而非替代**：AST 只说明命令长什么样，审批只说明意图是否可接受，只有沙箱能兜住
静态分析看不见的行为（任意脚本、未建模的写入参数、时间竞态）。每次调用先重新解析命令并计算
权限决定，出现 `ask` 时再匹配持久授权规则。

工具准备阶段只分析一次命令，形成不可变的 `ToolExecutionPlan`（AST 分析、资源声明、权限决定、
后台模式）。调度器、权限中间件和 `ShellJobManager` 共享同一份快照，不再各自重新解析。显式
`background` 同时决定调度策略和进程模式；未知程序与后台长任务只声明工作区写 claim，不再无条件
成为整批调用的全局屏障。

## 子 Agent

默认由主 Agent 独立完成搜索、编辑、测试、调试和审查。任务复杂、步骤较多或可以并行本身都不是创建
子 Agent 的理由；只有用户明确要求子 Agent、独立研究、分工或多 Agent 协作时才调用 `spawn_agent`。
没有明确委派时，主 Agent 可以直接建议协作，但不会擅自启动。

只有两种角色：

| 角色 | 能力 |
| --- | --- |
| `explorer` | 只读研究：代码探索、网页搜索、审查、测试方案设计；不修改文件，不复制工作区 |
| `worker` | 在共享工作区按分配的 `allowed_paths` 完成开发任务；修改立即可见，并发写入范围不得重叠 |

网页搜索、代码探索和测试方案不再单独建角色；它们只是 Explorer 接收到的不同任务。

## 状态与持久化边界

| 数据 | 位置 | 说明 |
| --- | --- | --- |
| 会话手记 | `~/.veriforge/workspaces/<name>-<hash>/sessions/<id>/journal.jsonl` | 只有成功写入日志后才更新活动视图 |
| 运行 Trace | `~/.veriforge/workspaces/<name>-<hash>/traces/<session>/` | 每个 Agent 实例一个文件，按 id 关联 |
| 长期记忆 | `~/.veriforge/memory-v3/memory.db` | 唯一数据源；`views/` 只是按主题生成的只读视图 |
| Shell 大输出 | `~/.veriforge/workspaces/<name>-<hash>/sessions/<id>/observations/<agent>/` | 超限才落盘，模型用 `read_file` 以绝对路径回读 |

项目目录保持为普通 Git 工作区；状态根由规范化后的项目绝对路径哈希确定，同一路径始终复用同一目录，
不同项目互相隔离。`VERIFORGE_HOME` 可覆盖用户级状态根。显式 `state_root` 会同时传给会话存储、
`WorkspaceService`、`WorkspaceAccessCoordinator`、`RecoveryService` 和 `ShellJobManager`，避免
会话与并发数据库落在不同状态根。工作区写入工具只能写项目目录，沙箱额外目录
授权也拒绝指向状态根；`danger-full-access` 绕过平台沙箱，是唯一的例外。

## 验证

```bash
python -m pytest -q                                  # 完整非外部测试
python -m pytest -q -m "not session and not integration and not external"   # 快速反馈
python -m pytest -q -m integration -rs               # 真实 ACL / LSP / Node 边界
```

`session`、`integration`、`external` 三个标记见 `pyproject.toml`。前两个在 CI 分开跑，
`external` 必须显式启用；跳过不等于通过。
