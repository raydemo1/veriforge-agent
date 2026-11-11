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

## 状态与持久化边界

| 数据 | 位置 | 说明 |
| --- | --- | --- |
| 逻辑会话手记 | `.harness/sessions/<id>/journal.jsonl` | 只有成功写入日志后才更新活动视图 |
| 运行 Trace | `.harness/traces/<session>/` | 每个 Agent 实例一个文件，按 id 关联 |
| 长期记忆 | `~/.harness/memory-v3/memory.db` | 唯一数据源；`views/` 只是按主题生成的只读视图 |
| Shell 大输出 | `.harness/observations/<session>/` | 超限才落盘，模型用 `read_file` 回读 |

## 验证

```bash
python -m pytest -q                                  # 完整非外部测试
python -m pytest -q -m "not session and not integration and not external"   # 快速反馈
python -m pytest -q -m integration -rs               # 真实 ACL / LSP / Node 边界
```

`session`、`integration`、`external` 三个标记见 `pyproject.toml`。前两个在 CI 分开跑，
`external` 必须显式启用；跳过不等于通过。
