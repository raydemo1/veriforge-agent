# 架构

本文介绍运行时的模块划分、一轮对话和一次工具调用经过的路径，以及所有长生命周期资源都要遵守的约定。用户可见的用法见 [README](../README.md)。

## 模块分层

```text
          cli.py · headless.py · tui_bridge.py · core/ · tui/      入口与界面
                                 │
                              agent/                                对话循环
                                 │
                             runtime/                               工具、权限、调度
                                 │
              memory/ ──── workspace/ ──── sessions/                 状态与副作用
                                 │
                             support/                               叶子原语
```

| 包 | 职责 | 允许依赖 |
| --- | --- | --- |
| `support/` | 取消令牌、token 计数、LLM 客户端与 provider 适配 | 仅 `config` |
| `sessions/` | 会话元数据、事件总线、JSONL 日志、观测与报告 | `support` |
| `workspace/` | 路径规则、文件写保护、Shell 进程与后台任务、沙箱、write lease、Worktree | `support`、`sessions` |
| `memory/` | 长期记忆的存储、检索、后台提取与整理 | `support`、`sessions`、`workspace` |
| `runtime/` | 工具注册表、参数校验、权限策略、调度、中间件、MCP、LSP、改动验证 | 以上全部 |
| `agent/` | 对话循环、上下文生命周期、工具执行编排、子 Agent | 以上全部 |
| `core/`、`tui/`、入口文件 | 交互会话、多会话注册表、终端界面桥接 | 以上全部 |
| `skills/` | 按需加载的 Markdown 知识目录 | 无代码依赖 |

依赖只能自上而下。`memory`、`sessions`、`workspace` 不能 import `agent`；多个层都要用的原语下沉到 `support` 或 `workspace/paths.py`。`tests/test_layering.py` 用 AST 检查这条规则，违反时测试直接失败。

前端 `frontend/opentui/` 是独立的 Bun 进程，通过 stdin/stdout 上的 NDJSON 与 `tui_bridge.py` 通信。协议定义在 `harness_code_agent/tui/protocol.py` 和 `frontend/opentui/src/protocol.ts` 两侧，`UI_PROTOCOL_VERSION` 必须一致，改动协议时两边一起改并递增版本号。

## 一轮对话

`agent/conversation.py` 中的 `AgentConversation` 负责一轮对话，循环直到模型不再调用工具或触发停止条件：

1. **准备上下文**：检查 token 用量，必要时自动压缩（见 [上下文与会话](context-and-sessions.md)），然后构建本次请求的消息。
2. **请求模型**：`agent/llm_channel.py` 发出请求（流式或非流式），负责重试 429 / 5xx / 连接错误、尊重 `Retry-After`、检测流式空闲超时。provider 差异集中在 `support/llm.py` 的 `ProviderAdapter`。
3. **执行工具**：模型返回的工具调用交给 `agent/tool_executor.py`，见下一节。
4. **决定是否继续**：`agent/turn_controller.py` 是继续 / 停止的唯一决策点，统一处理时间预算、token 与工具调用预算、输出被截断等情况，并记录唯一的停止原因。
5. **回合结束**：中间件做退出前检查，`runtime/verification/` 检查本轮改动过的文件，结果写入会话日志。

中间件（`runtime/middleware/`）通过固定的钩子参与这个过程：对话开始和结束、上下文压缩后、工具获准执行前、工具失败后。用户可以在 `~/.veriforge/middlewares.json` 中声明自定义中间件；出于安全考虑，不读取仓库内的中间件配置，否则克隆一个仓库就可能执行其中的 Python 代码。

## 一次工具调用

```text
模型输出的 tool_calls
   │
   ├─ 准备：参数校验 → 命令解析 → 资源声明 → 权限决定   （一次性生成 ToolExecutionPlan）
   ├─ 调度：按资源冲突分波，无冲突的并行
   ├─ 审批：需要时请求用户 / 审查模型
   ├─ 执行：调用处理函数，记录事件与失败分类
   └─ 回写：按预算内联结果，超限的落盘并返回读取引用
```

| 阶段 | 位置 |
| --- | --- |
| 参数校验 | `runtime/tool_call_validation.py`。Schema 未声明 `additionalProperties` 时按闭集处理，多余参数会被拒绝 |
| 资源声明与调度 | `runtime/execution_planner.py`、`agent/tool_executor.py`。工具声明自己读写哪些文件、目录或全局资源；未声明的扩展工具默认独占 |
| 权限决定 | `runtime/permissions.py`。只根据「工具 + 参数 + 命令事实」做 allow / ask / deny，细节见 [权限与沙箱](permissions-and-sandbox.md) |
| 审批 | `runtime/permission_middleware.py`、`runtime/approvals.py` |
| 执行 | `runtime/tool_runner.py`。处理函数抛出的任何异常都被转成失败结果，不会中断对话循环 |
| 结果回写 | `agent/observations.py`。单条结果默认最多内联 5000 token；超出部分完整写入文件，模型拿到状态、大小、首尾预览和 `next_read` 参数，用 `read_file` 继续分段读取 |

准备阶段对每个调用只分析一次，生成不可变的 `ToolExecutionPlan`（命令 AST 分析结果、资源声明、权限决定、是否后台运行）。调度器、权限中间件和 Shell 任务管理器共用这份快照，不会各自重新解析，也就不会出现「审批时看到的命令」和「实际执行的命令」不一致。

模型能看到的工具描述和运行时的保障是两回事：Schema 只描述接口（参数含义、返回值、分页规则），权限、隔离和预算由运行时强制执行，不依赖模型遵守描述。

### 工具披露

常用的文件、Shell、搜索、记忆工具始终挂载。子 Agent 协作类工具（`spawn_agent` 等）默认不挂载，模型通过 `search_tools` 按需发现后再长期挂载，以减少每次请求的 Schema 体积。MCP 工具同样按需加载。

## 子 Agent

主 Agent 默认独立完成搜索、编辑、测试和审查。只有用户明确要求委派、独立研究或多 Agent 协作时，才会调用 `spawn_agent`。子 Agent 只有两种角色：

| 角色 | 能力 |
| --- | --- |
| `explorer` | 只读：代码探索、网页搜索、审查、测试方案设计 |
| `worker` | 在共享工作区内，只能修改分配给它的 `allowed_paths`；Shell 的可写目录也收窄到这个范围 |

并发 worker 的写入范围如果重叠，`spawn_agent` 会直接拒绝，需要重新划分或顺序执行。子 Agent 使用自己的会话日志和 Trace，沿用父会话的项目记忆身份。并发数由 `HARNESS_MAX_CONCURRENT_AGENTS`（默认 3）和 `HARNESS_MAX_OPEN_AGENTS`（默认 8）限制。

## 资源生命周期约定

Shell 任务、MCP 连接、子 Agent 和会话运行时都遵守同一组约定。违反任何一条，都会出现「看起来结束了，实际还占着资源」的状态。

1. **终态必须有事实依据**。只有拿到确定的完成或失败证据，才进入 `exited` / `stopped` / `failed`。
2. **谁申请，谁释放**。调用方不替资源的拥有者兜底释放。
3. **取消请求和真实退出是两回事**。`stop` 只是请求，确认退出前状态保持 `stopping`。终态只代表执行结果，不代表资源已释放：一个 `exited` 的任务可能仍持有写租约，它必须仍然可以被停止，并阻止会话休眠。
4. **没确认退出，就不能假设资源已释放**。写租约、临时目录、连接保持占用并可观测；释放动作本身失败时保留句柄，以便重试。
5. **关闭失败要可诊断、可重试**。不能静默吞掉，也不能谎报成功。

`workspace/shell_jobs.py` 是这套约定最完整的实现，它分别记录：

- `launcher_exited` + `exit_code`：启动的直接进程是否退出；
- `processes_drained`：受管进程树是否全部结束；
- `readers_drained`：输出读取线程是否结束；
- `resources_released`：写租约和私有临时目录是否真的释放。

由此得到的行为：

| 场景 | 行为 |
| --- | --- |
| 前台命令的直接进程退出 | 记录退出码，给残留子进程一个有上限的排空窗口（默认 2 秒），超时后终止 |
| 后台任务的启动进程退出、子进程仍在运行 | 继续显示 `running`，并报告 `launcher_exited` |
| 主动停止 / 取消 | 终止整个进程树，确认退出后才释放租约 |
| 终态任务仍持有租约 | 仍可用 `stop_shell_job` 清理；会话关闭和休眠会被它阻塞 |
| 读不到进程状态（如 `AccessDenied`） | 按存活处理，租约保留 |
| 释放租约或清理临时目录失败 | 保留句柄、返回未完成，可重试 |

后台任务的排空没有时间上限，因为子进程就是任务本身；前台命令有上限，否则一次工具调用会一直挂到超时。读取线程的 `join` 也有上限，超时只记录警告，不会假装输出已完整。

## 状态存放位置

项目目录保持为普通的 Git 工作区，运行时状态都放在用户目录下：

| 数据 | 位置 |
| --- | --- |
| 项目级状态根 | `~/.veriforge/workspaces/<项目名>-<路径哈希>/`（`VERIFORGE_HOME` 可覆盖 `~/.veriforge`） |
| 会话日志 | `<状态根>/sessions/<会话 ID>/journal.jsonl` |
| Trace | `<状态根>/traces/<会话 ID>/`，每个 Agent 实例一个文件 |
| 工具大输出 | `<状态根>/sessions/<会话 ID>/observations/<agent>/` |
| MCP 配置 | `<状态根>/mcp.json` |
| 长期记忆 | `~/.veriforge/memory-v3/memory.db`（`VERIFORGE_MEMORY_ROOT` 可覆盖） |

状态根由项目的规范化绝对路径哈希决定，同一路径始终复用同一目录。工作区写入工具只能写项目目录，沙箱的额外目录授权也不能指向状态根，因此 Agent 无法通过改文件来篡改审批规则或会话记录（`danger-full-access` 模式除外）。

Trace 中的 `ui_request_id`、`turn_id`、`llm_call_id`、`tool_call_id`、`shell_job_id` 可以把一次界面请求、模型调用和工具执行串起来排查。
