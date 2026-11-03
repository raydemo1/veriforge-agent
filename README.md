# VeriForge

VeriForge 是一个面向真实代码仓库的可验证 Coding-Agent Runtime。

它把模型、工具、权限、会话和验收放在同一条可追踪链路里，支持修复代码、补测试、代码审查、计划设计和应用构建。

基于 OpenAI-compatible Chat Completions API，可接入 DeepSeek、OpenAI 等服务。

![VeriForge OpenTUI](https://raw.githubusercontent.com/raydemo1/veriforge-agent/main/docs/images/veriforge-tui.png)

## 特性

- Profile：`general`、`coding-agent`、`plan`、`review`、`app-builder`、`terminal`
- 工具治理：文件、Shell、Web、浏览器、MCP 和 Agent 统一经过 registry、权限和 middleware
- 资源感知调度：无冲突调用可并行，冲突资源按顺序执行，结果按模型顺序回写
- 安全修改：工作区路径保护、快照、审批、危险命令拦截和退出前验证
- 代码理解：单一只读 `code_intelligence` 工具，支持定义、引用、文件符号和诊断
- 多语言验证：本轮修改触发 Python AST/Ruff、TypeScript 静态检查和 Go 格式检查；项目代码执行经过 Shell 权限体系
- 子代理协作：只读 Agent 并行调查，worker 在隔离副本中生成可复查提案
- 上下文与记忆：可恢复的 JSONL 会话手账、结构化压缩、Markdown 长期记忆和可重建索引
- OpenTUI：命令与 `@` 补全、审批、会话、运行观察和模型设置

## 快速开始

环境要求：Python 3.10+、Bun 1.4+、Git，以及一个 OpenAI-compatible API key。

```bash
pip install -e .
cd frontend/opentui
bun install
cd ../..
```

复制配置模板并填写 API 信息：

```bash
cp .env.template .env
```

Windows PowerShell：

```powershell
Copy-Item .env.template .env
```

启动 TUI：

```bash
veriforge
```

也可以直接提交任务：

```bash
veriforge "Fix the failing tests"
veriforge -p "Review this repository"
```

CLI `-p`、无 TTY 的批处理，以及 Terminal-Bench / Claw-SWE-Bench runner 都通过同一个 Headless API 执行单轮任务：

```python
from harness_code_agent.headless import run_task

result = run_task(cwd="/path/to/project", task="Fix the failing tests", profile="coding-agent")
print(result.session_id, result.status)
if result.turn_result is not None:
    print(result.turn_result.text)
raise SystemExit(result.exit_code)
```

`RunResult` 返回工作区、会话 ID、记录目录 `harness_root`、`TurnResult`、运行状态和错误信息。`completed` 表示本轮正常结束，任务是否通过仍由测试或评测器判断；`failed` 表示初始化、执行或关闭抛出异常；`interrupted` 表示键盘中断，对应退出码分别为 `0`、`1`、`130`。执行失败后仍返回已启动会话的 ID 和记录目录。系统终止信号引发的 `SystemExit` 在资源关闭后继续传播，保留原退出码。

`on_session_started` 回调在提交任务前接收 `SessionInfo(session_id, cwd, harness_root)`，可立即写入 manifest 或在运行期间导出日志。接口统一负责关闭会话，保留正常的 Recovery 与权限检查；无界面运行不从 stdin 请求审批或问题答案，需要人工审批的操作拒绝执行，问题按取消处理。评测 runner 的权限模式仍由各自运行环境配置。

常用入口：

```text
/mcp          管理 MCP 服务
/compact      压缩上下文
/context      查看上下文预算
/memory       搜索、验证与管理长期记忆
/fork         创建会话分支
/observe      查看运行状态
```

输入 `/` 或 `@` 会打开遮罩补全面板。Enter/Tab 接受候选，点击外层关闭；修改或清空指令后会重新匹配。普通状态面板可点击外层或按 Esc 关闭。

上下文默认达到窗口的 80% 时压缩，同时预留模型输出空间。压缩只更新当前上下文视图，原始会话日志保留。

工具结果默认最多内联 5,000 token。超限结果完整落盘，返回状态、大小、文件引用及前后各最多 500 token 的预览，整条消息仍受 5K 上限约束；用 `read_file` 按 `next_read` 连续回读，长单行可通过 `start_column` 续读。短结果不额外生成文件。

长期记忆以 `~/.harness/memory-v3/memory.db` 为唯一数据源，用户偏好与各仓库分区隔离，同仓库的 worktree 共用项目记忆。`views/` 中的 Markdown 是按主题生成的只读视图，直接修改不会写回数据库。六类记忆为：偏好、项目事实、规范、决定、经验和工作流程。

在本机 `.env` 中配置 `SILICONFLOW_API_KEY` 后，检索使用硅基流动 `BAAI/bge-m3`（1024 维）与 SQLite 全文检索，两路独立召回再合并排序。查询不额外调用聊天模型，但会产生 embedding 请求；重复查询在会话内缓存。语义服务失败或索引未齐全会明确标记，不把检索失败解释为没有记忆。可用 `HARNESS_MEMORY_ROOT` 指定独立存储根目录。

记忆按三个独立维度组织：`scope` 区分个人与项目，`topic` 聚合同一知识领域，`kind` 标记偏好、事实、规范、决定、经验或流程。同一主题可包含不同 `kind`；已登记的主题别名复用原主题。个人空间仅保存来自用户的跨项目偏好，不接受项目文件依赖或验证结果。

新会话加载统一的启动快照：个人默认最多 300 token，项目约定最多 500 token，导航说明及主题目录最多 400 token，头部和引用信息计入预算。正文只选择有效、无条件适用的偏好和规范；条件规则、项目事实、决定、经验和流程默认只提供主题入口。项目同事实键的全局约定覆盖个人默认，不改写原记录；过期、待确认和挂起的内容不会作为确定事实展示。目录优先列出范围和主题名称，在预算允许时补充事实键和适用条件；可用 `memory_read` 按范围和主题名称展开。各范围显示已展示和总主题数，未展示完时提供可直接用于 `memory_read` 的续读参数，不自动生成自由摘要或执行向量检索。

启动快照按稳定事实键排序，在 Profile 切换和压缩时复用，恢复或回撤只在新的上下文边界刷新一次；原始日志保留。子代理沿用父会话的项目记忆身份。后续由模型在需要历史信息时主动调用 `memory_search` 或 `memory_read`，不逐轮自动检索或注入记忆正文。搜索支持分页、路径和类别筛选，读取支持展开主题、单条事实和修订历史。已经在上下文中的事实可直接复用。

明确保存通过 `memory_write` 即时完成，必须逐字引用当前原始用户输入作为证据；纠正使用原 ID 和版本，保留旧修订。自动提取只读取单独记录的原始用户输入和原生结构化验证，不读取回忆、模型推断或任意工具输出。每个成功主轮结束后立即安排后台增量提取，前台回答不等待；连续轮次合并尚未处理的范围，下一轮执行期间暂停相关会话的提取和整理。提取候选、跨会话整理和向量更新分别持久化、分阶段处理。关闭进程或会话不等待后台生成，未完成工作在后续启动时恢复；`/memory generate off` 取消当前会话尚未完成的生成任务，并撤销这些候选对旧事实的挂起；重新开启后恢复候选及任务，无需等待空闲期。关闭期间的输入不补做自动抽取。此开关不禁止显式保存、确认和遗忘，这些操作仍需原文证据并经过正常工具权限检查。

整理按事实身份和全文索引寻找已有记录，只对当前有效视图做逻辑更新，原始候选与修订历史保留。历史事件不反复成为当前事实，未决冲突只返回不确定提示。文件证据变化会使相关事实过期；较旧的后台结果不能覆盖较新的已确认状态。向量更新在事实整理之后异步处理，索引失败独立重试，不阻塞事实生效；旧向量随事实更新立即失效。

管理命令包括 `/memory status`、`search <query>`、`topics [scope]`、`show <id>`、`history <id>`、`edit <id> <version> <body>`、`validate <id> <version>`、`restore <id> <revision> <version>`、`forget <id> <version>`、`retry`，以及 `use on|off`、`generate on|off`。ID 可带 `project:` 或 `user:` 前缀。遗忘删除长期记忆及其修订并抑制旧来源再次提取，原始会话日志保留；用户明确要求重新记住时，`memory_write(restore_forgotten=true)` 使用新证据重新保存。

独立验证脚本：`eval/scripts/check_memory_consolidation.py`、`check_memory_runtime.py`；真实模型评测必须显式带 `--run`，场景与请求上限写入各次评测产物，不纳入默认测试集。

一个工作台可以同时运行多个会话，每个会话独立保存对话、队列、审批、模型设置、草稿和后台任务。切换标签不会中断执行，关闭标签只隐藏会话，历史会话恢复后沿用原 ID；「分支」从最近完整稳定的历史边界创建新会话。输入区的停止只取消当前回合，保留排队消息和后台任务；Ctrl+C 或关闭工作台会清理所有会话及其托管进程。

会话默认共享 Local。不同文件可以并行修改，同一文件写入前校验版本；真实冲突最多由 Agent 自动协调两轮；三方合并能安全通过时自动应用，仍有重叠或无法验证时展示最终差异，确认后才覆盖 Local。短命未知命令执行期间互斥写入；长期后台 Shell 允许其他会话继续写入，接受它直接改文件时的并发冲突风险。未知范围操作不扫描整个项目，只记录无法完整回撤的边界，回撤跨过该边界时才警告并拒绝。

需要隔离时选择「隔离工作区」，会 Fork 一个新会话，原会话和任务继续保留。Worktree 继承当前 Local 的未提交内容和非忽略的未跟踪文件，暂不支持从已有 Worktree 再派生；「应用到 Local」只计算创建后的任务改动，并检查 Local 后续修改，发生冲突时不会直接覆盖。

Windows 默认使用原生 PowerShell 7，使用随项目 Bun 依赖安装的 DSH Windows ACL 独立库限制写入，无需安装 Codex 或 DSH 应用。项目默认只读；Agent 在 `run_bash(write_paths=["dist", "cache"])` 中声明所需目录，进程只能写这些目录及私有临时目录。目录许可持续到整个进程树退出，其他目录仍可修改；后台任务不能申请整个共享 Local。Worktree 仅由用户主动创建，运行时不会因权限不足或写入范围未知而自动创建。此后端提供文件写入限制，不隔离读取和网络，沿用上游声明的 NTFS 硬链接等限制。Linux/macOS 使用同一进程托管链路执行 Bash，文件隔离由实际运行环境提供；Linux 评测沿用外层评测容器，不在工具内部再次启动 Docker。

稳定回合结束后保存回撤点，悬停或键盘聚焦该轮即可选择「回撤到此处」。「会话开始」可撤回第一轮。回撤沿用当前会话 ID，只撤销其拥有且仍可安全恢复的文件变化；其他会话或外部程序的后续修改会阻止回撤。原始记录保留，活动对话、计划和任务状态回到目标边界。后台 Shell 不阻止新恢复点；回撤时校验文件版本，并提示仍在运行的 Shell 可能带来的直接写入冲突。替换运行时前需先停止子代理。文件与会话记录通过可恢复事务一起更新。数据库、远端操作等外部副作用不在恢复范围内。

退出前验证只检查本轮改动，在没有 Git 的目录中也能发现 shell 修改；已有脏文件未在本轮改动时不会进入检查。自动执行 Python AST 和可用的 Ruff、最近 `tsconfig.json` 对应的 `tsc --noEmit`、`gofmt -d`。ESLint 会加载项目配置或插件，`go test` 会执行测试代码，`cargo check` 可能执行构建脚本或过程宏，因此自动验证将它们标为「已跳过」，同时提供对应目录和参数；需要运行时由 Agent 通过 `run_bash` 经过现有会话权限体系执行。优先复用项目已安装的工具，不自动安装依赖。缺失或超时的工具也标为「已跳过」，不计为通过。

源码验证与 LSP 文件变更扫描固定排除 `.harbor` 评测产物，以及依赖、缓存和构建目录。

`code_intelligence` 按需启动已有的 basedpyright/pyright、typescript-language-server、gopls 或 rust-analyzer，通过 pygls 通信。`definition` / `references` 接收文件路径和从 1 开始的行、列；`symbols` / `diagnostics` 只需文件路径。服务器缺失、崩溃或响应异常时返回 `available: false`，Agent 可继续使用文本搜索和文件读取。它不执行重命名、格式化或 workspace edit；LSP 诊断用于理解代码，不作为退出前验证证据。

安装 Python 与 TypeScript 语言服务器（Node.js 22.22.2+）：

```bash
npm install --global --ignore-scripts pyright typescript-language-server typescript@6
```

Windows 可将工具和下载缓存放到 D 盘，并将 `D:\Tools\lsp` 加入用户 PATH 后重新打开终端：

```powershell
npm install --global --prefix D:\Tools\lsp --cache D:\Tools\npm-cache --ignore-scripts pyright typescript-language-server typescript@6
```

Python 通信与文件监听依赖 `pygls` 和 `watchfiles` 随 `pip install -e .` 安装。只自动启动 PATH 中位于工作区之外的语言服务器；工作区 `.venv`、`node_modules`、工作区内的 PATH 入口及链接回工作区的程序均不自动执行。Windows 下支持全局 npm 的 `.cmd` 入口，并检查实际 Node 和 JS 文件的位置。TypeScript 固定使用工作区之外安装的 `tsserver`，不注册项目插件；因此语义结果可能与项目本地 TypeScript 版本或插件提供的结果不同。服务器启动时建立源码文件列表，后续查询通过原生文件监听发送增量变更，不再逐次读取并 hash 全仓源码。会话关闭时释放监听器和服务器进程。

语言服务器关闭自动类型包下载、Rust 构建脚本/过程宏执行和保存时编译检查；Go 使用只读模块设置。Rust 导航可能因此缺少构建生成代码或过程宏展开信息。项目的编译和测试按需通过 `run_bash` 执行。

## 工作模式

| Profile | 用途 |
| --- | --- |
| `general` | 普通问答和只读仓库检查 |
| `coding-agent` | 修改代码、测试和验证 |
| `plan` | 调查和生成计划，不直接修改代码 |
| `review` | 只读代码审查 |
| `app-builder` | 构建 Web 应用并进行浏览器验证 |
| `terminal` | Terminal-Bench / Harbor 任务 |

```bash
veriforge --profile coding-agent "Fix the parser bug"
veriforge --profile plan "Design the parser migration"
veriforge --profile review "Review the current branch"
```

## Runtime 设计

| 模块 | 职责 |
| --- | --- |
| `agent/` | 对话循环、统一上下文生命周期、provider、取消和子代理 |
| `runtime/` | 工具 registry、权限、调度、middleware 和 MCP |
| `workspace/` | 文件保护、快照、Shell 和后台任务 |
| `sessions/` | session metadata、事件、逻辑消息手账和报告 |
| `memory/` | SQLite 事实与修订、主题视图、向量＋全文检索、分阶段后台整理 |
| `profiles/` | 不同任务模式的 prompt、工具面和验收策略 |
| `frontend/opentui/` | Bun + React + TypeScript 终端界面 |
| `eval/` | 基准任务、运行器和结果账本 |

### 执行边界

- `run_bash` 每次从 workspace 根目录启动新的 Shell，不继承上次调用的 cwd、环境变量或函数。
- 文件、目录和全局资源使用统一 effect 声明；未声明 effect 的扩展工具默认独占。
- 同一文件读写、目录读写和验证输出按资源顺序执行；不同资源的安全调用可并行。
- `workspace-write` 默认需要批准 risky Shell 和未知工具；危险删除、覆盖和系统命令直接拒绝。
- worker 只在隔离副本中修改；提案需复查后才能三方合并，冲突不会直接覆盖主工作区。
- Docker 模式用于隔离 Shell，默认关闭网络；它不是绝对安全边界。

## 配置

常用环境变量：

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` | API key |
| `OPENAI_BASE_URL` | OpenAI-compatible API 地址 |
| `HARNESS_MODEL` | 默认模型 |
| `HARNESS_MODEL_INTENSITY` | `fast` / `normal` / `hard` / `max` |
| `HARNESS_ROUTER_API_KEY` | 丑橘 `jev` 分组的独立路由 key |
| `HARNESS_ROUTER_BASE_URL` | 路由 API 根地址，默认 `https://chouju.best/v1` |
| `HARNESS_ROUTER_MODEL` | 路由模型，默认 `jev-latest` |
| `HARNESS_ROUTER_TIMEOUT_SECONDS` | 路由请求超时，默认 3 秒，不重试 |
| `HARNESS_PERMISSION_MODE` | `workspace-write` / `llm-auto` / `danger-full-access` |
| `HARNESS_MODEL_INPUT_MODE` | `text` 或 `multimodal` |

完整配置见 `.env.template`。

自动模式下，每个用户 turn 由 Jev 通过 `/v1/systemone` 的 Choice 问题选择工作模式，并参考上一轮任务和回答。选择 `auto` 时全部任务意图交给 Jev，不再使用关键词规则或 BM25；手动选择其他模式后固定执行，不请求路由模型。专用模式中的普通问答使用 `direct_answer`，保留当前工作上下文。

路由器独立配置 API key、地址和模型，复用会话内的 HTTP 连接，会话关闭时释放连接。主模型和 `fast` 通道的配置不受影响。缺少路由配置、请求失败、响应无效或置信度低于 0.6 时保持当前模式，`profile_route_decision` 记录失败原因、实际模型、置信度和候选概率。路由模型仅判断工作模式，工具权限仍由 Runtime 控制。

会话只在首次执行时初始化，切换标签只读取已保留的界面与历史。连续空闲两分钟后自动释放运行时，下次执行自动恢复；排队消息、审批、后台进程、子代理、未退出工具、未应用的代码提案和待解决冲突都会阻止休眠。项目内共享模型与工具执行额度，任务等待额度时可正常取消。MCP 目录缓存可复用，连接按会话隔离、按需建立。顶部「资源」可查看运行会话、等待原因、后台进程的内存与 CPU 累计时间及最近失败；关闭面板后停止进程资源采样。

Trace 位于 `.harness/traces/<Session ID>/`，每个 Agent 实例单独写入文件，通过 `ui_request_id`、`turn_id`、`llm_call_id`、`tool_call_id`、`shell_job_id` 关联请求与执行。

## 测试

安装 Python 测试依赖并运行日常测试：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

默认执行完整非外部测试，排除标记为 `integration` 的原生工具测试和 `external` 的外部服务验收；单元测试和使用仓库内假服务器的 LSP 协议测试不要求安装真实语言服务器。测试统一使用假模型凭据，不依赖开发机的 API key；搜索参数与过滤规则通过 mock 验证，不要求安装 `rg`。

日常可只跑快速回归；本轮涉及的 Session 集成测试按需追加，提交前仍跑完整非外部测试：

```bash
python -m pytest -q -m "not integration and not session and not external"
python -m pytest -q tests/test_multi_session_workbench.py tests/test_project_resources.py
python -m pytest -q --durations=30
```

真实 Windows ACL、LSP 与 Node.js 执行边界测试单独运行：

```bash
python -m pytest -q -m integration -rs
```

本地缺少 Node.js、Python/TypeScript 语言服务器或外部 TypeScript 编译器时，对应测试明确跳过并显示原因。工具安装方法见上文；Windows 可继续使用 `D:\Tools\lsp`，安装目录需要加入当前终端的 `PATH`。跳过不代表验证通过。

`.github/workflows/python-tests.yml` 在 Linux 和 Windows 分别运行日常测试与真实 LSP 测试。LSP 任务固定 Node.js `24.18.0`、Pyright `1.1.414`、typescript-language-server `6.0.1` 和 TypeScript `6.0.3`，在 runner 临时目录中安装，位于工作区之外。安装和版本检查在测试前完成；测试不自动下载工具。CI 使用严格命令，缺少工具直接失败：

```bash
python -m pytest tests/test_code_intelligence.py tests/test_language_service_integration.py -q -m integration --require-integration-tools -rs
# Windows
python -m pytest tests/test_native_shell.py -q -m integration --require-integration-tools -rs
```

OpenTUI：

```bash
cd frontend/opentui
bun test
bun run check
```

## 评测

```bash
python eval/scripts/run_profile_router_eval.py
python eval/scripts/run_basic_metrics_eval.py --dry-run
python eval/scripts/run_terminal_bench_eval.py --dry-run
python eval/scripts/rebuild_eval_results.py --results-root eval/results --jobs-root jobs
```

Profile 路由评测会用真实接口检查 38 条固定用例，默认每 5.2 秒发起一次请求以满足丑橘每分钟 12 次的限制，结果保存到 `eval/local_results/profile_router_jev.json`。

评测结果和运行说明：

- [eval/README.md](eval/README.md)
- [eval/benchmarks/README.md](eval/benchmarks/README.md)
- [eval/results/SUMMARY.md](eval/results/SUMMARY.md)

## 相关文档

- [面试问答](docs/interview-qa.md)
- [环境变量模板](.env.template)

## 注意

- 不要提交 `.env` 或 API key。
- `app-builder` 的浏览器验证需要 Playwright Chromium。
- `terminal` 仅用于显式 benchmark 任务，不参与普通 profile 自动路由。
- 被中断的评测应标记为 interrupted/incomplete，不要当作最终结果。
