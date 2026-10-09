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

每轮对话结束后会自动保存回撤点。鼠标悬停到该轮对话，或用键盘聚焦该轮，即可显示「回撤到此处」。确认后保留这一轮，对话和文件一起恢复到该轮结束时的状态；原来的后续历史保留在原会话中。「会话开始」也支持回撤，可撤回第一轮。首次允许修改前保存原始状态，相同内容的快照自动复用。

回撤使用 `.harness/recovery` 中独立的文件快照，不创建 Git commit，也不修改 Git 暂存区。源码、未跟踪文件、二进制文件和 shell 修改均纳入快照；`.git`、`.harness`、依赖目录、虚拟环境、缓存及 `build`/`dist` 目录排除。硬链接、目录联接和特殊文件会阻止快照保存。数据库、远端操作等外部副作用不在恢复范围内。后台命令和子代理需先停止；手动修改的覆盖风险会显示在同一次确认中。恢复事务中断后，下次启动会先恢复事务前的文件状态。

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
| `memory/` | Markdown 记忆正文、BM25 检索、适用性验证和后台提炼 |
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
| `HARNESS_WINDOWS_SHELL` | `pwsh` 或 `wsl` |
| `HARNESS_SANDBOX_MODE` | `host` 或 `docker` |
| `HARNESS_MODEL_INPUT_MODE` | `text` 或 `multimodal` |

完整配置见 `.env.template`。

自动模式下，每个用户 turn 由 Jev 通过 `/v1/systemone` 的 Choice 问题选择工作模式，并参考上一轮任务和回答。选择 `auto` 时全部任务意图交给 Jev，不再使用关键词规则或 BM25；手动选择其他模式后固定执行，不请求路由模型。专用模式中的普通问答使用 `direct_answer`，保留当前工作上下文。

路由器独立配置 API key、地址和模型，复用会话内的 HTTP 连接，会话关闭时释放连接。主模型和 `fast` 通道的配置不受影响。缺少路由配置、请求失败、响应无效或置信度低于 0.6 时保持当前模式，`profile_route_decision` 记录失败原因、实际模型、置信度和候选概率。路由模型仅判断工作模式，工具权限仍由 Runtime 控制。

## 测试

安装 Python 测试依赖并运行日常测试：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

默认排除标记为 `integration` 的外部工具测试；单元测试和使用仓库内假服务器的 LSP 协议测试不要求安装真实语言服务器。测试统一使用假模型凭据，不依赖开发机的 API key；搜索参数与过滤规则通过 mock 验证，不要求安装 `rg`。

真实 LSP 与 Node.js 执行边界测试单独运行：

```bash
python -m pytest -q -m integration -rs
```

本地缺少 Node.js、Python/TypeScript 语言服务器或外部 TypeScript 编译器时，对应测试明确跳过并显示原因。工具安装方法见上文；Windows 可继续使用 `D:\Tools\lsp`，安装目录需要加入当前终端的 `PATH`。跳过不代表验证通过。

`.github/workflows/python-tests.yml` 在 Linux 和 Windows 分别运行日常测试与真实 LSP 测试。LSP 任务固定 Node.js `24.18.0`、Pyright `1.1.414`、typescript-language-server `6.0.1` 和 TypeScript `6.0.3`，在 runner 临时目录中安装，位于工作区之外。安装和版本检查在测试前完成；测试不自动下载工具。CI 使用严格命令，缺少工具直接失败：

```bash
python -m pytest -q -m integration --require-integration-tools -rs
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
