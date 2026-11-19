# VeriForge

VeriForge 是一个在本地代码仓库里工作的 Coding Agent。它把模型调用、工具执行、权限审批、会话记录和改动验证串在同一条可追踪的链路上，适合修 Bug、补测试、做代码审查和规划实现。

它通过 OpenAI 兼容的 Chat Completions API 调用模型，默认按 DeepSeek 调优，也可以接入 OpenAI 或其他兼容网关。

![VeriForge OpenTUI](https://raw.githubusercontent.com/raydemo1/veriforge-agent/main/docs/images/veriforge-tui.png)

## 特性

- **单一通用 Agent**：没有按任务类型切换的 profile。行为差异只来自权限模式和显式的 `/plan` 计划流。
- **统一的工具治理**：文件、Shell、Web、MCP 和子 Agent 都经过同一套注册表、参数校验、权限策略和中间件。
- **安全的 Shell**：先静态解析命令、再按事实做权限决定，最后放进平台沙箱（Windows ACL / Linux bubblewrap）执行。
- **并行调度**：无资源冲突的工具调用并行执行，冲突的按顺序执行，结果按模型给出的顺序回写。
- **自动验证**：每轮结束后检查本轮改动的文件（Python AST / Ruff、TypeScript `tsc`、`gofmt`），结果写入会话记录。
- **代码理解**：只读的 `code_intelligence` 工具按需启动语言服务器，提供定义、引用、符号和诊断。
- **可恢复的会话**：JSONL 会话日志、上下文压缩、回撤到任意稳定回合、分支会话和隔离 Worktree。
- **长期记忆**：SQLite 存储的事实与修订历史，全文 + 向量混合检索，后台增量提取与整理。
- **终端界面**：基于 Bun + React 的 OpenTUI，支持多会话标签、`/` 命令与 `@` 补全、审批面板和运行观察。

## 快速开始

### 环境要求

- Python 3.14（Windows 需要 64 位）
- [Bun](https://bun.sh) 1.4+（终端界面和 Windows 沙箱需要）
- Git
- 一个 OpenAI 兼容的 API key
- Linux 上的受限 Shell 需要安装 `bwrap`（bubblewrap）；Windows 需要 PowerShell 7

### 安装

Linux / macOS：

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
bun install --frozen-lockfile --cwd frontend/opentui
```

Windows PowerShell：

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
bun install --frozen-lockfile --cwd frontend/opentui
```

### 配置

```bash
cp .env.template .env
```

至少填写 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。所有可用变量及说明都在 [.env.template](.env.template) 里，常用项见下文 [配置](#配置)。

### 运行

在你要处理的项目目录里启动：

```bash
veriforge                              # 打开终端界面
veriforge "修复失败的测试"               # 打开终端界面并立即提交第一个任务
veriforge -p "审查这个仓库"              # 无界面执行一个任务，输出结果后退出
```

没有 TTY 时（例如管道输入），`veriforge` 会自动以无界面模式运行，并从 stdin 读取任务。

其他子命令：

```bash
veriforge session show latest          # 查看最近一次会话的摘要
veriforge session observe latest       # 查看会话的运行观察（加 --export 导出）
veriforge mcp list                     # 管理 MCP 服务：add / list / remove / describe
veriforge --setup-tui                  # 从 wheel 安装时，准备终端界面的 Bun 依赖
```

### 在代码中调用

命令行的 `-p` 模式和评测 runner 都使用同一个 Headless API：

```python
from harness_code_agent.headless import run_task

result = run_task(cwd="/path/to/project", task="Fix the failing tests")
print(result.session_id, result.status)       # completed / failed / interrupted
raise SystemExit(result.exit_code)            # 0 / 1 / 130
```

`completed` 只表示本轮正常结束，任务是否真正完成仍由测试或评测器判断。无界面运行不会从 stdin 请求审批：需要人工审批的操作直接拒绝，向用户提问按取消处理。

## 使用

### 命令

| 命令 | 作用 |
| --- | --- |
| `/plan` | 只调查、产出计划，不修改代码 |
| `/plan execute` | 执行已批准的计划 |
| `/compact` | 压缩当前对话上下文 |
| `/context` | 查看上下文预算的组成 |
| `/memory` | 查看和管理长期记忆，见 [docs/memory.md](docs/memory.md) |
| `/mcp` | 管理 MCP 服务和工具 |
| `/fork` | 从当前会话创建分支 |
| `/observe` | 打开当前项目的运行观察 |
| `/<skill>` | 直接加载一个内置 Skill，例如 `/tdd`、`/code-review` |

输入 `/` 或 `@` 会弹出补全。`@file:路径` 把文件作为附件，`@session:ID` 引用历史会话。

### 权限模式

通过 `HARNESS_PERMISSION_MODE` 设置，也可以在界面里切换：

| 模式 | 行为 |
| --- | --- |
| `workspace-write`（默认） | 工作区内的写入和未知程序在沙箱内直接执行；网络访问、破坏性 Git、大范围删除和额外权限需要你审批；工作区外写入默认拒绝 |
| `read-only` | 只允许读取。编辑工具和会写入的 Shell 命令一律拒绝 |
| `llm-auto` | 沙箱和策略与 `workspace-write` 相同，但需要审批的操作交给一个审查模型决定；审查失败按拒绝处理 |
| `danger-full-access` | 不使用沙箱，只拦截系统 / 磁盘级破坏。仅用于评测容器 |
| `terminal-eval` | Terminal-Bench 独立 runner 使用的预设 |

无论哪种模式，系统级破坏（如 `rm -rf /`、格式化磁盘）都会被拒绝。细节见 [docs/permissions-and-sandbox.md](docs/permissions-and-sandbox.md)。

### 计划流

需要先想清楚再动手时使用 `/plan`。规划期间运行时把权限收紧到 `read-only`，这是强制约束而不是提示词约定。计划就绪后：

- 继续输入文字会被当作修改意见，计划随之更新，不会开始执行；
- `/plan execute` 或计划面板上的确认才会开始执行，并恢复原来的权限模式；
- 如果执行没能真正开始（模型不可用、被取消等），计划保持待审批状态，可以再次执行。

### 多会话与工作区

一个工作台可以同时运行多个会话，每个会话有自己的对话、队列、审批和后台任务。切换标签不会中断执行；会话空闲两分钟后自动释放运行时，下次使用时恢复。

- **共享工作区（默认）**：所有会话改同一个目录。不同文件可以并行修改；同一文件写入前会校验版本，冲突时先尝试自动三方合并，无法安全合并时展示差异等你确认。
- **隔离工作区**：从当前会话 Fork 出一个 Git Worktree 会话，继承未提交的改动；完成后用「应用到 Local」把改动合回主工作区。
- **回撤**：每个稳定回合结束后保存回撤点，可以回到之前任意回合。回撤只撤销本会话自己的文件改动；如果文件之后又被其他会话或外部程序修改，回撤会被拒绝。数据库、远端请求等外部副作用不在回撤范围内。

更多细节见 [docs/context-and-sessions.md](docs/context-and-sessions.md)。

## 配置

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 模型服务的凭据和地址 |
| `HARNESS_PROVIDER` | `auto`（默认，按域名识别）、`openai`、`deepseek` 或 `openai-compatible` |
| `HARNESS_MODEL` | 非 DeepSeek provider 使用的模型名 |
| `HARNESS_MODEL_INTENSITY` | 模型档位：`fast` / `normal` / `hard`（默认）/ `max` |
| `HARNESS_MODEL_FAST` … `HARNESS_MODEL_MAX` | 覆盖各档位使用的模型 |
| `HARNESS_AVAILABLE_MODELS` | 界面模型选择器中可选的模型，逗号分隔 |
| `HARNESS_PERMISSION_MODE` | 权限模式，见上文 |
| `HARNESS_MODEL_INPUT_MODE` | `text`（默认）或 `multimodal`（允许发送图片） |
| `HARNESS_CONTEXT_WINDOW_TOKENS` | 上下文窗口大小，默认 200000；达到 80% 时自动压缩 |
| `SILICONFLOW_API_KEY` | 长期记忆向量检索使用的 embedding 凭据（独立于模型 key）|
| `VERIFORGE_HOME` | 用户级状态目录，默认 `~/.veriforge` |

完整列表（超时、Shell 资源限制、预算等）见 [.env.template](.env.template)。

运行时状态不写进你的项目目录，而是放在 `~/.veriforge/workspaces/<项目名>-<路径哈希>/` 下：会话日志、Trace、大输出文件和 MCP 配置都在这里。长期记忆在 `~/.veriforge/memory-v3/`。

## 项目结构

```text
harness_code_agent/      Python 运行时
├── support/             叶子层原语：取消令牌、token 计数、LLM 客户端适配
├── workspace/           路径规则、文件保护、Shell 与后台任务、沙箱、Worktree
├── sessions/            会话元数据、事件、JSONL 日志、观测与报告
├── memory/              长期记忆：SQLite、检索、后台提取与整理
├── runtime/             工具注册与执行、权限、调度、中间件、MCP、LSP、验证
├── agent/               对话循环、上下文管理、压缩、子 Agent
├── core/ · tui/         交互会话与终端界面的 Python 侧逻辑
├── skills/catalog/      内置 Skill（Markdown）
├── cli.py · headless.py · tui_bridge.py    入口
frontend/opentui/        终端界面（Bun + React + TypeScript）
eval/                    评测脚本、基准适配器与报告
tests/                   测试
docs/                    设计文档
```

## 开发

```bash
pip install -e ".[test]"
python -m pytest -q                          # Python 测试
cd frontend/opentui && bun run check && bun test   # 前端类型检查与测试
```

测试分层、集成测试、CI 和代码约定见 [docs/development.md](docs/development.md)。

## 文档

- [架构](docs/architecture.md)：模块分层、一轮对话和一次工具调用的完整路径、资源生命周期
- [权限与沙箱](docs/permissions-and-sandbox.md)：Shell 的四层安全边界、权限模式、额外权限与持久授权
- [上下文与会话](docs/context-and-sessions.md)：上下文版本、压缩、会话日志、多会话、回撤与 Worktree
- [长期记忆](docs/memory.md)：数据模型、检索、自动提取与管理命令
- [开发指南](docs/development.md)：环境、测试、CI 与约定
- [评测](eval/README.md)：本地指标、Terminal-Bench 2.1 / 4.1 与历史评测报告

## 许可证

[MIT](LICENSE)
