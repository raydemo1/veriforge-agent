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

常用入口：

```text
/checkpoint   管理检查点
/mcp          管理 MCP 服务
/compact      压缩上下文
/context      查看上下文预算
/memory       搜索、验证与管理长期记忆
/fork         创建会话分支
/observe      查看运行状态
```

输入 `/` 或 `@` 会打开遮罩补全面板。Enter/Tab 接受候选，点击外层关闭；修改或清空指令后会重新匹配。普通状态面板可点击外层或按 Esc 关闭。

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
| `HARNESS_PERMISSION_MODE` | `workspace-write` / `llm-auto` / `danger-full-access` |
| `HARNESS_WINDOWS_SHELL` | `pwsh` 或 `wsl` |
| `HARNESS_SANDBOX_MODE` | `host` 或 `docker` |
| `HARNESS_MODEL_INPUT_MODE` | `text` 或 `multimodal` |

完整配置见 `.env.template`。

## 测试

Python runtime：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

OpenTUI：

```bash
cd frontend/opentui
bun test
bun run check
```

## 评测

```bash
python eval/scripts/run_basic_metrics_eval.py --dry-run
python eval/scripts/run_terminal_bench_eval.py --dry-run
python eval/scripts/rebuild_eval_results.py --results-root eval/results --jobs-root jobs
```

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
