# 开发指南

## 环境

```bash
python3.14 -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[test]"                     # 运行时 + 测试依赖
pip install -e ".[eval]"                     # 需要跑评测时再装
bun install --frozen-lockfile --cwd frontend/opentui
```

依赖版本只在 `pyproject.toml` 中维护；`requirements.txt` 只是 `-e .` 的兼容入口。

仓库用 `.gitattributes` 统一为 LF 换行（`.cmd` / `.bat` 除外），Windows 上无需设置 `core.autocrlf`。

## 测试

```bash
python -m pytest -q                                          # 默认：全部非集成、非外部测试，按 CPU 数并行
python -m pytest -q -n0 tests/test_x.py::test_y              # 单个测试串行运行，省去启动 worker 的开销
python -m pytest -q -m "not session"                         # 快速反馈：跳过完整 Session 初始化的用例
python -m pytest -q -m integration -rs                       # 需要真实工具的集成测试
```

测试标记（定义在 `pyproject.toml`）：

| 标记 | 含义 | 默认是否运行 |
| --- | --- | --- |
| `session` | 初始化完整 Session、子进程传输或多会话执行，较慢 | 是 |
| `integration` | 需要本机安装的沙箱或语言服务器 | 否 |
| `external` | 显式启用的外部服务验收 | 否 |

- 所有测试使用假的模型凭据，不依赖本机的 API key。
- 每个测试有 60 秒的硬超时，卡住时会直接失败并打印线程栈。
- 测试默认通过 pytest-xdist 并行执行（`-n auto`），每个测试必须能独立运行，不依赖执行顺序或其他测试留下的状态。
- 同一行为的多个输入写进一个表驱动测试（`subTest` 或 `parametrize`），而不是每个输入一个测试函数；需要 git 仓库或会话的测试尽量在同一个夹具里覆盖多个场景。
- 本地缺少集成测试需要的工具时，对应测试会跳过并说明原因。**跳过不等于通过**；CI 使用 `--require-integration-tools`，缺工具直接失败。
- Linux 上的 Shell 测试需要 `bwrap`。沙箱内的 `/tmp` 是私有的，因此测试用的 Python 解释器不能位于 `/tmp` 下。

### 集成测试需要的工具

语言服务器测试需要 Node.js 22.22.2+，以及安装在工作区之外的语言服务器：

```bash
npm install --global --ignore-scripts pyright typescript-language-server typescript@6
python -m pytest tests/test_code_intelligence.py tests/test_language_service_integration.py -q -m integration -rs
```

运行时只会自动启动 `PATH` 中位于工作区之外的语言服务器，工作区内的 `.venv`、`node_modules` 以及链接回工作区的程序都不会被自动执行。

Windows 沙箱测试需要 PowerShell 7 和已安装的 Bun 依赖：

```bash
python -m pytest tests/test_native_shell.py -q -m integration -rs
```

### 前端

```bash
cd frontend/opentui
bun run check        # TypeScript 类型检查
bun test             # 单元测试
bun run dev          # 用模拟数据启动界面，不连接 Python 运行时
```

## CI

`.github/workflows/python-tests.yml` 包含四个任务：

| 任务 | 内容 |
| --- | --- |
| Runtime | Linux 和 Windows 上的默认测试集 |
| OpenTUI frontend | 前端类型检查和单元测试 |
| LSP integration | 固定版本的 Node.js、Pyright、typescript-language-server、TypeScript，安装在 runner 临时目录 |
| Windows ACL integration | 真实的 Windows 沙箱测试 |

## 代码约定

- **分层**：依赖只能自上而下，见 [架构](architecture.md#模块分层)。`tests/test_layering.py` 会检查。
- **异常边界**：`except Exception` 只允许出现在 `pyproject.toml` 中列出的「失败边界」文件里（工具处理、模型调用、插件加载、工作线程、界面桥接等），或者带 `# noqa: BLE001` 和原因注释。ruff 的 BLE001 规则负责检查。
- **前后端协议**：修改 `tui/protocol.py` 或 `frontend/opentui/src/protocol.ts` 时两边同步，并递增 `UI_PROTOCOL_VERSION`。
- **测试替身**：只给测试用的 Provider 等放在 `tests/fakes.py`，不放进生产代码。
- **Lint**：`ruff check .`。

## 打包

```bash
python -m pip wheel --no-deps . -w dist
```

wheel 包含 OpenTUI 源码、依赖锁文件和 Windows ACL 补丁。在目标机器上安装后运行 `veriforge --setup-tui`，它会在 `~/.veriforge/opentui/` 下安装 Bun 依赖并准备 Windows 沙箱依赖。从 wheel 安装时不会读取仓库里的 `.env`，需要通过环境变量设置 `OPENAI_API_KEY` 等配置。

## Skill

内置 Skill 位于 `harness_code_agent/skills/catalog/`，每个目录一个 `SKILL.md`。加载机制和来源说明见 [catalog/README.md](../harness_code_agent/skills/catalog/README.md)。
