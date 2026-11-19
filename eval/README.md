# 评测

`eval/` 存放所有评测代码、基准适配器、固定任务集和历史评测报告。运行产物写入 `eval/results/`，不进入 Git。

所有 runner 都需要 Python 3.14，并先安装评测依赖：

```bash
pip install -e ".[eval]"
```

大部分 runner 默认只打印计划（`--dry-run` 或缺省 `--run`），需要显式参数才会发出计费的真实请求。

## 目录

```text
eval/
├── scripts/      评测入口和结果账本工具
├── tasks/        固定的轻量任务集
├── benchmarks/   外部基准的适配器：Terminal-Bench 2.1 / 4.1（通过 Harbor）
├── reports/      历史评测报告（按日期命名）
└── results/      运行产物和汇总（不进入 Git）
```

## 本地评测

| 脚本 | 作用 |
| --- | --- |
| `scripts/run_basic_metrics_eval.py` | 本地指标套件：`memory`、`latency`（默认），`cache` 需要显式选择 |
| `scripts/run_dev_task_eval.py` | 本地开发任务回归。`--validate` 离线检查夹具，`--run` 调用模型 |
| `scripts/run_verifier_ablation.py` | 改动验证器的消融实验（见下文） |
| `scripts/run_memory_candidate_eval.py` | 记忆候选提取评测 |
| `scripts/check_memory_consolidation.py`、`scripts/check_memory_runtime.py` | 记忆整理与运行时的离线检查 |
| `scripts/deepseek_context_eval.py` | DeepSeek 上下文 / 缓存场景测量 |

```bash
python eval/scripts/run_basic_metrics_eval.py --dry-run
python eval/scripts/run_basic_metrics_eval.py --suites memory,latency
python eval/scripts/run_basic_metrics_eval.py --suites cache
```

### 验证器消融实验

```bash
python eval/scripts/run_verifier_ablation.py --repeats 2                 # 打印计划
python eval/scripts/run_verifier_ablation.py --build-image               # 构建评测镜像
python eval/scripts/run_verifier_ablation.py --run --repeats 2 --rates 0.006 0.30 1.20
```

每组在同一夹具的干净副本上对比三种配置：A 关闭验证器，B 只记录证据，C 使用当前的退出行为。三组在 Docker 中使用相同的 Agent、预算和权限运行；Agent 退出后，隐藏测试在另一个无网络的容器中执行。

- `--rates` 依次是缓存命中输入、缓存未命中输入、输出的价格（美元 / 百万 token）；`--tool-cost` 可以为每次工具调用设一个固定成本。
- 实验复现的是 `b2391f3` 归档的运行时，检出时必须包含该提交。
- runner 限制工具调用次数和单次实验的墙钟时间，不限制累计 token。
- 结果写入 `ablation.json`：每次成功的成本包含失败的实验，任何用量或判定缺失时留空；延迟只计 Agent 执行时间。
- 夹具只覆盖 Python，目前是测量流程的试点。

### A/B 模型配置

默认的 `.env` 是最终评测使用的配置。设置 `HARNESS_EVAL_PROFILE=ab` 后，评测进程会额外加载被 Git 忽略的 `.env.ab`：

```powershell
$env:HARNESS_EVAL_PROFILE = 'ab'
try {
    python eval/scripts/run_memory_candidate_eval.py
} finally {
    Remove-Item Env:HARNESS_EVAL_PROFILE
}
```

`.env.ab` 必须提供 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`HARNESS_PROVIDER`、`HARNESS_MODEL` 以及 `HARNESS_MODEL_FAST/NORMAL/HARD/MAX`，缺失时直接报错，不会回退到最终评测的模型。开发任务 runner 用 `--model-profile ab` 选择这份配置。A/B 调优的结果不能当作最终模型的测量结果。

## Terminal-Bench 2.1

通过 Harbor 运行，需要 Docker（或使用 `--env daytona`）。

```bash
python eval/scripts/run_terminal_bench_eval.py --dry-run
python eval/scripts/run_terminal_bench_eval.py --tbench-task-set 24task      # 固定子集：8task / 24task / full

python eval/benchmarks/run_terminal_bench.py --task fix-git                   # 单个任务
python eval/benchmarks/run_terminal_bench.py --task fix-git --task query-optimize
python eval/benchmarks/run_terminal_bench.py --full                           # 完整数据集
python eval/benchmarks/run_terminal_bench.py --task fix-git --env daytona
python eval/benchmarks/run_terminal_bench.py --task fix-git --force-build     # 强制构建环境镜像
```

运行流程：

1. 需要时下载仓库本地的 `terminal-bench-2-1` 数据集；
2. 保留可拉取的任务镜像，只把失效的 `docker_image` 修复为已知的 Docker Hub 备用镜像；
3. Harbor 通过 `--path` 使用本地数据集，避免任务元数据漂移；
4. `benchmarks/harbor_agent.py` 中的 `HarnessAgent` 在任务容器内安装独立的 Python 3.14 和 VeriForge，任务自带的 Python 命令仍使用镜像中的解释器；
5. 在容器内以 headless 方式运行 Agent，权限模式为 `danger-full-access`（容器本身是隔离边界）；
6. Harbor 用任务的 `tests/test.sh` 判定结果。

任务元数据在 `benchmarks/data/tb2_tasks.json`，由 `benchmarks/tb2_metadata.py` 读取。

当前账本记录：VeriForge + `DeepSeek-V4-Flash-Preview` 在 Terminal-Bench 2.1 上通过 `56/89`（62.9%）。作为参照，官方 DeepSeek Harness 在 Max 推理强度下的同模型成绩为 61.8%。

## Terminal-Bench 4.1（VeriLoop 20 题子集）

VeriLoop 的自进化实验使用 TB4.1 的一个固定 20 题子集，不跑全量。清单、切分（evolve 10 题、held-out 10 题）和挑选规则在 `tasks/veriloop_tb41_20task.json`，全部 66 题的元数据（类别、专家估时、任务和判题环境所需的 CPU/内存/GPU）在 `benchmarks/data/tb41_tasks.json`。

```bash
python eval/scripts/run_terminal_bench_eval.py --tbench-task-set tb41-20task --dry-run
python eval/scripts/run_terminal_bench_eval.py --tbench-task-set tb41-20task                  # 20 题，每题 1 次
python eval/scripts/run_terminal_bench_eval.py --tbench-task-set tb41-20task --split evolve   # 只跑一个切分
python eval/benchmarks/run_terminal_bench.py --benchmark tb4.1 --task music-harmony          # 单个任务
```

和 TB2.1 的区别：

- 直接从 Harbor 注册表运行 `terminal-bench/terminal-bench@4.1.0`，任务名带组织前缀（`terminal-bench/<task>`），不下载本地数据集，也不修复镜像（TB4.1 的任务从自己的 Dockerfile 构建）。
- 判题器在独立环境中运行（`environment_mode = "separate"`）。代理绕行设置通过 Harbor 的 `--ve NO_PROXY=...` 传给判题环境，不再改写 `task.toml`。
- TB4.1 的官方 Agent 超时是 8 小时。评测里 Agent 的预算默认是每题 3600 秒（`--task-budget` 或任务集的 `run_policy.agent_budget_sec`），通过 `HARNESS_TASK_BUDGET` 传进容器；Harbor 自身的 Agent 超时设为预算再加 900 秒，保证 Agent 先自己停下并导出会话。因此结果不能和排行榜上 8 小时的成绩直接比较。
- 子集只包含任务环境和判题环境都不超过 4 核、8 GB、无 GPU 的题目。在 6 核 / 6 GB 的机器上建议并发保持 1（默认值）。

## 结果账本

```bash
python eval/scripts/rebuild_eval_results.py --results-root eval/results --jobs-root jobs
```

它扫描各次运行的 `summary.json`、Harbor 的 `result.json`、VeriForge 产物和标准输出，重新生成 `results/ledger.json`、`results/results.json`、`results/retention_plan.json` 和面向人阅读的 `results/SUMMARY.md`。`SUMMARY.md` 只能包含真实运行产生的指标；被中断的评测要标记为 interrupted / incomplete，不能当作最终结果。

## 历史报告

| 报告 | 内容 |
| --- | --- |
| [2026-10-23-memory-lifecycle](reports/2026-10-23-memory-lifecycle.md) | 记忆分类、主题聚合与更新的问题定位 |
| [2026-10-23-tool-disclosure](reports/2026-10-23-tool-disclosure.md) | 工具披露方式与缓存命中的对比 |
| [2026-10-24-memory-consolidation](reports/2026-10-24-memory-consolidation.md) | 记忆整理与混合检索评测 |
| [2026-10-24-memory-v3-upgrade](reports/2026-10-24-memory-v3-upgrade.md) | 记忆 v3 升级的阶段记录 |
| [2026-10-24-memory-v3-acceptance](reports/2026-10-24-memory-v3-acceptance.md) | 记忆 v3 最终验收 |
| [2026-11-03-multi-session-resources](reports/2026-11-03-multi-session-resources.md) | 多会话资源管理的测量与验收 |
