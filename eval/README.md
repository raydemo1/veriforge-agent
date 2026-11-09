# Eval

`eval/` is the single home for evaluation code, benchmark adapters, fixed task sets, and generated results.

## Directory Map

```text
eval/
├── scripts/       # Evaluation runners and ledger rebuild tools
├── tasks/         # Fixed lightweight task definitions used by scripts
├── benchmarks/    # External benchmark adapters: Terminal-Bench 2.1 via Harbor, Claw-SWE-Bench
└── results/       # Timestamped raw outputs plus ledger.json/results.json/SUMMARY.md
```

## File Relationships

- `scripts/run_basic_metrics_eval.py` runs local basic metrics suites: memory and latency by default; cache is available when explicitly selected.
- `scripts/run_terminal_bench_eval.py` runs the fixed Terminal-Bench subsets and delegates each task to `benchmarks/run_terminal_bench.py`.
- `scripts/run_claw_swe_bench_eval.py` runs Claw-SWE-Bench through the local harness adapter.
- `scripts/deepseek_context_eval.py` runs the DeepSeek context/cache scenarios and writes timestamped outputs under `results/`.
- `scripts/rebuild_eval_results.py` scans raw `summary.json`, Harbor `result.json`, VeriForge artifacts, stdout, and stderr to regenerate `results/ledger.json`, `results/results.json`, `results/retention_plan.json`, and `results/SUMMARY.md`.
- `tasks/memory_ab.json`, `tasks/latency_smoke.json`, `tasks/terminal_bench_8task.json`, and `tasks/terminal_bench_24task.json` define the lightweight interview-project eval workload.
- `benchmarks/run_terminal_bench.py` launches the fixed Terminal-Bench subset through Harbor.
- `benchmarks/run_claw_swe_bench.py` launches Claw-SWE-Bench through the upstream orchestrator with the local harness adapter.
- `benchmarks/harbor_agent.py` is Harbor's installed-agent adapter for running VeriForge with the `terminal` profile inside task containers.
- `benchmarks/harness_claw_adapter.py` runs VeriForge with the `coding-agent` profile inside each Claw-SWE-Bench container and lets the upstream runner collect patches. Claw remains an external evaluation scenario rather than a separate product profile.
- `benchmarks/tb2_tasks.json` stores Terminal-Bench task metadata used by the terminal profile and launcher.

## Common Commands

```bash
python eval/scripts/run_basic_metrics_eval.py --dry-run
python eval/scripts/run_basic_metrics_eval.py --suites memory,latency
python eval/scripts/run_basic_metrics_eval.py --suites cache
python eval/scripts/run_basic_metrics_eval.py --suites latency
python eval/scripts/run_terminal_bench_eval.py --dry-run
python eval/scripts/run_terminal_bench_eval.py --tbench-task-set 24task
python eval/scripts/run_claw_swe_bench_eval.py --dry-run --claw-limit 5
python eval/scripts/run_claw_swe_bench_eval.py --claw-limit 5
python eval/scripts/rebuild_eval_results.py --results-root eval/results --jobs-root jobs
python eval/benchmarks/run_terminal_bench.py --task fix-git
python eval/benchmarks/run_claw_swe_bench.py --limit 5
```

## A/B tests and final evaluation

### Static verifier pilot

```powershell
python eval/scripts/run_verifier_ablation.py --repeats 2
python eval/scripts/run_verifier_ablation.py --build-image
python eval/scripts/run_verifier_ablation.py --run --repeats 2 --rates 0.006 0.30 1.20
```

The first command prints the plan. The last command uses the configured model
and makes billable requests. Set `--rates` to the cache-hit input, cache-miss
input, and output prices in USD per million tokens. The example applies one
DeepSeek Flash price schedule to every trial. Local tool calls have zero
marginal billing by default. `--tool-cost` can assign a flat USD rate per call.

Both pilot commands reproduce the historical comparisons using the runtime
archived from `b2391f3`. The checkout must include that commit. Each result
records the full runtime revision alongside the model and container image.

Each block pairs A (verifier disabled), B (evidence only), and C (current
exit behavior) on a clean copy of the same fixture. All three run the pinned
`coding-agent` profile with the same budgets and permissions inside Docker.
The runner caps tool calls and trial wall time. There is no cumulative token cap.
The agent container receives the fixture and prompt. The hidden tests run
after it exits in a separate container with no network access.

`ablation.json` contains the plan, group metrics, and per-trial evidence.
Cost per success includes failed trials and is left blank if any usage or
judge result is missing. Latency measures agent execution and excludes setup
and hidden tests. Timed-out runs are counted separately. Repeated tests count
exact command repeats among recognizable test commands. Requests after a C
intervention are observed overhead. The paired B/C comparison estimates the
effect on total requests. In C, the first interrupted candidate is graded
separately to flag already-passing code and possible rescues or regressions.
These fixtures are a pilot for the measurement path and cover Python only.

### Profile prompt pilot

```powershell
python eval/scripts/run_verifier_ablation.py --experiment profile-prompt --run --repeats 2 --rates 0.006 0.30 1.20
```

Both groups use passive verification and the same model, permissions, tools,
and budgets. P0 uses the original `coding-agent` profile prompt. P1 compresses
that prompt while keeping the todo policy and the shared runtime instructions.
Results include the exact profile prompts, input tokens on the first model
request, and its tool schema hash so the paired conditions can be audited.

### Model configuration

The default `.env` remains the final-evaluation configuration. To run a local
evaluation or tuning script with the separate `.env.ab` model profile, use:

```powershell
python eval/scripts/run_with_ab_model.py eval/scripts/run_memory_candidate_eval.py
python eval/scripts/run_with_ab_model.py eval/scripts/run_memory_candidate_eval.py --run
```

The first command prints the evaluation protocol without making model calls;
`--run` executes it. The wrapper selects the A/B profile for its child process
and its subprocesses; other commands keep the default configuration. Set
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `HARNESS_PROVIDER`, `HARNESS_MODEL`, and
`HARNESS_MODEL_FAST/NORMAL/HARD/MAX` in the ignored `.env.ab` file. The base
URL ends at `/v1`; the client adds `/chat/completions`. A missing A/B file is an
error rather than a fallback to the final model. Record the model and evaluation
conditions with results; A/B tuning outcomes are not final-model measurements.

Claw-SWE-Bench uses the Hugging Face dataset `TokenRhythm/Claw-SWE-Bench`
with the `lite` config by default. It requires Docker SWE-bench images and the
optional host dependency `datasets`; the launcher will clone the upstream
`opensquilla/claw-swe-bench` orchestrator into `.harbor/datasets/` on first use.

Use `eval/results/SUMMARY.md` as the human-facing summary. It should only contain metrics produced from real runs.
The current Terminal-Bench 2.1 ledger reports `56/89` passed (`62.9%`)
with `DeepSeek-V4-Flash-Preview` through VeriForge.
The official reference is `61.8%` for the same model on Terminal-Bench 2.1 at
Max reasoning intensity with the official DeepSeek Harness.
