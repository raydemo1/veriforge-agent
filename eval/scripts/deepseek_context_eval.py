"""Evaluate DeepSeek prompt-cache behavior for the project context lifecycle.

The script intentionally lives under Eval/ and writes all run artifacts under
Eval/results/. It reuses the project's provider adapter, usage normalization,
prompt-cache shape diagnostics, and compaction helpers.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing_extensions import Self

from harness_code_agent import config
from harness_code_agent.agent import context
from harness_code_agent.agent.context_manager import ContextManager
from harness_code_agent.agent.conversation import Agent
from harness_code_agent.agent.providers import current_adapter, get_client
from harness_code_agent.agent.utils import (
    _usage_to_dict,
    capture_prompt_cache_shape,
    compare_prompt_cache_shapes,
)

DEFAULT_CONTEXT_FILES = [
    "README.md",
    "harness_code_agent/agent/context.py",
    "harness_code_agent/agent/conversation.py",
    "harness_code_agent/agent/providers.py",
    "harness_code_agent/agent/utils.py",
    "harness_code_agent/sessions/events.py",
    "harness_code_agent/sessions/observability.py",
]


@dataclass
class EvalCall:
    scenario: str
    turn: int
    kind: str
    label: str
    timestamp: str
    latency_ms: int
    model: str
    provider: str
    message_count: int
    estimated_request_tokens: int
    log_rewrite_version: int
    finish_reason: str | None
    usage: dict[str, Any]
    diagnostics: dict[str, Any]
    response_preview: str

    @property
    def cache_hit_ratio(self) -> float:
        prompt_tokens = int(self.usage.get("prompt_tokens") or 0)
        hit_tokens = int(self.usage.get("cache_hit_tokens") or self.usage.get("cached_tokens") or 0)
        if prompt_tokens <= 0:
            return 0.0
        return hit_tokens / prompt_tokens


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def write(self, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    run_dir = make_run_dir(args)
    write_run_config(run_dir, args)
    print(f"Writing eval artifacts to: {run_dir}")

    evaluator = DeepSeekContextEvaluator(args=args, run_dir=run_dir)
    calls = evaluator.run()
    write_summary(run_dir, calls)
    print_summary(calls, run_dir)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real DeepSeek context/cache evaluations for VeriForge.",
    )
    parser.add_argument(
        "--scenarios",
        default="stable_warmup,schema_reorder,compaction_rewrite",
        help="Comma-separated scenarios: stable_warmup,schema_reorder,compaction_rewrite",
    )
    parser.add_argument("--turns", type=int, default=5, help="Turns for stable_warmup.")
    parser.add_argument(
        "--project-context-tokens",
        type=int,
        default=12_000,
        help="Approximate token budget for real project context included in prompts.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=160,
        help="Maximum completion tokens for measured calls.",
    )
    parser.add_argument(
        "--summary-output-tokens",
        type=int,
        default=800,
        help="Maximum completion tokens for the compaction summarizer call.",
    )
    parser.add_argument(
        "--post-rewrite-turns",
        type=int,
        default=3,
        help="Measured turns to run after compaction rewrite.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "eval" / "results"),
        help="Directory for eval result folders.",
    )
    parser.add_argument("--run-name", default="", help="Optional suffix for the result directory.")
    parser.add_argument(
        "--context-file",
        action="append",
        default=[],
        help="Additional project-relative file to include in the stable context.",
    )
    parser.add_argument(
        "--no-profile-extra",
        action="store_true",
        help="Use model-only chat kwargs instead of the project's resolved model profile extras.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run local checks without calling the API.",
    )
    return parser.parse_args(argv)


class DeepSeekContextEvaluator:
    def __init__(self, *, args: argparse.Namespace, run_dir: Path) -> None:
        self.args = args
        self.run_dir = run_dir
        self.adapter = current_adapter()
        self.client = get_client()
        self.agent = Agent(
            "deepseek-context-eval",
            make_system_prompt(),
            use_tools=True,
            tool_schemas=tool_schemas("canonical"),
        )
        self.project_context = load_project_context(
            target_tokens=args.project_context_tokens,
            extra_files=args.context_file,
        )
        self.calls: list[EvalCall] = []
        self.raw_events = JsonlWriter(run_dir / "raw_events.jsonl")
        self.raw_usage = JsonlWriter(run_dir / "raw_usage.jsonl")

    def run(self) -> list[EvalCall]:
        try:
            for scenario in scenario_names(self.args.scenarios):
                if scenario == "stable_warmup":
                    self.run_stable_warmup()
                elif scenario == "schema_reorder":
                    self.run_schema_reorder()
                elif scenario == "compaction_rewrite":
                    self.run_compaction_rewrite()
                else:
                    raise ValueError(f"Unknown scenario: {scenario}")
        finally:
            self.raw_events.close()
            self.raw_usage.close()
        return self.calls

    def run_stable_warmup(self) -> None:
        schemas = tool_schemas("canonical")
        messages = self.base_messages("stable_warmup")
        previous_shape = None
        for turn in range(1, max(1, self.args.turns) + 1):
            messages.append({
                "role": "user",
                "content": (
                    f"Stable warmup turn {turn}. Inspect the project context above and give "
                    "one concise sentence about the context lifecycle cache behavior. "
                    "Do not call tools."
                ),
            })
            assistant, usage, finish_reason, latency_ms = self.call_model(
                messages,
                schemas,
                max_tokens=self.args.max_output_tokens,
            )
            call, previous_shape = self.record_call(
                scenario="stable_warmup",
                turn=turn,
                kind="measured",
                label=f"stable_warmup_turn_{turn}",
                messages=messages,
                schemas=schemas,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                previous_shape=previous_shape,
                log_rewrite_version=0,
                response_preview=preview(assistant.get("content")),
            )
            messages.append(assistant)
            print_call(call)

    def run_schema_reorder(self) -> None:
        messages = self.base_messages("schema_reorder")
        messages.append({
            "role": "user",
            "content": (
                "Schema reorder probe. Answer in one short sentence and do not call tools."
            ),
        })
        variants = [
            ("canonical_a", tool_schemas("canonical")),
            ("canonical_a_repeat", tool_schemas("canonical")),
            ("reordered_b", tool_schemas("reordered")),
        ]
        previous_shape = None
        for turn, (label, schemas) in enumerate(variants, start=1):
            assistant, usage, finish_reason, latency_ms = self.call_model(
                messages,
                schemas,
                max_tokens=self.args.max_output_tokens,
            )
            call, previous_shape = self.record_call(
                scenario="schema_reorder",
                turn=turn,
                kind="measured",
                label=label,
                messages=messages,
                schemas=schemas,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                previous_shape=previous_shape,
                log_rewrite_version=0,
                response_preview=preview(assistant.get("content")),
            )
            print_call(call)

    def run_compaction_rewrite(self) -> None:
        schemas = tool_schemas("canonical")
        messages = self.base_messages("compaction_rewrite")
        messages.extend(make_old_work_log_messages())
        messages.append({
            "role": "user",
            "content": (
                "Before compaction, summarize one risk in the current context mechanism. "
                "Answer in one short sentence and do not call tools."
            ),
        })

        previous_shape = None
        assistant, usage, finish_reason, latency_ms = self.call_model(
            messages,
            schemas,
            max_tokens=self.args.max_output_tokens,
        )
        call, previous_shape = self.record_call(
            scenario="compaction_rewrite",
            turn=1,
            kind="measured",
            label="before_rewrite",
            messages=messages,
            schemas=schemas,
            usage=usage,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            previous_shape=previous_shape,
            log_rewrite_version=0,
            response_preview=preview(assistant.get("content")),
        )
        print_call(call)
        messages.append(assistant)

        summary_usage_box: list[dict[str, Any]] = []

        def summarize_with_deepseek(summary_messages: list[dict]) -> str:
            summary_assistant, summary_usage, summary_finish, summary_latency = self.call_model(
                summary_messages,
                tools=None,
                max_tokens=self.args.summary_output_tokens,
            )
            summary_usage_box.append(summary_usage)
            summary_call, _ = self.record_call(
                scenario="compaction_rewrite",
                turn=2,
                kind="summarizer",
                label="rewrite_summary_call",
                messages=summary_messages,
                schemas=None,
                usage=summary_usage,
                finish_reason=summary_finish,
                latency_ms=summary_latency,
                previous_shape=None,
                log_rewrite_version=0,
                response_preview=preview(summary_assistant.get("content")),
            )
            print_call(summary_call)
            return str(summary_assistant.get("content") or "")

        current_turn_start = max(1, len(messages) - 2)
        compacted = ContextManager().compact(
            messages,
            summarize_with_deepseek,
            current_turn_start_index=current_turn_start,
            state={},
            force=True,
        )
        messages = compacted.messages if compacted is not None else messages
        for post_turn in range(1, max(1, self.args.post_rewrite_turns) + 1):
            messages.append({
                "role": "user",
                "content": (
                    f"After compaction rewrite turn {post_turn}, state whether the preserved "
                    "summary is enough to continue. Answer in one short sentence and do not call tools."
                ),
            })
            assistant, usage, finish_reason, latency_ms = self.call_model(
                messages,
                schemas,
                max_tokens=self.args.max_output_tokens,
            )
            call, previous_shape = self.record_call(
                scenario="compaction_rewrite",
                turn=2 + post_turn,
                kind="measured",
                label=f"after_rewrite_turn_{post_turn}",
                messages=messages,
                schemas=schemas,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                previous_shape=previous_shape,
                log_rewrite_version=1,
                response_preview=preview(assistant.get("content")),
            )
            messages.append(assistant)
            print_call(call)

    def base_messages(self, scenario: str) -> list[dict]:
        return [
            {"role": "system", "content": self.agent.system_prompt},
            {
                "role": "user",
                "content": (
                    f"[EVAL PROJECT CONTEXT: {scenario}]\n"
                    "The following is real project context sampled from this repository.\n\n"
                    f"{self.project_context}"
                ),
            },
        ]

    def call_model(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        max_tokens: int,
    ) -> tuple[dict, dict[str, Any], str | None, int]:
        chat_args: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self.args.no_profile_extra:
            chat_args["model"] = config.MODEL
        else:
            chat_args["profile"] = config.resolve_model_profile(config.MODEL_INTENSITY)
        if tools is not None:
            chat_args["tools"] = tools
            chat_args["tool_choice"] = "none"

        kwargs = self.adapter.chat_kwargs(**chat_args)
        started = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not response.choices:
            raise RuntimeError("API returned no choices")
        choice = response.choices[0]
        usage = _usage_to_dict(getattr(response, "usage", None)) or {}
        assistant = self.adapter.assistant_message_from_response(choice.message)
        return assistant, usage, getattr(choice, "finish_reason", None), latency_ms

    def record_call(
        self,
        *,
        scenario: str,
        turn: int,
        kind: str,
        label: str,
        messages: list[dict],
        schemas: list[dict] | None,
        usage: dict[str, Any],
        finish_reason: str | None,
        latency_ms: int,
        previous_shape,
        log_rewrite_version: int,
        response_preview: str,
    ) -> tuple[EvalCall, Any]:
        shape = capture_prompt_cache_shape(
            self.agent,
            schemas,
            log_rewrite_version=log_rewrite_version,
        )
        diagnostics = compare_prompt_cache_shapes(previous_shape, shape, usage)
        call = EvalCall(
            scenario=scenario,
            turn=turn,
            kind=kind,
            label=label,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            latency_ms=latency_ms,
            model=config.MODEL,
            provider=self.adapter.name,
            message_count=len(messages),
            estimated_request_tokens=context.count_request_tokens(messages, tool_schemas=schemas),
            log_rewrite_version=log_rewrite_version,
            finish_reason=finish_reason,
            usage=usage,
            diagnostics=diagnostics,
            response_preview=response_preview,
        )
        payload = call_payload(call)
        self.raw_events.write(payload)
        self.raw_usage.write({
            "scenario": call.scenario,
            "turn": call.turn,
            "kind": call.kind,
            "label": call.label,
            "usage": call.usage,
            "cache_hit_ratio": call.cache_hit_ratio,
            "diagnostics": call.diagnostics,
        })
        self.calls.append(call)
        return call, shape


def make_system_prompt() -> str:
    return (
        "You are evaluating the VeriForge context lifecycle. "
        "Keep answers short. Do not claim to have executed tools. "
        "Focus on prompt-cache stability, compaction rewrite effects, and token usage."
    )


def load_project_context(*, target_tokens: int, extra_files: list[str]) -> str:
    files = list(dict.fromkeys(DEFAULT_CONTEXT_FILES + extra_files))
    chunks: list[str] = []
    for rel in files:
        path = PROJECT_ROOT / rel
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.append(f"\n\n===== {rel} =====\n{text}")
        if context.count_tokens([{"role": "user", "content": ''.join(chunks)}]) >= target_tokens:
            break
    joined = "".join(chunks)
    return trim_to_estimated_tokens(joined, max(1, target_tokens))


def trim_to_estimated_tokens(text: str, target_tokens: int) -> str:
    if context.count_tokens([{"role": "user", "content": text}]) <= target_tokens:
        return text
    lo = 0
    hi = len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        tokens = context.count_tokens([{"role": "user", "content": candidate}])
        if tokens <= target_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best + "\n\n[TRUNCATED TO EVAL TOKEN BUDGET]"


def tool_schemas(variant: str) -> list[dict]:
    read_file = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string", "enum": ["utf-8"]},
                },
                "required": ["path", "encoding"],
            },
        },
    }
    search_text = {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search project files for text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern", "path"],
            },
        },
    }
    write_note = {
        "type": "function",
        "function": {
            "name": "write_eval_note",
            "description": "Write an evaluation note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }
    if variant == "reordered":
        return [
            {
                "function": {
                    "parameters": {
                        "required": ["content", "path"],
                        "properties": {
                            "content": {"type": "string"},
                            "path": {"type": "string"},
                        },
                        "type": "object",
                    },
                    "description": "Write an evaluation note.",
                    "name": "write_eval_note",
                },
                "type": "function",
            },
            {
                "function": {
                    "name": "search_text",
                    "parameters": {
                        "required": ["path", "pattern"],
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "pattern": {"type": "string"},
                        },
                    },
                    "description": "Search project files for text.",
                },
                "type": "function",
            },
            {
                "function": {
                    "description": "Read a UTF-8 text file from the workspace.",
                    "parameters": {
                        "properties": {
                            "encoding": {"enum": ["utf-8"], "type": "string"},
                            "path": {"type": "string"},
                        },
                        "required": ["encoding", "path"],
                        "type": "object",
                    },
                    "name": "read_file",
                },
                "type": "function",
            },
        ]
    return [read_file, search_text, write_note]


def make_old_work_log_messages() -> list[dict]:
    repeated = (
        "Context work log: inspected Reasonix-style prefix stability, added cache diagnostics, "
        "normalized DeepSeek usage, stripped response-only reasoning_content before provider calls, "
        "and evaluated token-bounded recent tail plus compaction economics. "
    )
    old_blob = "\n".join(f"{i:03d}. {repeated}" for i in range(140))
    return [
        {"role": "assistant", "content": "Earlier implementation notes follow."},
        {"role": "user", "content": old_blob},
        {"role": "assistant", "content": "The older notes are ready for compaction."},
    ]


def scenario_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names or ["stable_warmup"]


def make_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{safe_name(args.run_name)}" if args.run_name else ""
    run_dir = Path(args.output_root) / f"{timestamp}_deepseek_context_eval{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_config(run_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "provider": current_adapter().name,
        "base_url": config.BASE_URL,
        "model": config.MODEL,
        "model_intensity": config.MODEL_INTENSITY,
        "context_window_tokens": config.CONTEXT_WINDOW_TOKENS,
        "compress_threshold": config.COMPRESS_THRESHOLD,
        "args": {
            key: value
            for key, value in vars(args).items()
            if key != "self_test"
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def write_summary(run_dir: Path, calls: list[EvalCall]) -> None:
    summary = summarize_calls(calls)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    (run_dir / "summary.md").write_text(render_summary_markdown(summary, calls), encoding="utf-8", newline="\n")


def summarize_calls(calls: list[EvalCall]) -> dict[str, Any]:
    by_scenario: dict[str, list[EvalCall]] = {}
    for call in calls:
        if call.kind == "measured":
            by_scenario.setdefault(call.scenario, []).append(call)

    scenarios = {}
    for scenario, items in by_scenario.items():
        prompt_tokens = [int(item.usage.get("prompt_tokens") or 0) for item in items]
        hit_tokens = [int(item.usage.get("cache_hit_tokens") or item.usage.get("cached_tokens") or 0) for item in items]
        miss_tokens = [int(item.usage.get("cache_miss_tokens") or 0) for item in items]
        hit_ratios = [item.cache_hit_ratio for item in items]
        prefix_changes = [
            {
                "turn": item.turn,
                "label": item.label,
                "reasons": item.diagnostics.get("prefix_change_reasons", []),
            }
            for item in items
            if item.diagnostics.get("prefix_changed")
        ]
        scenarios[scenario] = {
            "turns": len(items),
            "avg_hit_ratio": average(hit_ratios),
            "first_hit_ratio": hit_ratios[0] if hit_ratios else 0.0,
            "last_hit_ratio": hit_ratios[-1] if hit_ratios else 0.0,
            "prompt_tokens": prompt_tokens,
            "cache_hit_tokens": hit_tokens,
            "cache_miss_tokens": miss_tokens,
            "prefix_changes": prefix_changes,
            "estimated_request_tokens": [item.estimated_request_tokens for item in items],
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenarios": scenarios,
        "all_calls": [call_payload(call) for call in calls],
    }


def render_summary_markdown(summary: dict[str, Any], calls: list[EvalCall]) -> str:
    lines = [
        "# DeepSeek Context Cache Eval",
        "",
        f"Created at: {summary['created_at']}",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | Turns | Avg Hit Ratio | First -> Last | Prefix Changes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for scenario, item in summary["scenarios"].items():
        changes = item["prefix_changes"]
        change_text = "none" if not changes else ", ".join(
            f"{change['label']}:{'+'.join(change['reasons'])}" for change in changes
        )
        lines.append(
            f"| {scenario} | {item['turns']} | {format_ratio(item['avg_hit_ratio'])} | "
            f"{format_ratio(item['first_hit_ratio'])} -> {format_ratio(item['last_hit_ratio'])} | "
            f"{change_text} |"
        )
    lines.extend(["", "## Calls", ""])
    lines.append("| Scenario | Label | Kind | Prompt | Hit | Miss | Hit Ratio | Prefix Change |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for call in calls:
        usage = call.usage
        reasons = call.diagnostics.get("prefix_change_reasons", [])
        change = "+".join(reasons) if reasons else "none"
        lines.append(
            f"| {call.scenario} | {call.label} | {call.kind} | "
            f"{int(usage.get('prompt_tokens') or 0)} | "
            f"{int(usage.get('cache_hit_tokens') or usage.get('cached_tokens') or 0)} | "
            f"{int(usage.get('cache_miss_tokens') or 0)} | "
            f"{format_ratio(call.cache_hit_ratio)} | {change} |"
        )
    lines.append("")
    return "\n".join(lines)


def call_payload(call: EvalCall) -> dict[str, Any]:
    payload = asdict(call)
    payload["cache_hit_ratio"] = call.cache_hit_ratio
    return payload


def print_summary(calls: list[EvalCall], run_dir: Path) -> None:
    print("\nSummary")
    for call in calls:
        if call.kind != "measured":
            continue
        usage = call.usage
        print(
            f"  {call.scenario}/{call.label}: "
            f"prompt={int(usage.get('prompt_tokens') or 0)} "
            f"hit={int(usage.get('cache_hit_tokens') or usage.get('cached_tokens') or 0)} "
            f"miss={int(usage.get('cache_miss_tokens') or 0)} "
            f"ratio={format_ratio(call.cache_hit_ratio)} "
            f"prefix_change={call.diagnostics.get('prefix_change_reasons', [])}"
        )
    print(f"\nArtifacts: {run_dir}")


def print_call(call: EvalCall) -> None:
    usage = call.usage
    print(
        f"{call.scenario}/{call.label}: "
        f"prompt={int(usage.get('prompt_tokens') or 0)} "
        f"hit={int(usage.get('cache_hit_tokens') or usage.get('cached_tokens') or 0)} "
        f"miss={int(usage.get('cache_miss_tokens') or 0)} "
        f"ratio={format_ratio(call.cache_hit_ratio)} "
        f"prefix_change={call.diagnostics.get('prefix_change_reasons', [])} "
        f"latency={call.latency_ms}ms"
    )


def preview(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def format_ratio(value: float) -> str:
    return f"{value * 100:.1f}%"


def safe_name(value: str) -> str:
    chars = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip()]
    return "".join(chars).strip("_")


def run_self_test() -> None:
    agent = Agent("self-test", make_system_prompt(), use_tools=True)
    shape_a = capture_prompt_cache_shape(agent, tool_schemas("canonical"), log_rewrite_version=0)
    shape_b = capture_prompt_cache_shape(agent, tool_schemas("reordered"), log_rewrite_version=0)
    assert shape_a.tools_hash == shape_b.tools_hash

    fake_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "cache_hit_tokens": 80,
        "cached_tokens": 80,
        "cache_miss_tokens": 20,
    }
    diag = compare_prompt_cache_shapes(shape_a, shape_b, fake_usage)
    assert diag["prefix_changed"] is False
    assert diag["cache_hit_tokens"] == 80
    assert diag["cache_miss_tokens"] == 20

    shape_c = capture_prompt_cache_shape(agent, tool_schemas("canonical"), log_rewrite_version=1)
    rewrite_diag = compare_prompt_cache_shapes(shape_a, shape_c, fake_usage)
    assert rewrite_diag["prefix_changed"] is True
    assert rewrite_diag["prefix_change_reasons"] == ["log_rewrite"]

    sample = "hello " * 1000
    trimmed = trim_to_estimated_tokens(sample, 100)
    assert context.count_tokens([{"role": "user", "content": trimmed}]) <= 120

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        calls = [
            EvalCall(
                scenario="self",
                turn=1,
                kind="measured",
                label="fake",
                timestamp="2026-06-08T00:00:00",
                latency_ms=1,
                model="fake",
                provider="deepseek",
                message_count=2,
                estimated_request_tokens=100,
                log_rewrite_version=0,
                finish_reason="stop",
                usage=fake_usage,
                diagnostics=diag,
                response_preview="ok",
            )
        ]
        write_summary(run_dir, calls)
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "summary.md").exists()
    print("self-test passed")


if __name__ == "__main__":
    raise SystemExit(main())
