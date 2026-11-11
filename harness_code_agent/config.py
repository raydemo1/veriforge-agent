"""
Harness configuration.
Uses OpenAI-compatible API so it works with any provider.

Setup:
  cp .env.template .env   # then fill in your real values
"""
import math
import os
from dataclasses import dataclass
from pathlib import Path

from .provider_resolution import resolve_provider_name as _resolve_provider_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv():
    """Load .env (and the A/B overlay) without overriding the real environment.

    Priority, highest first: an explicitly exported process variable, the
    ``.env.ab`` overlay, ``.env``, then the defaults in this module.  A file is
    only a source of defaults — a permission mode or model exported by an eval
    runner must not be silently replaced because the working copy has a .env.
    """
    profile = os.environ.get("HARNESS_EVAL_PROFILE", "").strip().lower()
    if profile not in {"", "ab"}:
        raise ValueError("HARNESS_EVAL_PROFILE must be empty or 'ab'")
    paths = [PROJECT_ROOT / ".env"]
    if profile == "ab":
        overlay = PROJECT_ROOT / ".env.ab"
        if not overlay.is_file():
            raise FileNotFoundError("A/B profile requires .env.ab")
        paths.append(overlay)
    merged = {}
    for env_path in paths:
        if not env_path.exists():
            continue
        entries = {}
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key:
                entries[key] = value
        if profile == "ab" and env_path.name == ".env.ab":
            required = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "HARNESS_PROVIDER", "HARNESS_MODEL"}
            required.update(f"HARNESS_MODEL_{intensity}" for intensity in ("FAST", "NORMAL", "HARD", "MAX"))
            if any(not entries.get(key) for key in required):
                raise ValueError("A/B profile must provide a key, base URL, provider, model, and all intensity models")
        # Later files win over earlier ones; the process environment wins over both.
        merged.update(entries)
    for key, value in merged.items():
        os.environ.setdefault(key, value)


_load_dotenv()

# --- API ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
BASE_MODEL = os.environ.get("HARNESS_MODEL", "gpt-4o")
PROVIDER = os.environ.get("HARNESS_PROVIDER", "auto")
STREAM = os.environ.get("HARNESS_STREAM", "auto")
MODEL_INPUT_MODE = os.environ.get("HARNESS_MODEL_INPUT_MODE", "text").strip().lower()
if MODEL_INPUT_MODE not in {"text", "multimodal"}:
    raise ValueError("HARNESS_MODEL_INPUT_MODE must be 'text' or 'multimodal'")

# A streaming response must make progress periodically.  This is deliberately
# shorter than the general client timeout so a half-open model stream cannot
# occupy the agent worker for several retry windows.
LLM_STREAM_IDLE_TIMEOUT_SECONDS = float(
    os.environ.get("HARNESS_LLM_STREAM_IDLE_TIMEOUT_SECONDS", "60")
)
# Hard ceiling for one non-streaming assistant request, plus bounded transport
# retries (connection errors / 429 / 5xx) handled by the SDK and channel layer.
LLM_REQUEST_TIMEOUT_SECONDS = float(
    os.environ.get("HARNESS_LLM_REQUEST_TIMEOUT_SECONDS", "300")
)
LLM_MAX_RETRIES = int(os.environ.get("HARNESS_LLM_MAX_RETRIES", "2"))

MODEL_INTENSITIES = ("fast", "normal", "hard", "max")
MODEL_OVERRIDES = {
    intensity: os.environ.get(f"HARNESS_MODEL_{intensity.upper()}")
    for intensity in MODEL_INTENSITIES
}

# Models and reasoning efforts selectable from the TUI.  Values follow the
# DeepSeek API: reasoning_effort accepts low/high/max (default high).
AVAILABLE_MODELS = ("deepseek-flash", "deepseek-v4-pro")
REASONING_EFFORTS = ("low", "high", "max")

@dataclass
class ModelSelection:
    model: str | None = None
    reasoning_effort: str | None = None

    def select(self, *, model=None, reasoning_effort=None):
        if model is not None and model not in AVAILABLE_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
        if model is not None:
            self.model = model
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    model: str
    thinking: bool | None = None
    reasoning_effort: str | None = None
    input_mode: str = "text"


def _normalize_model_intensity(value: str | None) -> str:
    intensity = (value or "hard").strip().lower()
    if intensity not in MODEL_INTENSITIES:
        allowed = ", ".join(MODEL_INTENSITIES)
        raise ValueError(f"Unsupported HARNESS_MODEL_INTENSITY: {value!r}; expected one of: {allowed}")
    return intensity


def _model_override_for_intensity(intensity: str) -> str | None:
    return MODEL_OVERRIDES.get(intensity)


def _default_model_profile(provider: str, intensity: str) -> ModelProfile:
    if provider == "deepseek":
        defaults = {
            "fast": ModelProfile(provider=provider, model="deepseek-flash", thinking=False),
            "normal": ModelProfile(
                provider=provider,
                model="deepseek-flash",
                thinking=True,
                reasoning_effort="high",
            ),
            "hard": ModelProfile(
                provider=provider,
                model="deepseek-v4-pro",
                thinking=True,
                reasoning_effort="high",
            ),
            "max": ModelProfile(
                provider=provider,
                model="deepseek-v4-pro",
                thinking=True,
                reasoning_effort="max",
            ),
        }
        return defaults[intensity]
    return ModelProfile(provider=provider, model=BASE_MODEL)


def resolve_model_profile(intensity: str, selection: ModelSelection | None = None) -> ModelProfile:
    normalized = _normalize_model_intensity(intensity)
    override_model = _model_override_for_intensity(normalized)
    provider = _resolve_provider_name(provider=PROVIDER, base_url=BASE_URL)
    profile = _default_model_profile(provider, normalized)
    if override_model:
        profile = ModelProfile(
            provider=profile.provider,
            model=override_model,
            thinking=profile.thinking,
            reasoning_effort=profile.reasoning_effort,
            input_mode=MODEL_INPUT_MODE,
        )
    elif profile.input_mode != MODEL_INPUT_MODE:
        profile = ModelProfile(
            provider=profile.provider,
            model=profile.model,
            thinking=profile.thinking,
            reasoning_effort=profile.reasoning_effort,
            input_mode=MODEL_INPUT_MODE,
        )
    selection = selection or ModelSelection()
    runtime_override = {"model": selection.model, "reasoning_effort": selection.reasoning_effort}
    if normalized != "fast" and (runtime_override["model"] or runtime_override["reasoning_effort"]):
        model = runtime_override["model"] or profile.model
        provider = _resolve_provider_name(provider=PROVIDER, base_url=BASE_URL)
        profile = ModelProfile(
            provider=provider,
            model=model,
            thinking=True if runtime_override["reasoning_effort"] else profile.thinking,
            reasoning_effort=runtime_override["reasoning_effort"] or profile.reasoning_effort or "high",
            input_mode=MODEL_INPUT_MODE,
        )
    return profile


MODEL_INTENSITY = _normalize_model_intensity(os.environ.get("HARNESS_MODEL_INTENSITY", "hard"))
MODEL_PROFILE = resolve_model_profile(MODEL_INTENSITY)
MODEL = MODEL_PROFILE.model

# --- Token budgets ---
CONTEXT_WINDOW_TOKENS = int(os.environ.get("HARNESS_CONTEXT_WINDOW_TOKENS", "200000"))
COMPRESS_THRESHOLD = int(os.environ.get("COMPRESS_THRESHOLD", str(int(CONTEXT_WINDOW_TOKENS * 0.80))))

TOOL_OUTPUT_MAX_TOKENS = max(1000, int(os.environ.get("HARNESS_TOOL_OUTPUT_MAX_TOKENS", "5000")))

# --- Agent limits ---
MAX_AGENT_TOTAL_TOKENS = int(os.environ.get("MAX_AGENT_TOTAL_TOKENS", "0"))
MAX_AGENT_TOOL_CALLS = int(os.environ.get("MAX_AGENT_TOOL_CALLS", "200"))
AGENT_BUDGET_WARN_FRACTION = float(os.environ.get("AGENT_BUDGET_WARN_FRACTION", "0.8"))

# --- Subagent fan-out limits ---
MAX_CONCURRENT_AGENTS = max(1, int(os.environ.get("HARNESS_MAX_CONCURRENT_AGENTS", "3")))
MAX_OPEN_AGENTS = max(1, int(os.environ.get("HARNESS_MAX_OPEN_AGENTS", "8")))

# --- Tool execution safety nets ---
# There is deliberately no global per-tool wall-clock deadline: timeout is a
# capability of the concrete execution backend, not a universal tool semantic.
# run_bash owns its own timeout (default below, hard ceiling enforced by clamp;
# the backend kills the process tree when it fires).
SHELL_DEFAULT_TIMEOUT_SECONDS = float(
    os.environ.get("HARNESS_SHELL_DEFAULT_TIMEOUT_SECONDS", "300")
)
TOOL_MAX_TIMEOUT_SECONDS = float(
    os.environ.get("HARNESS_TOOL_MAX_TIMEOUT_SECONDS", "1800")
)
# Host-memory safety for shell output capture: the reader drains continuously
# (commands are never interrupted for producing too much output), keeping at
# most this many characters of head+tail preview in memory.  Output larger
# than the budget is streamed to an artifact file and the ToolResult carries
# capture metadata. Final model messages use TOOL_OUTPUT_MAX_TOKENS.
SHELL_OUTPUT_PREVIEW_CHARS = int(
    os.environ.get("HARNESS_SHELL_OUTPUT_PREVIEW_CHARS", "8192")
)
# Shell job lifecycle.  A foreground command's direct process exiting does not
# mean the job is over: descendants that outlive it are given this long to
# drain and are then terminated, so one tool call cannot hang until its own
# timeout.  Background jobs instead follow their descendants for as long as
# they live, and reader threads are never joined without a bound.
def _finite_seconds(name: str, default: float, *, maximum: float = 3600.0) -> float:
    """A duration that a deadline comparison can actually use.

    ``float("nan")`` would make every deadline comparison false and turn a
    bounded drain back into an unbounded one, so non-finite values are rejected
    at import instead of at the first job.
    """
    raw = os.environ.get(name)
    try:
        value = float(default if raw is None else raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number of seconds") from None
    if not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a finite number between 0 and {maximum:g}")
    return value


SHELL_JOB_EXIT_DRAIN_SECONDS = _finite_seconds("HARNESS_SHELL_JOB_EXIT_DRAIN_SECONDS", 2.0)
SHELL_JOB_TERMINATION_CONFIRM_SECONDS = _finite_seconds("HARNESS_SHELL_JOB_TERMINATION_CONFIRM_SECONDS", 3.0)
SHELL_JOB_READER_JOIN_SECONDS = _finite_seconds("HARNESS_SHELL_JOB_READER_JOIN_SECONDS", 5.0)

# --- Workspace write quota ---
# Cumulative structured writes (write_file/apply_patch) allowed per session;
# 0 disables the quota.  Writes also fail when free disk space drops below
# HARNESS_MIN_FREE_DISK_MB.
WORKSPACE_WRITE_QUOTA_BYTES = (
    int(os.environ.get("HARNESS_WORKSPACE_WRITE_QUOTA_MB", "0")) * 1024 * 1024
)
MIN_FREE_DISK_BYTES = int(os.environ.get("HARNESS_MIN_FREE_DISK_MB", "100")) * 1024 * 1024

# --- Shell process resource limits ---
# POSIX process limits: address space KB, CPU seconds, single-file blocks.
# All default to 0
# (disabled) because address-space caps can false-kill JVM/Node workloads.
SHELL_MEMORY_KB = int(os.environ.get("HARNESS_SHELL_MEMORY_KB", "0"))
SHELL_CPU_SECONDS = int(os.environ.get("HARNESS_SHELL_CPU_SECONDS", "0"))
SHELL_FSIZE_BLOCKS = int(os.environ.get("HARNESS_SHELL_FSIZE_BLOCKS", "0"))

TRACE_STDERR = os.environ.get("HARNESS_TRACE_STDERR", "").lower() in {"1", "true", "yes", "on"}

# --- Paths ---
WORKSPACE = os.path.abspath(os.environ.get("HARNESS_WORKSPACE", "./workspace"))
