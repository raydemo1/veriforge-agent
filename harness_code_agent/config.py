"""
Harness configuration.
Uses OpenAI-compatible API so it works with any provider.

Setup:
  cp .env.template .env   # then fill in your real values
"""
import os
from dataclasses import dataclass
from pathlib import Path

from .provider_resolution import resolve_provider_name as _resolve_provider_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv():
    """Load .env file if it exists. No third-party dependency needed."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # .env takes priority over shell env vars
        if key:
            os.environ[key] = value


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
AVAILABLE_MODELS = ("deepseek-v4-flash", "deepseek-v4-flash-vision-exp", "deepseek-v4-pro")
REASONING_EFFORTS = ("low", "high", "max")

# Runtime selection made from the TUI; None means "use the intensity preset".
_MODEL_OVERRIDE = {"model": None, "reasoning_effort": None}


def set_model_override(model: str | None = None, reasoning_effort: str | None = None) -> None:
    if model is not None and model not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model!r}; expected one of: {', '.join(AVAILABLE_MODELS)}")
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            f"Unsupported reasoning effort: {reasoning_effort!r}; expected one of: {', '.join(REASONING_EFFORTS)}"
        )
    _MODEL_OVERRIDE["model"] = model
    _MODEL_OVERRIDE["reasoning_effort"] = reasoning_effort


def get_model_override() -> dict[str, str | None]:
    return dict(_MODEL_OVERRIDE)


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
            "fast": ModelProfile(provider=provider, model="deepseek-v4-flash", thinking=False),
            "normal": ModelProfile(
                provider=provider,
                model="deepseek-v4-flash",
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


def resolve_model_profile(intensity: str) -> ModelProfile:
    normalized = _normalize_model_intensity(intensity)
    override_model = _model_override_for_intensity(normalized)
    provider = _resolve_provider_name(
        provider=PROVIDER,
        base_url=BASE_URL,
        model=override_model or BASE_MODEL,
    )
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
    runtime_override = get_model_override()
    if normalized != "fast" and (runtime_override["model"] or runtime_override["reasoning_effort"]):
        model = runtime_override["model"] or profile.model
        provider = _resolve_provider_name(provider=PROVIDER, base_url=BASE_URL, model=model)
        profile = ModelProfile(
            provider=provider,
            model=model,
            thinking=True if runtime_override["reasoning_effort"] else profile.thinking,
            reasoning_effort=runtime_override["reasoning_effort"] or profile.reasoning_effort or "high",
            input_mode=MODEL_INPUT_MODE,
        )
    return profile


MODEL_INTENSITY = _normalize_model_intensity(os.environ.get("HARNESS_MODEL_INTENSITY", "hard"))
MODEL_PROFILES = {intensity: resolve_model_profile(intensity) for intensity in MODEL_INTENSITIES}
MODEL_PROFILE = MODEL_PROFILES[MODEL_INTENSITY]
MODEL = MODEL_PROFILE.model

# --- Token budgets ---
CONTEXT_WINDOW_TOKENS = int(os.environ.get("HARNESS_CONTEXT_WINDOW_TOKENS", "200000"))
COMPRESS_THRESHOLD = int(os.environ.get("COMPRESS_THRESHOLD", str(int(CONTEXT_WINDOW_TOKENS * 0.85))))

# Tool output larger than this (chars) is stored to file; only a compact
# ref + head+tail preview goes into the message list.  Smaller outputs are
# inlined as usual.
TOOL_OUTPUT_INLINE_LIMIT = int(os.environ.get("HARNESS_TOOL_OUTPUT_INLINE_LIMIT", "4000"))

# --- Agent limits ---
MAX_AGENT_ITERATIONS = int(os.environ.get("MAX_AGENT_ITERATIONS", "60"))
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
# preview + artifact ref + total size.
SHELL_OUTPUT_PREVIEW_CHARS = int(
    os.environ.get("HARNESS_SHELL_OUTPUT_PREVIEW_CHARS", "8192")
)

# --- Workspace write quota ---
# Cumulative structured writes (write_file/apply_patch) allowed per session;
# 0 disables the quota.  Writes also fail when free disk space drops below
# HARNESS_MIN_FREE_DISK_MB.
WORKSPACE_WRITE_QUOTA_BYTES = (
    int(os.environ.get("HARNESS_WORKSPACE_WRITE_QUOTA_MB", "0")) * 1024 * 1024
)
MIN_FREE_DISK_BYTES = int(os.environ.get("HARNESS_MIN_FREE_DISK_MB", "100")) * 1024 * 1024

# --- Shell process resource limits ---
# Docker sandbox: passed to `docker run` as --memory / --cpus; 0 leaves them unset.
DOCKER_MEMORY_MB = int(os.environ.get("HARNESS_DOCKER_MEMORY_MB", "0"))
DOCKER_CPUS = float(os.environ.get("HARNESS_DOCKER_CPUS", "0"))
# POSIX host sandbox: injected as ulimit -v (address space KB), -t (CPU
# seconds), -f (single-file blocks) before each command.  All default to 0
# (disabled) because address-space caps can false-kill JVM/Node workloads.
SHELL_MEMORY_KB = int(os.environ.get("HARNESS_SHELL_MEMORY_KB", "0"))
SHELL_CPU_SECONDS = int(os.environ.get("HARNESS_SHELL_CPU_SECONDS", "0"))
SHELL_FSIZE_BLOCKS = int(os.environ.get("HARNESS_SHELL_FSIZE_BLOCKS", "0"))

TRACE_STDERR = os.environ.get("HARNESS_TRACE_STDERR", "").lower() in {"1", "true", "yes", "on"}
WINDOWS_SHELL = os.environ.get("HARNESS_WINDOWS_SHELL", "pwsh")
SANDBOX_MODE = os.environ.get("HARNESS_SANDBOX_MODE", "host")
DOCKER_IMAGE = os.environ.get("HARNESS_DOCKER_IMAGE", "python:3.12")
DOCKER_NETWORK = os.environ.get("HARNESS_DOCKER_NETWORK", "none")
DOCKER_USER = os.environ.get("HARNESS_DOCKER_USER", "")

# --- Paths ---
WORKSPACE = os.path.abspath(os.environ.get("HARNESS_WORKSPACE", "./workspace"))
PROGRESS_FILE = "progress.md"
