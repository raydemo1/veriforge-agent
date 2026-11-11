"""
Harbor adapter — runs our harness agent on Terminal-Bench 2.1 via Harbor framework.

Harbor has two agent types:
  - External (BaseAgent): agent runs outside container, sends commands via environment.exec()
  - Installed (BaseInstalledAgent): agent is installed inside the container

We use Installed agent — a small headless runner executes natively inside the
container, so run_bash just works as subprocess without any bridging.

Usage:
  # Install harbor
  pip install harbor

  # Test on hello-world task
  harbor run -d "terminal-bench@2.1" \
    --agent-import-path eval.benchmarks.harbor_agent:HarnessAgent \
    --task-names hello-world

  # Full benchmark
  harbor run -d "terminal-bench@2.1" \
    --agent-import-path eval.benchmarks.harbor_agent:HarnessAgent

  # With Daytona (no Docker needed locally)
  harbor run -d "terminal-bench@2.1" \
    --agent-import-path eval.benchmarks.harbor_agent:HarnessAgent \
    --env daytona
"""
from __future__ import annotations

import shlex
import shutil
import tempfile
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from eval.benchmarks.harbor_env import runner_env_vars
from eval.benchmarks.usage_metrics import parse_eval_metrics_from_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_STANDALONE_TARBALL = "python-3.12.13-x86_64-unknown-linux-gnu.tar.gz"
CONTAINER_PYTHON_TARBALL = f"/tmp/{PYTHON_STANDALONE_TARBALL}"
DEFAULT_DEBIAN_APT_MIRROR = "http://mirrors.tuna.tsinghua.edu.cn"
DEFAULT_UBUNTU_APT_MIRROR = "http://mirrors.tuna.tsinghua.edu.cn/ubuntu"
DEFAULT_APT_MIRROR_PAIRS = (
    (DEFAULT_DEBIAN_APT_MIRROR, DEFAULT_UBUNTU_APT_MIRROR),
    ("http://mirrors.aliyun.com", "http://mirrors.aliyun.com/ubuntu"),
    ("http://mirrors.ustc.edu.cn", "http://mirrors.ustc.edu.cn/ubuntu"),
    ("http://deb.debian.org", "http://archive.ubuntu.com/ubuntu"),
)
DEFAULT_PYPI_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
DEFAULT_NO_PROXY_HOSTS = (
    "localhost",
    "127.0.0.1",
    "host.docker.internal",
    "http.docker.internal",
    "astral.sh",
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "raw.githubusercontent.com",
    "pypi.org",
    "pypi.python.org",
    "files.pythonhosted.org",
    "download.pytorch.org",
    "download-r2.pytorch.org",
    "pytorch.s3.amazonaws.com",
    "s3.amazonaws.com",
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
    "us.aws.cdn.hf.co",
    "eu-central-1.aws.cdn.hf.co",
    "cas-bridge.xethub.hf.co",
    "cas-server.xethub.hf.co",
    "transfer.xethub.hf.co",
    "xethub.hf.co",
    "registry.npmmirror.com",
    "mirrors.tuna.tsinghua.edu.cn",
    "pypi.tuna.tsinghua.edu.cn",
    "deb.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "mirrors.aliyun.com",
    "mirrors.ustc.edu.cn",
    "releases.astral.sh",
)


def _no_proxy_export_command() -> str:
    hosts = ",".join(DEFAULT_NO_PROXY_HOSTS)
    quoted_hosts = shlex.quote(hosts)
    return (
        f"HCA_NO_PROXY_HOSTS=${{HCA_NO_PROXY_HOSTS:-{quoted_hosts}}}; "
        "export NO_PROXY=\"${NO_PROXY:+$NO_PROXY,}${HCA_NO_PROXY_HOSTS}\"; "
        "export no_proxy=\"${no_proxy:+$no_proxy,}${HCA_NO_PROXY_HOSTS}\"; "
    )


def _configure_debian_apt_mirror_command() -> str:
    """Return a shell snippet that makes apt installs less proxy-sensitive."""
    debian_mirror = shlex.quote(DEFAULT_DEBIAN_APT_MIRROR)
    ubuntu_mirror = shlex.quote(DEFAULT_UBUNTU_APT_MIRROR)
    return (
        f"APT_MIRROR=${{HCA_APT_MIRROR:-{debian_mirror}}}; "
        f"UBUNTU_APT_MIRROR=${{HCA_UBUNTU_APT_MIRROR:-{ubuntu_mirror}}}; "
        "if command -v apt-get >/dev/null 2>&1; then "
        "  if [ -f /etc/apt/sources.list.d/debian.sources ]; then "
        "    sed -i "
        "      -e \"s|https://mirrors.tuna.tsinghua.edu.cn/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://mirrors.tuna.tsinghua.edu.cn/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://mirrors.tuna.tsinghua.edu.cn/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|http://mirrors.tuna.tsinghua.edu.cn/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|https://mirrors.aliyun.com/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://mirrors.aliyun.com/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://mirrors.aliyun.com/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|http://mirrors.aliyun.com/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|https://mirrors.ustc.edu.cn/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://mirrors.ustc.edu.cn/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://mirrors.ustc.edu.cn/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|http://mirrors.ustc.edu.cn/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|https://deb.debian.org/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://deb.debian.org/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://security.debian.org/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://security.debian.org/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://deb.debian.org/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|http://deb.debian.org/debian|${APT_MIRROR}/debian|g\" "
        "      /etc/apt/sources.list.d/debian.sources || true; "
        "  fi; "
        "  if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then "
        "    sed -i "
        "      -e \"s|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|https://mirrors.aliyun.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://mirrors.aliyun.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|https://mirrors.ustc.edu.cn/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://mirrors.ustc.edu.cn/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|https://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|https://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      /etc/apt/sources.list.d/ubuntu.sources || true; "
        "  fi; "
        "  if [ -f /etc/apt/sources.list ]; then "
        "    sed -i "
        "      -e \"s|https://mirrors.tuna.tsinghua.edu.cn/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://mirrors.tuna.tsinghua.edu.cn/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://mirrors.tuna.tsinghua.edu.cn/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|http://mirrors.tuna.tsinghua.edu.cn/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|https://mirrors.aliyun.com/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://mirrors.aliyun.com/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://mirrors.aliyun.com/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|http://mirrors.aliyun.com/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|https://mirrors.ustc.edu.cn/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://mirrors.ustc.edu.cn/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://mirrors.ustc.edu.cn/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|http://mirrors.ustc.edu.cn/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|https://deb.debian.org/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://deb.debian.org/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://security.debian.org/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|http://security.debian.org/debian-security|${APT_MIRROR}/debian-security|g\" "
        "      -e \"s|https://deb.debian.org/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|http://deb.debian.org/debian|${APT_MIRROR}/debian|g\" "
        "      -e \"s|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|https://mirrors.aliyun.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://mirrors.aliyun.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|https://mirrors.ustc.edu.cn/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://mirrors.ustc.edu.cn/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|https://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|https://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      -e \"s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g\" "
        "      /etc/apt/sources.list || true; "
        "  fi; "
        "fi; "
    )


def _apt_get_install_command(packages: str) -> str:
    mirror_pairs = " ".join(
        f"{debian}|{ubuntu}" for debian, ubuntu in DEFAULT_APT_MIRROR_PAIRS
    )
    quoted_pairs = shlex.quote(mirror_pairs)
    return (
        f"{_no_proxy_export_command()} "
        "export DEBIAN_FRONTEND=noninteractive; "
        f"APT_MIRROR_PAIRS=${{HCA_APT_MIRROR_PAIRS:-{quoted_pairs}}}; "
        "APT_INSTALL_OK=0; "
        "for APT_PAIR in $APT_MIRROR_PAIRS; do "
        "  export HCA_APT_MIRROR=${APT_PAIR%%|*}; "
        "  export HCA_UBUNTU_APT_MIRROR=${APT_PAIR#*|}; "
        "  echo \"Trying apt mirror pair: $HCA_APT_MIRROR / $HCA_UBUNTU_APT_MIRROR\"; "
        f"  {_configure_debian_apt_mirror_command()} "
        "  if apt-get "
        "      -o Acquire::Retries=3 "
        "      -o Acquire::http::Timeout=30 "
        "      -o Acquire::https::Timeout=30 "
        "      update -qq && "
        "     apt-get "
        "      -o Dpkg::Lock::Timeout=300 "
        "      -y -qq --no-install-recommends install "
        f"      {packages}; then "
        "    APT_INSTALL_OK=1; "
        "    break; "
        "  fi; "
        "done; "
        "[ \"$APT_INSTALL_OK\" = \"1\" ]"
    )


def _pip_config_command() -> str:
    index_url = shlex.quote(DEFAULT_PYPI_INDEX_URL)
    return (
        f"{_no_proxy_export_command()} "
        "mkdir -p /etc && "
        "printf '[global]\\nindex-url = %s\\ntimeout = 120\\nretries = 10\\n' "
        f"{index_url} > /etc/pip.conf"
    )


class HarnessAgent(BaseInstalledAgent):
    """
    Installs our harness inside the Harbor container and runs it
    with --profile terminal for each task.
    """

    @staticmethod
    def name() -> str:
        return "harness-agent"

    def __init__(self, model_name: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name

    async def install(self, environment: BaseEnvironment) -> None:
        """Install dependencies and clone our repo into the container.

        Strategy: never use apt-get for python (too slow/unreliable on Daytona).
        1. Ensure git exists (apt-get only for git, which is tiny and fast)
        2. Upload a lightweight repo snapshot (wheel deps, no runtime logs)
        3. If no python3 → install standalone python from an uploaded tarball
           or download it from GitHub (~34MB)
        4. Install only the lightweight runtime deps needed by the headless
           terminal runner. Avoid the full project install because eval/UI deps
           make per-task container setup slow.
        """
        # Step 1: Get the current local harness code into the container so evals
        # test this checkout, including uncommitted adapter/runtime changes.
        await self.exec_as_root(environment, command="rm -rf /home/user/harness-agent && mkdir -p /home/user/harness-agent")
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "harness-agent"
            _copy_repo_snapshot(PROJECT_ROOT, snapshot)
            await environment.upload_dir(snapshot, "/home/user/harness-agent")
        python_tarball = PROJECT_ROOT / "vendor_wheels" / PYTHON_STANDALONE_TARBALL
        if python_tarball.exists():
            await environment.upload_file(python_tarball, CONTAINER_PYTHON_TARBALL)

        # Step 2: Ensure baseline system tooling exists. Several prebuilt
        # Terminal-Bench images are intentionally tiny; installing these during
        # setup prevents later agent/verifier phases from racing apt locks or
        # spending task budget on infrastructure packages.
        await self.exec_as_root(
            environment,
            command=(
                "command -v git >/dev/null 2>&1 && "
                "command -v curl >/dev/null 2>&1 && "
                "command -v unzip >/dev/null 2>&1 && "
                "command -v g++ >/dev/null 2>&1 || "
                "( command -v apt-get >/dev/null 2>&1 && "
                f"  {_apt_get_install_command('ca-certificates git curl unzip build-essential')} ) || "
                "( command -v apk >/dev/null 2>&1 && apk add --no-cache git ) || "
                "( command -v yum >/dev/null 2>&1 && yum install -y -q git ) || "
                "{ echo 'FATAL: baseline tooling is missing and no supported package manager was found'; exit 1; }"
            ),
            timeout_sec=900,
        )

        await self.exec_as_root(environment, command=_pip_config_command())

        # Step 3: Ensure python3 >= 3.11 (openai + pydantic v2 need it)
        # Old containers ship Python 3.9/3.10 where import openai crashes
        # on pydantic v2 or anyio incompatibilities. Check the actual version
        # and install standalone 3.12 if it's too old or missing entirely.
        await self.exec_as_root(
            environment,
            command=(
                "NEED_INSTALL=0; "
                "if command -v python3 >/dev/null 2>&1; then "
                "  PY_VER=$(python3 -c 'import sys; print(sys.version_info[:2])' 2>/dev/null) && "
                "  PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])' 2>/dev/null) && "
                "  PY_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])' 2>/dev/null) && "
                "  echo \"python3 found: $(python3 --version) (parsed: $PY_MAJOR.$PY_MINOR)\" && "
                "  if [ \"$PY_MAJOR\" -lt 3 ] 2>/dev/null || [ \"$PY_MINOR\" -lt 11 ] 2>/dev/null; then "
                "    echo \"Python $PY_MAJOR.$PY_MINOR is too old (need >= 3.11), upgrading...\"; "
                "    NEED_INSTALL=1; "
                "  fi; "
                "else "
                "  echo 'No python3 found'; "
                "  NEED_INSTALL=1; "
                "fi; "
                "if [ \"$NEED_INSTALL\" = \"1\" ]; then "
                f"  VENDOR_TGZ={shlex.quote(CONTAINER_PYTHON_TARBALL)} && "
                "  if [ -f \"$VENDOR_TGZ\" ]; then "
                "    echo 'Installing standalone Python 3.12 from vendored tarball...' && "
                "    cp \"$VENDOR_TGZ\" /tmp/python.tar.gz; "
                "  else "
                # Fallback: download from npmmirror (China) first, then GitHub.
                # Stripped build (~34MB) vs full build (~111MB): agent runtime needs no debug symbols.
                "    echo 'Installing standalone Python 3.12...' && "
                "    URL_NPM='https://registry.npmmirror.com/-/binary/python-build-standalone/"
                "20260623/cpython-3.12.13+20260623-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz' && "
                "    URL_GH='https://github.com/astral-sh/python-build-standalone/releases/"
                "download/20260623/cpython-3.12.13+20260623-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz' && "
                "    ( curl -fsSL -o /tmp/python.tar.gz \"$URL_NPM\" 2>/dev/null || "
                "      wget -q -O /tmp/python.tar.gz \"$URL_NPM\" 2>/dev/null || "
                "      curl -fsSL -o /tmp/python.tar.gz \"$URL_GH\" 2>/dev/null || "
                "      wget -q -O /tmp/python.tar.gz \"$URL_GH\" 2>/dev/null || "
                f"      ( {_apt_get_install_command('curl')} 2>/dev/null && "
                "        curl -fsSL -o /tmp/python.tar.gz \"$URL_NPM\" ) "
                "    ); "
                "  fi && "
                "  mkdir -p /opt/python && "
                "  tar -xzf /tmp/python.tar.gz -C /opt/python --strip-components=1 && "
                # Symlink to /usr/local/bin so it shadows the old system python3
                "  ln -sf /opt/python/bin/python3 /usr/local/bin/python3 && "
                "  ln -sf /opt/python/bin/pip3 /usr/local/bin/pip3 && "
                # Also update the bare 'python' command if it exists
                "  ln -sf /opt/python/bin/python3 /usr/local/bin/python && "
                "  rm -f /tmp/python.tar.gz \"$VENDOR_TGZ\" && "
                # Force hash table refresh so bash picks up the new binary.
                # IMPORTANT: keep `&&` (not `;`) so a failed download breaks the chain
                # instead of being masked by the echo below. The trailing
                # `command -v python3` guard turns a silent skip into a loud failure.
                "  hash -r 2>/dev/null && "
                "  echo \"standalone python installed: $(/usr/local/bin/python3 --version)\" && "
                "  command -v python3 >/dev/null 2>&1 || { echo 'FATAL: python3 still missing after install'; exit 1; }; "
                "else "
                "  echo 'Python version OK, no upgrade needed'; "
                "fi"
            ),
        )

        # Step 4: Install runtime dependencies. Keep this intentionally small:
        # the agent's terminal profile needs the OpenAI stack plus psutil for
        # shell job management. Full requirements.txt includes eval/UI/browser
        # packages and can exceed Harbor's setup timeout inside every task.
        await self.exec_as_root(
            environment,
            command=(
                "PYTHON=$(command -v python3); "
                "$PYTHON -m ensurepip --upgrade >/dev/null 2>&1 || true; "
                "$PYTHON -m pip install --break-system-packages --no-index --force-reinstall -q "
                "  --find-links=/home/user/harness-agent/vendor_wheels openai && "
                "( $PYTHON -m pip install --break-system-packages -q 'psutil>=5.9.8,<8' || "
                "  $PYTHON -m pip install --break-system-packages -q "
                "    -i https://pypi.org/simple 'psutil>=5.9.8,<8' ) && "
                "$PYTHON -c 'import openai, psutil' || "
                "( SITE=$($PYTHON -c 'import site; print(site.getsitepackages()[0])') && "
                "  mkdir -p \"$SITE\" && "
                "  for whl in /home/user/harness-agent/vendor_wheels/*.whl; do "
                "    $PYTHON -m zipfile -e \"$whl\" \"$SITE\" 2>/dev/null; "
                "  done && "
                "  $PYTHON -m pip install --break-system-packages -q 'psutil>=5.9.8,<8' && "
                "  $PYTHON -c 'import openai, psutil; print(\"minimal harness deps installed\")' ) || "
                "( command -v apt-get >/dev/null 2>&1 && "
                f"  {_apt_get_install_command('python3-psutil')} && "
                "  $PYTHON -c 'import openai, psutil' ) || "
                "( echo 'FATAL: failed to install harness dependencies'; exit 1 )"
            ),
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run our harness with --profile terminal on the given task."""
        escaped = shlex.quote(instruction)
        task_name = _task_name_from_context(context)
        task_name_arg = f" --task-name {shlex.quote(task_name)}" if task_name else ""

        # Pass secrets through Harbor's env channel instead of embedding them
        # in the logged shell command.
        env_vars = runner_env_vars()

        # Run VeriForge from the task workspace while importing the cloned agent code.
        result = await self.exec_as_agent(
            environment,
            command=(
                f"WORKSPACE=/app; "
                f"[ -d \"$WORKSPACE\" ] || WORKSPACE=/workspace; "
                f"[ -d \"$WORKSPACE\" ] || WORKSPACE=/root; "
                f"[ -d \"$WORKSPACE\" ] || WORKSPACE=/; "
                f"cd \"$WORKSPACE\" && "
                f"{_no_proxy_export_command()} "
                "if [ -n \"${HCA_MODEL_API_NO_PROXY_HOSTS:-}\" ]; then "
                "  export NO_PROXY=\"${NO_PROXY:+$NO_PROXY,}${HCA_MODEL_API_NO_PROXY_HOSTS}\"; "
                "  export no_proxy=\"${no_proxy:+$no_proxy,}${HCA_MODEL_API_NO_PROXY_HOSTS}\"; "
                "fi && "
                f"PYTHONPATH=/home/user/harness-agent "
                f"python3 /home/user/harness-agent/eval/benchmarks/hca_terminal_runner.py "
                f"--workspace \"$WORKSPACE\"{task_name_arg} {escaped}"
            ),
            env=env_vars,
        )
        metrics = parse_eval_metrics_from_text(getattr(result, "stdout", "") or "")
        if metrics:
            _populate_agent_context(context, metrics)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Called after run() completes. Could parse logs if needed."""


def _copy_repo_snapshot(source: Path, dest: Path) -> None:
    source = source.resolve()

    def ignore(path: str, names: list[str]) -> set[str]:
        current = Path(path).resolve()
        ignored: set[str] = set()
        for name in names:
            item = current / name
            rel = item.relative_to(source).as_posix()
            if name in {".git", ".pytest_cache", "__pycache__"} or name == ".harbor" and current == source or name == ".harness" and current == source or name == "jobs" and current == source or name == "workspace" and current == source or rel == "eval/results" or _is_vendored_python_tarball(rel) or name == ".env" or name.startswith(".env.") or name.endswith(".pyc"):
                ignored.add(name)
        return ignored

    shutil.copytree(source, dest, ignore=ignore)


def _is_vendored_python_tarball(rel_path: str) -> bool:
    return rel_path.startswith("vendor_wheels/python-") and rel_path.endswith(".tar.gz")


def _task_name_from_context(context: AgentContext) -> str:
    candidates: list[object] = [
        getattr(context, "task_name", None),
        getattr(context, "task_id", None),
        getattr(context, "name", None),
    ]
    task = getattr(context, "task", None)
    if task is not None:
        candidates.extend(
            [
                getattr(task, "name", None),
                getattr(task, "task_name", None),
                getattr(task, "id", None),
            ]
        )
    metadata = getattr(context, "metadata", None)
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("task_name"),
                metadata.get("task_id"),
                metadata.get("name"),
            ]
        )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _populate_agent_context(context: AgentContext, metrics: dict) -> None:
    tokens = metrics.get("tokens") or {}
    usage_cost = metrics.get("usage_cost") or {}
    context.n_input_tokens = tokens.get("prompt_tokens")
    context.n_cache_tokens = tokens.get("cached_tokens")
    context.n_output_tokens = tokens.get("completion_tokens")
    context.cost_usd = usage_cost.get("estimated_cost_usd")
    metadata = dict(context.metadata or {})
    metadata["hca_eval_metrics"] = metrics
    context.metadata = metadata
