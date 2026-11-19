from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from eval.benchmarks.usage_metrics import collect_harbor_job_usage

DATASET_ARCHIVE_URL = (
    "https://github.com/harbor-framework/terminal-bench-2-1/archive/refs/heads/main.zip"
)
HARBOR_DATASET_ID = "terminal-bench@2.1"
LOCAL_DATASET_DIR_NAME = "terminal-bench-2-1"
TERMINAL_BENCH_LABEL = "Terminal-Bench 2.1"
# Terminal-Bench 4.1 runs straight from the Harbor registry: its tasks build
# from their own Dockerfiles, so there are no prebuilt images to repair, and
# registry task names carry the organization prefix.
TB41_HARBOR_DATASET_ID = "terminal-bench/terminal-bench@4.1.0"
TB41_TASK_PREFIX = "terminal-bench/"
TB41_AGENT_TIMEOUT_SEC = 28800.0
DEFAULT_TB41_TASK_BUDGET_SEC = 3600.0
# Harbor's own agent timeout stays a little above the agent's budget so the
# agent stops itself first and the session artifacts are exported.
TB41_AGENT_TIMEOUT_MARGIN_SEC = 900.0
BENCHMARKS = ("tb2.1", "tb4.1")
DOCKERHUB_IMAGE_NAMESPACE = "alexgshaw"
DOCKERHUB_IMAGE_TAG = "20251031"
DEFAULT_AGENT_SETUP_TIMEOUT_SEC = 1200.0
DEFAULT_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER = 3.0
DEFAULT_DOCKER_PULL_TIMEOUT_SEC = 1800
DEFAULT_VERIFIER_NO_PROXY_HOSTS = (
    "localhost",
    "127.0.0.1",
    "host.docker.internal",
    "http.docker.internal",
    "astral.sh",
    "releases.astral.sh",
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
    "mirrors.tuna.tsinghua.edu.cn",
    "pypi.tuna.tsinghua.edu.cn",
    "deb.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "mirrors.aliyun.com",
    "mirrors.ustc.edu.cn",
)


def resolve_harbor_executable(env: dict[str, str]) -> str:
    harbor_path = shutil.which("harbor") or shutil.which("harbor.exe")
    if harbor_path:
        return harbor_path

    candidates = [
        Path(env.get("USERPROFILE", "")) / ".local" / "bin" / "harbor.exe",
        Path(env.get("USERPROFILE", "")) / ".local" / "bin" / "harbor",
        Path(env.get("APPDATA", "")) / "Python" / "Python312" / "Scripts" / "harbor.exe",
        Path(env.get("USERPROFILE", "")) / "AppData" / "Roaming" / "Python" / "Python312" / "Scripts" / "harbor.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return "harbor"


def build_launch_environment(
    repo_root: Path,
    base_env: dict[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ.copy())
    temp_dir = (repo_root / ".harbor" / "tmp").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env.setdefault("MAX_AGENT_TOOL_CALLS", "400")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("RICH_FORCE_TERMINAL", "0")

    dotenv_file = dotenv_path or (repo_root / ".env")
    if dotenv_file.exists():
        for raw_line in dotenv_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in env:
                env[key] = value

    pythonpath = env.get("PYTHONPATH")
    paths = [str(repo_root.resolve())]
    if pythonpath:
        paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def default_local_dataset_path(repo_root: Path) -> Path:
    return (repo_root / ".harbor" / "datasets" / LOCAL_DATASET_DIR_NAME).resolve()


def resolve_harbor_dataset_path(dataset_path: Path) -> Path:
    tasks_root = dataset_path / "tasks"
    if is_valid_harbor_dataset(dataset_path):
        return tasks_root
    raise RuntimeError(
        f"Terminal-Bench 2.1 dataset at {dataset_path} must contain tasks/*/task.toml."
    )


def is_valid_harbor_dataset(dataset_path: Path) -> bool:
    return any(_task_files(dataset_path, None))


def _download_and_extract_dataset_archive(dataset_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dataset_path.parent)) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        archive_path = temp_dir / f"{LOCAL_DATASET_DIR_NAME}.zip"
        urllib.request.urlretrieve(DATASET_ARCHIVE_URL, archive_path)
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)

        extracted_roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(extracted_roots) != 1:
            raise RuntimeError(
                f"Expected exactly one extracted root in {extract_dir}, found {len(extracted_roots)}"
            )

        source_root = extracted_roots[0]
        dataset_path.mkdir(parents=True, exist_ok=True)
        for child in source_root.iterdir():
            shutil.move(str(child), str(dataset_path / child.name))


def ensure_local_dataset(dataset_path: Path) -> Path:
    resolved = dataset_path.resolve()
    if is_valid_harbor_dataset(resolved):
        return resolved

    if resolved.exists():
        shutil.rmtree(resolved)

    _download_and_extract_dataset_archive(resolved)

    if not is_valid_harbor_dataset(resolved):
        raise RuntimeError(
            f"Local dataset checkout at {resolved} does not contain Harbor tasks."
        )

    return resolved


def repair_task_images(dataset_path: Path, tasks: list[str] | None = None) -> int:
    repaired = 0
    task_files = _task_files(dataset_path, tasks)
    for task_file in task_files:
        task_name = task_file.parent.name
        content = task_file.read_text(encoding="utf-8")
        image = _task_docker_image(content)
        if not image or _docker_image_exists(image):
            continue

        fallback = f"{DOCKERHUB_IMAGE_NAMESPACE}/{task_name}:{DOCKERHUB_IMAGE_TAG}"
        if not _docker_image_exists(fallback):
            continue

        updated = _replace_task_docker_image(content, fallback)
        if updated != content:
            task_file.write_text(updated, encoding="utf-8")
            repaired += 1
    return repaired


def patch_verifier_proxy_env(
    dataset_path: Path,
    tasks: list[str] | None = None,
    *,
    no_proxy_hosts: tuple[str, ...] = DEFAULT_VERIFIER_NO_PROXY_HOSTS,
) -> int:
    """Inject verifier no_proxy settings into the local Harbor dataset copy.

    Harbor reads [verifier.env] from task.toml and forwards it to verifier
    execution. The local Docker proxy can abort CONNECT requests to uv/PyPI
    bootstrap hosts, so verifier scripts should bypass that proxy for these
    hosts while leaving Docker's proxy available for other traffic.
    """
    patched = 0
    no_proxy_value = ",".join(no_proxy_hosts)
    for task_file in _task_files(dataset_path, tasks):
        content = task_file.read_text(encoding="utf-8")
        updated = _upsert_verifier_env_no_proxy(content, no_proxy_value)
        if updated != content:
            task_file.write_text(updated, encoding="utf-8", newline="\n")
            patched += 1
    return patched


def pre_pull_task_images(
    dataset_path: Path,
    tasks: list[str] | None = None,
    *,
    timeout_sec: int = DEFAULT_DOCKER_PULL_TIMEOUT_SEC,
) -> list[str]:
    """Pull missing prebuilt task images before Harbor starts environments.

    Harbor's environment-start timeout includes Docker image pulls. Pulling
    large or absent images up front keeps that timeout focused on container
    startup and makes failures easier to diagnose.
    """
    pulled: list[str] = []
    for image in _task_docker_images(dataset_path, tasks):
        if _docker_image_present(image):
            continue
        print(f"Pulling missing task image: {image}", flush=True)
        try:
            completed = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out after {timeout_sec} seconds while pulling Docker image {image}."
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"Failed to pull Docker image {image}: {detail}")
        pulled.append(image)
    return pulled


def _task_files(dataset_path: Path, tasks: list[str] | None) -> list[Path]:
    tasks_root = dataset_path / "tasks"
    if tasks:
        return [
            tasks_root / task / "task.toml"
            for task in tasks
            if (tasks_root / task / "task.toml").exists()
        ]
    return list(tasks_root.glob("*/task.toml"))


def _task_docker_image(content: str) -> str:
    match = re.search(r'^docker_image\s*=\s*"([^"]+)"\s*$', content, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _task_docker_images(dataset_path: Path, tasks: list[str] | None) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()
    for task_file in _task_files(dataset_path, tasks):
        image = _task_docker_image(task_file.read_text(encoding="utf-8"))
        if image and image not in seen:
            seen.add(image)
            images.append(image)
    return images


def _upsert_verifier_env_no_proxy(content: str, no_proxy_value: str) -> str:
    lines = content.splitlines()
    header_index = next((idx for idx, line in enumerate(lines) if line.strip() == "[verifier.env]"), None)
    if header_index is None:
        prefix = content.rstrip("\n")
        return (
            prefix
            + "\n\n[verifier.env]\n"
            + f'NO_PROXY = "{_toml_escape(no_proxy_value)}"\n'
            + f'no_proxy = "{_toml_escape(no_proxy_value)}"\n'
        )

    end_index = len(lines)
    for idx in range(header_index + 1, len(lines)):
        if re.match(r"^\s*\[[^\]]+\]\s*$", lines[idx]):
            end_index = idx
            break

    changed = False
    insert_at = header_index + 1
    for key in ("NO_PROXY", "no_proxy"):
        found = False
        for idx in range(header_index + 1, end_index):
            match = re.match(rf'^(\s*{re.escape(key)}\s*=\s*)"([^"]*)"\s*$', lines[idx])
            if not match:
                continue
            merged = _merge_csv_values(match.group(2), no_proxy_value)
            replacement = f'{match.group(1)}"{_toml_escape(merged)}"'
            if replacement != lines[idx]:
                lines[idx] = replacement
                changed = True
            found = True
            break
        if not found:
            lines.insert(insert_at, f'{key} = "{_toml_escape(no_proxy_value)}"')
            insert_at += 1
            end_index += 1
            changed = True

    if not changed:
        return content
    return "\n".join(lines) + "\n"


def _merge_csv_values(existing: str, additions: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for raw in [*existing.split(","), *additions.split(",")]:
        item = raw.strip()
        if not item or item.lower() in seen:
            continue
        seen.add(item.lower())
        values.append(item)
    return ",".join(values)


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _replace_task_docker_image(content: str, image: str) -> str:
    return re.sub(
        r'^docker_image\s*=\s*".*"$',
        f'docker_image = "{image}"',
        content,
        flags=re.MULTILINE,
    )


def _docker_image_exists(image: str) -> bool:
    try:
        completed = subprocess.run(
            ["docker", "manifest", "inspect", image],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _docker_image_present(image: str) -> bool:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def build_harbor_run_command(
    harbor_executable: str,
    tasks: list[str],
    runner_env: str | None,
    dataset_path: Path | None,
    force_build: bool,
    jobs_dir: Path | None = None,
    agent_setup_timeout_sec: float | None = None,
    environment_build_timeout_multiplier: float | None = None,
    dataset_id: str = HARBOR_DATASET_ID,
    agent_timeout_multiplier: float | None = None,
    verifier_env: dict[str, str] | None = None,
) -> list[str]:
    command = [harbor_executable, "run"]
    if jobs_dir is not None:
        command.extend(["--jobs-dir", str(jobs_dir)])
    if dataset_path is None:
        command.extend(["-d", dataset_id])
    else:
        command.extend(["--path", str(dataset_path)])

    command.extend(["--agent-import-path", "eval.benchmarks.harbor_agent:HarnessAgent"])

    if runner_env:
        command.extend(["--env", runner_env])
    if force_build:
        command.append("--force-build")
    if agent_setup_timeout_sec is not None:
        multiplier = agent_setup_timeout_sec / 600.0
        command.extend(["--agent-setup-timeout-multiplier", str(multiplier)])
    if environment_build_timeout_multiplier is not None:
        command.extend(
            [
                "--environment-build-timeout-multiplier",
                str(environment_build_timeout_multiplier),
            ]
        )
    if agent_timeout_multiplier is not None:
        command.extend(["--agent-timeout-multiplier", f"{agent_timeout_multiplier:.6g}"])
    for key, value in (verifier_env or {}).items():
        command.extend(["--ve", f"{key}={value}"])
    for task in tasks:
        command.extend(["--include-task-name", task])

    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Run {TERMINAL_BENCH_LABEL} from a local dataset with repaired prebuilt image "
            "references, or Terminal-Bench 4.1 from the Harbor registry."
        )
    )
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARKS,
        default="tb2.1",
        help="Terminal-Bench version to run (default: tb2.1).",
    )
    parser.add_argument(
        "--task-budget",
        type=float,
        default=DEFAULT_TB41_TASK_BUDGET_SEC,
        help=(
            "tb4.1 only: the agent's wall-clock budget per task in seconds "
            f"(default {DEFAULT_TB41_TASK_BUDGET_SEC:.0f}). Harbor's agent timeout is set "
            f"{TB41_AGENT_TIMEOUT_MARGIN_SEC:.0f}s above it."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name to run. Repeat to include multiple tasks.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full local dataset. If omitted and no --task is given, the full dataset is still run.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=f"Local dataset directory. Defaults to repo-local .harbor/datasets/{LOCAL_DATASET_DIR_NAME}",
    )
    parser.add_argument(
        "--env",
        dest="runner_env",
        default=None,
        help="Harbor environment backend, for example daytona.",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Force Harbor to build environments locally instead of using task docker_image.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=None,
        help="Optional Harbor jobs directory. Useful for isolating parallel task runs.",
    )
    parser.add_argument(
        "--agent-setup-timeout",
        type=float,
        default=DEFAULT_AGENT_SETUP_TIMEOUT_SEC,
        help="Harbor agent setup timeout in seconds. Defaults higher than Harbor's built-in 360s to absorb slow apt/pip setup.",
    )
    parser.add_argument(
        "--environment-build-timeout-multiplier",
        type=float,
        default=DEFAULT_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER,
        help="Multiplier for Harbor environment build/start timeout, useful when Docker must pull large images.",
    )
    parser.add_argument(
        "--skip-image-prepull",
        action="store_true",
        help="Do not pre-pull missing prebuilt Docker images before invoking Harbor.",
    )
    parser.add_argument(
        "--no-patch-verifier-proxy",
        action="store_true",
        help="Do not inject NO_PROXY/no_proxy into task verifier.env sections in the local dataset copy.",
    )
    return parser.parse_args(argv)


def tb41_registry_task_names(tasks: list[str]) -> list[str]:
    """Registry task names for --include-task-name (Harbor matches org/name)."""
    return [task if task.startswith(TB41_TASK_PREFIX) else TB41_TASK_PREFIX + task for task in tasks]


def tb41_agent_timeout_multiplier(task_budget_sec: float) -> float:
    return (task_budget_sec + TB41_AGENT_TIMEOUT_MARGIN_SEC) / TB41_AGENT_TIMEOUT_SEC


def build_tb41_run_command(
    args: argparse.Namespace,
    harbor_executable: str,
) -> list[str]:
    if args.full or not args.task:
        raise ValueError("tb4.1 runs need explicit --task names; the full dataset is not supported here.")
    if args.task_budget <= 0:
        raise ValueError("--task-budget must be positive.")
    verifier_env: dict[str, str] = {}
    if not args.no_patch_verifier_proxy:
        no_proxy_value = ",".join(DEFAULT_VERIFIER_NO_PROXY_HOSTS)
        verifier_env = {"NO_PROXY": no_proxy_value, "no_proxy": no_proxy_value}
    return build_harbor_run_command(
        harbor_executable=harbor_executable,
        tasks=tb41_registry_task_names(args.task),
        runner_env=args.runner_env,
        dataset_path=None,
        force_build=args.force_build,
        jobs_dir=args.jobs_dir,
        agent_setup_timeout_sec=args.agent_setup_timeout,
        environment_build_timeout_multiplier=args.environment_build_timeout_multiplier,
        dataset_id=TB41_HARBOR_DATASET_ID,
        agent_timeout_multiplier=tb41_agent_timeout_multiplier(args.task_budget),
        verifier_env=verifier_env,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    env = build_launch_environment(repo_root)
    harbor_executable = resolve_harbor_executable(env)

    if args.benchmark == "tb4.1":
        return _run_tb41(args, repo_root, env, harbor_executable)

    tasks = [] if args.full else args.task
    selected_tasks = None if args.full or not args.task else tasks
    dataset_path = ensure_local_dataset(args.dataset_path or default_local_dataset_path(repo_root))
    if args.runner_env != "daytona" and not docker_daemon_running():
        print("Docker daemon is not running. Please start Docker and try again.", file=sys.stderr)
        return 125
    verifier_proxy_patched = 0
    if not args.no_patch_verifier_proxy:
        verifier_proxy_patched = patch_verifier_proxy_env(dataset_path, selected_tasks)
    repaired_count = repair_task_images(dataset_path, selected_tasks)
    pulled_images: list[str] = []
    if args.runner_env != "daytona" and not args.force_build and not args.skip_image_prepull:
        pulled_images = pre_pull_task_images(dataset_path, selected_tasks)
    command = build_harbor_run_command(
        harbor_executable=harbor_executable,
        tasks=tasks,
        runner_env=args.runner_env,
        dataset_path=resolve_harbor_dataset_path(dataset_path),
        force_build=args.force_build,
        jobs_dir=args.jobs_dir,
        agent_setup_timeout_sec=args.agent_setup_timeout,
        environment_build_timeout_multiplier=args.environment_build_timeout_multiplier,
    )

    print(f"Using TEMP/TMP: {env['TEMP']}")
    print(f"Prepared local dataset: {dataset_path}")
    print(f"Patched verifier proxy env for {verifier_proxy_patched} task(s)")
    print(f"Repaired {repaired_count} broken task docker_image entries")
    print(f"Pre-pulled {len(pulled_images)} missing task image(s)")
    print(f"Running: {' '.join(command)}")
    return _run_harbor(command, repo_root, env, args.jobs_dir)


def _run_tb41(
    args: argparse.Namespace,
    repo_root: Path,
    env: dict[str, str],
    harbor_executable: str,
) -> int:
    try:
        command = build_tb41_run_command(args, harbor_executable)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.runner_env != "daytona" and not docker_daemon_running():
        print("Docker daemon is not running. Please start Docker and try again.", file=sys.stderr)
        return 125
    # The in-container runner reads the budget from this variable; Harbor
    # forwards it through the agent's environment (see harbor_env.py).
    env["HARNESS_TASK_BUDGET"] = f"{args.task_budget:g}"
    print(f"Using TEMP/TMP: {env['TEMP']}")
    print(f"Dataset: {TB41_HARBOR_DATASET_ID} (Harbor registry)")
    print(f"Agent budget per task: {args.task_budget:g}s")
    print(f"Running: {' '.join(command)}")
    return _run_harbor(command, repo_root, env, args.jobs_dir)


def _run_harbor(
    command: list[str],
    repo_root: Path,
    env: dict[str, str],
    jobs_dir: Path | None,
) -> int:
    jobs_root = (jobs_dir or (repo_root / "jobs")).resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)
    before_jobs = _job_names(jobs_root)
    completed = subprocess.run(command, cwd=repo_root, env=env, check=False)
    for job_dir in _new_job_dirs(jobs_root, before_jobs):
        summary = collect_harbor_job_usage(job_dir)
        print("HCA_TERMINAL_BENCH_RESULT:" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return completed.returncode


def docker_daemon_running() -> bool:
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{json .ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _job_names(jobs_root: Path) -> set[str]:
    if not jobs_root.exists():
        return set()
    return {path.name for path in jobs_root.iterdir() if path.is_dir()}


def _new_job_dirs(jobs_root: Path, before: set[str]) -> list[Path]:
    if not jobs_root.exists():
        return []
    new_dirs = [path for path in jobs_root.iterdir() if path.is_dir() and path.name not in before]
    return sorted(new_dirs, key=lambda item: item.stat().st_mtime)


if __name__ == "__main__":
    raise SystemExit(main())
