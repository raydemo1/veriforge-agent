"""Headless browser testing tools."""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

from ...agent.cancellation import CancelledError
from ..tool_result import ToolResult

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def _playwright_browser_roots() -> list[Path]:
    """Return browser cache roots without assuming a particular browser version."""
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if configured and configured != "0":
        return [Path(configured).expanduser()]
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        return [Path(local_app_data) / "ms-playwright"] if local_app_data else []
    if sys.platform == "darwin":
        return [Path.home() / "Library" / "Caches" / "ms-playwright"]
    return [Path.home() / ".cache" / "ms-playwright"]


def _installed_chromium_executables() -> list[Path]:
    """Find installed Chromium binaries, ordered by detected build number."""
    patterns = (
        "chromium_headless_shell-*/*/chrome-headless-shell.exe",
        "chromium_headless_shell-*/*/chrome-headless-shell",
        "chromium-*/*/chrome.exe",
        "chromium-*/*/chrome",
        "chromium-*/*/Chromium.app/Contents/MacOS/Chromium",
    )
    candidates: dict[str, Path] = {}
    for root in _playwright_browser_roots():
        if not root.is_dir():
            continue
        for pattern in patterns:
            for executable in root.glob(pattern):
                if executable.is_file():
                    candidates[str(executable.resolve()).lower()] = executable

    def sort_key(executable: Path) -> tuple[tuple[int, ...], int, str]:
        numbers = tuple(int(value) for value in re.findall(r"\d+", str(executable.parent.parent)))
        headless = 1 if "headless" in executable.name.lower() else 0
        return numbers, headless, str(executable).lower()

    return sorted(candidates.values(), key=sort_key, reverse=True)


def _launch_chromium(playwright):
    """Launch Playwright's browser, adapting to an already-installed local build."""
    try:
        return playwright.chromium.launch(headless=True), None
    except Exception:
        for executable in _installed_chromium_executables():
            try:
                browser = playwright.chromium.launch(headless=True, executable_path=str(executable))
                return browser, f"Browser: detected local Chromium ({executable})"
            except Exception:  # noqa: S112 - try the next detected browser binary
                continue
        raise


def _ensure_dev_server(runtime_state, start_command: str, port: int, startup_wait: int, cancellation_token) -> str:
    """Start a conversation-owned dev server and wait cancellably for startup."""
    manager = getattr(runtime_state, "shell_job_manager", None)
    if manager is None:
        return "[error] No background job manager available"
    existing_id = getattr(runtime_state, "browser_job_id", None)
    if existing_id:
        try:
            existing = manager.get(existing_id)
        except Exception:
            existing = None
        if existing is not None and existing.status == "running":
            return f"Dev server already running (pid={existing.pid})"

    job = manager.start(start_command)
    runtime_state.browser_job_id = job.job_id
    deadline = time.monotonic() + max(0, startup_wait)
    while time.monotonic() < deadline and job.status == "running":
        _check_cancelled(cancellation_token)
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    _check_cancelled(cancellation_token)
    if job.status != "running":
        detail = manager.read_output(job.job_id, 2_000).strip()
        return f"[error] Dev server exited immediately: {detail or job.error or job.status}"
    return f"Dev server started (pid={job.pid}, port={port})"


def stop_dev_server(runtime_state=None) -> ToolResult:
    """Stop the background dev server."""
    manager = getattr(runtime_state, "shell_job_manager", None)
    job_id = getattr(runtime_state, "browser_job_id", None)
    if manager is None or not job_id:
        return ToolResult(
            tool="stop_dev_server",
            status="success",
            output="No dev server running",
            metadata={"status_source": "native"},
        )
    try:
        job = manager.stop(job_id)
    except Exception as exc:
        return ToolResult(
            tool="stop_dev_server",
            status="failed",
            output=f"[error] Failed to stop dev server: {exc}",
            error=str(exc),
            metadata={"status_source": "shell_job", "job_id": job_id},
        )
    runtime_state.browser_job_id = None
    return ToolResult(
        tool="stop_dev_server",
        status="success",
        output=f"Dev server stopped (job_id={job_id}, pid={job.pid})",
        metadata={"status_source": "shell_job", "job_id": job_id},
    )


def browser_test(
    url: str,
    actions: list[dict] | None = None,
    screenshot: bool = True,
    start_command: str | None = None,
    port: int = 5173,
    startup_wait: int = 8,
    runtime_state=None,
    tool_context=None,
    cancellation_token=None,
) -> ToolResult:
    """
    Launch a headless browser, navigate to a URL, perform actions, and
    optionally take a screenshot. Returns a text report of what happened.

    actions is a list of dicts, each with:
      - type: "click" | "fill" | "wait" | "evaluate" | "scroll"
      - selector: CSS selector (for click/fill)
      - value: text to type (for fill), JS code (for evaluate)
      - delay: ms to wait (for wait)

    If start_command is provided, starts a dev server first.
    """
    if not HAS_PLAYWRIGHT:
        output = (
            "[error] Playwright not installed. "
            "Install with: pip install playwright && python -m playwright install chromium"
        )
        return ToolResult(
            tool="browser_test",
            status="failed",
            output=output,
            error="Playwright not installed",
            metadata={"status_source": "runtime"},
        )

    report_lines = []
    failed = False
    error_message = None
    console_errors: list[str] = []
    final_url = url
    artifact_path: Path | None = None

    # Optionally start dev server
    if start_command:
        try:
            srv_result = _ensure_dev_server(
                runtime_state,
                start_command,
                port,
                startup_wait,
                cancellation_token,
            )
        except CancelledError:
            stop_dev_server(runtime_state)
            raise
        report_lines.append(f"Server: {srv_result}")
        if srv_result.startswith("[error]"):
            failed = True
            error_message = srv_result.removeprefix("[error] ")

    try:
        _check_cancelled(cancellation_token)
        with sync_playwright() as p:
            browser, browser_detail = _launch_chromium(p)
            if browser_detail:
                report_lines.append(browser_detail)
            page = browser.new_page(viewport={"width": 1280, "height": 720})

            # Navigate
            try:
                _check_cancelled(cancellation_token)
                page.goto(url, timeout=15000)
                _check_cancelled(cancellation_token)
                report_lines.append(f"Navigated to {url} — title: {page.title()}")
            except Exception as e:
                report_lines.append(f"[error] Navigation failed: {e}")
                browser.close()
                return ToolResult(
                    tool="browser_test",
                    status="failed",
                    output="\n".join(report_lines),
                    error=f"Navigation failed: {e}",
                    metadata={
                        "url": url,
                        "status_source": "browser",
                        "checks": [{
                            "name": "Browser verification",
                            "status": "failed",
                            "detail": f"navigation failed: {url}",
                        }],
                    },
                )

            # Check for console errors
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

            # Execute actions
            for action in (actions or []):
                _check_cancelled(cancellation_token)
                action_type = action.get("type", "")
                selector = action.get("selector", "")
                value = action.get("value", "")
                delay = action.get("delay", 1000)

                try:
                    if action_type == "click":
                        page.click(selector, timeout=5000)
                        report_lines.append(f"Clicked: {selector}")
                    elif action_type == "fill":
                        page.fill(selector, value, timeout=5000)
                        report_lines.append(f"Filled '{selector}' with '{value[:50]}'")
                    elif action_type == "wait":
                        remaining = max(0, int(delay))
                        while remaining:
                            _check_cancelled(cancellation_token)
                            chunk = min(100, remaining)
                            page.wait_for_timeout(chunk)
                            remaining -= chunk
                        report_lines.append(f"Waited {delay}ms")
                    elif action_type == "evaluate":
                        result = page.evaluate(value)
                        report_lines.append(f"JS eval result: {str(result)[:500]}")
                    elif action_type == "scroll":
                        page.evaluate(f"window.scrollBy(0, {value or 500})")
                        report_lines.append(f"Scrolled by {value or 500}px")
                    else:
                        report_lines.append(f"[warn] Unknown action type: {action_type}")
                except Exception as e:
                    report_lines.append(f"[error] Action {action_type}('{selector}'): {e}")
                    failed = True
                    error_message = f"Action {action_type}('{selector}'): {e}"

                for _ in range(3):
                    _check_cancelled(cancellation_token)
                    page.wait_for_timeout(100)

            # Gather page info
            final_url = page.url
            report_lines.append(f"Final URL: {page.url}")
            report_lines.append(f"Visible text (first 2000 chars): {page.inner_text('body')[:2000]}")

            if console_errors:
                report_lines.append(f"Console errors ({len(console_errors)}):")
                for err in console_errors[:10]:
                    report_lines.append(f"  - {err[:200]}")

            # Screenshot
            if screenshot:
                if tool_context is None:
                    raise RuntimeError("browser_test requires a workspace context for screenshots")
                ss_path = Path(tool_context.workspace.root) / ".harness" / "artifacts" / "browser.png"
                ss_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(ss_path), full_page=False)
                artifact_path = ss_path
                report_lines.append(f"Screenshot saved to {ss_path.relative_to(tool_context.workspace.root)}")

            browser.close()

    except Exception as e:
        report_lines.append(f"[error] Browser test failed: {e}")
        failed = True
        error_message = f"Browser test failed: {e}"

    check_status = "failed" if failed else ("warning" if console_errors else "passed")
    check_detail = (
        f"{len(console_errors)} console error(s)"
        if console_errors and not failed
        else final_url
    )
    metadata: dict = {
        "url": url,
        "status_source": "browser",
        "checks": [{
            "name": "Browser verification",
            "status": check_status,
            "detail": check_detail,
        }],
    }
    if artifact_path is not None:
        rel = artifact_path.relative_to(tool_context.workspace.root).as_posix()
        metadata["artifacts"] = [{
            "kind": "image",
            "path": rel,
            "title": artifact_path.name,
            "detail": final_url,
        }]

    return ToolResult(
        tool="browser_test",
        status="failed" if failed else "success",
        output="\n".join(report_lines),
        error=error_message,
        metadata=metadata,
    )


def _check_cancelled(cancellation_token) -> None:
    if cancellation_token is not None:
        cancellation_token.check()
