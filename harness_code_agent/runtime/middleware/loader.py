"""User middleware loader.

Loads custom ``AgentMiddleware`` subclasses declared in the user's
persistent JSON config::

    ~/.harness/middlewares.json

Only this user-home config is read. Workspace-local middleware configs are
intentionally unsupported: otherwise cloning a repository and starting the
agent could import Python code shipped by that repository.

A missing config means "no custom middleware". Any other problem — invalid
JSON, a missing file or class, a class that does not extend
AgentMiddleware — fails startup with a clear error rather than silently
running without a policy the user explicitly configured.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from .base import AgentMiddleware

CONFIG_PATH = Path.home() / ".harness" / "middlewares.json"


class MiddlewareConfigError(Exception):
    """The user middleware config cannot be loaded safely."""


def load_user_middlewares(config_path: Path | str | None = None) -> list[AgentMiddleware]:
    """Instantiate every enabled middleware declared in the user config.

    Array order is the middleware order. Returns an empty list only when no
    config file exists.
    """
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    if not path.exists():
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MiddlewareConfigError(f"Failed to read middleware config {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MiddlewareConfigError(f"Failed to parse middleware config {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise MiddlewareConfigError(
            f"Failed to parse middleware config {path}: top-level value must be an object"
        )
    entries = data.get("middlewares")
    if not isinstance(entries, list):
        raise MiddlewareConfigError(
            f"Failed to parse middleware config {path}: 'middlewares' must be a list"
        )

    middlewares: list[AgentMiddleware] = []
    for position, item in enumerate(entries):
        if not isinstance(item, dict):
            raise MiddlewareConfigError(
                f"Failed to parse middleware config {path}: entry {position} must be an object"
            )
        if not item.get("enabled", True):
            continue
        raw_path = item.get("path")
        class_name = item.get("class")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise MiddlewareConfigError(
                f"Failed to parse middleware config {path}: entry {position} is missing 'path'"
            )
        if not isinstance(class_name, str) or not class_name.strip():
            raise MiddlewareConfigError(
                f"Failed to parse middleware config {path}: entry {position} is missing 'class'"
            )
        middlewares.append(_load_middleware(raw_path, class_name))
    return middlewares


def _load_middleware(raw_path: str, class_name: str) -> AgentMiddleware:
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise MiddlewareConfigError(
            f"Failed to load middleware {class_name}:\nfile not found: {Path(raw_path).expanduser()}"
        ) from exc

    module_name = "_harness_user_middleware_" + "".join(
        ch if ch.isalnum() else "_" for ch in str(path)
    )
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise MiddlewareConfigError(
                f"Failed to load middleware {class_name}:\ncannot import {path} as a Python module"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except MiddlewareConfigError:
        sys.modules.pop(module_name, None)
        raise
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise MiddlewareConfigError(
            f"Failed to load middleware {class_name}:\n"
            f"error while importing {path}: {type(exc).__name__}: {exc}"
        ) from exc

    cls = getattr(module, class_name, None)
    if cls is None:
        raise MiddlewareConfigError(
            f"Failed to load middleware {class_name}:\nclass not found: {class_name} in {path}"
        )
    if not isinstance(cls, type) or not issubclass(cls, AgentMiddleware):
        raise MiddlewareConfigError(
            f"Failed to load middleware {class_name}:\n"
            f"{class_name} in {path} is not an AgentMiddleware subclass"
        )
    try:
        return cls()
    except Exception as exc:
        raise MiddlewareConfigError(
            f"Failed to load middleware {class_name}:\n"
            f"could not instantiate {class_name} (it must take no arguments): {exc}"
        ) from exc
