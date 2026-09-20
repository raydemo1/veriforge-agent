from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandResult:
    text: str = ""
    should_continue: bool = True
    action: str | None = None
    payload: object | None = None


CommandHandler = Callable[[Any, list[str], "SlashCommandRegistry"], CommandResult]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    group: str
    usage: str
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()
    agent_command: bool = False

    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


class SlashCommandRegistry:
    def __init__(self, specs: list[CommandSpec]):
        self.specs = specs
        self._by_name = {
            name: spec
            for spec in specs
            for name in spec.names()
        }

    def execute(self, line: str, session: Any) -> CommandResult:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            return CommandResult(f"错误：{exc}")
        if not parts:
            return CommandResult()
        command, args = parts[0], parts[1:]
        spec = self._by_name.get(command)
        if spec is None:
            return CommandResult(f"未知的斜杠命令：{command}")
        try:
            return spec.handler(session, args, self)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return CommandResult(f"错误：{exc}")

    def command_names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def is_agent_command(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError:
            return False
        if not parts:
            return False
        spec = self._by_name.get(parts[0])
        return bool(spec and spec.agent_command)

    def candidates(self) -> list[CommandSpec]:
        return list(self.specs)

    def format_help(self) -> str:
        lines = ["VeriForge 命令：", ""]
        width = max((len(spec.usage) for spec in self.specs), default=0)
        for spec in self.specs:
            lines.append(f"{spec.usage:<{width}}  {spec.description}")
        return "\n".join(lines)


def default_command_registry(skill_registry=None) -> SlashCommandRegistry:
    specs = [
        CommandSpec("/checkpoint", "工作流", "/checkpoint", "打开检查点管理", _checkpoint_picker),
        CommandSpec("/mcp", "工作流", "/mcp", "打开 MCP 服务与工具管理", _mcp_picker),
        CommandSpec("/compact", "工作流", "/compact", "压缩当前对话上下文", _compact_now),
        CommandSpec("/context", "会话", "/context", "查看上下文预算组成", _context_status),
        CommandSpec("/memory", "会话", "/memory", "查看与管理长期记忆", _memory_command),
        CommandSpec("/fork", "会话", "/fork", "从当前会话创建并进入分支", _fork_current),
        CommandSpec("/observe", "会话", "/observe", "打开当前项目的运行观察", _observe_panel),
    ]
    if skill_registry is None:
        from ..skills import SkillRegistry

        skill_registry = SkillRegistry()
    occupied = {
        name
        for spec in specs
        for name in spec.names()
    }
    for item in getattr(skill_registry, "user_commands", []):
        skill_name = str(item.get("name", "")).strip()
        if not skill_name:
            continue
        command_name = f"/{skill_name}"
        if command_name in occupied:
            raise ValueError(
                f"User skill command {command_name} conflicts with built-in command"
            )
        hint = str(item.get("argument_hint", "")).strip()
        description = str(item.get("description", "")).strip()
        usage = f"{command_name} {hint}".rstrip()
        specs.append(
            CommandSpec(
                command_name,
                "Skills",
                usage,
                description,
                _user_skill(command_name),
                agent_command=True,
            )
        )
        occupied.add(command_name)
    return SlashCommandRegistry(specs)


def _user_skill(command_name: str) -> CommandHandler:
    def handler(session: Any, args: list[str], registry: SlashCommandRegistry) -> CommandResult:
        line = " ".join([command_name, *args])
        result = session.submit_skill_command(line)
        return CommandResult(result.text)

    return handler


def _panel_action(action: str, args: list[str], usage: str) -> CommandResult:
    _no_args(args, f"用法：{usage}")
    return CommandResult(action=action)


def _checkpoint_picker(session: Any, args: list[str], registry: SlashCommandRegistry) -> CommandResult:
    return _panel_action("checkpoint", args, "/checkpoint")


def _mcp_picker(session: Any, args: list[str], registry: SlashCommandRegistry) -> CommandResult:
    return _panel_action("mcp", args, "/mcp")


def _compact_now(session: Any, args: list[str], registry: SlashCommandRegistry) -> CommandResult:
    _no_args(args, "用法：/compact")
    _require_bound(session)
    return CommandResult(action="compact")


def _context_status(session: Any, args: list[str], registry: SlashCommandRegistry) -> CommandResult:
    _no_args(args, "用法：/context")
    return CommandResult(session.context_status())


def _memory_command(session: Any, args: list[str], registry: SlashCommandRegistry) -> CommandResult:
    return CommandResult(session.memory_command(args))


def _fork_current(session: Any, args: list[str], registry: SlashCommandRegistry) -> CommandResult:
    return _panel_action("fork", args, "/fork")


def _observe_panel(session: Any, args: list[str], registry: SlashCommandRegistry) -> CommandResult:
    return _panel_action("observe", args, "/observe")


def _no_args(args: list[str], usage: str) -> None:
    if args:
        raise ValueError(usage)


def _require_bound(session: Any) -> None:
    if not getattr(session, "is_bound", True):
        raise ValueError("No active session yet. Submit a task first.")
