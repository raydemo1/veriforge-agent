"""Tool registry and schema filtering primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .execution_planner import CallEffect
from .permissions import VALID_TOOL_PERMISSIONS
from .tool_call_validation import strict_tool_schema

if TYPE_CHECKING:
    from .tool_context import ToolContext

EffectResolver = Callable[[dict, "ToolContext | None"], CallEffect]
TOOL_CAPABILITY_MAIN = "main"
TOOL_CAPABILITY_READONLY_AGENT = "readonly_agent"
TOOL_CAPABILITY_WORKER_AGENT = "worker_agent"
VALID_TOOL_CAPABILITIES = {
    TOOL_CAPABILITY_MAIN,
    TOOL_CAPABILITY_READONLY_AGENT,
    TOOL_CAPABILITY_WORKER_AGENT,
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: dict
    handler: Callable
    permission: str
    effect_resolver: EffectResolver
    disclosure: str = "core"
    capabilities: frozenset[str] = frozenset({TOOL_CAPABILITY_MAIN})


VALID_TOOL_DISCLOSURES = {"core", "deferred"}


class ToolRegistry:
    """Thin registry boundary for built-in and future profile-provided tools."""

    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}

    def register(
        self,
        schema: dict,
        handler: Callable,
        *,
        permission: str | None = None,
        effect: EffectResolver | CallEffect | None = None,
        disclosure: str = "core",
        capabilities: set[str] | frozenset[str] | None = None,
    ) -> None:
        name = schema.get("function", {}).get("name")
        if not name:
            raise ValueError("Tool schema missing function.name")
        if permission is None:
            raise ValueError(f"Tool {name} missing permission classification")
        if permission not in VALID_TOOL_PERMISSIONS:
            raise ValueError(
                f"Tool {name} has unknown permission classification: {permission}"
            )
        if disclosure not in VALID_TOOL_DISCLOSURES:
            raise ValueError(
                f"Tool {name} has unknown disclosure classification: {disclosure}"
            )
        resolved_capabilities = frozenset(capabilities or {TOOL_CAPABILITY_MAIN})
        unknown_capabilities = resolved_capabilities - VALID_TOOL_CAPABILITIES
        if unknown_capabilities:
            names = ", ".join(sorted(unknown_capabilities))
            raise ValueError(f"Tool {name} has unknown capabilities: {names}")
        if isinstance(effect, CallEffect):
            resolver = lambda _args, _context, value=effect: value
        elif effect is not None:
            resolver = effect
        else:
            resolver = lambda _args, _context: CallEffect.global_exclusive()
        self._specs[name] = ToolSpec(
            name=name,
            schema=strict_tool_schema(schema),
            handler=handler,
            permission=permission,
            effect_resolver=resolver,
            disclosure=disclosure,
            capabilities=resolved_capabilities,
        )

    def register_spec(self, spec: ToolSpec) -> None:
        self.register(
            spec.schema,
            spec.handler,
            permission=spec.permission,
            effect=spec.effect_resolver,
            disclosure=spec.disclosure,
            capabilities=spec.capabilities,
        )

    def get(self, name: str) -> Callable | None:
        spec = self._specs.get(name)
        return spec.handler if spec is not None else None

    def spec_for(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def schema_for(self, name: str) -> dict | None:
        spec = self._specs.get(name)
        return spec.schema if spec is not None else None

    def permission_for(self, name: str) -> str | None:
        spec = self._specs.get(name)
        return spec.permission if spec is not None else None

    def effect_for(
        self, name: str, args: dict, context: ToolContext | None = None
    ) -> CallEffect:
        spec = self._specs.get(name)
        resolver = spec.effect_resolver if spec is not None else None
        return (
            resolver(dict(args or {}), context)
            if resolver is not None
            else CallEffect.global_exclusive()
        )

    def effect_resolver_for(self, name: str) -> EffectResolver | None:
        spec = self._specs.get(name)
        return spec.effect_resolver if spec is not None else None

    def disclosure_for(self, name: str) -> str | None:
        spec = self._specs.get(name)
        return spec.disclosure if spec is not None else None

    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def schemas(self) -> list[dict]:
        return [self._specs[name].schema for name in sorted(self._specs)]

    def dispatch(self) -> dict[str, Callable]:
        return {name: spec.handler for name, spec in self._specs.items()}

    def copy(self) -> ToolRegistry:
        clone = ToolRegistry()
        clone._specs = dict(self._specs)
        return clone


def tool_schemas_for_profile(
    *,
    allowed_permissions: set[str] | None = None,
    include_names: set[str] | None = None,
    exclude_names: set[str] | None = None,
    registry: ToolRegistry | None = None,
    disclosure: set[str] | None = None,
    capabilities: set[str] | frozenset[str] | None = None,
) -> list[dict]:
    if registry is None:
        from .builtins.registry import BUILTIN_TOOL_REGISTRY

        registry = BUILTIN_TOOL_REGISTRY
    allowed_permissions = (
        set(allowed_permissions) if allowed_permissions is not None else None
    )
    include_names = set(include_names or set())
    exclude_names = set(exclude_names or set())
    disclosures = {"core"} if disclosure is None else set(disclosure)
    required_capabilities = frozenset(capabilities or {TOOL_CAPABILITY_MAIN})
    unknown_capabilities = required_capabilities - VALID_TOOL_CAPABILITIES
    if unknown_capabilities:
        names = ", ".join(sorted(unknown_capabilities))
        raise ValueError(f"Unknown tool capability classification for profile: {names}")
    if allowed_permissions is not None:
        unknown_permissions = allowed_permissions - VALID_TOOL_PERMISSIONS
        if unknown_permissions:
            names = ", ".join(sorted(unknown_permissions))
            raise ValueError(
                f"Unknown tool permission classification for profile: {names}"
            )
    unknown_disclosures = disclosures - VALID_TOOL_DISCLOSURES
    if unknown_disclosures:
        names = ", ".join(sorted(unknown_disclosures))
        raise ValueError(f"Unknown tool disclosure classification for profile: {names}")

    schemas: list[dict] = []
    for spec in registry.specs():
        if not (spec.capabilities & required_capabilities):
            continue
        if spec.name in exclude_names:
            continue
        if spec.disclosure not in disclosures:
            continue
        if spec.name in include_names or (
            allowed_permissions is not None and spec.permission in allowed_permissions
        ):
            schemas.append(spec.schema)
    return schemas
