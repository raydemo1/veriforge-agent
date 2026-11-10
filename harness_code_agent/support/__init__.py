"""Leaf primitives shared by every package: no imports from agent/runtime/... .

Layering rule for this repository:

    support                     (this package: stdlib + config only)
      ^
    runtime, workspace, memory, sessions
      ^
    agent
      ^
    core, tui, tui_bridge, cli, headless

Anything in ``support`` may be used by every layer. Nothing here may import
``agent``, ``runtime``, ``sessions``, ``workspace``, ``memory``, ``core`` or
``tui``: that is what keeps the dependency graph acyclic.
"""
