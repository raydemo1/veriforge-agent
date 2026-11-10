"""Errors raised by the runtime layer that callers above it must handle."""
from __future__ import annotations


class SubagentCapacityError(ValueError):
    """The maximum number of open agents has been reached."""
