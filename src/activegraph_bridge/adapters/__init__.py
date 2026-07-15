"""Agent adapters: framework knowledge behind a small protocol."""

from .base import (
    AdapterCapabilities,
    AgentAdapter,
    register_adapter,
    resolve_adapter,
)
from .generic import GenericAdapter

__all__ = [
    "AdapterCapabilities",
    "AgentAdapter",
    "GenericAdapter",
    "register_adapter",
    "resolve_adapter",
]
