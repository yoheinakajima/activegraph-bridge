"""The adapter protocol: where framework-specific knowledge lives.

The bridge core knows nothing about any agent framework. An
:class:`AgentAdapter` supplies the four things that differ between
frameworks:

1. **Recognition** — ``supports(target)``.
2. **Invocation** — how to call the agent (``invoke``/``ainvoke``), and
   which methods stream.
3. **Instrumentation** — a context manager that hooks the framework's
   seams (middleware, callbacks-with-control, client injection) so
   model/tool calls route through the session. The generic adapter's is
   a no-op: generic agents are instrumented by bridged tools,
   instrumented clients, and ``effect(...)`` calls in agent code.
4. **State** — optional ``checkpoint``/``restore`` for framework-native
   snapshots.

Capabilities are the honesty mechanism: an adapter must not claim
``can_short_circuit_calls`` unless its instrumentation can *prevent* the
live call during replay and return the recorded value. Observation-only
callbacks earn inspection, never replay.

Third-party adapters register via the ``activegraph_bridge.adapters``
entry-point group, or explicitly with :func:`register_adapter` — the
core stays dependency-free either way.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Callable, ContextManager, Protocol, runtime_checkable

__all__ = ["AdapterCapabilities", "AgentAdapter", "register_adapter", "resolve_adapter"]


@dataclass(frozen=True)
class AdapterCapabilities:
    """What an adapter can honestly promise. Recorded on every run."""

    intercepts_models: bool = False
    intercepts_tools: bool = False
    can_short_circuit_calls: bool = False
    captures_framework_steps: bool = False
    supports_streaming: bool = False
    supports_checkpoint: bool = False
    supports_restore: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@runtime_checkable
class AgentAdapter(Protocol):
    """Framework-specific glue behind a small, stable protocol."""

    name: str
    capabilities: AdapterCapabilities

    def supports(self, target: Any) -> bool: ...

    def invoke(self, target: Any, method: str, args: tuple, kwargs: dict) -> Any: ...

    async def ainvoke(self, target: Any, method: str, args: tuple, kwargs: dict) -> Any: ...

    def instrument(self, target: Any, session: Any) -> ContextManager[None]: ...

    def checkpoint(self, target: Any) -> Any | None: ...

    def restore(self, factory: Callable[[], Any] | None, checkpoint: Any) -> Any: ...


_REGISTRY: list[AgentAdapter] = []
_ENTRY_POINTS_LOADED = False


def register_adapter(adapter: AgentAdapter, *, prepend: bool = True) -> None:
    """Register an adapter for ``adapter="auto"`` resolution.

    Later registrations win by default so specific adapters shadow the
    generic fallback.
    """
    if prepend:
        _REGISTRY.insert(0, adapter)
    else:
        _REGISTRY.append(adapter)


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="activegraph_bridge.adapters"):
            try:
                register_adapter(ep.load()(), prepend=False)
            except Exception:  # a broken third-party adapter must not break wrap()
                continue
    except Exception:
        pass


def resolve_adapter(adapter: Any, target: Any) -> AgentAdapter:
    """Resolve the ``adapter=`` argument of ``wrap()``.

    ``"auto"`` walks registered adapters (specific first, generic last);
    a string selects by name; an instance passes through.
    """
    from .generic import GenericAdapter

    if isinstance(adapter, AgentAdapter) and not isinstance(adapter, str):
        return adapter
    _load_entry_points()
    if adapter == "auto" or adapter is None:
        for candidate in _REGISTRY:
            try:
                if candidate.supports(target):
                    return candidate
            except Exception:
                continue
        return GenericAdapter()
    if isinstance(adapter, str):
        for candidate in _REGISTRY:
            if candidate.name == adapter:
                return candidate
        if adapter == "generic":
            return GenericAdapter()
        from ..errors import BridgeConfigurationError

        known = sorted({c.name for c in _REGISTRY} | {"generic"})
        raise BridgeConfigurationError(
            f"unknown adapter {adapter!r}",
            what_failed=f"wrap(..., adapter={adapter!r}) matched no registered adapter.",
            why=(
                "Adapters are looked up by name in the registry (built-ins plus "
                "the activegraph_bridge.adapters entry-point group)."
            ),
            how_to_fix=(
                f"Use one of {known}, pass an AgentAdapter instance, or leave "
                f"adapter='auto'."
            ),
        )
    from ..errors import BridgeConfigurationError

    raise BridgeConfigurationError(
        f"adapter must be 'auto', a name, or an AgentAdapter; got {type(adapter).__name__}"
    )


@contextmanager
def no_instrumentation():
    yield


NULL_INSTRUMENTATION = nullcontext
