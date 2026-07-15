"""Bridge error hierarchy.

Every bridge error inherits from :class:`BridgeError`, which inherits
from :class:`activegraph.ActiveGraphError` — so ``except
ActiveGraphError`` in host applications covers the bridge too, and every
error renders in ActiveGraph's locked structured format (summary, what
failed, why, how to fix).

The categories mirror the guarantees the bridge makes:

- :class:`ReplayDivergence` — a shadow execution requested something the
  recording did not contain, or in a different shape. Raised during
  ``verify()`` and during a fork's prefix replay.
- :class:`UnrecordedEffectError` — live I/O was attempted while effects
  were supposed to be served from the record. This is the fail-closed
  teeth behind "strict verification makes zero live calls".
- :class:`EffectBlockedError` — a write effect was refused by the
  side-effect policy (replay, verify, and fork tails never execute
  writes unless explicitly authorized).
- :class:`ReconstructionError` — a replay/verify/fork needed a fresh
  agent and no reconstruction strategy (factory, adapter reset, or
  checkpoint) exists.
- :class:`NotForkableError` — the selected fork point is not a safe
  boundary.
- :class:`OverrideError` — a fork override did not apply cleanly.
"""

from __future__ import annotations

from typing import Any

from activegraph.errors import (
    ActiveGraphError,
    ConfigurationError,
    ExecutionError,
    ReplayError,
)

__all__ = [
    "BridgeError",
    "BridgeConfigurationError",
    "ReplayDivergence",
    "UnrecordedEffectError",
    "EffectBlockedError",
    "ReplayedEffectFailure",
    "ReconstructionError",
    "NotForkableError",
    "OverrideError",
]

_DOCS_BASE = "https://github.com/yoheinakajima/activegraph-bridge/blob/main/docs"


class BridgeError(ActiveGraphError):
    """Root of every activegraph-bridge error."""

    _doc_slug = "bridge-error"

    @property
    def doc_url(self) -> str:  # bridge docs live in this repo, not docs.activegraph.ai
        return f"{_DOCS_BASE}/errors.md#{self._doc_slug}"


class BridgeConfigurationError(BridgeError, ConfigurationError):
    """Invalid ``wrap(...)`` / ``fork(...)`` / store configuration."""

    _doc_slug = "bridge-configuration-error"


class ReplayDivergence(BridgeError, ReplayError):
    """A re-execution requested effects that do not match the recording.

    Carries ``expected`` and ``got`` dicts describing both sides of the
    mismatch so tools can render a precise diff.
    """

    _doc_slug = "replay-divergence"

    def __init__(
        self,
        summary: str,
        *,
        expected: dict[str, Any] | None = None,
        got: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.expected = expected or {}
        self.got = got or {}
        context = dict(kwargs.pop("context", None) or {})
        context.setdefault("expected", self.expected)
        context.setdefault("got", self.got)
        super().__init__(summary, context=context, **kwargs)


class UnrecordedEffectError(BridgeError, ReplayError):
    """Live I/O attempted while effects must be served from the record.

    Raised by the shadow-execution guard when agent code reaches for a
    socket, subprocess, or an unmediated call during ``verify()`` or a
    fork's recorded prefix. Fail-closed by design: a verification that
    silently allowed one live call could not honestly claim replay.
    """

    _doc_slug = "unrecorded-effect"


class EffectBlockedError(BridgeError, ExecutionError):
    """A write effect was refused by the active side-effect policy."""

    _doc_slug = "effect-blocked"

    def __init__(
        self,
        summary: str,
        *,
        kind: str = "",
        name: str = "",
        mode: str = "",
        **kwargs: Any,
    ) -> None:
        self.kind = kind
        self.name = name
        self.mode = mode
        context = dict(kwargs.pop("context", None) or {})
        context.update({"kind": kind, "name": name, "mode": mode})
        super().__init__(summary, context=context, **kwargs)


class ReplayedEffectFailure(BridgeError, ExecutionError):
    """A recorded effect failure re-raised during replay.

    When the original exception class can be imported it is re-raised
    directly; this class is the fallback when the recorded exception
    cannot be faithfully rehydrated. ``original_class`` names what the
    live run actually raised.
    """

    _doc_slug = "replayed-effect-failure"

    def __init__(self, message: str, *, original_class: str = "") -> None:
        self.original_class = original_class
        super().__init__(message)


class ReconstructionError(BridgeError, ReplayError):
    """No way to build a clean agent for replay, verify, or fork."""

    _doc_slug = "reconstruction-error"


class NotForkableError(BridgeError, ReplayError):
    """The selected event is not a safe fork boundary."""

    _doc_slug = "not-forkable"


class OverrideError(BridgeError, ReplayError):
    """A fork override could not be applied at the fork point."""

    _doc_slug = "override-error"
