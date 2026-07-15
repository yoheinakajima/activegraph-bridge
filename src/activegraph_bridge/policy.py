"""Side-effect policy: reads and writes are not treated alike.

Every effect declares a side-effect class:

- ``"read"``     — observing the world (search, DB query, model call)
- ``"write"``    — changing the world (send email, update CRM)
- ``"unknown"``  — the conservative default; treated as a write wherever
  the distinction matters

The policy answers one question — *may this effect execute live, right
now?* — and the answer depends on the session mode:

================  ======================  =====================================
mode              read                    write / unknown
================  ======================  =====================================
record (live)     execute and record      ``on_write`` (default: execute)
verify / replay   served from record      **never executes**
fork prefix       served from record      **never executes**
fork tail         execute and record      ``on_fork_write`` (default: block)
================  ======================  =====================================

"Never executes" is not configurable. What *is* configurable is how a
fork tail handles writes: block (fail closed), simulate through a
caller-supplied simulator, route through an approval handler, or — with
explicit authorization — execute live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .errors import EffectBlockedError

__all__ = ["SideEffect", "SideEffectPolicy", "WriteDecision"]

SideEffect = Literal["read", "write", "unknown"]

WriteDecision = Literal["execute", "simulate", "block", "approval"]


@dataclass(frozen=True)
class SideEffectPolicy:
    """How write effects are handled in live and fork-tail execution.

    ``on_write``       — live recording. Default ``"execute"``: the agent
                         is running for real, so its writes happen,
                         and the log captures what was committed.
    ``on_fork_write``  — a fork's divergent tail. Default ``"block"``:
                         a counterfactual must not send real emails. Set
                         to ``"simulate"`` (with a ``simulator``),
                         ``"approval"`` (with an ``approve`` handler), or
                         ``"execute"`` for an explicitly-live fork.
    ``simulator``      — ``(kind, name, request) -> simulated response``.
    ``approve``        — ``(kind, name, request) -> bool``; also used when
                         ``on_write="approval"`` during live recording.
    ``approved_by``    — identity recorded on approved writes.
    """

    on_write: WriteDecision = "execute"
    on_fork_write: WriteDecision = "block"
    simulator: Callable[[str, str, Any], Any] | None = None
    approve: Callable[[str, str, Any], bool] | None = None
    approved_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def decide(self, *, side_effect: SideEffect, fork_tail: bool) -> WriteDecision:
        """Resolve the decision for one live-execution attempt.

        Only consulted in modes where live execution is possible at all
        (record, fork tail). Reads always execute; writes and unknowns
        follow the configured decision.
        """
        if side_effect == "read":
            return "execute"
        return self.on_fork_write if fork_tail else self.on_write

    def blocked(self, *, kind: str, name: str, mode: str, decision: WriteDecision) -> EffectBlockedError:
        """Build the fail-closed error for a refused write."""
        return EffectBlockedError(
            f"write effect {name or kind!r} blocked in {mode} mode",
            kind=kind,
            name=name,
            mode=mode,
            what_failed=(
                f"The agent attempted the write effect {name or kind!r} while the "
                f"session was in {mode} mode with write decision {decision!r}."
            ),
            why=(
                "Replays, verifications, and fork tails default to fail-closed for "
                "writes: a re-execution that silently re-sent an email or re-charged "
                "a card would make 'replay' a dangerous word. Reads are served or "
                "re-executed freely; writes need explicit policy."
            ),
            how_to_fix=(
                "Choose one, in order of preference:\n"
                "  1. Provide a simulator: SideEffectPolicy(on_fork_write='simulate',\n"
                "     simulator=lambda kind, name, request: {...})\n"
                "  2. Route through approval: SideEffectPolicy(on_fork_write='approval',\n"
                "     approve=handler)\n"
                "  3. Authorize live writes for this fork explicitly:\n"
                "     run.fork(..., side_effects='live')\n"
                "\n"
                "If this effect is actually read-only, declare it: "
                "side_effect='read' on the tool or effect call."
            ),
        )
