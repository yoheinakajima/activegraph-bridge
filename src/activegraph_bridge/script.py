"""Effect scripts: a recorded run parsed into a servable sequence.

A shadow execution (``verify()``, or a fork's recorded prefix) does not
walk raw events — it walks a **script**: the ordered sequence of
invocations, effects, and checkpoints reconstructed from the log. The
cursor over that script is what turns "we logged what happened" into
"we can serve what happened, in order, and notice the moment reality
disagrees".

Matching discipline:

- ``strict`` — the next unconsumed entry must match exactly (kind, name,
  and request hash). Any deviation is a divergence.
- ``auto`` (default) — position mismatches search the current
  invocation's remaining effects for an exact content match, consuming
  it out of order and noting a finding. This tolerates completion-order
  variance in concurrent agents without ever inventing a response:
  content must still match exactly, only position may flex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from activegraph.core.event import Event

from . import events as ev
from .errors import ReplayDivergence

__all__ = ["EffectEntry", "InvocationEntry", "EffectScript", "Cursor"]


@dataclass(frozen=True)
class EffectEntry:
    """One recorded effect: the request and its recorded outcome."""

    request_event_id: str
    kind: str
    name: str
    category: str
    side_effect: str
    request_hash: str
    request: Any
    codec_name: str
    quiescent: bool
    response: Any = None  # encoded response document (None if failed)
    error: Any = None  # encoded exception document (None if succeeded)
    served_from: str = "live"
    response_event_id: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class InvocationEntry:
    """One recorded invocation boundary (start or end)."""

    boundary: str  # "start" | "end"
    ordinal: int
    method: str = "invoke"
    input_hash: str = ""
    input: Any = None
    output_hash: str = ""
    output: Any = None
    error: Any = None
    start_event_id: str = ""


@dataclass(frozen=True)
class CheckpointEntry:
    ordinal: int
    label: str | None
    state: Any
    event_id: str


ScriptEntry = EffectEntry | InvocationEntry | CheckpointEntry


@dataclass
class EffectScript:
    """The recorded execution, in servable order."""

    entries: list[ScriptEntry] = field(default_factory=list)

    @classmethod
    def from_events(cls, events: list[Event]) -> "EffectScript":
        entries: list[ScriptEntry] = []
        pairs = {req.id: resp for req, resp in ev.effect_pairs(events)}
        open_invocations: dict[int, Event] = {}
        for e in events:
            if e.type == ev.INVOCATION_STARTED:
                ordinal = int(e.payload.get("ordinal", 1))
                open_invocations[ordinal] = e
                entries.append(
                    InvocationEntry(
                        boundary="start",
                        ordinal=ordinal,
                        method=str(e.payload.get("method", "invoke")),
                        input_hash=str(e.payload.get("input_hash", "")),
                        input=e.payload.get("input"),
                        start_event_id=e.id,
                    )
                )
            elif e.type in (ev.INVOCATION_COMPLETED, ev.INVOCATION_FAILED):
                ordinal = int(e.payload.get("ordinal", 1))
                start = open_invocations.pop(ordinal, None)
                entries.append(
                    InvocationEntry(
                        boundary="end",
                        ordinal=ordinal,
                        method=str((start.payload.get("method") if start else "invoke") or "invoke"),
                        output_hash=str(e.payload.get("output_hash", "")),
                        output=e.payload.get("output"),
                        error=e.payload.get("error"),
                        start_event_id=start.id if start else "",
                    )
                )
            elif e.type == ev.EFFECT_REQUESTED:
                resp = pairs.get(e.id)
                entries.append(
                    EffectEntry(
                        request_event_id=e.id,
                        kind=str(e.payload.get("kind", "")),
                        name=str(e.payload.get("name", "")),
                        category=str(e.payload.get("category", "external")),
                        side_effect=str(e.payload.get("side_effect", "unknown")),
                        request_hash=str(e.payload.get("request_hash", "")),
                        request=e.payload.get("request"),
                        codec_name=str(e.payload.get("codec", "auto")),
                        quiescent=bool(e.payload.get("quiescent", True)),
                        response=(resp.payload.get("response") if resp is not None and resp.type == ev.EFFECT_RESPONDED else None),
                        error=(resp.payload.get("error") if resp is not None and resp.type == ev.EFFECT_FAILED else None),
                        served_from=str(resp.payload.get("served_from", "live")) if resp is not None else "live",
                        response_event_id=resp.id if resp is not None else None,
                    )
                )
            elif e.type == ev.CHECKPOINT_RECORDED:
                entries.append(
                    CheckpointEntry(
                        ordinal=int(e.payload.get("ordinal", 1)),
                        label=e.payload.get("label"),
                        state=e.payload.get("state"),
                        event_id=e.id,
                    )
                )
        return cls(entries=entries)

    # -- queries -------------------------------------------------------------

    def invocation_starts(self) -> list[InvocationEntry]:
        return [
            e
            for e in self.entries
            if isinstance(e, InvocationEntry) and e.boundary == "start"
        ]

    def top_level_invocations(self) -> list[tuple[InvocationEntry, InvocationEntry | None]]:
        """(start, end) pairs for depth-0 invocations, in order.

        Nested invocations (a wrapped agent invoked inside another wrapped
        agent's run) stay inside the script and are consumed by the
        wrapper during shadow execution; only depth-0 pairs are re-driven
        by the engine.
        """
        out: list[tuple[InvocationEntry, InvocationEntry | None]] = []
        depth = 0
        current: InvocationEntry | None = None
        for entry in self.entries:
            if isinstance(entry, InvocationEntry):
                if entry.boundary == "start":
                    depth += 1
                    if depth == 1:
                        current = entry
                else:
                    if depth == 1 and current is not None:
                        out.append((current, entry))
                        current = None
                    depth = max(0, depth - 1)
        if current is not None:
            out.append((current, None))  # crashed mid-invocation
        return out

    def index_of_event(self, event_id: str) -> int | None:
        for i, entry in enumerate(self.entries):
            eid = getattr(entry, "request_event_id", None) or getattr(
                entry, "start_event_id", None
            ) or getattr(entry, "event_id", None)
            if eid == event_id:
                return i
        return None

    def truncated_before(self, event_id: str) -> "EffectScript":
        """The prefix script strictly before the entry anchored at event_id."""
        idx = self.index_of_event(event_id)
        if idx is None:
            raise ValueError(
                f"event {event_id!r} does not anchor any script entry"
            )
        return EffectScript(entries=list(self.entries[:idx]))


class Cursor:
    """Serving position over an :class:`EffectScript`.

    One cursor per shadow execution. ``take_effect`` is the workhorse:
    given the request the re-executing agent just made, it returns the
    recorded entry to serve — or raises :class:`ReplayDivergence` with
    both sides of the mismatch.
    """

    def __init__(self, script: EffectScript, *, match: str = "auto") -> None:
        if match not in ("auto", "strict"):
            raise ValueError(f"match must be 'auto' or 'strict', got {match!r}")
        self.script = script
        self.match = match
        self._consumed: set[int] = set()
        self._pos = 0
        self.reordered: list[str] = []  # request hashes served out of order

    # -- position helpers ----------------------------------------------------

    def _advance_past_consumed(self) -> None:
        while self._pos < len(self.script.entries) and (
            self._pos in self._consumed
            or isinstance(self.script.entries[self._pos], CheckpointEntry)
        ):
            # Checkpoints are replay no-ops: they were state snapshots of
            # the recording run, not effects the agent needs served.
            self._pos += 1

    def peek(self) -> ScriptEntry | None:
        self._advance_past_consumed()
        if self._pos >= len(self.script.entries):
            return None
        return self.script.entries[self._pos]

    @property
    def exhausted(self) -> bool:
        return self.peek() is None

    def remaining_effects(self) -> list[EffectEntry]:
        self._advance_past_consumed()
        return [
            e
            for i, e in enumerate(self.script.entries)
            if i >= self._pos and i not in self._consumed and isinstance(e, EffectEntry)
        ]

    # -- consuming -----------------------------------------------------------

    def _consume(self, index: int) -> None:
        self._consumed.add(index)
        if index == self._pos:
            self._pos += 1
        self._advance_past_consumed()

    def take_effect(
        self, *, kind: str, name: str, request_hash: str
    ) -> EffectEntry | None:
        """Serve the next recorded effect matching this request.

        Returns ``None`` when the cursor is exhausted (the caller decides
        whether that means divergence — verify — or the fork point).
        Raises :class:`ReplayDivergence` on a content mismatch.
        """
        entry = self.peek()
        if entry is None:
            return None
        if isinstance(entry, EffectEntry):
            if (
                entry.kind == kind
                and entry.name == name
                and entry.request_hash == request_hash
            ):
                self._consume(self._pos)
                return entry
            if self.match == "auto":
                found = self._search_current_stretch(kind, name, request_hash)
                if found is not None:
                    index, matched = found
                    self._consume(index)
                    self.reordered.append(request_hash)
                    return matched
            raise ReplayDivergence(
                f"re-execution requested {name or kind!r} but the recording "
                f"expected {entry.name or entry.kind!r} here",
                expected={
                    "kind": entry.kind,
                    "name": entry.name,
                    "request_hash": entry.request_hash,
                    "request": entry.request,
                    "event": entry.request_event_id,
                },
                got={"kind": kind, "name": name, "request_hash": request_hash},
                what_failed=(
                    f"At recorded position {self._pos}, the agent requested effect "
                    f"kind={kind!r} name={name!r} hash={request_hash[:12]}…, but the "
                    f"recording holds kind={entry.kind!r} name={entry.name!r} "
                    f"hash={entry.request_hash[:12]}… (event {entry.request_event_id})."
                ),
                why=(
                    "Replay serves recorded responses only when the re-executed "
                    "request is byte-identical (canonical JSON hash) to what was "
                    "recorded. A different request means the agent's code, prompt "
                    "assembly, or unrecorded inputs changed — serving the old "
                    "response would silently lie about what the agent would do."
                ),
                how_to_fix=(
                    "If the divergence is intentional (changed code or input), use "
                    "run.fork() instead of run.verify() — forks expect a divergent "
                    "tail. If it is not intentional, look for unmediated "
                    "nondeterminism feeding the request: current time, random "
                    "values, or uuids should come from the session's deterministic "
                    "sources (activegraph_bridge.det)."
                ),
            )
        # Next entry is an invocation boundary: the agent is making a call
        # the recording says shouldn't exist inside this invocation.
        raise ReplayDivergence(
            f"re-execution requested {name or kind!r} but the recording expected "
            f"the invocation boundary ({entry.boundary}) here",
            expected={"boundary": entry.boundary, "ordinal": entry.ordinal},
            got={"kind": kind, "name": name, "request_hash": request_hash},
            what_failed=(
                f"The agent requested one more effect ({name or kind!r}) than the "
                f"recording contains for this stretch of the run."
            ),
            why=(
                "Extra calls mean the re-executed control flow diverged from the "
                "recorded run — a response served here would have no recorded basis."
            ),
            how_to_fix=(
                "Check for nondeterministic control flow (time, randomness, "
                "iteration order over sets/dicts of unstable identity), or fork "
                "instead of verifying if the change is intentional."
            ),
        )

    def _search_current_stretch(
        self, kind: str, name: str, request_hash: str
    ) -> tuple[int, EffectEntry] | None:
        """Auto-match: exact content match within the current invocation's
        remaining effects (stop scanning at the next invocation boundary)."""
        for i in range(self._pos, len(self.script.entries)):
            if i in self._consumed:
                continue
            entry = self.script.entries[i]
            if isinstance(entry, InvocationEntry):
                break
            if (
                isinstance(entry, EffectEntry)
                and entry.kind == kind
                and entry.name == name
                and entry.request_hash == request_hash
            ):
                return i, entry
        return None

    def take_invocation(self, *, boundary: str, input_hash: str | None = None) -> InvocationEntry | None:
        """Consume the next entry as an invocation boundary, validating it."""
        entry = self.peek()
        if entry is None:
            return None
        if not isinstance(entry, InvocationEntry) or entry.boundary != boundary:
            described = (
                f"effect {entry.name or entry.kind!r}"
                if isinstance(entry, EffectEntry)
                else f"invocation {getattr(entry, 'boundary', '?')}"
            )
            if boundary == "end" and isinstance(entry, EffectEntry):
                raise ReplayDivergence(
                    "re-execution finished the invocation before making every recorded call",
                    expected={
                        "kind": entry.kind,
                        "name": entry.name,
                        "request_hash": entry.request_hash,
                        "event": entry.request_event_id,
                    },
                    got={"boundary": "end"},
                    what_failed=(
                        f"The invocation returned, but the recording still holds "
                        f"{len(self.remaining_effects())} unserved effect(s), next: "
                        f"{entry.name or entry.kind!r} (event {entry.request_event_id})."
                    ),
                    why=(
                        "Fewer calls than recorded is divergence just like extra "
                        "calls: the re-executed agent took a different path."
                    ),
                    how_to_fix=(
                        "Look for control flow driven by unrecorded state (module "
                        "globals, caches warm from a previous run, wall-clock time), "
                        "or fork instead of verifying if the change is intentional."
                    ),
                )
            raise ReplayDivergence(
                f"expected invocation {boundary}, recording holds {described}",
                expected={"boundary": boundary},
                got={"entry": described},
            )
        if (
            boundary == "start"
            and input_hash is not None
            and entry.input_hash
            and entry.input_hash != input_hash
        ):
            raise ReplayDivergence(
                "invocation input does not match the recording",
                expected={"input_hash": entry.input_hash, "input": entry.input},
                got={"input_hash": input_hash},
                what_failed=(
                    f"Invocation {entry.ordinal} was re-driven with an input whose "
                    f"canonical hash {input_hash[:12]}… differs from the recorded "
                    f"{entry.input_hash[:12]}…."
                ),
                why=(
                    "verify() replays the recorded inputs exactly; a different "
                    "input is a counterfactual, which is what fork() is for."
                ),
                how_to_fix=(
                    "Use run.fork(before=run.events.invocation(n), "
                    "overrides={'input': ...}) to explore a changed input."
                ),
            )
        self._consume(self._pos)
        return entry

    def entries_iter(self) -> Iterator[ScriptEntry]:
        return iter(self.script.entries)
