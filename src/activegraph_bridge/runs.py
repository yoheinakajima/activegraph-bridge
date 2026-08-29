"""Run handles: everything you can do with a recorded run.

A :class:`Run` is a durable reference — ``(store, run_id)`` plus,
when available, the wrap spec that knows how to rebuild the agent. The
four operations are deliberately distinct, because they promise
different things:

- :meth:`Run.replay` — rebuild the ActiveGraph projection from the
  event log. ActiveGraph's native meaning of replay: pure, offline,
  always available.
- :meth:`Run.playback_output` — return the recorded final output
  without executing anything. Works even for envelope-grade runs.
- :meth:`Run.verify` — fresh agent, recorded inputs, served effects,
  strict event comparison, zero live calls. Success is what earns
  ``boundary-verified``.
- :meth:`Run.fork` — branch before any safe boundary, optionally
  overriding the recorded response at the fork point (or the input, for
  invocation boundaries), then let the child develop a live tail.

Event selectors resolve human intent ("the first ``lookup_order``
call") to real event ids::

    run.events.tool_call("lookup_order", occurrence=1)
    run.events.model_call(2)
    run.events.invocation(1)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, TypeVar

from activegraph.core.clock import Clock
from activegraph.core.event import Event
from activegraph.core.graph import Graph
from activegraph.core.ids import IDGen
from activegraph.runtime.diff import Diff, compute_diff
from activegraph.store.base import RunRecord, replay_into

from . import events as ev
from ._canonical import content_hash
from ._store import BridgeStore, resolve_store
from .codecs import AutoCodec, EffectCodec
from .engine import (
    VerificationResult,
    WrapSpec,
    afork_execute,
    ashadow_verify,
    fork_execute,
    requires_async_drive,
    shadow_verify,
)
from .errors import (
    BridgeConfigurationError,
    NotForkableError,
    OverrideError,
    ReconstructionError,
)
from .report import ReplayabilityReport, compute_report
from .receipts import (
    EnvironmentAttestation,
    EnvironmentVerifier,
    ForkReceipt,
    effect_evidence,
    event_log_hash,
)
from .session import ForkOverride

__all__ = [
    "Run",
    "Fork",
    "RunDiff",
    "Replay",
    "EventRef",
    "load_run",
    "list_runs",
]

_T = TypeVar("_T")


def _run_async_operation(
    operation: Callable[[], Coroutine[Any, Any, _T]], *, async_api: str
) -> _T:
    """Run async work from sync APIs only when no event loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation())
    raise BridgeConfigurationError(
        "the synchronous API cannot drive async agent code inside an event loop",
        what_failed="A recorded async invocation was opened through a sync operation.",
        why="Starting a nested event loop would be unsafe and is rejected by asyncio.",
        how_to_fix=f"Use `{async_api}` in async code.",
    )


def _fingerprint_from_events(events: list[Event]) -> dict[str, Any]:
    for event in events:
        if event.type == ev.RUN_STARTED:
            return dict(event.payload.get("fingerprint") or {})
    return {}


# --------------------------------------------------------------------------- #
# selectors                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EventRef:
    """A resolved reference into a run's event log.

    ``event_id`` is the real ActiveGraph event id (usable directly with
    any native tooling); the rest is context for humans and for
    forkability checks.
    """

    event_id: str
    type: str
    kind: str = ""
    name: str = ""
    occurrence: int = 1
    ordinal: int | None = None
    quiescent: bool = True
    request_hash: str = ""
    footprint: str = "unknown"
    replay_source: str = "recorded"
    observables: tuple[str, ...] = ()

    @property
    def forkable(self) -> bool:
        if self.type == ev.EFFECT_REQUESTED:
            return self.quiescent
        return self.type in (ev.INVOCATION_STARTED, ev.CHECKPOINT_RECORDED)

    @property
    def resume_strategy(self) -> str:
        return "checkpoint" if self.type == ev.CHECKPOINT_RECORDED else "reexecute"

    def __str__(self) -> str:
        label = self.name or self.kind or self.type
        return f"{label}#{self.occurrence} ({self.event_id})"


class RunEvents:
    """Semantic selectors over a run's event log."""

    def __init__(self, events: list[Event]):
        self._events = events

    def all(self) -> list[Event]:
        return list(self._events)

    def __getitem__(self, event_id: str) -> Event:
        for e in self._events:
            if e.id == event_id:
                return e
        raise KeyError(event_id)

    def _effect_refs(self) -> list[EventRef]:
        refs: list[EventRef] = []
        seen: dict[tuple[str, str], int] = {}
        for e in self._events:
            if e.type != ev.EFFECT_REQUESTED:
                continue
            key = (str(e.payload.get("kind", "")), str(e.payload.get("name", "")))
            seen[key] = seen.get(key, 0) + 1
            refs.append(
                EventRef(
                    event_id=e.id,
                    type=e.type,
                    kind=key[0],
                    name=key[1],
                    occurrence=seen[key],
                    ordinal=e.payload.get("ordinal"),
                    quiescent=bool(e.payload.get("quiescent", True)),
                    request_hash=str(e.payload.get("request_hash", "")),
                    footprint=str(e.payload.get("footprint", "unknown")),
                    replay_source=str(e.payload.get("replay_source", "recorded")),
                    observables=tuple(
                        sorted(str(item) for item in (e.payload.get("observables") or []))
                    ),
                )
            )
        return refs

    def effects(
        self,
        *,
        kind: str | None = None,
        name: str | None = None,
        category: str | None = None,
    ) -> list[EventRef]:
        by_id = {e.id: e for e in self._events}
        out = []
        for ref in self._effect_refs():
            if kind is not None and ref.kind != kind:
                continue
            if name is not None and ref.name != name:
                continue
            if category is not None:
                request = by_id[ref.event_id]
                if str(request.payload.get("category")) != category:
                    continue
            out.append(ref)
        return out

    def _one(self, refs: list[EventRef], occurrence: int, what: str) -> EventRef:
        matching = [r for r in refs]
        if occurrence < 1 or occurrence > len(matching):
            raise KeyError(
                f"no {what} with occurrence={occurrence} "
                f"(found {len(matching)} in this run)"
            )
        ref = matching[occurrence - 1]
        return EventRef(
            event_id=ref.event_id,
            type=ref.type,
            kind=ref.kind,
            name=ref.name,
            occurrence=occurrence,
            ordinal=ref.ordinal,
            quiescent=ref.quiescent,
            request_hash=ref.request_hash,
            footprint=ref.footprint,
            replay_source=ref.replay_source,
            observables=ref.observables,
        )

    def effect(
        self,
        name: str | None = None,
        *,
        kind: str | None = None,
        occurrence: int = 1,
    ) -> EventRef:
        return self._one(
            self.effects(kind=kind, name=name), occurrence, f"effect {name or kind!r}"
        )

    def tool_call(self, name: str | None = None, occurrence: int = 1) -> EventRef:
        refs = [
            r
            for r in self.effects(category="tool")
            if name is None or r.name == name
        ]
        return self._one(refs, occurrence, f"tool call {name!r}" if name else "tool call")

    def model_call(self, occurrence: int = 1) -> EventRef:
        return self._one(self.effects(category="model"), occurrence, "model call")

    def retrieval(self, occurrence: int = 1) -> EventRef:
        return self._one(self.effects(category="retrieval"), occurrence, "retrieval")

    def invocation(self, ordinal: int = 1) -> EventRef:
        for e in self._events:
            if (
                e.type == ev.INVOCATION_STARTED
                and int(e.payload.get("ordinal", 0)) == ordinal
            ):
                return EventRef(
                    event_id=e.id,
                    type=e.type,
                    ordinal=ordinal,
                    occurrence=ordinal,
                )
        raise KeyError(f"no invocation with ordinal={ordinal}")

    def checkpoint(self, occurrence: int = 1) -> EventRef:
        refs = [
            EventRef(event_id=e.id, type=e.type, occurrence=i + 1)
            for i, e in enumerate(
                e for e in self._events if e.type == ev.CHECKPOINT_RECORDED
            )
        ]
        return self._one(refs, occurrence, "checkpoint")

    def fork_points(self) -> list[EventRef]:
        points = [
            EventRef(event_id=e.id, type=e.type, ordinal=e.payload.get("ordinal"))
            for e in self._events
            if e.type in (ev.INVOCATION_STARTED, ev.CHECKPOINT_RECORDED)
        ]
        points.extend(r for r in self._effect_refs() if r.quiescent)
        order = {e.id: i for i, e in enumerate(self._events)}
        points.sort(key=lambda r: order.get(r.event_id, 0))
        return points


# --------------------------------------------------------------------------- #
# replay projection                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class Replay:
    """The result of rebuilding a run's graph projection from its log."""

    graph: Graph
    events: int
    objects: int
    relations: int

    def __str__(self) -> str:
        return (
            f"Replay: {self.events} events -> {self.objects} objects, "
            f"{self.relations} relations (run {self.graph.run_id})"
        )


# --------------------------------------------------------------------------- #
# the run handle                                                               #
# --------------------------------------------------------------------------- #


class Run:
    """Handle on one recorded run. See module docstring."""

    def __init__(
        self,
        store: BridgeStore,
        run_id: str,
        *,
        spec: WrapSpec | None = None,
    ) -> None:
        self._store = store
        self.run_id = run_id
        self._spec = spec

    def __repr__(self) -> str:
        return f"Run({self.run_id!r})"

    # -- raw access ---------------------------------------------------------

    @property
    def store(self) -> BridgeStore:
        return self._store

    def raw_events(self) -> list[Event]:
        return self._store.load_events(self.run_id)

    @property
    def events(self) -> RunEvents:
        return RunEvents(self.raw_events())

    @property
    def record(self) -> RunRecord | None:
        return self._store.get_run(self.run_id)

    @property
    def label(self) -> str | None:
        rec = self.record
        return rec.label if rec else None

    @property
    def codec(self) -> EffectCodec:
        return self._spec.codec if self._spec is not None else AutoCodec()

    # -- attach (for runs loaded without a live spec) -------------------------

    def attach(
        self,
        agent_or_factory: Any,
        *,
        method: str = "invoke",
        adapter: Any = "auto",
        policy: Any = None,
        codec: EffectCodec | None = None,
        match: str = "auto",
    ) -> "Run":
        """Bind an agent (or factory) so verify/fork work on a loaded run.

        Runs created in this process by ``wrap()`` already carry their
        spec; ``attach`` is for runs loaded from disk in a later
        process, where only you know how to rebuild the agent.
        """
        from .wrapper import build_spec

        self._spec = build_spec(
            agent_or_factory,
            method=method,
            adapter=adapter,
            store=self._store,
            policy=policy,
            codec=codec,
            match=match,
        )
        return self

    def _require_spec(self, operation: str) -> WrapSpec:
        if self._spec is None:
            raise ReconstructionError(
                f"{operation} needs the agent, and this run handle has none attached",
                what_failed=(
                    f"{operation} was called on a run loaded from the store "
                    f"without an attached agent."
                ),
                why=(
                    "Replaying agent code requires the agent. The event log "
                    "holds every recorded effect, but only your code can "
                    "rebuild the thing that consumes them."
                ),
                how_to_fix=(
                    "Attach the same agent this run was recorded from:\n"
                    "    run = load_run(store, run_id).attach(build_agent)\n"
                    "then call verify()/fork() as usual."
                ),
            )
        return self._spec

    # -- the four operations ---------------------------------------------------

    def replay(self) -> Replay:
        """Rebuild the graph projection from the event log (native replay)."""
        events = self.raw_events()
        graph = Graph(ids=IDGen(), run_id=self.run_id)
        n = replay_into(graph, events)
        graph.ids.reseed_from_events(events)
        return Replay(
            graph=graph,
            events=n,
            objects=len(graph.all_objects()),
            relations=len(graph.all_relations()),
        )

    def graph(self) -> Graph:
        """The projected graph (shorthand for ``replay().graph``)."""
        return self.replay().graph

    def playback_output(self, *, invocation: int | None = None) -> Any:
        """The recorded output, decoded — no agent execution.

        ``invocation`` selects a specific invocation ordinal; default is
        the last completed one.
        """
        completed = [
            e for e in self.raw_events() if e.type == ev.INVOCATION_COMPLETED
        ]
        if invocation is not None:
            completed = [
                e for e in completed if int(e.payload.get("ordinal", 0)) == invocation
            ]
        if not completed:
            raise KeyError(
                f"run {self.run_id} has no completed invocation"
                + (f" with ordinal={invocation}" if invocation is not None else "")
            )
        return self.codec.decode_response(completed[-1].payload.get("output"))

    def verify(
        self,
        agent: Any = None,
        *,
        match: str | None = None,
        persist: bool = True,
    ) -> VerificationResult:
        """Shadow-execute a fresh agent against the recording.

        Zero model, tool, or network calls happen: every effect is served
        from the log and compared by canonical hash; unmediated I/O
        aborts the verification. On success (and ``persist=True``) a
        ``bridge.verification`` event is appended so the run's report
        shows ``boundary-verified`` from now on, in any process.
        """
        spec = self._require_spec("verify()")
        events = self.raw_events()
        if requires_async_drive(events):
            result = _run_async_operation(
                lambda: ashadow_verify(
                    spec, events, self.run_id, agent=agent, match=match
                ),
                async_api="await run.averify()",
            )
        else:
            result = shadow_verify(
                spec, events, self.run_id, agent=agent, match=match
            )
        self._persist_verification(result, persist=persist)
        return result

    async def averify(
        self,
        agent: Any = None,
        *,
        match: str | None = None,
        persist: bool = True,
    ) -> VerificationResult:
        """Awaitable verification for async code and async agent surfaces."""
        spec = self._require_spec("averify()")
        result = await ashadow_verify(
            spec, self.raw_events(), self.run_id, agent=agent, match=match
        )
        self._persist_verification(result, persist=persist)
        return result

    def fork(
        self,
        before: EventRef | str | None = None,
        overrides: dict[str, Any] | None = None,
        *,
        label: str | None = None,
        side_effects: str = "fail_closed",
        force: bool = False,
    ) -> "Fork":
        """Branch this run before a recorded boundary.

        ``before`` — an :class:`EventRef` from ``run.events`` selectors,
        a raw event id, or ``None`` for "before the first invocation".

        ``overrides`` — the change to apply at the fork point:
        ``{"response": ...}`` (alias ``"tool_result"``/``"result"``/
        ``"output"``) replaces the recorded response of the selected
        effect; ``{"input": ...}`` replaces the input of the selected
        invocation. Empty means pure divergence (changed code/config).

        ``side_effects`` — ``"fail_closed"`` (default) or ``"live"``
        (explicitly authorize the tail's writes).

        The fork copies the shared prefix into a child run (ActiveGraph's
        native fork lineage), and ``fork.execute()`` re-executes a clean
        agent: prefix effects are validated and served, the override is
        applied at the boundary, and the tail records live.
        """
        if side_effects not in ("fail_closed", "live"):
            raise BridgeConfigurationError(
                f"side_effects must be 'fail_closed' or 'live', got {side_effects!r}"
            )
        events = self.raw_events()
        anchor = self._resolve_anchor(before, events)
        idx = next(i for i, e in enumerate(events) if e.id == anchor.event_id)
        if idx == 0:
            raise NotForkableError(
                "cannot fork before the first event of a run",
                what_failed="The fork anchor is the run's very first event.",
                why="A fork copies a non-empty shared prefix; there is none here.",
                how_to_fix="Fork at the first invocation instead: run.fork() with no anchor.",
            )
        if not anchor.forkable and not force:
            raise NotForkableError(
                f"event {anchor.event_id} is not a safe fork boundary",
                what_failed=(
                    f"The selected event ({anchor.type}, {anchor.name or anchor.kind}) "
                    f"is not quiescent: other effects were in flight when it was "
                    f"recorded."
                ),
                why=(
                    "Forking mid-flight would cut the prefix between a request "
                    "and its response — the child would inherit a half-finished "
                    "effect no re-execution can honestly serve."
                ),
                how_to_fix=(
                    "Choose a quiescent boundary (run.events.fork_points() lists "
                    "them), or pass force=True if you accept the risk."
                ),
            )
        override = self._build_override(anchor, overrides or {})
        at_event = events[idx - 1].id
        prefix_events = events[:idx]
        prefix_hash = event_log_hash(prefix_events)
        parent_log_hash_at_fork = event_log_hash(events)
        source_fingerprint = _fingerprint_from_events(events)

        child_run_id = IDGen().run()
        self._store.fork_run(
            parent_run_id=self.run_id,
            new_run_id=child_run_id,
            at_event_id=at_event,
            label=label,
        )
        # Stamp the child with its fork configuration — auditability of
        # *what was changed* is half the value of a counterfactual.
        child_events = self._store.load_events(child_run_id)
        if event_log_hash(child_events[:idx]) != prefix_hash:
            raise BridgeConfigurationError(
                "the copied child prefix does not match the source prefix",
                what_failed="ActiveGraph fork_run returned a different prefix hash.",
                why="A fork receipt cannot be issued over a prefix that was not copied exactly.",
                how_to_fix="Inspect the event store fork implementation before executing this branch.",
            )
        ids = IDGen()
        ids.reseed_from_events(child_events)
        self._store.open_run(child_run_id).append(
            Event(
                id=ids.event(),
                type=ev.FORK_CONFIGURED,
                payload={
                    "parent_run_id": self.run_id,
                    "forked_before": anchor.event_id,
                    "at_event": at_event,
                    "override": {
                        "kind": anchor.kind,
                        "name": anchor.name,
                        "response": override.has_response,
                        "input": override.has_input,
                    }
                    if override is not None
                    else None,
                    "side_effects": side_effects,
                    "prefix_event_count": idx,
                    "prefix_hash": prefix_hash,
                    "parent_log_hash_at_fork": parent_log_hash_at_fork,
                },
                actor="bridge",
                timestamp=Clock().now(),
            )
        )
        return Fork(
            parent=self,
            child_run_id=child_run_id,
            anchor=anchor,
            override=override,
            side_effects=side_effects,
            label=label,
            prefix_event_count=idx,
            parent_event_count_at_fork=len(events),
            prefix_hash=prefix_hash,
            parent_log_hash_at_fork=parent_log_hash_at_fork,
            copied_through_event_id=at_event,
            source_fingerprint=source_fingerprint,
        )

    def diff(self, other: "Run | Fork") -> "RunDiff":
        """Structural comparison with a fork (or any other run)."""
        other_run = other.run if isinstance(other, Fork) else other
        return RunDiff.compute(self, other_run)

    # -- report ----------------------------------------------------------------

    @property
    def report(self) -> ReplayabilityReport:
        return compute_report(self.raw_events())

    @property
    def fork_receipts(self) -> list[ForkReceipt]:
        """Persisted fork receipts, oldest first."""
        return [
            ForkReceipt.from_dict(event.payload)
            for event in self.raw_events()
            if event.type == ev.FORK_RECEIPT
        ]

    def fork_points(self) -> list[EventRef]:
        return self.events.fork_points()

    # -- helpers -----------------------------------------------------------------

    def _resolve_anchor(
        self, before: EventRef | str | None, events: list[Event]
    ) -> EventRef:
        view = RunEvents(events)
        if before is None:
            return view.invocation(1)
        if isinstance(before, EventRef):
            return before
        if isinstance(before, str):
            e = view[before]  # raises KeyError if absent
            if e.type == ev.EFFECT_REQUESTED:
                refs = [r for r in view._effect_refs() if r.event_id == e.id]
                if refs:
                    return refs[0]
            return EventRef(
                event_id=e.id, type=e.type, ordinal=e.payload.get("ordinal")
            )
        raise BridgeConfigurationError(
            f"before= must be an EventRef, event id, or None; got {type(before).__name__}"
        )

    def _build_override(
        self, anchor: EventRef, overrides: dict[str, Any]
    ) -> ForkOverride | None:
        if not overrides:
            return None
        unknown = set(overrides) - {"response", "tool_result", "result", "output", "input"}
        if unknown:
            raise OverrideError(
                f"unknown override key(s): {sorted(unknown)}",
                what_failed=f"fork(overrides={{...}}) received {sorted(unknown)}.",
                why=(
                    "Overrides replace exactly one thing at the fork point: the "
                    "selected effect's response, or the selected invocation's input."
                ),
                how_to_fix=(
                    "Use {'response': value} (aliases: tool_result/result/output) "
                    "for effect anchors, or {'input': value} for invocation anchors."
                ),
            )
        response_keys = [k for k in ("response", "tool_result", "result", "output") if k in overrides]
        has_input = "input" in overrides
        if response_keys and has_input:
            raise OverrideError("a fork override replaces a response or an input, not both")
        if response_keys:
            if anchor.type != ev.EFFECT_REQUESTED:
                raise OverrideError(
                    f"response override requires an effect anchor, got {anchor.type}",
                    what_failed=(
                        f"fork(before={anchor.event_id}, overrides={{'{response_keys[0]}': ...}}) "
                        f"anchored at a {anchor.type} event."
                    ),
                    why="Only a recorded effect has a response to replace.",
                    how_to_fix=(
                        "Anchor at the effect whose result you want to change, e.g.\n"
                        "    run.fork(before=run.events.tool_call('lookup_order', occurrence=1), ...)"
                    ),
                )
            if len(response_keys) > 1:
                raise OverrideError(f"multiple response override keys: {response_keys}")
            return ForkOverride(
                target_event_id=anchor.event_id,
                kind=anchor.kind,
                name=anchor.name,
                request_hash=anchor.request_hash,
                footprint=anchor.footprint,  # type: ignore[arg-type]
                replay_source=anchor.replay_source,  # type: ignore[arg-type]
                observables=anchor.observables,
                response=overrides[response_keys[0]],
                has_response=True,
            )
        if anchor.type != ev.INVOCATION_STARTED:
            raise OverrideError(
                f"input override requires an invocation anchor, got {anchor.type}",
                what_failed=(
                    f"fork(before={anchor.event_id}, overrides={{'input': ...}}) "
                    f"anchored at a {anchor.type} event."
                ),
                why="Only an invocation boundary has an input to replace.",
                how_to_fix=(
                    "Anchor at the invocation:\n"
                    "    run.fork(before=run.events.invocation(1), overrides={'input': ...})"
                ),
            )
        return ForkOverride(
            target_event_id=anchor.event_id,
            kind="",
            name="",
            request_hash="",
            input=overrides["input"],
            has_input=True,
        )

    def _append_meta(self, type_: str, payload: dict[str, Any]) -> None:
        events = self.raw_events()
        ids = IDGen()
        ids.reseed_from_events(events)
        self._store.open_run(self.run_id).append(
            Event(
                id=ids.event(),
                type=type_,
                payload=payload,
                actor="bridge",
                timestamp=Clock().now(),
            )
        )

    def _persist_verification(
        self, result: VerificationResult, *, persist: bool
    ) -> None:
        if not persist:
            return
        self._append_meta(
            ev.VERIFICATION_RECORDED,
            {
                "ok": result.ok,
                "effects_served": result.effects_served,
                "invocations": result.invocations,
                "reordered": result.reordered,
                "divergence": (
                    str(result.divergence).splitlines()[0]
                    if result.divergence
                    else None
                ),
            },
        )


# --------------------------------------------------------------------------- #
# forks                                                                        #
# --------------------------------------------------------------------------- #


class Fork:
    """A configured, not-yet-executed branch of a run."""

    def __init__(
        self,
        *,
        parent: Run,
        child_run_id: str,
        anchor: EventRef,
        override: ForkOverride | None,
        side_effects: str,
        label: str | None,
        prefix_event_count: int,
        parent_event_count_at_fork: int,
        prefix_hash: str,
        parent_log_hash_at_fork: str,
        copied_through_event_id: str,
        source_fingerprint: dict[str, Any],
    ) -> None:
        self.parent = parent
        self.child_run_id = child_run_id
        self.anchor = anchor
        self.override = override
        self.side_effects = side_effects
        self.label = label
        self.prefix_event_count = prefix_event_count
        self.parent_event_count_at_fork = parent_event_count_at_fork
        self.prefix_hash = prefix_hash
        self.parent_log_hash_at_fork = parent_log_hash_at_fork
        self.copied_through_event_id = copied_through_event_id
        self.source_fingerprint = source_fingerprint
        self.executed = False
        self.output: Any = None
        self.receipt: ForkReceipt | None = None

    def __repr__(self) -> str:
        return f"Fork({self.child_run_id!r}, before={self.anchor.event_id!r})"

    @property
    def run(self) -> Run:
        """The child run handle (usable before and after execute())."""
        return Run(self.parent.store, self.child_run_id, spec=self.parent._spec)

    def environment_claims(
        self, target_fingerprint: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Claims an environment attestation must bind before execution."""
        if target_fingerprint is None:
            target_fingerprint = self.parent._require_spec(
                "fork.environment_claims()"
            ).fingerprint()
        return {
            "parent_run_id": self.parent.run_id,
            "child_run_id": self.child_run_id,
            "prefix_hash": self.prefix_hash,
            "forked_before_event_id": self.anchor.event_id,
            "target_fingerprint_hash": content_hash(target_fingerprint),
        }

    def execute(
        self,
        agent: Any = None,
        *,
        match: str | None = None,
        target_environment: EnvironmentAttestation | None = None,
        environment_verifier: EnvironmentVerifier | None = None,
    ) -> Any:
        """Re-execute to the fork point on recorded effects, apply the
        override, and record the divergent tail. Returns the (last)
        invocation's output."""
        self._ensure_not_executed("Fork.execute()")
        spec = self.parent._require_spec("fork.execute()")
        target_fingerprint, environment_verified, verifier_id = (
            self._verify_environment(
                spec,
                target_environment=target_environment,
                environment_verifier=environment_verifier,
            )
        )
        parent_events = self.parent.raw_events()
        try:
            if requires_async_drive(parent_events):
                output, session = _run_async_operation(
                    lambda: afork_execute(
                        spec,
                        child_run_id=self.child_run_id,
                        parent_events=parent_events,
                        override=self.override,
                        side_effects=self.side_effects,
                        agent=agent,
                        match=match,
                    ),
                    async_api="await fork.aexecute()",
                )
            else:
                output, session = fork_execute(
                    spec,
                    child_run_id=self.child_run_id,
                    parent_events=parent_events,
                    override=self.override,
                    side_effects=self.side_effects,
                    agent=agent,
                    match=match,
                )
        except BaseException:
            self.executed = True
            raise
        self.receipt = self._persist_receipt(
            session,
            parent_events=parent_events,
            target_fingerprint=target_fingerprint,
            target_environment=target_environment,
            environment_verified=environment_verified,
            verifier_id=verifier_id,
        )
        self.executed = True
        self.output = output
        return output

    async def aexecute(
        self,
        agent: Any = None,
        *,
        match: str | None = None,
        target_environment: EnvironmentAttestation | None = None,
        environment_verifier: EnvironmentVerifier | None = None,
    ) -> Any:
        """Awaitable fork execution for async code and agent surfaces."""
        self._ensure_not_executed("Fork.aexecute()")
        spec = self.parent._require_spec("fork.aexecute()")
        target_fingerprint, environment_verified, verifier_id = (
            self._verify_environment(
                spec,
                target_environment=target_environment,
                environment_verifier=environment_verifier,
            )
        )
        parent_events = self.parent.raw_events()
        try:
            output, session = await afork_execute(
                spec,
                child_run_id=self.child_run_id,
                parent_events=parent_events,
                override=self.override,
                side_effects=self.side_effects,
                agent=agent,
                match=match,
            )
        except BaseException:
            self.executed = True
            raise
        self.receipt = self._persist_receipt(
            session,
            parent_events=parent_events,
            target_fingerprint=target_fingerprint,
            target_environment=target_environment,
            environment_verified=environment_verified,
            verifier_id=verifier_id,
        )
        self.executed = True
        self.output = output
        return output

    def _verify_environment(
        self,
        spec: WrapSpec,
        *,
        target_environment: EnvironmentAttestation | None,
        environment_verifier: EnvironmentVerifier | None,
    ) -> tuple[dict[str, Any], bool, str | None]:
        target_fingerprint = spec.fingerprint()
        if target_environment is None:
            if environment_verifier is not None:
                raise BridgeConfigurationError(
                    "an environment verifier was supplied without an attestation"
                )
            return target_fingerprint, False, None
        if environment_verifier is None:
            raise BridgeConfigurationError(
                "target_environment requires an environment_verifier"
            )
        if not environment_verifier.verify(target_environment):
            raise BridgeConfigurationError(
                "target-environment attestation signature is invalid"
            )
        required = self.environment_claims(target_fingerprint)
        missing = {
            key: value
            for key, value in required.items()
            if target_environment.claims.get(key) != value
        }
        if missing:
            raise BridgeConfigurationError(
                "target-environment attestation does not bind this fork",
                what_failed=f"Attested claims differ for {sorted(missing)}.",
                why="A reusable or stale environment receipt cannot discharge this fork's premise.",
                how_to_fix="Issue a fresh attestation from fork.environment_claims().",
            )
        return target_fingerprint, True, environment_verifier.verifier_id

    def _persist_receipt(
        self,
        session: Any,
        *,
        parent_events: list[Event],
        target_fingerprint: dict[str, Any],
        target_environment: EnvironmentAttestation | None,
        environment_verified: bool,
        verifier_id: str | None,
    ) -> ForkReceipt:
        prefix = parent_events[: self.prefix_event_count]
        inherited = effect_evidence(prefix)
        inherited_ids = [item["request_event_id"] for item in inherited]
        served_ids = list(session.served_effect_request_ids)
        prefix_external_calls = int(session.prefix_external_calls)
        zero_reexecution = (
            prefix_external_calls == 0
            and sorted(served_ids) == sorted(inherited_ids)
        )
        child_before_receipt = self.run.raw_events()
        receipt = ForkReceipt(
            parent_run_id=self.parent.run_id,
            child_run_id=self.child_run_id,
            forked_before_event_id=self.anchor.event_id,
            copied_through_event_id=self.copied_through_event_id,
            parent_event_count_at_fork=self.parent_event_count_at_fork,
            prefix_event_count=self.prefix_event_count,
            parent_log_hash_at_fork=self.parent_log_hash_at_fork,
            prefix_hash=self.prefix_hash,
            child_log_hash_before_receipt=event_log_hash(child_before_receipt),
            source_fingerprint=self.source_fingerprint,
            target_fingerprint=target_fingerprint,
            target_environment=(
                target_environment.to_dict() if target_environment is not None else None
            ),
            environment_verified=environment_verified,
            environment_verifier=verifier_id,
            inherited_effects=inherited,
            served_effect_request_ids=served_ids,
            prefix_external_calls=prefix_external_calls,
            tail_executed_effect_request_ids=list(
                session.executed_effect_request_ids
            ),
            zero_reexecution_verified=zero_reexecution,
            external_continuation=(
                "verified"
                if zero_reexecution and environment_verified
                else "conditional"
            ),
            status="completed",
        )
        receipt = ForkReceipt.from_dict(receipt.to_dict())
        self.run._append_meta(ev.FORK_RECEIPT, receipt.to_dict())
        return receipt

    def _ensure_not_executed(self, operation: str) -> None:
        if not self.executed:
            return
        raise BridgeConfigurationError(
            "this fork has already executed; fork the parent again for a new branch",
            what_failed=f"{operation} was called after this fork already executed.",
            why=(
                "A fork's child log already contains its divergent tail after "
                "the first execution; running again would append a second, "
                "interleaved tail and corrupt the branch."
            ),
            how_to_fix="Call run.fork(...) again to create a fresh branch.",
        )

    def diff(self) -> "RunDiff":
        return self.parent.diff(self)


# --------------------------------------------------------------------------- #
# diff                                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class RunDiff:
    """Effect-level and graph-level comparison of two runs.

    ``shared_effects`` is the length of the identical effect prefix
    (kind, name, request hash, response hash). ``graph`` is ActiveGraph's
    native structural :class:`~activegraph.runtime.diff.Diff` over the
    projected graphs — bridge runs are ordinary graphs, so the native
    diff applies unmodified.
    """

    parent_run_id: str
    child_run_id: str
    shared_effects: int
    parent_tail: list[EventRef]
    child_tail: list[EventRef]
    parent_output: Any
    child_output: Any
    outputs_differ: bool
    graph: Diff

    @classmethod
    def compute(cls, parent: Run, child: Run) -> "RunDiff":
        p_refs = parent.events._effect_refs()
        c_refs = child.events._effect_refs()
        p_events = {e.id: e for e in parent.raw_events()}
        c_events = {e.id: e for e in child.raw_events()}

        def signature(ref: EventRef, events: dict[str, Event]) -> tuple:
            responses = [
                e
                for e in events.values()
                if e.caused_by == ref.event_id
                and e.type in (ev.EFFECT_RESPONDED, ev.EFFECT_FAILED)
            ]
            response_hash = responses[0].payload.get("response_hash") if responses else None
            return (ref.kind, ref.name, ref.request_hash, response_hash)

        shared = 0
        for p, c in zip(p_refs, c_refs):
            if signature(p, p_events) == signature(c, c_events):
                shared += 1
            else:
                break

        def out(run: Run) -> Any:
            try:
                return run.playback_output()
            except KeyError:
                return None

        p_out, c_out = out(parent), out(child)
        parent_graph = parent.graph()
        child_graph = child.graph()
        return cls(
            parent_run_id=parent.run_id,
            child_run_id=child.run_id,
            shared_effects=shared,
            parent_tail=p_refs[shared:],
            child_tail=c_refs[shared:],
            parent_output=p_out,
            child_output=c_out,
            outputs_differ=_encoded(parent, p_out) != _encoded(child, c_out),
            graph=compute_diff(parent_graph, child_graph, parent.run_id, child.run_id),
        )

    def __str__(self) -> str:
        lines = [
            f"Diff {self.parent_run_id} vs {self.child_run_id}",
            f"Shared effect prefix: {self.shared_effects}",
            f"Parent tail: {len(self.parent_tail)} effect(s)"
            + (f" — {', '.join(str(r) for r in self.parent_tail[:4])}" if self.parent_tail else ""),
            f"Child tail:  {len(self.child_tail)} effect(s)"
            + (f" — {', '.join(str(r) for r in self.child_tail[:4])}" if self.child_tail else ""),
            f"Outputs differ: {'yes' if self.outputs_differ else 'no'}",
            f"Divergent graph objects: {len(self.graph.divergent_objects)}",
        ]
        return "\n".join(lines)


def _encoded(run: Run, value: Any) -> Any:
    try:
        return run.codec.encode_response(value)
    except Exception:
        return repr(value)


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #


def load_run(store: str | BridgeStore, run_id: str | None = None) -> Run:
    """Open a recorded run by id (default: the most recent in the store).

    The returned handle can inspect, replay, and play back immediately;
    ``.attach(build_agent)`` enables verify and fork.
    """
    resolved = resolve_store(store)
    chosen = run_id or resolved.most_recent_run_id()
    if chosen is None:
        raise FileNotFoundError(f"no runs found in {resolved.url}")
    return Run(resolved, chosen)


def list_runs(store: str | BridgeStore) -> list[RunRecord]:
    """All runs in a store, with fork lineage (parent, fork point, label)."""
    return resolve_store(store).list_runs()
