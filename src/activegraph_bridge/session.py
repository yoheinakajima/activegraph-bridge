"""The execution session: the membrane between agent code and the world.

An :class:`ExecutionSession` is a context-local object (propagated
through ``contextvars``, so it survives ``await`` and nests naturally)
that connects everything the wrapped agent does to the active
ActiveGraph run. Agent code never touches it directly in the common
case — bridged tools, instrumented clients, and deterministic sources
find it via :func:`current_session`.

The session runs in one of four modes:

==========  ================================================================
``record``  Live execution. Every mediated call executes for real (writes
            per policy) and lands in the event log as an
            ``effect.requested`` / ``effect.responded`` pair.
``shadow``  Verification. Nothing executes; every call is served from the
            recorded script and compared against it. Divergence raises.
``prefix``  A fork replaying its inherited prefix. Identical to shadow,
            except running out of script is not an error — it is the fork
            point, where the session transitions to ``tail``.
``tail``    A fork past its fork point: live recording again, but with the
            fork's fail-closed write policy.
==========  ================================================================

The mode table is the whole trick: agent code is identical in all four;
only the membrane's answers change.
"""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from activegraph.core.event import Event
from activegraph.core.graph import Graph

from . import events as ev
from ._canonical import content_hash
from .codecs import (
    AutoCodec,
    EffectCodec,
    decode_exception,
    encode_exception,
    find_lossy,
)
from .errors import ReplayDivergence, UnrecordedEffectError
from .policy import (
    Footprint,
    ReplaySource,
    SideEffectPolicy,
    derive_footprint,
)
from .report import Finding
from .script import Cursor

__all__ = [
    "SessionMode",
    "ExecutionSession",
    "current_session",
    "effect",
    "aeffect",
    "checkpoint",
]

_ACTIVE: ContextVar["ExecutionSession | None"] = ContextVar(
    "activegraph_bridge_session", default=None
)
# Depth counter for I/O the bridge itself is performing on purpose (a live
# effect execution). Context-local so concurrent tasks don't mask each other.
_MEDIATED_DEPTH: ContextVar[int] = ContextVar(
    "activegraph_bridge_mediated_depth", default=0
)


def current_session() -> "ExecutionSession | None":
    """The session active in this context, if any.

    Returns ``None`` outside any wrapped execution — bridge primitives
    (tools, instrumented clients, deterministic sources) treat that as
    "run live, unrecorded" so instrumented code keeps working without
    the bridge.
    """
    return _ACTIVE.get()


class SessionMode(str, Enum):
    RECORD = "record"
    SHADOW = "shadow"
    PREFIX = "prefix"
    TAIL = "tail"


@dataclass
class ForkOverride:
    """The change a fork applies at its fork point."""

    target_event_id: str
    kind: str
    name: str
    request_hash: str
    footprint: Footprint = "idempotent"
    replay_source: ReplaySource = "recorded"
    observables: tuple[str, ...] = ()
    response: Any = None
    has_response: bool = False
    input: Any = None
    has_input: bool = False
    applied: bool = False


@dataclass
class _Invocation:
    ordinal: int
    method: str
    start_event_id: str = ""
    started_at: float = 0.0


class ExecutionSession:
    """Mediates one run's worth of agent execution. See module docstring."""

    def __init__(
        self,
        *,
        mode: SessionMode,
        graph: Graph | None = None,
        cursor: Cursor | None = None,
        policy: SideEffectPolicy | None = None,
        codec: EffectCodec | None = None,
        projector: Any = None,
        override: ForkOverride | None = None,
        run_id: str = "",
    ) -> None:
        if mode in (SessionMode.RECORD, SessionMode.TAIL) and graph is None:
            raise ValueError(f"{mode.value} mode requires a graph")
        if mode in (SessionMode.SHADOW, SessionMode.PREFIX) and cursor is None:
            raise ValueError(f"{mode.value} mode requires a cursor")
        self.mode = mode
        self.graph = graph
        self.cursor = cursor
        self.policy = policy or SideEffectPolicy()
        self.codec: EffectCodec = codec or AutoCodec()
        self.projector = projector
        self.override = override
        self.run_id = run_id or (graph.run_id if graph is not None else "")

        self.findings: list[Finding] = []
        self.effects_served = 0
        self.live_calls = 0
        self.prefix_external_calls = 0
        self.served_effect_request_ids: list[str] = []
        self.executed_effect_request_ids: list[str] = []
        # One agent instance per wrapper per session: nested wrapped agents
        # are rebuilt fresh for every shadow/fork session, exactly like the
        # top-level agent. Keyed by id(wrapper).
        self.agent_cache: dict[int, Any] = {}
        self._lock = threading.RLock()
        self._effect_ordinal = 0
        self._checkpoint_ordinal = 0
        self._invocation_ordinal = 0
        self._invocation_stack: list[_Invocation] = []
        self._in_flight = 0
        self._parent_events: list[str] = []  # causal chain for caused_by
        self._token: Token | None = None
        self._finalized = False

    # ------------------------------------------------------------------ #
    # activation                                                          #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "ExecutionSession":
        if _ACTIVE.get() is not None:
            raise RuntimeError(
                "an ExecutionSession is already active in this context; "
                "sessions do not nest (nested wrapped agents share the "
                "outer session automatically)"
            )
        self._token = _ACTIVE.set(self)
        from .determinism import ensure_guard_installed

        ensure_guard_installed()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None

    # ------------------------------------------------------------------ #
    # emission (record / tail)                                            #
    # ------------------------------------------------------------------ #

    def _emit(
        self, type_: str, payload: dict[str, Any], *, caused_by: str | None = None
    ) -> Event:
        assert self.graph is not None
        with self._lock:
            event = Event(
                id=self.graph.ids.event(),
                type=type_,
                payload=payload,
                actor="bridge",
                caused_by=caused_by,
                timestamp=self.graph.clock.now(),
            )
            self.graph.emit(event)
            if self.projector is not None:
                self.projector.project(self.graph, event)
            return event

    @property
    def _causal_parent(self) -> str | None:
        if self._parent_events:
            return self._parent_events[-1]
        return None

    @property
    def is_serving(self) -> bool:
        return self.mode in (SessionMode.SHADOW, SessionMode.PREFIX)

    @property
    def fork_tail(self) -> bool:
        return self.mode is SessionMode.TAIL

    # ------------------------------------------------------------------ #
    # invocations                                                         #
    # ------------------------------------------------------------------ #

    def begin_invocation(
        self,
        *,
        method: str,
        input_value: Any = None,
        encoded_input: Any = None,
        input_hash: str | None = None,
    ) -> _Invocation:
        """Open an invocation boundary (top-level from the engine, nested
        from a wrapped agent invoked inside another's session).

        The engine re-driving a recording passes ``encoded_input`` /
        ``input_hash`` straight from the log so validation never depends
        on encode/decode round-trips; live calls pass ``input_value``.
        """
        if encoded_input is None:
            encoded_input = self.codec.encode_request(input_value)
        encoded = encoded_input
        if input_hash is None:
            input_hash = content_hash(encoded)
        with self._lock:
            if self.is_serving:
                assert self.cursor is not None
                if self.mode is SessionMode.PREFIX and self.cursor.exhausted:
                    self._become_tail()
                else:
                    entry = self.cursor.take_invocation(
                        boundary="start", input_hash=input_hash
                    )
                    if entry is None:  # shadow ran past the recording
                        raise ReplayDivergence(
                            "re-execution started an invocation past the end of the recording",
                            got={"method": method, "input_hash": input_hash},
                        )
                    inv = _Invocation(
                        ordinal=entry.ordinal, method=method, started_at=time.monotonic()
                    )
                    inv.start_event_id = entry.start_event_id
                    self._invocation_stack.append(inv)
                    if entry.start_event_id:
                        self._parent_events.append(entry.start_event_id)
                    return inv
            # record / tail: emit
            self._invocation_ordinal += 1
            inv = _Invocation(
                ordinal=self._invocation_ordinal,
                method=method,
                started_at=time.monotonic(),
            )
            event = self._emit(
                ev.INVOCATION_STARTED,
                {
                    "ordinal": inv.ordinal,
                    "method": method,
                    "input": encoded,
                    "input_hash": input_hash,
                    "depth": len(self._invocation_stack),
                },
                caused_by=self._causal_parent,
            )
            inv.start_event_id = event.id
            self._invocation_stack.append(inv)
            self._parent_events.append(event.id)
            return inv

    def finish_invocation(
        self,
        inv: _Invocation,
        *,
        output: Any = None,
        error: BaseException | None = None,
    ) -> None:
        latency = time.monotonic() - inv.started_at
        with self._lock:
            if self._invocation_stack and self._invocation_stack[-1] is inv:
                self._invocation_stack.pop()
            if self._parent_events and self._parent_events[-1] == inv.start_event_id:
                self._parent_events.pop()
            if self.is_serving:
                assert self.cursor is not None
                if self.mode is SessionMode.PREFIX and self.cursor.exhausted:
                    self._become_tail()
                else:
                    entry = self.cursor.take_invocation(boundary="end")
                    if entry is None:
                        raise ReplayDivergence(
                            "re-execution finished an invocation past the end of the recording",
                            got={"ordinal": inv.ordinal},
                        )
                    if error is None and entry.error is None:
                        encoded = self.codec.encode_response(output)
                        got_hash = content_hash(encoded)
                        if entry.output_hash and got_hash != entry.output_hash:
                            raise ReplayDivergence(
                                "invocation output does not match the recording",
                                expected={
                                    "output_hash": entry.output_hash,
                                    "output": entry.output,
                                },
                                got={"output_hash": got_hash, "output": encoded},
                                what_failed=(
                                    f"Invocation {inv.ordinal} re-executed cleanly but "
                                    f"produced a different final output."
                                ),
                                why=(
                                    "Every recorded effect matched, so the divergence "
                                    "comes from computation the membrane did not see — "
                                    "typically direct time/random/uuid reads or mutable "
                                    "state carried across runs."
                                ),
                                how_to_fix=(
                                    "Route nondeterminism through activegraph_bridge.det "
                                    "(now/random/uuid4), or record the source as an "
                                    "effect. The divergent fields in got.output point at "
                                    "what changed."
                                ),
                            )
                    elif (error is None) != (entry.error is None):
                        raise ReplayDivergence(
                            "invocation outcome (success/failure) differs from the recording",
                            expected={"error": entry.error},
                            got={"error": str(error) if error else None},
                        )
                    return
            # record / tail
            if not self._finalizable():
                return
            if error is not None:
                self._emit(
                    ev.INVOCATION_FAILED,
                    {
                        "ordinal": inv.ordinal,
                        "error": encode_exception(error),
                        "latency_seconds": round(latency, 6),
                    },
                    caused_by=inv.start_event_id or None,
                )
                return
            encoded = self.codec.encode_response(output)
            lossy = find_lossy(encoded)
            for note in lossy:
                self.findings.append(Finding("blocker", "lossy-envelope", note))
            self._emit(
                ev.INVOCATION_COMPLETED,
                {
                    "ordinal": inv.ordinal,
                    "output": encoded,
                    "output_hash": content_hash(encoded),
                    "latency_seconds": round(latency, 6),
                    "lossy": lossy,
                },
                caused_by=inv.start_event_id or None,
            )

    def _finalizable(self) -> bool:
        return self.graph is not None and not self._finalized

    # ------------------------------------------------------------------ #
    # the effect broker                                                   #
    # ------------------------------------------------------------------ #

    def effect(
        self,
        kind: str,
        request: Any,
        execute: Callable[[], Any],
        *,
        name: str = "",
        side_effect: str = "unknown",
        footprint: Footprint | None = None,
        replay_source: ReplaySource = "recorded",
        observables: tuple[str, ...] = (),
        codec: EffectCodec | None = None,
        category: str | None = None,
    ) -> Any:
        """Mediate one call: the recorded-effect broker.

        In ``record``/``tail`` mode: canonicalize and hash the request,
        append ``effect.requested``, execute (subject to the write
        policy), append ``effect.responded``/``effect.failed``, and
        return the live value — rehydration is only ever needed on
        replay, so the agent always sees exactly what the SDK returned.

        In ``shadow``/``prefix`` mode: nothing executes. The recorded
        response is served (decoded by the codec), the recorded failure
        is re-raised, and any mismatch with the recording raises
        :class:`ReplayDivergence`. A ``prefix`` cursor running dry is the
        fork point: the session transitions to ``tail`` and this same
        call becomes the first divergent (live or overridden) effect.
        """
        codec = codec or self.codec
        encoded_request = codec.encode_request(request)
        request_hash = content_hash(encoded_request)
        category = category or _categorize(kind)
        footprint = footprint or derive_footprint(side_effect)  # type: ignore[arg-type]

        if self.is_serving:
            served = self._serve(
                kind=kind,
                name=name,
                request_hash=request_hash,
                encoded_request=encoded_request,
                codec=codec,
            )
            if served is not _FALL_THROUGH:
                return served
        return self._record(
            kind=kind,
            name=name,
            category=category,
            side_effect=side_effect,
            footprint=footprint,
            replay_source=replay_source,
            observables=observables,
            request=request,
            encoded_request=encoded_request,
            request_hash=request_hash,
            codec=codec,
            execute=execute,
            is_async=False,
        )

    async def aeffect(
        self,
        kind: str,
        request: Any,
        execute: Callable[[], Awaitable[Any]],
        *,
        name: str = "",
        side_effect: str = "unknown",
        footprint: Footprint | None = None,
        replay_source: ReplaySource = "recorded",
        observables: tuple[str, ...] = (),
        codec: EffectCodec | None = None,
        category: str | None = None,
    ) -> Any:
        """Async twin of :meth:`effect` (awaits ``execute()``)."""
        codec = codec or self.codec
        encoded_request = codec.encode_request(request)
        request_hash = content_hash(encoded_request)
        category = category or _categorize(kind)
        footprint = footprint or derive_footprint(side_effect)  # type: ignore[arg-type]

        if self.is_serving:
            served = self._serve(
                kind=kind,
                name=name,
                request_hash=request_hash,
                encoded_request=encoded_request,
                codec=codec,
            )
            if served is not _FALL_THROUGH:
                return served
        return await self._record(
            kind=kind,
            name=name,
            category=category,
            side_effect=side_effect,
            footprint=footprint,
            replay_source=replay_source,
            observables=observables,
            request=request,
            encoded_request=encoded_request,
            request_hash=request_hash,
            codec=codec,
            execute=execute,
            is_async=True,
        )

    # -- serving (shadow / prefix) ----------------------------------------

    def _serve(
        self,
        *,
        kind: str,
        name: str,
        request_hash: str,
        encoded_request: Any,
        codec: EffectCodec,
    ) -> Any:
        with self._lock:
            assert self.cursor is not None
            entry = self.cursor.take_effect(
                kind=kind, name=name, request_hash=request_hash
            )
            if entry is None:
                if self.mode is SessionMode.SHADOW:
                    raise ReplayDivergence(
                        f"re-execution requested {name or kind!r} beyond the end of the recording",
                        got={"kind": kind, "name": name, "request_hash": request_hash},
                        what_failed=(
                            f"The recording is fully consumed, but the agent "
                            f"requested one more effect: {name or kind!r}."
                        ),
                        why=(
                            "verify() proves the recording is complete; an extra "
                            "call means the re-executed path diverged."
                        ),
                        how_to_fix=(
                            "Check for nondeterministic control flow, or fork "
                            "instead of verifying if this divergence is intended."
                        ),
                    )
                # prefix exhausted → this call *is* the fork point
                self._become_tail()
                override = self.override
                if override is not None and not override.applied and override.has_response:
                    return self._apply_override(
                        override,
                        kind=kind,
                        name=name,
                        request_hash=request_hash,
                        encoded_request=encoded_request,
                        codec=codec,
                    )
                return _FALL_THROUGH  # first live call of the tail
            self.effects_served += 1
            self.served_effect_request_ids.append(entry.request_event_id)
            if entry.failed:
                raise decode_exception(entry.error)
            return codec.decode_response(entry.response)

    def _become_tail(self) -> None:
        """Fork point reached: from here the child records live."""
        if self.mode is SessionMode.TAIL:
            return
        if self.graph is None:
            raise RuntimeError(
                "prefix cursor exhausted but no graph is attached for the tail"
            )
        self.mode = SessionMode.TAIL
        # Causal chain and ordinals continue from what the prefix recorded.
        self._invocation_ordinal = max(
            (int(e.payload.get("ordinal", 0)) for e in self.graph.events
             if e.type == ev.INVOCATION_STARTED),
            default=0,
        )
        self._effect_ordinal = max(
            (int(e.payload.get("ordinal", 0)) for e in self.graph.events
             if e.type == ev.EFFECT_REQUESTED),
            default=0,
        )
        self._checkpoint_ordinal = max(
            (int(e.payload.get("ordinal", 0)) for e in self.graph.events
             if e.type == ev.CHECKPOINT_RECORDED),
            default=0,
        )
        # Invocations opened while serving the prefix are now live frames;
        # finish_invocation emits their completions into the tail because
        # the mode check happens at finish time, not open time.

    def _apply_override(
        self,
        override: ForkOverride,
        *,
        kind: str,
        name: str,
        request_hash: str,
        encoded_request: Any,
        codec: EffectCodec,
    ) -> Any:
        from .errors import OverrideError

        if (kind, name) != (override.kind, override.name):
            raise OverrideError(
                f"fork override targets {override.name or override.kind!r} but the "
                f"first divergent call is {name or kind!r}",
                what_failed=(
                    f"The fork was created before event {override.target_event_id} "
                    f"({override.name or override.kind!r}), but when re-execution "
                    f"reached the fork point the agent requested "
                    f"{name or kind!r} instead."
                ),
                why=(
                    "An override replaces the response of the exact call it names. "
                    "If the re-executed agent asks for something else first, the "
                    "prefix was not deterministic and the override would land on "
                    "the wrong call."
                ),
                how_to_fix=(
                    "Re-run run.verify() to locate the nondeterminism, or fork "
                    "without an override and steer the divergence in agent code."
                ),
            )
        if request_hash != override.request_hash:
            self.findings.append(
                Finding(
                    "note",
                    "override-request-drift",
                    f"fork-point request hash {request_hash[:12]}… differs from the "
                    f"recorded {override.request_hash[:12]}… (agent code or config "
                    f"changed since the recording); override applied anyway",
                )
            )
        override.applied = True
        encoded_response = codec.encode_response(override.response)
        req_event = self._emit(
            ev.EFFECT_REQUESTED,
            self._request_payload(
                kind=kind,
                name=name,
                category=_categorize(kind),
                side_effect="read",
                footprint=override.footprint,
                replay_source=override.replay_source,
                observables=override.observables,
                encoded_request=encoded_request,
                request_hash=request_hash,
                codec_name=getattr(codec, "name", "auto"),
            ),
            caused_by=self._causal_parent,
        )
        self._emit(
            ev.EFFECT_RESPONDED,
            {
                "kind": kind,
                "name": name,
                "request_hash": request_hash,
                "response": encoded_response,
                "response_hash": content_hash(encoded_response),
                "served_from": "override",
                "latency_seconds": 0.0,
                "ordinal": req_event.payload.get("ordinal"),
                "lossy": find_lossy(encoded_response),
                "lifecycle": "committed",
            },
            caused_by=req_event.id,
        )
        return codec.decode_response(encoded_response)

    # -- recording (record / tail) -----------------------------------------

    def _request_payload(
        self,
        *,
        kind: str,
        name: str,
        category: str,
        side_effect: str,
        footprint: Footprint,
        replay_source: ReplaySource,
        observables: tuple[str, ...],
        encoded_request: Any,
        request_hash: str,
        codec_name: str,
    ) -> dict[str, Any]:
        self._effect_ordinal += 1
        return {
            "kind": kind,
            "name": name,
            "category": category,
            "side_effect": side_effect,
            "footprint": footprint,
            "replay_source": replay_source,
            "observables": sorted(set(observables)),
            "lifecycle": "requested",
            "request": encoded_request,
            "request_hash": request_hash,
            "ordinal": self._effect_ordinal,
            "quiescent": self._in_flight == 0,
            "lane": threading.current_thread().name,
            "codec": codec_name,
            "invocation": (
                self._invocation_stack[-1].ordinal if self._invocation_stack else None
            ),
        }

    def _record(
        self,
        *,
        kind: str,
        name: str,
        category: str,
        side_effect: str,
        footprint: Footprint,
        replay_source: ReplaySource,
        observables: tuple[str, ...],
        request: Any,
        encoded_request: Any,
        request_hash: str,
        codec: EffectCodec,
        execute: Callable[[], Any],
        is_async: bool,
    ) -> Any:
        decision = self.policy.decide(
            side_effect=side_effect,  # type: ignore[arg-type]
            footprint=footprint,
            fork_tail=self.fork_tail,
        )
        mode_name = self.mode.value
        with self._lock:
            payload = self._request_payload(
                kind=kind,
                name=name,
                category=category,
                side_effect=side_effect,
                footprint=footprint,
                replay_source=replay_source,
                observables=observables,
                encoded_request=encoded_request,
                request_hash=request_hash,
                codec_name=getattr(codec, "name", "auto"),
            )
            for note in find_lossy(encoded_request):
                self.findings.append(Finding("warning", "lossy-capture", note))
            req_event = self._emit(
                ev.EFFECT_REQUESTED, payload, caused_by=self._causal_parent
            )
            self._parent_events.append(req_event.id)
            self._in_flight += 1

        def finish_ok(value: Any, served_from: str, started: float) -> Any:
            encoded_response = codec.encode_response(value)
            lossy = find_lossy(encoded_response)
            with self._lock:
                self._in_flight -= 1
                if self._parent_events and self._parent_events[-1] == req_event.id:
                    self._parent_events.pop()
                for note in lossy:
                    self.findings.append(Finding("warning", "lossy-capture", note))
                self.live_calls += 1 if served_from == "live" else 0
                self._emit(
                    ev.EFFECT_RESPONDED,
                    {
                        "kind": kind,
                        "name": name,
                        "request_hash": request_hash,
                        "response": encoded_response,
                        "response_hash": content_hash(encoded_response),
                        "served_from": served_from,
                        "approved_by": (
                            self.policy.approved_by if served_from == "approved" else None
                        ),
                        "latency_seconds": round(time.monotonic() - started, 6),
                        "ordinal": payload["ordinal"],
                        "lossy": lossy,
                        "lifecycle": "committed",
                    },
                    caused_by=req_event.id,
                )
            return value

        def finish_err(exc: BaseException, served_from: str, started: float) -> None:
            with self._lock:
                self._in_flight -= 1
                if self._parent_events and self._parent_events[-1] == req_event.id:
                    self._parent_events.pop()
                self._emit(
                    ev.EFFECT_FAILED,
                    {
                        "kind": kind,
                        "name": name,
                        "request_hash": request_hash,
                        "error": encode_exception(exc),
                        "served_from": served_from,
                        "latency_seconds": round(time.monotonic() - started, 6),
                        "ordinal": payload["ordinal"],
                        "lifecycle": "failed",
                    },
                    caused_by=req_event.id,
                )

        if decision == "block":
            exc = self.policy.blocked(
                kind=kind, name=name, mode=mode_name, decision=decision
            )
            finish_err(exc, "blocked", time.monotonic())
            raise exc

        if decision == "approval":
            approved = bool(
                self.policy.approve(kind, name, request) if self.policy.approve else False
            )
            if not approved:
                exc = self.policy.blocked(
                    kind=kind, name=name, mode=mode_name, decision=decision
                )
                finish_err(exc, "approval-denied", time.monotonic())
                raise exc
            decision = "execute"
            served_from = "approved"
        else:
            served_from = "live"

        if decision == "simulate":
            started = time.monotonic()
            if self.policy.simulator is None:
                exc = self.policy.blocked(
                    kind=kind, name=name, mode=mode_name, decision=decision
                )
                finish_err(exc, "blocked", started)
                raise exc
            try:
                value = self.policy.simulator(kind, name, request)
            except BaseException as sim_exc:
                finish_err(sim_exc, "simulated", started)
                raise
            return finish_ok(value, "simulated", started)

        # decision == "execute": live call, guarded as mediated I/O.
        if self.mode is SessionMode.PREFIX:
            self.prefix_external_calls += 1
        self.executed_effect_request_ids.append(req_event.id)
        if is_async:
            return self._execute_async(execute, finish_ok, finish_err, served_from)
        started = time.monotonic()
        depth_token = _MEDIATED_DEPTH.set(_MEDIATED_DEPTH.get() + 1)
        try:
            value = execute()
        except BaseException as exc:
            finish_err(exc, served_from, started)
            raise
        finally:
            _MEDIATED_DEPTH.reset(depth_token)
        return finish_ok(value, served_from, started)

    async def _execute_async(
        self,
        execute: Callable[[], Awaitable[Any]],
        finish_ok: Callable[..., Any],
        finish_err: Callable[..., None],
        served_from: str,
    ) -> Any:
        started = time.monotonic()
        depth_token = _MEDIATED_DEPTH.set(_MEDIATED_DEPTH.get() + 1)
        try:
            value = await execute()
        except BaseException as exc:
            finish_err(exc, served_from, started)
            raise
        finally:
            _MEDIATED_DEPTH.reset(depth_token)
        return finish_ok(value, served_from, started)

    # ------------------------------------------------------------------ #
    # checkpoints, hazards, run lifecycle                                 #
    # ------------------------------------------------------------------ #

    def checkpoint(self, state: Any, *, label: str | None = None) -> str | None:
        """Record a restorable agent-state snapshot (record/tail modes).

        A no-op while serving: checkpoints belong to the run that
        recorded them. Returns the checkpoint event id, or ``None`` when
        serving.
        """
        if self.is_serving:
            return None
        encoded = self.codec.encode_response(state)
        with self._lock:
            self._checkpoint_ordinal += 1
            event = self._emit(
                ev.CHECKPOINT_RECORDED,
                {
                    "ordinal": self._checkpoint_ordinal,
                    "label": label,
                    "state": encoded,
                    "state_hash": content_hash(encoded),
                    "invocation": (
                        self._invocation_stack[-1].ordinal
                        if self._invocation_stack
                        else None
                    ),
                    "lossy": find_lossy(encoded),
                },
                caused_by=self._causal_parent,
            )
            return event.id

    def in_mediated_execution(self) -> bool:
        return _MEDIATED_DEPTH.get() > 0

    def note_hazard(self, *, kind: str, detail: str, where: str) -> None:
        """Called by the audit guard when unmediated I/O is observed.

        Recording modes log it (an honesty downgrade); serving modes
        fail closed with :class:`UnrecordedEffectError`.
        """
        if self.is_serving:
            raise UnrecordedEffectError(
                f"live {kind} attempted during {self.mode.value} execution: {detail}",
                what_failed=(
                    f"While serving recorded effects, agent code attempted live "
                    f"{kind} ({detail}) from {where}."
                ),
                why=(
                    "verify() and fork prefixes promise zero live I/O. An "
                    "unmediated call means part of the agent's behavior was never "
                    "recorded, so replaying it honestly is impossible."
                ),
                how_to_fix=(
                    "Route this call through the membrane: wrap the client with "
                    "activegraph_bridge.instrument.wrap_client, declare the "
                    "function as a bridged tool, or wrap the call in "
                    "activegraph_bridge.effect(...)."
                ),
            )
        finding = Finding("blocker", "unrecorded-io", f"unrecorded {kind} from {where}")
        with self._lock:
            self.findings.append(finding)
            if self.graph is not None:
                self._emit(
                    ev.HAZARD_DETECTED,
                    {
                        "kind": kind,
                        "detail": f"unrecorded {kind} from {where}",
                        "where": where,
                        "during": self.mode.value,
                    },
                    caused_by=self._causal_parent,
                )

    def finalize(self, *, status: str) -> None:
        """Emit the run-completed summary (record/tail only, once)."""
        if self.graph is None or self._finalized:
            return
        with self._lock:
            self._finalized = True
            from .report import compute_report

            events = self.graph.events
            self._emit(
                ev.RUN_COMPLETED,
                {
                    "status": status,
                    "invocations": self._invocation_ordinal,
                    "effects": ev.summarize_effect_counts(events),
                    "checkpoints": self._checkpoint_ordinal,
                    "grade": compute_report(events).grade.value,
                },
            )


_FALL_THROUGH = object()  # sentinel: prefix exhausted, no override → go live


def _categorize(kind: str) -> str:
    """Effect category from its kind prefix (drives the graph projection)."""
    head = kind.split(".", 1)[0]
    if head in ("llm", "model", "openai", "anthropic", "gemini", "chat", "completion"):
        return "model"
    if head == "tool":
        return "tool"
    if head in ("retrieval", "vectorstore", "search", "rag"):
        return "retrieval"
    if head in ("memory", "history"):
        return "memory"
    if head in ("time", "random", "id", "uuid", "env"):
        return "determinism"
    return "external"


# -------------------------------------------------------------------------- #
# module-level conveniences                                                   #
# -------------------------------------------------------------------------- #


def effect(
    kind: str,
    request: Any,
    execute: Callable[[], Any],
    *,
    name: str = "",
    side_effect: str = "unknown",
    footprint: Footprint | None = None,
    replay_source: ReplaySource = "recorded",
    observables: tuple[str, ...] = (),
    codec: EffectCodec | None = None,
    category: str | None = None,
) -> Any:
    """Mediate one call through the active session.

    The universal escape hatch: anything an agent does that is
    nondeterministic or externally observable can be wrapped in an
    ``effect``. Outside a session, ``execute()`` runs directly —
    instrumented code works with or without the bridge::

        result = effect(
            "openai.responses.create",
            request=params,
            execute=lambda: client.responses.create(**params),
            side_effect="read",
        )
    """
    session = current_session()
    if session is None:
        return execute()
    return session.effect(
        kind,
        request,
        execute,
        name=name,
        side_effect=side_effect,
        footprint=footprint,
        replay_source=replay_source,
        observables=observables,
        codec=codec,
        category=category,
    )


async def aeffect(
    kind: str,
    request: Any,
    execute: Callable[[], Awaitable[Any]],
    *,
    name: str = "",
    side_effect: str = "unknown",
    footprint: Footprint | None = None,
    replay_source: ReplaySource = "recorded",
    observables: tuple[str, ...] = (),
    codec: EffectCodec | None = None,
    category: str | None = None,
) -> Any:
    """Async twin of :func:`effect`."""
    session = current_session()
    if session is None:
        return await execute()
    return await session.aeffect(
        kind,
        request,
        execute,
        name=name,
        side_effect=side_effect,
        footprint=footprint,
        replay_source=replay_source,
        observables=observables,
        codec=codec,
        category=category,
    )


def checkpoint(state: Any, *, label: str | None = None) -> str | None:
    """Record a restorable agent-state snapshot in the active session.

    No-op outside a session (returns ``None``), so agents can checkpoint
    unconditionally.
    """
    session = current_session()
    if session is None:
        return None
    return session.checkpoint(state, label=label)
