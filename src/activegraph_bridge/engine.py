"""The execution engine: one machine for record, verify, and fork.

Everything that runs a wrapped agent runs through here, in one of three
shapes that differ only in how their session answers:

- **record** — open a fresh run, execute live, capture everything.
- **shadow** (``run.verify()``) — fresh agent, recorded inputs, served
  effects, strict comparison, zero live calls.
- **fork** — shadow over the child's inherited prefix, then live
  recording for the divergent tail.

The engine deliberately re-executes *real agent code* during shadow and
fork — that is what reconstructs hidden Python state (message lists,
framework internals) that no event log can carry. The membrane makes the
re-execution safe: every nondeterministic input the agent consumed the
first time is served back to it byte-identically.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

from activegraph.core.clock import Clock
from activegraph.core.graph import Graph
from activegraph.core.ids import IDGen
from activegraph.store.base import replay_into

from . import events as ev
from ._canonical import content_hash
from ._store import BridgeStore
from .codecs import EffectCodec
from .errors import (
    BridgeConfigurationError,
    EffectBlockedError,
    ReconstructionError,
    ReplayDivergence,
    UnrecordedEffectError,
)
from .policy import SideEffectPolicy
from .report import Finding
from .determinism import runtime_fingerprint
from .script import Cursor, EffectScript, InvocationEntry
from .session import ExecutionSession, ForkOverride, SessionMode

__all__ = [
    "WrapSpec",
    "Recording",
    "record_invoke",
    "arecord_invoke",
    "shadow_verify",
    "ashadow_verify",
    "fork_execute",
    "afork_execute",
    "requires_async_drive",
]


@dataclass
class WrapSpec:
    """Everything ``wrap()`` decided, shared by every run of the agent."""

    target: Any
    factory: Callable[[], Any] | None
    method: str
    adapter: Any
    store: BridgeStore
    policy: SideEffectPolicy
    codec: EffectCodec
    projector_factory: Callable[[], Any] | None
    match: str
    label: str | None
    clock_factory: Callable[[], Clock]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reconstruction(self) -> str:
        """How a clean agent can be obtained for replay/verify/fork.

        ``fresh_factory``       — the factory builds one (the gold path).
        ``stateless_callable``  — the target is a plain function: the
                                  function object *is* its clean self
                                  (closures over mutable state are the
                                  caller's risk, and output divergence
                                  catches them).
        ``shared_instance``     — a live object was wrapped; there is no
                                  honest way to rebuild it.
        """
        if self.factory is not None:
            return "fresh_factory"
        if inspect.isfunction(self.target) or inspect.ismethod(self.target):
            return "stateless_callable"
        return "shared_instance"

    def build_agent(self) -> Any:
        if self.factory is not None:
            return self.factory()
        return self.target

    def describe_target(self) -> str:
        t = self.factory if self.factory is not None else self.target
        name = getattr(t, "__qualname__", None) or getattr(t, "__name__", None)
        return name or type(self.target).__name__

    def fingerprint(self) -> dict[str, Any]:
        return runtime_fingerprint(
            self.factory if self.factory is not None else self.target,
            {
                "method": self.method,
                "adapter": getattr(self.adapter, "name", "?"),
                "match": self.match,
                "reconstruction": self.reconstruction,
            },
        )


# --------------------------------------------------------------------------- #
# recording                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class Recording:
    """One open recording run: its graph, session, and lifecycle."""

    spec: WrapSpec
    session: ExecutionSession
    graph: Graph
    run_id: str
    agent: Any

    @classmethod
    def open(
        cls,
        spec: WrapSpec,
        *,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        goal: str | None = None,
    ) -> "Recording":
        run_id = IDGen().run()
        event_store = spec.store.create_run(
            run_id, label=label or spec.label, goal=goal
        )
        try:
            graph = Graph(ids=IDGen(), clock=spec.clock_factory(), run_id=run_id)
            graph.attach_store(event_store)
            projector = spec.projector_factory() if spec.projector_factory else None
            session = ExecutionSession(
                mode=SessionMode.RECORD,
                graph=graph,
                policy=spec.policy,
                codec=spec.codec,
                projector=projector,
            )
            session._emit(
                ev.RUN_STARTED,
                {
                    "mode": "record",
                    "label": label or spec.label,
                    "metadata": dict(spec.metadata) | dict(metadata or {}),
                    "adapter": getattr(spec.adapter, "name", "?"),
                    "adapter_capabilities": spec.adapter.capabilities.to_dict(),
                    "target": spec.describe_target(),
                    "method": spec.method,
                    "reconstruction": spec.reconstruction,
                    "fingerprint": spec.fingerprint(),
                    "match": spec.match,
                },
            )
            return cls(
                spec=spec,
                session=session,
                graph=graph,
                run_id=run_id,
                agent=spec.build_agent(),
            )
        except BaseException:
            event_store.close()
            raise

    def finalize(self, status: str) -> None:
        try:
            self.session.finalize(status=status)
        finally:
            store = self.graph.store
            close = getattr(store, "close", None)
            if close:
                close()


def _invocation_input(args: tuple, kwargs: dict) -> dict[str, Any]:
    return {"args": list(args), "kwargs": dict(kwargs)}


def _decode_input(codec: EffectCodec, encoded: Any) -> tuple[tuple, dict]:
    decoded = codec.decode_response(encoded)
    if isinstance(decoded, dict) and "args" in decoded and "kwargs" in decoded:
        return tuple(decoded["args"]), dict(decoded["kwargs"])
    return (decoded,), {}


def record_invoke(
    session: ExecutionSession,
    spec: WrapSpec,
    agent: Any,
    *,
    method: str,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Run one invocation against an active session (any mode).

    This single function serves live recording, nested invocations,
    shadow re-execution, and fork tails — the session decides what each
    boundary and effect means.
    """
    inv = session.begin_invocation(
        method=method, input_value=_invocation_input(args, kwargs)
    )
    try:
        with spec.adapter.instrument(agent, session):
            output = spec.adapter.invoke(agent, method, args, kwargs)
            # Only true iterators are streams (generators, SDK stream
            # objects). Models and containers define __iter__ but not
            # __next__ and must be captured as plain outputs.
            if isinstance(output, Iterator):
                return _record_stream(session, inv, output)
    except BaseException as exc:
        session.finish_invocation(inv, error=exc)
        raise
    session.finish_invocation(inv, output=output)
    return output


async def arecord_invoke(
    session: ExecutionSession,
    spec: WrapSpec,
    agent: Any,
    *,
    method: str,
    args: tuple,
    kwargs: dict,
) -> Any:
    inv = session.begin_invocation(
        method=method, input_value=_invocation_input(args, kwargs)
    )
    try:
        with spec.adapter.instrument(agent, session):
            output = await spec.adapter.ainvoke(agent, method, args, kwargs)
            if isinstance(output, AsyncIterator):
                return _record_astream(session, inv, output)
            if isinstance(output, Iterator):
                return _record_stream(session, inv, output)
    except BaseException as exc:
        session.finish_invocation(inv, error=exc)
        raise
    session.finish_invocation(inv, output=output)
    return output


def _record_stream(session: ExecutionSession, inv: Any, stream: Any) -> Any:
    """Tee a streamed invocation: yield chunks live, record the assembly."""

    def generator():
        chunks: list[Any] = []
        try:
            for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except BaseException as exc:
            session.finish_invocation(inv, error=exc)
            raise
        session.finish_invocation(inv, output={"stream": True, "chunks": chunks})

    return generator()


def _record_astream(
    session: ExecutionSession, inv: Any, stream: AsyncIterator[Any]
) -> AsyncIterator[Any]:
    """Tee an async stream while preserving chunk order and session effects."""

    async def generator() -> AsyncIterator[Any]:
        chunks: list[Any] = []
        try:
            async for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except BaseException as exc:
            session.finish_invocation(inv, error=exc)
            raise
        session.finish_invocation(inv, output={"stream": True, "chunks": chunks})

    return generator()


# --------------------------------------------------------------------------- #
# verify (shadow)                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class VerificationResult:
    """The outcome of a shadow execution.

    ``ok`` means: every recorded effect was requested again with an
    identical canonical hash, every response was served from the record,
    zero live calls happened, and every invocation reproduced its
    recorded output hash.
    """

    ok: bool
    run_id: str
    effects_served: int = 0
    invocations: int = 0
    reordered: int = 0
    divergence: Exception | None = None
    findings: list[Finding] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    def raise_if_failed(self) -> "VerificationResult":
        if not self.ok and self.divergence is not None:
            raise self.divergence
        if not self.ok:
            raise ReplayDivergence("verification failed", got={"run_id": self.run_id})
        return self

    def __str__(self) -> str:
        status = "verified" if self.ok else "FAILED"
        lines = [
            f"Verification: {status}",
            f"Served:       {self.effects_served} recorded effects across "
            f"{self.invocations} invocation(s), 0 live calls",
        ]
        if self.reordered:
            lines.append(
                f"Note:         {self.reordered} effect(s) matched out of order "
                f"(concurrent completion)"
            )
        if self.divergence is not None:
            first_line = str(self.divergence).splitlines()[0]
            lines.append(f"Divergence:   {first_line}")
        return "\n".join(lines)


def _require_reconstructable(spec: WrapSpec, agent: Any | None, purpose: str) -> Any:
    if agent is not None:
        return agent
    if spec.reconstruction == "stateless_callable":
        return spec.target
    if spec.factory is None:
        raise ReconstructionError(
            f"{purpose} needs a clean agent and no reconstruction strategy exists",
            what_failed=(
                f"{purpose} re-executes agent code from the beginning, which "
                f"requires a fresh agent. This run was recorded from a live "
                f"instance passed directly to wrap()."
            ),
            why=(
                "A shared instance carries hidden state mutated by the original "
                "run (message history, caches). Re-executing on top of that "
                "state would compare against a contaminated baseline — the "
                "result would be neither a verification nor an honest fork."
            ),
            how_to_fix=(
                "Wrap a factory instead of an instance:\n"
                "    agent = wrap(build_agent, store=...)   # def build_agent(): ...\n"
                "or pass a clean agent explicitly:\n"
                f"    run.{'verify' if purpose == 'verify()' else 'fork'}(..., agent=build_agent())"
            ),
        )
    return spec.build_agent()


def shadow_verify(
    spec: WrapSpec,
    events: list,
    run_id: str,
    *,
    agent: Any | None = None,
    match: str | None = None,
) -> VerificationResult:
    """Re-execute a recorded run with served effects and strict comparison."""
    script = EffectScript.from_events(events)
    cursor = Cursor(script, match=match or spec.match)
    session = ExecutionSession(
        mode=SessionMode.SHADOW,
        cursor=cursor,
        policy=spec.policy,
        codec=spec.codec,
        run_id=run_id,
    )
    agent = _require_reconstructable(spec, agent, "verify()")
    top_level = script.top_level_invocations()
    divergence: Exception | None = None
    try:
        with session:
            for start, _end in top_level:
                _drive_invocation(session, spec, agent, start)
            leftover = cursor.remaining_effects()
            if leftover:
                raise ReplayDivergence(
                    "re-execution finished with recorded effects left unserved",
                    expected={
                        "next": {
                            "kind": leftover[0].kind,
                            "name": leftover[0].name,
                            "event": leftover[0].request_event_id,
                        },
                        "remaining": len(leftover),
                    },
                    got={},
                )
    except (ReplayDivergence, UnrecordedEffectError) as exc:
        divergence = exc

    return VerificationResult(
        ok=divergence is None,
        run_id=run_id,
        effects_served=session.effects_served,
        invocations=len(top_level),
        reordered=len(cursor.reordered),
        divergence=divergence,
        findings=list(session.findings),
    )


async def ashadow_verify(
    spec: WrapSpec,
    events: list,
    run_id: str,
    *,
    agent: Any | None = None,
    match: str | None = None,
) -> VerificationResult:
    """Async counterpart to :func:`shadow_verify` for any invocation surface."""
    script = EffectScript.from_events(events)
    cursor = Cursor(script, match=match or spec.match)
    session = ExecutionSession(
        mode=SessionMode.SHADOW,
        cursor=cursor,
        policy=spec.policy,
        codec=spec.codec,
        run_id=run_id,
    )
    agent = _require_reconstructable(spec, agent, "verify()")
    top_level = script.top_level_invocations()
    divergence: Exception | None = None
    try:
        with session:
            for start, _end in top_level:
                await _adrive_invocation(session, spec, agent, start)
            leftover = cursor.remaining_effects()
            if leftover:
                raise ReplayDivergence(
                    "re-execution finished with recorded effects left unserved",
                    expected={
                        "next": {
                            "kind": leftover[0].kind,
                            "name": leftover[0].name,
                            "event": leftover[0].request_event_id,
                        },
                        "remaining": len(leftover),
                    },
                    got={},
                )
    except (ReplayDivergence, UnrecordedEffectError) as exc:
        divergence = exc

    return VerificationResult(
        ok=divergence is None,
        run_id=run_id,
        effects_served=session.effects_served,
        invocations=len(top_level),
        reordered=len(cursor.reordered),
        divergence=divergence,
        findings=list(session.findings),
    )


def requires_async_drive(events: list) -> bool:
    """Whether a recording contains a natively asynchronous invocation."""
    return any(
        start.method in {"ainvoke", "arun", "astream"}
        for start, _end in EffectScript.from_events(events).top_level_invocations()
    )


def _drive_invocation(
    session: ExecutionSession,
    spec: WrapSpec,
    agent: Any,
    start: InvocationEntry,
    *,
    input_override: Any = None,
) -> Any:
    """Re-drive one recorded top-level invocation against the session."""
    if input_override is not None:
        encoded = session.codec.encode_request(input_override)
        args, kwargs = _decode_input(session.codec, encoded)
        inv = session.begin_invocation(
            method=start.method, encoded_input=encoded, input_hash=content_hash(encoded)
        )
    else:
        args, kwargs = _decode_input(session.codec, start.input)
        inv = session.begin_invocation(
            method=start.method,
            encoded_input=start.input,
            input_hash=start.input_hash or None,
        )
    try:
        with spec.adapter.instrument(agent, session):
            output = spec.adapter.invoke(agent, start.method, args, kwargs)
            if inspect.isawaitable(output) or isinstance(output, AsyncIterator):
                if inspect.iscoroutine(output):
                    output.close()
                raise BridgeConfigurationError(
                    "an asynchronous invocation requires the async execution path",
                    what_failed=(
                        f"The recorded {start.method!r} invocation returned an "
                        "awaitable while a synchronous verify/fork driver was active."
                    ),
                    why="Async agent code must be awaited so its recorded effects execute.",
                    how_to_fix="Use `await run.averify()` or `await fork.aexecute()`.",
                )
            if isinstance(output, Iterator):
                output = {"stream": True, "chunks": list(output)}
    except (BridgeConfigurationError, ReplayDivergence, UnrecordedEffectError):
        raise
    except EffectBlockedError as exc:
        # A policy refusal is the caller's decision point, not an agent
        # failure to record and move past: land it in the log, then raise.
        session.finish_invocation(inv, error=exc)
        raise
    except BaseException as exc:
        session.finish_invocation(inv, error=exc)
        return None
    session.finish_invocation(inv, output=output)
    return output


async def _adrive_invocation(
    session: ExecutionSession,
    spec: WrapSpec,
    agent: Any,
    start: InvocationEntry,
    *,
    input_override: Any = None,
) -> Any:
    """Await and re-drive one invocation, consuming sync or async streams."""
    if input_override is not None:
        encoded = session.codec.encode_request(input_override)
        args, kwargs = _decode_input(session.codec, encoded)
        inv = session.begin_invocation(
            method=start.method, encoded_input=encoded, input_hash=content_hash(encoded)
        )
    else:
        args, kwargs = _decode_input(session.codec, start.input)
        inv = session.begin_invocation(
            method=start.method,
            encoded_input=start.input,
            input_hash=start.input_hash or None,
        )
    try:
        with spec.adapter.instrument(agent, session):
            output = await spec.adapter.ainvoke(agent, start.method, args, kwargs)
            if isinstance(output, AsyncIterator):
                chunks = [chunk async for chunk in output]
                output = {"stream": True, "chunks": chunks}
            elif isinstance(output, Iterator):
                output = {"stream": True, "chunks": list(output)}
    except (ReplayDivergence, UnrecordedEffectError):
        raise
    except EffectBlockedError as exc:
        session.finish_invocation(inv, error=exc)
        raise
    except BaseException as exc:
        session.finish_invocation(inv, error=exc)
        return None
    session.finish_invocation(inv, output=output)
    return output


# --------------------------------------------------------------------------- #
# fork                                                                         #
# --------------------------------------------------------------------------- #


def fork_execute(
    spec: WrapSpec,
    *,
    child_run_id: str,
    parent_events: list,
    override: ForkOverride | None,
    side_effects: str,
    agent: Any | None = None,
    match: str | None = None,
) -> tuple[Any, ExecutionSession]:
    """Run a fork: serve the inherited prefix, then record the tail.

    ``side_effects`` is the fork's write posture: ``"fail_closed"``
    (default — blocked unless the policy simulates/approves) or
    ``"live"`` (writes execute as in live recording; requires the caller
    to have said so explicitly).
    """
    child_events = spec.store.load_events(child_run_id)
    prefix_script = EffectScript.from_events(child_events)
    cursor = Cursor(prefix_script, match=match or spec.match)

    graph = Graph(ids=IDGen(), clock=spec.clock_factory(), run_id=child_run_id)
    replay_into(graph, child_events)
    graph.ids.reseed_from_events(child_events)
    event_store = spec.store.open_run(child_run_id)
    graph.attach_store(event_store)
    try:
        policy = spec.policy
        if side_effects == "live":
            # Explicit authorization: the tail writes like a live recording.
            from dataclasses import replace

            policy = replace(policy, on_fork_write=policy.on_write)

        projector = spec.projector_factory() if spec.projector_factory else None
        session = ExecutionSession(
            mode=SessionMode.PREFIX,
            graph=graph,
            cursor=cursor,
            policy=policy,
            codec=spec.codec,
            projector=projector,
            override=override,
        )
        agent = _require_reconstructable(spec, agent, "fork.execute()")

        # The drive plan comes from the parent's full recording: the fork
        # replays each recorded invocation (prefix ones validate + serve;
        # post-fork-point ones run live in the tail with their recorded
        # inputs — the counterfactual keeps the conversation script).
        full_script = EffectScript.from_events(parent_events)
        outputs: list[Any] = []
        status = "completed"
        try:
            with session:
                for start, _end in full_script.top_level_invocations():
                    input_override = None
                    if (
                        override is not None
                        and override.has_input
                        and override.target_event_id == start.start_event_id
                    ):
                        input_override = override.input
                    outputs.append(
                        _drive_invocation(
                            session, spec, agent, start, input_override=input_override
                        )
                    )
        except BaseException:
            status = "failed"
            raise
        finally:
            session.finalize(status=status)
        return (outputs[-1] if outputs else None), session
    finally:
        event_store.close()


async def afork_execute(
    spec: WrapSpec,
    *,
    child_run_id: str,
    parent_events: list,
    override: ForkOverride | None,
    side_effects: str,
    agent: Any | None = None,
    match: str | None = None,
) -> tuple[Any, ExecutionSession]:
    """Async fork execution with the same prefix and policy semantics."""
    child_events = spec.store.load_events(child_run_id)
    prefix_script = EffectScript.from_events(child_events)
    cursor = Cursor(prefix_script, match=match or spec.match)

    graph = Graph(ids=IDGen(), clock=spec.clock_factory(), run_id=child_run_id)
    replay_into(graph, child_events)
    graph.ids.reseed_from_events(child_events)
    event_store = spec.store.open_run(child_run_id)
    graph.attach_store(event_store)
    try:
        policy = spec.policy
        if side_effects == "live":
            from dataclasses import replace

            policy = replace(policy, on_fork_write=policy.on_write)

        projector = spec.projector_factory() if spec.projector_factory else None
        session = ExecutionSession(
            mode=SessionMode.PREFIX,
            graph=graph,
            cursor=cursor,
            policy=policy,
            codec=spec.codec,
            projector=projector,
            override=override,
        )
        agent = _require_reconstructable(spec, agent, "fork.execute()")

        full_script = EffectScript.from_events(parent_events)
        outputs: list[Any] = []
        status = "completed"
        try:
            with session:
                for start, _end in full_script.top_level_invocations():
                    input_override = None
                    if (
                        override is not None
                        and override.has_input
                        and override.target_event_id == start.start_event_id
                    ):
                        input_override = override.input
                    outputs.append(
                        await _adrive_invocation(
                            session, spec, agent, start, input_override=input_override
                        )
                    )
        except BaseException:
            status = "failed"
            raise
        finally:
            session.finalize(status=status)
        return (outputs[-1] if outputs else None), session
    finally:
        event_store.close()
