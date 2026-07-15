"""``wrap()``: give an existing agent an ActiveGraph execution membrane.

The wrapped agent keeps its own orchestration, prompts, loops, and
framework control flow — the bridge mediates what it *does*: model
calls, tools, retrieval, time, randomness, external writes, inputs and
outputs. One line for the simplest path::

    agent = wrap(existing_agent)
    answer = agent.invoke(payload)
    print(agent.last_run.report)

The recommended path is factory-backed, because replay and forking need
a clean agent::

    def build_agent():
        return ExistingAgent(model="...", tools=[...])

    agent = wrap(build_agent, store="sqlite:///agent-runs.db")

    with agent.execution(label="customer-support") as run:
        answer = agent.invoke({"messages": [...]})

    run.verify()                      # fresh agent, recorded effects, zero live calls
    fork = run.fork(
        before=run.events.tool_call("lookup_order", occurrence=1),
        overrides={"tool_result": {"status": "shipped"}},
    )
    alternative = fork.execute()
    print(run.diff(fork))

Invocation surface: ``invoke`` / ``ainvoke`` / ``stream`` / ``astream``
/ ``batch`` (plus calling the wrapper directly). Unknown attributes pass
through to the underlying agent, so the wrapper stays drop-in.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from activegraph.core.clock import Clock

from ._store import BridgeStore, resolve_store
from .adapters import resolve_adapter
from .codecs import AutoCodec, EffectCodec
from .engine import Recording, WrapSpec, arecord_invoke, record_invoke
from .errors import BridgeConfigurationError
from .policy import SideEffectPolicy
from .projection import DefaultProjector
from .runs import Run
from .session import ExecutionSession, _ACTIVE, current_session

__all__ = ["wrap", "recorded_agent", "WrappedAgent"]

DEFAULT_STORE = "sqlite:///agent-runs.db"


def build_spec(
    agent_or_factory: Any,
    *,
    method: str = "invoke",
    adapter: Any = "auto",
    store: str | BridgeStore = DEFAULT_STORE,
    policy: SideEffectPolicy | None = None,
    codec: EffectCodec | None = None,
    projector: Any = "default",
    match: str = "auto",
    label: str | None = None,
    clock: Callable[[], Clock] | None = None,
    metadata: dict[str, Any] | None = None,
) -> WrapSpec:
    """Resolve wrap() arguments into the immutable spec runs share."""
    is_factory = (
        inspect.isfunction(agent_or_factory) or inspect.ismethod(agent_or_factory)
    ) and _looks_like_factory(agent_or_factory)
    factory = agent_or_factory if is_factory else None
    target = None if is_factory else agent_or_factory

    probe = factory() if is_factory else target
    resolved_adapter = resolve_adapter(adapter, probe)
    if not resolved_adapter.supports(probe):
        raise BridgeConfigurationError(
            f"adapter {resolved_adapter.name!r} does not support this target",
            what_failed=(
                f"wrap() resolved adapter {resolved_adapter.name!r} but its "
                f"supports() check rejected {type(probe).__name__}."
            ),
            why="An adapter that cannot drive the target would fail at first invoke.",
            how_to_fix=(
                "Pass adapter='auto' (the default) to search all registered "
                "adapters, or an explicit AgentAdapter instance."
            ),
        )

    if projector == "default":
        projector_factory: Callable[[], Any] | None = DefaultProjector
    elif projector in (None, "none"):
        projector_factory = None
    elif callable(projector) and inspect.isclass(projector):
        projector_factory = projector
    elif callable(getattr(projector, "project", None)):
        cls = type(projector)
        projector_factory = cls  # fresh instance per run
    else:
        raise BridgeConfigurationError(
            f"projector must be 'default', 'none', a GraphProjector class or "
            f"instance; got {type(projector).__name__}"
        )

    return WrapSpec(
        target=probe if not is_factory else None,
        factory=factory,
        method=method,
        adapter=resolved_adapter,
        store=resolve_store(store),
        policy=policy or SideEffectPolicy(),
        codec=codec or AutoCodec(),
        projector_factory=projector_factory,
        match=match,
        label=label,
        clock_factory=clock or Clock,
        metadata=dict(metadata or {}),
    )


def _looks_like_factory(fn: Callable) -> bool:
    """A zero-arg callable is a factory; anything with required params is
    the agent itself (a callable agent like ``run_agent(payload)``)."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for p in signature.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is p.empty:
            return False
    return True


def wrap(
    agent_or_factory: Any,
    *,
    method: str = "invoke",
    adapter: Any = "auto",
    store: str | BridgeStore = DEFAULT_STORE,
    policy: SideEffectPolicy | None = None,
    codec: EffectCodec | None = None,
    projector: Any = "default",
    match: str = "auto",
    label: str | None = None,
    clock: Callable[[], Clock] | None = None,
    metadata: dict[str, Any] | None = None,
) -> "WrappedAgent":
    """Wrap an existing agent (or, preferably, its factory).

    Parameters
    ----------
    agent_or_factory:
        A zero-argument factory (recommended — enables verify and fork
        via fresh re-execution), a live agent instance, or a callable
        agent taking the payload directly.
    method:
        The primary invocation method on the target (default
        ``"invoke"``; ``"run"`` and plain callables also just work).
    adapter:
        ``"auto"`` (default), an adapter name, or an ``AgentAdapter``.
    store:
        ``sqlite:///path.db`` (default ``sqlite:///agent-runs.db``),
        a bare path, ``"memory://"`` for tests, or a ``BridgeStore``.
    policy:
        :class:`~activegraph_bridge.policy.SideEffectPolicy` controlling
        writes in live runs and fork tails.
    codec:
        Default :class:`~activegraph_bridge.codecs.EffectCodec` for
        requests/responses (per-effect codecs override).
    projector:
        ``"default"`` for the AgentExecutionPack graph projection,
        ``"none"`` to record events only, or your own
        :class:`~activegraph_bridge.projection.GraphProjector` for
        domain objects.
    match:
        Replay matching discipline: ``"auto"`` (tolerates concurrent
        completion reordering; content must still match exactly) or
        ``"strict"``.
    """
    spec = build_spec(
        agent_or_factory,
        method=method,
        adapter=adapter,
        store=store,
        policy=policy,
        codec=codec,
        projector=projector,
        match=match,
        label=label,
        clock=clock,
        metadata=metadata,
    )
    return WrappedAgent(spec)


class WrappedAgent:
    """The proxy that preserves the original invocation surface.

    Inside an ``execution()`` block, invocations join that block's run.
    Outside one, each invocation records its own single-invocation run.
    Invoked while *another* wrapped agent's session is active, the
    invocation is recorded as a nested invocation of the outer run —
    causality follows the call graph.
    """

    def __init__(self, spec: WrapSpec) -> None:
        self._spec = spec
        self._last_run: Run | None = None

    # -- introspection ---------------------------------------------------------

    @property
    def spec(self) -> WrapSpec:
        return self._spec

    @property
    def last_run(self) -> Run | None:
        """The most recent run this wrapper recorded (this process)."""
        return self._last_run

    def runs(self) -> list:
        """All runs in this wrapper's store (records, newest last)."""
        return self._spec.store.list_runs()

    def __repr__(self) -> str:
        return f"WrappedAgent({self._spec.describe_target()}, store={self._spec.store.url!r})"

    def __getattr__(self, item: str) -> Any:
        # Transparent proxy for everything that isn't an invocation surface.
        target = self._spec.target if self._spec.factory is None else self._spec.factory
        return getattr(target, item)

    # -- execution contexts -----------------------------------------------------

    @contextmanager
    def execution(
        self,
        *,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        goal: str | None = None,
    ) -> Iterator[Run]:
        """Group invocations into one run and expose its handle::

            with agent.execution(label="support", metadata={...}) as run:
                answer = agent.invoke(payload)
            run.verify()

        The run handle is live inside the block and durable after it.
        Safe under concurrency: the binding is context-local, so parallel
        execution blocks (threads or asyncio tasks) each get their own
        run.
        """
        recording = Recording.open(
            self._spec, label=label, metadata=metadata, goal=goal
        )
        run = Run(self._spec.store, recording.run_id, spec=self._spec)
        self._last_run = run
        recording.session.agent_cache[id(self)] = recording.agent
        status = "completed"
        try:
            with recording.session:
                yield run
        except BaseException:
            status = "failed"
            raise
        finally:
            recording.finalize(status)

    # -- invocation surface -------------------------------------------------------

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke_sync(self._spec.method, args, kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke_sync("run", args, kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke_sync(self._spec.method, args, kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        return self._invoke_sync("stream", args, kwargs)

    def batch(self, inputs: list, **kwargs: Any) -> list:
        """Sequential batch: each input is its own recorded invocation."""
        return [self._invoke_sync(self._spec.method, (item,), kwargs) for item in inputs]

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke_async("ainvoke", args, kwargs)

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke_async("astream", args, kwargs)

    # -- internals ----------------------------------------------------------------

    def _agent_for(self, session: ExecutionSession) -> Any:
        agent = session.agent_cache.get(id(self))
        if agent is None:
            agent = self._spec.build_agent()
            session.agent_cache[id(self)] = agent
        return agent

    def _invoke_sync(self, method: str, args: tuple, kwargs: dict) -> Any:
        session = current_session()
        if session is not None:
            # Joined an active membrane: this block's run, or an outer
            # agent's run (nested invocation). Same mechanics either way.
            return record_invoke(
                session,
                self._spec,
                self._agent_for(session),
                method=method,
                args=args,
                kwargs=kwargs,
            )
        # Standalone: one run for this single invocation.
        recording = Recording.open(self._spec)
        self._last_run = Run(self._spec.store, recording.run_id, spec=self._spec)
        try:
            with recording.session:
                output = record_invoke(
                    recording.session,
                    self._spec,
                    recording.agent,
                    method=method,
                    args=args,
                    kwargs=kwargs,
                )
        except BaseException:
            recording.finalize("failed")
            raise
        if isinstance(output, Iterator):
            # Keep the membrane active around each pull, finalize at
            # exhaustion — streams outlive the invoke() call.
            return _streaming_run(recording, output)
        recording.finalize("completed")
        return output

    async def _invoke_async(self, method: str, args: tuple, kwargs: dict) -> Any:
        session = current_session()
        if session is not None:
            return await arecord_invoke(
                session,
                self._spec,
                self._agent_for(session),
                method=method,
                args=args,
                kwargs=kwargs,
            )
        recording = Recording.open(self._spec)
        self._last_run = Run(self._spec.store, recording.run_id, spec=self._spec)
        status = "completed"
        try:
            with recording.session:
                return await arecord_invoke(
                    recording.session,
                    self._spec,
                    recording.agent,
                    method=method,
                    args=args,
                    kwargs=kwargs,
                )
        except BaseException:
            status = "failed"
            raise
        finally:
            recording.finalize(status)


def _streaming_run(recording: Recording, inner: Iterator[Any]) -> Iterator[Any]:
    """Drive a standalone streamed invocation with the session re-activated
    around every pull, finalizing the run when the stream ends."""

    def generator() -> Iterator[Any]:
        status = "completed"
        try:
            while True:
                token = _ACTIVE.set(recording.session)
                try:
                    try:
                        chunk = next(inner)
                    except StopIteration:
                        return
                finally:
                    _ACTIVE.reset(token)
                yield chunk
        except BaseException:
            status = "failed"
            raise
        finally:
            recording.finalize(status)

    return generator()


def recorded_agent(
    fn: Callable | None = None,
    *,
    store: str | BridgeStore = DEFAULT_STORE,
    **wrap_kwargs: Any,
) -> Any:
    """Decorator form: record a callable agent's executions.

    ::

        @recorded_agent(store="sqlite:///runs.db")
        def run_agent(payload):
            return existing_agent.run(payload)

        answer = run_agent({"question": "..."})
        run_agent.last_run.report

    Uses the exact same engine as :func:`wrap` — the decorated function
    is the agent, each call is a recorded invocation.
    """

    def apply(f: Callable) -> WrappedAgent:
        wrapped = wrap(f, store=store, **wrap_kwargs)
        wrapped.__doc__ = f.__doc__
        return wrapped

    if fn is not None:
        return apply(fn)
    return apply
