"""activegraph-bridge: wrap any agent in an ActiveGraph execution membrane.

Your agent keeps its own orchestration; the bridge mediates everything
it does that is nondeterministic or externally observable — model calls,
tools, retrieval, time, randomness, writes — and makes the resulting
event log *executable*: replay it, verify it with zero live calls, fork
it at any recorded boundary with a changed tool result or input.

Three commitments, everywhere:

1. **Mediate, don't observe.** A callback can produce a trace; only a
   membrane can serve recorded reality back to re-executing code.
2. **Report honest replayability.** Every run earns a grade (observed /
   envelope / boundary / checkpointed / native) computed from what was
   actually captured — never from what was intended.
3. **Fail closed on writes.** Replays, verifications, and fork tails do
   not send the email twice unless you explicitly authorize it.

Quickstart::

    from activegraph_bridge import wrap

    agent = wrap(build_agent, store="sqlite:///agent-runs.db")

    with agent.execution(label="customer-support") as run:
        answer = agent.invoke({"messages": [...]})

    print(run.report)          # Replayability: boundary (unverified) ...
    run.verify()               # fresh agent, recorded effects, zero live calls
    print(run.report)          # Replayability: boundary-verified ...

    fork = run.fork(
        before=run.events.tool_call("lookup_order", occurrence=1),
        overrides={"tool_result": {"status": "shipped"}},
    )
    alternative = fork.execute()
    print(run.diff(fork))
"""

from . import det, instrument
from .adapters import (
    AdapterCapabilities,
    AgentAdapter,
    GenericAdapter,
    register_adapter,
)
from .codecs import AttrBox, AutoCodec, EffectCodec, JsonCodec, OpaqueValue
from .engine import VerificationResult, WrapSpec
from .errors import (
    BridgeConfigurationError,
    BridgeError,
    EffectBlockedError,
    NotForkableError,
    OverrideError,
    ReconstructionError,
    ReplayDivergence,
    ReplayedEffectFailure,
    UnrecordedEffectError,
)
from .policy import SideEffectPolicy
from .projection import DefaultProjector, GraphProjector
from .report import Finding, Grade, ReplayabilityReport
from .runs import EventRef, Fork, Replay, Run, RunDiff, list_runs, load_run
from .session import ExecutionSession, aeffect, checkpoint, current_session, effect
from .tools import bridge_tool, wrap_tool
from .wrapper import WrappedAgent, recorded_agent, wrap

__version__ = "0.1.0"

__all__ = [
    # the front door
    "wrap",
    "recorded_agent",
    "WrappedAgent",
    # runs and their operations
    "Run",
    "Fork",
    "RunDiff",
    "Replay",
    "EventRef",
    "load_run",
    "list_runs",
    "VerificationResult",
    # the membrane, for agent code
    "effect",
    "aeffect",
    "checkpoint",
    "current_session",
    "ExecutionSession",
    "bridge_tool",
    "wrap_tool",
    "det",
    "instrument",
    # policy, codecs, projection
    "SideEffectPolicy",
    "EffectCodec",
    "AutoCodec",
    "JsonCodec",
    "AttrBox",
    "OpaqueValue",
    "GraphProjector",
    "DefaultProjector",
    # honesty
    "Grade",
    "Finding",
    "ReplayabilityReport",
    # adapters
    "AgentAdapter",
    "AdapterCapabilities",
    "GenericAdapter",
    "register_adapter",
    "WrapSpec",
    # errors
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
