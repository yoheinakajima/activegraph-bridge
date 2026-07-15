# Error catalog

Every bridge error inherits from `BridgeError`, which inherits from
`activegraph.ActiveGraphError` — `except ActiveGraphError` covers the
bridge, and every structured error renders in ActiveGraph's locked
format (summary, what failed, why, how to fix, doc link). Anchors below
match each error's `doc_url`.

### bridge-error

`BridgeError` — the root. Never raised bare; catch it to handle
anything bridge-specific.

### bridge-configuration-error

`BridgeConfigurationError` (also an ActiveGraph `ConfigurationError`) —
invalid `wrap()` arguments, unknown adapters, unsupported store URLs
(postgres is refused up front because native forking is SQLite-first),
double-executed forks.

### replay-divergence

`ReplayDivergence` (also a `ReplayError`) — a re-execution requested
something the recording does not contain at that point: different
request hash, extra call, missing call, changed invocation input, or a
different final output. Carries `expected` and `got` dicts with the
recorded event id, so tools can render a precise two-sided diff.
`verify()` returns it inside `VerificationResult.divergence`; a fork's
*prefix* raises it directly (a fork must not build on an unfaithful
prefix).

Common causes and fixes are spelled out in the message: unmediated
time/randomness (`det.*` is the fix), mutable globals or warm caches,
or intentional code changes (fork instead of verifying).

### unrecorded-effect

`UnrecordedEffectError` (also a `ReplayError`) — agent code reached for
live I/O (socket, subprocess, shell) while effects were being served.
The audit guard aborts the operation before it happens. Fix: route the
call through `instrument.wrap_client`, a `@bridge_tool`, or
`effect(...)`.

### effect-blocked

`EffectBlockedError` (also an `ExecutionError`) — a write effect was
refused by the side-effect policy (fork tails block writes by default;
live runs can be configured to `approval`/`block`). The refusal is also
recorded as an `effect.failed` event with `served_from="blocked"`. The
message lists the three ways forward: a simulator, an approval handler,
or `side_effects="live"`.

### replayed-effect-failure

`ReplayedEffectFailure` — a recorded effect failure whose original
exception class could not be imported at replay time. Carries
`original_class`. Recorded failures whose classes import are re-raised
as themselves; this is the honest fallback, never a swallow.

### reconstruction-error

`ReconstructionError` (also a `ReplayError`) — verify/fork needed a
clean agent and none can be built: the run was recorded from a live
instance and no factory, reset, or checkpoint strategy exists. Fixes:
wrap a factory, or pass a clean agent explicitly
(`run.verify(agent=...)`), or `run.attach(build_agent)` for runs loaded
from disk.

### not-forkable

`NotForkableError` (also a `ReplayError`) — the selected fork anchor is
not a safe boundary (a non-quiescent effect recorded while other
effects were in flight, or the very first event of a run).
`run.fork_points()` lists every safe boundary; `force=True` overrides
at your own risk.

### override-error

`OverrideError` (also a `ReplayError`) — a fork override could not
apply: unknown override keys, a response override anchored at an
invocation (or vice versa), or the re-executed agent requested a
different call at the fork point than the override targets (which means
the prefix was not deterministic — verify first).
