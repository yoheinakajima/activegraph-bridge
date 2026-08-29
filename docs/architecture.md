# Architecture

The bridge is an **execution membrane**: the wrapped agent stays in
charge of its own control flow, and everything nondeterministic or
externally observable that it does passes through one mediation point.
This document walks the layers from the outside in.

```
wrap(build_agent) ──► WrappedAgent            (the proxy you invoke)
                        │
                        ▼
                  ExecutionSession            (context-local membrane; 4 modes)
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
        effect broker        invocation boundaries
   (session.effect/aeffect)  (begin/finish_invocation)
              │                    │
              ▼                    ▼
     ActiveGraph event log  +  graph projection      (one store, one run)
```

## The pieces

### `WrappedAgent` (`wrapper.py`)

A transparent proxy preserving the original invocation surface —
`invoke`, `ainvoke`, `stream`, `astream`, `batch`, plain call — plus
`execution()` for grouping invocations into one run, and `last_run` /
`runs()` for finding what was recorded. Unknown attributes pass through
to the target.

Invocation routing is one rule: **if a session is active in this
context, join it; otherwise open a fresh run.** That single rule gives
you execution blocks, standalone one-shot runs, and nested wrapped
agents (which record as nested invocations of the outer run) without
any special cases in agent code.

### `ExecutionSession` (`session.py`)

The context-local object (a `contextvars` binding, so it survives
`await` and is isolated between threads/tasks) that answers every
mediated call. It runs in one of four modes:

| mode | boundaries | effects | writes |
| --- | --- | --- | --- |
| `record` | emitted | execute live + record | per live policy |
| `shadow` (verify) | validated against the script | served from the script | never |
| `prefix` (fork, before the fork point) | validated | served | never |
| `tail` (fork, after the fork point) | emitted | execute live + record | per fork policy |

A `prefix` session **transitions to `tail` in place** the moment its
cursor runs dry — that moment *is* the fork point. Agent code cannot
tell which mode it is running in; that is the whole point.

### The effect broker (`ExecutionSession.effect` / `aeffect`)

The one primitive everything else is sugar over:

```python
response = session.effect(
    kind="openai.responses.create",
    request=canonical_request,
    execute=lambda: client.responses.create(**request),
    side_effect="read",
    codec=my_codec,          # optional; AutoCodec by default
)
```

Record path: canonicalize → hash → append `effect.requested` → execute
(subject to the write policy) → encode → append `effect.responded` /
`effect.failed` → **return the live value** (rehydration is a replay
concern; live agents always see exactly what the SDK returned).

Serve path: match the request against the recorded script by
`(kind, name, canonical-hash)` → decode and return the recorded
response, or re-raise the recorded failure. Any mismatch raises
`ReplayDivergence` carrying both sides.

`bridge_tool`, `instrument.wrap_client`, and `det.*` are all thin
wrappers around this broker — which is why they all inherit record,
verify, fork, and fail-closed semantics for free.

### Scripts and cursors (`script.py`)

A shadow execution walks a **script** — the recorded run parsed into
ordered invocation boundaries, effects, and checkpoints — through a
**cursor** that serves and validates. Matching is `strict`
(position + content) or `auto` (content-exact, position may flex within
the current invocation — tolerating concurrent completion reordering
without ever inventing a response).

### The engine (`engine.py`)

One machine, three shapes:

- `record_invoke` / `arecord_invoke` — drive one invocation against any
  session, teeing sync or async stream chunks into the envelope.
- `shadow_verify` / `ashadow_verify` — fresh agent, recorded inputs,
  full-script consumption check, `VerificationResult`.
- `fork_execute` / `afork_execute` — prefix cursor over the child's
  inherited events, override application, live tail into the child store.

Successful forks append a hash-bound `bridge.fork_receipt`. It records the
copied prefix, source and target fingerprints, inherited effect descriptors,
served request identities, zero prefix external calls, tail executions, and
the result of target-environment verification. Receipt verification is a pure
operation over the parent events, child events, receipt, and configured trust
root.

Verification and forking **re-execute real agent code**. That is
deliberate: re-execution is what reconstructs hidden Python state
(message lists, framework internals) that no log can carry — strategy
#1 (fresh re-execution) of the state model. Adapters can add strategy
#2 (framework checkpoints); native ActiveGraph agents are strategy #3.

### Stores (`_store.py`)

Runs live in ordinary ActiveGraph event stores. SQLite is the default
(multi-run files, durable, forkable via the native
`SQLiteEventStore.fork_run` prefix copy, CLI-inspectable);
`memory://` mirrors the same semantics in-process for tests. Postgres
is refused up front rather than shipping with a broken `fork()`.

### Projection (`projection.py`)

During live recording, `DefaultProjector` folds bridge events into
native graph objects and relations (the AgentExecutionPack: `agent`,
`invocation`, `input`, `model_call`, `tool_call`, `retrieval`,
`external_effect`, `checkpoint`, `output`, `error`). Because these are
ordinary `object.created` / `relation.created` events, replaying the
log rebuilds the graph with zero bridge code involved, and ActiveGraph's
native `compute_diff` powers `run.diff(fork)`.

Determinism micro-effects (recorded `det.now()` reads and friends) stay
in the event log but are not projected — servable, not graph noise.

### Honesty layer (`report.py`, `determinism.py`)

`compute_report(events)` is a pure function of the log — grade,
captured counts, fork points, blockers, fingerprint. Hazard detection
is a process-wide `sys.addaudithook` watching `socket.connect`,
`subprocess.Popen`, and `os.system`: recording sessions log a
`hazard.detected` blocker with file:line; serving sessions **abort the
operation** (`UnrecordedEffectError`) — the fail-closed teeth behind
"strict verification makes zero live calls".

## Extension points

- **Adapters** (`adapters/`): the `AgentAdapter` protocol —
  `supports` / `invoke` / `ainvoke` / `instrument` / `checkpoint` /
  `restore` + an `AdapterCapabilities` declaration. Register via
  `register_adapter()` or the `activegraph_bridge.adapters` entry-point
  group. An adapter must not claim `can_short_circuit_calls` unless its
  instrumentation can *prevent* live calls during replay — capabilities
  are the honesty contract.
- **Codecs** (`codecs.py`): `EffectCodec` for integrations with exact
  wire types. `AutoCodec` covers JSON, Pydantic, dataclasses, bytes,
  datetimes, Decimals, exceptions — and falls back to `repr` *loudly*
  (a `lossy-capture` finding, never a silent lie). Decoded stand-ins
  (`AttrBox`) are round-trip stable: re-encoding yields the original
  recorded document, so served responses can flow into later requests
  without spurious divergence.
- **Projectors** (`projection.py`): implement `GraphProjector.project`
  to turn generic effects into domain objects (`order_status`,
  `policy_claim`, `recommended_action`) — where fork diffs become
  semantically meaningful.
- **Policies** (`policy.py`): simulation and approval handlers for
  writes, per live-run and per fork-tail.

## Design commitments

1. **The wrapper mediates effects; it does not merely observe them.**
2. **Every run reports its actual replayability level.** The grade is
   computed from the log, never asserted by configuration.
3. **Factory construction and checkpoints are first-class**, because
   hidden agent state is the hard part of forking.
