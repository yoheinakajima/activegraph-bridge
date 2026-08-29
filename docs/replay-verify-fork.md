# Replay, playback, verify, fork

"Replay" means too many things. The bridge exposes four distinct
operations so each can promise something exact.

## `run.replay()` — rebuild the projection

ActiveGraph's native replay meaning: fold the event log back into a
graph. Pure and offline — no agent code, no network, works on any run
forever.

```python
replay = run.replay()
replay.graph                       # a native activegraph.Graph
replay.graph.objects(type="tool_call")
replay.graph.relations(type="triggered")
```

## `run.playback_output()` — the recorded answer

Decode the recorded final output of the last (or a chosen) invocation.
Works even for envelope-grade runs; executes nothing.

```python
run.playback_output()
run.playback_output(invocation=2)
```

## `run.verify()` — earn the badge

A **shadow execution**:

1. A clean agent is constructed (factory, or the stateless callable
   itself; a shared instance is refused with `ReconstructionError`).
2. Each recorded invocation is re-driven with its recorded input.
3. Every mediated call the agent makes is matched against the recording
   — event kind, name, and canonical request hash. Matches are served
   from the record. **Nothing executes**: not tools, not SDKs, and not
   sockets (the audit guard aborts any attempt).
4. Recorded failures are re-raised; invocation outputs are compared by
   canonical hash; leftover or missing calls are divergence.

```python
result = run.verify()
result.ok                  # bool; result is truthy on success
result.effects_served      # calls answered from the record
result.divergence          # the ReplayDivergence, when not ok
```

For a recording containing `ainvoke` or `astream`, synchronous code may
still call `run.verify()`; it drives the async execution in a private
event loop. Code already running in an event loop uses the native async
form instead:

```python
result = await run.averify()
```

Success appends a `bridge.verification` event, so `run.report` shows
`boundary-verified` from then on — in any process that loads the run.

Divergence errors carry both sides (`expected` vs `got`) and the exact
recorded event id, plus a structured explanation of the usual causes
(unmediated time/randomness, mutable globals, changed code).

## `run.fork()` — branch history

```python
fork = run.fork(
    before=run.events.tool_call("lookup_order", occurrence=1),   # or an event id
    overrides={"tool_result": {"status": "shipped"}},            # or {"input": ...}
    label="what-if-shipped",
    side_effects="fail_closed",                                  # default
)
alternative = fork.execute()
child = fork.run                    # a full Run: report, replay, verify, fork again
diff = run.diff(fork)
```

Async agent surfaces use `alternative = await fork.aexecute()`. As with
verification, `fork.execute()` remains a convenience for synchronous
callers that are not already inside an event loop.

What `fork()` does at creation time:

1. Resolves `before` to a real event and checks it is a **safe
   boundary** — an invocation start, a quiescent completed effect, or a
   checkpoint. Mid-flight events are refused (`NotForkableError`)
   unless `force=True`.
2. Copies the parent's event prefix into a child run using ActiveGraph's
   native fork primitive — the child has real
   `parent_run_id` / `forked_at_event_id` lineage.
3. Stamps the child with `bridge.fork_configured` (what was changed is
   half the value of a counterfactual).

What `fork.execute()` does:

1. Rebuilds a clean agent (fresh factory; adapters may restore a
   checkpoint instead).
2. Re-drives the parent's invocations. Calls covered by the copied
   prefix are validated and served — the prefix is *proven*, not
   assumed.
3. At the fork point (the cursor running dry), applies the override:
   the response override answers the fork-point call with your value
   (recorded as `served_from="override"`); an input override replaces
   the invocation's input.
4. Everything after runs **live**, recorded into the child, with the
   fork write policy (blocked by default — see
   [side-effects](side-effects.md)).

Multi-turn runs fork naturally: `before=run.events.invocation(2)`
serves turn 1 from the record (reconstructing the agent's conversation
state) and re-runs turns 2+ live with their recorded inputs — the same
conversation script against the divergent state.

### Fork receipts

`fork.execute()` always emits a `bridge.fork_receipt` after a successful
execution. The receipt records:

- the exact copied-prefix and pre-receipt child-log hashes;
- source and target runtime fingerprints;
- every inherited effect descriptor and terminal outcome;
- the request event ids served from the record;
- zero prefix external calls and the separately listed tail executions;
- the target-environment attestation and verifier identity, when supplied.

An unattested fork can establish zero re-execution but remains a Conditional
external continuation because the runtime has no evidence that the target
environment represents the retained prefix. Discharge that premise with a
configured verifier:

```python
from activegraph_bridge import HmacEnvironmentAttestor, verify_fork_receipt

attestor = HmacEnvironmentAttestor(secret, issuer="runtime", key_id="prod-v1")
environment = attestor.issue(
    environment_id="fork-worker-7",
    snapshot_id="snapshot-42",
    claims=fork.environment_claims(),
)
fork.execute(
    target_environment=environment,
    environment_verifier=attestor,
)

receipt = fork.receipt
check = verify_fork_receipt(
    receipt,
    parent_events=run.raw_events(),
    child_events=fork.run.raw_events(),
    environment_verifier=attestor,
)
assert check.ok
assert receipt.external_continuation == "verified"
```

The HMAC key is never written to the log. Its `key_id` and verifier identity
are. Authentication is relative to the caller's configured trust root; it is
not provider attestation. The receipt schema is
[`schemas/fork-receipt-v1.schema.json`](../schemas/fork-receipt-v1.schema.json).
The checked-in
[`evidence/post-oracle-fork-v1`](../evidence/post-oracle-fork-v1/) fixture
exercises this path offline and verifies that a committed recorded oracle was
served without another call.

### Selectors

`run.events` resolves human intent to real event ids:

```python
run.events.tool_call("lookup_order", occurrence=2)
run.events.model_call(1)
run.events.retrieval(1)
run.events.effect("mystery")            # any effect by name
run.events.invocation(2)
run.events.checkpoint(1)
run.fork_points()                       # every safe boundary, in log order
```

Each returns an `EventRef` with `.event_id`, `.forkable`, and
`.resume_strategy` — and raw event ids are accepted anywhere a ref is.

## `run.diff(fork)` — what actually changed

Two layers, one call:

- **Effect level**: the length of the identical effect prefix
  (kind + name + request hash + response hash), and each side's tail.
- **Graph level**: ActiveGraph's native structural `Diff` over the
  projected graphs — divergent objects and relations by id.

Structural diff is deliberately literal; semantic equivalence ("do
these two answers mean the same thing?") belongs to a domain
projector, whose objects make `diff.graph.divergent_objects` speak
your language instead of JSON's.
