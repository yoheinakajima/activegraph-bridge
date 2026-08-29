# activegraph-bridge

**Wrap any existing agent in an [ActiveGraph](https://pypi.org/project/activegraph/) execution membrane — and get recorded effects, honest replay, offline verification, and forkable history without rewriting the agent.**

```python
from activegraph_bridge import wrap

agent = wrap(build_agent, store="sqlite:///agent-runs.db")

with agent.execution(label="customer-support") as run:
    answer = agent.invoke({"messages": [{"role": "user", "content": "Why was my order delayed?"}]})

run.verify()     # fresh agent, recorded effects, ZERO live calls
fork = run.fork(
    before=run.events.tool_call("lookup_order", occurrence=1),
    overrides={"tool_result": {"status": "shipped", "estimated_delivery": "2026-07-18"}},
)
alternative_answer = fork.execute()
print(run.diff(fork))
```

For a post-effect continuation claim, bind the target environment and retain
the emitted receipt:

```python
from activegraph_bridge import HmacEnvironmentAttestor, verify_fork_receipt

fork = run.fork(before=run.events.tool_call("next_step"))
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
assert receipt.zero_reexecution_verified
assert receipt.external_continuation == "verified"
```

The receipt hash-binds the copied prefix and child log, names every inherited
effect served from the record, records the process-produced prefix-call counter,
and carries a signature-checked, fork-bound caller assertion about the target
environment. The bridge verifies the assertion's origin and binding, not the
truth of its contents; deployments are responsible for ensuring the configured
issuer validates an environment before issuing it. Without such an assertion
the bridge still emits a receipt, but its continuation verdict remains
`conditional`.

Your agent keeps its own orchestration, prompts, loops, and framework
control flow. The bridge mediates what the agent *does* — model calls,
tool calls, retrieval, time, randomness, identifiers, external writes,
inputs and outputs — and lands all of it in an ordinary ActiveGraph
event log, where the log is authoritative and the graph is a projection.

## Why a membrane, not a tracing decorator

A callback listener can produce a good trace. It cannot honestly offer
**executable replay**, because it cannot stop the original model or tool
call from happening again and return the recorded response instead.

The bridge is built the other way around: every mediated call flows
through a recorded-effect broker that can answer it three ways —

- **live** (record mode): execute for real, capture request + response
- **served** (verify / fork prefix): return the recorded response;
  the SDK, the tool body, and the network are never touched
- **overridden** (fork point): return the counterfactual you asked for

That single mechanism is what makes `verify()` and `fork()` real
operations instead of marketing.

## Install

```bash
pip install activegraph-bridge        # pulls in activegraph
```

Python ≥ 3.11, same as ActiveGraph. The core has no other dependencies.

## Sixty seconds, end to end

```python
from activegraph_bridge import bridge_tool, instrument, wrap

@bridge_tool(side_effect="read")
def lookup_order(order_id: str) -> dict:
    return orders_db.get(order_id)

@bridge_tool(side_effect="write")
def send_email(to: str, body: str) -> dict:
    return mailer.send(to, body)

class SupportAgent:
    def __init__(self, client): ...
    def invoke(self, payload): ...          # calls lookup_order(), the model, maybe send_email()

def build_agent():                          # a factory: how to make a CLEAN agent
    return SupportAgent(instrument.openai(OpenAI()))

agent = wrap(build_agent, store="sqlite:///agent-runs.db")

with agent.execution(label="customer-support", metadata={"customer_id": "cust_123"}) as run:
    answer = agent.invoke({"order_id": "ord_1", "question": "Why was my order delayed?"})
```

Now the run is a durable, inspectable, *executable* artifact:

```python
print(run.report)
# Replayability: boundary (unverified)
# Captured:      1 model call, 2 tool calls
# Fork points:   4
# Blockers:      none

run.playback_output()      # the recorded answer, no execution at all
run.replay()               # rebuild the ActiveGraph graph projection from the log
run.verify()               # shadow-execute a fresh agent on recorded effects — zero live calls
print(run.report)
# Replayability: boundary-verified
# ...
```

And history becomes something you can branch:

```python
fork = run.fork(
    before=run.events.tool_call("lookup_order", occurrence=1),
    overrides={"tool_result": {"status": "shipped", "estimated_delivery": "2026-07-18"}},
)
alternative_answer = fork.execute()   # prefix served from the record, tail runs live
diff = run.diff(fork)                 # effect-level + native graph diff
```

During `fork.execute()` the write tool `send_email` **cannot fire** —
fork tails fail closed on writes unless you simulate, approve, or pass
`side_effects="live"` explicitly.

Provider-model calls are recorded as `one_shot + recorded`: their response can
be served during a prefix, but a new model call in a fork tail is blocked by
default. Set `SideEffectPolicy(on_fork_one_shot="execute")` only when the new
oracle interaction and cost are explicitly authorized.

## The four operations, and what each honestly promises

| Operation | What happens | Executes agent code? | Live calls? |
| --- | --- | --- | --- |
| `run.replay()` | Rebuild the graph projection from the event log (ActiveGraph's native replay) | no | none |
| `run.playback_output()` | Decode the recorded final output | no | none |
| `run.verify()` | Fresh agent re-executes against served effects; every request compared by canonical hash; output compared | yes | **none — enforced** |
| `fork.execute()` | Re-execute to the fork point on served effects, apply the override, record a live divergent tail | yes | tail only |

## Integration grades: guarantees you earned, not claimed

Every run computes its grade from what was actually captured. The report
never describes an observed run as replayable.

| Grade | What the wrapper captured | Honest guarantee |
| --- | --- | --- |
| `observed` | Not even a faithful envelope (e.g. output captured as `repr`) | Inspection and lineage only |
| `envelope` | The invocation's input and output | Playback; fork before the invocation |
| `boundary` | Every model/tool/retrieval/external call, plus a clean-agent strategy | Re-execution on recorded effects; forks at effect boundaries. `boundary-verified` once `verify()` passes |
| `checkpointed` | Boundary + restorable agent state | Resume forks from snapshots (adapter-provided) |
| `native` | Agent state lives in ActiveGraph itself | Full native replay/fork/diff (reserved for native integrations) |

What moves a run *up* the ladder:

- **Pass a factory** (`wrap(build_agent)`), not a live instance — replay
  needs a clean agent. Instances still work, at envelope grade, and the
  report says exactly why.
- **Declare tools** with `@bridge_tool(side_effect="read"|"write")`.
- **Instrument SDK clients**: `instrument.openai(client)`,
  `instrument.anthropic(client)`, or `instrument.wrap_client(anything,
  methods=[...])` — scoped proxies, no monkeypatching.
- **Route nondeterminism through `det`**: `det.now()`, `det.random()`,
  `det.uuid4()` are recorded effects with real-value fallback outside
  the bridge.
- **Wrap anything else** in `effect(kind, request, execute, side_effect=...)`.

What moves it *down* — and gets reported with file:line:

- direct sockets, subprocesses, `os.system` (detected by a process
  audit hook; **fatal during verification**, a visible blocker during
  recording)
- values that can only be captured as `repr()`
- wrapping a mutable instance with no reset strategy

```text
Replayability: playback-only (envelope)
Captured:      2 tool calls
Fork points:   1
Blockers:
  - unrecorded socket from custom_search.py:81
  - agent supplied as a live instance; no reset or checkpoint strategy
```

## Side effects fail closed

Effects retain the backward-compatible execution policy
`side_effect="read" | "write" | "unknown"`, and additionally record three
semantic dimensions: external footprint, replay source, and lifecycle. They
also carry observable tags for property-scoped world claims.

| Mode | Read | Write |
| --- | --- | --- |
| Live record | execute + record | execute per policy (`execute` / `approval` / `simulate` / `block`) |
| Verify / replay | served from record | **never executes** |
| Fork prefix | served from record | **never executes** |
| Fork tail | execute + record, except `one_shot` reads such as model calls are blocked by default | **blocked** by default; `simulate` / `approval` via policy; live only with `side_effects="live"` |

Writes record idempotency keys, approval identity, and whether they
actually committed. A blocked write is an `effect.failed` event in the
audit trail *and* an `EffectBlockedError` in your face.

## One log, two layers — and native ActiveGraph tooling works

A bridge run is a standard ActiveGraph event store (SQLite by default;
`memory://` for tests). Two layers live in the same log:

1. **Bridge events** — the replay substrate:
   `invocation.started/completed`, `effect.requested/responded/failed`,
   `checkpoint.recorded`, `hazard.detected`, `bridge.verification`.
2. **The AgentExecutionPack projection** — ordinary `object.created` /
   `relation.created` events emitted through `Graph.add_object`:
   `agent`, `invocation`, `input`, `model_call`, `tool_call`,
   `retrieval`, `external_effect`, `checkpoint`, `output`, `error`,
   related by `part_of`, `triggered`, `caused`, `used`, `produced`,
   `checkpoint_of`.

Because the projection uses native event types, ActiveGraph's own
machinery applies unmodified: `run.diff(fork)` uses the native
structural `Diff`, forks are real `fork_run` rows with
`parent_run_id`/`forked_at_event_id` lineage, and the ActiveGraph CLI
can inspect bridge stores.

Want domain-level diffs instead of structural ones? Ship a projector:

```python
agent = wrap(build_agent, projector=SupportCaseProjector())   # your GraphProjector
```

## Growing beyond one agent

- **Nested agents**: a wrapped agent invoked inside another wrapped
  agent's run records as a nested invocation in the same causal log —
  and is rebuilt fresh during that run's verification.
- **Multi-turn**: several `invoke()`s inside one `execution()` block
  share the agent instance; verify/fork re-drive the whole conversation
  to reconstruct its hidden state. Forks at `run.events.invocation(n)`
  replay turns 1..n-1 from the record.
- **Async and streams**: `ainvoke`/`astream` mirror the sync surface;
  streamed invocations record their chunk sequence and assembled output.
  In async application code, use `await run.averify()` and
  `await fork.aexecute()`; the synchronous helpers automatically drive
  async recordings only when no event loop is already running.

```python
answer = await agent.ainvoke(payload)
chunks = [chunk async for chunk in agent.astream(payload)]
assert (await agent.last_run.averify()).ok

fork = agent.last_run.fork(before=agent.last_run.events.model_call(1))
alternative = await fork.aexecute()
```
- **Adapters**: framework-specific knowledge (middleware hooks,
  checkpoint/restore) lives behind the small `AgentAdapter` protocol and
  the `activegraph_bridge.adapters` entry-point group, so integrations
  version independently of the core. The built-in generic adapter
  drives any callable or `invoke`-style object.

## Loading runs later

```python
from activegraph_bridge import load_run, list_runs

list_runs("sqlite:///agent-runs.db")                # RunRecords with fork lineage
run = load_run("sqlite:///agent-runs.db", run_id)   # inspect / playback / replay
run.attach(build_agent).verify()                    # re-execution needs the agent back
```

Every run records a fingerprint (Python version, package versions, agent
code hash, configuration hash), so a same-code strict replay is
distinguishable from a changed-code counterfactual — code changes are a
*reason* to fork, and the log says which kind of fork you ran.

## Documentation

- [Architecture](docs/architecture.md) — the membrane, session modes, the effect broker
- [Replay, verify, fork](docs/replay-verify-fork.md) — exact semantics of each operation
- [Integration grades](docs/grades.md) — how grades are computed, raising yours
- [Side-effect policy](docs/side-effects.md) — fail-closed writes, simulation, approval
- [Determinism](docs/determinism.md) — hazard detection, deterministic sources, fingerprints
- [Fork receipts](docs/replay-verify-fork.md#fork-receipts) — environment binding and zero-reexecution evidence
- [Error catalog](docs/errors.md)

Runnable examples (offline, no API keys) live in [`examples/`](examples/).

## Status

Unreleased v0.2 candidate — the core semantics and portable receipt path,
checked by an uncompromising test suite
(strict verification makes zero live calls; forks preserve verified
prefixes; unrecorded I/O blocks the verified badge; writes cannot fire
during replay or normal forks; inherited one-shot effects are served from the
record; target environments and child logs are hash-bound in fork receipts;
fresh-process verification reproduces output and graph). Framework adapters
(LangGraph, CrewAI, …) and Postgres stores are the next ring outward — see
[docs/architecture.md](docs/architecture.md#extension-points).

The checked-in [post-oracle fixture](evidence/post-oracle-fork-v1/) is an
offline, language-neutral witness: one committed recorded oracle interaction,
one actual child fork, and a generator-observed absence of any additional
fixture-oracle call in the child.

## License

Apache-2.0
