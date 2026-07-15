# Integration grades

The bridge's core honesty rule: **a run's guarantees are computed from
its event log, never asserted by configuration.** `run.report` is a pure
function of the recorded events — the same log yields the same report
today, in CI, and in a different process next year.

```python
print(run.report)
# Replayability: boundary-verified
# Captured:      7 model calls, 4 tool calls, 2 retrieval calls
# Fork points:   11
# Blockers:      none
```

## The ladder

| Grade | Earned when | Honest guarantee |
| --- | --- | --- |
| `observed` | The run couldn't even capture a faithful envelope (output recorded as `repr()` only), or never completed an invocation | Inspection and lineage only |
| `envelope` | Input and output captured faithfully, but re-execution can't be promised — a hazard fired, or the agent was wrapped as a live instance | `playback_output()`; fork before the invocation |
| `boundary` | Every boundary call individually mediated **and** a clean-agent strategy exists (factory or stateless callable) **and** no blockers | Re-execution on recorded effects; forks at effect boundaries |
| `checkpointed` | Boundary + recorded checkpoints + an adapter that can restore them | Forks resume from snapshots instead of re-executing the whole prefix |
| `native` | Agent state and coordination live in ActiveGraph itself | Full native replay/fork/diff. Reserved — the bridge never grants itself this grade |

`boundary` and `checkpointed` display as *unverified* until a
`verify()` succeeds; the recorded `bridge.verification` event upgrades
the label to `boundary-verified` durably.

A zero-effect run with a factory grades `boundary` — a pure function of
its input is re-executable by construction, and `verify()` proves it
(the hazard guard catches hidden I/O; output comparison catches hidden
state).

## Blockers and findings

Findings come in three severities:

- **blocker** — caps the grade (unrecorded I/O with file:line; live
  instance without reset; lossy envelope capture)
- **warning** — honesty notes that survive (a lossy *effect* response:
  playback fine, faithful re-serve impossible for that value)
- **note** — advisory (`side_effect="unknown"` counts; out-of-order
  matches during verification)

```text
Replayability: playback-only (envelope)
Captured:      2 tool calls
Fork points:   1
Blockers:
  - unrecorded socket from custom_search.py:81
  - agent supplied as a live instance; no reset or checkpoint strategy
```

## Raising your grade

| Symptom in the report | Fix |
| --- | --- |
| `agent supplied as a live instance…` | Wrap a factory: `wrap(build_agent)` |
| `unrecorded socket from x.py:41` | Route the call through `instrument.wrap_client`, a `@bridge_tool`, or `effect(...)` |
| `…captured as repr() only` | Register a faithful `EffectCodec` for that effect kind, or return encodable types |
| verification diverges on hashes with no code change | Replace direct `time`/`random`/`uuid` reads with `det.now()` / `det.random()` / `det.uuid4()` |
| `N effect(s) with side_effect='unknown'` | Declare `side_effect="read"` or `"write"` on the tool/effect |

## The fingerprint

Every run records the environment it was captured in:

```json
{
  "python_version": "3.11.15",
  "activegraph_version": "1.10.0",
  "bridge_version": "0.1.0",
  "agent_code_hash": "sha256-of-the-agent's-source-file",
  "configuration_hash": "sha256-of-the-wrap-configuration"
}
```

Code changes never *prohibit* verify or fork — a changed-code
counterfactual is one of the most useful forks there is. The
fingerprint exists so the log distinguishes same-code strict replay
from changed-code, changed-configuration, and changed-input
counterfactuals.
