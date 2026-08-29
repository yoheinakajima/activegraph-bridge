# Side-effect policy

Reads observe the world; writes change it. A replay that silently
re-sent an email would make "replay" a dangerous word, so the bridge is
**fail-closed**: writes never execute in any serving mode, and fork
tails block them unless you decide otherwise.

Execution policy and replay semantics are separate. Each request now records:

- `footprint`: `pure | idempotent | compensatable | one_shot | unknown`;
- `replay_source`: `deterministic | recorded | uncaptured`;
- lifecycle: `requested`, followed by `committed` or `failed`;
- `observables`: property-scoped external surfaces affected or consulted.

A provider-model call is `side_effect="read"` for mutation policy but
`footprint="one_shot"` for cost/oracle semantics and
`replay_source="recorded"` when its response is captured. This is why a prefix
can serve the result while a new fork-tail model call is blocked by default.

## Declaring

Every effect carries a class:

```python
@bridge_tool(side_effect="read")      # search, DB query, model call
@bridge_tool(side_effect="write")     # send email, update CRM
@bridge_tool                          # side_effect="unknown" — conservative default
```

`unknown` executes during live recording (zero-config still works) but
is treated as a write wherever the distinction matters, and the report
counts it so you can tighten declarations over time.

## The decision table

| Mode | Read | Write / unknown |
| --- | --- | --- |
| Live record | execute + record | `SideEffectPolicy.on_write` — default `execute`; also `approval`, `simulate`, `block` |
| Verify / replay | served from record | **never executes** (not configurable) |
| Fork prefix | served from record | **never executes** (not configurable) |
| Fork tail | execute + record | `SideEffectPolicy.on_fork_write` — default `block`; also `simulate`, `approval`, `execute` |
| Explicitly live fork | execute + record | `run.fork(..., side_effects="live")` — the tail writes like a live recording |

One-shot reads have an additional fail-closed fork-tail policy. To authorize a
new oracle interaction explicitly:

```python
policy = SideEffectPolicy(on_fork_one_shot="execute")
```

This does not authorize writes; `on_fork_write` remains independent.

## Configuring

```python
from activegraph_bridge import SideEffectPolicy, wrap

policy = SideEffectPolicy(
    on_write="approval",                       # live writes need a yes
    approve=lambda kind, name, request: ops.confirm(name, request),
    approved_by="ops@example.com",             # recorded on approved writes
    on_fork_write="simulate",                  # counterfactual writes are simulated
    simulator=lambda kind, name, request: {"sent": False, "simulated": True},
)

agent = wrap(build_agent, policy=policy)
```

Every outcome is in the audit trail: `effect.responded` records
`served_from` (`live` / `approved` / `simulated` / `override`), and a
refusal is a real `effect.failed` event with `served_from="blocked"` —
followed by an `EffectBlockedError` raised into the caller, whose
message walks through the three ways to proceed (simulate, approve,
or authorize live).

## Idempotency keys

If a write tool's signature declares `idempotency_key`, the bridge
injects a deterministic key derived from the run id and the canonical
request — identical requests within a run share a key, which is exactly
what external dedupe APIs want. The key is injected into the *call*,
not the hash, so replay matching is unaffected.

```python
@bridge_tool(side_effect="write")
def charge_card(customer: str, amount_cents: int, idempotency_key: str = "") -> dict:
    return payments.charge(customer, amount_cents, idempotency_key=idempotency_key)
```
