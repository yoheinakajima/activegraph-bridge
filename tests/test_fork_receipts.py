"""Hard effect-boundary forks with portable zero-reexecution receipts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from activegraph_bridge import (
    BridgeConfigurationError,
    EffectBlockedError,
    HmacEnvironmentAttestor,
    SideEffectPolicy,
    bridge_tool,
    instrument,
    verify_fork_receipt,
    wrap,
)


@bridge_tool(
    side_effect="read",
    footprint="pure",
    replay_source="recorded",
    observables=("agent.decision",),
)
def apply_policy(decision: str) -> dict[str, str]:
    return {"decision": decision}


class _Responses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return {"output_text": f"approve:{kwargs['input']}"}


class _OpenAI:
    def __init__(self) -> None:
        self.responses = _Responses()


def _agent(raw: _OpenAI, store_url: str):
    def build():
        client = instrument.openai(raw)

        class Agent:
            def invoke(self, payload):
                response = client.responses.create(
                    model="fixture-oracle-v1", input=payload["case"]
                )
                return apply_policy(response["output_text"])

        return Agent()

    return wrap(build, store=store_url)


def _attestor() -> HmacEnvironmentAttestor:
    return HmacEnvironmentAttestor(
        b"activegraph-bridge-public-conformance-key-v1",
        issuer="activegraph-bridge-conformance",
        key_id="fixture-v1",
    )


def test_post_oracle_fork_receipt_proves_zero_reexecution(store_url):
    raw = _OpenAI()
    agent = _agent(raw, store_url)
    original = agent.invoke({"case": "vendor-7"})
    run = agent.last_run
    assert original == {"decision": "approve:vendor-7"}
    assert raw.responses.calls == 1

    model_request = run.events.model_call(1)
    request_event = run.events[model_request.event_id]
    assert request_event.payload["footprint"] == "one_shot"
    assert request_event.payload["replay_source"] == "recorded"
    assert request_event.payload["observables"] == [
        "provider.cost",
        "provider.oracle",
    ]

    fork = run.fork(
        before=run.events.tool_call("apply_policy"),
        overrides={"tool_result": {"decision": "reject:vendor-7"}},
        label="post-oracle-zero-reexecution",
    )
    attestor = _attestor()
    environment = attestor.issue(
        environment_id="fixture-clean-room",
        snapshot_id="fixture-snapshot-v1",
        claims=fork.environment_claims(),
        issued_at="2026-08-28T00:00:00Z",
    )

    before = raw.responses.calls
    alternative = fork.execute(
        target_environment=environment,
        environment_verifier=attestor,
    )
    assert alternative == {"decision": "reject:vendor-7"}
    assert raw.responses.calls == before

    receipt = fork.receipt
    assert receipt is not None
    assert receipt.environment_verified
    assert receipt.zero_reexecution_verified
    assert receipt.external_continuation == "verified"
    assert receipt.prefix_external_calls == 0
    assert receipt.served_effect_request_ids == [model_request.event_id]
    assert receipt.inherited_effects[0]["footprint"] == "one_shot"
    assert receipt.inherited_effects[0]["lifecycle"] == "committed"
    assert fork.run.fork_receipts == [receipt]

    verification = verify_fork_receipt(
        receipt,
        parent_events=run.raw_events(),
        child_events=fork.run.raw_events(),
        environment_verifier=attestor,
    )
    assert verification.ok, verification.errors


def test_environment_attestation_is_bound_to_exact_fork(store_url):
    raw = _OpenAI()
    agent = _agent(raw, store_url)
    agent.invoke({"case": "vendor-8"})
    run = agent.last_run
    fork = run.fork(
        before=run.events.tool_call("apply_policy"),
        overrides={"tool_result": {"decision": "reject:vendor-8"}},
    )
    attestor = _attestor()
    environment = attestor.issue(
        environment_id="fixture-clean-room",
        snapshot_id="fixture-snapshot-v1",
        claims=fork.environment_claims(),
        issued_at="2026-08-28T00:00:00Z",
    )
    tampered = replace(
        environment,
        claims=environment.claims | {"prefix_hash": "0" * 64},
    )
    with pytest.raises(BridgeConfigurationError, match="signature"):
        fork.execute(
            target_environment=tampered,
            environment_verifier=attestor,
        )
    assert raw.responses.calls == 1


def test_valid_attestation_for_another_fork_is_rejected(store_url):
    raw = _OpenAI()
    agent = _agent(raw, store_url)
    agent.invoke({"case": "vendor-8b"})
    run = agent.last_run
    first = run.fork(
        before=run.events.tool_call("apply_policy"),
        overrides={"tool_result": {"decision": "reject:first"}},
    )
    second = run.fork(
        before=run.events.tool_call("apply_policy"),
        overrides={"tool_result": {"decision": "reject:second"}},
    )
    attestor = _attestor()
    environment = attestor.issue(
        environment_id="fixture-clean-room",
        snapshot_id="fixture-snapshot-v1",
        claims=first.environment_claims(),
        issued_at="2026-08-28T00:00:00Z",
    )
    with pytest.raises(BridgeConfigurationError, match="does not bind"):
        second.execute(
            target_environment=environment,
            environment_verifier=attestor,
        )
    assert raw.responses.calls == 1


def test_fork_tail_oracle_call_is_blocked_without_explicit_policy(store_url):
    raw = _OpenAI()
    agent = _agent(raw, store_url)
    agent.invoke({"case": "vendor-9"})
    run = agent.last_run
    fork = run.fork(before=run.events.model_call(1))

    with pytest.raises(EffectBlockedError):
        fork.execute()
    assert raw.responses.calls == 1
    with pytest.raises(BridgeConfigurationError, match="already executed"):
        fork.execute()


def test_fork_tail_oracle_call_requires_explicit_policy(store_url):
    raw = _OpenAI()

    def build():
        client = instrument.openai(raw)

        class Agent:
            def invoke(self, payload):
                return client.responses.create(
                    model="fixture-oracle-v1", input=payload["case"]
                )

        return Agent()

    agent = wrap(
        build,
        store=store_url,
        policy=SideEffectPolicy(on_fork_one_shot="execute"),
    )
    agent.invoke({"case": "vendor-9b"})
    fork = agent.last_run.fork(before=agent.last_run.events.model_call(1))
    fork.execute()
    assert raw.responses.calls == 2


def test_unattested_receipt_remains_conditional(store_url):
    raw = _OpenAI()
    agent = _agent(raw, store_url)
    agent.invoke({"case": "vendor-10"})
    run = agent.last_run
    fork = run.fork(
        before=run.events.tool_call("apply_policy"),
        overrides={"tool_result": {"decision": "reject:vendor-10"}},
    )
    fork.execute()
    assert fork.receipt is not None
    assert fork.receipt.zero_reexecution_verified
    assert fork.receipt.external_continuation == "conditional"
