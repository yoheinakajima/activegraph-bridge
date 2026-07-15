"""Write effects fail closed: replay, verify, and fork tails never write."""

from __future__ import annotations

import pytest

from activegraph_bridge import EffectBlockedError, SideEffectPolicy, wrap
from tests.conftest import EMAIL_OUTBOX, make_factory


@pytest.fixture()
def notifying_agent(store_url):
    return wrap(make_factory(notify=True), store=store_url)


def test_write_executes_and_records_in_live_run(notifying_agent):
    with notifying_agent.execution() as run:
        notifying_agent.invoke({"order_id": "ord_1", "question": "?", "email": "a@b.c"})
    assert len(EMAIL_OUTBOX) == 1
    events = run.raw_events()
    write_requests = [
        e
        for e in events
        if e.type == "effect.requested" and e.payload.get("side_effect") == "write"
    ]
    assert len(write_requests) == 1


def test_write_never_executes_during_verify(notifying_agent):
    with notifying_agent.execution() as run:
        notifying_agent.invoke({"order_id": "ord_1", "question": "?", "email": "a@b.c"})
    outbox_after_record = len(EMAIL_OUTBOX)

    result = run.verify()
    assert result.ok, result.divergence
    assert len(EMAIL_OUTBOX) == outbox_after_record  # served, not sent


def test_write_blocked_in_fork_tail_by_default(notifying_agent):
    """A fork tail cannot send the email unless explicitly authorized."""
    with notifying_agent.execution() as run:
        notifying_agent.invoke({"order_id": "ord_1", "question": "?", "email": "a@b.c"})
    sent_live = len(EMAIL_OUTBOX)

    fork = run.fork(
        before=run.events.tool_call("check_carrier"),
        overrides={"tool_result": {"carrier": "SLOW", "scans": 1}},
    )
    # the agent's write attempt inside the tail raises EffectBlockedError,
    # which propagates through the agent (it doesn't catch it)
    with pytest.raises(EffectBlockedError):
        fork.execute()
    assert len(EMAIL_OUTBOX) == sent_live  # nothing was sent

    # and the refusal is in the child's audit trail
    child_events = fork.run.raw_events()
    assert any(
        e.type == "effect.failed" and e.payload.get("served_from") == "blocked"
        for e in child_events
    )


def test_fork_tail_write_can_be_simulated(store_url):
    policy = SideEffectPolicy(
        on_fork_write="simulate",
        simulator=lambda kind, name, request: {"sent": False, "simulated": True},
    )
    agent = wrap(make_factory(notify=True), store=store_url, policy=policy)
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1", "question": "?", "email": "a@b.c"})
    sent_live = len(EMAIL_OUTBOX)

    fork = run.fork(
        before=run.events.tool_call("check_carrier"),
        overrides={"tool_result": {"carrier": "SLOW", "scans": 1}},
    )
    alt = fork.execute()
    assert alt is not None
    assert len(EMAIL_OUTBOX) == sent_live
    child_events = fork.run.raw_events()
    assert any(
        e.payload.get("served_from") == "simulated"
        for e in child_events
        if e.type == "effect.responded"
    )


def test_fork_tail_write_with_explicit_live_authorization(notifying_agent):
    with notifying_agent.execution() as run:
        notifying_agent.invoke({"order_id": "ord_1", "question": "?", "email": "a@b.c"})
    sent_live = len(EMAIL_OUTBOX)

    fork = run.fork(
        before=run.events.tool_call("check_carrier"),
        overrides={"tool_result": {"carrier": "SLOW", "scans": 1}},
        side_effects="live",
    )
    fork.execute()
    assert len(EMAIL_OUTBOX) == sent_live + 1  # explicitly authorized


def test_unknown_side_effect_treated_as_write_in_fork(store_url):
    from activegraph_bridge import bridge_tool

    ledger = []

    @bridge_tool  # side_effect defaults to "unknown"
    def mystery(x: int) -> int:
        ledger.append(x)
        return x * 2

    def build():
        class A:
            def invoke(self, payload):
                first = lookup(payload)
                return {"v": mystery(first)}

        def lookup(payload):
            return payload["x"]

        return A()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        agent.invoke({"x": 21})
    n = len(ledger)

    fork = run.fork(before=run.events.effect("mystery"))
    with pytest.raises(EffectBlockedError):
        fork.execute()
    assert len(ledger) == n  # the conservative default held


def test_live_write_policy_approval(store_url):
    decisions = []
    policy = SideEffectPolicy(
        on_write="approval",
        approve=lambda kind, name, request: decisions.append(name) or (name == "send_email"),
        approved_by="ops@example.com",
    )
    agent = wrap(make_factory(notify=True), store=store_url, policy=policy)
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1", "question": "?", "email": "a@b.c"})
    assert decisions == ["send_email"]
    assert len(EMAIL_OUTBOX) == 1
    responded = [
        e for e in run.raw_events() if e.type == "effect.responded"
        and e.payload.get("served_from") == "approved"
    ]
    assert responded and responded[0].payload.get("approved_by") == "ops@example.com"
