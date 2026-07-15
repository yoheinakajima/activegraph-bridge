"""Forking: verified prefix, applied override, divergent live tail."""

from __future__ import annotations

import pytest

from activegraph_bridge import OverrideError, load_run, wrap
from tests.conftest import TOOL_CALLS, make_factory


def test_fork_replaces_second_tool_result(agent, recorded_run):
    """The flagship acceptance test: replacing the second tool result
    preserves the exact recorded prefix and produces a divergent tail."""
    before = dict(TOOL_CALLS)
    fork = recorded_run.fork(
        before=recorded_run.events.tool_call("check_carrier", occurrence=1),
        overrides={"tool_result": {"carrier": "FASTSHIP", "scans": 9}},
    )
    alt = fork.execute()

    # prefix (lookup_order) was served, not re-executed
    assert TOOL_CALLS["lookup"] == before["lookup"]
    assert TOOL_CALLS["carrier"] == before["carrier"]

    # the override changed the answer
    assert "FASTSHIP" in alt["answer"]
    assert "FASTSHIP" not in recorded_run.playback_output()["answer"]

    diff = recorded_run.diff(fork)
    assert diff.shared_effects == 1  # exact verified prefix: lookup_order
    assert diff.outputs_differ
    assert len(diff.child_tail) >= 1

    # fork lineage is first-class ActiveGraph metadata
    child_record = fork.run.record
    assert child_record.parent_run_id == recorded_run.run_id
    assert child_record.forked_at_event_id is not None


def test_fork_override_recorded_as_override(recorded_run):
    fork = recorded_run.fork(
        before=recorded_run.events.tool_call("check_carrier"),
        overrides={"response": {"carrier": "X", "scans": 0}},
    )
    fork.execute()
    child_events = fork.run.raw_events()
    served = [
        e.payload.get("served_from")
        for e in child_events
        if e.type == "effect.responded"
    ]
    assert "override" in served
    assert any(e.type == "bridge.fork_configured" for e in child_events)


def test_fork_before_first_effect_replays_nothing(recorded_run):
    """Forking before the very first tool call: empty prefix, all live."""
    fork = recorded_run.fork(
        before=recorded_run.events.tool_call("lookup_order"),
        overrides={"tool_result": {"status": "shipped", "eta": "2026-07-18"}},
    )
    alt = fork.execute()
    assert "shipped" in alt["answer"].lower() or "Status: shipped" in alt["answer"]
    diff = recorded_run.diff(fork)
    assert diff.shared_effects == 0


def test_fork_with_input_override(recorded_run):
    fork = recorded_run.fork(
        before=recorded_run.events.invocation(1),
        overrides={"input": {"args": [{"order_id": "ord_2", "question": "eta?"}], "kwargs": {}}},
    )
    alt = fork.execute()
    assert "processing" in alt["answer"]
    assert recorded_run.diff(fork).outputs_differ


def test_fork_without_override_is_pure_divergence(recorded_run, monkeypatch):
    """No override: the tail re-executes live against current reality."""
    from tests.conftest import ORDERS

    monkeypatch.setitem(ORDERS, "ord_1", {"status": "shipped", "eta": "2026-07-16"})
    fork = recorded_run.fork(before=recorded_run.events.tool_call("lookup_order"))
    alt = fork.execute()
    assert "shipped" in alt["answer"]


def test_fork_default_anchor_is_first_invocation(recorded_run):
    fork = recorded_run.fork(
        overrides={"input": {"args": [{"order_id": "ord_2", "question": "x"}], "kwargs": {}}}
    )
    alt = fork.execute()
    assert "processing" in alt["answer"]


def test_fork_execute_is_once_only(recorded_run):
    from activegraph_bridge import BridgeConfigurationError

    fork = recorded_run.fork(before=recorded_run.events.tool_call("check_carrier"))
    fork.execute()
    with pytest.raises(BridgeConfigurationError):
        fork.execute()


def test_fork_at_second_invocation_replays_first(agent, store_url):
    with agent.execution(label="multi") as run:
        agent.invoke({"order_id": "ord_1", "question": "one"})
        agent.invoke({"order_id": "ord_2", "question": "two"})

    before = dict(TOOL_CALLS)
    fork = run.fork(
        before=run.events.invocation(2),
        overrides={"input": {"args": [{"order_id": "ord_1", "question": "again?"}], "kwargs": {}}},
    )
    alt = fork.execute()
    # first invocation fully served (agent state reconstructed), second live
    assert alt["turns"] == 2
    assert TOOL_CALLS["lookup"] == before["lookup"] + 1  # only the live tail called it
    assert "Delayed" in alt["answer"]


def test_fork_response_override_requires_effect_anchor(recorded_run):
    with pytest.raises(OverrideError):
        recorded_run.fork(
            before=recorded_run.events.invocation(1),
            overrides={"tool_result": {"nope": True}},
        )


def test_fork_input_override_requires_invocation_anchor(recorded_run):
    with pytest.raises(OverrideError):
        recorded_run.fork(
            before=recorded_run.events.tool_call("lookup_order"),
            overrides={"input": {}},
        )


def test_fork_unknown_override_key_rejected(recorded_run):
    with pytest.raises(OverrideError):
        recorded_run.fork(
            before=recorded_run.events.tool_call("lookup_order"),
            overrides={"tool_output": {}},
        )


def test_fork_points_listed(recorded_run):
    points = recorded_run.fork_points()
    assert len(points) == 3  # invocation start + 2 quiescent tool calls
    assert all(p.forkable for p in points)


def test_memory_store_fork_works():
    agent = wrap(make_factory(), store="memory://forktest")
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1", "question": "?"})
    fork = run.fork(
        before=run.events.tool_call("lookup_order"),
        overrides={"tool_result": {"status": "shipped", "eta": None}},
    )
    alt = fork.execute()
    assert "shipped" in alt["answer"]


def test_forked_child_run_is_loadable(recorded_run, store_url):
    fork = recorded_run.fork(before=recorded_run.events.tool_call("check_carrier"))
    fork.execute()
    child = load_run(store_url, fork.child_run_id)
    assert child.playback_output() is not None
    assert any(r.run_id == fork.child_run_id for r in child.store.list_runs())
