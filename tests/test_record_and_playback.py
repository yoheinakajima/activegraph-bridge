"""Recording, playback, replay projection, and run listings."""

from __future__ import annotations

import pytest

from activegraph_bridge import Grade, list_runs, load_run, wrap
from activegraph_bridge import events as ev_mod
from tests.conftest import SupportAgent, make_factory
from activegraph_bridge.testing import ScriptedModel


def test_invocation_surface_preserved(agent):
    out = agent.invoke({"order_id": "ord_1", "question": "hi"})
    assert out["answer"].startswith("[empathetic]")
    assert out["turns"] == 1


def test_playback_output_matches_live_output(recorded_run):
    assert recorded_run.playback_output()["answer"].startswith("[empathetic]")


def test_event_log_contains_effect_pairs(recorded_run):
    events = recorded_run.raw_events()
    types = [e.type for e in events]
    assert types[0] == "bridge.run_started"
    assert types.count("effect.requested") == 2  # lookup_order + check_carrier
    assert types.count("effect.responded") == 2
    assert "invocation.started" in types and "invocation.completed" in types
    # responses are causally linked to their requests
    pairs = ev_mod.effect_pairs(events)
    assert all(resp is not None and resp.caused_by == req.id for req, resp in pairs)


def test_replay_rebuilds_native_projection(recorded_run):
    replay = recorded_run.replay()
    graph = replay.graph
    assert graph.objects(type="agent")
    assert graph.objects(type="invocation")
    assert len(graph.objects(type="tool_call")) == 2
    assert graph.objects(type="output")
    # relations from the AgentExecutionPack
    kinds = {r.type for r in graph.all_relations()}
    assert {"part_of", "used", "produced", "triggered"} <= kinds


def test_report_boundary_unverified_before_verify(recorded_run):
    report = recorded_run.report
    assert report.grade is Grade.BOUNDARY
    assert not report.verified
    assert "boundary (unverified)" in str(report)
    assert report.effect_counts.get("tool") == 2
    assert report.fork_points >= 3  # invocation + two quiescent tool calls


def test_runs_are_listed_with_labels(agent, store_url):
    with agent.execution(label="support") as run:
        agent.invoke({"order_id": "ord_1", "question": "?"})
    records = list_runs(store_url)
    assert any(r.run_id == run.run_id and r.label == "support" for r in records)


def test_load_run_from_disk_without_spec(recorded_run, store_url):
    loaded = load_run(store_url, recorded_run.run_id)
    assert loaded.playback_output() == recorded_run.playback_output()
    assert loaded.report.grade is Grade.BOUNDARY


def test_standalone_invoke_records_its_own_run(agent):
    out = agent.invoke({"order_id": "ord_2", "question": "eta?"})
    run = agent.last_run
    assert run is not None
    assert run.playback_output() == out


def test_multiple_invocations_in_one_execution(agent):
    with agent.execution(label="thread") as run:
        agent.invoke({"order_id": "ord_1", "question": "one"})
        out2 = agent.invoke({"order_id": "ord_2", "question": "two"})
    assert out2["turns"] == 2  # same agent instance across the block
    assert run.playback_output()["turns"] == 2
    assert run.playback_output(invocation=1)["turns"] == 1
    assert run.report.invocations == 2


def test_memory_store_backend():
    agent = wrap(make_factory(), store="memory://test-backend")
    out = agent.invoke({"order_id": "ord_1", "question": "?"})
    assert agent.last_run.playback_output() == out


def test_batch_records_one_invocation_each(agent):
    with agent.execution() as run:
        outs = agent.batch(
            [
                {"order_id": "ord_1", "question": "a"},
                {"order_id": "ord_2", "question": "b"},
            ]
        )
    assert [o["turns"] for o in outs] == [1, 2]
    assert run.report.invocations == 2


def test_failed_invocation_recorded(store_url):
    def build():
        return SupportAgent(ScriptedModel(["x"]))

    agent = wrap(build, store=store_url)
    with pytest.raises(KeyError):
        agent.invoke({"order_id": "missing", "question": "?"})
    events = agent.last_run.raw_events()
    assert any(e.type == "invocation.failed" for e in events)
    assert any(e.type == "effect.failed" for e in events)  # lookup_order raised
