"""Verification: strict shadow execution with zero live calls.

These are the uncompromising acceptance tests: verification must make
no model/tool/network calls, must fail on any divergence, and must
never claim a badge it did not earn.
"""

from __future__ import annotations

import pytest

from activegraph_bridge import Grade, ReplayDivergence, load_run, wrap
from activegraph_bridge import det
from activegraph_bridge.testing import ScriptedModel, assert_verified
from tests.conftest import TOOL_CALLS, lookup_order, make_factory, ORDERS


def test_verify_makes_zero_live_calls(recorded_run, monkeypatch):
    """The tool bodies must not execute during verification — even when
    the live data has changed underneath them."""
    monkeypatch.setitem(ORDERS, "ord_1", {"status": "CHANGED", "eta": "now"})
    before = dict(TOOL_CALLS)

    result = recorded_run.verify()

    assert result.ok, result.divergence
    assert result.effects_served == 2
    assert TOOL_CALLS == before  # not one live tool execution


def test_verify_upgrades_report_to_verified(recorded_run):
    assert_verified(recorded_run)
    report = recorded_run.report
    assert report.grade is Grade.BOUNDARY
    assert report.verified
    assert report.label == "boundary-verified"


def test_verify_persists_across_processes(recorded_run, store_url):
    recorded_run.verify()
    fresh = load_run(store_url, recorded_run.run_id)
    assert fresh.report.verified  # read back from the log, no spec needed


def test_verify_detects_changed_agent_code(store_url):
    agent = wrap(make_factory(), store=store_url)
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1", "question": "?"})

    # "Deploy" a code change: the agent now asks for a different order.
    class Tampered:
        def __init__(self, inner):
            self.inner = inner

        def invoke(self, payload):
            return self.inner.invoke({**payload, "order_id": "ord_2"})

    result = run.verify(agent=Tampered(make_factory()()))
    assert not result.ok
    assert isinstance(result.divergence, ReplayDivergence)
    assert "lookup_order" in str(result.divergence)


def test_verify_detects_output_divergence_from_unmediated_state(store_url):
    """Every effect matches, but hidden global state changes the output."""
    counter = {"n": 0}

    def build():
        class Sneaky:
            def invoke(self, payload):
                order = lookup_order(payload["order_id"])
                counter["n"] += 1  # unmediated, survives across executions
                return {"status": order["status"], "n": counter["n"]}

        return Sneaky()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1", "question": "?"})

    result = run.verify()
    assert not result.ok
    assert "output does not match" in str(result.divergence)


def test_verify_serves_recorded_failures(store_url):
    """A recorded tool failure replays as the same exception class."""

    def build():
        class Fragile:
            def invoke(self, payload):
                try:
                    lookup_order("missing")
                except KeyError:
                    return {"handled": True}
                return {"handled": False}

        return Fragile()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        out = agent.invoke({})
    assert out == {"handled": True}
    assert_verified(run)


def test_verify_with_det_sources(store_url):
    """time/random/uuid reads are recorded effects and replay exactly."""

    def build():
        class Timestamped:
            def invoke(self, payload):
                order = lookup_order(payload["order_id"])
                return {
                    "status": order["status"],
                    "at": det.now().isoformat(),
                    "jitter": det.random(),
                    "id": str(det.uuid4()),
                }

        return Timestamped()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        first = agent.invoke({"order_id": "ord_1"})
    assert_verified(run)
    # and playback returns the recorded values
    assert run.playback_output() == first


def test_verify_requires_reconstruction_strategy(store_url):
    """An instance-wrapped agent cannot verify (no clean rebuild)."""
    from activegraph_bridge import ReconstructionError
    from tests.conftest import SupportAgent

    instance = SupportAgent(ScriptedModel(["a"]))
    agent = wrap(instance, store=store_url)
    agent.invoke({"order_id": "ord_1", "question": "?"})
    run = agent.last_run
    with pytest.raises(ReconstructionError):
        run.verify()
    # but an explicitly supplied clean agent works
    result = run.verify(agent=SupportAgent(ScriptedModel(["a"])))
    assert result.ok, result.divergence


def test_verify_missing_calls_detected(store_url):
    """Re-execution that skips a recorded call fails verification."""
    flag = {"skip": False}

    def build():
        class Skipper:
            def invoke(self, payload):
                order = lookup_order(payload["order_id"])
                if not flag["skip"]:
                    lookup_order(payload["order_id"])  # second call
                return {"status": order["status"]}

        return Skipper()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1"})
    flag["skip"] = True
    result = run.verify()
    assert not result.ok
    assert "before making every recorded call" in str(result.divergence)


def test_verify_extra_calls_detected(store_url):
    flag = {"extra": False}

    def build():
        class Chatty:
            def invoke(self, payload):
                order = lookup_order(payload["order_id"])
                if flag["extra"]:
                    lookup_order(payload["order_id"])
                return {"status": order["status"]}

        return Chatty()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1"})
    flag["extra"] = True
    result = run.verify()
    assert not result.ok


def test_loaded_run_verifies_after_attach(recorded_run, store_url):
    loaded = load_run(store_url, recorded_run.run_id).attach(make_factory())
    assert loaded.verify().ok
