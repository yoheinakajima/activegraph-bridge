"""Hazard detection and grade honesty.

A run is never described as more replayable than it proved itself to
be: unrecorded I/O caps the grade and blocks verification; instance
wrapping caps at envelope; lossy captures are reported.
"""

from __future__ import annotations

import socket
import subprocess

from activegraph_bridge import Grade, UnrecordedEffectError, wrap
from activegraph_bridge.testing import ScriptedModel
from tests.conftest import SupportAgent, lookup_order, make_factory


def _touch_network() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # connect() on UDP does no handshake — no packets, but the
        # socket.connect audit event fires exactly like TCP.
        s.connect(("127.0.0.1", 9))
    finally:
        s.close()


def test_direct_socket_downgrades_recording(store_url):
    """An uninstrumented network call prevents the 'verified' label."""

    def build():
        class Leaky:
            def invoke(self, payload):
                order = lookup_order(payload["order_id"])
                _touch_network()  # bypasses the membrane
                return {"status": order["status"]}

        return Leaky()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1"})

    report = run.report
    assert report.grade is Grade.ENVELOPE  # capped: no boundary claim
    assert any(f.code == "unrecorded-io" for f in report.blockers)
    assert "playback-only" in str(report)
    events = run.raw_events()
    assert any(e.type == "hazard.detected" for e in events)
    # playback still works — honesty, not punishment
    assert run.playback_output()["status"] == "delayed"


def test_direct_socket_fails_verification_closed(store_url):
    """During verify, the socket is aborted before it opens."""
    leak = {"on": False}

    def build():
        class Sometimes:
            def invoke(self, payload):
                order = lookup_order(payload["order_id"])
                if leak["on"]:
                    _touch_network()
                return {"status": order["status"]}

        return Sometimes()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1"})  # clean recording
    assert run.report.grade is Grade.BOUNDARY

    leak["on"] = True  # the re-executed code now reaches for the network
    result = run.verify()
    assert not result.ok
    assert isinstance(result.divergence, UnrecordedEffectError)


def test_subprocess_detected(store_url):
    def build():
        class Sheller:
            def invoke(self, payload):
                subprocess.run(["true"], check=False)
                return {"ok": True}

        return Sheller()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        agent.invoke({})
    assert any(f.code == "unrecorded-io" for f in run.report.blockers)


def test_instance_wrapping_caps_at_envelope(store_url):
    """No factory, mutable singleton: playback yes, replay claims no."""
    instance = SupportAgent(ScriptedModel(["a"]))
    agent = wrap(instance, store=store_url)
    agent.invoke({"order_id": "ord_1", "question": "?"})
    report = agent.last_run.report
    assert report.grade is Grade.ENVELOPE
    assert report.reconstruction == "shared_instance"
    assert any(f.code == "no-reconstruction" for f in report.blockers)
    assert "playback-only" in str(report)


def test_pure_factory_agent_is_boundary_by_construction(store_url):
    """Zero effects + factory + clean guard: re-executable by construction."""

    def build():
        class Pure:
            def invoke(self, payload):
                return {"double": payload["x"] * 2}

        return Pure()

    agent = wrap(build, store=store_url)
    agent.invoke({"x": 21})
    run = agent.last_run
    assert run.report.grade is Grade.BOUNDARY
    assert run.verify().ok
    assert run.report.verified


def test_lossy_envelope_capture_is_observed_grade(store_url):
    """An output that can only be repr()'d cannot even claim playback."""

    class Socketish:  # no model_dump/to_dict/dataclass — repr only
        def __repr__(self) -> str:
            return "<Socketish 0x1>"

    def build():
        class Opaque:
            def invoke(self, payload):
                return Socketish()

        return Opaque()

    agent = wrap(build, store=store_url)
    agent.invoke({})
    report = agent.last_run.report
    assert report.grade is Grade.OBSERVED
    assert "inspection-only" in str(report)
    assert any(f.code == "lossy-envelope" for f in report.findings)


def test_grade_survives_reload(store_url):
    from activegraph_bridge import load_run

    agent = wrap(make_factory(), store=store_url)
    with agent.execution() as run:
        agent.invoke({"order_id": "ord_1", "question": "?"})
    run.verify()
    reloaded = load_run(store_url, run.run_id)
    assert reloaded.report.grade is Grade.BOUNDARY
    assert reloaded.report.verified
