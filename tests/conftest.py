"""Shared fixtures: a deterministic multi-tool support agent.

The agent is deliberately shaped like real integrations: a stateful
class holding conversation history, a fake model client rebuilt by the
factory, one read tool, one write tool, and control flow that depends on
tool results (so overriding a result genuinely changes the path).
"""

from __future__ import annotations

import pytest

from activegraph_bridge import bridge_tool, wrap
from activegraph_bridge.testing import ScriptedModel

ORDERS = {
    "ord_1": {"status": "delayed", "eta": "2026-07-25"},
    "ord_2": {"status": "processing", "eta": None},
}

EMAIL_OUTBOX: list[dict] = []

# Live-execution counters: incremented only when a tool body actually runs.
# The zero-live-calls acceptance tests assert these stay frozen during
# verify() and fork prefixes.
TOOL_CALLS = {"lookup": 0, "carrier": 0, "email": 0}


@bridge_tool(side_effect="read")
def lookup_order(order_id: str) -> dict:
    TOOL_CALLS["lookup"] += 1
    return dict(ORDERS[order_id])


@bridge_tool(side_effect="read")
def check_carrier(order_id: str) -> dict:
    TOOL_CALLS["carrier"] += 1
    return {"carrier": "ACME", "scans": 3}


@bridge_tool(side_effect="write")
def send_email(to: str, body: str) -> dict:
    TOOL_CALLS["email"] += 1
    EMAIL_OUTBOX.append({"to": to, "body": body})
    return {"sent": True, "n": len(EMAIL_OUTBOX)}


class SupportAgent:
    """Two read tools, a model 'decision', a conditional write."""

    def __init__(self, model: ScriptedModel, notify: bool = False):
        self.model = model
        self.notify = notify
        self.history: list = []

    def invoke(self, payload: dict) -> dict:
        self.history.append(payload)
        order = lookup_order(payload["order_id"])
        carrier = check_carrier(payload["order_id"])
        tone = self.model.create(order=order, question=payload["question"])
        if order["status"] == "delayed":
            answer = f"[{tone}] Delayed via {carrier['carrier']}, ETA {order['eta']}."
            if self.notify:
                send_email(payload.get("email", "user@example.com"), answer)
        else:
            answer = f"[{tone}] Status: {order['status']} via {carrier['carrier']}."
        return {"answer": answer, "turns": len(self.history)}


def make_factory(*, notify: bool = False, tones: list[str] | None = None):
    def build_agent():
        return SupportAgent(ScriptedModel(list(tones or ["empathetic", "cheerful"])), notify=notify)

    return build_agent


@pytest.fixture()
def store_url(tmp_path):
    return f"sqlite:///{tmp_path}/runs.db"


@pytest.fixture()
def agent(store_url):
    return wrap(make_factory(), store=store_url)


@pytest.fixture()
def recorded_run(agent):
    with agent.execution(label="support") as run:
        agent.invoke({"order_id": "ord_1", "question": "where is my order?"})
    return run


@pytest.fixture(autouse=True)
def clean_outbox():
    EMAIL_OUTBOX.clear()
    yield
    EMAIL_OUTBOX.clear()
