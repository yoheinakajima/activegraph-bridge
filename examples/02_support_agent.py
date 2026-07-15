"""The customer-support scenario: multi-tool, write effects, counterfactuals.

Demonstrates the full policy story:
- read tools served on replay,
- a write tool (send_email) that executes live, is SERVED during verify,
  and is BLOCKED in a fork tail until simulated or authorized,
- a fork that replaces one recorded tool result and changes the outcome.

Runs completely offline:  python examples/02_support_agent.py
"""

from __future__ import annotations

import os
import tempfile

from activegraph_bridge import EffectBlockedError, SideEffectPolicy, bridge_tool, det, wrap
from activegraph_bridge.testing import ScriptedModel

STORE = f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'support.db')}"

ORDERS = {"ord_1": {"status": "delayed", "eta": "2026-07-25"}}
OUTBOX: list[dict] = []


@bridge_tool(side_effect="read")
def lookup_order(order_id: str) -> dict:
    return dict(ORDERS[order_id])


@bridge_tool(side_effect="write")
def send_apology(to: str, body: str) -> dict:
    OUTBOX.append({"to": to, "body": body})
    return {"sent": True}


class SupportAgent:
    def __init__(self, model: ScriptedModel):
        self.model = model

    def invoke(self, payload: dict) -> dict:
        order = lookup_order(payload["order_id"])
        tone = self.model.create(order=order)
        checked_at = det.now().isoformat()  # recorded, replays exactly
        if order["status"] == "delayed":
            answer = f"[{tone}] Your order is delayed — new ETA {order['eta']}."
            send_apology(payload["email"], answer)
        else:
            answer = f"[{tone}] Your order is {order['status']}."
        return {"answer": answer, "checked_at": checked_at}


def build_agent() -> SupportAgent:
    return SupportAgent(ScriptedModel(["empathetic"]))


def main() -> None:
    agent = wrap(build_agent, store=STORE)

    with agent.execution(label="ticket-4812", metadata={"customer_id": "cust_123"}) as run:
        answer = agent.invoke({"order_id": "ord_1", "question": "why late?", "email": "amy@example.com"})
    print("live:", answer)
    print("emails actually sent:", len(OUTBOX))

    # ---- verify: the apology email is SERVED, not re-sent ----------------
    result = run.verify()
    print()
    print(result)
    print("emails after verify:", len(OUTBOX), "(unchanged)")

    # ---- fork: what if the order had shipped? -----------------------------
    fork = run.fork(
        before=run.events.tool_call("lookup_order", occurrence=1),
        overrides={"tool_result": {"status": "shipped", "eta": None}},
    )
    alternative = fork.execute()
    print()
    print("counterfactual:", alternative)
    print("emails after fork:", len(OUTBOX), "(the shipped path never writes)")
    print()
    print(run.diff(fork))

    # ---- fork tails fail closed on writes ---------------------------------
    # This counterfactual KEEPS the delayed status (a different ETA), so the
    # re-executed agent tries to send the apology again — and is refused.
    fork2 = run.fork(
        before=run.events.tool_call("lookup_order", occurrence=1),
        overrides={"tool_result": {"status": "delayed", "eta": "2026-08-01"}},
    )
    try:
        fork2.execute()
    except EffectBlockedError as e:
        print()
        print("fork tail write blocked, as it should be:")
        print("   ", str(e).splitlines()[0])
    print("emails after blocked fork:", len(OUTBOX), "(still unchanged)")

    # ---- opt in to simulation for counterfactual writes -------------------
    simulating = wrap(
        build_agent,
        store=STORE,
        policy=SideEffectPolicy(
            on_fork_write="simulate",
            simulator=lambda kind, name, request: {"sent": False, "simulated": True},
        ),
    )
    with simulating.execution(label="ticket-4813") as run3:
        simulating.invoke({"order_id": "ord_1", "question": "?", "email": "amy@example.com"})
    fork3 = run3.fork(
        before=run3.events.tool_call("lookup_order", occurrence=1),
        overrides={"tool_result": {"status": "delayed", "eta": "2026-08-01"}},
    )
    outbox_before_fork3 = len(OUTBOX)
    alt3 = fork3.execute()
    print()
    print("simulated-write counterfactual:", alt3["answer"])
    print(
        f"emails after simulated fork: {len(OUTBOX)} "
        f"(unchanged from {outbox_before_fork3} — the write was simulated)"
    )


if __name__ == "__main__":
    main()
