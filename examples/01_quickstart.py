"""Quickstart: wrap a plain-Python agent, record, verify, fork.

Runs completely offline — no API keys, no network. The "model" is a
deterministic fake; the point is the membrane around it.

    python examples/01_quickstart.py
"""

from __future__ import annotations

import os
import tempfile

from activegraph_bridge import bridge_tool, wrap
from activegraph_bridge.testing import ScriptedModel

STORE = f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'quickstart.db')}"

INVENTORY = {"blue-widget": 3, "red-widget": 0}


@bridge_tool(side_effect="read")
def check_stock(sku: str) -> dict:
    return {"sku": sku, "available": INVENTORY.get(sku, 0)}


class ShopAgent:
    """An 'existing agent': its own control flow, a model, a tool."""

    def __init__(self, model: ScriptedModel):
        self.model = model

    def invoke(self, payload: dict) -> dict:
        stock = check_stock(payload["sku"])
        tone = self.model.create(question=payload["question"], stock=stock)
        if stock["available"] > 0:
            return {"reply": f"[{tone}] Yes — {stock['available']} in stock."}
        return {"reply": f"[{tone}] Sorry, out of stock."}


def build_agent() -> ShopAgent:  # the factory: how to make a CLEAN agent
    return ShopAgent(ScriptedModel(["friendly"]))


def main() -> None:
    agent = wrap(build_agent, store=STORE)

    # ---- record ---------------------------------------------------------
    with agent.execution(label="quickstart") as run:
        answer = agent.invoke({"sku": "blue-widget", "question": "any left?"})
    print("live answer: ", answer)

    # ---- inspect --------------------------------------------------------
    print()
    print(run.report)
    print()
    print("playback:    ", run.playback_output())
    print("replay:      ", run.replay())

    # ---- verify: fresh agent, recorded effects, zero live calls ---------
    result = run.verify()
    print()
    print(result)
    print()
    print(run.report)

    # ---- fork: what if the stock lookup had said 0? ----------------------
    fork = run.fork(
        before=run.events.tool_call("check_stock", occurrence=1),
        overrides={"tool_result": {"sku": "blue-widget", "available": 0}},
    )
    alternative = fork.execute()
    print()
    print("fork answer: ", alternative)
    print()
    print(run.diff(fork))


if __name__ == "__main__":
    main()
