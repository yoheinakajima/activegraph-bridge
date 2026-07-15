"""Instrumenting an SDK client — no monkeypatching, no framework.

`instrument.wrap_client` proxies exactly the method paths you name; the
recorded responses (Pydantic models included) rehydrate on replay so
idiomatic `resp.choices[0].message.content` code works with zero live
calls. The same `instrument.openai(...)` / `instrument.anthropic(...)`
presets wrap real SDK clients identically.

Runs completely offline:  python examples/03_instrumented_sdk.py
"""

from __future__ import annotations

import os
import tempfile

from activegraph_bridge import instrument, wrap

STORE = f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'sdk.db')}"


# A stand-in with the OpenAI client's shape (swap in OpenAI() for real).
class Completions:
    def __init__(self) -> None:
        self.live_calls = 0

    def create(self, **kwargs):
        self.live_calls += 1
        user = kwargs["messages"][-1]["content"]
        return {
            "id": f"chatcmpl-{self.live_calls}",
            "choices": [{"message": {"role": "assistant", "content": f"echo: {user}"}}],
        }


class Chat:
    def __init__(self) -> None:
        self.completions = Completions()


class FakeOpenAI:
    def __init__(self) -> None:
        self.chat = Chat()


raw_client = FakeOpenAI()


def build_agent():
    client = instrument.openai(raw_client)  # scoped proxy over this instance

    class ChatAgent:
        def invoke(self, payload: dict) -> dict:
            resp = client.chat.completions.create(
                model="gpt-x",
                messages=[{"role": "user", "content": payload["prompt"]}],
            )
            return {"reply": resp["choices"][0]["message"]["content"]}

    return ChatAgent()


def main() -> None:
    agent = wrap(build_agent, store=STORE)

    with agent.execution(label="sdk-demo") as run:
        out = agent.invoke({"prompt": "hello membrane"})
    print("live:", out, f"(SDK calls: {raw_client.chat.completions.live_calls})")

    result = run.verify()
    print()
    print(result)
    print(f"SDK calls after verify: {raw_client.chat.completions.live_calls} (unchanged)")

    print()
    print(run.report)
    print()
    for ref in run.events.effects(category="model"):
        print("model call:", ref)


if __name__ == "__main__":
    main()
