"""The membrane's primitives: effect(), instrumented clients, det, codecs,
nested agents, async, streaming, checkpoints, and the decorator form."""

from __future__ import annotations

import dataclasses

from activegraph_bridge import (
    AttrBox,
    checkpoint,
    effect,
    instrument,
    recorded_agent,
    wrap,
)
from activegraph_bridge.codecs import AutoCodec
from activegraph_bridge.testing import assert_verified
from tests.conftest import lookup_order, make_factory


def test_effect_passthrough_outside_session():
    assert effect("misc.compute", {"x": 1}, lambda: 42) == 42


def test_bridge_tool_passthrough_outside_session():
    assert lookup_order("ord_1")["status"] == "delayed"


def test_wrap_client_records_and_serves(store_url):
    class FakeSDK:
        class chat:  # noqa: N801 - mimics SDK attribute shape
            def __init__(self, outer):
                self.completions = outer

        def __init__(self):
            self.chat = FakeSDK.chat(self)
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return {"id": f"resp_{self.calls}", "echo": kwargs["prompt"]}

    sdk = FakeSDK()

    def build():
        client = instrument.wrap_client(
            sdk, methods=["chat.completions.create"], kind_prefix="fake"
        )

        class Agent:
            def invoke(self, payload):
                r = client.chat.completions.create(prompt=payload["q"])
                return {"reply": r["echo"], "id": r["id"]}

        return Agent()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        out = agent.invoke({"q": "hello"})
    assert out == {"reply": "hello", "id": "resp_1"}
    assert sdk.calls == 1

    assert run.verify().ok
    assert sdk.calls == 1  # verification hit the recording, not the SDK

    refs = run.events.effects(category="model")
    assert refs and refs[0].kind == "fake.chat.completions.create"


def test_pydantic_response_roundtrip(store_url):
    from pydantic import BaseModel

    class Choice(BaseModel):
        text: str

    class Completion(BaseModel):
        id: str
        choices: list[Choice]

    class SDK:
        def create(self, **kwargs):
            return Completion(id="c1", choices=[Choice(text="hi " + kwargs["q"])])

    def build():
        client = instrument.wrap_client(SDK(), methods=["create"], kind_prefix="pyd")

        class Agent:
            def invoke(self, payload):
                resp = client.create(q=payload["q"])
                return {"text": resp.choices[0].text}  # idiomatic SDK access

        return Agent()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        out = agent.invoke({"q": "there"})
    assert out == {"text": "hi there"}
    assert_verified(run)


def test_attrbox_preserves_idiomatic_access_and_roundtrips():
    codec = AutoCodec()
    doc = {
        "$bridge": "model",
        "class": "not_importable_module:Nope",
        "data": {"choices": [{"message": {"content": "hey"}}], "usage": {"t": 3}},
    }
    box = codec.decode_response(doc)
    assert isinstance(box, AttrBox)
    assert box.choices[0].message.content == "hey"
    assert box.usage.t == 3
    # round-trip stability: re-encoding yields the original document
    assert codec.encode_response(box) == doc


@dataclasses.dataclass
class Point:
    x: int
    y: int


def test_dataclass_roundtrip():
    codec = AutoCodec()
    encoded = codec.encode_response(Point(1, 2))
    decoded = codec.decode_response(encoded)
    assert decoded == Point(1, 2)  # importable class rehydrates faithfully


def test_local_class_falls_back_to_attrbox():
    codec = AutoCodec()

    @dataclasses.dataclass
    class Local:  # not importable at replay time
        x: int

    decoded = codec.decode_response(codec.encode_response(Local(5)))
    assert isinstance(decoded, AttrBox)
    assert decoded.x == 5  # attribute access still works


def test_openai_preset_shapes(store_url):
    """The openai() preset intercepts the documented method paths on a
    duck-typed client, sync and async."""

    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kw):
            self.calls += 1
            return {"choices": [{"message": {"content": kw["messages"][-1]["content"].upper()}}]}

    class Chat:
        def __init__(self):
            self.completions = Completions()

    class FakeOpenAI:
        def __init__(self):
            self.chat = Chat()
            self.api_key = "sk-fake"  # untouched attributes pass through

    raw = FakeOpenAI()

    def build():
        client = instrument.openai(raw)
        assert client.api_key == "sk-fake"

        class Agent:
            def invoke(self, payload):
                r = client.chat.completions.create(
                    model="gpt-x", messages=[{"role": "user", "content": payload["q"]}]
                )
                return {"a": r["choices"][0]["message"]["content"]}

        return Agent()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        out = agent.invoke({"q": "hello"})
    assert out == {"a": "HELLO"}
    assert raw.chat.completions.calls == 1
    assert run.verify().ok
    assert raw.chat.completions.calls == 1  # zero live calls during verify
    assert run.events.model_call(1).kind == "openai.chat.completions.create"


def test_nested_wrapped_agents_share_one_run(store_url):
    inner = wrap(make_factory(), store=store_url)

    def build_outer():
        class Router:
            def invoke(self, payload):
                sub = inner.invoke({"order_id": payload["order_id"], "question": "sub"})
                return {"routed": sub["answer"]}

        return Router()

    outer = wrap(build_outer, store=store_url)
    with outer.execution(label="routed") as run:
        out = outer.invoke({"order_id": "ord_1"})
    assert "Delayed" in out["routed"]

    events = run.raw_events()
    started = [e for e in events if e.type == "invocation.started"]
    assert len(started) == 2  # outer + nested, one causal log
    assert {e.payload.get("depth") for e in started} == {0, 1}

    assert run.verify().ok  # nested agent rebuilt fresh and served too


async def test_async_agent_roundtrip(store_url):
    def build():
        class AsyncAgent:
            async def ainvoke(self, payload):
                order = lookup_order(payload["order_id"])
                return {"status": order["status"]}

        return AsyncAgent()

    agent = wrap(build, store=store_url)
    out = await agent.ainvoke({"order_id": "ord_1"})
    assert out == {"status": "delayed"}
    assert agent.last_run.playback_output() == out


def test_stream_recorded_and_played_back(store_url):
    def build():
        class Streamer:
            def stream(self, payload):
                order = lookup_order(payload["order_id"])
                for word in ("your", "order", "is", order["status"]):
                    yield word

        return Streamer()

    agent = wrap(build, store=store_url)
    chunks = list(agent.stream({"order_id": "ord_1"}))
    assert chunks == ["your", "order", "is", "delayed"]
    recorded = agent.last_run.playback_output()
    assert recorded == {"stream": True, "chunks": chunks}
    assert agent.last_run.verify().ok


def test_checkpoint_recorded(store_url):
    def build():
        class Saver:
            def __init__(self):
                self.notes = []

            def invoke(self, payload):
                self.notes.append(payload["q"])
                checkpoint({"notes": list(self.notes)}, label="after-note")
                return {"n": len(self.notes)}

        return Saver()

    agent = wrap(build, store=store_url)
    with agent.execution() as run:
        agent.invoke({"q": "a"})
    events = run.raw_events()
    cps = [e for e in events if e.type == "checkpoint.recorded"]
    assert len(cps) == 1 and cps[0].payload["label"] == "after-note"
    assert run.report.checkpoints == 1
    assert run.verify().ok  # checkpoints are replay no-ops


def test_recorded_agent_decorator(store_url):
    @recorded_agent(store=store_url)
    def run_agent(payload):
        order = lookup_order(payload["order_id"])
        return {"status": order["status"]}

    out = run_agent({"order_id": "ord_2"})
    assert out == {"status": "processing"}
    assert run_agent.last_run is not None
    assert run_agent.last_run.playback_output() == out
    # the decorated callable is its own agent — verify re-executes it
    assert run_agent.last_run.verify().ok


def test_concurrent_executions_are_isolated(store_url):
    import threading

    agent = wrap(make_factory(), store=store_url)
    results: dict[str, str] = {}

    def worker(name: str, order: str) -> None:
        with agent.execution(label=name) as run:
            agent.invoke({"order_id": order, "question": name})
            results[name] = run.run_id

    threads = [
        threading.Thread(target=worker, args=("t1", "ord_1")),
        threading.Thread(target=worker, args=("t2", "ord_2")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results.values())) == 2  # two separate runs, no bleed
