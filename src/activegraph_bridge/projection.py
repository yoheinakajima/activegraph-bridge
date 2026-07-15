"""The AgentExecutionPack: projecting bridge events into a native graph.

Recording produces two intertwined layers in one event log: the bridge
events (the replay substrate) and a graph projection built from them —
ordinary ActiveGraph objects and relations, emitted through
``Graph.add_object`` / ``Graph.add_relation`` with ``caused_by`` pointing
at the bridge event each one materializes.

Because the projection uses native event types, everything ActiveGraph
already knows how to do with a graph works on wrapped-agent runs:
structural diff of parent vs fork, causal chains, the CLI inspector.

Default object types::

    agent  invocation  input  model_call  tool_call  retrieval
    memory_op  external_effect  checkpoint  output  error

Default relations::

    invocation --part_of--> agent
    invocation --used--> input
    call --part_of--> invocation
    parent --triggered--> call          (enclosing effect or invocation)
    previous call --caused--> call      (the causal spine)
    checkpoint --checkpoint_of--> invocation
    invocation --produced--> output | error

Domain projection: pass your own :class:`GraphProjector` to ``wrap()``
to turn generic effects into domain objects (``order_status``,
``policy_claim``, ``recommended_action`` …) — that is where fork diffs
become semantically meaningful instead of structurally exact.
Determinism micro-effects (recorded time/random/uuid reads) stay in the
event log but are not projected; they would be graph noise.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from activegraph.core.event import Event
from activegraph.core.graph import Graph

from . import events as ev

__all__ = ["GraphProjector", "DefaultProjector", "CATEGORY_OBJECT_TYPES"]

CATEGORY_OBJECT_TYPES = {
    "model": "model_call",
    "tool": "tool_call",
    "retrieval": "retrieval",
    "memory": "memory_op",
    "external": "external_effect",
}

_PREVIEW_CHARS = 240


@runtime_checkable
class GraphProjector(Protocol):
    """Folds bridge events into graph objects and relations.

    Called by the session after each bridge event is emitted, in live
    recording modes only (record and fork tails) — replays rebuild the
    projection from the recorded ``object.created`` / ``relation.created``
    events instead of re-projecting.
    """

    def project(self, graph: Graph, event: Event) -> None: ...


def _preview(value: Any) -> str:
    import json

    try:
        text = json.dumps(value, default=str)
    except Exception:
        text = repr(value)
    return text[:_PREVIEW_CHARS]


class DefaultProjector:
    """The generic agent-execution projection described above.

    Stateful per run (it remembers the agent and invocation object ids)
    but self-healing: a fork's tail starts with a fresh projector whose
    lookups fall back to querying the graph, so objects created in the
    copied prefix connect correctly to tail objects.
    """

    def __init__(self) -> None:
        self._agent_obj: str | None = None
        self._invocation_objs: dict[int, str] = {}
        self._effect_objs: dict[str, str] = {}  # request event id -> object id
        self._last_effect_obj: dict[int, str] = {}  # invocation ordinal -> object id
        self._requests: dict[str, Event] = {}  # request event id -> event

    # -- lookups that survive fork tails -----------------------------------

    def _agent(self, graph: Graph) -> str | None:
        if self._agent_obj is None:
            objs = graph.objects(type="agent")
            self._agent_obj = objs[0].id if objs else None
        return self._agent_obj

    def _invocation(self, graph: Graph, ordinal: int | None) -> str | None:
        if ordinal is None:
            return None
        obj_id = self._invocation_objs.get(ordinal)
        if obj_id is None:
            for o in graph.objects(type="invocation"):
                if o.data.get("ordinal") == ordinal:
                    self._invocation_objs[ordinal] = o.id
                    return o.id
        return obj_id

    # -- the fold ------------------------------------------------------------

    def project(self, graph: Graph, event: Event) -> None:
        if event.type == ev.EFFECT_REQUESTED:
            self._requests[event.id] = event  # cached for the response fold
            return
        handler = {
            ev.RUN_STARTED: self._on_run_started,
            ev.INVOCATION_STARTED: self._on_invocation_started,
            ev.EFFECT_RESPONDED: self._on_effect_done,
            ev.EFFECT_FAILED: self._on_effect_done,
            ev.CHECKPOINT_RECORDED: self._on_checkpoint,
            ev.INVOCATION_COMPLETED: self._on_invocation_done,
            ev.INVOCATION_FAILED: self._on_invocation_done,
        }.get(event.type)
        if handler is not None:
            handler(graph, event)

    def _on_run_started(self, graph: Graph, event: Event) -> None:
        p = event.payload
        obj = graph.add_object(
            "agent",
            {
                "adapter": p.get("adapter"),
                "target": p.get("target"),
                "reconstruction": p.get("reconstruction"),
                "code_hash": (p.get("fingerprint") or {}).get("agent_code_hash"),
            },
            actor="bridge",
            caused_by=event.id,
        )
        self._agent_obj = obj.id

    def _on_invocation_started(self, graph: Graph, event: Event) -> None:
        p = event.payload
        ordinal = int(p.get("ordinal", 1))
        inv = graph.add_object(
            "invocation",
            {
                "ordinal": ordinal,
                "method": p.get("method"),
                "input_hash": p.get("input_hash"),
                "status": "running",
            },
            actor="bridge",
            caused_by=event.id,
        )
        self._invocation_objs[ordinal] = inv.id
        agent = self._agent(graph)
        if agent:
            graph.add_relation(inv.id, agent, "part_of", actor="bridge", caused_by=event.id)
        input_obj = graph.add_object(
            "input",
            {"hash": p.get("input_hash"), "preview": _preview(p.get("input"))},
            actor="bridge",
            caused_by=event.id,
        )
        graph.add_relation(inv.id, input_obj.id, "used", actor="bridge", caused_by=event.id)

    def _on_effect_done(self, graph: Graph, event: Event) -> None:
        request_id = event.caused_by
        if request_id is None:
            return
        request = self._requests.pop(request_id, None)
        if request is None:  # tail-resume: the request may sit in the prefix
            request = next(
                (e for e in graph.events if e.id == request_id), None
            )
        if request is None:
            return
        rp = request.payload
        category = str(rp.get("category", "external"))
        if category == "determinism":
            return  # recorded, servable, but not graph-worthy
        failed = event.type == ev.EFFECT_FAILED
        obj = graph.add_object(
            CATEGORY_OBJECT_TYPES.get(category, "external_effect"),
            {
                "kind": rp.get("kind"),
                "name": rp.get("name"),
                "side_effect": rp.get("side_effect"),
                "request_hash": rp.get("request_hash"),
                "status": "failed" if failed else "ok",
                "served_from": event.payload.get("served_from", "live"),
                "latency_seconds": event.payload.get("latency_seconds", 0.0),
                "request_preview": _preview(rp.get("request")),
                "response_preview": _preview(
                    event.payload.get("error") if failed else event.payload.get("response")
                ),
            },
            actor="bridge",
            caused_by=event.id,
        )
        self._effect_objs[request_id] = obj.id
        inv_ordinal = rp.get("invocation")
        inv_obj = self._invocation(graph, inv_ordinal)
        if inv_obj:
            graph.add_relation(obj.id, inv_obj, "part_of", actor="bridge", caused_by=event.id)
        # triggered: the enclosing effect if this call was nested, else the invocation
        parent_obj = self._effect_objs.get(request.caused_by or "") or inv_obj
        if parent_obj and parent_obj != obj.id:
            graph.add_relation(parent_obj, obj.id, "triggered", actor="bridge", caused_by=event.id)
        # caused: the sequential spine within the invocation
        if isinstance(inv_ordinal, int):
            prev = self._last_effect_obj.get(inv_ordinal)
            if prev and prev != obj.id:
                graph.add_relation(prev, obj.id, "caused", actor="bridge", caused_by=event.id)
            self._last_effect_obj[inv_ordinal] = obj.id

    def _on_checkpoint(self, graph: Graph, event: Event) -> None:
        p = event.payload
        obj = graph.add_object(
            "checkpoint",
            {
                "ordinal": p.get("ordinal"),
                "label": p.get("label"),
                "state_hash": p.get("state_hash"),
            },
            actor="bridge",
            caused_by=event.id,
        )
        inv_obj = self._invocation(graph, p.get("invocation"))
        if inv_obj:
            graph.add_relation(obj.id, inv_obj, "checkpoint_of", actor="bridge", caused_by=event.id)

    def _on_invocation_done(self, graph: Graph, event: Event) -> None:
        p = event.payload
        ordinal = int(p.get("ordinal", 1))
        failed = event.type == ev.INVOCATION_FAILED
        inv_obj = self._invocation(graph, ordinal)
        if failed:
            err = graph.add_object(
                "error",
                {
                    "class": (p.get("error") or {}).get("class"),
                    "message": (p.get("error") or {}).get("message"),
                },
                actor="bridge",
                caused_by=event.id,
            )
            if inv_obj:
                graph.add_relation(inv_obj, err.id, "produced", actor="bridge", caused_by=event.id)
                graph.patch_object(inv_obj, {"status": "failed"}, actor="bridge", caused_by=event.id)
            return
        out = graph.add_object(
            "output",
            {"hash": p.get("output_hash"), "preview": _preview(p.get("output"))},
            actor="bridge",
            caused_by=event.id,
        )
        if inv_obj:
            graph.add_relation(inv_obj, out.id, "produced", actor="bridge", caused_by=event.id)
            graph.patch_object(inv_obj, {"status": "completed"}, actor="bridge", caused_by=event.id)
