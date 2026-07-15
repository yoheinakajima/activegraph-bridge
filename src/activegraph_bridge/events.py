"""The bridge's event vocabulary.

A bridge run is an ordinary ActiveGraph event log. Two families of events
live in it:

1. **Bridge events** (defined here) — the record of what the wrapped
   agent did: run lifecycle, invocations, mediated effects, checkpoints,
   hazards, verification outcomes. These are the replay substrate.
2. **Native projection events** (``object.created`` / ``relation.created``,
   emitted through ``Graph.add_object`` / ``add_relation``) — the
   AgentExecutionPack graph derived from the bridge events. Because the
   projection uses ActiveGraph's own event types, native tooling
   (``compute_diff``, causal traces, the CLI inspector) works on bridge
   runs unmodified.

Event payloads are JSON documents (ActiveGraph's storage contract);
values pass through an :class:`~activegraph_bridge.codecs.EffectCodec`
before landing here.

Causality: ``effect.responded`` / ``effect.failed`` set ``caused_by`` to
their ``effect.requested`` event; effects set ``caused_by`` to their
enclosing effect (nested calls) or invocation; projection events set
``caused_by`` to the bridge event they materialize.
"""

from __future__ import annotations

from typing import Any

from activegraph.core.event import Event

__all__ = [
    "RUN_STARTED",
    "RUN_COMPLETED",
    "INVOCATION_STARTED",
    "INVOCATION_COMPLETED",
    "INVOCATION_FAILED",
    "EFFECT_REQUESTED",
    "EFFECT_RESPONDED",
    "EFFECT_FAILED",
    "CHECKPOINT_RECORDED",
    "HAZARD_DETECTED",
    "VERIFICATION_RECORDED",
    "FORK_CONFIGURED",
    "BRIDGE_EVENT_TYPES",
    "META_EVENT_TYPES",
    "effect_pairs",
]

RUN_STARTED = "bridge.run_started"
RUN_COMPLETED = "bridge.run_completed"
INVOCATION_STARTED = "invocation.started"
INVOCATION_COMPLETED = "invocation.completed"
INVOCATION_FAILED = "invocation.failed"
EFFECT_REQUESTED = "effect.requested"
EFFECT_RESPONDED = "effect.responded"
EFFECT_FAILED = "effect.failed"
CHECKPOINT_RECORDED = "checkpoint.recorded"
HAZARD_DETECTED = "hazard.detected"
VERIFICATION_RECORDED = "bridge.verification"
FORK_CONFIGURED = "bridge.fork_configured"

BRIDGE_EVENT_TYPES = frozenset(
    {
        RUN_STARTED,
        RUN_COMPLETED,
        INVOCATION_STARTED,
        INVOCATION_COMPLETED,
        INVOCATION_FAILED,
        EFFECT_REQUESTED,
        EFFECT_RESPONDED,
        EFFECT_FAILED,
        CHECKPOINT_RECORDED,
        HAZARD_DETECTED,
        VERIFICATION_RECORDED,
        FORK_CONFIGURED,
    }
)

# Events that describe the run *about itself* rather than what the agent
# did. Excluded from execution comparisons (diff, verify scripts).
META_EVENT_TYPES = frozenset(
    {RUN_STARTED, RUN_COMPLETED, VERIFICATION_RECORDED, FORK_CONFIGURED, HAZARD_DETECTED}
)


def effect_pairs(events: list[Event]) -> list[tuple[Event, Event | None]]:
    """Pair each ``effect.requested`` with its response/failure event.

    Pairing is by ``caused_by`` linkage, mirroring how ActiveGraph's own
    ``ToolCache.from_events`` walks ``tool.requested``/``tool.responded``.
    An unpaired request (crash mid-effect) yields ``(request, None)``.
    """
    responses: dict[str, Event] = {}
    for e in events:
        if e.type in (EFFECT_RESPONDED, EFFECT_FAILED) and e.caused_by:
            responses[e.caused_by] = e
    return [
        (e, responses.get(e.id)) for e in events if e.type == EFFECT_REQUESTED
    ]


def summarize_effect_counts(events: list[Event]) -> dict[str, int]:
    """Count captured effects by category (model/tool/retrieval/...)."""
    counts: dict[str, int] = {}
    for e in events:
        if e.type == EFFECT_REQUESTED:
            category = str(e.payload.get("category", "external"))
            counts[category] = counts.get(category, 0) + 1
    return counts


def error_doc(exc: BaseException) -> dict[str, Any]:
    """The JSON shape shared by invocation.failed / effect.failed."""
    from .codecs import encode_exception

    return encode_exception(exc)
