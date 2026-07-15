"""Replayability grades, findings, and the honest run report.

The bridge never describes a run as more replayable than it proved
itself to be. Every run earns a grade from what was actually captured:

==============  ==============================================================
grade           honest guarantee
==============  ==============================================================
observed        Inspection and lineage only — something prevented a faithful
                envelope (for example a lossy output capture).
envelope        The whole invocation was recorded as one effect: recorded
                output can be played back; forks are possible before the
                invocation.
boundary        Each model/tool/retrieval/external call was individually
                mediated, and a clean-agent reconstruction strategy exists:
                agent code can be re-executed against recorded effects, and
                forks can branch at effect boundaries. Displayed as
                ``boundary-verified`` only after ``run.verify()`` passes.
checkpointed    Boundary, plus restorable agent state snapshots — forks can
                resume from a checkpoint instead of re-executing the prefix.
native          Agent state and coordination live in ActiveGraph itself.
                Reserved for native integrations; the bridge does not grant
                this grade.
==============  ==============================================================

Grades are computed from the event log alone, so a run loaded from disk
years later reports the same truth it earned when recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from . import events as ev

if TYPE_CHECKING:
    from activegraph.core.event import Event

__all__ = ["Grade", "Finding", "ReplayabilityReport"]


class Grade(str, Enum):
    """Integration grades, ordered weakest to strongest."""

    OBSERVED = "observed"
    ENVELOPE = "envelope"
    BOUNDARY = "boundary"
    CHECKPOINTED = "checkpointed"
    NATIVE = "native"

    @property
    def rank(self) -> int:
        return _GRADE_ORDER.index(self)


_GRADE_ORDER = [
    Grade.OBSERVED,
    Grade.ENVELOPE,
    Grade.BOUNDARY,
    Grade.CHECKPOINTED,
    Grade.NATIVE,
]


@dataclass(frozen=True)
class Finding:
    """One observation about a run's replayability.

    ``severity`` is ``"blocker"`` (caps the grade), ``"warning"``
    (grade survives, honesty note), or ``"note"``.
    """

    severity: str
    code: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.detail}"


@dataclass
class ReplayabilityReport:
    """What this run can honestly promise, and why.

    Render with ``print(run.report)``::

        Replayability: boundary-verified
        Captured:      7 model calls, 4 tool calls, 2 retrieval calls
        Fork points:   11
        Blockers:      none

    or, for a run that cannot be replayed::

        Replayability: playback-only (envelope)
        Captured:      invocation envelope only
        Fork points:   1
        Blockers:
          - unrecorded socket I/O from custom_search.py:81
          - agent supplied as a live instance; no reset or checkpoint strategy
    """

    grade: Grade
    verified: bool
    effect_counts: dict[str, int] = field(default_factory=dict)
    invocations: int = 0
    fork_points: int = 0
    checkpoints: int = 0
    findings: list[Finding] = field(default_factory=list)
    reconstruction: str = "none"
    fingerprint: dict[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def label(self) -> str:
        """The one-line honest claim."""
        if self.grade is Grade.NATIVE:
            return "native"
        if self.grade is Grade.CHECKPOINTED:
            return "checkpointed-verified" if self.verified else "checkpointed (unverified)"
        if self.grade is Grade.BOUNDARY:
            return "boundary-verified" if self.verified else "boundary (unverified)"
        if self.grade is Grade.ENVELOPE:
            return "playback-only (envelope)"
        return "inspection-only (observed)"

    def __str__(self) -> str:
        captured = ", ".join(
            f"{n} {category} call{'s' if n != 1 else ''}"
            for category, n in sorted(self.effect_counts.items(), key=lambda kv: -kv[1])
            if category != "determinism"
        )
        det = self.effect_counts.get("determinism", 0)
        if det:
            captured = f"{captured} (+{det} determinism reads)" if captured else f"{det} determinism reads"
        if not captured:
            captured = "invocation envelope only"
        lines = [
            f"Replayability: {self.label}",
            f"Captured:      {captured}",
            f"Fork points:   {self.fork_points}",
        ]
        if self.checkpoints:
            lines.append(f"Checkpoints:   {self.checkpoints}")
        if self.blockers:
            lines.append("Blockers:")
            lines.extend(f"  - {b.detail}" for b in self.blockers)
        else:
            lines.append("Blockers:      none")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w.detail}" for w in self.warnings)
        return "\n".join(lines)


def compute_report(events: list["Event"]) -> ReplayabilityReport:
    """Derive the honest report from a run's event log.

    Pure function of the log — the same events always yield the same
    report, whether computed live at the end of a recording or years
    later from a loaded store.
    """
    findings: list[Finding] = []
    reconstruction = "none"
    fingerprint: dict[str, Any] = {}
    capabilities: dict[str, Any] = {}
    invocations = 0
    checkpoints = 0
    completed = False
    verified = False

    for e in events:
        if e.type == ev.RUN_STARTED:
            reconstruction = str(e.payload.get("reconstruction", "none"))
            fingerprint = dict(e.payload.get("fingerprint", {}) or {})
            capabilities = dict(e.payload.get("adapter_capabilities", {}) or {})
        elif e.type == ev.INVOCATION_STARTED:
            invocations += 1
        elif e.type == ev.INVOCATION_COMPLETED:
            completed = True
        elif e.type == ev.CHECKPOINT_RECORDED:
            checkpoints += 1
        elif e.type == ev.HAZARD_DETECTED:
            findings.append(
                Finding(
                    severity="blocker",
                    code="unrecorded-io",
                    detail=str(e.payload.get("detail", "unrecorded I/O")),
                )
            )
        elif e.type == ev.EFFECT_RESPONDED:
            for note in e.payload.get("lossy", []) or []:
                findings.append(Finding("warning", "lossy-capture", str(note)))
        elif e.type == ev.INVOCATION_FAILED:
            findings.append(
                Finding(
                    "note",
                    "invocation-failed",
                    f"invocation failed: {((e.payload.get('error') or {}).get('message', ''))}",
                )
            )
        elif e.type == ev.VERIFICATION_RECORDED:
            verified = bool(e.payload.get("ok"))

    pairs = ev.effect_pairs(events)
    effect_counts = ev.summarize_effect_counts(events)
    boundary_effects = sum(
        n for category, n in effect_counts.items() if category != "determinism"
    )

    # Lossy captures on the *envelope* (invocation output) are blockers for
    # even playback fidelity; lossy effect responses are warnings that block
    # verification but not playback. Envelope lossiness is detected here.
    envelope_lossy = [
        Finding("blocker", "lossy-envelope", str(note))
        for e in events
        if e.type == ev.INVOCATION_COMPLETED
        for note in (e.payload.get("lossy") or [])
    ]
    findings.extend(envelope_lossy)

    if reconstruction not in ("fresh_factory", "checkpoint", "stateless_callable"):
        findings.append(
            Finding(
                severity="blocker" if boundary_effects else "warning",
                code="no-reconstruction",
                detail=(
                    "agent supplied as a live instance; no reset or checkpoint "
                    "strategy (pass a factory to wrap() for re-execution and forking)"
                ),
            )
        )

    unknown_writes = sum(
        1
        for req, _ in pairs
        if req.payload.get("side_effect") == "unknown"
        and req.payload.get("category") != "determinism"
    )
    if unknown_writes:
        findings.append(
            Finding(
                "note",
                "unknown-side-effects",
                f"{unknown_writes} effect(s) with side_effect='unknown' are "
                f"treated as writes during forks; declare read/write to unlock "
                f"cleaner fork policies",
            )
        )

    blockers = [f for f in findings if f.severity == "blocker"]

    # Grade ladder, honest and monotone.
    if (not completed and invocations == 0) or envelope_lossy:
        grade = Grade.OBSERVED
    elif blockers:
        # Hazards or a missing reconstruction strategy cap the run at
        # envelope: recorded output plays back, re-execution is not promised.
        grade = Grade.ENVELOPE
    elif boundary_effects == 0:
        # No mediated calls. With a reconstruction strategy the agent is
        # re-executable by construction (nothing nondeterministic was
        # observed); without one, playback is all that can be claimed.
        grade = (
            Grade.BOUNDARY
            if reconstruction in ("fresh_factory", "stateless_callable")
            else Grade.ENVELOPE
        )
    elif checkpoints and capabilities.get("supports_restore"):
        grade = Grade.CHECKPOINTED
    else:
        grade = Grade.BOUNDARY

    fork_points = _count_fork_points(events, grade)

    return ReplayabilityReport(
        grade=grade,
        verified=verified and grade.rank >= Grade.BOUNDARY.rank,
        effect_counts=effect_counts,
        invocations=invocations,
        fork_points=fork_points,
        checkpoints=checkpoints,
        findings=findings,
        reconstruction=reconstruction,
        fingerprint=fingerprint,
    )


def _count_fork_points(events: list["Event"], grade: Grade) -> int:
    """Safe fork boundaries: invocation starts always; completed quiescent
    effects and checkpoints only when the run is boundary-grade or better."""
    n = 0
    responded = {e.caused_by for e in events if e.type in (ev.EFFECT_RESPONDED, ev.EFFECT_FAILED)}
    for e in events:
        if e.type == ev.INVOCATION_STARTED:
            n += 1
        elif grade.rank >= Grade.BOUNDARY.rank and e.type == ev.EFFECT_REQUESTED:
            if e.payload.get("quiescent", True) and e.id in responded:
                n += 1
        elif grade.rank >= Grade.BOUNDARY.rank and e.type == ev.CHECKPOINT_RECORDED:
            n += 1
    return n
