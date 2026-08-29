"""Signed environment assertions and hash-bound fork receipts.

The bridge can already *enforce* that a fork prefix is served from the log.
This module makes that enforcement portable evidence. A receipt binds the
source prefix, child log, effect identities, target-environment attestation,
and zero-reexecution counters into one canonical JSON document.

HMAC is intentionally a configured trust mechanism, not a global identity
system. Production callers keep the key outside the log and decide which key
ids they trust. The public conformance fixture uses an explicitly published
fixture key to exercise the verification path; it is not a production trust
root.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from activegraph.core.event import Event

from . import events as ev
from ._canonical import canonical_json, content_hash
from .policy import derive_footprint

__all__ = [
    "EnvironmentAttestation",
    "EnvironmentVerifier",
    "ForkReceipt",
    "HmacEnvironmentAttestor",
    "ReceiptVerification",
    "effect_evidence",
    "event_log_hash",
    "verify_fork_receipt",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def event_log_hash(events: list[Event]) -> str:
    """SHA-256 over the ordered canonical event documents."""
    return content_hash([event.to_dict() for event in events])


@dataclass(frozen=True)
class EnvironmentAttestation:
    """A signed claim that a target environment represents a source prefix."""

    environment_id: str
    snapshot_id: str
    issuer: str
    key_id: str
    claims: dict[str, Any]
    issued_at: str
    signature: str
    expires_at: str | None = None
    algorithm: str = "hmac-sha256"
    schema_version: str = "activegraph-bridge/environment-attestation-v1"

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "environment_id": self.environment_id,
            "snapshot_id": self.snapshot_id,
            "issuer": self.issuer,
            "key_id": self.key_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "claims": self.claims,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.unsigned_dict() | {"signature": self.signature}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EnvironmentAttestation":
        return cls(
            environment_id=str(value["environment_id"]),
            snapshot_id=str(value["snapshot_id"]),
            issuer=str(value["issuer"]),
            key_id=str(value["key_id"]),
            claims=dict(value.get("claims") or {}),
            issued_at=str(value["issued_at"]),
            expires_at=(
                str(value["expires_at"])
                if value.get("expires_at") is not None
                else None
            ),
            signature=str(value["signature"]),
            algorithm=str(value.get("algorithm", "hmac-sha256")),
            schema_version=str(
                value.get(
                    "schema_version",
                    "activegraph-bridge/environment-attestation-v1",
                )
            ),
        )


@runtime_checkable
class EnvironmentVerifier(Protocol):
    """Configured trust root for a target-environment assertion."""

    @property
    def verifier_id(self) -> str: ...

    def verify(self, attestation: EnvironmentAttestation) -> bool: ...


class HmacEnvironmentAttestor:
    """Issue and verify HMAC-SHA256 environment assertions."""

    def __init__(self, secret: bytes, *, issuer: str, key_id: str) -> None:
        if not secret:
            raise ValueError("environment attestation secret must not be empty")
        self._secret = bytes(secret)
        self.issuer = issuer
        self.key_id = key_id

    @property
    def verifier_id(self) -> str:
        return f"hmac-sha256:{self.issuer}:{self.key_id}"

    def _signature(self, document: dict[str, Any]) -> str:
        digest = hmac.new(
            self._secret,
            canonical_json(document).encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def issue(
        self,
        *,
        environment_id: str,
        snapshot_id: str,
        claims: dict[str, Any],
        issued_at: str | None = None,
        expires_at: str | None = None,
    ) -> EnvironmentAttestation:
        unsigned = {
            "schema_version": "activegraph-bridge/environment-attestation-v1",
            "algorithm": "hmac-sha256",
            "environment_id": environment_id,
            "snapshot_id": snapshot_id,
            "issuer": self.issuer,
            "key_id": self.key_id,
            "issued_at": issued_at or _now_iso(),
            "expires_at": expires_at,
            "claims": dict(claims),
        }
        return EnvironmentAttestation(
            environment_id=environment_id,
            snapshot_id=snapshot_id,
            issuer=self.issuer,
            key_id=self.key_id,
            claims=dict(claims),
            issued_at=str(unsigned["issued_at"]),
            expires_at=expires_at,
            signature=self._signature(unsigned),
        )

    def verify(self, attestation: EnvironmentAttestation) -> bool:
        if (
            attestation.algorithm != "hmac-sha256"
            or attestation.issuer != self.issuer
            or attestation.key_id != self.key_id
        ):
            return False
        if attestation.expires_at is not None:
            try:
                expires = datetime.fromisoformat(
                    attestation.expires_at.replace("Z", "+00:00")
                )
            except ValueError:
                return False
            if expires <= datetime.now(timezone.utc):
                return False
        expected = self._signature(attestation.unsigned_dict())
        return hmac.compare_digest(expected, attestation.signature)


def effect_evidence(events: list[Event]) -> list[dict[str, Any]]:
    """Project effect request/outcome pairs into language-neutral evidence."""
    evidence: list[dict[str, Any]] = []
    for request, outcome in ev.effect_pairs(events):
        side_effect = str(request.payload.get("side_effect", "unknown"))
        footprint = str(
            request.payload.get("footprint")
            or derive_footprint(side_effect)  # type: ignore[arg-type]
        )
        if outcome is None:
            lifecycle = "requested"
        elif outcome.type == ev.EFFECT_RESPONDED:
            lifecycle = "committed"
        else:
            lifecycle = "failed"
        evidence.append(
            {
                "request_event_id": request.id,
                "outcome_event_id": outcome.id if outcome is not None else None,
                "kind": str(request.payload.get("kind", "")),
                "name": str(request.payload.get("name", "")),
                "footprint": footprint,
                "replay_source": str(
                    request.payload.get(
                        "replay_source", "recorded" if outcome is not None else "uncaptured"
                    )
                ),
                "lifecycle": lifecycle,
                "observables": sorted(
                    str(value)
                    for value in (request.payload.get("observables") or [])
                ),
                "request_hash": str(request.payload.get("request_hash", "")),
                "response_hash": (
                    str(outcome.payload.get("response_hash", ""))
                    if outcome is not None
                    else ""
                ),
            }
        )
    return evidence


@dataclass(frozen=True)
class ForkReceipt:
    """Portable evidence emitted after one fork execution."""

    parent_run_id: str
    child_run_id: str
    forked_before_event_id: str
    copied_through_event_id: str
    parent_event_count_at_fork: int
    prefix_event_count: int
    parent_log_hash_at_fork: str
    prefix_hash: str
    child_log_hash_before_receipt: str
    source_fingerprint: dict[str, Any]
    target_fingerprint: dict[str, Any]
    target_environment: dict[str, Any] | None
    environment_verified: bool
    environment_verifier: str | None
    inherited_effects: list[dict[str, Any]]
    served_effect_request_ids: list[str]
    prefix_external_calls: int
    tail_executed_effect_request_ids: list[str]
    zero_reexecution_verified: bool
    external_continuation: str
    status: str
    created_at: str = field(default_factory=_now_iso)
    receipt_hash: str = ""
    schema_version: str = "activegraph-bridge/fork-receipt-v1"

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parent_run_id": self.parent_run_id,
            "child_run_id": self.child_run_id,
            "forked_before_event_id": self.forked_before_event_id,
            "copied_through_event_id": self.copied_through_event_id,
            "parent_event_count_at_fork": self.parent_event_count_at_fork,
            "prefix_event_count": self.prefix_event_count,
            "parent_log_hash_at_fork": self.parent_log_hash_at_fork,
            "prefix_hash": self.prefix_hash,
            "child_log_hash_before_receipt": self.child_log_hash_before_receipt,
            "source_fingerprint": self.source_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "target_environment": self.target_environment,
            "environment_verified": self.environment_verified,
            "environment_verifier": self.environment_verifier,
            "inherited_effects": self.inherited_effects,
            "served_effect_request_ids": self.served_effect_request_ids,
            "prefix_external_calls": self.prefix_external_calls,
            "tail_executed_effect_request_ids": self.tail_executed_effect_request_ids,
            "zero_reexecution_verified": self.zero_reexecution_verified,
            "external_continuation": self.external_continuation,
            "status": self.status,
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        document = self.unsigned_dict()
        return document | {"receipt_hash": self.receipt_hash or content_hash(document)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ForkReceipt":
        return cls(
            parent_run_id=str(value["parent_run_id"]),
            child_run_id=str(value["child_run_id"]),
            forked_before_event_id=str(value["forked_before_event_id"]),
            copied_through_event_id=str(value["copied_through_event_id"]),
            parent_event_count_at_fork=int(value["parent_event_count_at_fork"]),
            prefix_event_count=int(value["prefix_event_count"]),
            parent_log_hash_at_fork=str(value["parent_log_hash_at_fork"]),
            prefix_hash=str(value["prefix_hash"]),
            child_log_hash_before_receipt=str(value["child_log_hash_before_receipt"]),
            source_fingerprint=dict(value.get("source_fingerprint") or {}),
            target_fingerprint=dict(value.get("target_fingerprint") or {}),
            target_environment=(
                dict(value["target_environment"])
                if value.get("target_environment") is not None
                else None
            ),
            environment_verified=bool(value.get("environment_verified")),
            environment_verifier=(
                str(value["environment_verifier"])
                if value.get("environment_verifier") is not None
                else None
            ),
            inherited_effects=list(value.get("inherited_effects") or []),
            served_effect_request_ids=[
                str(item) for item in (value.get("served_effect_request_ids") or [])
            ],
            prefix_external_calls=int(value.get("prefix_external_calls", 0)),
            tail_executed_effect_request_ids=[
                str(item)
                for item in (value.get("tail_executed_effect_request_ids") or [])
            ],
            zero_reexecution_verified=bool(value.get("zero_reexecution_verified")),
            external_continuation=str(value.get("external_continuation", "refused")),
            status=str(value.get("status", "unknown")),
            created_at=str(value["created_at"]),
            receipt_hash=str(value.get("receipt_hash", "")),
            schema_version=str(
                value.get("schema_version", "activegraph-bridge/fork-receipt-v1")
            ),
        )


@dataclass(frozen=True)
class ReceiptVerification:
    ok: bool
    errors: list[str]


def verify_fork_receipt(
    receipt: ForkReceipt,
    *,
    parent_events: list[Event],
    child_events: list[Event],
    environment_verifier: EnvironmentVerifier | None = None,
) -> ReceiptVerification:
    """Recompute every portable receipt claim and report exact failures."""
    errors: list[str] = []
    if receipt.schema_version != "activegraph-bridge/fork-receipt-v1":
        errors.append("unsupported receipt schema")
    document = receipt.unsigned_dict()
    if receipt.receipt_hash != content_hash(document):
        errors.append("receipt hash mismatch")

    parent_at_fork = parent_events[: receipt.parent_event_count_at_fork]
    prefix = parent_events[: receipt.prefix_event_count]
    child_prefix = child_events[: receipt.prefix_event_count]
    if event_log_hash(parent_at_fork) != receipt.parent_log_hash_at_fork:
        errors.append("parent log hash mismatch")
    if event_log_hash(prefix) != receipt.prefix_hash:
        errors.append("parent prefix hash mismatch")
    if event_log_hash(child_prefix) != receipt.prefix_hash:
        errors.append("child prefix hash mismatch")
    if not prefix or prefix[-1].id != receipt.copied_through_event_id:
        errors.append("copied-through event does not match prefix boundary")
    if (
        receipt.prefix_event_count >= len(parent_events)
        or parent_events[receipt.prefix_event_count].id
        != receipt.forked_before_event_id
    ):
        errors.append("forked-before event does not match parent boundary")
    source_fingerprint = next(
        (
            dict(event.payload.get("fingerprint") or {})
            for event in parent_at_fork
            if event.type == ev.RUN_STARTED
        ),
        {},
    )
    if source_fingerprint != receipt.source_fingerprint:
        errors.append("source fingerprint mismatch")

    before_receipt: list[Event] = []
    found_receipt = False
    for event in child_events:
        if event.type == ev.FORK_RECEIPT and str(
            event.payload.get("receipt_hash", "")
        ) == receipt.receipt_hash:
            found_receipt = True
            break
        before_receipt.append(event)
    if not found_receipt:
        errors.append("receipt event missing from child log")
    elif event_log_hash(before_receipt) != receipt.child_log_hash_before_receipt:
        errors.append("child log hash mismatch")

    expected_effects = effect_evidence(prefix)
    if expected_effects != receipt.inherited_effects:
        errors.append("inherited effect evidence mismatch")
    expected_ids = [item["request_event_id"] for item in expected_effects]
    if sorted(receipt.served_effect_request_ids) != sorted(expected_ids):
        errors.append("not every inherited effect was served exactly from the record")
    if receipt.prefix_external_calls != 0:
        errors.append("prefix external call count is not zero")

    environment_ok = False
    if receipt.target_environment is not None and environment_verifier is not None:
        attestation = EnvironmentAttestation.from_dict(receipt.target_environment)
        environment_ok = environment_verifier.verify(attestation)
        if not environment_ok:
            errors.append("target-environment signature verification failed")
        required_claims = {
            "parent_run_id": receipt.parent_run_id,
            "child_run_id": receipt.child_run_id,
            "prefix_hash": receipt.prefix_hash,
            "forked_before_event_id": receipt.forked_before_event_id,
            "target_fingerprint_hash": content_hash(receipt.target_fingerprint),
        }
        if any(attestation.claims.get(k) != v for k, v in required_claims.items()):
            environment_ok = False
            errors.append("target-environment claims do not bind this fork")
    elif receipt.environment_verified:
        errors.append("receipt claims environment verification without a verifier")
    if receipt.environment_verified != environment_ok:
        errors.append("environment-verification verdict mismatch")
    if environment_ok:
        assert environment_verifier is not None
        if receipt.environment_verifier != environment_verifier.verifier_id:
            errors.append("environment verifier identity mismatch")

    computed_zero = (
        receipt.status == "completed"
        and receipt.prefix_external_calls == 0
        and sorted(receipt.served_effect_request_ids) == sorted(expected_ids)
    )
    if receipt.zero_reexecution_verified != computed_zero:
        errors.append("zero-reexecution verdict mismatch")
    computed_continuation = (
        "verified" if computed_zero and environment_ok else "conditional"
    )
    if receipt.external_continuation != computed_continuation:
        errors.append("external-continuation verdict mismatch")

    return ReceiptVerification(ok=not errors, errors=errors)
