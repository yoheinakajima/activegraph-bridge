"""Offline verifier for the checked-in post-oracle fork receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from activegraph.core.event import Event

from activegraph_bridge import (
    ForkReceipt,
    HmacEnvironmentAttestor,
    verify_fork_receipt,
)


ROOT = Path(__file__).resolve().parent


def _json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _events(name: str) -> list[Event]:
    return [
        Event(**json.loads(line))
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    manifest = _json("manifest.json")
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"FAIL checksum {name}: {actual} != {expected}")

    receipt = ForkReceipt.from_dict(_json("receipt.json"))
    attestor = HmacEnvironmentAttestor(
        b"activegraph-bridge-public-conformance-key-v1",
        issuer="activegraph-bridge-conformance",
        key_id="fixture-v1",
    )
    result = verify_fork_receipt(
        receipt,
        parent_events=_events("parent.jsonl"),
        child_events=_events("child.jsonl"),
        environment_verifier=attestor,
    )
    if not result.ok:
        raise SystemExit("FAIL receipt: " + "; ".join(result.errors))
    if manifest["fork_inherited_oracle_calls"] != 0:
        raise SystemExit("FAIL inherited oracle call count is not zero")
    if receipt.external_continuation != "verified":
        raise SystemExit("FAIL continuation is not verified")
    print(
        "POST-ORACLE FORK RECEIPT PASS — committed oracle served from record; "
        "0 inherited external calls"
    )


if __name__ == "__main__":
    main()
