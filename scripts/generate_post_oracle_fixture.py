"""Generate the public post-oracle fork receipt conformance fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from activegraph_bridge import (
    HmacEnvironmentAttestor,
    bridge_tool,
    instrument,
    verify_fork_receipt,
    wrap,
)


@bridge_tool(
    side_effect="read",
    footprint="pure",
    replay_source="recorded",
    observables=("agent.decision",),
)
def apply_fixture_policy(decision: str) -> dict[str, str]:
    return {"decision": decision}


class FixtureResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return {"output_text": f"approve:{kwargs['input']}"}


class FixtureOpenAI:
    def __init__(self) -> None:
        self.responses = FixtureResponses()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, events) -> None:
    path.write_text(
        "".join(
            json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="activegraph-bridge-receipt-") as tmp:
        raw = FixtureOpenAI()

        def build_agent():
            client = instrument.openai(raw)

            class Agent:
                def invoke(self, payload):
                    response = client.responses.create(
                        model="fixture-oracle-v1", input=payload["case"]
                    )
                    return apply_fixture_policy(response["output_text"])

            return Agent()

        agent = wrap(
            build_agent,
            store=f"sqlite:///{tmp}/runs.db",
            label="public-post-oracle-fixture",
            metadata={"network": "none", "provider": "fixture-only"},
        )
        original = agent.invoke({"case": "vendor-public-1"})
        run = agent.last_run
        if run is None:
            raise RuntimeError("fixture source run was not recorded")
        verification = run.verify()
        if not verification.ok:
            raise RuntimeError(f"source verification failed: {verification.divergence}")
        before_fork_calls = raw.responses.calls

        fork = run.fork(
            before=run.events.tool_call("apply_fixture_policy"),
            overrides={"tool_result": {"decision": "reject:vendor-public-1"}},
            label="public-post-oracle-fork",
        )
        attestor = HmacEnvironmentAttestor(
            b"activegraph-bridge-public-conformance-key-v1",
            issuer="activegraph-bridge-conformance",
            key_id="fixture-v1",
        )
        environment = attestor.issue(
            environment_id="fixture-clean-room",
            snapshot_id="fixture-snapshot-v1",
            claims=fork.environment_claims(),
            issued_at="2026-08-28T00:00:00Z",
        )
        alternative = fork.execute(
            target_environment=environment,
            environment_verifier=attestor,
        )
        if raw.responses.calls != before_fork_calls:
            raise RuntimeError("fork re-executed the inherited oracle")
        if fork.receipt is None:
            raise RuntimeError("fork did not emit a receipt")
        receipt_check = verify_fork_receipt(
            fork.receipt,
            parent_events=run.raw_events(),
            child_events=fork.run.raw_events(),
            environment_verifier=attestor,
        )
        if not receipt_check.ok:
            raise RuntimeError(f"receipt verification failed: {receipt_check.errors}")

        parent_path = output / "parent.jsonl"
        child_path = output / "child.jsonl"
        receipt_path = output / "receipt.json"
        environment_path = output / "environment-attestation.json"
        _write_jsonl(parent_path, run.raw_events())
        _write_jsonl(child_path, fork.run.raw_events())
        _write_json(receipt_path, fork.receipt.to_dict())
        _write_json(environment_path, environment.to_dict())

        files = [parent_path, child_path, receipt_path, environment_path]
        file_hashes = {path.name: _sha256(path) for path in files}
        manifest = {
            "schema_version": "activegraph-bridge/post-oracle-fixture-v1",
            "claim_scope": (
                "Offline conformance evidence for an actual bridge fork after a "
                "committed recorded oracle effect; no real provider was called."
            ),
            "source_run_id": run.run_id,
            "child_run_id": fork.child_run_id,
            "source_output": original,
            "child_output": alternative,
            "source_oracle_calls": before_fork_calls,
            "fork_inherited_oracle_calls": raw.responses.calls - before_fork_calls,
            "prefix_hash": fork.receipt.prefix_hash,
            "receipt_hash": fork.receipt.receipt_hash,
            "external_continuation": fork.receipt.external_continuation,
            "environment_verifier": fork.receipt.environment_verifier,
            "files": file_hashes,
        }
        manifest_path = output / "manifest.json"
        _write_json(manifest_path, manifest)
        checksums = file_hashes | {"manifest.json": _sha256(manifest_path)}
        (output / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/post-oracle-fork-v1"),
    )
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
