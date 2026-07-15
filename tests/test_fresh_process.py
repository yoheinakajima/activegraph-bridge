"""The strongest honesty test: verification from a brand-new process.

Nothing in-memory can leak into the result — the fresh interpreter sees
only the SQLite store and the agent code, and must reproduce the same
final output and the same projected graph with zero live calls.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_fresh_process_verification(agent, store_url):
    with agent.execution(label="cross-process") as run:
        agent.invoke({"order_id": "ord_1", "question": "where is it?"})
    local_replay = run.replay()
    local_output = run.playback_output()

    script = f"""
import json
from activegraph_bridge import load_run
from tests.conftest import make_factory, TOOL_CALLS

run = load_run({store_url!r}, {run.run_id!r}).attach(make_factory())
before = dict(TOOL_CALLS)
result = run.verify()
assert result.ok, result.divergence
assert TOOL_CALLS == before, "live tool executed in fresh process"
replay = run.replay()
print(json.dumps({{
    "output": run.playback_output(),
    "objects": replay.objects,
    "relations": replay.relations,
    "verified": run.report.verified,
    "grade": run.report.grade.value,
}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=60,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    import json

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["output"] == local_output
    assert payload["objects"] == local_replay.objects
    assert payload["relations"] == local_replay.relations
    assert payload["verified"] is True
    assert payload["grade"] == "boundary"
