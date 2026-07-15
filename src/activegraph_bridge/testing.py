"""Testing helpers: fakes and assertions for bridge integrations.

- :class:`ScriptedModel` — a deterministic stand-in for a model SDK
  client, useful in examples and adapter tests. Pair it with
  ``instrument.wrap_client`` to exercise the full record/verify/fork
  loop offline.
- :func:`assert_verified` — verify a run and fail with the full
  divergence report if it does not earn its badge. The backbone of
  "strict replay tests" for your own agents::

      def test_agent_replays(tmp_path):
          agent = wrap(build_agent, store=f"sqlite:///{tmp_path}/runs.db")
          agent.invoke({"question": "..."})
          assert_verified(agent.last_run)
"""

from __future__ import annotations

from typing import Any

from .runs import Run

__all__ = ["ScriptedModel", "assert_verified"]


class ScriptedModel:
    """A fake model client that returns scripted responses in order.

    ``create(**request)`` pops the next response. Deterministic given
    the call sequence, stateful like a real client (so tests exercise
    real reconstruction), and loud when the script runs dry.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._script = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.calls.append(request)
        if not self._script:
            raise RuntimeError(
                "ScriptedModel exhausted: more model calls than scripted responses"
            )
        return self._script.pop(0)


def assert_verified(run: Run, agent: Any = None, **kwargs: Any) -> None:
    """Verify ``run`` and raise ``AssertionError`` with details on failure."""
    result = run.verify(agent, **kwargs)
    if not result.ok:
        raise AssertionError(
            f"run {run.run_id} failed verification:\n{result}\n\n"
            f"{result.divergence}"
        )
