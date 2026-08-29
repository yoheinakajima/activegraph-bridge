"""Deterministic sources: time, randomness, and identifiers as effects.

Direct reads of ``time.time()``, ``random.random()``, or ``uuid.uuid4()``
are classic replay hazards — they bypass the membrane, so a re-execution
sees different values and diverges. These drop-in sources route the read
through the active session as a tiny recorded effect (category
``determinism``): live runs record the real value, replays and forks are
served the recorded one.

Outside any session they return real values, so sprinkling them through
agent code costs nothing when the bridge is not in play::

    from activegraph_bridge import det

    stamp = det.now()          # datetime, recorded
    jitter = det.random()      # float in [0, 1), recorded
    request_id = det.uuid4()   # uuid.UUID, recorded
"""

from __future__ import annotations

import datetime as _dt
import random as _random
import time as _time
import uuid as _uuid

from .session import current_session

__all__ = ["now", "time", "random", "randint", "uuid4"]


def _mediate(kind: str, request: dict, live: object) -> object:
    session = current_session()
    if session is None:
        return live() if callable(live) else live
    return session.effect(
        kind,
        request,
        live if callable(live) else (lambda: live),
        name=kind,
        side_effect="read",
        footprint="pure",
        replay_source="recorded",
        observables=(),
        category="determinism",
    )


def now() -> _dt.datetime:
    """Current UTC time as an aware datetime, recorded for replay."""
    return _mediate(
        "time.now", {}, lambda: _dt.datetime.now(tz=_dt.timezone.utc)
    )  # type: ignore[return-value]


def time() -> float:
    """Epoch seconds, recorded for replay."""
    return _mediate("time.time", {}, _time.time)  # type: ignore[return-value]


def random() -> float:
    """Uniform float in [0, 1), recorded for replay."""
    return _mediate("random.random", {}, _random.random)  # type: ignore[return-value]


def randint(a: int, b: int) -> int:
    """Random integer in [a, b], recorded for replay."""
    return _mediate(
        "random.randint", {"a": a, "b": b}, lambda: _random.randint(a, b)
    )  # type: ignore[return-value]


def uuid4() -> _uuid.UUID:
    """Random UUID, recorded for replay (stored as its string form)."""
    value = _mediate("id.uuid4", {}, lambda: str(_uuid.uuid4()))
    return _uuid.UUID(str(value))
