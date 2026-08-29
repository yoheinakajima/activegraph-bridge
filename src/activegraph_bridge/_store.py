"""Store resolution: one URL, three backends, identical semantics.

The bridge stores runs in ordinary ActiveGraph event stores:

- ``sqlite:///agent-runs.db`` (or a bare path) — the default. Multi-run
  files, durable, forkable via ``SQLiteEventStore.fork_run``, readable
  by the ActiveGraph CLI.
- ``memory://`` (or ``:memory:``) — process-local, for tests and
  ephemeral exploration. Forking is implemented as a prefix copy so the
  full API works; nothing survives the interpreter.
- ``postgres://…`` — deliberately not supported yet; ActiveGraph's own
  fork primitive is SQLite-first (CONTRACT v0.8 #5) and the bridge
  refuses rather than offering a store where half the API would fail.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional, TypeVar

from activegraph.core.event import Event
from activegraph.store.base import RunRecord
from activegraph.store.sqlite import SQLiteEventStore

from .errors import BridgeConfigurationError

__all__ = ["BridgeStore", "resolve_store"]

_T = TypeVar("_T")


def _now_iso() -> str:
    return (
        datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


class BridgeStore:
    """Backend-neutral handle on a multi-run store. Subclasses: sqlite, memory."""

    url: str

    def create_run(
        self,
        run_id: str,
        *,
        label: str | None = None,
        goal: str | None = None,
        parent_run_id: str | None = None,
        forked_at_event_id: str | None = None,
    ):  # -> EventStore
        raise NotImplementedError

    def open_run(self, run_id: str):  # -> EventStore
        raise NotImplementedError

    def load_events(self, run_id: str) -> list[Event]:
        store = self.open_run(run_id)
        try:
            return list(store.iter_events())
        finally:
            close = getattr(store, "close", None)
            if close:
                close()

    def append_event(self, run_id: str, event: Event) -> None:
        """Append through a short-lived handle and close it deterministically."""
        store = self.open_run(run_id)
        try:
            store.append(event)
        finally:
            close = getattr(store, "close", None)
            if close:
                close()

    def fork_run(
        self, *, parent_run_id: str, new_run_id: str, at_event_id: str, label: str | None
    ) -> int:
        raise NotImplementedError

    def list_runs(self) -> list[RunRecord]:
        raise NotImplementedError

    def get_run(self, run_id: str) -> RunRecord | None:
        return next((r for r in self.list_runs() if r.run_id == run_id), None)

    def most_recent_run_id(self) -> str | None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# SQLite                                                                       #
# --------------------------------------------------------------------------- #


class SqliteBridgeStore(BridgeStore):
    _locks_guard = threading.Lock()
    _locks: dict[str, threading.RLock] = {}

    def __init__(self, path: str, url: str) -> None:
        self.path = path
        self.url = url
        lock_key = os.path.realpath(os.path.abspath(path))
        with self._locks_guard:
            self._lock = self._locks.setdefault(lock_key, threading.RLock())

    def _retry_locked(self, operation: Callable[[], _T]) -> _T:
        """Serialize local opens and retry brief cross-process SQLite races."""
        delay = 0.01
        for attempt in range(8):
            try:
                with self._lock:
                    return operation()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 7:
                    raise
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def create_run(
        self,
        run_id: str,
        *,
        label: str | None = None,
        goal: str | None = None,
        parent_run_id: str | None = None,
        forked_at_event_id: str | None = None,
    ) -> SQLiteEventStore:
        def create() -> SQLiteEventStore:
            store = SQLiteEventStore(self.path, run_id=run_id)
            try:
                store.upsert_run(
                    parent_run_id=parent_run_id,
                    forked_at_event_id=forked_at_event_id,
                    label=label,
                    created_at=_now_iso(),
                    goal=goal,
                )
            except BaseException:
                store.close()
                raise
            return store

        return self._retry_locked(create)

    def open_run(self, run_id: str) -> SQLiteEventStore:
        return self._retry_locked(
            lambda: SQLiteEventStore(self.path, run_id=run_id)
        )

    def fork_run(
        self, *, parent_run_id: str, new_run_id: str, at_event_id: str, label: str | None
    ) -> int:
        return self._retry_locked(
            lambda: SQLiteEventStore.fork_run(
                self.path,
                parent_run_id=parent_run_id,
                new_run_id=new_run_id,
                at_event_id=at_event_id,
                label=label,
                created_at=_now_iso(),
            )
        )

    def list_runs(self) -> list[RunRecord]:
        return self._retry_locked(lambda: SQLiteEventStore.list_runs(self.path))

    def most_recent_run_id(self) -> str | None:
        return self._retry_locked(
            lambda: SQLiteEventStore.most_recent_run_id(self.path)
        )


# --------------------------------------------------------------------------- #
# Memory                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class _MemRun:
    record: RunRecord
    events: list[Event] = field(default_factory=list)
    by_id: dict[str, int] = field(default_factory=dict)


class _MemRunStore:
    """EventStore protocol over a shared in-process run entry."""

    def __init__(self, run: _MemRun) -> None:
        self._run = run
        self.run_id = run.record.run_id

    def append(self, event: Event) -> None:
        if event.id in self._run.by_id:
            from activegraph.store import DuplicateEventError

            raise DuplicateEventError(f"duplicate event id: {event.id}")
        self._run.by_id[event.id] = len(self._run.events)
        self._run.events.append(event)

    def iter_events(
        self, after: Optional[str] = None, until: Optional[str] = None
    ) -> Iterator[Event]:
        start = 0
        end = len(self._run.events)
        if after is not None:
            start = self._run.by_id[after] + 1
        if until is not None:
            end = self._run.by_id[until] + 1
        return iter(self._run.events[start:end])

    def get_event(self, event_id: str) -> Optional[Event]:
        idx = self._run.by_id.get(event_id)
        return self._run.events[idx] if idx is not None else None

    def count(self) -> int:
        return len(self._run.events)

    def truncate_after(self, event_id: str) -> None:
        cut = self._run.by_id[event_id] + 1
        for e in self._run.events[cut:]:
            del self._run.by_id[e.id]
        del self._run.events[cut:]

    def close(self) -> None:
        pass


class MemoryBridgeStore(BridgeStore):
    """Process-local multi-run store keyed by its URL.

    Two ``wrap(..., store="memory://demo")`` calls in one process share
    runs, mirroring how two SQLite opens of the same path share a file.
    """

    _registry: dict[str, dict[str, _MemRun]] = {}
    _lock = threading.Lock()

    def __init__(self, url: str) -> None:
        self.url = url
        with self._lock:
            self._runs = self._registry.setdefault(url, {})

    def create_run(
        self,
        run_id: str,
        *,
        label: str | None = None,
        goal: str | None = None,
        parent_run_id: str | None = None,
        forked_at_event_id: str | None = None,
    ) -> _MemRunStore:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                run = _MemRun(
                    record=RunRecord(
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        forked_at_event_id=forked_at_event_id,
                        label=label,
                        created_at=_now_iso(),
                        goal=goal,
                        frame_id=None,
                    )
                )
                self._runs[run_id] = run
        return _MemRunStore(run)

    def open_run(self, run_id: str) -> _MemRunStore:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"no run {run_id!r} in {self.url}")
        return _MemRunStore(run)

    def fork_run(
        self, *, parent_run_id: str, new_run_id: str, at_event_id: str, label: str | None
    ) -> int:
        with self._lock:
            parent = self._runs.get(parent_run_id)
            if parent is None:
                raise KeyError(f"no run {parent_run_id!r} in {self.url}")
            cut = parent.by_id.get(at_event_id)
            if cut is None:
                raise KeyError(
                    f"event {at_event_id!r} not found in run {parent_run_id!r}"
                )
            child = _MemRun(
                record=RunRecord(
                    run_id=new_run_id,
                    parent_run_id=parent_run_id,
                    forked_at_event_id=at_event_id,
                    label=label,
                    created_at=_now_iso(),
                    goal=parent.record.goal,
                    frame_id=None,
                )
            )
            for e in parent.events[: cut + 1]:
                child.by_id[e.id] = len(child.events)
                child.events.append(e)
            self._runs[new_run_id] = child
            return len(child.events)

    def list_runs(self) -> list[RunRecord]:
        return [r.record for r in self._runs.values()]

    def most_recent_run_id(self) -> str | None:
        latest: str | None = None
        latest_created = ""
        for run_id, run in self._runs.items():
            if run.record.created_at >= latest_created:
                latest_created = run.record.created_at
                latest = run_id
        return latest


# --------------------------------------------------------------------------- #
# resolution                                                                   #
# --------------------------------------------------------------------------- #


def resolve_store(store: str | BridgeStore) -> BridgeStore:
    """Turn a URL (or bare path) into a BridgeStore."""
    if isinstance(store, BridgeStore):
        return store
    if not isinstance(store, str) or not store:
        raise BridgeConfigurationError(
            f"store must be a URL string or BridgeStore, got {store!r}",
            what_failed=f"wrap(...) received store={store!r}.",
            why="The store is where runs live; without one there is nothing to replay.",
            how_to_fix=(
                "Pass one of:\n"
                "    store='sqlite:///agent-runs.db'   durable, forkable (default style)\n"
                "    store='agent-runs.db'             same, bare-path shorthand\n"
                "    store='memory://'                 process-local, for tests"
            ),
        )
    if store in (":memory:",) or store.startswith("memory://"):
        return MemoryBridgeStore(store if store.startswith("memory://") else "memory://")
    if store.startswith(("postgres://", "postgresql://")):
        raise BridgeConfigurationError(
            "postgres stores are not supported by the bridge yet",
            what_failed=f"wrap(...) received the postgres store URL {store!r}.",
            why=(
                "ActiveGraph's fork primitive is SQLite-first (CONTRACT v0.8 #5); "
                "offering a postgres bridge store today would mean fork() failing "
                "at runtime on half the API. The bridge refuses up front instead."
            ),
            how_to_fix=(
                "Use sqlite:///path/to/runs.db (durable and forkable), then "
                "migrate finished runs with `activegraph migrate` if they need "
                "to live in postgres."
            ),
        )
    if store.startswith("sqlite:"):
        from activegraph.store.url import parse_store_url

        parsed = parse_store_url(store)
        return SqliteBridgeStore(parsed.sqlite_path or "", store)
    # bare path shorthand
    return SqliteBridgeStore(store, f"sqlite:///{store}")
