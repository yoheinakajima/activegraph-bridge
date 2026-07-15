"""Replay-hazard detection and run fingerprinting.

ActiveGraph's replay contract names the usual suspects for divergence:
direct I/O, wall-clock time, randomness, generated identifiers, mutable
global state. The bridge attacks them from two sides:

- **Detection** (this module): a process-wide ``sys.addaudithook`` that
  watches for sockets, subprocesses, and shell escapes. During live
  recording an unmediated hit becomes a ``hazard.detected`` event — the
  run keeps working but can no longer claim to be re-executable, and the
  replayability report says exactly where the leak is. During shadow
  execution (verify, fork prefix) the same hit **fails closed**: the
  operation is aborted with :class:`UnrecordedEffectError`, because a
  verification that let one live call through would be a lie.

- **Mediation** (:mod:`activegraph_bridge.det`): deterministic sources
  for time, randomness, and identifiers that record on first execution
  and serve on replay.

Audit hooks cannot be uninstalled (that is their point), so the hook is
installed once, lazily, and does nothing unless a session is active in
the current context. Python audit events do not propagate to threads the
agent spawns itself — detection is per-context, which covers direct
calls and asyncio; hazards in agent-spawned threads are documented as
out of scope for v0.1.
"""

from __future__ import annotations

import hashlib
import inspect
import platform
import sys
import sysconfig
import threading
from typing import Any

__all__ = ["ensure_guard_installed", "runtime_fingerprint"]

# Audit events that mean "agent code is reaching for the world directly".
# Deliberately small: cheap to check, hard to false-positive. File opens
# are not watched by default — imports and stores open files constantly.
_WATCHED = {
    "socket.connect": "socket",
    "subprocess.Popen": "subprocess",
    "os.system": "shell",
}

_installed = False
_install_lock = threading.Lock()

_BRIDGE_DIR = __file__.rsplit("/", 1)[0]
_STDLIB_DIR = sysconfig.get_paths().get("stdlib", "")


def ensure_guard_installed() -> None:
    """Install the audit hook once per process (idempotent, cheap)."""
    global _installed
    if _installed:
        return
    with _install_lock:
        if _installed:
            return
        sys.addaudithook(_hook)
        _installed = True


def _hook(event: str, args: tuple[Any, ...]) -> None:
    kind = _WATCHED.get(event)
    if kind is None:
        return
    # Import late and defensively: the hook fires for the whole process
    # forever, so it must never break unrelated code.
    try:
        from .session import current_session
    except Exception:
        return
    session = current_session()
    if session is None or session.in_mediated_execution():
        return
    detail = _describe(event, args)
    where = _caller_location()
    # note_hazard records in live modes and raises UnrecordedEffectError in
    # serving modes — aborting the socket/subprocess before it happens.
    session.note_hazard(kind=kind, detail=detail, where=where)


def _describe(event: str, args: tuple[Any, ...]) -> str:
    try:
        if event == "socket.connect" and len(args) >= 2:
            return f"socket.connect to {args[1]!r}"
        if event == "subprocess.Popen" and args:
            return f"subprocess {args[0]!r}"
        if event == "os.system" and args:
            return f"os.system({args[0]!r})"
    except Exception:
        pass
    return event


def _caller_location() -> str:
    """First stack frame outside the bridge, stdlib, and activegraph."""
    frame = sys._getframe()
    try:
        while frame is not None:
            filename = frame.f_code.co_filename
            if (
                not filename.startswith(_BRIDGE_DIR)
                and (not _STDLIB_DIR or not filename.startswith(_STDLIB_DIR))
                and "/activegraph/" not in filename
                and filename != "<string>"
            ):
                short = filename.rsplit("/", 1)[-1]
                return f"{short}:{frame.f_lineno}"
            frame = frame.f_back
    except Exception:
        pass
    finally:
        del frame
    return "unknown"


def runtime_fingerprint(target: Any, config: dict[str, Any]) -> dict[str, Any]:
    """The environment stamp recorded on every run.

    Code changes never *prohibit* replaying or forking — they are one of
    the best reasons to fork — but the log must make same-code strict
    replay distinguishable from a changed-code counterfactual. This
    stamp is that distinction.
    """
    from . import __version__ as bridge_version

    try:
        import activegraph

        ag_version = activegraph.__version__
    except Exception:  # pragma: no cover - activegraph is a hard dependency
        ag_version = "unknown"

    return {
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "activegraph_version": ag_version,
        "bridge_version": bridge_version,
        "agent_code_hash": _code_hash(target),
        "configuration_hash": _config_hash(config),
    }


def _code_hash(target: Any) -> str | None:
    """SHA-256 of the module source defining the agent (best effort)."""
    try:
        obj = target
        if not (inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismethod(obj)):
            obj = type(obj)
        source_file = inspect.getsourcefile(obj)
        if not source_file:
            return None
        with open(source_file, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def _config_hash(config: dict[str, Any]) -> str:
    from ._canonical import content_hash

    safe = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, type(None)))}
    return content_hash(safe)
