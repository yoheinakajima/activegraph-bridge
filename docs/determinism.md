# Determinism and coverage

ActiveGraph's replay contract names the classic divergence sources:
direct I/O, wall-clock time, randomness, generated identifiers, mutable
global state. The bridge attacks them from two directions.

## Mediation: `det`

Drop-in sources that record on first execution and serve on replay —
and fall back to real values outside any session, so they cost nothing
to adopt:

```python
from activegraph_bridge import det

stamp = det.now()            # aware UTC datetime
epoch = det.time()           # float seconds
jitter = det.random()        # float in [0, 1)
roll = det.randint(1, 6)
request_id = det.uuid4()     # uuid.UUID
```

Each is a tiny recorded effect (category `determinism`) — served during
verify and fork prefixes, counted separately in the report, excluded
from the graph projection.

## Detection: the audit guard

A process-wide `sys.addaudithook` (installed lazily on first session
activation; Python audit hooks are permanent by design, so the hook is
a no-op unless a session is active in the current context) watches:

- `socket.connect`
- `subprocess.Popen`
- `os.system`

Behavior by mode:

- **Recording**: the run keeps working, but a `hazard.detected` event
  lands in the log with the offending `file.py:lineno`, and the report
  gains a blocker — the run can no longer claim to be re-executable.
- **Verify / fork prefix**: the operation is **aborted** before it
  happens (`UnrecordedEffectError`). This is the enforcement behind
  "strict verification makes zero model, tool, or network calls."

I/O the bridge itself performs on purpose (the live execution inside a
mediated effect) is exempted via a context-local depth counter, so
recording an API call does not flag its own socket.

### Scope and honest limitations

- Detection is per-context: direct calls and asyncio tasks are covered;
  threads the agent spawns itself do not inherit the context and are
  not guarded in v0.1.
- `time.time()` / `random()` / `uuid4()` *reads* are not observable by
  audit hooks. They cannot fake a verification, though: any read that
  influences a request hash or the final output surfaces as an exact
  divergence, and the error message points at `det` as the fix.
- File opens are not watched by default (imports and stores open files
  constantly); treat the filesystem as an effect via `effect(...)` when
  it matters.

## What verification actually checks

1. Invocation inputs (canonical hash) — recorded vs re-driven.
2. Every mediated request — kind, name, canonical request hash, in
   order (`match="strict"`) or content-exact with position flexibility
   inside an invocation (`match="auto"`, tolerates concurrent
   completion reordering).
3. Failure parity — recorded failures must fail again, successes must
   succeed.
4. Complete consumption — no missing calls, no extra calls.
5. Final outputs (canonical hash), per invocation.
6. Zero unmediated I/O, enforced by the guard.

## The fingerprint

Recorded on every run (see `bridge.run_started`): Python version,
platform, activegraph + bridge versions, a SHA-256 of the agent's
source file, and a hash of the wrap configuration. Replays across code
changes are legitimate — the fingerprint keeps them *labeled*.
