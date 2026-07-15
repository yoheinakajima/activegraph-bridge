"""The generic adapter: plain callables and invoke-style objects.

This is the universal fallback — it can drive anything you can call:

- a function or coroutine function (``target(payload)``)
- any object with the configured method (``target.invoke(payload)``),
  plus ``ainvoke``/``stream``/``astream``/``batch`` if present

It performs no framework instrumentation of its own; boundary-grade
capture comes from what the agent's code already routes through the
membrane (bridged tools, instrumented SDK clients, ``effect()`` calls,
``det`` sources). That composition is exactly why
``can_short_circuit_calls`` is honestly ``True``: everything the bridge
captured, the bridge can serve back instead of re-executing.

State: a generic agent has no framework snapshot, so ``checkpoint``
returns ``None`` and ``restore`` rebuilds via the factory — strategy #1
(fresh re-execution) from the bridge's state model.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, ContextManager

from .base import AdapterCapabilities, no_instrumentation

__all__ = ["GenericAdapter"]


class GenericAdapter:
    name = "generic"
    capabilities = AdapterCapabilities(
        intercepts_models=False,  # instrumented clients do, the adapter itself doesn't
        intercepts_tools=False,
        can_short_circuit_calls=True,
        captures_framework_steps=False,
        supports_streaming=True,
        supports_checkpoint=False,
        supports_restore=False,
    )

    def supports(self, target: Any) -> bool:
        return callable(target) or any(
            callable(getattr(target, m, None))
            for m in ("invoke", "ainvoke", "run", "arun", "stream", "astream")
        )

    # -- invocation ----------------------------------------------------------

    def _resolve(self, target: Any, method: str) -> Callable[..., Any]:
        fn = getattr(target, method, None)
        if callable(fn):
            return fn
        if method in ("invoke", "run", "__call__") and callable(target):
            return target
        raise AttributeError(
            f"{type(target).__name__} has no callable {method!r} "
            f"(and is not itself callable)"
        )

    def invoke(self, target: Any, method: str, args: tuple, kwargs: dict) -> Any:
        return self._resolve(target, method)(*args, **kwargs)

    async def ainvoke(self, target: Any, method: str, args: tuple, kwargs: dict) -> Any:
        # ainvoke tolerates a missing async method when the target itself
        # is a coroutine-friendly callable (async def agent(payload): ...).
        fn = getattr(target, method, None)
        if not callable(fn):
            if callable(target):
                fn = target
            else:
                fn = self._resolve(target, method)  # raises with the clear message
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    # -- instrumentation / state ----------------------------------------------

    def instrument(self, target: Any, session: Any) -> ContextManager[None]:
        return no_instrumentation()

    def checkpoint(self, target: Any) -> Any | None:
        return None

    def restore(self, factory: Callable[[], Any] | None, checkpoint: Any) -> Any:
        if factory is None:
            raise ValueError("generic restore requires a factory")
        return factory()
