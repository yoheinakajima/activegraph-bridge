"""Bridged tools: one decorator to make a function replay-honest.

::

    from activegraph_bridge import bridge_tool

    @bridge_tool(side_effect="read")
    def lookup_order(order_id: str) -> dict:
        return crm.get(order_id)

    @bridge_tool(side_effect="write")
    def send_email(to: str, subject: str, body: str, idempotency_key: str = ""):
        return mailer.send(to, subject, body, key=idempotency_key)

Inside a wrapped execution, every call becomes a recorded effect —
canonicalized arguments, content hash, response captured. During
``verify()`` and fork prefixes the recorded result is served and the
function body **does not run**; during fork tails write tools follow the
fail-closed policy. Outside any session the function runs untouched, so
decorating costs nothing for undecorated use.

The request payload shape (``{"tool": name, "args": {...}}``) matches
ActiveGraph's native tool-cache hashing convention, so bridged tool
calls hash identically to native ones with the same arguments.

Idempotency: if the function signature declares an ``idempotency_key``
parameter and the caller does not supply one, the bridge injects a
deterministic key derived from the run and the canonical request —
identical requests in one run share a key, which is exactly what
external dedupe wants. The key is injected into the call, not the hash.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from ._canonical import content_hash
from .codecs import EffectCodec
from .policy import Footprint, ReplaySource, SideEffect
from .session import current_session

__all__ = ["bridge_tool", "wrap_tool"]


def bridge_tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    side_effect: SideEffect = "unknown",
    footprint: Footprint | None = None,
    replay_source: ReplaySource = "recorded",
    observables: tuple[str, ...] = (),
    codec: EffectCodec | None = None,
) -> Any:
    """Declare a function as a mediated tool. See module docstring."""

    def apply(f: Callable) -> Callable:
        return wrap_tool(
            f,
            name=name,
            side_effect=side_effect,
            footprint=footprint,
            replay_source=replay_source,
            observables=observables,
            codec=codec,
        )

    if fn is not None:
        return apply(fn)
    return apply


def wrap_tool(
    fn: Callable,
    *,
    name: str | None = None,
    side_effect: SideEffect = "unknown",
    footprint: Footprint | None = None,
    replay_source: ReplaySource = "recorded",
    observables: tuple[str, ...] = (),
    codec: EffectCodec | None = None,
) -> Callable:
    """Function form of :func:`bridge_tool` (useful for tools you import)."""
    tool_name = name or fn.__name__
    signature = inspect.signature(fn)
    wants_key = "idempotency_key" in signature.parameters
    is_async = inspect.iscoroutinefunction(fn)

    def bind_request(args: tuple, kwargs: dict) -> tuple[dict[str, Any], dict]:
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arguments.pop("idempotency_key", None)
        except TypeError:
            arguments = {"args": list(args), "kwargs": dict(kwargs)}
        return {"tool": tool_name, "args": arguments}, arguments

    def prepare(args: tuple, kwargs: dict, session: Any) -> tuple[dict, dict]:
        request, _ = bind_request(args, kwargs)
        call_kwargs = dict(kwargs)
        if wants_key and "idempotency_key" not in call_kwargs and session is not None:
            call_kwargs["idempotency_key"] = content_hash(
                {"run": session.run_id, "request": request}
            )[:32]
        return request, call_kwargs

    if is_async:

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            session = current_session()
            if session is None:
                return await fn(*args, **kwargs)
            request, call_kwargs = prepare(args, kwargs, session)
            return await session.aeffect(
                f"tool.{tool_name}",
                request,
                lambda: fn(*args, **call_kwargs),
                name=tool_name,
                side_effect=side_effect,
                footprint=footprint,
                replay_source=replay_source,
                observables=observables,
                codec=codec,
                category="tool",
            )

        wrapper: Callable = async_wrapper
    else:

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            session = current_session()
            if session is None:
                return fn(*args, **kwargs)
            request, call_kwargs = prepare(args, kwargs, session)
            return session.effect(
                f"tool.{tool_name}",
                request,
                lambda: fn(*args, **call_kwargs),
                name=tool_name,
                side_effect=side_effect,
                footprint=footprint,
                replay_source=replay_source,
                observables=observables,
                codec=codec,
                category="tool",
            )

        wrapper = sync_wrapper

    wrapper.tool_name = tool_name  # type: ignore[attr-defined]
    wrapper.side_effect = side_effect  # type: ignore[attr-defined]
    wrapper.footprint = footprint  # type: ignore[attr-defined]
    setattr(wrapper, "__wrapped__", fn)
    return wrapper
