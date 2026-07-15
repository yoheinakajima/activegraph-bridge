"""Scoped SDK client proxies: intercept exactly the methods that matter.

An instrumented client is a transparent proxy: every attribute passes
through untouched except the declared method paths, whose calls route
through the effect broker. The proxy is *scoped* — it wraps one client
instance you choose, not the SDK module — so other clients in the
process stay untouched, and nothing is monkeypatched.

::

    from activegraph_bridge import instrument

    client = instrument.openai(OpenAI())        # sync or async client
    client = instrument.anthropic(Anthropic())

    # any SDK, generically:
    client = instrument.wrap_client(
        vector_db,
        methods=["collections.query"],
        kind_prefix="vectordb",
        category="retrieval",
        side_effect="read",
    )

During recording the real SDK call happens and its response (Pydantic
models included) is captured faithfully. During verify/fork-prefix the
recorded response is rehydrated — as the SDK class when it imports, or
as an :class:`~activegraph_bridge.codecs.AttrBox` with identical
attribute access when it doesn't — and **no network call happens**.

Known limitation (v0.1): SDK *streaming* responses (``stream=True``)
are captured as opaque values — the run stays inspectable but will not
claim boundary-grade fidelity for those calls. Record non-streaming
variants for verification workflows, or wrap streaming calls in your
own effect with a chunk-assembling codec.
"""

from __future__ import annotations

import inspect
from typing import Any

from .codecs import EffectCodec
from .policy import SideEffect
from .session import aeffect, effect

__all__ = ["wrap_client", "openai", "anthropic"]


class _ClientProxy:
    """Attribute-transparent proxy routing declared method paths."""

    __slots__ = ("_target", "_routes", "_path", "_config")

    def __init__(self, target: Any, routes: frozenset[str], path: tuple[str, ...], config: dict) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_routes", routes)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_config", config)

    def __getattr__(self, item: str) -> Any:
        target = object.__getattribute__(self, "_target")
        routes = object.__getattribute__(self, "_routes")
        path = object.__getattribute__(self, "_path")
        config = object.__getattribute__(self, "_config")
        value = getattr(target, item)
        dotted = ".".join(path + (item,))
        if dotted in routes:
            return _wrap_method(value, dotted, config)
        if any(r.startswith(dotted + ".") for r in routes):
            return _ClientProxy(value, routes, path + (item,), config)
        return value

    def __setattr__(self, item: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), item, value)

    def __repr__(self) -> str:
        return f"instrumented({object.__getattribute__(self, '_target')!r})"


def _wrap_method(fn: Any, dotted: str, config: dict) -> Any:
    kind = f"{config['kind_prefix']}.{dotted}" if config["kind_prefix"] else dotted
    name = config["name"] or dotted
    common = dict(
        name=name,
        side_effect=config["side_effect"],
        codec=config["codec"],
        category=config["category"],
    )

    if inspect.iscoroutinefunction(fn):

        async def async_method(*args: Any, **kwargs: Any) -> Any:
            request = {"args": list(args), "kwargs": kwargs}
            return await aeffect(kind, request, lambda: fn(*args, **kwargs), **common)

        return async_method

    def method(*args: Any, **kwargs: Any) -> Any:
        request = {"args": list(args), "kwargs": kwargs}
        return effect(kind, request, lambda: fn(*args, **kwargs), **common)

    return method


def wrap_client(
    client: Any,
    *,
    methods: list[str],
    kind_prefix: str = "",
    category: str = "model",
    side_effect: SideEffect = "read",
    codec: EffectCodec | None = None,
    name: str | None = None,
) -> Any:
    """Instrument ``client`` so calls to ``methods`` become effects.

    ``methods`` are dotted paths relative to the client
    (``"chat.completions.create"``). Everything else proxies through
    unchanged. Works with sync and async clients; outside a session the
    calls run live and unrecorded.
    """
    return _ClientProxy(
        client,
        frozenset(methods),
        (),
        {
            "kind_prefix": kind_prefix,
            "category": category,
            "side_effect": side_effect,
            "codec": codec,
            "name": name,
        },
    )


def openai(client: Any) -> Any:
    """Instrument an OpenAI (or AsyncOpenAI) client's model surfaces."""
    return wrap_client(
        client,
        methods=[
            "chat.completions.create",
            "responses.create",
            "embeddings.create",
        ],
        kind_prefix="openai",
        category="model",
        side_effect="read",
    )


def anthropic(client: Any) -> Any:
    """Instrument an Anthropic (or AsyncAnthropic) client's model surfaces."""
    return wrap_client(
        client,
        methods=["messages.create"],
        kind_prefix="anthropic",
        category="model",
        side_effect="read",
    )
