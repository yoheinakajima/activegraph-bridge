"""Effect codecs: how live Python values become recorded JSON and back.

Agent code does not consume JSON — it consumes SDK response objects,
Pydantic models, dataclasses, exceptions. A codec is the two-way contract
that makes replay honest:

- ``encode_request`` / ``encode_response`` reduce a live value to a JSON
  document the event log can store and hash.
- ``decode_response`` rehydrates the recorded document into something the
  original agent code can use *without the live service*.

:class:`AutoCodec` is the default and handles the common shapes:

==============  =======================================  =========================
live value      recorded as                              rehydrated as
==============  =======================================  =========================
JSON primitive  itself                                   itself
Pydantic model  ``model_dump(mode="json")`` + class ref  the model class if it
                                                         imports, else `AttrBox`
dataclass       ``asdict`` + class ref                   the dataclass if it
                                                         imports, else `AttrBox`
bytes           base64                                   bytes
datetime/date   ISO-8601                                 datetime/date
Decimal         canonical string                         Decimal
Exception       class ref + message + args               the exception class if it
                                                         imports, else
                                                         ``ReplayedEffectFailure``
anything else   ``repr()`` (LOSSY)                       :class:`OpaqueValue`
==============  =======================================  =========================

A lossy encoding is never silent: the codec reports it, the session
records a finding, and the run's replayability grade is capped
accordingly. Custom codecs implement :class:`EffectCodec` and are the
right tool when an integration knows its exact wire types.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import importlib
from decimal import Decimal
from typing import Any, Iterator, Protocol, runtime_checkable

from ._canonical import JsonValue
from .errors import ReplayedEffectFailure

__all__ = [
    "EffectCodec",
    "AutoCodec",
    "JsonCodec",
    "AttrBox",
    "OpaqueValue",
    "encode_exception",
    "decode_exception",
    "find_lossy",
]

_TAG = "$bridge"  # envelope key marking a typed encoding


@runtime_checkable
class EffectCodec(Protocol):
    """The two-way serialization contract for one effect kind.

    Codecs are stateless: fidelity is judged from the encoded document
    itself (see :func:`find_lossy`), so one codec instance is safe to
    share across concurrent sessions.
    """

    name: str

    def encode_request(self, value: Any) -> JsonValue: ...

    def encode_response(self, value: Any) -> JsonValue: ...

    def decode_response(self, value: JsonValue) -> Any: ...


def find_lossy(doc: JsonValue) -> list[str]:
    """Walk an encoded document and list every lossy (repr-only) capture.

    Used by the session after each encode to record honesty findings —
    a run whose responses contain opaque captures can play back its
    output but cannot claim faithful re-execution, and the replayability
    report says so.
    """
    lossy: list[str] = []
    _walk_lossy(doc, lossy)
    return lossy


def _walk_lossy(doc: Any, out: list[str]) -> None:
    if isinstance(doc, list):
        for v in doc:
            _walk_lossy(v, out)
        return
    if not isinstance(doc, dict):
        return
    if doc.get(_TAG) == "opaque":
        out.append(f"value of type {doc.get('class', '?')} captured as repr() only")
        return
    for v in doc.values():
        _walk_lossy(v, out)


class AttrBox:
    """Attribute-and-index access over a plain JSON document.

    When a recorded Pydantic/SDK response class cannot be imported at
    replay time, the recorded document is wrapped in an ``AttrBox`` so
    idiomatic agent code — ``resp.choices[0].message.content`` — still
    works. Equality compares against the underlying document, and
    ``model_dump()`` / ``to_dict()`` are provided for code that
    round-trips responses back to dicts.

    Round-trip stable: a box remembers the exact encoded document it was
    decoded from, and re-encoding returns that document verbatim — so a
    served response fed into a later request hashes identically to the
    recording.
    """

    __slots__ = ("_data", "_encoded")

    def __init__(self, data: Any, *, encoded: Any = None) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_encoded", encoded)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return AttrBox(value)
        return value

    def __getattr__(self, item: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict) and item in data:
            return self._wrap(data[item])
        raise AttributeError(item)

    def __getitem__(self, item: Any) -> Any:
        return self._wrap(object.__getattribute__(self, "_data")[item])

    def __iter__(self) -> Iterator[Any]:
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return iter(data)
        return (self._wrap(v) for v in data)

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_data"))

    def __contains__(self, item: Any) -> bool:
        return item in object.__getattribute__(self, "_data")

    def __eq__(self, other: Any) -> bool:
        data = object.__getattribute__(self, "_data")
        if isinstance(other, AttrBox):
            other = object.__getattribute__(other, "_data")
        return data == other

    def __repr__(self) -> str:
        return f"AttrBox({object.__getattribute__(self, '_data')!r})"

    def get(self, key: str, default: Any = None) -> Any:
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return self._wrap(data.get(key, default))
        return default

    def model_dump(self, **_: Any) -> Any:
        return object.__getattribute__(self, "_data")

    def to_dict(self) -> Any:
        return object.__getattribute__(self, "_data")


class OpaqueValue:
    """A recorded value that could only be captured as ``repr()``.

    Reading ``.repr`` is fine; anything else raises so replay fails loud
    instead of silently handing agent code a husk.
    """

    __slots__ = ("repr", "class_name")

    def __init__(self, repr_: str, class_name: str) -> None:
        object.__setattr__(self, "repr", repr_)
        object.__setattr__(self, "class_name", class_name)

    def __getattr__(self, item: str) -> Any:
        raise AttributeError(
            f"recorded value of type {object.__getattribute__(self, 'class_name')!r} "
            f"was captured as repr() only; attribute {item!r} is not available on "
            f"replay. Register a faithful EffectCodec for this effect kind."
        )

    def __repr__(self) -> str:
        return f"OpaqueValue({object.__getattribute__(self, 'repr')})"


def _class_ref(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _import_ref(ref: str) -> type | None:
    try:
        module_name, qualname = ref.split(":", 1)
        target: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
        return target if isinstance(target, type) else None
    except Exception:
        return None


def encode_exception(exc: BaseException) -> JsonValue:
    """Encode an exception for the event log (class ref, message, args)."""
    return {
        _TAG: "exception",
        "class": _class_ref(exc),
        "message": str(exc),
        "args": [a if isinstance(a, (str, int, float, bool, type(None))) else repr(a) for a in exc.args],
    }


def decode_exception(doc: JsonValue) -> BaseException:
    """Rehydrate a recorded exception; fall back to ReplayedEffectFailure.

    Reconstruction is best-effort by design: the class is imported and
    called with the recorded message. Exceptions whose constructors need
    richer arguments come back as :class:`ReplayedEffectFailure` carrying
    the original class name — agent code catching broad classes keeps
    working, and nothing is ever swallowed.
    """
    class_ref = str(doc.get("class", ""))
    message = str(doc.get("message", ""))
    cls = _import_ref(class_ref)
    if cls is not None and issubclass(cls, BaseException):
        try:
            return cls(message)
        except Exception:
            pass
    return ReplayedEffectFailure(
        f"replayed failure ({class_ref}): {message}", original_class=class_ref
    )


class AutoCodec:
    """Default codec: faithful for the common shapes, loud when lossy.

    Stateless — one instance serves every session.
    """

    name = "auto"

    # -- encode ------------------------------------------------------------

    def encode_request(self, value: Any) -> JsonValue:
        return self._encode(value)

    def encode_response(self, value: Any) -> JsonValue:
        return self._encode(value)

    def _encode(self, value: Any) -> JsonValue:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, AttrBox):  # round-trip: emit the original document
            encoded = object.__getattribute__(value, "_encoded")
            if encoded is not None:
                return encoded
            return self._encode(object.__getattribute__(value, "_data"))
        if isinstance(value, OpaqueValue):  # round-trip: reproduce the opaque tag
            return {
                _TAG: "opaque",
                "class": object.__getattribute__(value, "class_name"),
                "repr": object.__getattribute__(value, "repr"),
            }
        if isinstance(value, (list, tuple)):
            return [self._encode(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return {_TAG: "set", "items": sorted(self._encode(v) for v in value)}
        if isinstance(value, dict):
            return {str(k): self._encode(v) for k, v in value.items()}
        if isinstance(value, bytes):
            return {_TAG: "bytes", "b64": base64.b64encode(value).decode("ascii")}
        if isinstance(value, Decimal):
            return {_TAG: "decimal", "value": str(value)}
        if isinstance(value, _dt.datetime):
            return {_TAG: "datetime", "iso": value.isoformat()}
        if isinstance(value, _dt.date):
            return {_TAG: "date", "iso": value.isoformat()}
        if isinstance(value, BaseException):
            return encode_exception(value)
        dump = getattr(value, "model_dump", None)
        if callable(dump):  # Pydantic v2 model (or anything speaking its protocol)
            try:
                return {
                    _TAG: "model",
                    "class": _class_ref(value),
                    "data": self._encode(dump(mode="json")),
                }
            except TypeError:
                return {_TAG: "model", "class": _class_ref(value), "data": self._encode(dump())}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                _TAG: "dataclass",
                "class": _class_ref(value),
                "data": self._encode(dataclasses.asdict(value)),
            }
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return {
                    _TAG: "model",
                    "class": _class_ref(value),
                    "data": self._encode(to_dict()),
                }
            except Exception:
                pass
        # Last resort: repr. Recorded, but tagged so find_lossy() reports
        # it and the run's replayability report stays honest.
        return {_TAG: "opaque", "class": _class_ref(value), "repr": repr(value)}

    # -- decode ------------------------------------------------------------

    def decode_response(self, value: JsonValue) -> Any:
        return self._decode(value)

    def _decode(self, value: JsonValue) -> Any:
        if isinstance(value, list):
            return [self._decode(v) for v in value]
        if not isinstance(value, dict):
            return value
        tag = value.get(_TAG)
        if tag is None:
            return {k: self._decode(v) for k, v in value.items()}
        if tag == "set":
            return set(self._decode(value["items"]))
        if tag == "bytes":
            return base64.b64decode(value["b64"])
        if tag == "decimal":
            return Decimal(value["value"])
        if tag == "datetime":
            return _dt.datetime.fromisoformat(value["iso"])
        if tag == "date":
            return _dt.date.fromisoformat(value["iso"])
        if tag == "exception":
            return decode_exception(value)
        if tag in ("model", "dataclass"):
            data = self._decode(value["data"])
            cls = _import_ref(str(value.get("class", "")))
            if cls is not None:
                validate = getattr(cls, "model_validate", None)
                if callable(validate):
                    try:
                        return validate(data)
                    except Exception:
                        pass
                if dataclasses.is_dataclass(cls) and isinstance(data, dict):
                    try:
                        return cls(**data)
                    except Exception:
                        pass
            return AttrBox(data, encoded=value)
        if tag == "opaque":
            return OpaqueValue(str(value.get("repr", "")), str(value.get("class", "")))
        return {k: self._decode(v) for k, v in value.items()}


class JsonCodec:
    """Strict pass-through codec for effects that already speak JSON.

    Values must already be JSON-shaped; anything else raises at record
    time. Use this for effects where you control both sides and want the
    log to contain exactly what the agent saw.
    """

    name = "json"

    def encode_request(self, value: Any) -> JsonValue:
        return self._check(value)

    def encode_response(self, value: Any) -> JsonValue:
        return self._check(value)

    def decode_response(self, value: JsonValue) -> Any:
        return value

    @staticmethod
    def _check(value: Any) -> JsonValue:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, list):
            return [JsonCodec._check(v) for v in value]
        if isinstance(value, dict):
            return {str(k): JsonCodec._check(v) for k, v in value.items()}
        raise TypeError(
            f"JsonCodec requires JSON-shaped values; got {type(value).__name__}. "
            f"Use AutoCodec (the default) or a custom EffectCodec."
        )


DEFAULT_CODEC = AutoCodec()
