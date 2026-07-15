"""Canonical JSON and content hashing.

Every request and response the bridge records is reduced to a canonical
JSON value before hashing, using the same conventions as ActiveGraph's
own replay caches (``activegraph.tools.cache``): Pydantic models dump to
JSON mode, Decimals become their canonical strings, and the final hash is
the SHA-256 of a sorted-key, separator-compact ``json.dumps``. Matching
the host library's convention is deliberate — a bridge-recorded tool call
hashes identically to a native ActiveGraph tool call with the same
arguments, so content-addressed tooling composes across both.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from activegraph.tools.cache import canonicalize_args

JsonValue = Any  # None | bool | int | float | str | list | dict — after canonicalization

__all__ = ["JsonValue", "canonical_json", "canonicalize", "content_hash"]


def canonicalize(value: Any) -> JsonValue:
    """Normalize ``value`` into a JSON-stable shape.

    Delegates to ActiveGraph's ``canonicalize_args`` (Pydantic-aware,
    Decimal-safe) and additionally flattens tuples and sets so hashing
    never depends on Python container identity.
    """
    if isinstance(value, tuple):
        return [canonicalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(canonicalize(v) for v in value)
    out = canonicalize_args(value)
    if isinstance(out, dict):
        return {k: canonicalize(v) for k, v in out.items()}
    if isinstance(out, list):
        return [canonicalize(v) for v in out]
    return out


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to the canonical JSON string used for hashing."""
    return json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    """SHA-256 hex digest of the canonical JSON form of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
