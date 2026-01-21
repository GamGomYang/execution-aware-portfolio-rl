from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compute_env_signature(
    asset_list: Sequence[str],
    L: int | None,
    Lv: int | None,
    feature_flags: Mapping[str, Any],
    cost_params: Mapping[str, Any],
    schema_version: str,
) -> str:
    payload = {
        "asset_list": list(asset_list),
        "num_assets": len(asset_list),
        "L": L,
        "Lv": Lv,
        "feature_flags": dict(feature_flags),
        "cost_params": dict(cost_params),
        "schema_version": schema_version,
    }
    return sha256_bytes(canonical_json(payload))
