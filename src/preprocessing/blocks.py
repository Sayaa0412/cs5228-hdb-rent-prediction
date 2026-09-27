"""HDB block lookup preparation and rental-table enrichment."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.core import _require_columns, normalize_text


BLOCK_KEYS = ["TOWN", "BLOCK", "STREET"]
BLOCK_FEATURES = ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED", "REGION"]


def prepare_block_lookup(blocks: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate the one-row-per-block auxiliary lookup."""
    _require_columns(blocks, [*BLOCK_KEYS, *BLOCK_FEATURES])
    lookup = blocks.copy()
    for name in BLOCK_KEYS:
        lookup[name] = normalize_text(lookup[name])
    if lookup.duplicated(BLOCK_KEYS).any():
        raise ValueError("HDB block lookup is not unique on town, block, street")
    return lookup[[*BLOCK_KEYS, *BLOCK_FEATURES]]


def join_block_features(core: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Add block-level spatial features to a preprocessed rental frame."""
    _require_columns(core, [*BLOCK_KEYS, "APPROVAL_YEAR"])
    lookup = prepare_block_lookup(blocks)
    out = core.merge(lookup, on=BLOCK_KEYS, how="left", validate="many_to_one")
    out["BUILDING_AGE"] = out["APPROVAL_YEAR"] - out["YEAR_COMPLETED"]
    return out
