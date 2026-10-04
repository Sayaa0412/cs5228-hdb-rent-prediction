"""Configurable HDB block feature construction."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.preprocessing.core import _require_columns, normalize_text


BLOCK_KEYS = ["TOWN", "BLOCK", "STREET"]
BLOCK_SOURCE_COLUMNS = [
    "LATITUDE",
    "LONGITUDE",
    "MAX_FLOOR",
    "YEAR_COMPLETED",
    "REGION",
]


def _source_column(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda frame: frame[name]


def _building_age(frame: pd.DataFrame) -> pd.Series:
    return frame["APPROVAL_YEAR"] - frame["YEAR_COMPLETED"]


# Each entry declares the lookup columns needed by one output feature and how
# to calculate it. Adding another block feature only requires a new entry.
BLOCK_FEATURE_BUILDERS: dict[
    str,
    tuple[list[str], Callable[[pd.DataFrame], pd.Series]],
] = {
    "LATITUDE": (["LATITUDE"], _source_column("LATITUDE")),
    "LONGITUDE": (["LONGITUDE"], _source_column("LONGITUDE")),
    "MAX_FLOOR": (["MAX_FLOOR"], _source_column("MAX_FLOOR")),
    "YEAR_COMPLETED": (["YEAR_COMPLETED"], _source_column("YEAR_COMPLETED")),
    "REGION": (["REGION"], _source_column("REGION")),
    "BUILDING_AGE": (["YEAR_COMPLETED"], _building_age),
}
BLOCK_FEATURES = tuple(BLOCK_FEATURE_BUILDERS)


def _requested_features(features: list[str]) -> list[str]:
    requested = list(dict.fromkeys(features))
    unknown = sorted(set(requested).difference(BLOCK_FEATURE_BUILDERS))
    if unknown:
        raise ValueError(f"Unknown block features: {unknown}")
    return requested


def prepare_block_lookup(
    blocks: pd.DataFrame,
    source_columns: list[str],
) -> pd.DataFrame:
    """Normalize one-row-per-block lookup columns needed by the request."""
    columns = list(dict.fromkeys(source_columns))
    _require_columns(blocks, [*BLOCK_KEYS, *columns])
    lookup = blocks[[*BLOCK_KEYS, *columns]].copy()
    for name in BLOCK_KEYS:
        lookup[name] = normalize_text(lookup[name])
    if lookup.duplicated(BLOCK_KEYS).any():
        raise ValueError("HDB block lookup is not unique on town, block, street")
    return lookup


def add_block_features(
    frame: pd.DataFrame,
    blocks: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Calculate the requested block features and add them to ``frame``."""
    requested = _requested_features(features)
    if not requested:
        return frame.copy()

    _require_columns(frame, BLOCK_KEYS)
    dependencies = list(dict.fromkeys(
        column
        for feature in requested
        for column in BLOCK_FEATURE_BUILDERS[feature][0]
    ))
    if "BUILDING_AGE" in requested:
        _require_columns(frame, ["APPROVAL_YEAR"])

    base = frame.drop(columns=requested, errors="ignore")
    joined = base.merge(
        prepare_block_lookup(blocks, dependencies),
        on=BLOCK_KEYS,
        how="left",
        validate="many_to_one",
    )
    for feature in requested:
        joined[feature] = BLOCK_FEATURE_BUILDERS[feature][1](joined)
    return joined[[*base.columns, *requested]]
