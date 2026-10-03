"""Public preprocessing entry point for raw rental records."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, overload

import pandas as pd

from src.preprocessing.amenities import make_amenity_features
from src.preprocessing.blocks import BLOCK_KEYS, join_block_features
from src.preprocessing.coe import add_lagged_coe
from src.preprocessing.core import prepare_core


TARGET_COLUMN = "MONTHLY_RENT"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUXILIARY_DATA = PROJECT_ROOT / "data" / "auxiliary"
Source = tuple[str, int, int]


def _source(name: str) -> Source:
    path = (AUXILIARY_DATA / name).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=16)
def _read_auxiliary(source: Source) -> pd.DataFrame:
    return pd.read_csv(source[0])


@lru_cache(maxsize=4)
def _amenity_lookup(sources: tuple[Source, ...]) -> pd.DataFrame:
    # File signatures invalidate cached distances when an auxiliary table changes.
    return make_amenity_features(*(_read_auxiliary(source) for source in sources))


@overload
def preprocess(
    X_raw: pd.DataFrame,
    *,
    label: Literal[True],
    auxiliary_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]: ...


@overload
def preprocess(
    X_raw: pd.DataFrame,
    *,
    label: Literal[False],
    auxiliary_config: dict | None = None,
) -> pd.DataFrame: ...


def preprocess(
    X_raw: pd.DataFrame,
    *,
    label: bool,
    auxiliary_config: dict | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Clean rental fields and build the configured auxiliary features.

    With ``label=True``, ``MONTHLY_RENT`` is required and returned separately.
    With ``label=False``, only clean features are returned; a target column is
    removed if one is present. The input frame is never mutated.
    """
    enabled = {"blocks": True, "amenities": True, "coe": True}
    if auxiliary_config is not None:
        unknown = set(auxiliary_config).difference(enabled)
        if unknown:
            raise ValueError(f"Unknown auxiliary features: {sorted(unknown)}")
        if any(type(value) is not bool for value in auxiliary_config.values()):
            raise ValueError("Auxiliary feature switches must be booleans")
        enabled = {name: auxiliary_config.get(name, False) for name in enabled}
    if enabled["amenities"] and not enabled["blocks"]:
        raise ValueError("Amenity features require blocks=True")
    if label and TARGET_COLUMN not in X_raw.columns:
        raise ValueError(f"label=True requires the target column {TARGET_COLUMN!r}")

    clean = prepare_core(X_raw)
    if enabled["blocks"]:
        blocks = _read_auxiliary(_source("sg-hdb-block.csv"))
        clean = join_block_features(clean, blocks)

    if enabled["amenities"]:
        sources = tuple(_source(name) for name in (
            "sg-hdb-block.csv", "sg-mrt-stations.csv",
            "sg-schools.csv", "sg-shopping-malls.csv",
        ))
        clean = clean.merge(
            _amenity_lookup(sources), on=BLOCK_KEYS,
            how="left", validate="many_to_one",
        )

    if enabled["coe"]:
        clean = add_lagged_coe(clean, _read_auxiliary(_source("sg-coe-prices.csv")))

    if label:
        y_clean = clean.pop(TARGET_COLUMN)
        return clean, y_clean

    return clean.drop(columns=TARGET_COLUMN, errors="ignore")
