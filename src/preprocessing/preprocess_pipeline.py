"""Public preprocessing entry point for raw rental records."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, overload

import pandas as pd

from src.preprocessing.amenities import (
    AMENITY_COLUMNS,
    make_amenity_features,
    merge_amenity_features,
    required_amenity_sources,
)
from src.preprocessing.blocks import BLOCK_FEATURES, add_block_features
from src.preprocessing.coe import COE_FEATURES, add_coe_features
from src.preprocessing.core import prepare_core


TARGET_COLUMN = "MONTHLY_RENT"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUXILIARY_DATA = PROJECT_ROOT / "data" / "auxiliary"
Source = tuple[str, int, int]

AUXILIARY_FEATURES = {
    "blocks": BLOCK_FEATURES,
    "amenities": AMENITY_COLUMNS,
    "coe": COE_FEATURES,
}
DEFAULT_AUXILIARY_CONFIG = {
    name: list(features)
    for name, features in AUXILIARY_FEATURES.items()
}


def _source(name: str) -> Source:
    path = (AUXILIARY_DATA / name).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=16)
def _read_auxiliary(source: Source) -> pd.DataFrame:
    return pd.read_csv(source[0])


@lru_cache(maxsize=8)
def _amenity_lookup(
    features: tuple[str, ...],
    sources: tuple[tuple[str, Source], ...],
) -> pd.DataFrame:
    tables = {name: _read_auxiliary(source) for name, source in sources}
    blocks = tables.pop("blocks")
    return make_amenity_features(blocks, tables, list(features))


def _normalize_auxiliary_config(
    auxiliary_config: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    if auxiliary_config is None:
        return {
            name: list(features)
            for name, features in DEFAULT_AUXILIARY_CONFIG.items()
        }

    unknown_groups = sorted(set(auxiliary_config).difference(AUXILIARY_FEATURES))
    if unknown_groups:
        raise ValueError(f"Unknown auxiliary groups: {unknown_groups}")

    normalized = {}
    for name, features in auxiliary_config.items():
        if not isinstance(features, list):
            raise ValueError(f"Auxiliary group {name!r} must contain a feature list")
        requested = list(dict.fromkeys(features))
        unknown_features = sorted(
            set(requested).difference(AUXILIARY_FEATURES[name])
        )
        if unknown_features:
            raise ValueError(f"Unknown {name} features: {unknown_features}")
        if requested:
            normalized[name] = requested
    return normalized


@overload
def preprocess(
    X_raw: pd.DataFrame,
    *,
    label: Literal[True],
    auxiliary_config: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.Series]: ...


@overload
def preprocess(
    X_raw: pd.DataFrame,
    *,
    label: Literal[False],
    auxiliary_config: dict[str, list[str]] | None = None,
) -> pd.DataFrame: ...


def preprocess(
    X_raw: pd.DataFrame,
    *,
    label: bool,
    auxiliary_config: dict[str, list[str]] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Clean rental fields and add only the configured auxiliary features."""
    requested = _normalize_auxiliary_config(auxiliary_config)
    if label and TARGET_COLUMN not in X_raw.columns:
        raise ValueError(f"label=True requires the target column {TARGET_COLUMN!r}")

    clean = prepare_core(X_raw)

    block_features = requested.get("blocks", [])
    if block_features:
        clean = add_block_features(
            clean,
            _read_auxiliary(_source("sg-hdb-block.csv")),
            block_features,
        )

    amenity_features = requested.get("amenities", [])
    if amenity_features:
        sources = [("blocks", _source("sg-hdb-block.csv"))]
        sources.extend(
            (name, _source(filename))
            for name, filename in required_amenity_sources(
                amenity_features,
            ).items()
        )
        clean = merge_amenity_features(
            clean,
            _amenity_lookup(tuple(amenity_features), tuple(sources)),
            amenity_features,
        )

    coe_features = requested.get("coe", [])
    if coe_features:
        clean = add_coe_features(
            clean,
            _read_auxiliary(_source("sg-coe-prices.csv")),
            coe_features,
        )

    if label:
        y_clean = clean.pop(TARGET_COLUMN)
        return clean, y_clean
    return clean.drop(columns=TARGET_COLUMN, errors="ignore")
