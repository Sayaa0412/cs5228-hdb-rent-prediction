"""Configurable amenity features calculated once per HDB block."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.preprocessing.blocks import BLOCK_KEYS
from src.preprocessing.core import _require_columns, normalize_text


MRT_FEATURES = ("NEAREST_MRT_KM", "MRT_WITHIN_1KM")
SCHOOL_FEATURES = (
    "NEAREST_PRIMARY_SCHOOL_KM",
    "PRIMARY_SCHOOLS_WITHIN_1KM",
)
MALL_FEATURES = ("NEAREST_MALL_KM", "MALLS_WITHIN_1KM")


def _open_mrt(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ["STATUS", "LATITUDE", "LONGITUDE"])
    return frame.loc[normalize_text(frame["STATUS"]).eq("open")]


def _primary_schools(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ["MAINLEVEL_CODE", "LATITUDE", "LONGITUDE"])
    level = normalize_text(frame["MAINLEVEL_CODE"])
    return frame.loc[
        level.eq("primary") | level.str.startswith("mixed level (p1", na=False)
    ]


def _all_places(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ["LATITUDE", "LONGITUDE"])
    return frame


def _distance_feature_group(
    blocks: pd.DataFrame,
    places: pd.DataFrame,
    feature_names: tuple[str, str],
) -> dict[str, np.ndarray]:
    nearest, count = distance_features(blocks, places)
    return {feature_names[0]: nearest, feature_names[1]: count}


# A group declares its source file, output columns, source preparation and
# calculator. New amenity families can be added without changing the pipeline.
AMENITY_FEATURE_GROUPS: dict[str, dict] = {
    "mrt": {
        "file": "sg-mrt-stations.csv",
        "features": MRT_FEATURES,
        "prepare": _open_mrt,
        "build": _distance_feature_group,
    },
    "schools": {
        "file": "sg-schools.csv",
        "features": SCHOOL_FEATURES,
        "prepare": _primary_schools,
        "build": _distance_feature_group,
    },
    "malls": {
        "file": "sg-shopping-malls.csv",
        "features": MALL_FEATURES,
        "prepare": _all_places,
        "build": _distance_feature_group,
    },
}
AMENITY_COLUMNS = tuple(
    feature
    for group in AMENITY_FEATURE_GROUPS.values()
    for feature in group["features"]
)


def _feature_groups(features: list[str]) -> dict[str, list[str]]:
    requested = list(dict.fromkeys(features))
    unknown = sorted(set(requested).difference(AMENITY_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown amenity features: {unknown}")
    return {
        name: [feature for feature in requested if feature in group["features"]]
        for name, group in AMENITY_FEATURE_GROUPS.items()
        if any(feature in group["features"] for feature in requested)
    }


def required_amenity_sources(features: list[str]) -> dict[str, str]:
    """Return source names and files needed for the requested output columns."""
    groups = _feature_groups(features)
    return {name: AMENITY_FEATURE_GROUPS[name]["file"] for name in groups}


def distance_features(
    blocks: pd.DataFrame,
    places: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest great-circle distance and count within one kilometre."""
    _require_columns(blocks, ["LATITUDE", "LONGITUDE"])
    _require_columns(places, ["LATITUDE", "LONGITUDE"])
    if places.empty:
        raise ValueError("No amenity coordinates found")

    lat1 = np.radians(blocks["LATITUDE"].to_numpy(dtype=float))[:, None]
    lon1 = np.radians(blocks["LONGITUDE"].to_numpy(dtype=float))[:, None]
    lat2 = np.radians(places["LATITUDE"].to_numpy(dtype=float))[None, :]
    lon2 = np.radians(places["LONGITUDE"].to_numpy(dtype=float))[None, :]
    haversine = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    return distances.min(axis=1), (distances <= 1.0).sum(axis=1)


def make_amenity_features(
    blocks: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    features: list[str],
) -> pd.DataFrame:
    """Build one-row-per-block values for only the requested features."""
    groups = _feature_groups(features)
    _require_columns(blocks, [*BLOCK_KEYS, "LATITUDE", "LONGITUDE"])
    out = blocks[[*BLOCK_KEYS, "LATITUDE", "LONGITUDE"]].copy()
    for name in BLOCK_KEYS:
        out[name] = normalize_text(out[name])
    if out.duplicated(BLOCK_KEYS).any():
        raise ValueError("HDB block lookup is not unique on town, block, street")

    for name, requested in groups.items():
        if name not in tables:
            raise ValueError(f"Missing amenity table: {name}")
        group = AMENITY_FEATURE_GROUPS[name]
        places = group["prepare"](tables[name])
        values = group["build"](out, places, group["features"])
        for feature in requested:
            out[feature] = values[feature]
    return out[[*BLOCK_KEYS, *features]]


def add_amenity_features(
    frame: pd.DataFrame,
    blocks: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    features: list[str],
) -> pd.DataFrame:
    """Calculate the requested amenity features and add them to ``frame``."""
    requested = list(dict.fromkeys(features))
    if not requested:
        return frame.copy()
    lookup = make_amenity_features(blocks, tables, requested)
    return merge_amenity_features(frame, lookup, requested)


def merge_amenity_features(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Add a cached one-row-per-block amenity lookup to ``frame``."""
    requested = list(dict.fromkeys(features))
    _require_columns(frame, BLOCK_KEYS)
    base = frame.drop(columns=requested, errors="ignore")
    return base.merge(
        lookup[[*BLOCK_KEYS, *requested]],
        on=BLOCK_KEYS,
        how="left",
        validate="many_to_one",
    )
