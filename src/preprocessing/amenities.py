"""Static amenity-distance features calculated once per HDB block."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.blocks import BLOCK_KEYS
from src.preprocessing.core import _require_columns, normalize_text


AMENITY_COLUMNS = [
    "NEAREST_MRT_KM",
    "MRT_WITHIN_1KM",
    "NEAREST_PRIMARY_SCHOOL_KM",
    "PRIMARY_SCHOOLS_WITHIN_1KM",
    "NEAREST_MALL_KM",
    "MALLS_WITHIN_1KM",
]


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
    mrt: pd.DataFrame,
    schools: pd.DataFrame,
    malls: pd.DataFrame,
) -> pd.DataFrame:
    """Build the six static amenity features used by the existing experiments."""
    _require_columns(blocks, [*BLOCK_KEYS, "LATITUDE", "LONGITUDE"])
    _require_columns(mrt, ["STATUS", "LATITUDE", "LONGITUDE"])
    _require_columns(schools, ["MAINLEVEL_CODE", "LATITUDE", "LONGITUDE"])
    _require_columns(malls, ["LATITUDE", "LONGITUDE"])

    out = blocks[[*BLOCK_KEYS, "LATITUDE", "LONGITUDE"]].copy()
    for name in BLOCK_KEYS:
        out[name] = normalize_text(out[name])
    if out.duplicated(BLOCK_KEYS).any():
        raise ValueError("HDB block lookup is not unique on town, block, street")

    open_mrt = mrt.loc[normalize_text(mrt["STATUS"]).eq("open")]
    school_level = normalize_text(schools["MAINLEVEL_CODE"])
    primary_schools = schools.loc[
        school_level.eq("primary")
        | school_level.str.startswith("mixed level (p1", na=False)
    ]

    feature_specs = (
        (open_mrt, "NEAREST_MRT_KM", "MRT_WITHIN_1KM"),
        (
            primary_schools,
            "NEAREST_PRIMARY_SCHOOL_KM",
            "PRIMARY_SCHOOLS_WITHIN_1KM",
        ),
        (malls, "NEAREST_MALL_KM", "MALLS_WITHIN_1KM"),
    )
    for places, distance_name, count_name in feature_specs:
        nearest, count = distance_features(out, places)
        out[distance_name] = nearest
        out[count_name] = count
    return out.drop(columns=["LATITUDE", "LONGITUDE"])
