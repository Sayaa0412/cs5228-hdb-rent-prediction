"""Public preprocessing entry point for raw rental records."""

from __future__ import annotations

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
    """Build the shared clean feature table used by every model.

    With ``label=True``, ``MONTHLY_RENT`` is required and returned separately.
    With ``label=False``, only clean features are returned; a target column is
    removed if one is present. The input frame is never mutated.
    """
    if auxiliary_config is not None:
        raise NotImplementedError("auxiliary_config is not implemented yet")

    clean = prepare_core(X_raw)
    blocks = pd.read_csv(AUXILIARY_DATA / "sg-hdb-block.csv")
    clean = join_block_features(clean, blocks)

    amenities = make_amenity_features(
        blocks,
        pd.read_csv(AUXILIARY_DATA / "sg-mrt-stations.csv"),
        pd.read_csv(AUXILIARY_DATA / "sg-schools.csv"),
        pd.read_csv(AUXILIARY_DATA / "sg-shopping-malls.csv"),
    )
    clean = clean.merge(
        amenities,
        on=BLOCK_KEYS,
        how="left",
        validate="many_to_one",
    )

    coe = pd.read_csv(AUXILIARY_DATA / "sg-coe-prices.csv")
    clean = add_lagged_coe(clean, coe)

    if label:
        if TARGET_COLUMN not in clean.columns:
            raise ValueError(
                f"label=True requires the target column {TARGET_COLUMN!r}"
            )
        y_clean = clean.pop(TARGET_COLUMN)
        return clean, y_clean

    return clean.drop(columns=TARGET_COLUMN, errors="ignore")
