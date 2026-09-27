"""Deterministic preprocessing for the core rental table."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


TEXT_COLUMNS = ("TOWN", "BLOCK", "STREET", "FLAT_MODEL")
REQUIRED_COLUMNS = (*TEXT_COLUMNS, "FLAT_TYPE", "RENT_APPROVAL_DATE", "LEASE_COMMENCE_DATE")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize casing and whitespace while retaining pandas string values."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def clean_flat_type(series: pd.Series) -> pd.Series:
    """Normalize variants such as ``4 room`` and ``4-room`` to ``4-room``."""
    return normalize_text(series).str.replace(
        r"^(\d)\s*[- ]\s*room$",
        r"\1-room",
        regex=True,
    )


def prepare_core(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw rental fields and add model-ready time features."""
    _require_columns(frame, REQUIRED_COLUMNS)
    out = frame.copy()
    for name in TEXT_COLUMNS:
        out[name] = normalize_text(out[name])
    out["FLAT_TYPE"] = clean_flat_type(out["FLAT_TYPE"])

    dates = pd.to_datetime(out["RENT_APPROVAL_DATE"], format="%Y-%m", errors="raise")
    lease_start = pd.to_numeric(out["LEASE_COMMENCE_DATE"], errors="raise")
    out["APPROVAL_YEAR"] = dates.dt.year
    out["MONTH_INDEX"] = (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    out["LEASE_AGE"] = dates.dt.year - lease_start
    return out
