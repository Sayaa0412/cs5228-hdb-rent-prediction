"""Leakage-resistant time-based dataset splitting."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.core import _require_columns


def forward_split(
    frame: pd.DataFrame,
    *,
    train_end: str,
    valid_start: str,
    valid_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split records into past training data and a later validation window."""
    _require_columns(frame, ["RENT_APPROVAL_DATE"])
    if train_end >= valid_start:
        raise ValueError("train_end must be earlier than valid_start")
    fitting = frame.loc[frame["RENT_APPROVAL_DATE"] <= train_end].copy()
    valid_mask = frame["RENT_APPROVAL_DATE"] >= valid_start
    if valid_end is not None:
        if valid_start > valid_end:
            raise ValueError("valid_start must not be later than valid_end")
        valid_mask &= frame["RENT_APPROVAL_DATE"] <= valid_end
    validation = frame.loc[valid_mask].copy()
    if fitting.empty or validation.empty:
        raise ValueError("Forward split produced an empty partition")
    if fitting["RENT_APPROVAL_DATE"].max() >= validation["RENT_APPROVAL_DATE"].min():
        raise ValueError("Forward split contains overlapping time periods")
    return fitting, validation
