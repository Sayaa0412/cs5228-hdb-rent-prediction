"""Time-aligned COE feature construction."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.core import _require_columns


COE_FEATURE = "PREV_MONTH_COE_MEDIAN_SGD"


def add_lagged_coe(frame: pd.DataFrame, coe: pd.DataFrame) -> pd.DataFrame:
    """Join the median COE premium from the preceding calendar month."""
    _require_columns(frame, ["RENT_APPROVAL_DATE"])
    _require_columns(coe, ["YEAR", "MONTH", "QUOTA_PREMIUM"])
    prices = coe.copy()
    prices["MONTH_KEY"] = pd.to_datetime(
        prices["YEAR"].astype(str) + "-" + prices["MONTH"],
        format="%Y-%b",
        errors="raise",
    ).dt.to_period("M").astype(str)
    prices["PREMIUM_SGD"] = pd.to_numeric(
        prices["QUOTA_PREMIUM"].astype("string").str.replace(r"[^0-9.]", "", regex=True),
        errors="raise",
    )
    monthly = prices.groupby("MONTH_KEY", observed=True)["PREMIUM_SGD"].median()
    out = frame.copy()
    previous_month = (pd.PeriodIndex(out["RENT_APPROVAL_DATE"], freq="M") - 1).astype(str)
    out[COE_FEATURE] = pd.Series(previous_month, index=out.index).map(monthly)
    return out
