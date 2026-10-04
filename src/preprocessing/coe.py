"""Configurable time-aligned COE feature construction."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.preprocessing.core import _require_columns


COE_FEATURE = "PREV_MONTH_COE_MEDIAN_SGD"


def _prepare_monthly_coe(coe: pd.DataFrame) -> pd.Series:
    _require_columns(coe, ["YEAR", "MONTH", "QUOTA_PREMIUM"])
    prices = coe.copy()
    prices["MONTH_KEY"] = pd.to_datetime(
        prices["YEAR"].astype(str) + "-" + prices["MONTH"],
        format="%Y-%b",
        errors="raise",
    ).dt.to_period("M").astype(str)
    prices["PREMIUM_SGD"] = pd.to_numeric(
        prices["QUOTA_PREMIUM"].astype("string").str.replace(
            r"[^0-9.]", "", regex=True,
        ),
        errors="raise",
    )
    return prices.groupby("MONTH_KEY", observed=True)["PREMIUM_SGD"].median()


def _previous_month_median(
    frame: pd.DataFrame,
    monthly_coe: pd.Series,
) -> pd.Series:
    previous_month = (
        pd.PeriodIndex(frame["RENT_APPROVAL_DATE"], freq="M") - 1
    ).astype(str)
    return pd.Series(previous_month, index=frame.index).map(monthly_coe)


# Register new COE-derived columns here. Builders share the prepared monthly
# lookup, so adding outputs does not require changing the pipeline.
COE_FEATURE_BUILDERS: dict[
    str,
    Callable[[pd.DataFrame, pd.Series], pd.Series],
] = {
    COE_FEATURE: _previous_month_median,
}
COE_FEATURES = tuple(COE_FEATURE_BUILDERS)


def add_coe_features(
    frame: pd.DataFrame,
    coe: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Calculate the requested lagged COE features and add them to ``frame``."""
    requested = list(dict.fromkeys(features))
    unknown = sorted(set(requested).difference(COE_FEATURE_BUILDERS))
    if unknown:
        raise ValueError(f"Unknown COE features: {unknown}")
    if not requested:
        return frame.copy()

    _require_columns(frame, ["RENT_APPROVAL_DATE"])
    monthly_coe = _prepare_monthly_coe(coe)
    out = frame.drop(columns=requested, errors="ignore").copy()
    for feature in requested:
        out[feature] = COE_FEATURE_BUILDERS[feature](out, monthly_coe)
    return out
