"""Test a one-month-lagged COE indicator without using future auction results.

The supplied COE table starts in January 2021. Rentals approved in that month
have no prior-month value and retain a missing numeric feature. Later rentals
use the median quota premium from all categories and rounds in the preceding
calendar month. Run from the project root: python src/coe_experiment.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from amenity_experiment import make_block_features
from catboost_experiment import CATEGORICAL, NUMERIC_BASE, PARAMETERS, predict_catboost, prepare_features
from horizon_validation import WINDOWS
from progress_analysis import DATA, OUTPUT, metrics


FEATURE = "PREV_MONTH_COE_MEDIAN_SGD"


def add_lagged_coe(frame: pd.DataFrame) -> pd.DataFrame:
    coe = pd.read_csv(DATA / "auxiliary" / "sg-coe-prices.csv")
    coe["MONTH_KEY"] = pd.to_datetime(
        coe.YEAR.astype(str) + "-" + coe.MONTH, format="%Y-%b", errors="raise"
    ).dt.to_period("M").astype(str)
    coe["PREMIUM_SGD"] = pd.to_numeric(
        coe.QUOTA_PREMIUM.str.replace(r"[^0-9.]", "", regex=True), errors="raise"
    )
    monthly = coe.groupby("MONTH_KEY", observed=True).PREMIUM_SGD.median()
    out = frame.copy()
    previous = (pd.PeriodIndex(out.RENT_APPROVAL_DATE, freq="M") - 1).astype(str)
    out[FEATURE] = pd.Series(previous, index=out.index).map(monthly)
    # No value from the rental's approval month or a later month is joined.
    if out.loc[out.RENT_APPROVAL_DATE == "2021-01", FEATURE].notna().any():
        raise ValueError("January 2021 cannot have a supplied prior-month COE value")
    return out


def predict_with_coe(fitting: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    columns = NUMERIC_BASE + [FEATURE] + CATEGORICAL
    train_x = fitting[columns].copy()
    valid_x = validation[columns].copy()
    for name in CATEGORICAL:
        train_x[name] = train_x[name].fillna("unknown").astype(str)
        valid_x[name] = valid_x[name].fillna("unknown").astype(str)
    model = CatBoostRegressor(**PARAMETERS)
    model.fit(train_x, fitting.MONTHLY_RENT, cat_features=CATEGORICAL)
    return model.predict(valid_x)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA / "train.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    enriched = add_lagged_coe(prepare_features(raw, blocks, make_block_features(blocks)))
    records = []
    for fit_end, valid_start, valid_end, months in WINDOWS:
        fit = enriched.loc[enriched.RENT_APPROVAL_DATE <= fit_end].copy()
        valid = enriched.loc[enriched.RENT_APPROVAL_DATE.between(valid_start, valid_end)].copy()
        y = valid.MONTHLY_RENT
        base = metrics(y, predict_catboost(fit, valid, amenities=False))
        expanded = metrics(y, predict_with_coe(fit, valid))
        record = {
            "horizon_months": months,
            "fit_through": fit_end,
            "validation_start": valid_start,
            "validation_end": valid_end,
            "baseline_catboost_block": base,
            "catboost_block_lagged_coe": expanded,
            "rmse_change_sgd": round(expanded["rmse_sgd"] - base["rmse_sgd"], 2),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "feature": FEATURE,
        "construction": "Median COE quota premium across categories and auction rounds in the previous calendar month",
        "windows": records,
        "limitation": "A previous-month COE premium is a market indicator, not a cause of HDB rent; the earliest rental month has no supplied lagged value.",
    }
    (OUTPUT / "coe_experiment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
