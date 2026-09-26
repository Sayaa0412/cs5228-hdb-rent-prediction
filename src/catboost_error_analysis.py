"""Audit CatBoost errors on the final six-month forward holdout.

Run from the project root: python src/catboost_error_analysis.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from amenity_experiment import make_block_features
from catboost_experiment import predict_catboost, prepare_features
from progress_analysis import DATA, OUTPUT, TRAIN_END, VALID_START, metrics


def grouped_error(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    return (
        frame.groupby(key, observed=True)
        .agg(
            rows=("ABS_ERROR", "size"),
            mae_sgd=("ABS_ERROR", "mean"),
            mean_signed_error_sgd=("SIGNED_ERROR", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw_train = pd.read_csv(DATA / "train.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    amenities = make_block_features(blocks)
    train = prepare_features(raw_train, blocks, amenities)
    fitting = train.loc[train.RENT_APPROVAL_DATE <= TRAIN_END].copy()
    validation = train.loc[train.RENT_APPROVAL_DATE >= VALID_START].copy()
    prediction = predict_catboost(fitting, validation, amenities=True)
    errors = validation[["RENT_APPROVAL_DATE", "TOWN", "FLAT_TYPE", "MONTHLY_RENT"]].copy()
    errors["PREDICTED_RENT"] = prediction
    errors["SIGNED_ERROR"] = prediction - errors.MONTHLY_RENT
    errors["ABS_ERROR"] = np.abs(errors.SIGNED_ERROR)
    errors["RENT_BAND"] = pd.cut(
        errors.MONTHLY_RENT,
        bins=[-np.inf, 2000, 3000, 4000, np.inf],
        labels=["up to 2000", "2001 to 3000", "3001 to 4000", "above 4000"],
    )
    for key, name in (
        ("FLAT_TYPE", "catboost_error_by_flat_type.csv"),
        ("RENT_APPROVAL_DATE", "catboost_error_by_month.csv"),
        ("TOWN", "catboost_error_by_town.csv"),
        ("RENT_BAND", "catboost_error_by_rent_band.csv"),
    ):
        grouped_error(errors, key).to_csv(OUTPUT / name, index=False)

    overall = metrics(errors.MONTHLY_RENT, prediction)
    by_type = grouped_error(errors, "FLAT_TYPE")
    substantial = by_type.loc[by_type.rows >= 100]
    highest = substantial.sort_values("mae_sgd", ascending=False).iloc[0]
    summary = {
        "validation": "Train through 2024-09; validate on 2024-10 to 2025-03",
        "validation_rows": len(errors),
        "overall": overall,
        "median_absolute_error_sgd": round(float(errors.ABS_ERROR.median()), 2),
        "p90_absolute_error_sgd": round(float(errors.ABS_ERROR.quantile(0.9)), 2),
        "highest_mae_flat_type_with_at_least_100_rows": {
            "flat_type": str(highest.FLAT_TYPE),
            "rows": int(highest.rows),
            "mae_sgd": round(float(highest.mae_sgd), 2),
        },
        "signed_error_definition": "prediction minus actual rent; positive values indicate overprediction",
    }
    (OUTPUT / "catboost_error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
