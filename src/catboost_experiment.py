"""Compare CatBoost with the existing HGB model on a forward holdout.

Run from the project root: python src/catboost_experiment.py
Hyperparameters are fixed in this script; the final holdout is not used for
early stopping or per-run tuning.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from amenity_experiment import AMENITY_COLUMNS, KEYS, make_block_features
from progress_analysis import (
    DATA,
    OUTPUT,
    SEED,
    TRAIN_END,
    VALID_START,
    fit_hgb,
    join_blocks,
    metrics,
    prepare_core,
)
from rolling_validation import WINDOWS


NUMERIC_BASE = [
    "MONTH_INDEX", "FLOOR_AREA_SQM", "LEASE_AGE", "LATITUDE", "LONGITUDE",
    "MAX_FLOOR", "BUILDING_AGE",
]
CATEGORICAL = ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "REGION", "STREET", "ADDRESS"]
PARAMETERS = {
    "loss_function": "RMSE",
    "iterations": 650,
    "depth": 7,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "random_seed": SEED,
    "thread_count": 6,
    "verbose": False,
    "allow_writing_files": False,
}


def add_address(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ADDRESS"] = out[KEYS].astype(str).agg("|".join, axis=1)
    return out


def prepare_features(
    raw: pd.DataFrame, blocks: pd.DataFrame, amenity_features: pd.DataFrame
) -> pd.DataFrame:
    enriched = join_blocks(prepare_core(raw), blocks)
    enriched = enriched.merge(amenity_features, on=KEYS, how="left", validate="many_to_one")
    enriched = add_address(enriched)
    if enriched[AMENITY_COLUMNS].isna().any().any():
        raise ValueError("Missing amenity features")
    return enriched


def predict_catboost(
    fitting: pd.DataFrame, validation: pd.DataFrame, *, amenities: bool
) -> np.ndarray:
    numeric = NUMERIC_BASE + (AMENITY_COLUMNS if amenities else [])
    columns = numeric + CATEGORICAL
    train_x = fitting[columns].copy()
    valid_x = validation[columns].copy()
    for name in CATEGORICAL:
        train_x[name] = train_x[name].fillna("unknown").astype(str)
        valid_x[name] = valid_x[name].fillna("unknown").astype(str)
    model = CatBoostRegressor(**PARAMETERS)
    model.fit(train_x, fitting["MONTHLY_RENT"], cat_features=CATEGORICAL)
    return model.predict(valid_x)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw_train = pd.read_csv(DATA / "train.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    amenity_features = make_block_features(blocks)
    enriched = prepare_features(raw_train, blocks, amenity_features)
    fitting = enriched.loc[enriched.RENT_APPROVAL_DATE <= TRAIN_END].copy()
    validation = enriched.loc[enriched.RENT_APPROVAL_DATE >= VALID_START].copy()
    target = validation.MONTHLY_RENT

    hgb = metrics(target, fit_hgb(fitting, validation, spatial=True))
    cat_base = metrics(target, predict_catboost(fitting, validation, amenities=False))
    cat_amenities = metrics(target, predict_catboost(fitting, validation, amenities=True))
    prior_hgb = json.loads((OUTPUT / "rolling_validation.json").read_text(encoding="utf-8"))
    rolling_checks = []
    for index, (end, start, stop) in enumerate(WINDOWS[:-1]):
        if prior_hgb[index]["validation_start"] != start or prior_hgb[index]["validation_end"] != stop:
            raise ValueError("HGB rolling validation windows do not match")
        historical_fit = enriched.loc[enriched.RENT_APPROVAL_DATE <= end].copy()
        historical_valid = enriched.loc[enriched.RENT_APPROVAL_DATE.between(start, stop)].copy()
        score = metrics(
            historical_valid.MONTHLY_RENT,
            predict_catboost(historical_fit, historical_valid, amenities=True),
        )
        rolling_checks.append({
            "fit_through": end,
            "validation_start": start,
            "validation_end": stop,
            "hgb_block_rmse_sgd": prior_hgb[index]["rmse_sgd"],
            "catboost_block_amenities_rmse_sgd": score["rmse_sgd"],
        })
    rolling_checks.append({
        "fit_through": TRAIN_END,
        "validation_start": VALID_START,
        "validation_end": str(validation.RENT_APPROVAL_DATE.max()),
        "hgb_block_rmse_sgd": hgb["rmse_sgd"],
        "catboost_block_amenities_rmse_sgd": cat_amenities["rmse_sgd"],
    })
    summary = {
        "validation": "Train through 2024-09; validate on 2024-10 to 2025-03",
        "training_rows": len(fitting),
        "validation_rows": len(validation),
        "catboost_parameters": PARAMETERS,
        "categorical_features": CATEGORICAL,
        "hgb_block_baseline": hgb,
        "catboost_block": cat_base,
        "catboost_block_amenities": cat_amenities,
        "rolling_checks": rolling_checks,
        "caveats": [
            "Fixed CatBoost settings are preliminary; this is not a full tuning study.",
            "The amenity tables lack complete historical opening dates.",
            "The holdout is six months, while the Kaggle test continues through July 2026.",
        ],
    }
    path = OUTPUT / "catboost_experiment.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
