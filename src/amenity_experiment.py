"""Test static amenity-distance features on the same forward holdout.

Run from the project root: python src/amenity_experiment.py
The source tables lack historical opening dates, so these features are
exploratory and should not be treated as a date-safe causal explanation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from progress_analysis import (
    DATA,
    OUTPUT,
    SEED,
    TRAIN_END,
    VALID_START,
    fit_hgb,
    join_blocks,
    metrics,
    normalize_text,
    prepare_core,
)
from rolling_validation import WINDOWS


KEYS = ["TOWN", "BLOCK", "STREET"]
AMENITY_COLUMNS = [
    "NEAREST_MRT_KM",
    "MRT_WITHIN_1KM",
    "NEAREST_PRIMARY_SCHOOL_KM",
    "PRIMARY_SCHOOLS_WITHIN_1KM",
    "NEAREST_MALL_KM",
    "MALLS_WITHIN_1KM",
]


def distance_features(blocks: pd.DataFrame, places: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distances in kilometres, calculated once per HDB block."""
    lat1 = np.radians(blocks["LATITUDE"].to_numpy(dtype=float))[:, None]
    lon1 = np.radians(blocks["LONGITUDE"].to_numpy(dtype=float))[:, None]
    lat2 = np.radians(places["LATITUDE"].to_numpy(dtype=float))[None, :]
    lon2 = np.radians(places["LONGITUDE"].to_numpy(dtype=float))[None, :]
    if lat2.shape[1] == 0:
        raise ValueError("No amenity coordinates found")
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    km = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return km.min(axis=1), (km <= 1.0).sum(axis=1)


def make_block_features(blocks: pd.DataFrame) -> pd.DataFrame:
    out = blocks[KEYS + ["LATITUDE", "LONGITUDE"]].copy()
    for key in KEYS:
        out[key] = normalize_text(out[key])
    if out.duplicated(KEYS).any():
        raise ValueError("HDB block lookup is not unique")

    mrt = pd.read_csv(DATA / "auxiliary" / "sg-mrt-stations.csv")
    mrt = mrt.loc[normalize_text(mrt["STATUS"]) == "open"]
    schools = pd.read_csv(DATA / "auxiliary" / "sg-schools.csv")
    school_level = normalize_text(schools["MAINLEVEL_CODE"])
    schools = schools.loc[school_level.eq("primary") | school_level.str.startswith("mixed level (p1", na=False)]
    malls = pd.read_csv(DATA / "auxiliary" / "sg-shopping-malls.csv")

    for places, distance_name, count_name in (
        (mrt, "NEAREST_MRT_KM", "MRT_WITHIN_1KM"),
        (schools, "NEAREST_PRIMARY_SCHOOL_KM", "PRIMARY_SCHOOLS_WITHIN_1KM"),
        (malls, "NEAREST_MALL_KM", "MALLS_WITHIN_1KM"),
    ):
        nearest, count = distance_features(out, places)
        out[distance_name] = nearest
        out[count_name] = count
    return out.drop(columns=["LATITUDE", "LONGITUDE"])


def predict_with_amenities(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    numeric = [
        "MONTH_INDEX", "FLOOR_AREA_SQM", "LEASE_AGE", "LATITUDE", "LONGITUDE",
        "MAX_FLOOR", "BUILDING_AGE", *AMENITY_COLUMNS,
    ]
    categorical = ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "REGION"]
    columns = numeric + categorical
    x_train = train[columns].copy()
    x_valid = valid[columns].copy()
    for name in categorical:
        x_train[name] = x_train[name].fillna("unknown").astype("category")
        x_valid[name] = pd.Categorical(
            x_valid[name].fillna("unknown"), categories=x_train[name].cat.categories
        )
    model = HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        learning_rate=0.07,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=SEED,
    )
    model.fit(x_train, train["MONTHLY_RENT"])
    return model.predict(x_valid)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw_train = pd.read_csv(DATA / "train.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    enriched = join_blocks(prepare_core(raw_train), blocks)
    block_features = make_block_features(blocks)
    enriched = enriched.merge(block_features, on=KEYS, how="left", validate="many_to_one")
    if enriched[AMENITY_COLUMNS].isna().any().any():
        raise ValueError("At least one rental record has missing amenity features")

    fitting = enriched.loc[enriched.RENT_APPROVAL_DATE <= TRAIN_END].copy()
    validation = enriched.loc[enriched.RENT_APPROVAL_DATE >= VALID_START].copy()
    baseline = fit_hgb(fitting, validation, spatial=True)
    expanded = predict_with_amenities(fitting, validation)
    baseline_scores = metrics(validation.MONTHLY_RENT, baseline)
    expanded_scores = metrics(validation.MONTHLY_RENT, expanded)
    rolling_checks = []
    for end, start, stop in WINDOWS[:-1]:
        historical_fit = enriched.loc[enriched.RENT_APPROVAL_DATE <= end].copy()
        historical_valid = enriched.loc[enriched.RENT_APPROVAL_DATE.between(start, stop)].copy()
        historical_baseline = metrics(
            historical_valid.MONTHLY_RENT,
            fit_hgb(historical_fit, historical_valid, spatial=True),
        )
        historical_expanded = metrics(
            historical_valid.MONTHLY_RENT,
            predict_with_amenities(historical_fit, historical_valid),
        )
        rolling_checks.append({
            "fit_through": end,
            "validation_start": start,
            "validation_end": stop,
            "baseline_rmse_sgd": historical_baseline["rmse_sgd"],
            "with_amenities_rmse_sgd": historical_expanded["rmse_sgd"],
            "rmse_change_sgd": round(
                historical_expanded["rmse_sgd"] - historical_baseline["rmse_sgd"], 2
            ),
        })
    rolling_checks.append({
        "fit_through": TRAIN_END,
        "validation_start": VALID_START,
        "validation_end": str(validation.RENT_APPROVAL_DATE.max()),
        "baseline_rmse_sgd": baseline_scores["rmse_sgd"],
        "with_amenities_rmse_sgd": expanded_scores["rmse_sgd"],
        "rmse_change_sgd": round(expanded_scores["rmse_sgd"] - baseline_scores["rmse_sgd"], 2),
    })
    summary = {
        "method": "Train through 2024-09; validate on 2024-10 to 2025-03",
        "feature_source": "Course-provided MRT, schools and shopping mall coordinate tables",
        "amenity_features": AMENITY_COLUMNS,
        "baseline_block_hgb": baseline_scores,
        "block_hgb_with_amenities": expanded_scores,
        "rmse_change_sgd": round(expanded_scores["rmse_sgd"] - baseline_scores["rmse_sgd"], 2),
        "rolling_checks": rolling_checks,
        "limitation": "Amenity tables do not contain complete opening dates. Stations marked open at data collection may not have been open for every historical rental month.",
    }
    path = OUTPUT / "amenity_experiment.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
