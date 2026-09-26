"""Reproducible exploratory analysis and forward-validation baselines.

Run from the project root: python src/progress_analysis.py
Outputs are written to analysis/ and can be regenerated from data/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "analysis"
TRAIN_END = "2024-09"
VALID_START = "2024-10"
SEED = 42


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def clean_flat_type(series: pd.Series) -> pd.Series:
    return normalize_text(series).str.replace(r"^(\d)\s*[- ]\s*room$", r"\1-room", regex=True)


def prepare_core(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for name in ("TOWN", "BLOCK", "STREET", "FLAT_MODEL"):
        out[name] = normalize_text(out[name])
    out["FLAT_TYPE"] = clean_flat_type(out["FLAT_TYPE"])
    dates = pd.to_datetime(out["RENT_APPROVAL_DATE"], format="%Y-%m", errors="raise")
    out["APPROVAL_YEAR"] = dates.dt.year
    out["MONTH_INDEX"] = (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    out["LEASE_AGE"] = dates.dt.year - out["LEASE_COMMENCE_DATE"]
    return out


def join_blocks(core: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    lookup = blocks.copy()
    for name in ("TOWN", "BLOCK", "STREET"):
        lookup[name] = normalize_text(lookup[name])
    keys = ["TOWN", "BLOCK", "STREET"]
    if lookup.duplicated(keys).any():
        raise ValueError("HDB block lookup is not unique on town, block, street")
    use = keys + ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED", "REGION"]
    out = core.merge(lookup[use], on=keys, how="left", validate="many_to_one")
    out["BUILDING_AGE"] = out["APPROVAL_YEAR"] - out["YEAR_COMPLETED"]
    return out


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_sgd": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse_sgd": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def grouped_median_prediction(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    keys = ["TOWN", "FLAT_TYPE"]
    pair = train.groupby(keys, observed=True)["MONTHLY_RENT"].median()
    town = train.groupby("TOWN", observed=True)["MONTHLY_RENT"].median()
    global_median = float(train["MONTHLY_RENT"].median())
    idx = pd.MultiIndex.from_frame(valid[keys])
    result = pd.Series(pair.reindex(idx).to_numpy(), index=valid.index)
    result = result.fillna(valid["TOWN"].map(town)).fillna(global_median)
    return result.to_numpy(dtype=float)


def fit_hgb(train: pd.DataFrame, valid: pd.DataFrame, *, spatial: bool) -> np.ndarray:
    numeric = ["MONTH_INDEX", "FLOOR_AREA_SQM", "LEASE_AGE"]
    categorical = ["TOWN", "FLAT_TYPE", "FLAT_MODEL"]
    if spatial:
        numeric += ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "BUILDING_AGE"]
        categorical += ["REGION"]
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
    raw_test = pd.read_csv(DATA / "test.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    train = join_blocks(prepare_core(raw_train), blocks)
    test = join_blocks(prepare_core(raw_test), blocks)

    build = train.loc[train["RENT_APPROVAL_DATE"] <= TRAIN_END].copy()
    valid = train.loc[train["RENT_APPROVAL_DATE"] >= VALID_START].copy()
    if build.empty or valid.empty or build["RENT_APPROVAL_DATE"].max() >= valid["RENT_APPROVAL_DATE"].min():
        raise ValueError("Invalid forward validation split")

    predictions = {
        "global_median": np.full(len(valid), float(build["MONTHLY_RENT"].median())),
        "town_flat_type_median": grouped_median_prediction(build, valid),
        "core_hgb": fit_hgb(build, valid, spatial=False),
        "core_plus_block_hgb": fit_hgb(build, valid, spatial=True),
    }
    scores = {name: metrics(valid["MONTHLY_RENT"], pred) for name, pred in predictions.items()}
    best_name = min(scores, key=lambda name: scores[name]["rmse_sgd"])
    best_pred = predictions[best_name]
    error_frame = valid[["RENT_APPROVAL_DATE", "TOWN", "FLAT_TYPE", "MONTHLY_RENT"]].copy()
    error_frame["PREDICTED_RENT"] = best_pred
    error_frame["ABS_ERROR"] = np.abs(error_frame["MONTHLY_RENT"] - best_pred)
    by_type = (
        error_frame.groupby("FLAT_TYPE", observed=True)
        .agg(rows=("ABS_ERROR", "size"), mae_sgd=("ABS_ERROR", "mean"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    by_month = (
        error_frame.groupby("RENT_APPROVAL_DATE", observed=True)
        .agg(rows=("ABS_ERROR", "size"), mae_sgd=("ABS_ERROR", "mean"))
        .reset_index()
    )
    month_stats = (
        train.groupby("RENT_APPROVAL_DATE", observed=True)["MONTHLY_RENT"]
        .agg(["size", "median", "mean"])
        .reset_index()
    )
    quarter_labels = pd.to_datetime(train["RENT_APPROVAL_DATE"], format="%Y-%m").dt.to_period("Q")
    quarter_stats = (
        train.assign(QUARTER=quarter_labels.astype(str))
        .groupby("QUARTER", observed=True)["MONTHLY_RENT"]
        .agg(["size", "median"])
        .reset_index()
    )
    town_stats = (
        train.groupby("TOWN", observed=True)["MONTHLY_RENT"]
        .agg(["size", "median"])
        .reset_index()
        .sort_values("median")
    )
    flat_stats = (
        train.groupby("FLAT_TYPE", observed=True)["MONTHLY_RENT"]
        .agg(["size", "median"])
        .reset_index()
        .sort_values("median")
    )
    by_type.to_csv(OUTPUT / "validation_error_by_flat_type.csv", index=False)
    by_month.to_csv(OUTPUT / "validation_error_by_month.csv", index=False)
    month_stats.to_csv(OUTPUT / "monthly_rent_stats.csv", index=False)
    quarter_stats.to_csv(OUTPUT / "quarterly_rent_stats.csv", index=False)
    town_stats.to_csv(OUTPUT / "town_rent_stats.csv", index=False)
    flat_stats.to_csv(OUTPUT / "flat_type_rent_stats.csv", index=False)

    summary = {
        "method": "Time ordered holdout: train through 2024-09, validate on 2024-10 to 2025-03",
        "random_seed": SEED,
        "train_rows": int(len(raw_train)),
        "test_rows": int(len(raw_test)),
        "build_rows": int(len(build)),
        "validation_rows": int(len(valid)),
        "train_date_range": [str(raw_train.RENT_APPROVAL_DATE.min()), str(raw_train.RENT_APPROVAL_DATE.max())],
        "test_date_range": [str(raw_test.RENT_APPROVAL_DATE.min()), str(raw_test.RENT_APPROVAL_DATE.max())],
        "train_missing_cells": int(raw_train.isna().sum().sum()),
        "test_missing_cells": int(raw_test.isna().sum().sum()),
        "constant_train_columns": [c for c in raw_train if raw_train[c].nunique(dropna=False) == 1],
        "exact_duplicate_train_rows": int(raw_train.duplicated().sum()),
        "exact_duplicate_test_rows": int(raw_test.duplicated().sum()),
        "block_match_rate_train": round(float(train["LATITUDE"].notna().mean()), 4),
        "block_match_rate_test": round(float(test["LATITUDE"].notna().mean()), 4),
        "post_completion_approvals_train": int((train["BUILDING_AGE"] < 0).sum()),
        "target": {
            "mean_sgd": round(float(raw_train.MONTHLY_RENT.mean()), 2),
            "median_sgd": round(float(raw_train.MONTHLY_RENT.median()), 2),
            "p01_sgd": round(float(raw_train.MONTHLY_RENT.quantile(.01)), 2),
            "p99_sgd": round(float(raw_train.MONTHLY_RENT.quantile(.99)), 2),
        },
        "validation_scores": scores,
        "best_validation_model_by_rmse": best_name,
        "notes": [
            "Validation simulates forward prediction but spans six months; the Kaggle test extends beyond this horizon.",
            "Identical feature rows need not be erroneous: separate rentals may share recorded attributes.",
            "The competition evaluates submissions with RMSE; see its Evaluation section.",
        ],
    }
    (OUTPUT / "progress_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
