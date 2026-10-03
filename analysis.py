"""Analyze holdout errors for one model variant."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import save_config
from src.models import create_model
from src.preprocessing import forward_split, preprocess


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN_END = "2024-09"
VALID_START = "2024-10"
VALID_END = "2025-03"


def grouped_error(frame: pd.DataFrame, key: str) -> list[dict]:
    grouped = (
        frame.groupby(key, observed=True)
        .agg(
            rows=("ABS_ERROR", "size"),
            mae_sgd=("ABS_ERROR", "mean"),
            mean_signed_error_sgd=("SIGNED_ERROR", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    grouped["mae_sgd"] = grouped["mae_sgd"].round(2)
    grouped["mean_signed_error_sgd"] = grouped[
        "mean_signed_error_sgd"
    ].round(2)
    return grouped.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    config = runpy.run_path(ROOT / args.config_path)
    model_name = config["model_name"]
    output_path = ROOT / config["analysis_path"]
    save_config(config, output_path)

    raw = pd.read_csv(DATA / "train.csv")
    train_raw, valid_raw = forward_split(
        raw,
        train_end=TRAIN_END,
        valid_start=VALID_START,
        valid_end=VALID_END,
    )
    X_train, y_train = preprocess(
        train_raw,
        label=True,
        auxiliary_config=config["auxiliary_config"],
    )
    X_valid, y_valid = preprocess(
        valid_raw,
        label=True,
        auxiliary_config=config["auxiliary_config"],
    )

    model = create_model(
        model_name,
        model_config=config["model_config"],
        training_config=config["training_config"],
    )
    model.train(X_train, y_train)
    prediction = model.predict(X_valid)

    errors = X_valid[["RENT_APPROVAL_DATE", "TOWN", "FLAT_TYPE"]].copy()
    errors["ACTUAL_RENT"] = y_valid.to_numpy()
    errors["PREDICTED_RENT"] = prediction
    errors["SIGNED_ERROR"] = prediction - errors["ACTUAL_RENT"]
    errors["ABS_ERROR"] = errors["SIGNED_ERROR"].abs()
    errors["RENT_BAND"] = pd.cut(
        errors["ACTUAL_RENT"],
        bins=[-np.inf, 2000, 3000, 4000, np.inf],
        labels=[
            "up to 2000",
            "2001 to 3000",
            "3001 to 4000",
            "above 4000",
        ],
    )

    by_flat_type = grouped_error(errors, "FLAT_TYPE")
    common_flat_types = [record for record in by_flat_type if record["rows"] >= 100]
    highest_group = max(common_flat_types, key=lambda record: record["mae_sgd"])
    highest = {
        "flat_type": highest_group["FLAT_TYPE"],
        "rows": highest_group["rows"],
        "mae_sgd": highest_group["mae_sgd"],
    }

    result = {
        "summary": {
            "fit_through": TRAIN_END,
            "validation_start": VALID_START,
            "validation_end": VALID_END,
            "fit_rows": len(X_train),
            "validation_rows": len(X_valid),
            "mae_sgd": round(float(mean_absolute_error(y_valid, prediction)), 2),
            "rmse_sgd": round(
                float(np.sqrt(mean_squared_error(y_valid, prediction))),
                2,
            ),
            "r2": round(float(r2_score(y_valid, prediction)), 4),
            "median_absolute_error_sgd": round(
                float(errors["ABS_ERROR"].median()),
                2,
            ),
            "p90_absolute_error_sgd": round(
                float(errors["ABS_ERROR"].quantile(0.9)),
                2,
            ),
            "highest_mae_flat_type_with_at_least_100_rows": highest,
            "signed_error_definition": (
                "prediction minus actual rent; positive values indicate "
                "overprediction"
            ),
        },
        "error_by_flat_type": by_flat_type,
        "error_by_month": grouped_error(errors, "RENT_APPROVAL_DATE"),
        "error_by_town": grouped_error(errors, "TOWN"),
        "error_by_rent_band": grouped_error(errors, "RENT_BAND"),
    }

    loss_path = (output_path.parent / "loss" / "holdout" if model_name == "mlp"
                 else output_path.with_name(f"{output_path.stem}_loss"))
    loss_files = model.save_loss(loss_path)
    if loss_files:
        result["loss_files"] = loss_files
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
