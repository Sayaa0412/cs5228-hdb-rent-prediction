"""Evaluate one model variant on the original forward-validation windows."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models import create_model
from src.preprocessing import forward_split, preprocess


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

HOLDOUT = ("2024-09", "2024-10", "2025-03")
ROLLING_WINDOWS = [
    ("2023-03", "2023-04", "2023-09"),
    ("2023-09", "2023-10", "2024-03"),
    ("2024-03", "2024-04", "2024-09"),
    ("2024-09", "2024-10", "2025-03"),
]
HORIZON_WINDOWS = [
    ("2023-09", "2023-10", "2025-03", 18),
    ("2024-03", "2024-04", "2025-03", 12),
    ("2024-09", "2024-10", "2025-03", 6),
]


def evaluate_split(
    raw: pd.DataFrame,
    *,
    model_name: str,
    model_config: dict,
    training_config: dict,
    auxiliary_config: dict | None,
    train_end: str,
    valid_start: str,
    valid_end: str,
) -> dict:
    train_raw, valid_raw = forward_split(
        raw,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
    )
    X_train, y_train = preprocess(
        train_raw,
        label=True,
        auxiliary_config=auxiliary_config,
    )
    X_valid, y_valid = preprocess(
        valid_raw,
        label=True,
        auxiliary_config=auxiliary_config,
    )

    model = create_model(
        model_name,
        model_config=model_config,
        training_config=training_config,
    )
    model.train(X_train, y_train)
    prediction = model.predict(X_valid)

    return {
        "fit_through": train_end,
        "validation_start": valid_start,
        "validation_end": valid_end,
        "fit_rows": len(X_train),
        "validation_rows": len(X_valid),
        "mae_sgd": round(float(mean_absolute_error(y_valid, prediction)), 2),
        "rmse_sgd": round(
            float(np.sqrt(mean_squared_error(y_valid, prediction))),
            2,
        ),
        "r2": round(float(r2_score(y_valid, prediction)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    config = runpy.run_path(ROOT / args.config_path)
    model_name = config["model_name"]
    raw = pd.read_csv(DATA / "train.csv")

    holdout = evaluate_split(
        raw,
        model_name=model_name,
        model_config=config["model_config"],
        training_config=config["training_config"],
        auxiliary_config=config["auxiliary_config"],
        train_end=HOLDOUT[0],
        valid_start=HOLDOUT[1],
        valid_end=HOLDOUT[2],
    )
    print(json.dumps(holdout), flush=True)

    rolling = []
    for train_end, valid_start, valid_end in ROLLING_WINDOWS:
        record = evaluate_split(
            raw,
            model_name=model_name,
            model_config=config["model_config"],
            training_config=config["training_config"],
            auxiliary_config=config["auxiliary_config"],
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        rolling.append(record)
        print(json.dumps(record), flush=True)

    horizon = []
    for train_end, valid_start, valid_end, months in HORIZON_WINDOWS:
        record = evaluate_split(
            raw,
            model_name=model_name,
            model_config=config["model_config"],
            training_config=config["training_config"],
            auxiliary_config=config["auxiliary_config"],
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        record = {"horizon_months": months, **record}
        horizon.append(record)
        print(json.dumps(record), flush=True)

    result = {
        "holdout": holdout,
        "rolling": rolling,
        "horizon": horizon,
    }
    output_path = ROOT / config["eval_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
