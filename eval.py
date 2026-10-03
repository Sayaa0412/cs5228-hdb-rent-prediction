"""Evaluate one model variant on the original forward-validation windows."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import pandas as pd

from src.evaluation import HOLDOUT, HORIZON_WINDOWS, ROLLING_WINDOWS, regression_metrics
from src.models import create_model
from src.preprocessing import forward_split, preprocess


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


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
        **regression_metrics(y_valid, prediction),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    config = runpy.run_path(ROOT / args.config_path)
    model_name = config["model_name"]
    raw = pd.read_csv(DATA / "train.csv")

    results = {}

    def evaluate_window(window: tuple[str, str, str]) -> dict:
        if window not in results:
            results[window] = evaluate_split(
                raw,
                model_name=model_name,
                model_config=config["model_config"],
                training_config=config["training_config"],
                auxiliary_config=config["auxiliary_config"],
                train_end=window[0], valid_start=window[1], valid_end=window[2],
            )
        return dict(results[window])

    holdout = evaluate_window(HOLDOUT)
    print(json.dumps(holdout), flush=True)

    rolling = []
    for train_end, valid_start, valid_end in ROLLING_WINDOWS:
        record = evaluate_window((train_end, valid_start, valid_end))
        rolling.append(record)
        print(json.dumps(record), flush=True)

    horizon = []
    for train_end, valid_start, valid_end, months in HORIZON_WINDOWS:
        record = evaluate_window((train_end, valid_start, valid_end))
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
