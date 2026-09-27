"""Train one registered model on all labeled rental records."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import pandas as pd

from src.models import create_model
from src.preprocessing import preprocess


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    config = runpy.run_path(ROOT / args.config_path)
    model_name = config["model_name"]
    variant = config["model_config"]["variant"]

    X_raw = pd.read_csv(DATA / "train.csv")
    X_train, y_train = preprocess(
        X_raw,
        label=True,
        auxiliary_config=config["auxiliary_config"],
    )

    model = create_model(
        model_name,
        model_config=config["model_config"],
        training_config=config["training_config"],
    )
    model.train(X_train, y_train)

    save_path = ROOT / config["save_path"]
    model.save(save_path)
    print(f"trained: {model_name}/{variant}")
    print(f"rows: {len(X_train)}, features: {X_train.shape[1]}")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
