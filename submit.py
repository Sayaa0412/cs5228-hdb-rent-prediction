"""Generate one Kaggle submission from a trained model checkpoint."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import joblib
import pandas as pd

from src.config import save_config
from src.preprocessing import preprocess


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", help="Trained checkpoint path, relative to the repository or absolute")
    args = parser.parse_args()

    config = runpy.run_path(ROOT / args.config_path)
    model_name = config["model_name"]
    variant = config["model_config"]["variant"]

    checkpoint_path = ROOT / (args.checkpoint_path or config["save_path"])
    model = joblib.load(checkpoint_path)
    submit_path = ROOT / config["submit_path"]
    # Record the loaded model's settings, which may differ from today's config.
    save_config({
        **config,
        "model_config": model.model_config,
        "training_config": model.training_config,
        "save_path": str(checkpoint_path),
    }, submit_path)

    X_raw = pd.read_csv(DATA / "test.csv")
    X_test = preprocess(
        X_raw,
        label=False,
        auxiliary_config=config["auxiliary_config"],
    )
    prediction = model.predict(X_test)

    sample = pd.read_csv(DATA / "example-submission.csv")
    submission = pd.DataFrame(
        {
            "Id": sample["Id"],
            "Predicted": prediction,
        }
    )

    submit_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submit_path, index=False)

    print(f"generated: {model_name}/{variant}")
    print(f"rows: {len(submission)}")
    print(f"saved: {submit_path}")


if __name__ == "__main__":
    main()
