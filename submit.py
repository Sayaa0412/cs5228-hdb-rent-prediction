"""Generate one Kaggle submission from a trained model checkpoint."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation import validate_predictions
from src.preprocessing import preprocess


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def build_submission(sample: pd.DataFrame, prediction, *, test_rows: int) -> pd.DataFrame:
    """Validate the sample IDs and predictions before constructing a CSV."""
    if list(sample.columns) != ["Id", "Predicted"] or len(sample) != test_rows:
        raise ValueError("Sample submission does not match the test set")
    if sample["Id"].isna().any() or sample["Id"].duplicated().any():
        raise ValueError("Submission IDs must be unique and nonmissing")
    values = validate_predictions(prediction, test_rows, positive=True)
    return pd.DataFrame({"Id": sample["Id"].to_numpy(), "Predicted": values})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    config = runpy.run_path(ROOT / args.config_path)
    model_name = config["model_name"]
    variant = config["model_config"]["variant"]

    model = joblib.load(ROOT / config["save_path"])

    X_raw = pd.read_csv(DATA / "test.csv")
    X_test = preprocess(
        X_raw,
        label=False,
        auxiliary_config=config["auxiliary_config"],
    )
    prediction = model.predict(X_test)

    sample = pd.read_csv(DATA / "example-submission.csv")
    submission = build_submission(sample, prediction, test_rows=len(X_test))

    submit_path = ROOT / config["submit_path"]
    submit_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submit_path, index=False)

    print(f"generated: {model_name}/{variant}")
    print(f"rows: {len(submission)}")
    print(f"saved: {submit_path}")


if __name__ == "__main__":
    main()
