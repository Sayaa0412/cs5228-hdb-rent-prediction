"""Fit the preliminary block model on all labeled rows and write a Kaggle CSV.

Run from the project root: python src/make_submission.py
This generates a candidate; it does not upload it to Kaggle.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from progress_analysis import DATA, fit_hgb, join_blocks, prepare_core


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submissions"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw_train = pd.read_csv(DATA / "train.csv")
    raw_test = pd.read_csv(DATA / "test.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    sample = pd.read_csv(DATA / "example-submission.csv")
    train = join_blocks(prepare_core(raw_train), blocks)
    test = join_blocks(prepare_core(raw_test), blocks)
    prediction = fit_hgb(train, test, spatial=True)
    if list(sample.columns) != ["Id", "Predicted"] or len(sample) != len(test):
        raise ValueError("Sample submission does not match the test set")
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Predictions must be positive and finite")
    submission = pd.DataFrame({"Id": sample["Id"], "Predicted": prediction})
    if submission["Id"].isna().any() or submission["Id"].duplicated().any():
        raise ValueError("Submission Id values must be unique and nonmissing")
    csv_path = OUTPUT / "block_hgb_initial.csv"
    submission.to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = {
        "status": "candidate_generated",
        "file": str(csv_path.relative_to(ROOT)),
        "sha256": digest,
        "rows": len(submission),
        "columns": list(submission.columns),
        "model": "HistGradientBoostingRegressor with core and HDB block features",
        "training_rows": len(train),
        "local_forward_validation_rmse_sgd": 457.68,
        "caution": "The local validation covers six months; the Kaggle test extends to July 2026.",
    }
    (OUTPUT / "block_hgb_initial_candidate.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
