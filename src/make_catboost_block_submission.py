"""Build a CatBoost Kaggle candidate without amenity tables.

Run from the project root: python src/make_catboost_block_submission.py
This script creates a local file only; it never uploads to Kaggle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from catboost_experiment import add_address, predict_catboost
from progress_analysis import DATA, join_blocks, prepare_core


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submissions"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw_train = pd.read_csv(DATA / "train.csv")
    raw_test = pd.read_csv(DATA / "test.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    sample = pd.read_csv(DATA / "example-submission.csv")
    train = add_address(join_blocks(prepare_core(raw_train), blocks))
    test = add_address(join_blocks(prepare_core(raw_test), blocks))
    prediction = predict_catboost(train, test, amenities=False)
    if list(sample.columns) != ["Id", "Predicted"] or len(sample) != len(test):
        raise ValueError("Sample submission does not match the test set")
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Predictions must be positive and finite")
    submission = pd.DataFrame({"Id": sample.Id, "Predicted": prediction})
    if submission.Id.isna().any() or submission.Id.duplicated().any():
        raise ValueError("Submission IDs must be unique and nonmissing")
    csv_path = OUTPUT / "catboost_block_candidate.csv"
    submission.to_csv(csv_path, index=False)
    metadata = {
        "status": "candidate_generated_not_uploaded",
        "file": str(csv_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": len(submission),
        "columns": list(submission.columns),
        "model": "CatBoostRegressor with core, HDB block and address features; no amenity tables",
        "training_rows": len(train),
        "local_forward_rmse_sgd": {"6_months": 451.27, "12_months": 475.58, "18_months": 501.21},
        "caution": "The public Kaggle score is unknown until this exact file is submitted. Historical holdouts do not cover the entire 2026 test period.",
    }
    metadata_path = OUTPUT / "catboost_block_candidate.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
