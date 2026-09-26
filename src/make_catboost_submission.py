"""Build, but do not upload, the CatBoost plus amenities Kaggle candidate.

Run from the project root: python src/make_catboost_submission.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from amenity_experiment import make_block_features
from catboost_experiment import prepare_features, predict_catboost
from progress_analysis import DATA


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submissions"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw_train = pd.read_csv(DATA / "train.csv")
    raw_test = pd.read_csv(DATA / "test.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    sample = pd.read_csv(DATA / "example-submission.csv")
    amenity_features = make_block_features(blocks)
    train = prepare_features(raw_train, blocks, amenity_features)
    test = prepare_features(raw_test, blocks, amenity_features)
    prediction = predict_catboost(train, test, amenities=True)
    if list(sample.columns) != ["Id", "Predicted"] or len(sample) != len(test):
        raise ValueError("Sample submission does not match the test set")
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Predictions must be positive and finite")
    submission = pd.DataFrame({"Id": sample.Id, "Predicted": prediction})
    if submission.Id.isna().any() or submission.Id.duplicated().any():
        raise ValueError("Submission IDs must be unique and nonmissing")
    csv_path = OUTPUT / "catboost_amenities_candidate.csv"
    submission.to_csv(csv_path, index=False)
    metadata = {
        "status": "candidate_generated_not_uploaded",
        "file": str(csv_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": len(submission),
        "columns": list(submission.columns),
        "model": "CatBoostRegressor with core, block, address and static amenity features",
        "training_rows": len(train),
        "local_forward_validation_rmse_sgd": 450.46,
        "caution": "The public Kaggle score is unknown until this exact file is submitted. Local validation uses a six-month historical holdout and amenity opening dates are incomplete.",
    }
    metadata_path = OUTPUT / "catboost_amenities_candidate.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
