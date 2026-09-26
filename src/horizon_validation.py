"""Compare fixed models over six-, twelve-, and eighteen-month forward holds.

All validation windows end in March 2025. The fitting data stop immediately
before each window; there is no target leakage across the cutoffs.
Run from the project root: python src/horizon_validation.py
"""

from __future__ import annotations

import json

import pandas as pd

from amenity_experiment import make_block_features
from catboost_experiment import predict_catboost, prepare_features
from progress_analysis import DATA, OUTPUT, fit_hgb, metrics


WINDOWS = [
    ("2023-09", "2023-10", "2025-03", 18),
    ("2024-03", "2024-04", "2025-03", 12),
    ("2024-09", "2024-10", "2025-03", 6),
]


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA / "train.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    enriched = prepare_features(raw, blocks, make_block_features(blocks))
    records = []
    for fit_end, valid_start, valid_end, months in WINDOWS:
        fit = enriched.loc[enriched.RENT_APPROVAL_DATE <= fit_end].copy()
        valid = enriched.loc[enriched.RENT_APPROVAL_DATE.between(valid_start, valid_end)].copy()
        if fit.empty or valid.empty or fit.RENT_APPROVAL_DATE.max() >= valid.RENT_APPROVAL_DATE.min():
            raise ValueError(f"Invalid forward split ending {fit_end}")
        y = valid.MONTHLY_RENT
        scores = {
            "hgb_block": metrics(y, fit_hgb(fit, valid, spatial=True)),
            "catboost_block": metrics(y, predict_catboost(fit, valid, amenities=False)),
            "catboost_block_amenities": metrics(y, predict_catboost(fit, valid, amenities=True)),
        }
        record = {
            "horizon_months": months,
            "fit_through": fit_end,
            "validation_start": valid_start,
            "validation_end": valid_end,
            "fit_rows": len(fit),
            "validation_rows": len(valid),
            "scores": scores,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "description": "Fixed model settings with validation windows ending March 2025",
        "windows": records,
        "caveat": "The amenity tables lack complete historical opening dates; historical access may be overstated.",
    }
    (OUTPUT / "horizon_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
