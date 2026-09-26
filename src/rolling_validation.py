"""Evaluate the preliminary block model on four consecutive forecast windows.

Run from the project root: python src/rolling_validation.py
"""

from __future__ import annotations

import json

import pandas as pd

from progress_analysis import DATA, OUTPUT, fit_hgb, join_blocks, metrics, prepare_core


WINDOWS = [
    ("2023-03", "2023-04", "2023-09"),
    ("2023-09", "2023-10", "2024-03"),
    ("2024-03", "2024-04", "2024-09"),
    ("2024-09", "2024-10", "2025-03"),
]


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    core = pd.read_csv(DATA / "train.csv")
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
    train = join_blocks(prepare_core(core), blocks)
    results = []
    for end, start, stop in WINDOWS:
        fitting = train.loc[train.RENT_APPROVAL_DATE <= end].copy()
        validation = train.loc[train.RENT_APPROVAL_DATE.between(start, stop)].copy()
        prediction = fit_hgb(fitting, validation, spatial=True)
        results.append({
            "fit_through": end,
            "validation_start": start,
            "validation_end": stop,
            "fit_rows": len(fitting),
            "validation_rows": len(validation),
            **metrics(validation.MONTHLY_RENT, prediction),
        })
    path = OUTPUT / "rolling_validation.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
