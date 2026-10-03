"""Synthetic inputs for preprocessing and model lifecycle tests."""

from pathlib import Path

import pandas as pd


def rental_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "RENT_APPROVAL_DATE": month,
            "TOWN": "test town", "BLOCK": "1", "STREET": "test street",
            "FLAT_TYPE": "4 room" if row % 2 else "3-room",
            "FLAT_MODEL": "standard", "LEASE_COMMENCE_DATE": 2000,
            "FLOOR_AREA_SQM": 60 + row % 4 * 10,
            "MONTHLY_RENT": 1800 + row * 50,
        }
        for row, month in enumerate(["2021-01", "2024-09", "2024-10", "2025-03"] * 4)
    ], index=range(50, 66))


def write_auxiliary(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "TOWN": "test town", "BLOCK": "1", "STREET": "test street",
        "LATITUDE": 1.30, "LONGITUDE": 103.80, "MAX_FLOOR": 12,
        "YEAR_COMPLETED": 1999, "REGION": "test region",
    }]).to_csv(folder / "sg-hdb-block.csv", index=False)
    pd.DataFrame([{"STATUS": "open", "LATITUDE": 1.301, "LONGITUDE": 103.801}]).to_csv(
        folder / "sg-mrt-stations.csv", index=False,
    )
    pd.DataFrame([{"MAINLEVEL_CODE": "primary", "LATITUDE": 1.302, "LONGITUDE": 103.802}]).to_csv(
        folder / "sg-schools.csv", index=False,
    )
    pd.DataFrame([{"LATITUDE": 1.303, "LONGITUDE": 103.803}]).to_csv(
        folder / "sg-shopping-malls.csv", index=False,
    )
    pd.DataFrame([
        {"YEAR": 2024, "MONTH": "Aug", "QUOTA_PREMIUM": "$80,000"},
        {"YEAR": 2024, "MONTH": "Sep", "QUOTA_PREMIUM": "$90,000"},
        {"YEAR": 2025, "MONTH": "Feb", "QUOTA_PREMIUM": "$95,000"},
    ]).to_csv(folder / "sg-coe-prices.csv", index=False)
