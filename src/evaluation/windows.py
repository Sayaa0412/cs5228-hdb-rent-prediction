"""The original project's fixed forward-validation periods."""

HOLDOUT = ("2024-09", "2024-10", "2025-03")
ROLLING_WINDOWS = [
    ("2023-03", "2023-04", "2023-09"),
    ("2023-09", "2023-10", "2024-03"),
    ("2024-03", "2024-04", "2024-09"),
    HOLDOUT,
]
HORIZON_WINDOWS = [
    ("2023-09", "2023-10", "2025-03", 18),
    ("2024-03", "2024-04", "2025-03", 12),
    (*HOLDOUT, 6),
]
