import contextlib
import importlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation import HOLDOUT, regression_metrics


class EvaluationTests(unittest.TestCase):
    def test_metrics_match_known_errors(self):
        self.assertEqual(regression_metrics([1000, 2000], [900, 2100]), {
            "mae_sgd": 100.0, "rmse_sgd": 100.0, "r2": 0.96,
        })
        with self.assertRaises(ValueError):
            regression_metrics([1000, 2000], [np.nan, 2100])

    def test_repeated_windows_train_only_once(self):
        entry = importlib.import_module("eval")
        config = runpy.run_path(str(entry.ROOT / "configs/median/global_median.py"))

        def result(raw, **kwargs):
            return {"fit_through": kwargs["train_end"], "rmse_sgd": 835.90}

        with tempfile.TemporaryDirectory() as directory:
            config["eval_path"] = "evaluation.json"
            with (
                patch.object(entry, "ROOT", Path(directory)),
                patch.object(entry.runpy, "run_path", return_value=config),
                patch.object(entry.pd, "read_csv", return_value=pd.DataFrame()),
                patch.object(entry, "evaluate_split", side_effect=result) as evaluate,
                patch("sys.argv", ["eval.py", "--config_path", "unused.py"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                entry.main()
            self.assertEqual(evaluate.call_count, 6)
            output = json.loads((Path(directory) / "evaluation.json").read_text())
            self.assertEqual(len(output["rolling"]), 4)
            self.assertEqual(len(output["horizon"]), 3)
            self.assertEqual(output["holdout"], output["rolling"][-1])
            self.assertNotIn("horizon_months", output["holdout"])

    def test_small_error_analysis_has_no_large_group(self):
        entry = importlib.import_module("analysis")
        raw = pd.DataFrame({
            "RENT_APPROVAL_DATE": ["2024-09", "2024-09", "2024-10", "2024-10"],
            "TOWN": ["test town"] * 4, "BLOCK": ["1"] * 4, "STREET": ["test street"] * 4,
            "FLAT_TYPE": ["3-room"] * 4, "FLAT_MODEL": ["standard"] * 4,
            "LEASE_COMMENCE_DATE": [2000] * 4, "FLOOR_AREA_SQM": [60] * 4,
            "MONTHLY_RENT": [2000, 2200, 2400, 2600],
        })
        config = runpy.run_path(str(entry.ROOT / "configs/median/global_median.py"))
        with tempfile.TemporaryDirectory() as directory:
            config["analysis_path"] = "analysis.json"
            with (
                patch.object(entry, "ROOT", Path(directory)),
                patch.object(entry.runpy, "run_path", return_value=config),
                patch.object(entry.pd, "read_csv", return_value=raw),
                patch("sys.argv", ["analysis.py", "--config_path", "unused.py"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                entry.main()
            output = json.loads((Path(directory) / "analysis.json").read_text())
            self.assertIsNone(output["summary"]["highest_mae_flat_type_with_at_least_100_rows"])
            self.assertEqual(entry.TRAIN_END, HOLDOUT[0])
