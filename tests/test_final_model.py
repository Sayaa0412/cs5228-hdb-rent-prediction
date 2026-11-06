"""Temporal and data contracts; no model training or experiment reruns."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import final_model as model
from tests.helpers import rental_frame, write_auxiliary


class FinalModelTests(unittest.TestCase):
    def test_features_keep_order_duplicates_and_unknown_block_values(self):
        raw = rental_frame().iloc[[5, 2, 2, 0]].copy()
        with tempfile.TemporaryDirectory() as directory:
            write_auxiliary(Path(directory))
            blocks = pd.read_csv(Path(directory) / "sg-hdb-block.csv")
        blocks["MAX_FLOOR"] = -1
        blocks["YEAR_COMPLETED"] = -1
        x = model.features(raw, blocks)
        self.assertEqual(len(x), len(raw))
        self.assertEqual(x.MONTH_INDEX.tolist(), [44, 45, 45, 0])
        self.assertTrue(x.MAX_FLOOR.isna().all())
        self.assertTrue(x.BUILDING_AGE.isna().all())
        self.assertNotIn("MONTHLY_RENT", x)
        changed = raw.assign(MONTHLY_RENT=99999)
        pd.testing.assert_frame_equal(x, model.features(changed, blocks))
        static = model.static_features(raw, x)
        self.assertEqual(static.LEASE_START_YEAR.tolist(), [2000] * 4)

    def test_history_distinguishes_match_absence_and_unavailable(self):
        source = pd.DataFrame(dict(ADDRESS=["a", "a", "b", "a"],
                                    FLAT_TYPE=["4-room"] * 4,
                                    MONTH_INDEX=[0, 0, 0, 12]))
        y = np.array([1000., 2000., 3000., 4000.])
        history = model.reference_history(source, y)
        query = pd.DataFrame(dict(ADDRESS=["a", "c", "a"], FLAT_TYPE=["4-room"] * 3,
                                   MONTH_INDEX=[12, 12, 25]))
        x = model.reference_features(history, query, 0, 12, (12,))
        self.assertAlmostEqual(x.PAST0_LOG_COUNT[0], np.log(3))
        self.assertEqual(x.PAST0_LOG_COUNT[1], 0)
        self.assertTrue(np.isnan(x.PAST0_LOG_RENT_ANOMALY[1]))
        self.assertEqual(x.PAST0_UNAVAILABLE.tolist(), [0, 0, 1])
        self.assertTrue(np.isnan(x.PAST0_LOG_COUNT[2]))
        # Changing a query month's own rent cannot change its earlier descriptor.
        changed = y.copy()
        changed[-1] = 99999
        updated = model.reference_features(model.reference_history(source, changed), query, 0, 12, (12,))
        pd.testing.assert_frame_equal(x, updated)

    def test_normalized_history_uses_exact_source_month(self):
        source = pd.DataFrame(dict(ADDRESS=["a", "a", "a"],
                                    FLAT_TYPE=["3-room"] * 3, MONTH_INDEX=[0, 0, 1]))
        table = model.normalized_history(source, [1000, 2000, 8000])
        query = source.iloc[:1].assign(MONTH_INDEX=24)
        extra = model.normalized_features(table, query, 0, 1)
        self.assertEqual(extra.HISTORY24_NORMALIZED_MEAN.iloc[0], 1500)
        self.assertAlmostEqual(extra.HISTORY24_LOG_COUNT.iloc[0], np.log(3))

    def test_crossfit_excludes_held_out_addresses_without_fitting(self):
        x = pd.DataFrame({"ADDRESS": ["a", "a", "b", "b", "c", "c"]})
        fitted_addresses = []

        def fake_index(frame, months, log_y):
            fitted_addresses.append(set(frame.ADDRESS))
            return pd.Series([0., .1], index=[0, 1])

        with patch.object(model, "index_fit", side_effect=fake_index):
            offsets, _, fold_ids = model.crossfit_index(x, [0, 1] * 3, np.ones(6) * 2000)
        for fold in range(3):
            self.assertFalse(fitted_addresses[fold] & set(x.loc[fold_ids == fold, "ADDRESS"]))
        np.testing.assert_allclose(offsets, [0., .1] * 3)

    def test_forecast_handles_same_month_and_rejects_unsupported_horizons(self):
        config = dict(level=.2, slope=.1, phi=.9, last_observed=.3)
        np.testing.assert_allclose(model.forecast_scale(config, [0, 1, 2]),
                                   np.exp([.3, .29, .371]))
        for horizon in (-1, 25):
            with self.assertRaises(ValueError):
                model.forecast_scale(config, [horizon])

    def test_evaluation_fits_only_prefix_and_pools_rows(self):
        import final

        raw = pd.DataFrame({"RENT_APPROVAL_DATE": pd.period_range("2021-01", "2025-03", freq="M").astype(str),
                            "MONTHLY_RENT": np.arange(51) + 2000.})
        fitted = {}

        def fake_fit(prefix, blocks, destination, device):
            fitted[destination] = prefix.copy()
            destination.mkdir()

        def fake_predict(query, blocks, destination):
            self.assertNotIn("MONTHLY_RENT", query)
            self.assertTrue(query.RENT_APPROVAL_DATE.min() > fitted[destination].RENT_APPROVAL_DATE.max())
            origin = pd.Period(fitted[destination].RENT_APPROVAL_DATE.max(), freq="M")
            horizon = pd.PeriodIndex(query.RENT_APPROVAL_DATE, freq="M").asi8 - origin.ordinal
            return pd.DataFrame({"forecast_month": horizon, "MONTHLY_RENT": 2000.})

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "evaluation"
            with patch.object(final, "fit_final", side_effect=fake_fit), patch.object(final, "predict_final", side_effect=fake_predict):
                final.evaluate(raw, pd.DataFrame(), destination, list(model.FOLDS), "GPU")
            metrics = pd.read_csv(destination / "metrics.csv")
        self.assertEqual(len(fitted), 4)
        self.assertEqual(metrics.iloc[-1]["n"], 24)
        self.assertAlmostEqual(metrics.iloc[-1].rmse ** 2, metrics.iloc[:-1].sse.sum() / 24)


if __name__ == "__main__":
    unittest.main()
