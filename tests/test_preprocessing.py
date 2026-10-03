import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

import src.preprocessing.preprocess_pipeline as pipeline
from src.preprocessing import forward_split, preprocess
from tests.helpers import rental_frame, write_auxiliary


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.paths = patch.object(pipeline, "AUXILIARY_DATA", self.folder)
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.addCleanup(self.temporary.cleanup)
        self.raw = rental_frame()

    def test_core_requires_no_auxiliary_files_and_preserves_labels(self):
        original = self.raw.copy(deep=True)
        X, y = preprocess(self.raw, label=True, auxiliary_config={})
        assert_frame_equal(self.raw, original)
        self.assertEqual(X.index.tolist(), self.raw.index.tolist())
        self.assertEqual(y.tolist(), self.raw.MONTHLY_RENT.tolist())
        self.assertNotIn("MONTHLY_RENT", X)
        self.assertNotIn("LATITUDE", X)
        self.assertEqual(X.FLAT_TYPE.unique().tolist(), ["3-room", "4-room"])

    def test_blocks_do_not_require_amenity_or_coe_tables(self):
        write_auxiliary(self.folder)
        for name in ("sg-mrt-stations.csv", "sg-schools.csv", "sg-shopping-malls.csv", "sg-coe-prices.csv"):
            (self.folder / name).unlink()
        X, y = preprocess(self.raw, label=True, auxiliary_config={"blocks": True})
        self.assertTrue(X.LATITUDE.notna().all())
        self.assertNotIn("NEAREST_MRT_KM", X)
        self.assertEqual(y.tolist(), self.raw.MONTHLY_RENT.tolist())

    def test_coe_uses_previous_month_and_keeps_earliest_month_missing(self):
        write_auxiliary(self.folder)
        (self.folder / "sg-mrt-stations.csv").unlink()
        X, _ = preprocess(self.raw, label=True, auxiliary_config={"blocks": True, "coe": True})
        feature = X.PREV_MONTH_COE_MEDIAN_SGD
        self.assertTrue(feature[X.RENT_APPROVAL_DATE.eq("2021-01")].isna().all())
        self.assertTrue(feature[X.RENT_APPROVAL_DATE.eq("2024-10")].eq(90000).all())
        self.assertNotIn("NEAREST_MRT_KM", X)

    def test_static_amenities_are_cached_and_refresh_after_source_change(self):
        write_auxiliary(self.folder)
        flags = {"blocks": True, "amenities": True}
        with patch.object(pipeline, "make_amenity_features", wraps=pipeline.make_amenity_features) as compute:
            first, _ = preprocess(self.raw, label=True, auxiliary_config=flags)
            preprocess(self.raw, label=True, auxiliary_config=flags)
            self.assertEqual(compute.call_count, 1)
            pd.DataFrame([{"LATITUDE": 1.40001, "LONGITUDE": 103.90}]).to_csv(
                self.folder / "sg-shopping-malls.csv", index=False,
            )
            changed, _ = preprocess(self.raw, label=True, auxiliary_config=flags)
            self.assertEqual(compute.call_count, 2)
            self.assertGreater(changed.NEAREST_MALL_KM.iloc[0], first.NEAREST_MALL_KM.iloc[0])

    def test_invalid_switches_are_rejected(self):
        for flags in ({"blocks": "yes"}, {"unknown": True}, {"amenities": True}):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                preprocess(self.raw, label=True, auxiliary_config=flags)

    def test_target_is_required_early_and_removed_for_prediction(self):
        with self.assertRaisesRegex(ValueError, "MONTHLY_RENT"):
            preprocess(self.raw.drop(columns="MONTHLY_RENT"), label=True)
        X = preprocess(self.raw, label=False, auxiliary_config={})
        self.assertNotIn("MONTHLY_RENT", X)

    def test_forward_split_rejects_overlap_and_keeps_boundaries(self):
        train, valid = forward_split(self.raw, train_end="2024-09", valid_start="2024-10", valid_end="2025-03")
        self.assertEqual(len(train), 8)
        self.assertEqual(len(valid), 8)
        with self.assertRaises(ValueError):
            forward_split(self.raw, train_end="2024-10", valid_start="2024-10")
