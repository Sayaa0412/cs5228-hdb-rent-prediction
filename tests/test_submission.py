import unittest

import numpy as np
import pandas as pd

from submit import build_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.sample = pd.DataFrame({"Id": [7, 3], "Predicted": [0, 0]}, index=[10, 20])

    def test_valid_submission_preserves_sample_id_order(self):
        result = build_submission(self.sample, [2300, 2700], test_rows=2)
        self.assertEqual(result.to_dict("list"), {"Id": [7, 3], "Predicted": [2300, 2700]})

    def test_invalid_predictions_are_rejected(self):
        for values in ([np.nan, 1], [np.inf, 1], [-1, 1], [0, 1], [1], [[1], [2]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_submission(self.sample, values, test_rows=2)

    def test_invalid_sample_schema_count_and_ids_are_rejected(self):
        for sample in (
            self.sample.rename(columns={"Id": "id"}), self.sample.iloc[:1],
            self.sample.assign(Id=[7, 7]), self.sample.assign(Id=[7, None]),
        ):
            with self.subTest(sample=sample.to_dict()), self.assertRaises(ValueError):
                build_submission(sample, [2300, 2700], test_rows=2)
