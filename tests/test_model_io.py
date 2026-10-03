import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import src.preprocessing.preprocess_pipeline as pipeline
from scripts.run_all import discover_configs
from src.models import create_model
from src.preprocessing import preprocess
from tests.helpers import rental_frame, write_auxiliary

ROOT = Path(__file__).resolve().parents[1]


class ModelIOTests(unittest.TestCase):
    def test_all_configs_train_predict_and_reload(self):
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            folder = Path(directory)
            write_auxiliary(folder)
            with patch.object(pipeline, "AUXILIARY_DATA", folder):
                configs = discover_configs(ROOT)
                self.assertEqual(len(configs), 8)
                for path in configs:
                    with self.subTest(config=path.relative_to(ROOT).as_posix()):
                        config = runpy.run_path(str(path))
                        training = config["training_config"].copy()
                        if config["model_name"] == "catboost":
                            training.update(iterations=5, thread_count=1)
                        elif config["model_name"] == "hgb":
                            training.update(max_iter=5, min_samples_leaf=2)
                        X, y = preprocess(rental_frame(), label=True, auxiliary_config=config["auxiliary_config"])
                        model = create_model(config["model_name"], model_config=config["model_config"], training_config=training)
                        model.train(X, y)
                        prediction = model.predict(X)
                        self.assertEqual(prediction.shape, (len(X),))
                        self.assertTrue(np.isfinite(prediction).all())
                        save_path = folder / "models" / f"{path.parent.name}_{path.stem}.joblib"
                        model.save(save_path)
                        np.testing.assert_array_equal(prediction, joblib.load(save_path).predict(X))

    def test_grouped_median_handles_unseen_groups(self):
        config = runpy.run_path(str(ROOT / "configs/median/town_flat_type.py"))
        model = create_model("median", model_config=config["model_config"], training_config={})
        fitting = pd.DataFrame({"TOWN": ["a", "a", "b"], "FLAT_TYPE": ["3-room", "4-room", "3-room"]})
        model.train(fitting, np.array([1000, 3000, 5000]))
        prediction = model.predict(pd.DataFrame({"TOWN": ["a", "a", "new"], "FLAT_TYPE": ["3-room", "new", "new"]}))
        np.testing.assert_array_equal(prediction, [1000, 2000, 3000])
