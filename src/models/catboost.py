"""CatBoost models from the original experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.models.base import BaseRegressionModel
from src.models.registry import register_model
from src.preprocessing import BLOCK_KEYS


@register_model("catboost")
class CatBoostRegressionModel(BaseRegressionModel):
    """CatBoost with block, amenity, or lagged COE features."""

    def _model_init(self) -> None:
        parameters = self.model_config.copy()
        parameters.pop("variant")
        parameters.pop("features")
        parameters.update(self.training_config)
        self.model = CatBoostRegressor(**parameters)

    def _preprocess(self, X_clean: pd.DataFrame) -> pd.DataFrame:
        features = self.model_config["features"]
        numeric = features["numeric"]
        categorical = features["categorical"]
        X_input = X_clean.copy()
        X_input["ADDRESS"] = X_input[BLOCK_KEYS].astype(str).agg("|".join, axis=1)
        X_input = X_input[numeric + categorical]
        for name in categorical:
            X_input[name] = X_input[name].fillna("unknown").astype(str)
        return X_input

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> None:
        self._model_init()
        categorical = self.model_config["features"]["categorical"]
        X_input = self._preprocess(X_train)
        self.model.fit(X_input, y_train, cat_features=categorical)

    def predict(self, X_clean: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._preprocess(X_clean))
