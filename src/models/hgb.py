"""Histogram gradient boosting models from the original experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.base import BaseRegressionModel
from src.models.registry import register_model


@register_model("hgb")
class HGBRegressionModel(BaseRegressionModel):
    """HGB with core, HDB block, or amenity features."""

    def _model_init(self) -> None:
        parameters = self.model_config.copy()
        parameters.pop("variant")
        parameters.pop("features")
        parameters.update(self.training_config)
        self.model = HistGradientBoostingRegressor(**parameters)

    def _preprocess(self, X_clean: pd.DataFrame) -> pd.DataFrame:
        features = self.model_config["features"]
        numeric = features["numeric"]
        categorical = features["categorical"]
        X_input = X_clean[numeric + categorical].copy()
        for name in categorical:
            X_input[name] = X_input[name].fillna("unknown").astype("category")
        return X_input

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> None:
        self._model_init()
        X_input = self._preprocess(X_train)
        self.model.fit(X_input, y_train)

    def predict(self, X_clean: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._preprocess(X_clean))
