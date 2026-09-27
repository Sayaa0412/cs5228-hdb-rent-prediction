"""Median baselines from the original experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseRegressionModel
from src.models.registry import register_model


@register_model("median")
class MedianRegressionModel(BaseRegressionModel):
    """Global or config-driven grouped median prediction."""

    def _model_init(self) -> None:
        self.model = {}

    def _preprocess(self, X_clean: pd.DataFrame) -> pd.DataFrame:
        features = self.model_config["features"]
        columns = features["numeric"] + features["categorical"]
        return X_clean[columns].copy()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> None:
        X_input = self._preprocess(X_train)
        target = pd.Series(np.asarray(y_train), index=X_input.index)
        self._model_init()
        self.model["global_median"] = float(target.median())

        categorical = self.model_config["features"]["categorical"]
        if categorical:
            frame = X_input.assign(MONTHLY_RENT=target)
            self.model["group_medians"] = []
            for width in range(len(categorical), 0, -1):
                columns = categorical[:width]
                medians = frame.groupby(columns, observed=True)[
                    "MONTHLY_RENT"
                ].median()
                self.model["group_medians"].append((columns, medians))

    def predict(self, X_clean: pd.DataFrame) -> np.ndarray:
        X_input = self._preprocess(X_clean)
        categorical = self.model_config["features"]["categorical"]
        if not categorical:
            return np.full(len(X_input), self.model["global_median"])

        prediction = pd.Series(np.nan, index=X_input.index)
        for columns, medians in self.model["group_medians"]:
            if len(columns) == 1:
                group_prediction = X_input[columns[0]].map(medians)
            else:
                index = pd.MultiIndex.from_frame(X_input[columns])
                group_prediction = pd.Series(
                    medians.reindex(index).to_numpy(),
                    index=X_input.index,
                )
            prediction = prediction.fillna(group_prediction)
        prediction = prediction.fillna(self.model["global_median"])
        return prediction.to_numpy(dtype=float)
