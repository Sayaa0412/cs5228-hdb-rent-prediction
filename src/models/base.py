"""Abstract interface shared by all regression models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BaseRegressionModel(ABC):
    """Define the required interface for a project regression model.

    Subclasses own their model initialization, preprocessing, training and
    prediction logic. The base class stores both configurations and provides
    the shared serialization method.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
    ) -> None:
        self.model_config = dict(model_config)
        self.training_config = dict(training_config)
        self.model: Any | None = None

    @abstractmethod
    def _model_init(self) -> None:
        """Initialize the underlying estimator and assign it to ``self.model``."""
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, X_clean: pd.DataFrame) -> Any:
        """Convert clean project features into this model's input format."""
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> None:
        """Fit the model and store the fitted estimator on this instance."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_clean: pd.DataFrame) -> np.ndarray:
        """Return one prediction for each row in ``X_clean``."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Save the fitted model to ``path``."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
