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
        self.loss_history: list[float] = []

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

    def save_loss(self, path: str | Path) -> dict[str, str]:
        """Export recorded epoch losses; models without a history produce no files."""
        if not self.loss_history:
            return {}
        from matplotlib.figure import Figure

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_path, png_path = path.with_suffix(".csv"), path.with_suffix(".png")
        history = pd.DataFrame({
            "epoch": range(1, len(self.loss_history) + 1),
            "train_mse_standardized": self.loss_history,
        })
        history.to_csv(csv_path, index=False)
        figure = Figure(figsize=(8, 4), layout="constrained")
        axis = figure.subplots()
        axis.plot(history["epoch"], history["train_mse_standardized"])
        axis.set(
            xlabel="Epoch", ylabel="Training MSE (standardized target)",
            title=f"Training loss: {path.name}",
        )
        axis.grid(alpha=0.25)
        figure.savefig(png_path, dpi=150)
        return {"csv": str(csv_path), "png": str(png_path)}
