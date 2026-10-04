"""Tabular regression with categorical embeddings and numeric features."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.models.base import BaseRegressionModel
from src.models.registry import register_model
from src.preprocessing import BLOCK_KEYS


class _TabularMLP(nn.Module):
    def __init__(self, numeric_count, category_sizes, embedding_dim, hidden_dims, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim, padding_idx=0) for size in category_sizes
        )
        width = numeric_count + len(category_sizes) * embedding_dim
        layers = []
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden
        self.head = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, numeric, categorical):
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.head(torch.cat([numeric, *embedded], dim=1)).squeeze(1)


@register_model("mlp")
class MLPRegressionModel(BaseRegressionModel):
    """Chronological epoch selection followed by full-prefix refitting."""

    def _model_init(self) -> None:
        self.model = _TabularMLP(
            len(self.model_config["features"]["numeric"]),
            [len(mapping) + 1 for mapping in self.category_maps.values()],
            self.model_config["embedding_dim"],
            self.model_config["hidden_dims"],
            self.model_config["dropout"],
        )

    def _preprocess(self, X_clean: pd.DataFrame) -> pd.DataFrame:
        features = self.model_config["features"]
        X_input = X_clean.copy()
        X_input["ADDRESS"] = X_input[BLOCK_KEYS].astype(str).agg("|".join, axis=1)
        X_input = X_input[features["numeric"] + features["categorical"]].copy()
        for name in features["categorical"]:
            X_input[name] = X_input[name].astype("string")
        return X_input

    def _encode(self, X_input: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        numeric = X_input[self.model_config["features"]["numeric"]]
        numeric = (numeric.fillna(self.numeric_mean) - self.numeric_mean) / self.numeric_std
        numeric_array = numeric.to_numpy(dtype=np.float32)
        if not np.isfinite(numeric_array).all():
            raise ValueError("MLP numeric features must be finite after mean imputation")
        categorical = np.column_stack([
            X_input[name].map(mapping).fillna(0).to_numpy(dtype=np.int64)
            for name, mapping in self.category_maps.items()
        ])
        return (
            torch.from_numpy(numeric_array),
            torch.from_numpy(categorical),
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> None:
        config = self.training_config
        for name in ("epochs", "validation_months", "eval_every"):
            value = config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"MLP {name} must be a positive integer")
        if X_train.empty or "RENT_APPROVAL_DATE" not in X_train:
            raise ValueError("MLP requires nonempty training data with RENT_APPROVAL_DATE")
        dates = pd.to_datetime(X_train["RENT_APPROVAL_DATE"], errors="coerce")
        if dates.isna().any():
            raise ValueError("MLP requires valid RENT_APPROVAL_DATE values")
        months = dates.dt.to_period("M")
        cutoff = months.max() - (config["validation_months"] - 1)
        fit = (months < cutoff).to_numpy()
        valid = ~fit
        if not fit.any() or not valid.any():
            raise ValueError(
                "MLP needs earlier training rows before the latest "
                f"{config['validation_months']} calendar months of inner validation"
            )
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if len(y) != len(X_train) or not np.isfinite(y).all():
            raise ValueError("MLP targets must be finite and match the training row count")
        self.best_checkpoint = None
        self.selection_history = []
        self.best_epoch = None
        self.best_validation_rmse = float("inf")
        self.training_summary = {
            "inner_fit": {"start": str(dates[fit].min()), "end": str(dates[fit].max()),
                          "rows": int(fit.sum())},
            "inner_validation": {"start": str(dates[valid].min()), "end": str(dates[valid].max()),
                                 "rows": int(valid.sum())},
            "validation_cutoff_month": str(cutoff),
            "max_selection_epochs": config["epochs"],
            "refit_rows": len(X_train),
        }
        print("MLP: Searching for the best epoch", flush=True)
        best_state = self._fit(
            X_train.iloc[fit], y[fit], config["epochs"],
            validation=(X_train.iloc[valid], y[valid]),
        )
        print(
            f"MLP: Best epoch = {self.best_epoch} | "
            f"Inner validation RMSE = {self.best_validation_rmse:.2f} SGD",
            flush=True,
        )
        self.training_summary.update(
            best_epoch=self.best_epoch, best_validation_rmse_sgd=self.best_validation_rmse,
            selection_epochs=len(self.loss_history), refit_epochs=self.best_epoch,
        )
        # Standalone CPU snapshot: no recursive checkpoint child or shared weights/stats.
        checkpoint = deepcopy(self)
        checkpoint.model.load_state_dict(best_state)
        checkpoint.fit_phase = "inner_selection"
        checkpoint.loss_history = checkpoint.loss_history[:self.best_epoch]
        self.best_checkpoint = checkpoint
        print(
            f"MLP: Starting training on all {len(X_train)} rows "
            f"for {self.best_epoch} epochs",
            flush=True,
        )
        self._fit(X_train, y, self.best_epoch)
        self.fit_phase = "full_prefix_refit"

    def _fit(self, X_train, y, epochs, validation=None):
        config = self.training_config
        torch.manual_seed(config["random_seed"])
        torch.set_num_threads(config["thread_count"])
        X_input = self._preprocess(X_train)
        numeric = X_input[self.model_config["features"]["numeric"]]
        if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
            raise ValueError("MLP numeric features cannot contain infinity")
        self.numeric_mean = numeric.mean().fillna(0)
        self.numeric_std = numeric.fillna(self.numeric_mean).std(ddof=0).fillna(1)
        self.numeric_std = self.numeric_std.mask(self.numeric_std <= 0, 1)
        self.category_maps = {
            name: {value: i + 1 for i, value in enumerate(X_input[name].dropna().unique())}
            for name in self.model_config["features"]["categorical"]
        }
        self.target_mean = float(y.mean())
        self.target_std = float(y.std()) or 1.0
        if not np.isfinite([self.target_mean, self.target_std]).all():
            raise ValueError("MLP target normalization must be finite")
        target = torch.from_numpy(((y - self.target_mean) / self.target_std).astype(np.float32))
        loader = DataLoader(
            TensorDataset(*self._encode(X_input), target),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(config["random_seed"]),
        )
        device = config["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_init()
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        loss_function = nn.MSELoss()
        self.loss_history = []
        best_state = None
        phase = "selection" if validation is not None else "refit"
        with tqdm(
            range(1, epochs + 1), desc=f"MLP {phase} ({device})", unit="epoch",
            disable=not config["verbose"], dynamic_ncols=True,
        ) as progress:
            for epoch in progress:
                total_loss = 0.0
                for numeric_batch, categorical_batch, target_batch in loader:
                    optimizer.zero_grad()
                    prediction = self.model(numeric_batch.to(device), categorical_batch.to(device))
                    loss = loss_function(prediction, target_batch.to(device))
                    if not torch.isfinite(loss):
                        raise ValueError("MLP training loss became nonfinite")
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(target_batch)
                self.loss_history.append(total_loss / len(y))
                if validation is not None:
                    record = {"epoch": epoch, "train_mse_standardized": self.loss_history[-1]}
                    if epoch == 1 or epoch % config["eval_every"] == 0 or epoch == epochs:
                        train_rmse = float(np.sqrt(np.mean((self.predict(X_train) - y) ** 2)))
                        val_rmse = float(np.sqrt(np.mean(
                            (self.predict(validation[0]) - validation[1]) ** 2
                        )))
                        self.model.train()
                        if not np.isfinite([train_rmse, val_rmse]).all():
                            raise ValueError("MLP monitoring RMSE became nonfinite")
                        record.update(train_rmse_sgd=train_rmse, validation_rmse_sgd=val_rmse)
                        tqdm.write(
                            f"MLP validation | epoch {epoch}/{epochs} | "
                            f"train_loss (standardized MSE): {self.loss_history[-1]:.6f} | "
                            f"train_RMSE (SGD): {train_rmse:.2f} | "
                            f"validation_RMSE (SGD): {val_rmse:.2f}"
                        )
                        if val_rmse < self.best_validation_rmse:
                            self.best_validation_rmse = val_rmse
                            self.best_epoch = epoch
                            best_state = {
                                name: value.detach().cpu().clone()
                                for name, value in self.model.state_dict().items()
                            }
                    self.selection_history.append(record)
        # Persist CPU tensors so inherited joblib checkpoints do not require a GPU.
        self.model.cpu()
        self.model.eval()
        return best_state

    def predict(self, X_clean: pd.DataFrame) -> np.ndarray:
        if X_clean.empty:
            return np.empty(0, dtype=np.float64)
        loader = DataLoader(
            TensorDataset(*self._encode(self._preprocess(X_clean))),
            batch_size=self.training_config["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            prediction = np.concatenate([
                self.model(numeric.to(device), categorical.to(device)).cpu().numpy()
                for numeric, categorical in loader
            ])
        return prediction.astype(np.float64) * self.target_std + self.target_mean

    def save_loss(self, path: str | Path) -> dict[str, str]:
        """Export refit loss, inner-selection diagnostics, and its best checkpoint."""
        from matplotlib.figure import Figure

        if not getattr(self, "loss_history", []):
            return {}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_path, png_path = path.with_suffix(".csv"), path.with_suffix(".png")
        loss = pd.DataFrame({
            "epoch": range(1, len(self.loss_history) + 1),
            "train_mse_standardized": self.loss_history,
        })
        loss.to_csv(csv_path, index=False)
        figure = Figure(figsize=(8, 4), layout="constrained")
        axis = figure.subplots()
        axis.plot(loss["epoch"], loss["train_mse_standardized"])
        axis.set(xlabel="Epoch", ylabel="Training MSE (standardized target)",
                 title=f"Training loss: {path.name}")
        axis.grid(alpha=0.25)
        figure.savefig(png_path, dpi=150)
        files = {"csv": str(csv_path), "png": str(png_path)}
        if not self.selection_history:
            return files
        selection_path = path.with_name(f"{path.stem}_selection.csv")
        validation_path = path.with_name(f"{path.stem}_validation.png")
        summary_path = path.with_name(f"{path.stem}_summary.json")
        checkpoint_path = path.with_name(f"{path.stem}_best_selection.joblib")
        history = pd.DataFrame(self.selection_history)
        history.to_csv(selection_path, index=False)
        monitored = history.dropna(subset=["validation_rmse_sgd"])
        figure = Figure(figsize=(8, 4), layout="constrained")
        axis = figure.subplots()
        axis.plot(monitored["epoch"], monitored["train_rmse_sgd"], label="Inner train")
        axis.plot(monitored["epoch"], monitored["validation_rmse_sgd"], label="Inner validation")
        axis.axvline(self.best_epoch, linestyle="--", color="gray", label=f"Best epoch {self.best_epoch}")
        axis.set(xlabel="Epoch", ylabel="RMSE (SGD)", title=f"Inner selection: {path.name}")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.savefig(validation_path, dpi=150)
        summary_path.write_text(json.dumps(self.training_summary, indent=2), encoding="utf-8")
        files.update(selection_csv=str(selection_path), validation_png=str(validation_path),
                     summary_json=str(summary_path))
        if self.best_checkpoint is not None:
            self.best_checkpoint.save(checkpoint_path)
            files["best_selection_checkpoint"] = str(checkpoint_path)
        return files
