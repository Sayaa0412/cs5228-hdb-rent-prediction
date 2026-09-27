"""Model registration, discovery and construction utilities."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

from src.models.base import BaseRegressionModel


ModelClass = type[BaseRegressionModel]
_MODEL_REGISTRY: dict[str, ModelClass] = {}
_DISCOVERY_COMPLETE = False


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Model name cannot be empty")
    return normalized


def register_model(name: str) -> Callable[[ModelClass], ModelClass]:
    """Register a model class under a case-insensitive name.

    Example::

        @register_model("catboost")
        class CatBoostRegressionModel(BaseRegressionModel):
            ...
    """
    normalized = _normalize_name(name)

    def decorator(model_class: ModelClass) -> ModelClass:
        if not isinstance(model_class, type) or not issubclass(
            model_class, BaseRegressionModel
        ):
            raise TypeError("Registered models must inherit BaseRegressionModel")
        existing = _MODEL_REGISTRY.get(normalized)
        if existing is not None and existing is not model_class:
            raise ValueError(
                f"Model name {normalized!r} is already registered by "
                f"{existing.__module__}.{existing.__name__}"
            )
        _MODEL_REGISTRY[normalized] = model_class
        return model_class

    return decorator


def discover_models() -> None:
    """Import model modules so their registration decorators execute."""
    global _DISCOVERY_COMPLETE
    if _DISCOVERY_COMPLETE:
        return

    package = importlib.import_module("src.models")
    skipped_modules = {"base", "registry"}
    for module in pkgutil.iter_modules(package.__path__):
        if module.name.startswith("_") or module.name in skipped_modules:
            continue
        importlib.import_module(f"{package.__name__}.{module.name}")
    _DISCOVERY_COMPLETE = True


def get_model_class(name: str) -> ModelClass:
    """Return the registered class for ``name`` after model discovery."""
    discover_models()
    normalized = _normalize_name(name)
    try:
        return _MODEL_REGISTRY[normalized]
    except KeyError:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "none"
        raise ValueError(
            f"Unknown model {normalized!r}. Available models: {available}"
        ) from None


def create_model(
    name: str,
    *,
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
) -> BaseRegressionModel:
    """Instantiate a registered model without importing its concrete module."""
    model_class = get_model_class(name)
    return model_class(
        model_config={} if model_config is None else model_config,
        training_config={} if training_config is None else training_config,
    )


def available_models() -> tuple[str, ...]:
    """Return all registered model names in deterministic order."""
    discover_models()
    return tuple(sorted(_MODEL_REGISTRY))
