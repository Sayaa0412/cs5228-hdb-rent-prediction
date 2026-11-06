"""Final report model: exact rental histories + market-normalized Ridge.

Shared definitions for the report comparisons. See experiments/README.md.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing.core import prepare_core
from .preprocessing.blocks import join_block_features


CAT = ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "REGION", "STREET", "ADDRESS"]
NUM = ["MONTH_INDEX", "FLOOR_AREA_SQM", "LEASE_AGE", "LATITUDE",
       "LONGITUDE", "MAX_FLOOR", "BUILDING_AGE"]
STATIC_NUM = ["FLOOR_AREA_SQM", "LATITUDE", "LONGITUDE", "MAX_FLOOR",
              "LEASE_START_YEAR", "COMPLETION_YEAR"]
FOLDS = {"f1": "2023-03", "f2": "2023-09", "f3": "2024-03", "f4": "2024-09"}


def features(raw, blocks):
    """Keep row order and duplicate-looking transactions; never use the target."""
    source = raw.reset_index(drop=True).copy()
    source["source_row"] = np.arange(len(source))
    lookup = blocks.copy()
    # -1 means unknown in the supplied block table.
    for name in ("YEAR_COMPLETED", "MAX_FLOOR"):
        lookup[name] = lookup[name].mask(lookup[name].eq(-1))
    clean = join_block_features(prepare_core(source), lookup)
    np.testing.assert_array_equal(clean.source_row, source.source_row)
    clean["ADDRESS"] = clean[["TOWN", "BLOCK", "STREET"]].astype(str).agg("|".join, axis=1)
    x = clean[NUM + CAT].copy()
    for name in CAT:
        x[name] = x[name].fillna("unknown").astype(str)
    return x


def static_features(raw, x):
    z = x.drop(columns=["MONTH_INDEX", "LEASE_AGE", "BUILDING_AGE"]).copy()
    year = pd.to_datetime(raw.RENT_APPROVAL_DATE.to_numpy(), format="%Y-%m").year
    z["LEASE_START_YEAR"] = year - x.LEASE_AGE
    z["COMPLETION_YEAR"] = year - x.BUILDING_AGE
    return z[STATIC_NUM + CAT]


def reference_history(x, y):
    """Address/type/month mean log rent relative to the type/month median."""
    source = pd.DataFrame({"key": x.ADDRESS + "|" + x.FLAT_TYPE,
                           "type": x.FLAT_TYPE, "month": x.MONTH_INDEX,
                           "value": np.log(np.asarray(y))})
    source["value"] -= source.groupby(["type", "month"]).value.transform("median")
    return source.groupby(["key", "month"]).value.agg(["mean", "size"])


def normalized_history(x, normalized_y):
    source = pd.DataFrame({"key": x.ADDRESS + "|" + x.FLAT_TYPE,
                           "month": x.MONTH_INDEX, "value": np.asarray(normalized_y)})
    return source.groupby(["key", "month"]).value.agg(["mean", "size"])


def _lookup(table, query, first, last, lag):
    month = query.MONTH_INDEX - lag
    available = month.between(first, last)
    keys = pd.MultiIndex.from_arrays([query.ADDRESS + "|" + query.FLAT_TYPE, month])
    values = table.reindex(keys).set_axis(query.index)
    return values, available


def reference_features(table, query, first, last, lags=(12, 24)):
    out = pd.DataFrame(index=query.index)
    for k, lag in enumerate(lags):
        if lag <= 0:
            raise ValueError("History lags must be strictly positive.")
        values, available = _lookup(table, query, first, last, lag)
        out[f"PAST{k}_LOG_COUNT"] = np.log1p(values["size"].fillna(0)).where(available)
        out[f"PAST{k}_LOG_RENT_ANOMALY"] = values["mean"].where(available)
        out[f"PAST{k}_UNAVAILABLE"] = (~available).astype(float)
    return out


def normalized_features(table, query, first, last):
    values, available = _lookup(table, query, first, last, 24)
    return pd.DataFrame({
        "HISTORY24_NORMALIZED_MEAN": values["mean"].where(available),
        "HISTORY24_LOG_COUNT": np.log1p(values["size"].fillna(0)).where(available),
        "HISTORY24_UNAVAILABLE": (~available).astype(float),
    }, index=query.index)


def index_fit(x, months, log_y):
    """Hedonic Ridge index: property penalty 20, month penalty 0.1."""
    numeric = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                            StandardScaler())
    xn = numeric.fit_transform(x[STATIC_NUM])
    cats = x[CAT].copy()
    cats["ADDRESS_TYPE"] = x.ADDRESS + "|" + x.FLAT_TYPE
    cats["MONTH"] = np.asarray(months).astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    xc = encoder.fit_transform(cats)
    design = sp.hstack([sp.csr_matrix(xn), xc], format="csr")
    start = len(STATIC_NUM) + sum(len(c) for c in encoder.categories_[:-1])
    month_scale = np.sqrt(20 / .1)
    column_scale = np.ones(design.shape[1])
    column_scale[start:] = month_scale
    model = Ridge(alpha=20, solver="lsqr", tol=1e-6, max_iter=2000)
    model.fit(design.multiply(column_scale).tocsr(), np.asarray(log_y))
    if int(np.max(model.n_iter_)) >= 2000:
        raise RuntimeError("Market-index Ridge reached its iteration limit.")
    month_ids = encoder.categories_[-1].astype(int)
    index = pd.Series(model.coef_[start:start + len(month_ids)] * month_scale,
                      index=month_ids).sort_index()
    index -= index.iloc[:6].mean()
    return index


def crossfit_index(x, months, y):
    """Three address-group folds exclude each address from its own normalizer."""
    groups = x.ADDRESS.to_numpy()
    months, log_y = np.asarray(months), np.log(np.asarray(y))
    offsets, fold_ids = np.full(len(x), np.nan), np.full(len(x), -1)
    for k, (train, valid) in enumerate(GroupKFold(n_splits=3).split(x, groups=groups)):
        index = index_fit(x.iloc[train], months[train], log_y[train])
        offsets[valid] = index.reindex(months[valid]).to_numpy()
        fold_ids[valid] = k
    if not np.isfinite(offsets).all():
        raise ValueError("A held-out month has no market index in its fitting fold.")
    return offsets, index_fit(x, months, log_y), fold_ids


def forecast_config(index):
    if len(index) < 6 or not np.array_equal(index.index, np.arange(index.index.min(), index.index.max() + 1)):
        raise ValueError("Market forecasting requires at least six consecutive months.")
    slope, level = np.polyfit(np.arange(-5, 1), index.iloc[-6:].to_numpy(), 1)
    return {"level": float(level), "slope": float(slope), "phi": .9,
            "last_observed": float(index.iloc[-1]), "cutoff_month": int(index.index.max())}


def forecast_scale(config, horizon):
    h = np.asarray(horizon, dtype=int)
    if (h < 0).any() or (h > 24).any():
        raise ValueError("Queries must be between the fitted month and 24 months after it.")
    phi = config["phi"]
    offset = config["level"] + config["slope"] * phi * (1 - phi ** h) / (1 - phi)
    offset = np.where(h == 0, config["last_observed"], offset)
    return np.exp(offset)


def reference_params(device="GPU"):
    params = dict(iterations=650, depth=7, learning_rate=.05, l2_leaf_reg=5,
                  loss_function="RMSE", eval_metric="RMSE", random_seed=42,
                  thread_count=4, task_type=device, boosting_type="Plain",
                  verbose=False, allow_writing_files=False)
    if device == "GPU":
        params.update(devices="0", gpu_ram_part=.6, border_count=128,
                      max_ctr_complexity=4, ctr_target_border_count=1,
                      one_hot_max_size=2, bootstrap_type="Bayesian", bagging_temperature=1)
    return params


def score(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return dict(n=len(error), rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.mean(np.abs(error))), bias=float(np.mean(error)),
                sse=float(np.sum(error ** 2)))
