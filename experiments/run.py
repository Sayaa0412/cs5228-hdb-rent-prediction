"""Fixed report comparisons. Running this module explicitly starts training."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from . import model as m
from .census import build as census_features

LAGS = {"exact": (12, 24), "minus1": (11, 23), "plus1": (13, 25),
        "only12": (12,), "only24": (24,)}


def processor(x, cats):
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                   StandardScaler()), [c for c in x if c not in cats]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", dtype=np.float64), cats),
    ], sparse_threshold=1)


def tree_fit(folder, name, x, y, train, query, device, cats=m.CAT, weights=None):
    print("Fitting", folder.name, name, flush=True)
    model = CatBoostRegressor(**m.reference_params(device))
    model.fit(x.loc[train], y, cat_features=cats, sample_weight=weights)
    model.save_model(str(folder / (name + ".cbm")))
    return model.predict(x.loc[query])


def classical(folder, x, y, train, query, device):
    cats = m.CAT + ["PLANNING_AREA", "SUBZONE"]
    prep = processor(x, cats)
    a = prep.fit_transform(x.loc[train]).tocsr()
    q = prep.transform(x.loc[query]).tocsr()
    joblib.dump(prep, folder / "preprocessor.joblib")
    models = {
        "ridge": Ridge(alpha=10, solver="lsqr", tol=1e-6, max_iter=5000),
        "cart": DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=12,
            min_samples_leaf=20, max_features=.5, bootstrap=True, random_state=42, n_jobs=4),
    }
    predictions = {}
    for name, model in models.items():
        print("Fitting", folder.name, name, flush=True)
        model.fit(a, y)
        predictions[name] = model.predict(q)
        joblib.dump(model, folder / (name + ".joblib"))
    predictions["catboost_25"] = tree_fit(folder, "catboost_25", x, y, train, query, device, cats)
    # Geography-only control separates place identity from census composition.
    predictions["catboost_15"] = tree_fit(folder, "catboost_15", x[m.NUM + cats],
        y, train, query, device, cats)
    predictions["catboost_13"] = tree_fit(folder, "catboost_13", x[m.NUM + m.CAT],
        y, train, query, device)
    return predictions


def histories(folder, raw, x, y, train, query, device):
    first, last = int(x.loc[train, "MONTH_INDEX"].min()), int(x.loc[train, "MONTH_INDEX"].max())
    h = x.loc[query, "MONTH_INDEX"].to_numpy() - last
    static = m.static_features(raw, x)
    offset, index, folds = m.crossfit_index(static.loc[train], x.loc[train, "MONTH_INDEX"], y)
    config = m.forecast_config(index)
    scale = np.exp(offset)
    qscale = m.forecast_scale(config, h)
    z = y / scale
    weights = scale ** 2 / np.mean(scale ** 2)
    pd.DataFrame({"source_row": np.flatnonzero(train), "market_offset": offset,
                  "crossfit_fold": folds}).to_csv(folder / "crossfit_index.csv", index=False)
    index.to_csv(folder / "market_index.csv")
    (folder / "forecast.json").write_text(json.dumps(config, indent=2))
    reference = tree_fit(folder, "reference", x, y, train, query, device)
    market = tree_fit(folder, "old_market", static, z, train, query, device, weights=weights) * qscale
    out = {"reference": reference, "old_market": market, "baseline_average": (reference + market) / 2}
    table = m.reference_history(x.loc[train], y)
    history_inputs = {}
    for name, lags in LAGS.items():
        extra = m.reference_features(table, x, first, last, lags)
        history_inputs[name] = extra
        pred = tree_fit(folder, name + "_reference", pd.concat([x, extra], axis=1),
                        y, train, query, device)
        out[name + "_reference"] = pred
        out[name + "_average"] = (pred + market) / 2
    routed = np.where(h <= 12, out["exact_reference"], out["only24_reference"])
    out["routed_reference"] = routed
    out["old_average"] = (routed + market) / 2
    # C31: changing the market tree's inputs, with the reference held fixed.
    for name in ("exact", "only24"):
        xx = pd.concat([static, history_inputs[name]], axis=1)
        out[name + "_history_market"] = tree_fit(folder, name + "_history_market", xx,
            z, train, query, device, weights=weights) * qscale
    c31 = np.where(h <= 12, out["exact_history_market"], out["only24_history_market"])
    out["C31_average"] = (routed + c31) / 2
    # C32/C33: same normalized historical representation and SGD-equivalent weights.
    norm_table = m.normalized_history(x.loc[train], z)
    extra = m.normalized_features(norm_table, x, first, last)
    xx = pd.concat([static, extra], axis=1)
    anchor = extra.HISTORY24_NORMALIZED_MEAN.fillna(0).to_numpy()
    for name in ("level", "difference"):
        subtract = anchor[train] if name == "difference" else 0
        add = anchor[query] if name == "difference" else 0
        p = tree_fit(folder, name + "_market", xx, z - subtract,
                     train, query, device, weights=weights)
        out[name + "_market"] = (p + add) * qscale
        out[name + "_average"] = (routed + out[name + "_market"]) / 2
    prep = processor(xx, m.CAT)
    a = prep.fit_transform(xx.loc[train])
    linear = Ridge(alpha=10, solver="lsqr", tol=1e-6, max_iter=5000)
    linear.fit(a, z, sample_weight=weights)
    if int(np.max(linear.n_iter_)) >= 5000:
        raise RuntimeError("Ridge reached its iteration limit")
    out["linear_market"] = linear.predict(prep.transform(xx.loc[query])) * qscale
    out["linear_average"] = (routed + out["linear_market"]) / 2
    joblib.dump(dict(preprocessor=prep, model=linear, columns=list(xx)), folder / "linear.joblib")
    return out, extra.loc[query, "HISTORY24_LOG_COUNT"].gt(0).to_numpy()


def run(args):
    raw = pd.read_csv(args.train, dtype={"BLOCK": str}).reset_index(drop=True)
    blocks = pd.read_csv(args.blocks, dtype={"BLOCK": str})
    raw = raw.loc[raw.RENT_APPROVAL_DATE <= "2025-03"].reset_index(drop=True)
    if not np.isfinite(raw.MONTHLY_RENT).all() or not raw.MONTHLY_RENT.gt(0).all():
        raise ValueError("Positive finite training rents are required")
    x = m.features(raw, blocks)
    enriched = None
    if args.suite == "classical":
        geo, extra, _, _ = census_features(x, blocks, args.census)
        enriched = pd.concat([x, geo, extra], axis=1)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.train, args.blocks]}
    metadata = dict(suite=args.suite, folds=args.folds, device=args.device, input_sha256=inputs,
        fixed_params=m.reference_params(args.device), validation_end="2025-03",
        note="New execution of packaged definitions; not the archived run that produced the report.")
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
    parts, rows, bands = [], [], []
    with threadpool_limits(limits=4):
        for fold in args.folds:
            cutoff = m.FOLDS[fold]
            train = raw.RENT_APPROVAL_DATE.le(cutoff).to_numpy()
            query = raw.RENT_APPROVAL_DATE.gt(cutoff).to_numpy()
            folder = args.output / fold
            folder.mkdir()
            y = raw.loc[train, "MONTHLY_RENT"].to_numpy()
            out = raw.loc[query, ["RENT_APPROVAL_DATE", "TOWN", "FLAT_TYPE"]].copy()
            out.insert(0, "source_row", np.flatnonzero(query))
            out["actual"] = raw.loc[query, "MONTHLY_RENT"]
            out["forecast_month"] = x.loc[query, "MONTH_INDEX"] - x.loc[train, "MONTH_INDEX"].max()
            if args.suite == "classical":
                predictions = classical(folder, enriched, y, train, query, args.device)
            else:
                predictions, has_history = histories(folder, raw, x, y, train, query, args.device)
                out["has_history"] = has_history
            for name, pred in predictions.items():
                if not np.isfinite(pred).all():
                    raise ValueError("Non-finite predictions: " + name)
                out[name] = pred
                for lo, hi in ((1, 6), (7, 12), (13, 18), (19, 24)):
                    if out.forecast_month.max() < hi:
                        continue
                    take = out.forecast_month.between(lo, hi)
                    s = dict(fold=fold, method=name, **m.score(out.loc[take, "actual"], out.loc[take, name]))
                    bands.append(dict(band=f"{lo}-{hi}", **s))
                    if lo == 1:
                        rows.append(s)
            out.to_csv(folder / "predictions.csv", index=False)
            parts.append(out.loc[out.forecast_month.le(6)])
            pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    pooled = pd.concat(parts)
    if not pooled.source_row.is_unique:
        raise ValueError("Primary evaluation windows overlap")
    for name in predictions:
        rows.append(dict(fold="pooled", method=name, **m.score(pooled.actual, pooled[name])))
    pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(bands).to_csv(args.output / "bands.csv", index=False)
    (args.output / "complete.json").write_text(json.dumps(dict(rows=len(pooled), folds=args.folds)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=["classical", "history"])
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--blocks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", nargs="+", choices=list(m.FOLDS), default=list(m.FOLDS))
    p.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    p.add_argument("--census", type=Path, default=Path(__file__).parent / "public_data/census2020/census2020_dwelling_raw.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
