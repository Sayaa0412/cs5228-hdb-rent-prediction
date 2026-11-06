"""Analyze saved C33 predictions only; never fit a model or change weights."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .model import score


def analyze(source, destination):
    parts, slices, bands, cumulative = [], [], [], []
    for fold in ("f1", "f2", "f3", "f4"):
        f = pd.read_csv(source / fold / "predictions.csv")
        f["reference"] = 2 * f.linear_average - f.linear_market
        f["fold"] = fold
        for method in ("old_average", "level_average", "linear_average"):
            for lo, hi in ((1, 6), (7, 12), (13, 18), (19, 24)):
                if f.forecast_month.max() >= hi:
                    q = f.loc[f.forecast_month.between(lo, hi)]
                    bands.append(dict(fold=fold, band=f"{lo}-{hi}", method=method, **score(q.actual, q[method])))
            for hi in (6, 12, 18, 24):
                if f.forecast_month.max() >= hi:
                    q = f.loc[f.forecast_month.between(1, hi)]
                    cumulative.append(dict(fold=fold, horizon=hi, method=method, **score(q.actual, q[method])))
        f = f.loc[f.forecast_month.between(1, 6)].copy()
        parts.append(f)
        for group, take in {"all": np.ones(len(f), bool), "low_rent_le2000": f.actual.le(2000),
                            "other_rent": f.actual.gt(2000), "history": f.has_history,
                            "no_history": ~f.has_history}.items():
            for method in ("old_average", "level_average", "linear_average"):
                q = f.loc[take]
                if len(q):
                    s = score(q.actual, q[method])
                    total_sse = np.sum((f[method] - f.actual) ** 2)
                    slices.append(dict(fold=fold, group=group, method=method, row_share=len(q)/len(f),
                                       sse_share=s["sse"]/total_sse, **s))
    pooled = pd.concat(parts)
    if not pooled.source_row.is_unique or len(pooled) != 71040:
        raise ValueError("Expected the four original disjoint primary windows (71,040 rows)")
    decomposition, metrics = [], []
    for scope, f in [("pooled", pooled)] + [(k, g) for k, g in pooled.groupby("fold")]:
        for method in ("old_average", "level_average", "linear_average", "reference", "level_market", "linear_market"):
            metrics.append(dict(fold=scope, method=method, **score(f.actual, f[method])))
        er = f.reference - f.actual
        for method in ("level_market", "linear_market"):
            em = f[method] - f.actual
            a, b, c = np.mean(er**2), np.mean(em**2), np.mean(er*em)
            mse = np.mean(((er+em)/2)**2)
            np.testing.assert_allclose(mse, (a+b+2*c)/4, rtol=1e-12)
            decomposition.append(dict(scope=scope, market=method, reference_mse=a, market_mse=b,
                error_cross_product=c, residual_correlation=np.corrcoef(er, em)[0, 1], ensemble_mse=mse))
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in {"metrics": metrics, "bands": bands, "cumulative": cumulative,
                       "slices": slices, "error_decomposition": decomposition}.items():
        pd.DataFrame(rows).to_csv(destination / (name + ".csv"), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    analyze(args.predictions, args.output)
