"""Train, evaluate, or predict with the final report model. No automatic submission."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.final_model import FOLDS, fit_final, predict_final, score


def evaluate(raw, blocks, destination, folds, device):
    destination.mkdir(parents=True, exist_ok=False)
    primary, bands, pooled = [], [], []
    for fold in folds:
        cutoff = FOLDS[fold]
        dates = pd.to_datetime(raw.RENT_APPROVAL_DATE, format="%Y-%m").dt.to_period("M")
        origin = pd.Period(cutoff, freq="M")
        train = dates <= origin
        valid = (dates > origin) & (dates <= min(origin + 24, pd.Period("2025-03")))
        if not train.any() or not valid.any() or dates.max() < origin + 6:
            raise ValueError(f"Data do not cover the complete primary window for {fold}.")
        model_dir = destination / fold
        print(f"Fitting {fold}, training through {cutoff}", flush=True)
        fit_final(raw.loc[train], blocks, model_dir, device)
        predictions = predict_final(raw.loc[valid].drop(columns="MONTHLY_RENT"), blocks, model_dir)
        predictions["source_row"] = raw.index[valid].to_numpy()
        predictions["actual"] = raw.loc[valid, "MONTHLY_RENT"].to_numpy()
        predictions.to_csv(model_dir / "predictions.csv", index=False)
        six = predictions.loc[predictions.forecast_month <= 6]
        pooled.append(six)
        primary.append(dict(fold=fold, **score(six.actual, six.MONTHLY_RENT)))
        for lo, hi in ((1, 6), (7, 12), (13, 18), (19, 24)):
            if predictions.forecast_month.max() < hi:
                continue
            band = predictions.loc[predictions.forecast_month.between(lo, hi)]
            bands.append(dict(fold=fold, band=f"{lo}-{hi}", **score(band.actual, band.MONTHLY_RENT)))
        pd.DataFrame(primary).to_csv(destination / "metrics.csv", index=False)
        pd.DataFrame(bands).to_csv(destination / "bands.csv", index=False)
    all_primary = pd.concat(pooled, ignore_index=True)
    if not all_primary.source_row.is_unique:
        raise ValueError("Primary windows overlap.")
    primary.append(dict(fold="pooled", **score(all_primary.actual, all_primary.MONTHLY_RENT)))
    pd.DataFrame(primary).to_csv(destination / "metrics.csv", index=False)
    (destination / "complete.json").write_text(json.dumps({
        "folds": folds, "primary_rows": len(all_primary),
        "note": "Exposed development windows, not untouched test performance.",
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(primary).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "predict"):
        sub = commands.add_parser(command)
        sub.add_argument("--data-dir", type=Path, default=Path("data"))
        sub.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten")
        if command == "predict":
            sub.add_argument("--model-dir", type=Path, required=True)
            sub.add_argument("--query", type=Path, help="Default: DATA_DIR/test.csv")
        else:
            sub.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
        if command == "evaluate":
            sub.add_argument("--folds", nargs="+", choices=list(FOLDS), default=list(FOLDS))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; choose a new directory.")
    blocks = pd.read_csv(args.data_dir / "auxiliary" / "sg-hdb-block.csv")
    if args.command == "predict":
        query = pd.read_csv(args.query or args.data_dir / "test.csv")
        result = predict_final(query, blocks, args.model_dir)
        args.out.mkdir(parents=True, exist_ok=False)
        result.to_csv(args.out / "predictions.csv", index=False)
        result[["MONTHLY_RENT"]].to_csv(args.out / "monthly_rent_predictions.csv", index=False)
        print(f"Saved {len(result)} predictions in original query order to {args.out}")
    else:
        raw = pd.read_csv(args.data_dir / "train.csv")
        if args.command == "train":
            print(json.dumps(fit_final(raw, blocks, args.out, args.device), indent=2))
        else:
            if len(set(args.folds)) != len(args.folds):
                parser.error("Each fold may be requested only once.")
            evaluate(raw, blocks, args.out, args.folds, args.device)


if __name__ == "__main__":
    main()
