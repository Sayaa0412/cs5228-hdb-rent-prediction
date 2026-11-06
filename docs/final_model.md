# Final report model

The retained model has two components:

1. **CatBoost reference:** the 13 selected property/location/date fields plus
   exact address × flat-type rental histories. Use the 12- and 24-month histories
   through forecast month 12, and a separately fitted 24-month-only model after that.
2. **Normalized Ridge:** estimate a monthly market scale, predict rent divided by
   that scale from static property fields and normalized 24-month history, then
   restore the forecast market scale. Average this prediction with the reference.

The final model does not need census, resale, HDB Property Information additions,
MRT openings, COE, or stock data. Those trials remain in the separate experiment
archive; they are not dependencies of the retained model.

## Files and commands

`src/final_model.py` holds the model and features. `final.py` is the entry point.
Run the following from the repository root after installing `requirements.txt`.
Only these course files are needed:

```text
data/train.csv
data/test.csv                         # needed only for prediction
data/auxiliary/sg-hdb-block.csv
```

```bash
# Fit the retained model to all labeled rows.
python final.py train --out output/final_model

# Predict from the saved model, preserving the test file's row order.
python final.py predict --model-dir output/final_model --out output/final_predictions

# Optional future evaluation: fit the four chronological development origins.
python final.py evaluate --out output/final_evaluation
```

`--data-dir PATH` selects another data directory. `evaluate --folds f4` runs just
one origin. Training defaults to `--device GPU`; `--device CPU` is also available.
Directories must be new, so existing results are never overwritten. A complete
training directory contains two reference models, the Ridge/history state, the
market-index diagnostics, and `complete.json`. Prediction needs this complete
directory and the query/block CSVs; it never refits a model.

`monthly_rent_predictions.csv` contains one `MONTHLY_RENT` column. The companion
`predictions.csv` includes the components, horizon, and zero-based input row number.
These are local predictions, not an automatically formatted or uploaded Kaggle
submission. Use the actual competition template if IDs or other column names are
required; do not guess them.

## Recorded results

These values come from the completed October 31 experiments, not a rerun of this
code consolidation. The earlier column already has exact calendar histories, but
uses the earlier boosted market component without normalized history.

| Train through | Evaluate | Rows | Earlier history ensemble | Final ensemble |
| --- | --- | ---: | ---: | ---: |
| 2023-03 | 2023-04–2023-09 | 17,905 | 600.320 | 596.456 |
| 2023-09 | 2023-10–2024-03 | 17,977 | 533.401 | 531.136 |
| 2024-03 | 2024-04–2024-09 | 17,960 | 484.155 | 480.727 |
| 2024-09 | 2024-10–2025-03 | 17,198 | 438.578 | 435.373 |
| Pooled | Four primary windows | 71,040 | 518.322 | **515.138** |

RMSE is in SGD. Pooled RMSE takes the square root of total squared error divided
by total rows; it is not the average of the four RMSEs. The full saved component
comparison and later six-month bands are in
[`benchmarks/final_model/`](../benchmarks/final_model/). `old_average` is the earlier
history ensemble above, `level_average` replaces its market component with boosted
normalized history, and `linear_average` is the final fixed ensemble. `level_market`
and `linear_market` are the two individual market components.

All four windows informed development. Longer horizons overlap across origins and
are not independent test sets. There are no held-out Kaggle labels here.

## Definitions needed to interpret or reproduce the model

- Normalize address text and join blocks many-to-one. Retain duplicate-looking
  rental records. Treat `-1` block completion years/floor counts as missing.
- A reference history contains mean log-rent anomaly, log count, and an unavailable
  flag from exactly `t-12` or `t-24`. The anomaly subtracts the flat-type/source-month
  median. No query labels or labels beyond the fitting origin enter the history.
- The market index is a sparse Ridge decomposition of log rent into property,
  address/type, and month effects. Property penalty is 20 and month penalty 0.1.
  Center the index on its first six months. Three folds grouped by address exclude
  each training address from its own target normalizer. This is retrospective
  estimation within the origin, not online estimation at each past transaction.
- The response Ridge uses 12 static fields and three normalized 24-month-history
  features. Fit numeric median imputation/scaling and categorical one-hot encoding
  on the prefix only. Use alpha 10 and weights proportional to squared market scale.
- Extrapolate the monthly log index using an OLS line on its last six observations
  and damping 0.9. Multiply the normalized prediction by this forecast scale.
- Each reference uses 650 trees, depth 7, learning rate 0.05, L2 penalty 5, and seed
  42. The final average is fixed at 0.5. No validation-based tuning, clipping, or
  true-rent routing occurs in this entry point.
- Queries in the fitted month (`h=0`) use its observed prefix index. Such rows occur
  in the supplied test file and are not out-of-time validation. Earlier queries
  and horizons beyond 24 months are rejected.

## Code provenance and checks

This compact entry point consolidates the existing final definitions from
`selected_model.py`, `renewal_inference.py`, `train_calendar_history_final.py`,
`calendar_linear.py`, and `train_calendar_linear_final.py` in the retained research
archive. It replaces their machine-specific paths and cached intermediate inputs
with arguments and computation from the supplied training prefix. The older
baseline implementations in this repository are unchanged.

The consolidation is checked with synthetic temporal/row-order tests and feature
comparisons against saved research artifacts. No experiments were rerun for this
repository update. A fresh end-to-end training run of this consolidated entry point
has **not** been performed, and identical fresh RMSEs are not guaranteed.

Run the non-training checks with:

```bash
python -m unittest tests.test_final_model
```

Keep the original data, trained models, row-level predictions, and complete
experiment archive separately. Only source code and small aggregate score tables
are added here.
