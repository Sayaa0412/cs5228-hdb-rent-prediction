# Report experiments

This folder adds the experiments supporting the final report without changing
the older model code on `lsf_dev`. No experiments were retrained while preparing
this package. Reported scores are copied from the original saved runs, not from
a new execution of the portable runner.

中文简述：已有分数看 `recorded_results/`，完整实验记录看 `records/`。
`run.py` 是整理后的主要实验入口；`diagnose.py` 只分析已有预测。
这里没有上传课程租金数据、逐行预测或模型文件，也没有重新训练。

## What is included

| Report evidence | Portable entry | Original source / saved result |
|---|---|---|
| Matched 25-input Ridge, CART, random forest, CatBoost (Fig. 2) | `run classical` | `course_classical_comparison.py`; `classical.csv` |
| Geography versus census composition | `run classical` | `census_gpu_experiment.py`; `census.csv` |
| Reference + normalized market baseline, fixed equal average | `run history` | C29 `old_reference` / `old_average`; `lags_C29.csv` |
| 11/23, 12/24, 13/25 months; 12-only and 24-only (Fig. 4) | `run history` | `renewal_gpu.py`, `renewal_ablation_gpu.py`; `lags_C29.csv`, `single_lags_C30.csv` |
| Market variants and final model (Table 4, Fig. 5) | `run history` | `renewal_market_gpu.py`, `seasonal_difference_gpu.py`, `calendar_linear.py`; `market_history_C31.csv`, `normalized_history_C32.csv`, `final_C33.csv` |
| Long-horizon bands | `run history`, `diagnose` | `final_C33_bands.csv`; `records/final_C33_all_windows.csv` |
| Complementary errors and low-rent group (Fig. 6, Tables 5–6) | `diagnose` | `analyze_calendar_linear.py`; `error_decomposition.csv`, `error_slices.csv` |

`original_sources/` preserves selected original scripts byte-for-byte, with
SHA-256 hashes. They document how the recorded results were produced. They are
**not standalone entry points**: they retain historical workspace paths, caches
and job-control dependencies. Use the portable entry points below instead.

`records/` preserves the full existing experiment ledger and source notes,
including unsuccessful directions. Its historical workspace paths and original
reproducibility notes are retained as provenance; they are not current commands.
This package covers the principal report comparisons, not portable retraining
of every exploratory model in that ledger. Earlier baseline variants in the
report remain archived results, not interchangeable reruns of the fixed controls.

## Running, only when a new run is wanted

From the repository root, in a separate Python environment:

```bash
python -m pip install -r experiments/requirements.txt
python -m experiments.run classical --train data/train.csv --blocks data/auxiliary/sg-hdb-block.csv --output experiments/runs/classical
python -m experiments.run history --train data/train.csv --blocks data/auxiliary/sg-hdb-block.csv --output experiments/runs/history
```

The course files must be supplied locally. `--folds f4` runs just the latest
origin; omit it to run all four. `--device CPU` is available where required by
the environment. Saved run results remain separate from `recorded_results/`.
An output directory must not already exist. The history command fits the
baseline and its variants in one place; it does not require old model files.

To analyze existing C33 prediction files without fitting or inference:

```bash
python -m experiments.diagnose --predictions experiments/runs/history --output experiments/runs/diagnostics
```

This also accepts the original C33 directory containing
`f1/predictions.csv` through `f4/predictions.csv`. These files stay local because
they contain actual rents. The command writes aggregate metrics, separate
forecast bands, cumulative horizons, group errors and the squared-error
decomposition. It does not select weights using the validation labels.

## Fixed protocol and limits

- Origins: March 2023, September 2023, March 2024, September 2024.
  Each primary window is its next six months; 71,040 disjoint rows in total.
  Later bands use only observed labels through March 2025. Overlapping longer
  windows are not independent replications.
- CatBoost: 650 trees, depth 7, learning rate 0.05, L2 5, seed 42, Plain boosting;
  no early stopping or outer-window hyperparameter selection in these controls.
- Market index: property Ridge penalty 20 and month penalty 0.1, three address
  folds; last-six-month linear trend, damping 0.9. Normalized responses use
  squared-scale weights to preserve the squared-error objective in SGD.
- History always uses strictly earlier months within the training prefix.
  It groups address and flat type; it does not identify repeated individual flats.
  Both 12/24-month reference inputs apply through horizon 12, then only 24 months.
  Final market Ridge uses the 24-month normalized history, alpha 10, and a fixed
  0.5 average with the reference. No future query labels build features.
- Census data are public **resident counts by dwelling type**, not numbers of
  flats. The 118 KB archived source, chronology and provenance are in
  `public_data/census2020/`. It was available at all four fitting origins;
  early training rows are not claimed to have had it at their approval date.
- Final recorded RMSE: f1 596.456, f2 531.136, f3 480.727, f4 435.373;
  pooled 515.138. These are exposed local evaluation windows, not Kaggle scores
  or an untouched estimate after model selection.

## Packaging checks

The preparation checks compare all 150,000 base and census feature rows with
the saved inputs, all five lag definitions at all four origins, and recomputed
diagnostic aggregates with the recorded C33 results. Syntax and command help
are checked. Details are in `packaging_checks.json`.

**No full training execution of this reorganized runner was performed.**
That is an explicit validation limit, not a claim of a freshly reproduced score.
Future runs can differ numerically; retain their own outputs and environment
information. The original scripts and recorded aggregates remain the evidence
for the report's exact numbers.
