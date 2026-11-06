# Existing implementation and evidence index

## Scope

These are documentation and result indexes, not a new portable source-code release. The existing final implementation depends on the research working tree and cached artifacts. Original course data, trained models and 50,000 row-level predictions are not uploaded into this Overleaf supplement. The source submission must include the relevant modules or a real repository link; no repository URL is invented here. This report revision did not rerun training, inference or baseline reproduction.

## Actual entry points and inputs

- Final full-data entry point: `work/train_calendar_linear_final.py`.
- Final four-window procedure: `work/calendar_linear.py` (saved executed copy: `outputs/calendar_linear_20261031_run01/runner.py`).
- Referenced modules include `seasonal_difference_gpu.py`, `census_gpu_experiment.py`, `course_gpu_common.py` and `outputs/selected_model_reproduction_v01/selected_model.py`; use the actual import chain rather than a baseline-only command.
- Required course inputs include `data/train.csv`, `data/test.csv` and `data/auxiliary/sg-hdb-block.csv` under the original project.
- Required saved inputs include `outputs/census_preparation_20261031/features.pkl`, the market_crossfit.csv and market_forecast.json in `outputs/selected_full_gpu_20261031_run01/model/`, and `outputs/calendar_history_final_20261031_run01/candidate_predictions.csv`.
- Final saved artifacts are in `outputs/calendar_linear_final_20261031_run01/`: linear_market.joblib, normalized_history.csv, candidate_predictions.csv, monthly_rent_predictions.csv, input_manifest.json, complete.json and export_check.json. Actual executed definitions are retained there.
- Existing scripts contain workspace-specific paths and guards against overwriting output directories. Adapt paths and choose a new output directory before any future reproduction. No command is advertised as tested in a fresh environment during this revision.

## Dependencies

Original project requirements: numpy>=2.0,<3; pandas>=2.2,<3; scikit-learn>=1.6,<2; matplotlib>=3.8,<4; catboost>=1.2.10,<2; joblib>=1.4,<2; torch>=2.6,<3; tqdm>=4.66,<5; PyYAML>=6,<7; tabm>=0.0.3,<1; rtdl_revisiting_models==0.0.2. Neural dependencies apply to archived neural experiments, not the final Ridge fit. These are existing declared ranges, not a newly verified environment lock.

## Mapping report evidence

| Report content | Existing source / supplementary index |
| --- | --- |
| Four primary windows and final improvement chain | `outputs/calendar_linear_diagnostics_20261031/comparison.csv`; copied final_component_comparison.csv; final_C33_all_windows.csv |
| Same-input Ridge/CART/forest/boosting and wider screening | four_window_results_by_source.csv; saved_metrics_all_scopes.csv, preserving each source/protocol; CS5228_全部实验总结.md |
| Lag controls and matched-history support | `outputs/report_revision_20261101_v03/tables/`; full experiment register, exact-calendar C29/C30 entries |
| Later bands and cumulative intervals | final_C33_all_windows.csv; previous-procedure scores in the archived source metric index |
| Matched market learners and complementarity | `outputs/calendar_linear_diagnostics_20261031/`; copied final_component_comparison.csv |
| Low-rent precision/recall | low_rent_rule_metrics.csv; distinct from the earlier gating experiment |
| Full features and availability | feature_contract.json; FEATURES_AND_SOURCES.md; dated release files |

RMSE pools per-record squared errors before taking the square root. The four six-month windows are distinct, but longer bands reuse outcomes across origins. No confidence or independence claim follows from those repetitions. Preserve source-row identity when joining predictions.

## Prediction format

The final local table has 50,000 predictions in original test-row order and one MONTHLY_RENT column. It is not a verified Kaggle submission. Original submit.py expects Id,Predicted using the real example-submission.csv; that template was unavailable. Never guess zero- or one-based IDs. `prepare_submission_file.py` in the final output directory can format existing predictions after the actual template is supplied, without refitting or uploading.
