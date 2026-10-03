# Component Guide

Root commands load a Python configuration, preprocess rental records, create a
registered model and perform the requested task. They use three public packages:
`src.preprocessing`, `src.models` and `src.evaluation`.

## Experiment Configuration

Every `configs/<model>/<variant>.py` supplies:

| Field | Purpose |
| --- | --- |
| `model_name` | Registry name: median, hgb or catboost |
| `auxiliary_config` | Switches for blocks, amenities and coe |
| `model_config` | Variant, feature columns and estimator construction parameters |
| `training_config` | Fit parameters such as iterations, learning rate and seed |
| `save_path` | Checkpoint location |
| `submit_path` | Prediction CSV location |
| `eval_path` | Forward-validation result location |
| `analysis_path` | Error-analysis result location |

Paths are relative to the repository root. Model parameters come from the configuration;
the wrappers do not contain a second set of default hyperparameters. `variant` and
`features` describe the experiment and are removed before constructing the estimator.
When estimator keys overlap, `training_config` takes precedence.

## Preprocessing

```python
from src.preprocessing import preprocess

X_train, y_train = preprocess(
    raw_train,
    label=True,
    auxiliary_config={"blocks": True, "amenities": True},
)
X_test = preprocess(
    raw_test,
    label=False,
    auxiliary_config={"blocks": True, "amenities": True},
)
```

The input is copied. `label=True` requires `MONTHLY_RENT` and returns features and
the corresponding target separately. `label=False` returns only features, removing
the target if supplied. Output row order follows the input; joins can reset index labels.

Core processing normalizes address and flat-type strings and constructs approval year,
month index and lease age. Auxiliary switches select the additional features:

| Configuration | Auxiliary files used |
| --- | --- |
| `{}` | None |
| `{"blocks": True}` | HDB blocks |
| `{"blocks": True, "amenities": True}` | HDB blocks, MRT, schools, malls |
| `{"blocks": True, "coe": True}` | HDB blocks and COE |

Unspecified switches are false when a dictionary is provided. Unknown switches and
non-boolean values are rejected. Amenities require `blocks=True`. Omitting the
configuration or passing `None` retains the original full-feature preprocessing
behavior for compatibility; maintained experiment files always use explicit switches.

Tables and static amenity features are cached within the process using resolved file
paths, modification times and sizes. Changes to an auxiliary file invalidate its cache.
Low-level joins copy their inputs, so callers do not mutate the cached tables.
COE features use the preceding calendar month; January 2021 has no supplied lagged value.

For a new feature, add a preprocessing component, connect it to `preprocess()`, and
include its columns in the relevant experiment configuration. Keep all label-dependent
transformations fitted on the training partition; the existing auxiliary features use
static tables or earlier calendar months.

## Models

```python
from src.models import create_model

model = create_model(
    config["model_name"],
    model_config=config["model_config"],
    training_config=config["training_config"],
)
model.train(X_train, y_train)
prediction = model.predict(X_valid)
model.save("chkpt/catboost/amenities.joblib")
```

Concrete models inherit `BaseRegressionModel` and implement `_model_init()`,
`_preprocess()`, `train()` and `predict()`. The base class saves the complete wrapper
with joblib. Shapes are `(n_samples, n_features)` for X and `(n_samples,)` for targets
and predictions. Use the same feature configuration for training and submission.

The model's `_preprocess()` selects configured columns and adapts them to the estimator.
HGB converts categorical columns to pandas categories; sklearn retains the fitted
encoding. CatBoost constructs the address category and fills missing categorical
values with `unknown`. Median models fall back from the configured complete grouping
through progressively shorter prefix groups, finally using the global median.

To add a model, create `src/models/<name>.py`, inherit the base interface and decorate
the class with `@register_model("<name>")`. Add configuration files under
`configs/<name>/`. Registry discovery imports model modules automatically, and the batch
runner discovers their configuration files. Imports must not read data or start training.

## Evaluation and Submission

`src.evaluation.windows` owns the holdout, rolling and horizon periods.
`regression_metrics()` validates targets and predictions and returns rounded MAE,
RMSE and R-squared. `forward_split()` rejects empty or overlapping time partitions.
`eval.py` fits each distinct period once, reusing its result when holdout, rolling and
horizon sections refer to the same window.

`analysis.py` uses the shared holdout and reports signed and absolute errors by flat type,
month, town and target-defined rent band. If no flat-type group has 100 observations,
the highest-error large-group field is null.

`submit.build_submission()` requires the exact sample columns `Id,Predicted`, matching
row counts, unique nonmissing IDs, and one positive finite scalar prediction per test
record. Validation happens before writing the output file.

## Batch Execution and Checks

`scripts/run_all.py` accepts `--task`, optional `--model` and `--variant` filters, and
`--dry-run`. It starts commands with `sys.executable` from the repository root and stops
on the first failed child process, returning a nonzero exit code.

Run `python -m unittest discover -s tests -v` before committing shared changes. The
synthetic tests need no course files. Baseline metadata records known full-data hashes
and historical scores; it is not regenerated by ordinary training or evaluation runs.
