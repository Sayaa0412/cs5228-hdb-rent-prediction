# CS5228 HDB rent prediction

Team 29: HAN MINGJIA, LIU HENGYAN, LIU RUIXI and LIU SIFEI.

Models, feature processing and forward validation for monthly HDB rent prediction.
This repository tracks code, experiment configurations, tests and selected baseline
metadata. Course data, presentations, prediction CSVs and trained checkpoints stay local.

## Setup

Use Python 3.11 or newer. From the repository root, create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

An existing Conda environment works with the same pip and Python commands.
If PowerShell activation is unavailable, use `.\.venv\Scripts\python.exe` directly.
The requirements specify a PyTorch version range without fixing a CUDA build.
On Windows or Linux, install the appropriate PyTorch build before installing the
requirements. For CPU-only use:

```bash
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For NVIDIA GPUs, choose the CUDA build compatible with your driver using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then run
`python -m pip install -r requirements.txt`. For example, the official wheel indexes
include `https://download.pytorch.org/whl/cu126` and
`https://download.pytorch.org/whl/cu130`; use the index appropriate for your environment.
On macOS, use the standard pip installation shown above. An already installed
PyTorch build satisfying the version range is retained by the requirements command.

## Local Data

Place the course-provided files in the following local paths:

```text
data/
  train.csv
  test.csv
  example-submission.csv
  auxiliary/
    sg-hdb-block.csv
    sg-mrt-stations.csv
    sg-schools.csv
    sg-shopping-malls.csv
    sg-coe-prices.csv
```

Training requires `train.csv`; submission generation requires `test.csv` and
`example-submission.csv`. Median and core HGB need no auxiliary files. Block variants
need the HDB block table. Amenity variants also need the MRT, school and mall tables.
The lagged COE variant needs the block and COE tables. Stock data is not used by the
current models. Do not force-add course data to Git.

## Run One Experiment

The same configuration identifies training, evaluation, analysis and submission:

```bash
python train.py --config_path configs/catboost/amenities.py
python eval.py --config_path configs/catboost/amenities.py
python analysis.py --config_path configs/catboost/amenities.py
python submit.py --config_path configs/catboost/amenities.py
```

`train.py` fits all labeled rows and saves a checkpoint. `submit.py` loads that
checkpoint, validates the predictions and sample IDs, and writes a local CSV; use
the same configuration as training. Evaluation and analysis independently fit models
on past records and evaluate later records. `submit.py` does not upload to Kaggle.

| Model | Variants |
| --- | --- |
| median | global_median, town_flat_type |
| hgb | core, block, amenities |
| catboost | block, amenities, lagged_coe |

Outputs are ignored by Git:

| Task | Default output |
| --- | --- |
| Training | `chkpt/<model>/<variant>.joblib` |
| Evaluation | `output/eval/<model>/<variant>.json` |
| Error analysis | `output/analysis/<model>/<variant>.json` |
| Submission | `output/submit/<model>/<variant>.csv` |

## Batch Runs

The original Bash command lists are also available. From the repository root,
with the intended Python environment active, run them using Bash (Git Bash on
Windows), or copy an individual command from a script:

```bash
bash bash/train.bash
bash bash/eval.bash
bash bash/analysis.bash
bash bash/submit.bash
```

These scripts preserve the original eight configuration commands. A failed
command does not stop later commands. Add new configuration commands to the lists
when needed, or use the Python runner below to discover configurations automatically.

The Python runner uses the active interpreter and stops when a task fails. It finds
configuration files automatically; adding a variant does not require editing a task list.

```bash
python scripts/run_all.py --task train
python scripts/run_all.py --task eval --model catboost
python scripts/run_all.py --task analysis --model hgb --variant block
python scripts/run_all.py --task submit
python scripts/run_all.py --task train --dry-run
```

Train the selected variants before generating their submissions. Full batch evaluation
includes six distinct training/validation periods per variant and can take several minutes.

## Baselines

[benchmarks/baseline.json](benchmarks/baseline.json) records the reference data hashes,
code versions, environment and selected model results. On the fixed holdout, training
ends in September 2024 and validation covers October 2024 through March 2025:

| Model | Reference RMSE (SGD) |
| --- | ---: |
| HGB core | 462.41 |
| HGB block | 457.68 |
| HGB amenities | 454.95 |
| CatBoost block | 451.27 |
| CatBoost amenities | 450.46 |
| CatBoost lagged COE | 451.48 |

Historical public Kaggle scores are recorded separately inside that metadata file.
They are different from local holdout scores and final private leaderboard results.
Static amenity tables lack complete historical opening dates, and a six-month
holdout does not cover the entire test horizon through July 2026.

## Repository Structure

```text
configs/                  Feature lists, model parameters and output paths
src/models/               Model interfaces, registry and implementations
src/preprocessing/        Cleaning, auxiliary joins, caching and time splitting
src/evaluation/           Shared metrics and validation windows
scripts/run_all.py        Cross-platform batch runner
bash/                     Copyable command lists for the original configurations
tests/                    Synthetic regression checks
benchmarks/baseline.json   Selected reproducible reference metadata
docs/TECHNICAL_GUIDE.md    Component contracts and extension examples
train.py                  Full-data training
eval.py                   Holdout, rolling and horizon validation
analysis.py               Holdout error analysis
submit.py                 Validated prediction CSV generation
```

See the [technical guide](docs/TECHNICAL_GUIDE.md) for component usage and extension.

## Checks and Collaboration

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic inputs and temporary files, so they need no course dataset.
They cover optional auxiliary dependencies, caching, time boundaries, all eight model
configurations, checkpoint reloads, invalid submissions and batch failure handling.

Before changing shared code, synchronize with `main` and work on a feature branch.
Commit code, the affected configurations and focused tests together. Compare changed
model or feature behavior on the same forward split. Update `benchmarks/baseline.json`
only when deliberately establishing a new reference; keep routine outputs local.
