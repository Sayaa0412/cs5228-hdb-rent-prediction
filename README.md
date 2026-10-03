# CS5228 HDB rent prediction

Team 29: HAN MINGJIA, LIU HENGYAN, LIU RUIXI and LIU SIFEI.

This project predicts monthly HDB rent in Singapore from flat details and approval dates. The [CS5228 Kaggle competition](https://www.kaggle.com/competitions/cs-5228-2610-project/overview) scores submissions by **root mean squared error (RMSE)**. A submission must have the columns `Id` and `Predicted`. The competition page currently lists **15 November 2026, 23:59 Singapore time** as its closing time. The final Canvas report deadline is separate and has not yet been stated on the [Project page](https://canvas.nus.edu.sg/courses/93866/pages/project).

## Course deliverables

- **Progress report:** roughly 10–15 slides, submitted as a PDF on Canvas. The last one or two slides should describe the plan for the rest of the project. One team member uploads it.
- **Final report:** a scientific paper with at most eight pages of main text, excluding references and appendices. It must name every team member, give their student IDs, and include a short contribution record in the appendix.
- **Final submission:** a ZIP containing the report PDF and source code, or the report PDF with a GitHub link for the code, as specified on Canvas.
- **Assessment:** methodology 60%, report quality 30%, and reproducibility 10%. The project is assessed on more than its Kaggle score.

## Current status

The [12-slide progress report](reports/progress/CS5228_Team29_Progress_Report_v8.pptx) covers the data, forward validation, preliminary models, error analysis, both Kaggle public scores and next steps. The course data and report templates are in `data/` and `cs5228-report-templates.zip`.

The first histogram gradient boosting (HGB) model with HDB block features reaches **RMSE SGD 457.68** and **MAE SGD 334.55** on the October 2024 to March 2025 holdout. Its [Kaggle submission](https://www.kaggle.com/competitions/cs-5228-2610-project/submissions), `block_hgb_initial.csv`, was marked Complete on 26 September 2026 with a **public RMSE of SGD 454.42263**. Kaggle states that its public leaderboard uses approximately 50% of the test data; the final private leaderboard uses the other 50%. This is not the same evaluation as the local holdout. Four consecutive six-month HGB checks produced RMSE values of **621.66, 548.01, 499.04 and 457.68**, so the recent score does not describe every market period.

An exploratory amenity feature experiment calculates nearest distances and one-kilometre counts for MRT stations marked open, primary schools and shopping malls. With the same model settings and latest forward holdout, RMSE moves from **457.68 to 454.95**. It improves all four rolling windows by **0.86 to 2.73 SGD**. These are local validation results; the expanded model has not been submitted to Kaggle. The auxiliary tables do not provide complete historical opening dates, so this feature set needs a temporal availability audit before being used as evidence of historical access to amenities.

A CatBoost comparison using address categories reaches **RMSE SGD 451.27** with block features and **450.46** with amenity features on the latest holdout. The amenity version improves on HGB in all four six-month windows. Both 50,000-row files are in `submissions/`: `catboost_block_candidate.csv` uses no amenity tables, while `catboost_amenities_candidate.csv` has the slightly lower local error. Both were submitted on 26 September 2026. Their Kaggle public RMSE scores are **445.55651** without amenities and **444.80298** with amenities, an improvement of **0.75353** for the latter. The initial HGB submission scored **454.42263**. Public leaderboard scores use a separate test subset and are not local holdout or final private leaderboard results. The amenity tables still need a historical opening-date audit, and the six-month holdout cannot establish performance through July 2026, the end of the unlabeled test period.

With fixed settings and longer validation periods ending in March 2025, CatBoost with amenities has RMSE **474.12** over 12 months and **500.37** over 18 months. The corresponding HGB scores are **481.63** and **508.58**. These checks show why the latest six-month score should not stand in for a longer forecast.

A previous-month COE premium was tested as a date-aligned economic feature. It raised CatBoost block RMSE by **0.21**, **0.59** and **11.06** SGD on the six-, twelve- and eighteen-month windows. We left it out of the candidate model.

CatBoost's overall MAE is **SGD 326.27**, but it is much less reliable for the 925 validation records with actual rent at or below SGD 2,000: their MAE is **SGD 1,037.35**, with average predictions **SGD 1,033.19 above** the actual rent. This band is defined using the observed target; it identifies cases to audit and does not explain why their rents are low. Executive flats also have higher MAE (**SGD 384.25**) than the overall figure.

## Environment setup

Run all commands from the repository root. Python 3.11 or newer is recommended. The project can use either Conda or a standard Python virtual environment; do not install the dependencies into the system Python environment.

### Conda

The Conda commands are the same on Windows, Linux and macOS:

```bash
conda create --name cs5228_project python=3.11 pip -y
conda activate cs5228_project
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Python venv on Windows

Create and activate the environment in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, the environment interpreter can be used directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Python venv on Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running experiments

Before running the entry points, place the course data under `data/`:

```text
data/
├── train.csv
├── test.csv
├── example-submission.csv
└── auxiliary/
    ├── sg-coe-prices.csv
    ├── sg-hdb-block.csv
    ├── sg-mrt-stations.csv
    ├── sg-schools.csv
    ├── sg-shopping-malls.csv
    └── sg-stock-prices.csv
```

Each Bash script runs all eight registered model variants. `eval.bash` and `analysis.bash` retrain temporary models on time-based holdouts, so the CatBoost runs take longer than training or submission generation.

| Task | Bash script | Input | Output |
| --- | --- | --- | --- |
| Train full-data models | `bash/train.bash` | `data/train.csv` | `chkpt/<model>/<variant>.joblib` |
| Run holdout, rolling and horizon validation | `bash/eval.bash` | `data/train.csv` | `output/eval/<model>/<variant>.json` |
| Run holdout error analysis | `bash/analysis.bash` | `data/train.csv` | `output/analysis/<model>/<variant>.json` |
| Generate Kaggle candidates | `bash/submit.bash` | Checkpoints and `data/test.csv` | `output/submit/<model>/<variant>.csv` |

`submit.bash` only generates local CSV files. It does not upload them to Kaggle. Run `train.bash` before `submit.bash` so every configured checkpoint exists. Evaluation and analysis do not use the full-data checkpoints and may be run independently.

### Windows

Activate the Conda or venv environment in PowerShell, then run the scripts with Git Bash:

```powershell
conda activate cs5228_project

bash bash/train.bash
bash bash/eval.bash
bash bash/analysis.bash
bash bash/submit.bash
```

On Windows, `bash` should resolve to Git Bash so it can use the active Windows Python environment. If Git Bash is not on `PATH`, invoke its executable directly; the installation path may differ between machines:

```powershell
& "C:\Program Files\Git\bin\bash.exe" bash/train.bash
```

For a venv, replace the Conda activation command with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux or macOS

Activate the selected environment, then execute the same task scripts:

```bash
conda activate cs5228_project

bash bash/train.bash
bash bash/eval.bash
bash bash/analysis.bash
bash bash/submit.bash
```

For a venv, replace the Conda activation command with:

```bash
source .venv/bin/activate
```

### Running one model variant

The root entry points accept one configuration file path. The configuration supplies the registered model name, variant, feature columns, hyperparameters and output paths, so the same file identifies the complete experiment:

```bash
python train.py --config_path configs/hgb/block.py
python eval.py --config_path configs/hgb/block.py
python analysis.py --config_path configs/hgb/block.py
python submit.py --config_path configs/hgb/block.py
```

The available combinations are:

| Model | Variants |
| --- | --- |
| `median` | `global_median`, `town_flat_type` |
| `hgb` | `core`, `block`, `amenities` |
| `catboost` | `block`, `amenities`, `lagged_coe` |
| `mlp` | `block` |

### MLP baseline

MLP is not included in the existing batch scripts; run it separately with the commands below.

The PyTorch MLP uses the same seven numeric and six categorical features as CatBoost block, including the same joined `ADDRESS`. Category embeddings are concatenated with mean-imputed, standardized numeric features, followed by configurable ReLU layers, dropout and a scalar head. Numeric statistics, category dictionaries and target normalization are fitted only on training rows; missing/unseen categories use a fixed zero embedding. Predictions are converted back to SGD.

Network dimensions, maximum selection epochs and AdamW settings are configured in `configs/mlp/block.py`; `device` accepts `cpu`, `auto` or an explicit PyTorch device. Each `train(X, y)` reserves the latest `validation_months=3` calendar months within its training input for inner validation, fitting all normalization and category maps only on earlier rows. Insufficient history raises an error; there is no random-split fallback. Validation runs at epoch 1, every `eval_every=10` epochs, and the final epoch. The minimum inner-validation RMSE selects the epoch (earliest on ties), with no early stopping.

The final model is initialized afresh, fits normalization/maps on the full supplied training prefix, and trains for exactly the selected epoch count. `predict()` and the main joblib checkpoint use this full-prefix refit, not the inner-selection model. External evaluation windows and the public train/predict interface are unchanged; external labels are not used for epoch selection. Training costs `epochs + best_epoch` epochs per fit; the eight external evaluation fits each now have selection and refit phases. Checkpoints store CPU tensors. These changes do not make previously inspected development windows independently untouched test sets.

MLP outputs are grouped in `block_YYMMDD_HHMM/` run directories, using local time when the configuration loads. Separate commands get separate timestamps; runs started in the same minute can overwrite the same output. For submission, explicitly provide the checkpoint printed by the training command (the path below is an example):

Each MLP run also saves `config.yaml` beside its result or
checkpoint, including feature lists, hyperparameters and resolved output paths.
Submission snapshots use the loaded checkpoint's model/training settings and
actual checkpoint path. Snapshots are saved before training or prediction, so a
YAML file alone does not indicate successful completion. Existing runs are not
backfilled because their exact configurations may no longer be available.

```bash
python train.py --config_path configs/mlp/block.py
python eval.py --config_path configs/mlp/block.py
python analysis.py --config_path configs/mlp/block.py
python submit.py --config_path configs/mlp/block.py --checkpoint_path chkpt/mlp/block_261003_1545/model.joblib
```

With `verbose=True`, MLP training displays selection/refit tqdm progress bars rather
than printing each epoch's loss. Loss history is recorded even with `verbose=False`.
Each fit exports a full-prefix refit CSV (`epoch,train_mse_standardized`) and PNG curve:

- `train.py`: `chkpt/mlp/block_<time>/` contains `model.joblib`, `config.yaml`
  and `loss/train.csv` / `loss/train.png`.
- `eval.py`: `output/eval/mlp/block_<time>/` contains `result.json`, `config.yaml`
  and `loss/` with separate `holdout`, `rolling_<start>_<end>` and
  `horizon_<months>m` CSV/PNG pairs.
- `analysis.py`: `output/analysis/mlp/block_<time>/` contains `result.json`,
  `config.yaml` and `loss/holdout.csv` / `loss/holdout.png`.
- `submit.py`: `output/submit/mlp/block_<time>/` contains `submission.csv`
  and `config.yaml`; prediction does not generate a training-loss curve.

Each loss stem also produces `_selection.csv` (every epoch's batch MSE plus train/validation SGD RMSE at monitoring epochs), `_validation.png` (both SGD RMSE curves with the best epoch marked), `_summary.json` (inner date ranges/row counts, chosen epoch/RMSE and actual selection/refit epochs), and `_best_selection.joblib`. The latter is a standalone inner-fit model with the exact best weights, normalization, category maps and configuration, distinct from the final full-prefix model. The main checkpoint also retains selected-epoch metadata. The configuration YAML records requested settings; the summary records actual epoch counts.

Evaluation and analysis JSON include all artifact paths in `loss_files`. Batch-MSE curves show the
sample-weighted average of training-batch MSE on standardized targets, with dropout
active and weights changing during each epoch. They are not validation loss or
the final model's SGD RMSE. The separate validation plot uses deterministic, dropout-disabled inference on both inner subsets in the same SGD units to reveal overfitting; it is not an external-evaluation curve. Existing results cannot recover past epoch losses;
new training is needed. Models without recorded histories produce no loss files.

## Repository structure

```text
.
├── analysis.py              # Holdout error analysis entry point
├── eval.py                  # Holdout, rolling and horizon validation
├── submit.py                # Kaggle CSV generation from checkpoints
├── train.py                 # Full labeled-data training
├── bash/                    # Batch entry points for all variants
├── configs/                 # Model, training and output-path configuration
│   ├── median/
│   ├── hgb/
│   ├── catboost/
│   └── mlp/
├── src/                     # Importable project library
│   ├── models/              # Abstract interface, registry and model classes
│   └── preprocessing/       # Cleaning, feature joins and time splitting
├── data/                    # Local course data; ignored by Git
├── chkpt/                   # Generated model checkpoints; ignored by Git
├── output/                  # Evaluation, analysis and submission files; ignored by Git
├── requirements.txt         # Python dependencies
└── TECHNICAL_GUIDE.md       # Detailed src component and extension guide
```

The shared `preprocess()` function converts raw train or test records into the same clean feature table. Each model class then selects and formats the columns required by its variant through its own `_preprocess()` implementation. New model modules register themselves through `src.models`, so the root entry points do not import concrete model classes individually.

Each file in `configs/<model>/<variant>.py` defines:

```text
model_name
auxiliary_config
model_config
training_config
save_path
submit_path
eval_path
analysis_path
```

`auxiliary_config` is currently reserved and must remain `None`. The root entry points pass it to `preprocess()`; non-`None` auxiliary configurations will be implemented in a later extension.

See `TECHNICAL_GUIDE.md` for the preprocessing contract, model interface, registry usage and shape requirements.

## Collaboration

Synchronize the branch before editing:

```bash
git pull --rebase origin main
```

Use a separate branch when multiple contributors are changing related files:

```bash
git switch -c feature/<short-description>
```

Keep changes scoped and commit the implementation together with its configuration and documentation updates. A new model or variant should include its model implementation, configuration file and entries in the relevant Bash scripts.

Before opening a pull request:

1. Run the affected model variant through the relevant root entry point.
2. Confirm generated paths match the corresponding config file.
3. Compare evaluation metrics with the previous implementation when migrating existing behavior.
4. Check `git status` and exclude local data, checkpoints, outputs, environments and IDE settings.
5. Rebase on the latest `main`, resolve conflicts locally and push the feature branch.

The repository ignores `data/`, `chkpt/`, `output/`, virtual environments and editor-local settings. Do not force-add these generated or machine-specific files. Use a pull request when changes overlap with another contributor's work or affect shared model and preprocessing interfaces.
