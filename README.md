# CS5228 HDB rent prediction

Team 29: HAN MINGJIA, LIU HENGYAN, LIU RUIXI and LIU SIFEI.

This project predicts monthly HDB rent in Singapore from flat details and approval dates. The [CS5228 Kaggle competition](https://www.kaggle.com/competitions/cs-5228-2610-project/overview) scores submissions by **root mean squared error (RMSE)**. A submission must have the columns `Id` and `Predicted`. The competition page currently lists **15 November 2026, 23:59 Singapore time** as its closing time. The final Canvas report deadline is a separate date and has not yet been stated on the [Project page](https://canvas.nus.edu.sg/courses/93866/pages/project).

## Current status

The 12-slide progress report draft is under team review and is not included in this repository yet. It uses a purple, pink and blue palette. It includes exploratory findings, a forward validation design, six preliminary models, error analysis, the first Kaggle public score and the next steps. The original course data and report templates are in `data/` and `cs5228-report-templates.zip`.

The first histogram gradient boosting (HGB) model with HDB block features reaches **RMSE SGD 457.68** and **MAE SGD 334.55** on the October 2024 to March 2025 holdout. Its [Kaggle submission](https://www.kaggle.com/competitions/cs-5228-2610-project/submissions), `block_hgb_initial.csv`, was marked Complete on 26 September 2026 with a **public RMSE of SGD 454.42263**. Kaggle states that its public leaderboard uses approximately 50% of the test data; the final private leaderboard uses the other 50%. This is not the same evaluation as the local holdout. Four consecutive six-month HGB checks produced RMSE values of **621.66, 548.01, 499.04 and 457.68**, so the recent score does not describe every market period.

An exploratory amenity feature experiment calculates nearest distances and one-kilometre counts for MRT stations marked open, primary schools and shopping malls. With the same model settings and latest forward holdout, RMSE moves from **457.68 to 454.95**. It improves all four rolling windows by **0.86 to 2.73 SGD**. These are local validation results; the expanded model has not been submitted to Kaggle. The auxiliary tables do not provide complete historical opening dates, so this feature set needs a temporal availability audit before being used as evidence of historical access to amenities.

A CatBoost comparison using address categories reaches **RMSE SGD 451.27** with block features and **450.46** with amenity features on the latest holdout. The amenity version improves on HGB in all four six-month windows. Its 50,000-row file, `catboost_amenities_candidate.csv`, is ready locally but **has not been submitted to Kaggle**. The six-month holdout also cannot establish performance through July 2026, the end of the unlabeled test period.

CatBoost's overall MAE is **SGD 326.27**, but it is much less reliable for the 925 validation records with actual rent at or below SGD 2,000: their MAE is **SGD 1,037.35**, with average predictions **SGD 1,033.19 above** the actual rent. This band is defined using the observed target; it identifies cases to audit and does not explain why their rents are low. Executive flats also have higher MAE (**SGD 384.25**) than the overall figure.

## Reproduce the analysis

Use Python 3.11 or newer from the repository root. On Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/progress_analysis.py
.\.venv\Scripts\python.exe src/rolling_validation.py
.\.venv\Scripts\python.exe src/amenity_experiment.py
.\.venv\Scripts\python.exe src/catboost_experiment.py
.\.venv\Scripts\python.exe src/catboost_error_analysis.py
.\.venv\Scripts\python.exe src/make_submission.py
.\.venv\Scripts\python.exe src/make_catboost_submission.py
```

The scripts read `data/train.csv`, `data/test.csv` and the relevant tables in `data/auxiliary/`. They write metrics and supporting tables to `analysis/` and candidate Kaggle files to `submissions/`. They never read hidden test labels. The main model fitting set ends in September 2024; the following six months form its validation set. The model seed is 42. The two `make_*submission.py` scripts fit on all labeled rows and create files with `Id,Predicted`; running them does not upload either file. Each generated candidate has a SHA-256 hash in its metadata. The scored HGB file's observed Kaggle result is recorded separately in `submissions/block_hgb_initial.json`.

## Repository contents

| Path | Purpose |
| --- | --- |
| `data/` | Course-provided competition data, kept in this private repository |
| `src/progress_analysis.py` | Data checks, feature preparation and preliminary validation |
| `src/amenity_experiment.py` | Exploratory amenity distances and matched forward validation |
| `src/catboost_experiment.py` | CatBoost comparison and four forward windows |
| `src/catboost_error_analysis.py` | Error audit by rent band, flat type, town and approval month |
| `analysis/` | Generated metrics and tables used in the progress report |
| `submissions/` | Candidate Kaggle submission and its metadata |
| `cs5228-report-templates.zip` | Course-provided final report templates |
| `CS5228_Project_Requirements.md` | Requirements summarized from Canvas |

## Collaboration

The repository is private because the competition data and project code are for Team 29. Team members should also join the official Kaggle team before sharing competition material. To avoid overwriting another member's work, start each session with `git pull --rebase origin main`, make a focused commit, then run `git push origin main`. If Git reports a conflict, resolve it before pushing. For larger or overlapping changes, use a separate branch and a pull request.

The next experiments will audit amenity opening dates where available, test date-safe economic indicators, check longer forecast horizons and compare further regression methods. Each experiment should record the split, features, RMSE and code version so that the final report can trace its results.
