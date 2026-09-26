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

## Reproduce the analysis

Use Python 3.11 or newer from the repository root. On Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/progress_analysis.py
.\.venv\Scripts\python.exe src/rolling_validation.py
.\.venv\Scripts\python.exe src/amenity_experiment.py
.\.venv\Scripts\python.exe src/catboost_experiment.py
.\.venv\Scripts\python.exe src/horizon_validation.py
.\.venv\Scripts\python.exe src/coe_experiment.py
.\.venv\Scripts\python.exe src/catboost_error_analysis.py
.\.venv\Scripts\python.exe src/make_submission.py
.\.venv\Scripts\python.exe src/make_catboost_submission.py
.\.venv\Scripts\python.exe src/make_catboost_block_submission.py
```

The scripts read `data/train.csv`, `data/test.csv` and the relevant tables in `data/auxiliary/`. They write metrics and supporting tables to `analysis/` and candidate Kaggle files to `submissions/`. They never read hidden test labels. The main model fitting set ends in September 2024; the following six months form its validation set. The model seed is 42. The three submission-building scripts fit on all labeled rows and create files with `Id,Predicted`; running them does not upload a file. Each generated candidate has a SHA-256 hash in its metadata. The observed Kaggle results are recorded separately in `submissions/block_hgb_initial.json`, `submissions/catboost_block_result.json` and `submissions/catboost_amenities_result.json`.

## Repository contents

| Path | Purpose |
| --- | --- |
| `data/` | Course-provided competition data, kept in this private repository |
| `src/progress_analysis.py` | Data checks, feature preparation and preliminary validation |
| `src/amenity_experiment.py` | Exploratory amenity distances and matched forward validation |
| `src/catboost_experiment.py` | CatBoost comparison and four forward windows |
| `src/horizon_validation.py` | Six-, twelve- and eighteen-month forward holds |
| `src/coe_experiment.py` | Previous-month COE feature comparison |
| `src/catboost_error_analysis.py` | Error audit by rent band, flat type, town and approval month |
| `src/make_catboost_block_submission.py` | Build the candidate without amenity tables |
| `analysis/` | Generated metrics and tables used in the progress report |
| `submissions/` | Kaggle prediction files, hashes and public results |
| `cs5228-report-templates.zip` | Course-provided final report templates |
| `reports/progress/` | Editable progress-report slides |

## Collaboration

Before editing, run `git pull --rebase origin main`. Commit your changes, then run `git push origin main`. If two people need to edit the same files, use separate branches and a pull request.

The next experiments will audit amenity opening dates where possible, review stock indicators using only earlier prices, and compare further regression methods. Each experiment should record its split, features, RMSE and code version.
