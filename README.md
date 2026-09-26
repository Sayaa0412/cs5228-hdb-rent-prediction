# CS5228 HDB rent prediction

Team 29: HAN MINGJIA, LIU HENGYAN, LIU RUIXI and LIU SIFEI.

This project predicts monthly HDB rent in Singapore from flat details and approval dates. The [CS5228 Kaggle competition](https://www.kaggle.com/competitions/cs-5228-2610-project/overview) scores submissions by **root mean squared error (RMSE)**. A submission must have the columns `Id` and `Predicted`. The competition page currently lists **15 November 2026, 23:59 Singapore time** as its closing time. The final Canvas report deadline is a separate date and has not yet been stated on the [Project page](https://canvas.nus.edu.sg/courses/93866/pages/project).

## Current status

The [editable progress report](reports/progress/CS5228_Team29_Progress_Report_v3.pptx) contains 12 slides. It includes exploratory findings, a forward validation design, four preliminary models, error analysis and the next steps. The results are local validation scores, not Kaggle leaderboard scores. The original course data and report templates are in `data/` and `cs5228-report-templates.zip`.

The best preliminary model is a histogram gradient boosting regressor with HDB block features. On the October 2024 to March 2025 holdout, it reaches **RMSE SGD 457.68** and **MAE SGD 334.55**. Four consecutive six-month checks produced RMSE values of **621.66, 548.01, 499.04 and 457.68**, so the recent score does not describe every market period. These checks also do not establish performance through July 2026, the end of the unlabeled test period.

## Reproduce the analysis

Use Python 3.11 or newer from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python src/progress_analysis.py
python src/rolling_validation.py
python src/make_submission.py
```

The scripts read `data/train.csv`, `data/test.csv` and `data/auxiliary/sg-hdb-block.csv`. They write metrics and supporting tables to `analysis/` and a candidate Kaggle file to `submissions/`. They never read hidden test labels. The main progress-report model fitting set ends in September 2024; the following six months form its validation set. The model seed is 42. `make_submission.py` fits on all labeled rows and creates a candidate with `Id,Predicted`; running it does not upload the file.

## Repository contents

| Path | Purpose |
| --- | --- |
| `data/` | Course-provided competition data, kept in this private repository |
| `src/progress_analysis.py` | Data checks, feature preparation and preliminary validation |
| `analysis/` | Generated metrics and tables used in the progress report |
| `submissions/` | Candidate Kaggle submission and its metadata |
| `reports/progress/` | Editable PowerPoint progress report |
| `cs5228-report-templates.zip` | Course-provided final report templates |
| `CS5228_Project_Requirements.md` | Requirements summarized from Canvas |

## Collaboration

The repository is private because the competition data and project code are for Team 29. Team members should also join the official Kaggle team before sharing competition material. To avoid overwriting another member's work, start each session with `git pull --rebase origin main`, make a focused commit, then run `git push origin main`. If Git reports a conflict, resolve it before pushing. For larger or overlapping changes, use a separate branch and a pull request.

The next experiments will test additional forward validation windows, distance to amenities, date-safe economic indicators and further regression methods. Each experiment should record the split, features, RMSE and code version so that the final report can trace its results.
