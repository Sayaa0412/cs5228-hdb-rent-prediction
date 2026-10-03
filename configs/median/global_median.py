"""Global median baseline configuration."""

model_name = "median"
auxiliary_config = {}

model_config = {
    "variant": "global_median",
    "features": {
        "numeric": [],
        "categorical": [],
    },
}

training_config = {}

save_path = "chkpt/median/global_median.joblib"
submit_path = "output/submit/median/global_median.csv"
eval_path = "output/eval/median/global_median.json"
analysis_path = "output/analysis/median/global_median.json"
