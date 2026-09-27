"""Town and flat-type median baseline configuration."""

model_name = "median"
auxiliary_config = None

model_config = {
    "variant": "town_flat_type",
    "features": {
        "numeric": [],
        "categorical": ["TOWN", "FLAT_TYPE"],
    },
}

training_config = {}

save_path = "chkpt/median/town_flat_type.joblib"
submit_path = "output/submit/median/town_flat_type.csv"
eval_path = "output/eval/median/town_flat_type.json"
analysis_path = "output/analysis/median/town_flat_type.json"
