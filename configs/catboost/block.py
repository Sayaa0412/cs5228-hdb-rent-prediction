"""HDB block-feature CatBoost configuration."""

model_name = "catboost"
auxiliary_config = {"blocks": True}

model_config = {
    "variant": "block",
    "features": {
        "numeric": [
            "MONTH_INDEX",
            "FLOOR_AREA_SQM",
            "LEASE_AGE",
            "LATITUDE",
            "LONGITUDE",
            "MAX_FLOOR",
            "BUILDING_AGE",
        ],
        "categorical": [
            "TOWN",
            "FLAT_TYPE",
            "FLAT_MODEL",
            "REGION",
            "STREET",
            "ADDRESS",
        ],
    },
    "loss_function": "RMSE",
    "depth": 7,
}

training_config = {
    "iterations": 650,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "random_seed": 42,
    "thread_count": 6,
    "verbose": False,
    "allow_writing_files": False,
}

save_path = "chkpt/catboost/block.joblib"
submit_path = "output/submit/catboost/block.csv"
eval_path = "output/eval/catboost/block.json"
analysis_path = "output/analysis/catboost/block.json"
