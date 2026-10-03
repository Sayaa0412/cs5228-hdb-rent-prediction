"""HDB block-feature HGB configuration."""

model_name = "hgb"
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
        "categorical": ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "REGION"],
    },
    "categorical_features": "from_dtype",
    "max_leaf_nodes": 31,
}

training_config = {
    "learning_rate": 0.07,
    "max_iter": 220,
    "min_samples_leaf": 30,
    "l2_regularization": 0.1,
    "early_stopping": False,
    "random_state": 42,
}

save_path = "chkpt/hgb/block.joblib"
submit_path = "output/submit/hgb/block.csv"
eval_path = "output/eval/hgb/block.json"
analysis_path = "output/analysis/hgb/block.json"
