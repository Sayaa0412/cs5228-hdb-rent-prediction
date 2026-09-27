"""HDB block and lagged-COE CatBoost configuration."""

model_name = "catboost"
auxiliary_config = None

model_config = {
    "variant": "lagged_coe",
    "features": {
        "numeric": [
            "MONTH_INDEX",
            "FLOOR_AREA_SQM",
            "LEASE_AGE",
            "LATITUDE",
            "LONGITUDE",
            "MAX_FLOOR",
            "BUILDING_AGE",
            "PREV_MONTH_COE_MEDIAN_SGD",
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

save_path = "chkpt/catboost/lagged_coe.joblib"
submit_path = "output/submit/catboost/lagged_coe.csv"
eval_path = "output/eval/catboost/lagged_coe.json"
analysis_path = "output/analysis/catboost/lagged_coe.json"
