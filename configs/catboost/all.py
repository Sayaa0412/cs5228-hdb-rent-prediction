"""CatBoost with block, amenity and lagged COE features."""

model_name = "catboost"
auxiliary_config = {
    "blocks": [
        "LATITUDE",
        "LONGITUDE",
        "MAX_FLOOR",
        "BUILDING_AGE",
        "REGION",
    ],
    "coe": ["PREV_MONTH_COE_MEDIAN_SGD"],
    "amenities": [
        "NEAREST_MRT_KM",
        "MRT_WITHIN_1KM",
        "NEAREST_PRIMARY_SCHOOL_KM",
        "PRIMARY_SCHOOLS_WITHIN_1KM",
        "NEAREST_MALL_KM",
        "MALLS_WITHIN_1KM",
    ],
}

model_config = {
    "variant": "all",
    "features": {
        "numeric": [
            "MONTH_INDEX",
            "FLOOR_AREA_SQM",
            "LEASE_AGE",
            "LATITUDE",
            "LONGITUDE",
            "MAX_FLOOR",
            "BUILDING_AGE",
            "NEAREST_MRT_KM",
            "MRT_WITHIN_1KM",
            "NEAREST_PRIMARY_SCHOOL_KM",
            "PRIMARY_SCHOOLS_WITHIN_1KM",
            "NEAREST_MALL_KM",
            "MALLS_WITHIN_1KM",
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

save_path = "chkpt/catboost/all.joblib"
submit_path = "output/submit/catboost/all.csv"
eval_path = "output/eval/catboost/all.json"
analysis_path = "output/analysis/catboost/all.json"
