"""HDB block and amenity-feature CatBoost configuration."""

model_name = "catboost"
auxiliary_config = None

model_config = {
    "variant": "amenities",
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

save_path = "chkpt/catboost/amenities.joblib"
submit_path = "output/submit/catboost/amenities.csv"
eval_path = "output/eval/catboost/amenities.json"
analysis_path = "output/analysis/catboost/amenities.json"
