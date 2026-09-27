"""HDB block and amenity-feature HGB configuration."""

model_name = "hgb"
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

save_path = "chkpt/hgb/amenities.joblib"
submit_path = "output/submit/hgb/amenities.csv"
eval_path = "output/eval/hgb/amenities.json"
analysis_path = "output/analysis/hgb/amenities.json"
