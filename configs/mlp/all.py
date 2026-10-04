"""MLP with block, amenity and lagged COE features."""

model_name = "mlp"
auxiliary_config = {
    "blocks": [
        "LATITUDE",
        "LONGITUDE",
        "MAX_FLOOR",
        "BUILDING_AGE",
        "REGION",
    ],
    "amenities": [
        "NEAREST_MRT_KM",
        "MRT_WITHIN_1KM",
        "NEAREST_PRIMARY_SCHOOL_KM",
        "PRIMARY_SCHOOLS_WITHIN_1KM",
        "NEAREST_MALL_KM",
        "MALLS_WITHIN_1KM",
    ],
    "coe": ["PREV_MONTH_COE_MEDIAN_SGD"],
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
    "embedding_dim": 32,
    "hidden_dims": [512, 256, 128],
    "dropout": 0.1,
}

training_config = {
    "epochs": 60,
    "validation_months": 3,
    "eval_every": 5,
    "batch_size": 4096,
    "lr": 0.0003,
    "weight_decay": 0.001,
    "random_seed": 42,
    "thread_count": 1,
    "device": "auto",
    "verbose": True,
}

save_path = "chkpt/mlp/all.joblib"
submit_path = "output/submit/mlp/all.csv"
eval_path = "output/eval/mlp/all.json"
analysis_path = "output/analysis/mlp/all.json"
