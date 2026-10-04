"""HDB block-feature MLP configuration."""

model_name = "mlp"
auxiliary_config = {
    "blocks": [
        "LATITUDE",
        "LONGITUDE",
        "MAX_FLOOR",
        "BUILDING_AGE",
        "REGION",
    ],
}

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
    "embedding_dim": 16,
    "hidden_dims": [512, 256, 128],
    "dropout": 0.1,
}

training_config = {
    "epochs": 60,
    "validation_months": 3,
    "eval_every": 10,
    "batch_size": 4096,
    "lr": 0.0005,
    "weight_decay": 0.0001,
    "random_seed": 42,
    "thread_count": 1,
    "device": "auto",
    "verbose": True,
}

save_path = "chkpt/mlp/block.joblib"
submit_path = "output/submit/mlp/block.csv"
eval_path = "output/eval/mlp/block.json"
analysis_path = "output/analysis/mlp/block.json"
