"""HDB block-feature MLP configuration."""

from datetime import datetime

model_name = "mlp"
auxiliary_config = None

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
    "embedding_dim": 4,
    "hidden_dims": [256, 128, 64],
    "dropout": 0.1,
}

training_config = {
    "epochs": 500,
    "validation_months": 3,
    "eval_every": 10,
    "batch_size": 512,
    "lr": 0.00003,
    "weight_decay": 0.0001,
    "random_seed": 42,
    "thread_count": 6,
    "device": "auto",
    "verbose": True,
}

run_time = datetime.now().strftime("%y%m%d_%H%M")
save_path = f"chkpt/mlp/block_{run_time}/model.joblib"
submit_path = f"output/submit/mlp/block_{run_time}/submission.csv"
eval_path = f"output/eval/mlp/block_{run_time}/result.json"
analysis_path = f"output/analysis/mlp/block_{run_time}/result.json"
