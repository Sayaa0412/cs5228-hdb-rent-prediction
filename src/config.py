"""Save the resolved MLP experiment configuration beside its output."""

from pathlib import Path


def save_config(config: dict, output_path: Path) -> None:
    if config["model_name"] != "mlp":
        return
    import yaml

    keys = (
        "model_name", "auxiliary_config", "model_config", "training_config",
        "run_time", "save_path", "submit_path", "eval_path", "analysis_path",
    )
    snapshot = {key: config[key] for key in keys if key in config}
    path = output_path.parent / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )
