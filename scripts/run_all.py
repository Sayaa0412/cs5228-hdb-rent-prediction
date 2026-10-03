"""Run one task for every matching experiment with the active Python."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def discover_configs(root: Path, *, model: str | None = None, variant: str | None = None) -> list[Path]:
    return sorted(
        path for path in (root / "configs").glob("*/*.py")
        if not path.name.startswith("_")
        and (model is None or path.parent.name == model)
        and (variant is None or path.stem == variant)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("train", "eval", "analysis", "submit"))
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = discover_configs(ROOT, model=args.model, variant=args.variant)
    if not configs:
        parser.error("No configurations match the requested model and variant")

    for config in configs:
        relative = config.relative_to(ROOT).as_posix()
        command = [sys.executable, str(ROOT / f"{args.task}.py"), "--config_path", relative]
        print(f"{args.task}: {relative}", flush=True)
        if not args.dry_run:
            try:
                subprocess.run(command, cwd=ROOT, check=True)
            except subprocess.CalledProcessError as error:
                print(f"Failed: {relative} (exit {error.returncode})", file=sys.stderr)
                return error.returncode if error.returncode > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
