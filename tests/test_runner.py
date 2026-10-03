import contextlib
import io
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts import run_all


class RunnerTests(unittest.TestCase):
    def test_filtering_discovers_requested_configs(self):
        paths = run_all.discover_configs(run_all.ROOT, model="catboost", variant="amenities")
        self.assertEqual([p.relative_to(run_all.ROOT).as_posix() for p in paths], ["configs/catboost/amenities.py"])

    def test_first_failure_stops_batch_and_returns_failure(self):
        with (
            patch("sys.argv", ["run_all.py", "--task", "train"]),
            patch.object(
                run_all.subprocess, "run",
                side_effect=subprocess.CalledProcessError(7, "python"),
            ) as execute,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(run_all.main(), 7)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(execute.call_args.args[0][0], sys.executable)

    def test_dry_run_does_not_launch_training(self):
        with (
            patch("sys.argv", ["run_all.py", "--task", "train", "--dry-run"]),
            patch.object(run_all.subprocess, "run") as execute,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_all.main(), 0)
        execute.assert_not_called()
