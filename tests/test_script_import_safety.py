from __future__ import annotations

import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ScriptImportSafetyTest(unittest.TestCase):
    def test_benchmark_scripts_help_exits_before_running_workloads(self) -> None:
        modules = (
            "scripts.run_ablation",
            "scripts.run_benchmark",
            "scripts.run_benchmark_fleet",
        )
        for module_name in modules:
            module = importlib.import_module(module_name)
            stdout = io.StringIO()
            with self.subTest(module=module_name):
                with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                    module.main(["--help"])
                self.assertEqual(raised.exception.code, 0)
                self.assertIn("usage:", stdout.getvalue())

    def test_summary_helpers_do_not_execute_on_import(self) -> None:
        modules = (
            "scripts.check_run_success",
            "scripts.compute_ablation_hv",
            "scripts.compute_ablation_hv_med",
            "scripts.compute_ablation_hv_surgical",
            "scripts.compute_rebenchmark_feasibility",
            "scripts.compute_rebenchmark_hv",
            "scripts.quick_ablation_summary",
        )
        for module_name in modules:
            sys.modules.pop(module_name, None)
            stdout = io.StringIO()
            with self.subTest(module=module_name):
                with contextlib.redirect_stdout(stdout):
                    importlib.import_module(module_name)
                self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
