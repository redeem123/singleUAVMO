from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_support import install_numba_stub

install_numba_stub()

from uav_benchmark import cli


class SACCliDispatchTest(unittest.TestCase):
    def test_list_algorithms_shows_named_profiles(self) -> None:
        argv = [
            "uav_benchmark.cli",
            "list-algorithms",
        ]
        with mock.patch("builtins.print") as mocked_print, mock.patch.object(sys, "argv", argv):
            cli.main()
        rendered = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
        self.assertIn("Named Profiles", rendered)
        self.assertIn("benchmark-core", rendered)
        self.assertIn("state-representation", rendered)

    def test_sac_pretrain_dispatches_to_script_entrypoint(self) -> None:
        argv = [
            "uav_benchmark.cli",
            "sac-pretrain",
            "--stage",
            "paper_stage1",
            "--state-representation",
            "TRFTS-HAND",
            "--checkpoint",
            "results/custom/controller.pt",
            "--results-dir",
            "results/custom",
            "--reset",
        ]
        with mock.patch.object(cli, "_maybe_reexec_for_torch"):
            with mock.patch("scripts.train_sac_smopso_controller.main") as mocked_main:
                with mock.patch.object(sys, "argv", argv):
                    cli.main()
        mocked_main.assert_called_once_with(
            [
                "--stage",
                "paper_stage1",
                "--gpu-mode",
                "force",
                "--state-representation",
                "TRFTS-HAND",
                "--checkpoint",
                "results/custom/controller.pt",
                "--results-dir",
                "results/custom",
                "--reset",
            ]
        )

    def test_sac_policy_eval_dispatches_to_script_entrypoint(self) -> None:
        argv = [
            "uav_benchmark.cli",
            "sac-policy-eval",
            "--checkpoint",
            "results/sac_smopso_pretrain/controller.pt",
            "--modes",
            "online",
            "frozen",
            "--fleet-sizes",
            "3",
            "5",
        ]
        with mock.patch.object(cli, "_maybe_reexec_for_torch"):
            with mock.patch("scripts.evaluate_sac_smopso_policy_modes.main") as mocked_main:
                with mock.patch.object(sys, "argv", argv):
                    cli.main()
        mocked_main.assert_called_once_with(
            [
                "--fleet-sizes",
                "3",
                "5",
                "--gpu-mode",
                "force",
                "--modes",
                "online",
                "frozen",
                "--checkpoint",
                "results/sac_smopso_pretrain/controller.pt",
                "--results-dir",
                "results/sac_smopso_policy_eval",
            ]
        )

    def test_sac_policy_eval_accepts_protocol_dispatch(self) -> None:
        argv = [
            "uav_benchmark.cli",
            "sac-policy-eval",
            "--protocol",
            "configs/paper_policy_mode_comparison.yaml",
            "--results-dir",
            "results/policy",
        ]
        with mock.patch.object(cli, "_maybe_reexec_for_torch"):
            with mock.patch("scripts.evaluate_sac_smopso_policy_modes.main") as mocked_main:
                with mock.patch.object(sys, "argv", argv):
                    cli.main()
        mocked_main.assert_called_once_with(
            [
                "--gpu-mode",
                "force",
                "--protocol",
                "configs/paper_policy_mode_comparison.yaml",
                "--results-dir",
                "results/policy",
            ]
        )

    def test_sac_encoder_ablation_dispatches_to_script_entrypoint(self) -> None:
        argv = [
            "uav_benchmark.cli",
            "sac-encoder-ablation",
            "--protocol",
            "configs/paper_relational_encoder_ablation.yaml",
            "--checkpoint-template",
            "results/sac_smopso_pretrain/{mode}/controller.pt",
        ]
        with mock.patch.object(cli, "_maybe_reexec_for_torch"):
            with mock.patch("scripts.run_sac_smopso_state_ablation.main") as mocked_main:
                with mock.patch.object(sys, "argv", argv):
                    cli.main()
        mocked_main.assert_called_once_with(
            [
                "--gpu-mode",
                "force",
                "--checkpoint-template",
                "results/sac_smopso_pretrain/{mode}/controller.pt",
                "--protocol",
                "configs/paper_relational_encoder_ablation.yaml",
                "--results-dir",
                "results/sac_smopso_state_ablation",
            ]
        )

    def test_paper_artifacts_defaults_to_full_benchmark_protocol(self) -> None:
        params = cli.BenchmarkParams()
        with mock.patch.object(cli, "_maybe_reexec_for_torch"):
            with mock.patch.object(cli, "_load_protocol", return_value={}) as mocked_load_protocol:
                with mock.patch.object(cli.BenchmarkParams, "from_mapping", return_value=params):
                    with mock.patch.object(cli, "run_benchmark"):
                        with mock.patch.object(cli, "generate_benchmark_report"):
                            with mock.patch.object(cli, "statistical_analysis"):
                                with mock.patch.object(cli, "generate_research_plots"):
                                    with mock.patch("builtins.print"):
                                        with mock.patch.object(sys, "argv", ["uav_benchmark.cli", "paper-artifacts"]):
                                            cli.main()
        self.assertTrue(str(mocked_load_protocol.call_args.args[0]).endswith("configs/full_benchmark.yaml"))

    def test_benchmark_reexecs_into_project_venv_for_sac(self) -> None:
        argv = [
            "uav_benchmark.cli",
            "benchmark",
            "--project-root",
            ".",
            "--results-dir",
            "results/demo",
            "--protocol",
            "configs/full_benchmark.yaml",
            "--extra-json",
            '{"allowExperimentalAlgorithms": true, "algorithms": ["SAC-SMOPSO"]}',
        ]
        target = Path("/tmp/project-venv-python")
        with mock.patch.object(cli, "_fallback_torch_python", return_value=target):
            with mock.patch.object(cli.os, "execve", side_effect=SystemExit(0)) as mocked_execve:
                with mock.patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit):
                        cli.main()
        exec_env = mocked_execve.call_args.args[2]
        self.assertNotIn("__PYVENV_LAUNCHER__", exec_env)
        self.assertNotIn("PYTHONHOME", exec_env)
        self.assertNotIn("PYTHONPATH", exec_env)
        mocked_execve.assert_called_once_with(str(target), [str(target), *argv], exec_env)

    def test_benchmark_safe_run_does_not_reexec_for_torch(self) -> None:
        argv = [
            "uav_benchmark.cli",
            "benchmark",
            "--project-root",
            ".",
            "--results-dir",
            "results/demo",
            "--protocol",
            "configs/full_benchmark.yaml",
        ]
        with mock.patch.object(cli, "_fallback_torch_python") as mocked_fallback:
            with mock.patch.object(cli, "run_benchmark"):
                with mock.patch.object(cli, "compute_metrics"):
                    with mock.patch.object(cli, "statistical_analysis"):
                        with mock.patch.object(cli, "generate_benchmark_report"):
                            with mock.patch.object(cli, "generate_research_plots"):
                                with mock.patch.object(sys, "argv", argv):
                                    cli.main()
        mocked_fallback.assert_not_called()

    def test_sac_workflows_promote_gpu_mode_to_force(self) -> None:
        args = argparse.Namespace(command="sac-pretrain", gpu_mode="auto", protocol=None, extra_json="")
        cli._prefer_gpu_force_for_torch(args)
        self.assertEqual(args.gpu_mode, "force")

    def test_explicit_gpu_off_is_preserved_for_sac_workflows(self) -> None:
        args = argparse.Namespace(command="sac-pretrain", gpu_mode="off", protocol=None, extra_json="")
        cli._prefer_gpu_force_for_torch(args)
        self.assertEqual(args.gpu_mode, "off")

    def test_export_relational_artifacts_dispatches_to_script_entrypoint(self) -> None:
        argv = [
            "uav_benchmark.cli",
            "export-relational-paper-artifacts",
            "--input",
            "results/demo/summary.json",
            "--output-dir",
            "results/demo/tables",
        ]
        with mock.patch.object(cli, "_maybe_reexec_for_torch"):
            with mock.patch("scripts.export_relational_paper_artifacts.main") as mocked_main:
                with mock.patch.object(sys, "argv", argv):
                    cli.main()
        mocked_main.assert_called_once_with(
            [
                "--input",
                "results/demo/summary.json",
                "--output-dir",
                "results/demo/tables",
            ]
        )


if __name__ == "__main__":
    unittest.main()
