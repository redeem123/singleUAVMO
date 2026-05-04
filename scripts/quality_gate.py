from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

QUALITY_PATHS = [
    "uav_benchmark/config.py",
    "uav_benchmark/model_contracts.py",
    "uav_benchmark/exceptions.py",
    "uav_benchmark/benchmark.py",
    "uav_benchmark/benchmark_selection.py",
    "uav_benchmark/bootstrap.py",
    "uav_benchmark/cli.py",
    "uav_benchmark/cli_sac.py",
    "uav_benchmark/utils/gpu.py",
    "uav_benchmark/utils/random.py",
    "uav_benchmark/analysis/metrics/feasibility.py",
    "uav_benchmark/analysis/metrics/report.py",
    "uav_benchmark/analysis/plotting/helpers.py",
    "uav_benchmark/analysis/plotting/visualizers.py",
    "uav_benchmark/algorithms/cgpo/__init__.py",
    "uav_benchmark/algorithms/cgpo/core.py",
    "uav_benchmark/algorithms/cgpo/ppf.py",
    "uav_benchmark/algorithms/cmosma/__init__.py",
    "uav_benchmark/algorithms/fastr_moea/__init__.py",
    "uav_benchmark/algorithms/fastr_moea/operators.py",
    "uav_benchmark/algorithms/gcnmoea/__init__.py",
    "uav_benchmark/algorithms/gcnmoea/selection.py",
    "uav_benchmark/algorithms/mogwo/__init__.py",
    "uav_benchmark/algorithms/mogwo/archive.py",
    "uav_benchmark/algorithms/mogwo/components.py",
    "uav_benchmark/algorithms/mogwo/constants.py",
    "uav_benchmark/algorithms/mogwo/engine.py",
    "uav_benchmark/algorithms/nmopso/__init__.py",
    "uav_benchmark/algorithms/nmopso/legacy_core.py",
    "uav_benchmark/algorithms/sac_smopso/__init__.py",
    "uav_benchmark/algorithms/sac_smopso/actions.py",
    "uav_benchmark/algorithms/sac_smopso/constants.py",
    "uav_benchmark/algorithms/sac_smopso/controller.py",
    "uav_benchmark/algorithms/sac_smopso/controller_checkpoint.py",
    "uav_benchmark/algorithms/sac_smopso/controller_networks.py",
    "uav_benchmark/algorithms/sac_smopso/controller_replay.py",
    "uav_benchmark/algorithms/sac_smopso/controller_types.py",
    "uav_benchmark/algorithms/sac_smopso/geometry.py",
    "uav_benchmark/algorithms/sac_smopso/initialization.py",
    "uav_benchmark/algorithms/sac_smopso/reservoir.py",
    "uav_benchmark/algorithms/sac_smopso/scoring.py",
    "uav_benchmark/algorithms/sac_smopso/state.py",
    "uav_benchmark/algorithms/sac_smopso/torch_support.py",
    "uav_benchmark/algorithms/sac_smopso/workflow.py",
    "uav_benchmark/algorithms/sem4d/__init__.py",
    "uav_benchmark/algorithms/sem4d/core.py",
    "uav_benchmark/algorithms/sem4d/evolution.py",
    "uav_benchmark/algorithms/tactic_moea/__init__.py",
    "uav_benchmark/algorithms/tactic_moea/topology.py",
    "uav_benchmark/algorithms/shared/fleet_artifacts.py",
    "uav_benchmark/algorithms/shared/fleet_common.py",
    "uav_benchmark/algorithms/shared/fleet_nsga_runner.py",
    "uav_benchmark/algorithms/shared/fleet_pso_runner.py",
    "uav_benchmark/algorithms/shared/fleet_runner.py",
    "uav_benchmark/algorithms/shared/nmopso_engine.py",
    "uav_benchmark/algorithms/shared/nmopso_features.py",
    "uav_benchmark/algorithms/shared/nmopso_helpers.py",
    "tests/test_algorithm_registration.py",
    "tests/test_benchmark_manifest.py",
    "tests/test_config_validation.py",
    "tests/test_runtime_environment.py",
    "tests/test_model_contracts.py",
    "tests/test_sac_smopso_controller_state.py",
    "tests/test_sac_smopso_fleet_smoke.py",
    "tests/test_sac_smopso_geometry_score.py",
    "tests/test_sac_smopso_policy_eval.py",
    "tests/test_sac_smopso_pretraining.py",
    "tests/test_sac_smopso_state_ablation.py",
    "tests/test_sac_smopso_workflow.py",
    "scripts/check_module_sizes.py",
    "scripts/quality_gate.py",
]


def _run(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UAV benchmark quality gate.")
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-coverage", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    python = sys.executable
    _run([python, "-m", "ruff", "check", *QUALITY_PATHS], project_root)
    _run([python, "-m", "ruff", "format", "--check", *QUALITY_PATHS], project_root)
    _run([python, "-m", "pyright", *QUALITY_PATHS], project_root)
    _run([python, "scripts/check_module_sizes.py", "--project-root", str(project_root)], project_root)
    if args.skip_coverage:
        _run([python, "-m", "pytest", "-q"], project_root)
    else:
        _run([python, "-m", "coverage", "run", "-m", "pytest", "-q"], project_root)
        _run([python, "-m", "coverage", "report"], project_root)

    if args.skip_smoke:
        return

    with tempfile.TemporaryDirectory(prefix="uav_quality_gate_") as tmpdir:
        results_dir = Path(tmpdir) / "smoke_results"
        extra = json.dumps({"algorithms": ["NMOPSO"], "problemNames": ["c_100"], "maxWorkers": 1})
        _run(
            [
                python,
                "-m",
                "uav_benchmark.cli",
                "list-algorithms",
            ],
            project_root,
        )
        _run(
            [
                python,
                "-m",
                "uav_benchmark.cli",
                "benchmark",
                "--project-root",
                str(project_root),
                "--results-dir",
                str(results_dir),
                "--generations",
                "1",
                "--population",
                "4",
                "--runs",
                "1",
                "--fleet-size",
                "1",
                "--fleet-sizes",
                "1",
                "--scenario-set",
                "custom",
                "--gpu-mode",
                "off",
                "--seed",
                "123",
                "--extra-json",
                extra,
            ],
            project_root,
        )
        _run([python, "-m", "uav_benchmark.cli", "compute-metrics", "--results-dir", str(results_dir)], project_root)
        _run(
            [
                python,
                "-m",
                "uav_benchmark.cli",
                "report-metrics",
                "--project-root",
                str(project_root),
                "--results-dir",
                str(results_dir),
            ],
            project_root,
        )


if __name__ == "__main__":
    main()
