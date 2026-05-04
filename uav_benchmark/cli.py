from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from uav_benchmark.algorithms import algorithm_profile_specs, algorithm_specs, resolve_algorithm_profile
from uav_benchmark.analysis.metrics.compute import MetricConfig, compute_metrics
from uav_benchmark.analysis.metrics.report import ReportConfig, generate_benchmark_report
from uav_benchmark.analysis.metrics.stats import statistical_analysis
from uav_benchmark.analysis.plotting.research import generate_research_plots
from uav_benchmark.analysis.plotting.visualizers import path_visualizer, peak_visualizer
from uav_benchmark.benchmark import _normalize_algorithm_name, run_benchmark, run_nmopso_ablation
from uav_benchmark.cli_sac import (
    TORCH_REQUIRED_COMMANDS,
    handle_sac_encoder_ablation,
    handle_sac_policy_eval,
    handle_sac_pretrain,
    register_sac_commands,
)
from uav_benchmark.config import BenchmarkParams

_BENCHMARK_FLEET_SIZE_DEFAULT = 1
_BENCHMARK_FLEET_SIZES_DEFAULT = "1,3"
_BENCHMARK_DEFAULT_PROBLEM_NAMES = ("c_100", "m_100", "s_120")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TORCH_ACCELERATED_ALGORITHMS = {"SAC-SMOPSO", "RA-SMPSO", "RA-NSGA-II"}


class _MarkExplicitAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


def _tokenize_mapping_value(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    if isinstance(raw, (list, tuple)):
        values: list[str] = []
        for item in raw:
            text = str(item).strip()
            if text:
                values.append(text)
        return tuple(values)
    text = str(raw).strip()
    return (text,) if text else ()


def _current_python_has_torch() -> bool:
    try:
        import torch  # type: ignore
    except (ImportError, OSError, RuntimeError):
        return False
    return torch is not None


def _python_has_torch(python_path: Path) -> bool:
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import torch"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _fallback_torch_python() -> Path | None:
    if _current_python_has_torch():
        return None
    scripts_dir = _PROJECT_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    candidates = [scripts_dir / "python", scripts_dir / "python3"]
    current = Path(sys.executable).absolute()
    for candidate in candidates:
        if not candidate.exists():
            continue
        absolute_candidate = candidate.absolute()
        if absolute_candidate == current:
            continue
        if _python_has_torch(candidate):
            return candidate
    return None


def _merged_protocol_mapping(args: argparse.Namespace) -> dict:
    mapping: dict = {}
    protocol = getattr(args, "protocol", None)
    if protocol:
        mapping.update(_load_protocol(Path(protocol).resolve()))
    extra_json = getattr(args, "extra_json", "")
    if extra_json:
        mapping.update(json.loads(extra_json))
    return mapping


def _requested_algorithm_names(mapping: dict) -> tuple[str, ...]:
    requested: list[str] = []
    profile_keys = ("algorithmProfiles", "algorithm_profiles", "algorithmSets", "algorithm_sets")
    profile_value_keys = ("algorithmProfile", "algorithm_profile", "algorithmSet", "algorithm_set")
    for key in profile_keys + profile_value_keys:
        for token in _tokenize_mapping_value(mapping.get(key)):
            requested.extend(resolve_algorithm_profile(token))
    for token in _tokenize_mapping_value(mapping.get("algorithms")):
        requested.append(_normalize_algorithm_name(token))
    deduped: list[str] = []
    seen: set[str] = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return tuple(deduped)


def _command_needs_torch_reexec(args: argparse.Namespace) -> bool:
    if args.command in TORCH_REQUIRED_COMMANDS:
        return True
    if args.command not in {"benchmark", "paper-artifacts"}:
        return False
    if getattr(args, "gpu_mode", "auto") == "off":
        return False
    requested = _requested_algorithm_names(_merged_protocol_mapping(args))
    return any(name in _TORCH_ACCELERATED_ALGORITHMS for name in requested)


def _maybe_reexec_for_torch(args: argparse.Namespace) -> None:
    if not _command_needs_torch_reexec(args):
        return
    fallback_python = _fallback_torch_python()
    if fallback_python is None:
        return
    clean_env = dict(os.environ)
    for key in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH"):
        clean_env.pop(key, None)
    print(
        f"Re-executing under {fallback_python} so Torch-backed SAC workflows can use GPU acceleration.",
        file=sys.stderr,
    )
    os.execve(str(fallback_python), [str(fallback_python), *sys.argv], clean_env)


def _prefer_gpu_force_for_torch(args: argparse.Namespace) -> None:
    if not _command_needs_torch_reexec(args):
        return
    if not hasattr(args, "gpu_mode"):
        return
    if args.gpu_mode == "off":
        return
    args.gpu_mode = "force"


def _parse_fleet_sizes(raw: str) -> tuple[int, ...]:
    if not raw:
        return ()
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _load_protocol(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load protocol files. Install pyyaml.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol file must contain a mapping: {path}")
    return payload


def _append_scalar_arg(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    argv.extend([flag, text])


def _append_multi_arg(argv: list[str], flag: str, values: list[object] | tuple[object, ...] | None) -> None:
    if not values:
        return
    argv.append(flag)
    argv.extend(str(value) for value in values)


def _ensure_project_root_on_path() -> None:
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _build_argv(
    args: argparse.Namespace,
    *,
    multi: tuple[tuple[str, str], ...] = (),
    scalar: tuple[tuple[str, str], ...] = (),
    flags: tuple[tuple[str, str], ...] = (),
) -> list[str]:
    argv: list[str] = []
    for flag, attr in multi:
        _append_multi_arg(argv, flag, getattr(args, attr, None))
    for flag, attr in scalar:
        _append_scalar_arg(argv, flag, getattr(args, attr, None))
    for flag, attr in flags:
        if getattr(args, attr, False):
            argv.append(flag)
    return argv


def _build_relational_export_argv(args: argparse.Namespace) -> list[str]:
    return _build_argv(
        args,
        scalar=(
            ("--input", "input"),
            ("--output-dir", "output_dir"),
        ),
    )


def _arg_overrides_default(args: argparse.Namespace, name: str, default: object) -> bool:
    explicit_marker = f"_{name}_explicit"
    if hasattr(args, explicit_marker):
        return bool(getattr(args, explicit_marker))
    if not hasattr(args, name):
        return False
    return getattr(args, name) != default


def _benchmark_fleet_sizes_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    if not hasattr(args, "fleet_sizes"):
        return ()
    if _arg_overrides_default(args, "fleet_sizes", _BENCHMARK_FLEET_SIZES_DEFAULT):
        return _parse_fleet_sizes(args.fleet_sizes)
    if _arg_overrides_default(args, "fleet_size", _BENCHMARK_FLEET_SIZE_DEFAULT):
        return ()
    return _parse_fleet_sizes(_BENCHMARK_FLEET_SIZES_DEFAULT)


def _build_params(args: argparse.Namespace) -> BenchmarkParams:
    extra = {}
    if args.extra_json:
        extra = json.loads(args.extra_json)
    # Only inject default problemNames when not loading a protocol
    # (protocols specify their own problemNames)
    if (
        not (hasattr(args, "protocol") and args.protocol)
        and "problemNames" not in extra
        and "problem_names" not in extra
        and "problems" not in extra
    ):
        extra["problemNames"] = list(_BENCHMARK_DEFAULT_PROBLEM_NAMES)
    params = BenchmarkParams(
        generations=args.generations,
        population=args.population,
        runs=args.runs,
        compute_metrics=True,
        use_parallel=False,
        parallel_mode="none",
        safe_dist=args.safe_dist,
        drone_size=args.drone_size,
        results_dir=Path(args.results_dir),
        seed=args.seed,
        mode="fleet",
        fleet_size=args.fleet_size if hasattr(args, "fleet_size") else 1,
        fleet_sizes=_benchmark_fleet_sizes_from_args(args),
        separation_min=args.separation_min if hasattr(args, "separation_min") else 10.0,
        max_turn_deg=args.max_turn_deg if hasattr(args, "max_turn_deg") else 75.0,
        evaluation_budget=args.evaluation_budget if hasattr(args, "evaluation_budget") else 0,
        scenario_set=args.scenario_set if hasattr(args, "scenario_set") else "paper_medium",
        gpu_mode=args.gpu_mode if hasattr(args, "gpu_mode") else "auto",
        extra=extra,
    )
    if hasattr(args, "protocol") and args.protocol:
        protocol_mapping = _load_protocol(Path(args.protocol).resolve())
        protocol_params = BenchmarkParams.from_mapping(protocol_mapping)
        if _arg_overrides_default(args, "results_dir", "results"):
            protocol_params.results_dir = Path(args.results_dir).resolve()
        protocol_params.extra.update(extra)
        # Command-line mode override is intentional.
        protocol_params.mode = params.mode
        protocol_params.gpu_mode = params.gpu_mode
        if _arg_overrides_default(args, "fleet_sizes", _BENCHMARK_FLEET_SIZES_DEFAULT) and params.fleet_sizes:
            protocol_params.fleet_sizes = params.fleet_sizes
        if _arg_overrides_default(args, "fleet_size", _BENCHMARK_FLEET_SIZE_DEFAULT):
            protocol_params.fleet_size = params.fleet_size
            if not _arg_overrides_default(args, "fleet_sizes", _BENCHMARK_FLEET_SIZES_DEFAULT):
                protocol_params.fleet_sizes = ()
        if _arg_overrides_default(args, "separation_min", 10.0):
            protocol_params.separation_min = params.separation_min
        if _arg_overrides_default(args, "max_turn_deg", 75.0):
            protocol_params.max_turn_deg = params.max_turn_deg
        if _arg_overrides_default(args, "evaluation_budget", 0):
            protocol_params.evaluation_budget = params.evaluation_budget
        if _arg_overrides_default(args, "scenario_set", "paper_medium"):
            protocol_params.scenario_set = params.scenario_set
        return protocol_params
    return params


def _print_algorithm_catalog() -> None:
    for availability, title in (("benchmark", "Benchmark-Safe"), ("experimental", "Experimental")):
        print(f"\n{title}")
        for spec in algorithm_specs(availability):
            print(f"- {spec.name}: {spec.summary}")
    print("\nNamed Profiles")
    for profile in algorithm_profile_specs():
        suffix = " [experimental opt-in]" if profile.requires_experimental else ""
        joined = ", ".join(profile.algorithms)
        print(f"- {profile.name}{suffix}: {profile.summary}")
        print(f"  algorithms: {joined}")


def _params_seed(params: BenchmarkParams) -> int:
    return int(params.seed) if params.seed is not None else 0


def _default_metric_config(seed: int) -> MetricConfig:
    return MetricConfig(hv_samples=2000, max_points=100, max_runs=0, seed=int(seed))


def _metric_config_from_args(args: argparse.Namespace) -> MetricConfig:
    return MetricConfig(
        hv_samples=args.hv_samples,
        max_points=args.max_points,
        max_runs=args.max_runs,
        seed=args.seed,
    )


def _baseline_algorithm_from_params(params: BenchmarkParams) -> str:
    if not isinstance(params.extra, dict):
        return "NMOPSO"
    raw_algorithms = params.extra.get("algorithms")
    if isinstance(raw_algorithms, (list, tuple)) and raw_algorithms:
        return _normalize_algorithm_name(str(raw_algorithms[0]))
    return "NMOPSO"


def _handle_archive(args: argparse.Namespace) -> None:
    import shutil
    from datetime import datetime

    results_path = Path(args.results_dir).resolve()
    archive_path = Path(args.archive_dir).resolve()
    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = archive_path / f"results_archive_{timestamp}"
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Moving {results_path} to {target_dir}")
    for item in results_path.iterdir():
        shutil.move(str(item), str(target_dir))
    print("Archive complete.")


def _handle_benchmark(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    params = _build_params(args)
    params.results_dir = params.results_dir.resolve()
    run_benchmark(project_root, params)

    metrics_cfg = _default_metric_config(_params_seed(params))
    compute_metrics(params.results_dir.resolve(), metrics_cfg)
    statistical_analysis(params.results_dir.resolve(), metrics_cfg)

    report_cfg = ReportConfig(
        project_root=project_root,
        results_dir=params.results_dir.resolve(),
        output_dir=None,
        hv_samples=2000,
        max_runs=0,
        baseline_algorithm=_baseline_algorithm_from_params(params),
        seed=_params_seed(params),
    )
    generate_benchmark_report(report_cfg)
    generate_research_plots(project_root, params.results_dir.resolve())


def _handle_ablation(args: argparse.Namespace) -> None:
    params = _build_params(args)
    params.extra["ablationStudy"] = True
    params.extra["legacyPathRunner"] = True
    run_nmopso_ablation(Path(args.project_root).resolve(), params)


def _handle_compute_metrics(args: argparse.Namespace) -> None:
    compute_metrics(Path(args.results_dir).resolve(), _metric_config_from_args(args))


def _handle_stats(args: argparse.Namespace) -> None:
    report = statistical_analysis(Path(args.results_dir).resolve(), _metric_config_from_args(args))
    for algorithm, rows in report.items():
        print(f"\n{algorithm}")
        for row in rows:
            mean_obj = " ".join(f"{value:.4f}" for value in row.mean_obj.tolist())
            std_obj = " ".join(f"{value:.4f}" for value in row.std_obj.tolist())
            print(f"{row.problem:30s} HV {row.mean_hv:.4f}±{row.std_hv:.4f} | OBJ {mean_obj} | STD {std_obj}")


def _handle_report_metrics(args: argparse.Namespace) -> None:
    cfg = ReportConfig(
        project_root=Path(args.project_root).resolve(),
        results_dir=Path(args.results_dir).resolve(),
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        hv_samples=args.hv_samples,
        max_runs=args.max_runs,
        baseline_algorithm=args.baseline_algorithm,
        seed=args.seed,
    )
    report = generate_benchmark_report(cfg)
    print(f"Summary rows: {report['summary_rows']}")
    print(f"Pairwise rows: {report['pairwise_rows']}")
    print(f"summary_csv: {report['summary_csv']}")
    if report["pairwise_csv"] is not None:
        print(f"pairwise_csv: {report['pairwise_csv']}")
    if report.get("win_tie_loss_csv") is not None:
        print(f"win_tie_loss_csv: {report['win_tie_loss_csv']}")
    print(f"summary_json: {report['summary_json']}")


def _handle_plots(args: argparse.Namespace) -> None:
    generate_research_plots(Path(args.project_root).resolve(), Path(args.results_dir).resolve())


def _handle_paper_artifacts(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    protocol = _load_protocol(Path(args.protocol).resolve())
    params = BenchmarkParams.from_mapping(protocol)
    params.mode = "fleet"
    params.gpu_mode = args.gpu_mode
    params.results_dir = Path(args.results_dir).resolve()
    run_benchmark(project_root, params)

    report_cfg = ReportConfig(
        project_root=project_root,
        results_dir=params.results_dir,
        output_dir=params.results_dir / "metrics",
        hv_samples=2000,
        max_runs=0,
        baseline_algorithm="NMOPSO",
        seed=_params_seed(params),
    )
    generate_benchmark_report(report_cfg)
    statistical_analysis(params.results_dir, _default_metric_config(_params_seed(params)))
    generate_research_plots(project_root, params.results_dir)
    print(params.results_dir / "metrics")
    print(params.results_dir / "Plots")


def _handle_export_relational_artifacts(args: argparse.Namespace) -> None:
    _ensure_project_root_on_path()
    from scripts.export_relational_paper_artifacts import main as export_relational_main

    export_relational_main(_build_relational_export_argv(args))


def _handle_path_visualizer(args: argparse.Namespace) -> None:
    output = path_visualizer(
        Path(args.project_root).resolve(),
        args.problem_name,
        args.run_num,
        args.algorithm,
        show=args.show,
        path_index=args.path_index,
        feasible_only=not args.allow_infeasible,
        display_lift=args.display_lift,
    )
    print(output)


def _handle_peak_visualizer(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    outputs = peak_visualizer(Path(args.project_root).resolve(), output_dir=output_dir)
    for path in outputs:
        print(path)


def _dispatch_command(args: argparse.Namespace) -> None:
    handlers = {
        "list-algorithms": lambda _args: _print_algorithm_catalog(),
        "archive": _handle_archive,
        "benchmark": _handle_benchmark,
        "ablation": _handle_ablation,
        "compute-metrics": _handle_compute_metrics,
        "stats": _handle_stats,
        "report-metrics": _handle_report_metrics,
        "plots": _handle_plots,
        "paper-artifacts": _handle_paper_artifacts,
        "sac-pretrain": handle_sac_pretrain,
        "sac-policy-eval": handle_sac_policy_eval,
        "sac-encoder-ablation": handle_sac_encoder_ablation,
        "export-relational-paper-artifacts": _handle_export_relational_artifacts,
        "path-visualizer": _handle_path_visualizer,
        "peak-visualizer": _handle_peak_visualizer,
    }
    handlers[args.command](args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python UAV benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-algorithms")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--project-root", default=".", type=str)
    benchmark_parser.add_argument("--results-dir", default="results", type=str)
    benchmark_parser.add_argument("--generations", default=500, type=int)
    benchmark_parser.add_argument("--population", default=100, type=int)
    benchmark_parser.add_argument("--runs", default=14, type=int)
    benchmark_parser.add_argument("--safe-dist", default=20.0, type=float)
    benchmark_parser.add_argument("--drone-size", default=1.0, type=float)
    benchmark_parser.add_argument("--seed", default=None, type=int)
    benchmark_parser.add_argument("--extra-json", default="", type=str)
    benchmark_parser.add_argument(
        "--fleet-size",
        default=_BENCHMARK_FLEET_SIZE_DEFAULT,
        type=int,
        action=_MarkExplicitAction,
    )
    benchmark_parser.add_argument(
        "--fleet-sizes",
        default=_BENCHMARK_FLEET_SIZES_DEFAULT,
        type=str,
        action=_MarkExplicitAction,
        help="Comma-separated fleet sizes, e.g. 3,5,8",
    )
    benchmark_parser.add_argument("--scenario-set", default="paper_medium", type=str)
    benchmark_parser.add_argument("--separation-min", default=10.0, type=float)
    benchmark_parser.add_argument("--max-turn-deg", default=75.0, type=float)
    benchmark_parser.add_argument("--evaluation-budget", default=0, type=int)
    benchmark_parser.add_argument("--gpu-mode", choices=("auto", "off", "force"), default="auto", type=str)
    benchmark_parser.add_argument("--protocol", default="", type=str, help="YAML protocol config path")

    ablation_parser = subparsers.add_parser("ablation")
    ablation_parser.add_argument("--project-root", default=".", type=str)
    ablation_parser.add_argument("--results-dir", default="results/NMOPSO_ABLATION", type=str)
    ablation_parser.add_argument("--generations", default=200, type=int)
    ablation_parser.add_argument("--population", default=80, type=int)
    ablation_parser.add_argument("--runs", default=6, type=int)
    ablation_parser.add_argument("--compute-metrics", action="store_true")
    ablation_parser.add_argument("--safe-dist", default=20.0, type=float)
    ablation_parser.add_argument("--drone-size", default=1.0, type=float)
    ablation_parser.add_argument("--seed", default=None, type=int)
    ablation_parser.add_argument("--extra-json", default="", type=str)

    metrics_parser = subparsers.add_parser("compute-metrics")
    metrics_parser.add_argument("--results-dir", default="results", type=str)
    metrics_parser.add_argument("--hv-samples", default=2000, type=int)
    metrics_parser.add_argument("--max-points", default=100, type=int)
    metrics_parser.add_argument("--max-runs", default=0, type=int)
    metrics_parser.add_argument("--seed", default=0, type=int)

    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--results-dir", default="results", type=str)
    stats_parser.add_argument("--hv-samples", default=2000, type=int)
    stats_parser.add_argument("--max-points", default=100, type=int)
    stats_parser.add_argument("--max-runs", default=0, type=int)
    stats_parser.add_argument("--seed", default=0, type=int)

    report_parser = subparsers.add_parser("report-metrics")
    report_parser.add_argument("--project-root", default=".", type=str)
    report_parser.add_argument("--results-dir", default="results", type=str)
    report_parser.add_argument("--output-dir", default="", type=str)
    report_parser.add_argument("--hv-samples", default=2000, type=int)
    report_parser.add_argument("--max-runs", default=0, type=int)
    report_parser.add_argument("--baseline-algorithm", default="NMOPSO", type=str)
    report_parser.add_argument("--seed", default=0, type=int)

    plots_parser = subparsers.add_parser("plots")
    plots_parser.add_argument("--project-root", default=".", type=str)
    plots_parser.add_argument("--results-dir", default="results", type=str)

    artifacts_parser = subparsers.add_parser("paper-artifacts")
    artifacts_parser.add_argument("--project-root", default=".", type=str)
    artifacts_parser.add_argument("--results-dir", default="results/paper_artifacts", type=str)
    artifacts_parser.add_argument("--protocol", default="configs/full_benchmark.yaml", type=str)
    artifacts_parser.add_argument("--gpu-mode", choices=("auto", "off", "force"), default="auto", type=str)

    register_sac_commands(subparsers)

    export_relational_parser = subparsers.add_parser("export-relational-paper-artifacts")
    export_relational_parser.add_argument("--input", required=True, type=str, help="Input summary JSON.")
    export_relational_parser.add_argument("--output-dir", required=True, type=str, help="Output directory.")

    path_parser = subparsers.add_parser("path-visualizer")
    path_parser.add_argument("problem_name", type=str)
    path_parser.add_argument("run_num", type=int)
    path_parser.add_argument("--project-root", default=".", type=str)
    path_parser.add_argument("--algorithm", default="NSGA-II", type=str)
    path_parser.add_argument("--show", action="store_true")
    path_parser.add_argument("--path-index", type=int, default=None)
    path_parser.add_argument("--display-lift", type=float, default=0.0)
    path_parser.add_argument("--allow-infeasible", action="store_true")

    peak_parser = subparsers.add_parser("peak-visualizer")
    peak_parser.add_argument("--project-root", default=".", type=str)
    peak_parser.add_argument("--output-dir", default="", type=str)

    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("--results-dir", default="results", type=str)
    archive_parser.add_argument("--archive-dir", default="archives", type=str)

    args = parser.parse_args()
    _prefer_gpu_force_for_torch(args)
    _maybe_reexec_for_torch(args)
    _dispatch_command(args)


if __name__ == "__main__":
    main()
