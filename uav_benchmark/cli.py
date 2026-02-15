from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_benchmark.analysis.plots_multi_uav import generate_multi_uav_plots
from uav_benchmark.analysis.compute_metrics import MetricConfig, compute_metrics
from uav_benchmark.analysis.benchmark_report import ReportConfig, generate_benchmark_report
from uav_benchmark.analysis.generate_research_plots import generate_research_plots
from uav_benchmark.analysis.statistical_analysis import statistical_analysis
from uav_benchmark.analysis.visualizers import path_visualizer, peak_visualizer
from uav_benchmark.benchmark import run_benchmark, run_nmopso_ablation
from uav_benchmark.config import BenchmarkParams


def _parse_fleet_sizes(raw: str) -> tuple[int, ...]:
    if not raw:
        return ()
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _load_protocol(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load protocol files. Install pyyaml.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol file must contain a mapping: {path}")
    return payload


def _build_params(args: argparse.Namespace) -> BenchmarkParams:
    extra = {}
    if args.extra_json:
        extra = json.loads(args.extra_json)
    params = BenchmarkParams(
        generations=args.generations,
        population=args.population,
        runs=args.runs,
        compute_metrics=args.compute_metrics,
        use_parallel=False,
        parallel_mode="none",
        safe_dist=args.safe_dist,
        drone_size=args.drone_size,
        results_dir=Path(args.results_dir),
        seed=args.seed,
        mode=args.mode if hasattr(args, "mode") else "single",
        fleet_size=args.fleet_size if hasattr(args, "fleet_size") else 1,
        fleet_sizes=_parse_fleet_sizes(args.fleet_sizes) if hasattr(args, "fleet_sizes") else (),
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
        protocol_params.results_dir = Path(args.results_dir).resolve()
        protocol_params.extra.update(extra)
        # Command-line mode override is intentional for aliases like benchmark-multi.
        protocol_params.mode = params.mode
        protocol_params.gpu_mode = params.gpu_mode
        if params.fleet_sizes:
            protocol_params.fleet_sizes = params.fleet_sizes
        if params.fleet_size > 1:
            protocol_params.fleet_size = params.fleet_size
        if params.separation_min > 0:
            protocol_params.separation_min = params.separation_min
        if params.max_turn_deg > 0:
            protocol_params.max_turn_deg = params.max_turn_deg
        if params.evaluation_budget >= 0:
            protocol_params.evaluation_budget = params.evaluation_budget
        if params.scenario_set:
            protocol_params.scenario_set = params.scenario_set
        return protocol_params
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Python UAV benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--project-root", default=".", type=str)
    benchmark_parser.add_argument("--results-dir", default="results", type=str)
    benchmark_parser.add_argument("--generations", default=500, type=int)
    benchmark_parser.add_argument("--population", default=100, type=int)
    benchmark_parser.add_argument("--runs", default=14, type=int)
    benchmark_parser.add_argument("--compute-metrics", action="store_true")
    benchmark_parser.add_argument("--safe-dist", default=20.0, type=float)
    benchmark_parser.add_argument("--drone-size", default=1.0, type=float)
    benchmark_parser.add_argument("--seed", default=None, type=int)
    benchmark_parser.add_argument("--extra-json", default="", type=str)
    benchmark_parser.add_argument("--mode", choices=("single", "multi"), default="single", type=str)
    benchmark_parser.add_argument("--fleet-size", default=1, type=int)
    benchmark_parser.add_argument("--fleet-sizes", default="", type=str, help="Comma-separated fleet sizes, e.g. 3,5,8")
    benchmark_parser.add_argument("--scenario-set", default="paper_medium", type=str)
    benchmark_parser.add_argument("--separation-min", default=10.0, type=float)
    benchmark_parser.add_argument("--max-turn-deg", default=75.0, type=float)
    benchmark_parser.add_argument("--evaluation-budget", default=0, type=int)
    benchmark_parser.add_argument("--gpu-mode", choices=("auto", "off", "force"), default="auto", type=str)
    benchmark_parser.add_argument("--protocol", default="", type=str, help="YAML protocol config path")
    benchmark_parser.add_argument(
        "--plots-after",
        action="store_true",
        help="Generate research plots automatically after benchmark completes",
    )

    multi_parser = subparsers.add_parser("benchmark-multi")
    multi_parser.add_argument("--project-root", default=".", type=str)
    multi_parser.add_argument("--results-dir", default="results", type=str)
    multi_parser.add_argument("--generations", default=300, type=int)
    multi_parser.add_argument("--population", default=80, type=int)
    multi_parser.add_argument("--runs", default=10, type=int)
    multi_parser.add_argument("--compute-metrics", action="store_true")
    multi_parser.add_argument("--safe-dist", default=20.0, type=float)
    multi_parser.add_argument("--drone-size", default=1.0, type=float)
    multi_parser.add_argument("--seed", default=11, type=int)
    multi_parser.add_argument("--extra-json", default="", type=str)
    multi_parser.add_argument("--mode", choices=("single", "multi"), default="multi", type=str)
    multi_parser.add_argument("--fleet-size", default=3, type=int)
    multi_parser.add_argument("--fleet-sizes", default="3,5,8", type=str)
    multi_parser.add_argument("--scenario-set", default="paper_medium", type=str)
    multi_parser.add_argument("--separation-min", default=10.0, type=float)
    multi_parser.add_argument("--max-turn-deg", default=75.0, type=float)
    multi_parser.add_argument("--evaluation-budget", default=0, type=int)
    multi_parser.add_argument("--gpu-mode", choices=("auto", "off", "force"), default="auto", type=str)
    multi_parser.add_argument("--protocol", default="", type=str)
    multi_parser.add_argument("--plots-after", action="store_true")

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
    artifacts_parser.add_argument("--protocol", default="configs/paper_medium_multi.yaml", type=str)
    artifacts_parser.add_argument("--gpu-mode", choices=("auto", "off", "force"), default="auto", type=str)

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

    args = parser.parse_args()

    if args.command == "benchmark" or args.command == "benchmark-multi":
        project_root = Path(args.project_root).resolve()
        params = _build_params(args)
        params.results_dir = params.results_dir.resolve()
        run_benchmark(project_root, params)
        if args.plots_after:
            if str(params.mode).lower() == "multi":
                generate_multi_uav_plots(project_root, params.results_dir.resolve())
            else:
                generate_research_plots(project_root, params.results_dir.resolve())
        return
    if args.command == "ablation":
        params = _build_params(args)
        params.extra["ablationStudy"] = True
        run_nmopso_ablation(Path(args.project_root).resolve(), params)
        return
    if args.command == "compute-metrics":
        cfg = MetricConfig(hv_samples=args.hv_samples, max_points=args.max_points, max_runs=args.max_runs, seed=args.seed)
        compute_metrics(Path(args.results_dir).resolve(), cfg)
        return
    if args.command == "stats":
        cfg = MetricConfig(hv_samples=args.hv_samples, max_points=args.max_points, max_runs=args.max_runs, seed=args.seed)
        report = statistical_analysis(Path(args.results_dir).resolve(), cfg)
        for algorithm, rows in report.items():
            print(f"\n{algorithm}")
            for row in rows:
                mean_obj = " ".join(f"{value:.4f}" for value in row.mean_obj.tolist())
                std_obj = " ".join(f"{value:.4f}" for value in row.std_obj.tolist())
                print(f"{row.problem:30s} HV {row.mean_hv:.4f}±{row.std_hv:.4f} | OBJ {mean_obj} | STD {std_obj}")
        return
    if args.command == "report-metrics":
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
        return
    if args.command == "plots":
        generate_research_plots(Path(args.project_root).resolve(), Path(args.results_dir).resolve())
        return
    if args.command == "paper-artifacts":
        project_root = Path(args.project_root).resolve()
        protocol_path = Path(args.protocol).resolve()
        protocol = _load_protocol(protocol_path)
        params = BenchmarkParams.from_mapping(protocol)
        params.mode = "multi"
        params.gpu_mode = args.gpu_mode
        params.results_dir = Path(args.results_dir).resolve()
        run_benchmark(project_root, params)
        cfg = ReportConfig(
            project_root=project_root,
            results_dir=params.results_dir,
            output_dir=params.results_dir / "metrics",
            hv_samples=2000,
            max_runs=0,
            baseline_algorithm="NMOPSO",
            seed=int(params.seed) if params.seed is not None else 0,
        )
        generate_benchmark_report(cfg)
        stats_cfg = MetricConfig(hv_samples=2000, max_points=100, max_runs=0, seed=int(params.seed) if params.seed is not None else 0)
        statistical_analysis(params.results_dir, stats_cfg)
        generate_multi_uav_plots(project_root, params.results_dir)
        print(params.results_dir / "metrics")
        print(params.results_dir / "plots_multi_uav")
        return
    if args.command == "path-visualizer":
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
        return
    if args.command == "peak-visualizer":
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None
        outputs = peak_visualizer(Path(args.project_root).resolve(), output_dir=output_dir)
        for path in outputs:
            print(path)
        return


if __name__ == "__main__":
    main()
