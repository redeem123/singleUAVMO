from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from uav_benchmark.analysis.metrics.report import ReportConfig, generate_benchmark_report
from uav_benchmark.io.matlab import load_mat

BASELINE = "CGPO_full"
HIGHER_IS_BETTER = {"hv", "feasible_ratio"}
LOWER_IS_BETTER = {"igd_plus", "runtime_sec", "mission_conflict_rate", "mission_makespan", "mission_energy"}

# Paper-honest variants of the lean three-mechanism CGPO.
MECHANISM_BY_VARIANT = {
    "CGPO_full": "all three published mechanisms (CIG + PPF + OVO)",
    "CGPO_random_only": "all three CGPO mechanisms",
    "CGPO_no_cig_edge_coupling": "constraint-interaction edge coupling",
    "CGPO_no_ppf_pressure": "Pareto pressure field (selection)",
    "CGPO_no_ovo_variation": "orchestrated variation operator",
    "CGPO_no_ovo_fleet_coordination": "OVO fleet-coordination push only",
}
FAILURE_MODE_BY_VARIANT = {
    "CGPO_full": "must dominate the random_only baseline on HV and feasibility",
    "CGPO_random_only": "establishes the NSGA-II floor with no graph awareness",
    "CGPO_no_cig_edge_coupling": "pairwise and terrain tension should decay more slowly; PPF + OVO degrade",
    "CGPO_no_ppf_pressure": "uniform parent sampling may waste evaluations on poor parents",
    "CGPO_no_ovo_variation": "Gaussian mutation should exhibit slower convergence than tension-aware OVO",
    "CGPO_no_ovo_fleet_coordination": "fleet conflict rate should remain higher in fleet >= 3 cases",
}

SINGLE_COMPONENT_VARIANTS = {
    "CGPO_no_ppf_pressure",
    "CGPO_no_ovo_variation",
    "CGPO_no_ovo_fleet_coordination",
    "CGPO_no_cig_edge_coupling",
    "CGPO_random_only",
}
REDUCED_VARIANTS: set[str] = set()


def _float(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value


def _problem_fleet(problem: str) -> int:
    if "_uav" not in problem:
        return 1
    try:
        return int(problem.rsplit("_uav", 1)[1])
    except ValueError:
        return 1


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _dominance_score(row: dict[str, str]) -> tuple[float, float, float, float]:
    feasible = _float(row.get("feasible_ratio_mean"))
    hv = _float(row.get("hv_mean"))
    igd = _float(row.get("igd_plus_mean"))
    runtime = _float(row.get("runtime_sec_mean"))
    return (
        feasible if np.isfinite(feasible) else -1.0,
        hv if np.isfinite(hv) else -1.0,
        -igd if np.isfinite(igd) else float("-inf"),
        -runtime if np.isfinite(runtime) else float("-inf"),
    )


def _better_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=_dominance_score)


def _delta_pct(new_value: float, base_value: float) -> float:
    if not np.isfinite(new_value) or not np.isfinite(base_value) or abs(base_value) <= 1e-12:
        return float("nan")
    return 100.0 * (new_value - base_value) / abs(base_value)


def _component_dominance_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows_by_problem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_problem[str(row.get("problem", ""))].append(row)

    out: list[dict[str, Any]] = []
    for problem, rows in sorted(rows_by_problem.items()):
        by_algorithm = {str(row.get("algorithm", "")): row for row in rows}
        main = by_algorithm.get(BASELINE)
        if main is None:
            continue
        single = _better_row([row for name, row in by_algorithm.items() if name in SINGLE_COMPONENT_VARIANTS])
        reduced = _better_row([row for name, row in by_algorithm.items() if name in REDUCED_VARIANTS])

        main_hv = _float(main.get("hv_mean"))
        main_feasible = _float(main.get("feasible_ratio_mean"))
        main_igd = _float(main.get("igd_plus_mean"))
        main_runtime = _float(main.get("runtime_sec_mean"))

        def value(row: dict[str, str] | None, key: str) -> float:
            return _float(row.get(key)) if row is not None else float("nan")

        single_hv = value(single, "hv_mean")
        single_feasible = value(single, "feasible_ratio_mean")
        single_igd = value(single, "igd_plus_mean")
        single_runtime = value(single, "runtime_sec_mean")
        reduced_hv = value(reduced, "hv_mean")
        reduced_feasible = value(reduced, "feasible_ratio_mean")
        reduced_igd = value(reduced, "igd_plus_mean")
        reduced_runtime = value(reduced, "runtime_sec_mean")

        main_beats_single = (
            np.isfinite(main_feasible)
            and np.isfinite(single_feasible)
            and np.isfinite(main_hv)
            and np.isfinite(single_hv)
            and main_feasible >= single_feasible - 1e-9
            and main_hv > single_hv + 1e-9
        )
        reduced_beats_single = (
            np.isfinite(reduced_feasible)
            and np.isfinite(single_feasible)
            and np.isfinite(reduced_hv)
            and np.isfinite(single_hv)
            and reduced_feasible >= single_feasible - 1e-9
            and reduced_hv > single_hv + 1e-9
        )
        if single is None:
            interpretation = "missing_single_component_baselines"
        elif main_beats_single:
            interpretation = "cgpo_r_adds_value_over_best_single"
        elif reduced is not None and reduced_beats_single:
            interpretation = "reduced_method_adds_value_over_best_single"
        else:
            interpretation = "dominant_single_component_explains_performance"

        out.append(
            {
                "problem": problem,
                "fleet_size": _problem_fleet(problem),
                "main_algorithm": BASELINE,
                "main_hv_mean": main_hv,
                "main_feasible_ratio_mean": main_feasible,
                "main_igd_plus_mean": main_igd,
                "main_runtime_sec_mean": main_runtime,
                "best_single_component": single.get("algorithm", "") if single is not None else "",
                "best_single_hv_mean": single_hv,
                "best_single_feasible_ratio_mean": single_feasible,
                "best_single_igd_plus_mean": single_igd,
                "best_single_runtime_sec_mean": single_runtime,
                "main_minus_best_single_hv": main_hv - single_hv
                if np.isfinite(main_hv) and np.isfinite(single_hv)
                else float("nan"),
                "main_minus_best_single_hv_pct": _delta_pct(main_hv, single_hv),
                "main_minus_best_single_feasible_ratio": (
                    main_feasible - single_feasible
                    if np.isfinite(main_feasible) and np.isfinite(single_feasible)
                    else float("nan")
                ),
                "main_minus_best_single_igd_plus": main_igd - single_igd
                if np.isfinite(main_igd) and np.isfinite(single_igd)
                else float("nan"),
                "best_reduced_variant": reduced.get("algorithm", "") if reduced is not None else "",
                "best_reduced_hv_mean": reduced_hv,
                "best_reduced_feasible_ratio_mean": reduced_feasible,
                "best_reduced_igd_plus_mean": reduced_igd,
                "best_reduced_runtime_sec_mean": reduced_runtime,
                "main_minus_best_reduced_hv": main_hv - reduced_hv
                if np.isfinite(main_hv) and np.isfinite(reduced_hv)
                else float("nan"),
                "main_minus_best_reduced_hv_pct": _delta_pct(main_hv, reduced_hv),
                "interpretation": interpretation,
            }
        )
    return out


def _metric_delta(row: dict[str, str]) -> float:
    baseline = _float(row.get("baseline_mean"))
    algorithm = _float(row.get("algorithm_mean"))
    if not np.isfinite(baseline) or not np.isfinite(algorithm):
        return float("nan")
    return algorithm - baseline


def _is_harmful(row: dict[str, str], alpha: float, effect_threshold: float) -> bool:
    metric = str(row.get("metric", ""))
    if metric not in HIGHER_IS_BETTER and metric not in LOWER_IS_BETTER:
        return False
    delta = _metric_delta(row)
    if not np.isfinite(delta):
        return False
    p_holm = _float(row.get("p_holm"))
    cliffs = abs(_float(row.get("cliffs_delta")))
    if not np.isfinite(p_holm) or p_holm > alpha or not np.isfinite(cliffs) or cliffs < effect_threshold:
        return False
    if metric in HIGHER_IS_BETTER:
        return delta < 0.0
    return delta > 0.0


def _decision_rows(
    pairwise_rows: list[dict[str, str]], summary_rows: list[dict[str, str]], alpha: float, effect_threshold: float
) -> list[dict[str, Any]]:
    rows_by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pairwise_rows:
        algorithm = str(row.get("algorithm", ""))
        if algorithm and algorithm != BASELINE and algorithm.startswith("CGPO_"):
            rows_by_variant[algorithm].append(row)

    summary_by_algorithm: dict[str, list[dict[str, str]]] = defaultdict(list)
    baseline_by_problem: dict[str, dict[str, str]] = {}
    for row in summary_rows:
        algorithm = str(row.get("algorithm", ""))
        summary_by_algorithm[algorithm].append(row)
        if algorithm == BASELINE:
            baseline_by_problem[str(row.get("problem", ""))] = row

    decisions: list[dict[str, Any]] = []
    for variant, rows in sorted(rows_by_variant.items()):
        harmful = [row for row in rows if _is_harmful(row, alpha, effect_threshold)]
        feasibility_harm = [row for row in harmful if row.get("metric") == "feasible_ratio"]
        hv_harm = [row for row in harmful if row.get("metric") == "hv"]
        igd_harm = [row for row in harmful if row.get("metric") == "igd_plus"]
        hard_fleet_harm = [
            row
            for row in harmful
            if _problem_fleet(str(row.get("problem", ""))) >= 3
            and row.get("metric") in {"feasible_ratio", "hv", "igd_plus", "mission_conflict_rate"}
        ]

        runtime_deltas = [
            _float(row.get("runtime_sec_median")) - _baseline_runtime(summary_rows, str(row.get("problem", "")))
            for row in summary_by_algorithm.get(variant, [])
        ]
        finite_runtime_deltas = [value for value in runtime_deltas if np.isfinite(value)]
        median_runtime_delta = float(np.median(finite_runtime_deltas)) if finite_runtime_deltas else float("nan")
        descriptive_harm = _descriptive_harm_count(
            summary_by_algorithm.get(variant, []),
            baseline_by_problem,
        )

        if descriptive_harm > 0 or feasibility_harm or len(hv_harm) >= 2 or hard_fleet_harm:
            recommendation = "keep"
        elif harmful:
            recommendation = "conditional"
        elif np.isfinite(median_runtime_delta) and median_runtime_delta < -0.01:
            recommendation = "remove_or_simplify"
        else:
            recommendation = "remove_if_simplicity_matters"

        decisions.append(
            {
                "variant": variant,
                "mechanism_removed": MECHANISM_BY_VARIANT.get(variant, variant.replace("CGPO_", "")),
                "expected_failure_mode": FAILURE_MODE_BY_VARIANT.get(variant, ""),
                "recommendation": recommendation,
                "harmful_case_count": len(harmful),
                "descriptive_harm_count": descriptive_harm,
                "feasibility_harm_count": len(feasibility_harm),
                "hv_harm_count": len(hv_harm),
                "igd_harm_count": len(igd_harm),
                "fleet_3_5_harm_count": len(hard_fleet_harm),
                "median_runtime_delta_sec": median_runtime_delta,
                "alpha": alpha,
                "effect_threshold_abs_cliffs_delta": effect_threshold,
            }
        )
    return decisions


def _descriptive_harm_count(variant_rows: list[dict[str, str]], baseline_by_problem: dict[str, dict[str, str]]) -> int:
    count = 0
    for row in variant_rows:
        problem = str(row.get("problem", ""))
        baseline = baseline_by_problem.get(problem)
        if baseline is None:
            continue
        base_feasible = _float(baseline.get("feasible_ratio_mean"))
        var_feasible = _float(row.get("feasible_ratio_mean"))
        if np.isfinite(base_feasible) and np.isfinite(var_feasible) and var_feasible < base_feasible - 0.25:
            count += 1
            continue
        base_hv = _float(baseline.get("hv_mean"))
        var_hv = _float(row.get("hv_mean"))
        if np.isfinite(base_hv) and np.isfinite(var_hv) and base_hv > 1e-9 and var_hv < 0.5 * base_hv:
            count += 1
            continue
        base_igd = _float(baseline.get("igd_plus_mean"))
        var_igd = _float(row.get("igd_plus_mean"))
        if np.isfinite(base_igd) and np.isfinite(var_igd) and var_igd > base_igd * 1.5:
            count += 1
    return count


def _trace_mean(run_dir: Path, trace_name: str) -> float:
    path = run_dir / f"rl_{trace_name}.mat"
    if not path.exists():
        return float("nan")
    try:
        data = load_mat(path)
        values = np.asarray(data.get(f"rl_{trace_name}", []), dtype=float).reshape(-1)
    except Exception:
        return float("nan")
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _trace_rows(results_dir: Path) -> list[dict[str, Any]]:
    # Trace fields emitted by the lean CGPO runner (cgpo.trace.CGPOTrace).
    traces = (
        # CIG diagnostics
        "cig_mean_tension",
        "cig_max_tension",
        "cig_terrain_edges",
        "cig_obstacle_edges",
        "cig_turn_edges",
        "cig_smoothing_edges",
        "cig_pairwise_edges",
        # PPF diagnostics
        "ppf_feasibility_pressure",
        "ppf_boundary_mass",
        "ppf_pressure_entropy",
        # OVO diagnostics
        "ovo_perturbation_scale",
        "ovo_coordinated_clusters",
        # Population diagnostics
        "offspring_feasible_ratio",
        # Retained as zero-valued compatibility diagnostics.
        "gfp_projection_norm",
        "gfp_violation_delta",
        "gfp_acceptance_rate",
    )
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(results_dir.glob("CGPO_*/*/Run_*")):
        if not run_dir.is_dir():
            continue
        row: dict[str, Any] = {
            "algorithm": run_dir.parent.parent.name,
            "problem": run_dir.parent.name,
            "run": run_dir.name.replace("Run_", ""),
        }
        for trace in traces:
            row[f"{trace}_mean"] = _trace_mean(run_dir, trace)
        rows.append(row)
    return rows


def _baseline_runtime(summary_rows: list[dict[str, str]], problem: str) -> float:
    for row in summary_rows:
        if row.get("algorithm") == BASELINE and row.get("problem") == problem:
            return _float(row.get("runtime_sec_median"))
    return float("nan")


def _write_manifest(
    results_dir: Path, output_dir: Path, args: argparse.Namespace, report_payload: dict[str, Any]
) -> None:
    benchmark_manifest = results_dir / "benchmark_manifest.json"
    command_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {
        "schemaVersion": 1,
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "resultsDir": str(results_dir.resolve()),
        "outputDir": str(output_dir.resolve()),
        "baselineAlgorithm": BASELINE,
        "commandArgs": command_args,
        "benchmarkManifest": json.loads(benchmark_manifest.read_text(encoding="utf-8"))
        if benchmark_manifest.exists()
        else None,
        "report": {key: str(value) for key, value in report_payload.items()},
    }
    (output_dir / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CGPO ablation benchmark outputs.")
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default="", type=Path)
    parser.add_argument("--hv-samples", default=2000, type=int)
    parser.add_argument("--max-runs", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument(
        "--effect-threshold",
        default=0.147,
        type=float,
        help="Absolute Cliff's delta threshold for non-negligible effects.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir if str(args.output_dir) else results_dir / "metrics").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_payload = generate_benchmark_report(
        ReportConfig(
            project_root=args.project_root.resolve(),
            results_dir=results_dir,
            output_dir=output_dir,
            hv_samples=int(args.hv_samples),
            max_runs=int(args.max_runs),
            baseline_algorithm=BASELINE,
            seed=int(args.seed),
        )
    )

    summary_csv = output_dir / "benchmark_metrics_summary.csv"
    pairwise_csv = output_dir / "pairwise_stats.csv"
    ablation_summary = output_dir / "ablation_metrics_summary.csv"
    ablation_pairwise = output_dir / "ablation_pairwise_stats.csv"
    if summary_csv.exists():
        shutil.copyfile(summary_csv, ablation_summary)
    if pairwise_csv.exists():
        shutil.copyfile(pairwise_csv, ablation_pairwise)

    summary_rows = _load_csv(ablation_summary)
    pairwise_rows = _load_csv(ablation_pairwise)
    decision_rows = _decision_rows(
        pairwise_rows, summary_rows, alpha=float(args.alpha), effect_threshold=float(args.effect_threshold)
    )
    _write_csv(output_dir / "ablation_decisions.csv", decision_rows)
    dominance_rows = _component_dominance_rows(summary_rows)
    _write_csv(output_dir / "component_dominance.csv", dominance_rows)

    runtime_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        runtime_rows.append(
            {
                "algorithm": row.get("algorithm", ""),
                "problem": row.get("problem", ""),
                "runs": row.get("runs", ""),
                "runtime_sec_mean": row.get("runtime_sec_mean", ""),
                "runtime_sec_std": row.get("runtime_sec_std", ""),
                "runtime_sec_median": row.get("runtime_sec_median", ""),
                "runtime_sec_iqr": row.get("runtime_sec_iqr", ""),
            }
        )
    _write_csv(output_dir / "ablation_runtime.csv", runtime_rows)
    trace_rows = _trace_rows(results_dir)
    _write_csv(output_dir / "cgpo_trace_summary.csv", trace_rows)
    _write_manifest(results_dir, output_dir, args, report_payload)

    print(f"ablation_summary={ablation_summary}")
    print(f"ablation_pairwise={ablation_pairwise}")
    print(f"ablation_decisions={output_dir / 'ablation_decisions.csv'}")
    print(f"component_dominance={output_dir / 'component_dominance.csv'}")
    print(f"ablation_runtime={output_dir / 'ablation_runtime.csv'}")
    print(f"cgpo_trace_summary={output_dir / 'cgpo_trace_summary.csv'}")


if __name__ == "__main__":
    main()
