from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.analysis.metrics.report import (  # noqa: E402
    _igd_plus,
    _load_feasible_mask,
    _load_mission_metric,
    _load_popobj_raw,
    _load_runtime,
)
from uav_benchmark.analysis.metrics.compute import _build_ref_points  # noqa: E402
from uav_benchmark.core.metrics import cal_metric  # noqa: E402
from uav_benchmark.io.matlab import load_mat  # noqa: E402

try:
    from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    friedmanchisquare = None
    mannwhitneyu = None
    wilcoxon = None


@dataclass(slots=True)
class RunMetric:
    algorithm: str
    problem: str
    run_id: int
    hv: float
    spread: float
    convergence_rate: float
    igd_plus: float
    feasible_ratio: float
    runtime_sec: float
    mission_conflict_rate: float
    mission_makespan: float
    mission_energy: float


def _run_index(run_dir: Path) -> int:
    parts = run_dir.name.split("_", 1)
    if len(parts) != 2:
        return 0
    try:
        return int(parts[1])
    except Exception:
        return 0


def _non_dominated(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    unique = np.unique(points, axis=0)
    keep = np.ones(unique.shape[0], dtype=bool)
    for i in range(unique.shape[0]):
        if not keep[i]:
            continue
        for j in range(unique.shape[0]):
            if i == j or not keep[j]:
                continue
            if np.all(unique[j] <= unique[i]) and np.any(unique[j] < unique[i]):
                keep[i] = False
                break
    return unique[keep]


def _cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    wins = 0
    losses = 0
    for value_left in left:
        wins += int(np.sum(value_left > right))
        losses += int(np.sum(value_left < right))
    total = left.size * right.size
    return float((wins - losses) / total) if total > 0 else float("nan")


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _metric_value(record: RunMetric, metric_name: str) -> float:
    return float(getattr(record, metric_name))


def _load_convergence_rate(run_dir: Path) -> float:
    gen_hv_path = run_dir / "gen_hv.mat"
    if not gen_hv_path.exists():
        return float("nan")
    try:
        payload = load_mat(gen_hv_path)
    except Exception:
        return float("nan")
    curve = np.asarray(payload.get("gen_hv", np.zeros((0, 2), dtype=float)), dtype=float)
    if curve.ndim != 2 or curve.shape[0] == 0:
        return float("nan")
    hv_curve = np.asarray(curve[:, 0], dtype=float).reshape(-1)
    hv_curve = hv_curve[np.isfinite(hv_curve)]
    if hv_curve.size == 0:
        return float("nan")
    hv_curve = np.maximum.accumulate(hv_curve)
    final_hv = float(np.max(hv_curve))
    if final_hv <= 0.0:
        return 0.0
    threshold = 0.90 * final_hv
    reached = np.where(hv_curve >= threshold)[0]
    gen_90 = int(reached[0]) + 1 if reached.size > 0 else int(hv_curve.size)
    return float(final_hv / float(max(1, gen_90)))


def _collect_records(results_dir: Path, hv_samples: int, max_runs: int) -> list[RunMetric]:
    ref_points = _build_ref_points(results_dir)
    snapshots: list[tuple[str, str, int, Path, np.ndarray, np.ndarray]] = []
    feasible_pools: dict[str, list[np.ndarray]] = {}

    for algorithm_dir in sorted(results_dir.iterdir()):
        if not algorithm_dir.is_dir() or algorithm_dir.name.startswith(".") or algorithm_dir.name == "Plots":
            continue
        algorithm_name = algorithm_dir.name
        for problem_dir in sorted(algorithm_dir.iterdir()):
            if not problem_dir.is_dir() or problem_dir.name.startswith("."):
                continue
            run_dirs = [entry for entry in sorted(problem_dir.glob("Run_*")) if entry.is_dir()]
            if max_runs > 0:
                run_dirs = run_dirs[:max_runs]
            for run_dir in run_dirs:
                run_id = _run_index(run_dir)
                pop_obj = _load_popobj_raw(run_dir)
                feasible_mask = _load_feasible_mask(run_dir, pop_obj)
                feasible_obj = pop_obj[feasible_mask] if pop_obj.size else np.zeros((0, 4), dtype=float)
                if feasible_obj.size > 0:
                    feasible_pools.setdefault(problem_dir.name, []).append(feasible_obj)
                snapshots.append((algorithm_name, problem_dir.name, run_id, run_dir, pop_obj, feasible_mask))

    reference_fronts: dict[str, np.ndarray] = {}
    for problem_name, stacks in feasible_pools.items():
        merged = np.vstack(stacks) if stacks else np.zeros((0, 4), dtype=float)
        reference_fronts[problem_name] = _non_dominated(merged) if merged.size else np.zeros((0, 4), dtype=float)

    records: list[RunMetric] = []
    for algorithm_name, problem_name, run_id, run_dir, pop_obj, feasible_mask in snapshots:
        total_count = int(pop_obj.shape[0]) if pop_obj.ndim == 2 else 0
        feasible_obj = pop_obj[feasible_mask] if pop_obj.size else np.zeros((0, 4), dtype=float)
        feasible_count = int(feasible_obj.shape[0])
        feasible_ratio = float(feasible_count / total_count) if total_count > 0 else 0.0

        objective_count = feasible_obj.shape[1] if feasible_obj.size else 4
        reference_point = ref_points.get(problem_name)
        hv = (
            cal_metric(1, feasible_obj, 0, objective_count, hv_samples, reference_point)
            if feasible_obj.size
            else 0.0
        )
        spread = cal_metric(2, feasible_obj, 0, objective_count) if feasible_obj.size else 0.0
        convergence_rate = _load_convergence_rate(run_dir)
        igd_plus = _igd_plus(feasible_obj, reference_fronts.get(problem_name, np.zeros((0, 4), dtype=float)))
        runtime_sec = _load_runtime(run_dir)
        mission_conflict_rate = _load_mission_metric(run_dir, "conflictRate")
        mission_makespan = _load_mission_metric(run_dir, "makespan")
        mission_energy = _load_mission_metric(run_dir, "energy")

        records.append(
            RunMetric(
                algorithm=algorithm_name,
                problem=problem_name,
                run_id=run_id,
                hv=float(hv),
                spread=float(spread),
                convergence_rate=float(convergence_rate),
                igd_plus=float(igd_plus),
                feasible_ratio=float(feasible_ratio),
                runtime_sec=float(runtime_sec),
                mission_conflict_rate=float(mission_conflict_rate),
                mission_makespan=float(mission_makespan),
                mission_energy=float(mission_energy),
            )
        )
    return records


def _paired_vectors(
    left: list[RunMetric],
    right: list[RunMetric],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_by_run: dict[int, float] = {}
    right_by_run: dict[int, float] = {}
    for record in left:
        value = _metric_value(record, metric_name)
        if np.isfinite(value):
            left_by_run[int(record.run_id)] = value
    for record in right:
        value = _metric_value(record, metric_name)
        if np.isfinite(value):
            right_by_run[int(record.run_id)] = value
    common = sorted(set(left_by_run.keys()) & set(right_by_run.keys()))
    if not common:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    left_vec = np.asarray([left_by_run[key] for key in common], dtype=float)
    right_vec = np.asarray([right_by_run[key] for key in common], dtype=float)
    return left_vec, right_vec


def _unpaired_vectors(
    left: list[RunMetric],
    right: list[RunMetric],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_vec = np.asarray([_metric_value(record, metric_name) for record in left], dtype=float)
    right_vec = np.asarray([_metric_value(record, metric_name) for record in right], dtype=float)
    left_vec = left_vec[np.isfinite(left_vec)]
    right_vec = right_vec[np.isfinite(right_vec)]
    return left_vec, right_vec


def _holm_adjust(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> None:
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx, row in enumerate(rows):
        key = tuple(row.get(group_key) for group_key in group_keys)
        groups.setdefault(key, []).append(idx)
    for indices in groups.values():
        valid = []
        for idx in indices:
            p_value = float(rows[idx].get("p_value", float("nan")))
            if np.isfinite(p_value):
                valid.append((idx, p_value))
        if not valid:
            continue
        valid.sort(key=lambda item: item[1])
        m = len(valid)
        adjusted = [min(1.0, (m - rank) * p_value) for rank, (_idx, p_value) in enumerate(valid)]
        for rank in range(1, len(adjusted)):
            adjusted[rank] = max(adjusted[rank], adjusted[rank - 1])
        for (idx, _), p_holm in zip(valid, adjusted):
            rows[idx]["p_holm"] = float(p_holm)


def _pairwise_rows(
    records: list[RunMetric],
    control_algorithm: str,
    metrics: list[str],
    pairwise_mode: str,
    min_paired: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    problems = sorted({record.problem for record in records})
    for problem in problems:
        control = [
            record for record in records
            if record.problem == problem and record.algorithm == control_algorithm
        ]
        if not control:
            continue
        competitors = sorted({
            record.algorithm for record in records
            if record.problem == problem and record.algorithm != control_algorithm
        })
        for metric_name in metrics:
            for competitor_name in competitors:
                competitor = [
                    record for record in records
                    if record.problem == problem and record.algorithm == competitor_name
                ]
                paired_left, paired_right = _paired_vectors(control, competitor, metric_name)
                unpaired_left, unpaired_right = _unpaired_vectors(control, competitor, metric_name)

                use_paired = False
                if pairwise_mode == "paired":
                    use_paired = True
                elif pairwise_mode == "unpaired":
                    use_paired = False
                else:
                    use_paired = paired_left.size >= min_paired

                if use_paired:
                    left, right = paired_left, paired_right
                    test_name = "wilcoxon_signed_rank"
                    if wilcoxon is not None and left.size > 0 and right.size > 0:
                        try:
                            p_value = float(wilcoxon(left, right, zero_method="wilcox", alternative="two-sided").pvalue)
                        except Exception:
                            p_value = float("nan")
                    else:
                        p_value = float("nan")
                else:
                    left, right = unpaired_left, unpaired_right
                    test_name = "mann_whitney_u"
                    if mannwhitneyu is not None and left.size > 0 and right.size > 0:
                        try:
                            p_value = float(mannwhitneyu(left, right, alternative="two-sided").pvalue)
                        except Exception:
                            p_value = float("nan")
                    else:
                        p_value = float("nan")

                rows.append(
                    {
                        "problem": problem,
                        "metric": metric_name,
                        "control": control_algorithm,
                        "algorithm": competitor_name,
                        "test": test_name,
                        "paired": float(1.0 if use_paired else 0.0),
                        "n_control_raw": int(unpaired_left.size),
                        "n_algorithm_raw": int(unpaired_right.size),
                        "n_effective": int(min(left.size, right.size) if use_paired else (left.size + right.size)),
                        "control_mean": _mean(left),
                        "algorithm_mean": _mean(right),
                        "delta_mean_algorithm_minus_control": _mean(right) - _mean(left),
                        "cliffs_delta_algorithm_vs_control": _cliffs_delta(right, left),
                        "p_value": float(p_value),
                        "p_holm": float("nan"),
                    }
                )
    _holm_adjust(rows, group_keys=("problem", "metric"))
    return rows


def _friedman_rows(
    records: list[RunMetric],
    control_algorithm: str,
    metrics: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []
    all_algorithms = sorted({record.algorithm for record in records})

    for metric_name in metrics:
        block_values: dict[tuple[str, int], dict[str, float]] = {}
        for record in records:
            value = _metric_value(record, metric_name)
            if not np.isfinite(value):
                continue
            key = (record.problem, int(record.run_id))
            block_values.setdefault(key, {})[record.algorithm] = value

        blocks: list[list[float]] = []
        for key in sorted(block_values.keys()):
            row = block_values[key]
            if all(algorithm in row for algorithm in all_algorithms):
                blocks.append([float(row[algorithm]) for algorithm in all_algorithms])
        matrix = np.asarray(blocks, dtype=float)

        if matrix.shape[0] >= 2 and len(all_algorithms) >= 3 and friedmanchisquare is not None:
            try:
                statistic, p_value = friedmanchisquare(*[matrix[:, idx] for idx in range(matrix.shape[1])])
                statistic_f = float(statistic)
                p_value_f = float(p_value)
            except Exception:
                statistic_f = float("nan")
                p_value_f = float("nan")
        else:
            statistic_f = float("nan")
            p_value_f = float("nan")

        summary_rows.append(
            {
                "metric": metric_name,
                "n_algorithms": int(len(all_algorithms)),
                "n_blocks": int(matrix.shape[0]),
                "friedman_statistic": statistic_f,
                "p_value": p_value_f,
            }
        )

        if control_algorithm not in all_algorithms or matrix.size == 0:
            continue
        control_idx = all_algorithms.index(control_algorithm)
        control_vec = matrix[:, control_idx]
        for algorithm_idx, algorithm_name in enumerate(all_algorithms):
            if algorithm_name == control_algorithm:
                continue
            comp_vec = matrix[:, algorithm_idx]
            if wilcoxon is not None and control_vec.size > 0 and comp_vec.size > 0:
                try:
                    post_p = float(wilcoxon(control_vec, comp_vec, zero_method="wilcox", alternative="two-sided").pvalue)
                except Exception:
                    post_p = float("nan")
            else:
                post_p = float("nan")
            posthoc_rows.append(
                {
                    "metric": metric_name,
                    "control": control_algorithm,
                    "algorithm": algorithm_name,
                    "n_blocks": int(control_vec.size),
                    "control_mean": _mean(control_vec),
                    "algorithm_mean": _mean(comp_vec),
                    "delta_mean_algorithm_minus_control": _mean(comp_vec) - _mean(control_vec),
                    "cliffs_delta_algorithm_vs_control": _cliffs_delta(comp_vec, control_vec),
                    "p_value": float(post_p),
                    "p_holm": float("nan"),
                }
            )

    _holm_adjust(posthoc_rows, group_keys=("metric",))
    return summary_rows, posthoc_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_metric_list(raw: str) -> list[str]:
    default_metrics = [
        "hv",
        "spread",
        "convergence_rate",
        "igd_plus",
        "feasible_ratio",
        "runtime_sec",
        "mission_conflict_rate",
        "mission_makespan",
        "mission_energy",
    ]
    if not raw.strip():
        return default_metrics
    values = [item.strip() for item in raw.split(",") if item.strip()]
    valid = set(default_metrics)
    return [item for item in values if item in valid]


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication-grade significance testing on benchmark results.")
    parser.add_argument("--results-dir", required=True, type=str)
    parser.add_argument("--output-dir", default="", type=str)
    parser.add_argument("--control-algorithm", default="MOGWO", type=str)
    parser.add_argument("--pairwise-mode", choices=("auto", "paired", "unpaired"), default="auto", type=str)
    parser.add_argument("--min-paired", default=3, type=int)
    parser.add_argument("--metrics", default="", type=str)
    parser.add_argument("--hv-samples", default=2000, type=int)
    parser.add_argument("--max-runs", default=0, type=int)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (results_dir / "metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _parse_metric_list(str(args.metrics))
    if not metrics:
        raise RuntimeError("No valid metrics selected.")

    records = _collect_records(results_dir=results_dir, hv_samples=int(args.hv_samples), max_runs=int(args.max_runs))
    pairwise = _pairwise_rows(
        records=records,
        control_algorithm=str(args.control_algorithm),
        metrics=metrics,
        pairwise_mode=str(args.pairwise_mode),
        min_paired=int(max(1, args.min_paired)),
    )
    friedman_summary, friedman_posthoc = _friedman_rows(
        records=records,
        control_algorithm=str(args.control_algorithm),
        metrics=metrics,
    )

    pairwise_csv = output_dir / "publication_pairwise_stats.csv"
    friedman_csv = output_dir / "publication_friedman.csv"
    friedman_posthoc_csv = output_dir / "publication_friedman_posthoc.csv"
    json_path = output_dir / "publication_stats.json"

    _write_csv(pairwise_csv, pairwise)
    _write_csv(friedman_csv, friedman_summary)
    _write_csv(friedman_posthoc_csv, friedman_posthoc)

    payload = {
        "resultsDir": str(results_dir),
        "controlAlgorithm": str(args.control_algorithm),
        "pairwiseMode": str(args.pairwise_mode),
        "metrics": metrics,
        "records": len(records),
        "pairwiseRows": pairwise,
        "friedmanRows": friedman_summary,
        "friedmanPosthocRows": friedman_posthoc,
        "scipyAvailable": bool(wilcoxon is not None and mannwhitneyu is not None and friedmanchisquare is not None),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(pairwise_csv)
    print(friedman_csv)
    print(friedman_posthoc_csv)
    print(json_path)


if __name__ == "__main__":
    main()
