from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.analysis.metrics.report import (  # noqa: E402
    _igd_plus,
    _load_feasible_mask,
    _load_popobj_raw,
)
from uav_benchmark.core.metrics import cal_metric  # noqa: E402
from uav_benchmark.io.matlab import load_mat  # noqa: E402

try:
    from scipy.stats import mannwhitneyu, wilcoxon  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mannwhitneyu = None
    wilcoxon = None


@dataclass(slots=True)
class CaseMetric:
    case: str
    problem: str
    run_id: int
    hv: float
    spread: float
    convergence_rate: float
    igd_plus: float
    feasible_ratio: float


_HIGHER_IS_BETTER = {"hv", "spread", "convergence_rate", "feasible_ratio"}


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


def _collect_snapshots(
    results_root: Path,
    *,
    case_names: tuple[str, ...],
    algorithm_name: str,
    max_runs: int,
) -> tuple[list[tuple[str, str, int, Path, np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    snapshots: list[tuple[str, str, int, Path, np.ndarray, np.ndarray]] = []
    feasible_pools: dict[str, list[np.ndarray]] = {}

    for case_name in case_names:
        algorithm_dir = results_root / case_name / algorithm_name
        if not algorithm_dir.exists():
            continue
        for problem_dir in sorted(algorithm_dir.iterdir()):
            if not problem_dir.is_dir() or problem_dir.name.startswith("."):
                continue
            run_dirs = [entry for entry in sorted(problem_dir.glob("Run_*")) if entry.is_dir()]
            if max_runs > 0:
                run_dirs = run_dirs[:max_runs]
            for run_dir in run_dirs:
                pop_obj = _load_popobj_raw(run_dir)
                feasible_mask = _load_feasible_mask(run_dir, pop_obj)
                feasible_obj = pop_obj[feasible_mask] if pop_obj.size else np.zeros((0, 4), dtype=float)
                if feasible_obj.size > 0:
                    feasible_pools.setdefault(problem_dir.name, []).append(feasible_obj)
                snapshots.append(
                    (
                        case_name,
                        problem_dir.name,
                        _run_index(run_dir),
                        run_dir,
                        pop_obj,
                        feasible_mask,
                    )
                )

    reference_fronts: dict[str, np.ndarray] = {}
    for problem_name, stacks in feasible_pools.items():
        merged = np.vstack(stacks) if stacks else np.zeros((0, 4), dtype=float)
        reference_fronts[problem_name] = _non_dominated(merged) if merged.size else np.zeros((0, 4), dtype=float)
    return snapshots, reference_fronts


def _build_reference_points(
    snapshots: list[tuple[str, str, int, Path, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    ref_points: dict[str, np.ndarray] = {}
    for _case_name, problem_name, _run_id, _run_dir, pop_obj, feasible_mask in snapshots:
        feasible_obj = pop_obj[feasible_mask] if pop_obj.size else np.zeros((0, 4), dtype=float)
        if feasible_obj.size == 0:
            continue
        max_values = np.max(feasible_obj, axis=0)
        if problem_name in ref_points:
            ref_points[problem_name] = np.maximum(ref_points[problem_name], max_values)
        else:
            ref_points[problem_name] = max_values
    for problem_name, reference in list(ref_points.items()):
        ref = np.asarray(reference, dtype=float) * 1.1
        ref[ref <= 0.0] = 1.0
        ref_points[problem_name] = ref
    return ref_points


def _collect_records(
    results_root: Path,
    *,
    case_names: tuple[str, ...],
    algorithm_name: str,
    max_runs: int,
    hv_samples: int,
) -> list[CaseMetric]:
    snapshots, reference_fronts = _collect_snapshots(
        results_root,
        case_names=case_names,
        algorithm_name=algorithm_name,
        max_runs=max_runs,
    )
    ref_points = _build_reference_points(snapshots)
    records: list[CaseMetric] = []

    for case_name, problem_name, run_id, run_dir, pop_obj, feasible_mask in snapshots:
        total_count = int(pop_obj.shape[0]) if pop_obj.ndim == 2 else 0
        feasible_obj = pop_obj[feasible_mask] if pop_obj.size else np.zeros((0, 4), dtype=float)
        feasible_count = int(feasible_obj.shape[0])
        feasible_ratio = float(feasible_count / total_count) if total_count > 0 else 0.0

        objective_count = feasible_obj.shape[1] if feasible_obj.size else 4
        reference_point = ref_points.get(problem_name)
        hv = cal_metric(1, feasible_obj, 0, objective_count, hv_samples, reference_point) if feasible_obj.size else 0.0
        spread = cal_metric(2, feasible_obj, 0, objective_count) if feasible_obj.size else 0.0
        convergence_rate = _load_convergence_rate(run_dir)
        igd_plus = _igd_plus(feasible_obj, reference_fronts.get(problem_name, np.zeros((0, 4), dtype=float)))

        records.append(
            CaseMetric(
                case=case_name,
                problem=problem_name,
                run_id=run_id,
                hv=float(hv),
                spread=float(spread),
                convergence_rate=float(convergence_rate),
                igd_plus=float(igd_plus),
                feasible_ratio=float(feasible_ratio),
            )
        )
    return records


def _metric_vector(records: list[CaseMetric], metric_name: str) -> np.ndarray:
    values = np.asarray([float(getattr(record, metric_name)) for record in records], dtype=float)
    return values[np.isfinite(values)]


def _paired_vectors(
    left: list[CaseMetric],
    right: list[CaseMetric],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_by_key = {
        (record.problem, int(record.run_id)): float(getattr(record, metric_name))
        for record in left
        if np.isfinite(float(getattr(record, metric_name)))
    }
    right_by_key = {
        (record.problem, int(record.run_id)): float(getattr(record, metric_name))
        for record in right
        if np.isfinite(float(getattr(record, metric_name)))
    }
    common = sorted(set(left_by_key.keys()) & set(right_by_key.keys()))
    if not common:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    left_vec = np.asarray([left_by_key[key] for key in common], dtype=float)
    right_vec = np.asarray([right_by_key[key] for key in common], dtype=float)
    return left_vec, right_vec


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
        for (idx, _), p_holm in zip(valid, adjusted, strict=False):
            rows[idx]["p_holm"] = float(p_holm)


def _compare_vectors(left: np.ndarray, right: np.ndarray) -> tuple[str, float]:
    if left.size == 0 or right.size == 0:
        return "none", float("nan")
    if wilcoxon is not None and left.size == right.size and left.size > 0:
        if np.allclose(left, right, equal_nan=False):
            return "wilcoxon_signed_rank", 1.0
        try:
            result = wilcoxon(left, right, zero_method="wilcox", alternative="two-sided")
            return (
                "wilcoxon_signed_rank",
                float(getattr(result, "pvalue", float("nan"))),
            )
        except Exception:
            pass
    if mannwhitneyu is not None:
        try:
            result = mannwhitneyu(left, right, alternative="two-sided")
            return (
                "mann_whitney_u",
                float(getattr(result, "pvalue", float("nan"))),
            )
        except Exception:
            pass
    return "none", float("nan")


def _is_control_better(metric_name: str, control_mean: float, case_mean: float) -> bool:
    if metric_name in _HIGHER_IS_BETTER:
        return bool(control_mean > case_mean)
    return bool(control_mean < case_mean)


def _pairwise_rows(
    records: list[CaseMetric],
    *,
    control_case: str,
    metrics: tuple[str, ...],
    alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = sorted({record.case for record in records if record.case != control_case})
    problems = sorted({record.problem for record in records})

    for case_name in cases:
        for metric_name in metrics:
            control_pool = [record for record in records if record.case == control_case]
            case_pool = [record for record in records if record.case == case_name]
            left_all, right_all = _paired_vectors(control_pool, case_pool, metric_name)
            test_name, p_value = _compare_vectors(left_all, right_all)
            control_mean = float(np.mean(left_all)) if left_all.size > 0 else float("nan")
            case_mean = float(np.mean(right_all)) if right_all.size > 0 else float("nan")
            rows.append(
                {
                    "scope": "pooled",
                    "problem": "ALL",
                    "metric": metric_name,
                    "control": control_case,
                    "case": case_name,
                    "n_control": int(left_all.size),
                    "n_case": int(right_all.size),
                    "control_mean": control_mean,
                    "case_mean": case_mean,
                    "test_name": test_name,
                    "p_value": p_value,
                    "p_holm": float("nan"),
                    "cliffs_delta": _cliffs_delta(left_all, right_all),
                    "control_better": _is_control_better(metric_name, control_mean, case_mean)
                    if np.isfinite(control_mean) and np.isfinite(case_mean)
                    else False,
                    "significant": False,
                }
            )
            for problem_name in problems:
                control_records = [
                    record for record in records if record.case == control_case and record.problem == problem_name
                ]
                case_records = [
                    record for record in records if record.case == case_name and record.problem == problem_name
                ]
                left, right = _paired_vectors(control_records, case_records, metric_name)
                test_name, p_value = _compare_vectors(left, right)
                control_mean = float(np.mean(left)) if left.size > 0 else float("nan")
                case_mean = float(np.mean(right)) if right.size > 0 else float("nan")
                rows.append(
                    {
                        "scope": "problem",
                        "problem": problem_name,
                        "metric": metric_name,
                        "control": control_case,
                        "case": case_name,
                        "n_control": int(left.size),
                        "n_case": int(right.size),
                        "control_mean": control_mean,
                        "case_mean": case_mean,
                        "test_name": test_name,
                        "p_value": p_value,
                        "p_holm": float("nan"),
                        "cliffs_delta": _cliffs_delta(left, right),
                        "control_better": _is_control_better(metric_name, control_mean, case_mean)
                        if np.isfinite(control_mean) and np.isfinite(case_mean)
                        else False,
                        "significant": False,
                    }
                )

    _holm_adjust(rows, ("scope", "metric"))
    for row in rows:
        p_effective = float(row.get("p_holm", row.get("p_value", float("nan"))))
        row["significant"] = bool(
            np.isfinite(p_effective) and p_effective < alpha and bool(row.get("control_better", False))
        )
    return rows


def _summary_rows(records: list[CaseMetric], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name in sorted({record.case for record in records}):
        for problem_name in sorted({record.problem for record in records}):
            subset = [record for record in records if record.case == case_name and record.problem == problem_name]
            if not subset:
                continue
            row: dict[str, Any] = {
                "case": case_name,
                "problem": problem_name,
                "runs": int(len(subset)),
            }
            for metric_name in metrics:
                values = _metric_vector(subset, metric_name)
                row[f"{metric_name}_mean"] = float(np.mean(values)) if values.size > 0 else float("nan")
                row[f"{metric_name}_std"] = float(np.std(values)) if values.size > 0 else float("nan")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="LOO significance analysis for MOGWO component ablations.")
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--algorithm-name", default="MOGWO", type=str)
    parser.add_argument("--control-case", default="full", type=str)
    parser.add_argument("--cases", default="", type=str, help="Comma-separated subset of case names.")
    parser.add_argument("--metrics", default="hv,spread,convergence_rate,igd_plus", type=str)
    parser.add_argument("--max-runs", default=30, type=int)
    parser.add_argument("--hv-samples", default=2000, type=int)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    if not results_root.exists():
        raise SystemExit(f"Results root not found: {results_root}")

    if args.cases.strip():
        case_names = tuple(item.strip() for item in args.cases.split(",") if item.strip())
    else:
        case_names = tuple(
            entry.name for entry in sorted(results_root.iterdir()) if entry.is_dir() and not entry.name.startswith(".")
        )
    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else (results_root / "metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _collect_records(
        results_root,
        case_names=case_names,
        algorithm_name=str(args.algorithm_name),
        max_runs=int(args.max_runs),
        hv_samples=int(args.hv_samples),
    )
    if not records:
        raise SystemExit("No ablation records found.")

    summary_rows = _summary_rows(records, metrics)
    pairwise_rows = _pairwise_rows(
        records,
        control_case=str(args.control_case),
        metrics=metrics,
        alpha=float(args.alpha),
    )

    pooled_rows = [row for row in pairwise_rows if row["scope"] == "pooled" and row["case"] != str(args.control_case)]
    loo_pass = bool(pooled_rows and all(bool(row.get("significant", False)) for row in pooled_rows))

    summary_csv = output_dir / "ablation_summary.csv"
    pairwise_csv = output_dir / "ablation_pairwise.csv"
    summary_json = output_dir / "ablation_significance.json"

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["case", "problem", "runs"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with pairwise_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            list(pairwise_rows[0].keys())
            if pairwise_rows
            else [
                "scope",
                "problem",
                "metric",
                "control",
                "case",
                "n_control",
                "n_case",
                "control_mean",
                "case_mean",
                "test_name",
                "p_value",
                "p_holm",
                "cliffs_delta",
                "control_better",
                "significant",
            ]
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    payload = {
        "resultsRoot": str(results_root),
        "algorithmName": str(args.algorithm_name),
        "controlCase": str(args.control_case),
        "cases": list(case_names),
        "metrics": list(metrics),
        "alpha": float(args.alpha),
        "summaryCsv": str(summary_csv),
        "pairwiseCsv": str(pairwise_csv),
        "looPassAllMetrics": loo_pass,
        "pooledComparisons": pooled_rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"summary_csv: {summary_csv}")
    print(f"pairwise_csv: {pairwise_csv}")
    print(f"summary_json: {summary_json}")
    print(f"loo_pass_all_metrics: {loo_pass}")
    for row in pooled_rows:
        p_eff = row["p_holm"] if np.isfinite(float(row["p_holm"])) else row["p_value"]
        print(
            f"{row['metric']}: full vs {row['case']} "
            f"mean={row['control_mean']:.6f} vs {row['case_mean']:.6f}, "
            f"p={p_eff:.6g}, significant={row['significant']}"
        )


if __name__ == "__main__":
    main()
