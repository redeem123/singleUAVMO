from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.io.matlab import load_mat  # noqa: E402


DEFAULT_STAGE_KEYS = ("evalTimeSec", "updateTimeSec", "archiveTimeSec", "atlasTimeSec")


def _run_index(run_dir: Path) -> int:
    parts = run_dir.name.split("_", 1)
    if len(parts) != 2:
        return 0
    try:
        return int(parts[1])
    except Exception:
        return 0


def _safe_scalar(value: Any) -> float:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 0:
            return float("nan")
        return float(array[0])
    except Exception:
        return float("nan")


def _iter_run_dirs(results_dir: Path):
    for algorithm_dir in sorted(results_dir.iterdir()):
        if not algorithm_dir.is_dir() or algorithm_dir.name.startswith(".") or algorithm_dir.name == "Plots":
            continue
        for problem_dir in sorted(algorithm_dir.iterdir()):
            if not problem_dir.is_dir() or problem_dir.name.startswith("."):
                continue
            for run_dir in sorted(problem_dir.glob("Run_*")):
                if run_dir.is_dir():
                    yield algorithm_dir.name, problem_dir.name, run_dir


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


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def _median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate runtime complexity breakdown from run_stats.mat files.")
    parser.add_argument("--results-dir", required=True, type=str)
    parser.add_argument("--output-dir", default="", type=str)
    parser.add_argument("--stage-keys", default=",".join(DEFAULT_STAGE_KEYS), type=str)
    parser.add_argument("--algorithms", default="", type=str, help="Optional comma-separated algorithm filter")
    parser.add_argument("--problems", default="", type=str, help="Optional comma-separated problem filter")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (results_dir / "metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_keys = tuple(key.strip() for key in str(args.stage_keys).split(",") if key.strip())
    algorithm_filter = {item.strip() for item in str(args.algorithms).split(",") if item.strip()}
    problem_filter = {item.strip() for item in str(args.problems).split(",") if item.strip()}

    detailed_rows: list[dict[str, Any]] = []
    nonempty_stage_hits = {key: 0 for key in stage_keys}

    for algorithm_name, problem_name, run_dir in _iter_run_dirs(results_dir):
        if algorithm_filter and algorithm_name not in algorithm_filter:
            continue
        if problem_filter and problem_name not in problem_filter:
            continue

        stats_file = run_dir / "run_stats.mat"
        if not stats_file.exists():
            continue

        try:
            payload = load_mat(stats_file)
        except Exception:
            continue

        runtime = _safe_scalar(payload.get("runtimeSec", float("nan")))
        row: dict[str, Any] = {
            "algorithm": algorithm_name,
            "problem": problem_name,
            "run_id": int(_run_index(run_dir)),
            "runtimeSec": float(runtime),
        }
        for key in stage_keys:
            value = _safe_scalar(payload.get(key, float("nan")))
            row[key] = float(value)
            if np.isfinite(value):
                nonempty_stage_hits[key] += 1
            if np.isfinite(value) and np.isfinite(runtime) and runtime > 0:
                row[f"{key}_pct"] = float(100.0 * value / runtime)
            else:
                row[f"{key}_pct"] = float("nan")
        detailed_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in detailed_rows:
        groups.setdefault((str(row["algorithm"]), str(row["problem"])), []).append(row)

    for (algorithm_name, problem_name), rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "algorithm": algorithm_name,
            "problem": problem_name,
            "runs": int(len(rows)),
            "runtimeSec_mean": _mean([float(row["runtimeSec"]) for row in rows]),
            "runtimeSec_std": _std([float(row["runtimeSec"]) for row in rows]),
            "runtimeSec_median": _median([float(row["runtimeSec"]) for row in rows]),
        }
        for key in stage_keys:
            values = [float(row[key]) for row in rows]
            pcts = [float(row[f"{key}_pct"]) for row in rows]
            summary[f"{key}_mean"] = _mean(values)
            summary[f"{key}_std"] = _std(values)
            summary[f"{key}_pct_mean"] = _mean(pcts)
            summary[f"{key}_pct_std"] = _std(pcts)
        summary_rows.append(summary)

    detailed_csv = output_dir / "runtime_breakdown_detailed.csv"
    summary_csv = output_dir / "runtime_breakdown_summary.csv"
    json_path = output_dir / "runtime_breakdown.json"

    _write_csv(detailed_csv, detailed_rows)
    _write_csv(summary_csv, summary_rows)

    payload = {
        "resultsDir": str(results_dir),
        "stageKeys": list(stage_keys),
        "rowsDetailed": int(len(detailed_rows)),
        "rowsSummary": int(len(summary_rows)),
        "stageHits": {key: int(value) for key, value in nonempty_stage_hits.items()},
        "formula": "T_gen = T_eval + T_update + T_archive + T_atlas",
        "notes": [
            "Percent columns are computed as stage/runtimeSec * 100.",
            "If stage keys are absent in run_stats.mat, stage percentages remain NaN.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(detailed_csv)
    print(summary_csv)
    print(json_path)


if __name__ == "__main__":
    main()
