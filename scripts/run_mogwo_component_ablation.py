from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CASE_EXTRAS: dict[str, dict[str, Any]] = {
    # ── Tier 1: Anchors ──────────────────────────────────────────
    "full": {"mogwoVariant": "full"},
    "pure_gwo": {"mogwoVariant": "standard_gwo"},
    # ── Tier 2: LOO (Leave One Out from Full) ────────────────────
    "loo_repair_restart": {"mogwoVariant": "full", "mogwoUseRepairRestart": False},
    "loo_attention": {"mogwoVariant": "full", "mogwoUseAdaptiveAttention": False},
    "loo_archive_explorer": {"mogwoVariant": "full", "mogwoUseDualArchiveExplorer": False},
}


def _runner_registered() -> bool:
    try:
        from uav_benchmark.algorithms import ALL_REGISTRY
    except ImportError:
        return False
    return "MOGWO" in ALL_REGISTRY


def _coerce_names(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(value for item in raw if (value := str(item).strip()))
    return ()


def _select_cases(raw: str) -> list[str]:
    if not raw.strip():
        return list(CASE_EXTRAS.keys())
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in CASE_EXTRAS]
    if unknown:
        raise RuntimeError(f"Unknown ablation case(s): {', '.join(unknown)}")
    return selected


def _resolved_fleet_sizes(params: Any) -> tuple[int, ...]:
    raw = getattr(params, "fleet_sizes", ())
    fleet_sizes = tuple(int(item) for item in raw if int(item) >= 1)
    if fleet_sizes:
        return fleet_sizes
    return (max(1, int(getattr(params, "fleet_size", 1))),)


def _resolved_problem_names(params: Any) -> tuple[str, ...]:
    extra = getattr(params, "extra", {})
    if not isinstance(extra, dict):
        return ()
    nested_extra = extra.get("extra")
    if isinstance(nested_extra, dict) and "problemNames" in nested_extra:
        return _coerce_names(nested_extra.get("problemNames"))
    return _coerce_names(extra.get("problemNames"))


def _expected_final_popobj_count(params: Any) -> int:
    problem_names = _resolved_problem_names(params)
    fleet_sizes = _resolved_fleet_sizes(params)
    if not problem_names or not fleet_sizes:
        return 0
    return len(problem_names) * len(fleet_sizes) * max(1, int(getattr(params, "runs", 1)))


def _completed_final_popobj_count(case_dir: Path) -> int:
    if not case_dir.exists():
        return 0
    return sum(1 for _ in case_dir.glob("**/final_popobj.mat"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair MOGWO component ablation cases.")
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "full_benchmark.yaml"), type=str)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results" / "mogwo_component_ablation"), type=str)
    parser.add_argument(
        "--cases",
        default="",
        type=str,
        help="Comma-separated subset of ablation cases (e.g. full,pure_gwo,loo_repair_restart,...). Leave empty for all.",
    )
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume completed per-run artifacts when present. Disable only for intentional full reruns.",
    )
    parser.add_argument(
        "--skip-complete-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a case entirely when the expected final_popobj count is already present.",
    )
    args = parser.parse_args()

    try:
        from uav_benchmark.benchmark import run_benchmark  # type: ignore
        from uav_benchmark.cli import _load_protocol  # type: ignore
        from uav_benchmark.config import BenchmarkParams  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Unable to import benchmark pipeline in this code snapshot. Restore missing benchmark dependencies first."
        ) from exc

    if not _runner_registered():
        raise RuntimeError(
            "MOGWO runner is not registered in this code snapshot. "
            "Restore/add uav_benchmark.algorithms MOGWO implementation first."
        )

    protocol = _load_protocol(Path(args.protocol).expanduser().resolve())
    base_params = BenchmarkParams.from_mapping(protocol)
    base_params.mode = "fleet"
    base_params.results_dir = Path(args.results_root).expanduser().resolve()
    base_params.extra = dict(base_params.extra)
    base_params.extra["algorithms"] = ["MOGWO"]
    base_params.extra["allowExperimentalAlgorithms"] = True

    selected_cases = _select_cases(str(args.cases))
    resume_existing = bool(args.resume_existing)
    for case_name in selected_cases:
        case_params = replace(base_params)
        case_params.results_dir = base_params.results_dir / case_name
        case_params.extra = dict(base_params.extra)
        case_params.extra.update(CASE_EXTRAS[case_name])
        case_params.extra["resumeExistingRuns"] = resume_existing

        expected_count = _expected_final_popobj_count(case_params)
        completed_count = _completed_final_popobj_count(case_params.results_dir)
        print(
            f"[ABLT] {case_name} -> {case_params.results_dir} "
            f"(completed={completed_count}/{expected_count}, resumeExistingRuns={resume_existing})"
        )
        if args.skip_complete_cases and expected_count > 0 and completed_count >= expected_count:
            print(f"[SKIP] {case_name} already has the full expected artifact count.")
            continue
        if args.dry_run:
            continue
        run_benchmark(PROJECT_ROOT, case_params)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(2) from exc
