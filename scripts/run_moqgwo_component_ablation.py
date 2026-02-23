from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CASE_EXTRAS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_attention": {"moqgwoVariant": "no_attention"},
    "no_atlas": {"moqgwoUseAtlas": False},
    "standard_gwo": {"moqgwoVariant": "standard_gwo"},
}


def _runner_registered(benchmark_module: Any) -> bool:
    runner_map = getattr(benchmark_module, "_RUNNER_BY_NAME", {})
    return "MOQGWO" in runner_map


def _select_cases(raw: str) -> list[str]:
    if not raw.strip():
        return list(CASE_EXTRAS.keys())
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in CASE_EXTRAS]
    if unknown:
        raise RuntimeError(f"Unknown ablation case(s): {', '.join(unknown)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair MOQGWO component ablation cases.")
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "moqgwo_component_ablation.yaml"), type=str)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results" / "moqgwo_component_ablation"), type=str)
    parser.add_argument("--cases", default="", type=str, help="Comma-separated subset of: full,no_attention,no_atlas,standard_gwo")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    try:
        from uav_benchmark.benchmark import run_benchmark  # type: ignore
        import uav_benchmark.benchmark as benchmark_module  # type: ignore
        from uav_benchmark.cli import _load_protocol  # type: ignore
        from uav_benchmark.config import BenchmarkParams  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Unable to import benchmark pipeline in this code snapshot. "
            "Restore missing benchmark dependencies first."
        ) from exc

    if not _runner_registered(benchmark_module):
        raise RuntimeError(
            "MOQGWO runner is not registered in this code snapshot. "
            "Restore/add uav_benchmark.algorithms MOQGWO implementation first."
        )

    protocol = _load_protocol(Path(args.protocol).expanduser().resolve())
    base_params = BenchmarkParams.from_mapping(protocol)
    base_params.mode = "multi"
    base_params.results_dir = Path(args.results_root).expanduser().resolve()
    base_params.extra = dict(base_params.extra)
    base_params.extra["algorithms"] = ["MOQGWO"]

    selected_cases = _select_cases(str(args.cases))
    for case_name in selected_cases:
        case_params = replace(base_params)
        case_params.results_dir = base_params.results_dir / case_name
        case_params.extra = dict(base_params.extra)
        case_params.extra.update(CASE_EXTRAS[case_name])
        print(f"[ABLT] {case_name} -> {case_params.results_dir}")
        if args.dry_run:
            continue
        run_benchmark(PROJECT_ROOT, case_params)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(2)
