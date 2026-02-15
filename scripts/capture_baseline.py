from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.utils.fs import sha256_file


@dataclass(slots=True)
class BaselineEntry:
    algorithm: str
    problem: str
    run: str
    file: str
    sha256: str
    size_bytes: int


def capture_baseline(project_root: Path) -> dict:
    results_dir = project_root / "results"
    entries: list[dict] = []
    for algorithm_dir in sorted(results_dir.iterdir()):
        if not algorithm_dir.is_dir() or algorithm_dir.name.startswith("."):
            continue
        for problem_dir in sorted(algorithm_dir.iterdir()):
            if not problem_dir.is_dir() or problem_dir.name.startswith("."):
                continue
            for run_dir in sorted(problem_dir.glob("Run_*")):
                if not run_dir.is_dir():
                    continue
                for filename in ("final_popobj.mat", "gen_hv.mat"):
                    path = run_dir / filename
                    if not path.exists():
                        continue
                    entry = BaselineEntry(
                        algorithm=algorithm_dir.name,
                        problem=problem_dir.name,
                        run=run_dir.name,
                        file=str(path.relative_to(project_root)),
                        sha256=sha256_file(path),
                        size_bytes=path.stat().st_size,
                    )
                    entries.append(asdict(entry))
            final_hv = problem_dir / "final_hv.mat"
            if final_hv.exists():
                entry = BaselineEntry(
                    algorithm=algorithm_dir.name,
                    problem=problem_dir.name,
                    run="",
                    file=str(final_hv.relative_to(project_root)),
                    sha256=sha256_file(final_hv),
                    size_bytes=final_hv.stat().st_size,
                )
                entries.append(asdict(entry))
    summary = {
        "project_root": str(project_root),
        "entry_count": len(entries),
        "entries": entries,
    }
    return summary


def main() -> None:
    project_root = PROJECT_ROOT
    payload = capture_baseline(project_root)
    out_dir = project_root / "docs" / "python_port"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_manifest.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_file)


if __name__ == "__main__":
    main()
