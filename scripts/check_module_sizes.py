from __future__ import annotations

import argparse
from pathlib import Path

MAX_ACTIVE_SOURCE_LINES = 800

ACTIVE_SOURCE_ROOTS = ("uav_benchmark",)


def _iter_active_modules(project_root: Path) -> list[Path]:
    modules: list[Path] = []
    for raw_path in ACTIVE_SOURCE_ROOTS:
        path = project_root / raw_path
        if path.is_file():
            modules.append(path)
            continue
        if path.is_dir():
            modules.extend(sorted(path.rglob("*.py")))
    return sorted(set(modules))


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check active UAV benchmark module sizes.")
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--max-lines", default=MAX_ACTIVE_SOURCE_LINES, type=int)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    failures: list[tuple[Path, int]] = []
    for module in _iter_active_modules(project_root):
        lines = _line_count(module)
        if lines > args.max_lines:
            failures.append((module.relative_to(project_root), lines))

    if failures:
        print(f"Active source modules over {args.max_lines} lines:")
        for module, lines in failures:
            print(f"  {module}: {lines}")
        return 1

    print(f"Checked active source module sizes: <= {args.max_lines} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
