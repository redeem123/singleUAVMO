"""Remove generated workspace artifacts without touching source files."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "venv", "env", ".mypy_cache", ".ruff_cache"}
PRESERVED_PLACEHOLDERS = {"README.md", ".gitkeep"}


def _is_excluded(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True
    return any(part in EXCLUDED_DIRS for part in rel.parts)


def _walk_generated() -> tuple[list[Path], list[Path]]:
    cache_dirs: list[Path] = []
    cache_files: list[Path] = []

    for current, dirnames, filenames in os.walk(ROOT):
        path = Path(current)
        if _is_excluded(path):
            dirnames[:] = []
            continue

        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        if path.name == "__pycache__":
            cache_dirs.append(path)
            dirnames[:] = []
            continue

        for filename in filenames:
            file_path = path / filename
            if filename == ".DS_Store" or file_path.suffix in {".pyc", ".pyo"}:
                cache_files.append(file_path)

    explicit_dirs = [
        ROOT / ".pytest_cache",
        ROOT / ".hypothesis",
        ROOT / "uav_path_planning_benchmark.egg-info",
    ]
    cache_dirs.extend(path for path in explicit_dirs if path.exists())
    return sorted(set(cache_dirs)), sorted(set(cache_files))


def _workspace_contents(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.name not in PRESERVED_PLACEHOLDERS)


def _selected_targets(args: argparse.Namespace) -> list[Path]:
    targets: list[Path] = []

    if args.all or args.caches:
        cache_dirs, cache_files = _walk_generated()
        targets.extend(cache_dirs)
        targets.extend(cache_files)

    if args.all or args.scratch:
        targets.extend(path for path in [ROOT / "tmp", ROOT / "tmp_inspect", ROOT / "output"] if path.exists())

    if args.all or args.results:
        targets.extend(_workspace_contents(ROOT / "results"))

    if args.all or args.logs:
        targets.extend(_workspace_contents(ROOT / "logs"))

    deduped: dict[Path, Path] = {}
    for target in targets:
        deduped[target.resolve()] = target
    return sorted(deduped.values(), key=lambda path: str(path.relative_to(ROOT)))


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", action="store_true", help="remove Python/test caches and local egg-info")
    parser.add_argument("--scratch", action="store_true", help="remove tmp/, tmp_inspect/, and output/")
    parser.add_argument(
        "--results", action="store_true", help="clear generated results/ contents, preserving README.md"
    )
    parser.add_argument("--logs", action="store_true", help="clear generated logs/ contents, preserving README.md")
    parser.add_argument("--all", action="store_true", help="select caches, scratch, results, and logs")
    parser.add_argument("--yes", action="store_true", help="delete selected paths; without this, only print a dry run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not any([args.all, args.caches, args.scratch, args.results, args.logs]):
        print("Nothing selected. Use --caches, --scratch, --results, --logs, or --all.")
        return 2

    targets = _selected_targets(args)
    if not targets:
        print("Workspace is already clean for the selected groups.")
        return 0

    action = "Removing" if args.yes else "Would remove"
    for target in targets:
        print(f"{action}: {target.relative_to(ROOT)}")
        if args.yes:
            _remove(target)

    if not args.yes:
        print("Dry run only. Re-run with --yes to delete these paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
