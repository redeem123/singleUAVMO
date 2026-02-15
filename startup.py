from __future__ import annotations

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    problems = list((project_root / "problems").glob("*.mat"))
    print("Initializing UAV Path Planning Research Environment (Python)...")
    print(f"Project root: {project_root}")
    print(f"Available scenarios: {len(problems)}")
    print("Ready for benchmarking.")


if __name__ == "__main__":
    main()
