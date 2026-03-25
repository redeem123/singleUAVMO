from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.problem_generation import save_mountain


def main() -> None:
    output = Path(__file__).resolve().parent / "terrainStruct.mat"
    save_mountain(output, with_nofly=True)
    print(output)


if __name__ == "__main__":
    main()
