from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.utils.fs import sha256_file


def check_parity(project_root: Path, manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    missing = []
    mismatched = []
    verified = 0
    for entry in entries:
        rel_file = entry["file"]
        expected_sha = entry["sha256"]
        current_file = project_root / rel_file
        if not current_file.exists():
            missing.append(rel_file)
            continue
        actual_sha = sha256_file(current_file)
        if actual_sha != expected_sha:
            mismatched.append({"file": rel_file, "expected": expected_sha, "actual": actual_sha})
            continue
        verified += 1
    return {
        "total": len(entries),
        "verified": verified,
        "missing": missing,
        "mismatched": mismatched,
    }


def main() -> None:
    manifest = PROJECT_ROOT / "docs" / "python_port" / "baseline_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Baseline manifest not found: {manifest}")
    result = check_parity(PROJECT_ROOT, manifest)
    print(json.dumps(result, indent=2))
    if result["missing"] or result["mismatched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
