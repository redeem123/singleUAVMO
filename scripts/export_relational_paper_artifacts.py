from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export relational paper artifacts to JSON, CSV, and LaTeX.")
    parser.add_argument("--input", required=True, type=Path, help="Input summary JSON.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: object) -> str:
    text = str(value)
    return text.replace("_", "\\_")


def _write_latex_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(_latex_escape(col) for col in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row.get(col, "")) for col in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_table_family(output_dir: Path, stem: str, rows: list[dict[str, object]]) -> None:
    _write_csv(output_dir / f"{stem}.csv", rows)
    _write_latex_table(output_dir / f"{stem}.tex", rows)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _mode_summary(payload: dict[str, Any]) -> dict[str, Any]:
    direct = payload.get("modeSummary")
    if isinstance(direct, dict):
        return direct
    overall = payload.get("overall", {})
    if isinstance(overall, dict):
        nested = overall.get("modeSummary")
        if isinstance(nested, dict):
            return nested
    return {}


def _pairwise_summary(payload: dict[str, Any]) -> dict[str, Any]:
    direct = payload.get("pairwise")
    if isinstance(direct, dict):
        return direct
    overall = payload.get("overall", {})
    if isinstance(overall, dict):
        nested = overall.get("pairwise")
        if isinstance(nested, dict):
            return nested
    return {}


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("records", "evalRecords", "trainRecords"):
        values = payload.get(key, [])
        if not isinstance(values, list):
            continue
        rows: list[dict[str, Any]] = []
        for record in values:
            if isinstance(record, dict):
                rows.append(record)
        if rows:
            return rows
    return []


def _main_results_rows(payload: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode_name, summary in _mode_summary(payload).items():
        if not isinstance(summary, dict):
            continue
        row: dict[str, object] = {"mode": str(mode_name)}
        row.update(summary)
        rows.append(row)
    return rows


def _pairwise_rows(payload: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison, comparison_payload in _pairwise_summary(payload).items():
        if not isinstance(comparison_payload, dict):
            continue
        metrics_payload = comparison_payload.get("summary", comparison_payload)
        if not isinstance(metrics_payload, dict):
            continue
        for metric_name, summary in metrics_payload.items():
            if not isinstance(summary, dict):
                continue
            rows.append(
                {
                    "comparison": str(comparison),
                    "metric": str(metric_name),
                    "count": summary.get("count", 0),
                    "meanDelta": summary.get("meanDelta", 0.0),
                    "medianDelta": summary.get("medianDelta", 0.0),
                    "wins": summary.get("wins", 0),
                    "losses": summary.get("losses", 0),
                    "ties": summary.get("ties", 0),
                    "wilcoxonPValue": summary.get("wilcoxonPValue", summary.get("pValue", "")),
                }
            )
    return rows


def _policy_mode_rows(payload: dict[str, Any]) -> list[dict[str, object]]:
    policy_modes = {"online", "finetune", "frozen"}
    rows: list[dict[str, object]] = []
    for row in _main_results_rows(payload):
        if str(row.get("mode", "")) in policy_modes:
            rows.append(row)
    return rows


def _runtime_rows(payload: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in _records(payload):
        runtime_fields = {
            "runtimeSec": record.get("runtimeSec"),
            "rlControllerTimeSec": record.get("rlControllerTimeSec"),
            "gpuUpdateTimeSec": record.get("gpuUpdateTimeSec"),
        }
        if not any(value is not None for value in runtime_fields.values()):
            continue
        rows.append(
            {
                "problem": record.get("problem", ""),
                "seed": record.get("seed", 0),
                "mode": record.get("stateRepresentation", record.get("policyMode", record.get("mode", ""))),
                **runtime_fields,
            }
        )
    return rows


def _transfer_rows(payload: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in _records(payload):
        rows.append(
            {
                "problem": record.get("problem", ""),
                "seed": record.get("seed", 0),
                "mode": record.get("stateRepresentation", record.get("policyMode", record.get("mode", ""))),
                "conflictMean": record.get("conflictMean", 0.0),
                "violationMean": record.get("violationMean", 0.0),
                "firstFeasibleGeneration": record.get("firstFeasibleGeneration", 0.0),
                "feasibleMean": record.get("feasibleMean", 0.0),
            }
        )
    return rows


def _replica_rows(payload: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_replica = payload.get("byReplica", {})
    if not isinstance(by_replica, dict):
        return rows
    for replica_name, replica_payload in by_replica.items():
        if not isinstance(replica_payload, dict):
            continue
        replica_summary = replica_payload.get("modeSummary", {})
        if not isinstance(replica_summary, dict):
            continue
        for mode_name, summary in replica_summary.items():
            if not isinstance(summary, dict):
                continue
            row: dict[str, object] = {"replica": str(replica_name), "mode": str(mode_name)}
            row.update(summary)
            rows.append(row)
    return rows


def _command_note(input_path: Path, output_dir: Path) -> str:
    return f"python3 scripts/export_relational_paper_artifacts.py --input {input_path} --output-dir {output_dir}"


def _seed_list(payload: dict[str, Any]) -> list[int]:
    config = payload.get("config", {})
    if not isinstance(config, dict):
        return []
    seeds = config.get("seeds", config.get("evalSeeds", config.get("trainSeeds", [])))
    if not isinstance(seeds, list):
        return []
    values: list[int] = []
    for item in seeds:
        try:
            values.append(int(item))
        except Exception:
            continue
    return values


def _manifest(
    *,
    payload: dict[str, Any],
    input_path: Path,
    output_dir: Path,
    table_paths: list[Path],
) -> dict[str, Any]:
    config = payload.get("config", {})
    paper_slug = ""
    if isinstance(config, dict):
        paper_slug = str(config.get("paperSlug", "")).strip()
    if not paper_slug:
        paper_slug = input_path.stem
    notes: list[str] = []
    if not _runtime_rows(payload):
        notes.append("Runtime table is empty because the input summary did not include per-record runtime fields.")
    if payload.get("byReplica"):
        notes.append("Replica-level summaries were exported to replica_results.csv for variance reporting.")
    created_at = datetime.now(timezone.utc).isoformat()
    tables = [str(path.relative_to(output_dir)) for path in table_paths]
    configs = [config] if isinstance(config, dict) and config else []
    metrics = [
        "hypervolume",
        "pureDiversity",
        "feasibleMean",
        "conflictMean",
        "violationMean",
        "firstFeasibleGeneration",
        "runtimeSec",
    ]
    manifest = {
        "paper_slug": paper_slug,
        "title": "Relational adaptive search control artifact bundle",
        "claim": "Exports the summary tables required to report encoder, policy, and runtime evidence without dropping sample counts or p-values.",
        "created_at": created_at,
        "repo_root": str(PROJECT_ROOT),
        "git_commit": _git_commit(),
        "commands": [_command_note(input_path.resolve(), output_dir.resolve())],
        "configs": configs,
        "results_dirs": [str(output_dir.resolve()), str((output_dir / "summaries").resolve())],
        "seeds": _seed_list(payload),
        "metrics": metrics,
        "tables": tables,
        "figures": [],
        "notes": notes,
        "paperSlug": paper_slug,
        "createdAtUtc": created_at,
        "inputSummary": str(input_path.resolve()),
        "outputDir": str(output_dir.resolve()),
        "repoRoot": str(PROJECT_ROOT),
        "gitCommit": _git_commit(),
        "recordCount": int(payload.get("recordCount", len(_records(payload)))),
        "replicas": list(payload.get("replicas", [])) if isinstance(payload.get("replicas", []), list) else [],
    }
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summaries_dir / "input_summary.json", payload)

    main_rows = _main_results_rows(payload)
    pairwise_rows = _pairwise_rows(payload)
    runtime_rows = _runtime_rows(payload)
    policy_rows = _policy_mode_rows(payload)
    transfer_rows = _transfer_rows(payload)
    replica_rows = _replica_rows(payload)

    table_paths: list[Path] = []
    for stem, rows in (
        ("main_results", main_rows),
        ("ablation", pairwise_rows),
        ("runtime", runtime_rows),
        ("encoder_ablation", pairwise_rows),
        ("transfer_table", transfer_rows),
        ("policy_mode_table", policy_rows),
        ("runtime_table", runtime_rows),
        ("replica_results", replica_rows),
    ):
        _write_table_family(output_dir, stem, rows)
        table_paths.extend([output_dir / f"{stem}.csv", output_dir / f"{stem}.tex"])

    manifest = _manifest(
        payload=payload,
        input_path=args.input,
        output_dir=output_dir,
        table_paths=table_paths,
    )
    _write_json(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
