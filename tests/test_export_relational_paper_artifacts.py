from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_relational_paper_artifacts import main


class ExportRelationalPaperArtifactsTest(unittest.TestCase):
    def test_exporter_writes_json_csv_and_latex(self) -> None:
        payload = {
            "config": {"paper": "demo"},
            "records": [
                {
                    "problem": "p",
                    "seed": 1,
                    "stateRepresentation": "TRFTS",
                    "conflictMean": 0.1,
                    "violationMean": 0.2,
                    "runtimeSec": 12.5,
                }
            ],
            "modeSummary": {
                "TRFTS": {"hypervolumeMean": 0.5, "pureDiversityMean": 0.4},
                "online": {"hypervolumeMean": 0.3, "feasibleMean": 0.1},
            },
            "pairwise": {
                "TRFTS_vs_flat": {
                    "summary": {
                        "hypervolume": {
                            "meanDelta": 0.1,
                            "medianDelta": 0.05,
                            "wins": 2,
                            "losses": 1,
                            "ties": 0,
                            "wilcoxonPValue": 0.01,
                        },
                        "firstFeasibleGeneration": {"meanDelta": 1.0, "wins": 2, "losses": 0, "ties": 1},
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "summary.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            output_dir = root / "out"
            import sys

            argv = sys.argv
            try:
                sys.argv = [
                    "export_relational_paper_artifacts.py",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ]
                main()
            finally:
                sys.argv = argv
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "summaries" / "input_summary.json").exists())
            self.assertTrue((output_dir / "main_results.csv").exists())
            self.assertTrue((output_dir / "main_results.tex").exists())
            self.assertTrue((output_dir / "ablation.csv").exists())
            self.assertTrue((output_dir / "encoder_ablation.csv").exists())
            self.assertTrue((output_dir / "policy_mode_table.tex").exists())
            self.assertIn("runtimeSec", (output_dir / "runtime_table.csv").read_text(encoding="utf-8"))
            self.assertIn("wilcoxonPValue", (output_dir / "ablation.csv").read_text(encoding="utf-8"))

    def test_exporter_accepts_replica_aggregate_summary_shape(self) -> None:
        payload = {
            "recordCount": 6,
            "replicas": ["r1", "r2"],
            "overall": {
                "modeSummary": {
                    "flat": {"count": 2, "hypervolumeMean": 0.4},
                    "TRFTS": {"count": 2, "hypervolumeMean": 0.5},
                },
                "pairwise": {
                    "TRFTS_vs_flat": {
                        "hypervolume": {
                            "count": 2,
                            "meanDelta": 0.1,
                            "medianDelta": 0.1,
                            "wins": 2,
                            "losses": 0,
                            "ties": 0,
                            "wilcoxonPValue": 0.02,
                        }
                    }
                },
            },
            "byReplica": {
                "r1": {"modeSummary": {"flat": {"count": 1, "hypervolumeMean": 0.3}}},
                "r2": {"modeSummary": {"TRFTS": {"count": 1, "hypervolumeMean": 0.6}}},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "aggregate.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            output_dir = root / "out"
            argv = sys.argv
            try:
                sys.argv = [
                    "export_relational_paper_artifacts.py",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ]
                main()
            finally:
                sys.argv = argv
            self.assertTrue((output_dir / "replica_results.csv").exists())
            self.assertIn("TRFTS_vs_flat", (output_dir / "ablation.csv").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["recordCount"], 6)
            self.assertIn("replica_results.csv", "\n".join(manifest["tables"]))


if __name__ == "__main__":
    unittest.main()
