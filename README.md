# Python UAV Benchmark (Fleet-First)

## Brief Project Overview

This repository provides a reproducible benchmark framework for constrained UAV path planning and mission optimization. It uses one fleet-based architecture and compares multiple evolutionary baselines.

Current scope includes:

- base-fleet workflows (`fleet_size=1`),
- fleet mission benchmarking (homogeneous point-to-point),
- optional GPU acceleration (`--gpu-mode auto|off|force`).

## Project Goals

- Provide a reproducible benchmark framework for constrained UAV path planning.
- Keep one consistent mission-level optimization pipeline for every fleet size.
- Offer baseline evolutionary methods for comparative studies.
- Generate paper-ready metrics and artifacts from scripted runs.

## Setup / Run Instructions

```bash
python3 -m pip install -r requirements-python.txt
```

Optional GPU backends:

```bash
python3 -m pip install -r requirements-gpu.txt
```

Install dev tooling (lint/test):

```bash
python3 -m pip install -e ".[dev]"
```

## Quickstart

Base-fleet smoke (`fleet_size=1`):

```bash
python3 -m uav_benchmark.cli benchmark --project-root . --results-dir results/smoke_fleet1 --generations 5 --population 20 --runs 1 --fleet-size 1 --compute-metrics
```

Fleet smoke:

```bash
python3 -m uav_benchmark.cli benchmark-fleet --project-root . --results-dir results/smoke_fleet --protocol configs/smoke_fleet.yaml --compute-metrics --gpu-mode auto
```

Cleanup generated artifacts/caches:

```bash
python3 scripts/clean_workspace.py --results --caches
```

Additional run controls (`--extra-json`):

- `resumeExistingRuns: true|false`: skip completed `Run_*` folders and continue interrupted runs.
- `maxWorkers: N`: cap worker processes (defaults to available CPU count).
- `problemNames: ["c_100_uav3", ...]`: run only selected problems/scenarios.

## Fleet Paper Pipeline

Run the paper artifact pipeline:

```bash
python3 -m uav_benchmark.cli paper-artifacts --project-root . --results-dir results/paper_artifacts --protocol configs/paper_medium_fleet.yaml --gpu-mode auto
```

This runs benchmark + report + stats + fleet plots.

Helper scripts:

```bash
python3 scripts/run_moqgwo_component_ablation.py
python3 scripts/run_benchmark_fleet.py
```

- `run_attention_ablation.py`: strict attention ablation matrix with run-manifest and quality gates.
- `publication_readiness_audit.py`: mandatory publication-gate audit over ablation + benchmark artifacts.
- `export_publication_tables.py`: export paper-ready tables in CSV/Markdown/LaTeX.
- `run_publication_suite.py`: end-to-end publication bundle orchestration (run/audit/tables/package).

Publication docs:

- `docs/publication_pipeline.md`
- `docs/reproducibility.md`

## Core CLI Commands

```bash
python3 -m uav_benchmark.cli --help
python3 -m uav_benchmark.cli compute-metrics --results-dir results
python3 -m uav_benchmark.cli report-metrics --project-root . --results-dir results --baseline-algorithm NMOPSO
python3 -m uav_benchmark.cli stats --results-dir results
python3 -m uav_benchmark.cli plots --project-root . --results-dir results
python3 -m uav_benchmark.cli path-visualizer c_100 1 --algorithm NMOPSO --show
```

## Directory Structure and Key Files

- `uav_benchmark/`: Core Python package (algorithms, evaluators, analysis, CLI).
- `problems/`: Terrain/problem definitions (`terrainStruct_*.mat`).
- `configs/`: Benchmark protocol YAMLs (e.g., smoke and paper settings).
- `scripts/`: Active helper scripts for benchmark/publication workflows.
- `scripts/legacy/`: Legacy migration/parity wrappers preserved for traceability.
- `tests/`: Unit and smoke tests.
- `docs/`: Protocol/reproducibility notes and reference papers.
- `uav_benchmark/analysis/generate_research_plots.py`: Python launcher that runs embedded MATLAB plotting driver.
- `results/`: Generated benchmark outputs (ignored in Git for large artifacts).
- `requirements-python.txt`, `requirements-gpu.txt`: CPU/GPU dependency sets.
- `pyproject.toml`: Python project metadata.

## Fleet Result Artifacts

Each run (`Run_*`) stores:

- `final_popobj.mat`
- `run_stats.mat`
- `mission_stats.mat`
- `fleet_paths.mat`
- `conflict_log.mat`
- `bp_*.mat` (compatibility path exports)

`run_stats.mat` includes runtime and GPU telemetry:

- `gpuBackend`, `gpuMemPeakBytes`, `gpuUpdateTimeSec`

Metrics reports are written to `results/.../metrics/`:

- `benchmark_metrics_summary.csv`
- `pairwise_stats.csv`
- `win_tie_loss.csv`
- `benchmark_metrics_summary.json`

Benchmark-level reproducibility manifest:

- `results/<run_dir>/benchmark_manifest.json` (resolved problem/algorithm plan, per-task effective seeds, git/env metadata, plan hash)

## Notes

- Use `--gpu-mode force` to require a GPU backend; it falls back to CPU only when no backend is available and logs backend in `run_stats.mat`.
- Scenario generation for `paper_medium` creates `terrainStruct_<base>_uav<K>.mat` problems on demand.

## Contribution Guidelines

1. Fork the repository and create a feature branch.
2. Keep changes scoped and include/update tests under `tests/` when behavior changes.
3. Run tests locally:
   - `python3 -m unittest discover -s tests -p 'test_*.py'`
4. For benchmark-affecting changes, include a short run summary (settings + key metrics).
5. Open a pull request with:
   - change summary,
   - affected modules/files,
   - validation evidence (tests and/or benchmark output).
