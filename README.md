# Python UAV Benchmark (Fleet-First)

## Brief Project Overview

This repository provides a reproducible benchmark framework for constrained UAV path planning and mission optimization. It uses one fleet-based architecture and compares multiple evolutionary baselines.

### Recent Upgrades
- **Performance:** Numba JIT acceleration for numerical evaluation pipelines and KDTree spatial indexing for $O(K \log K)$ fleet conflict detection.
- **Accuracy:** Smooth bilinear terrain height interpolation.
- **Architecture:** Formalized `UAVAlgorithm` protocol with a centralized component registry.
- **Validation:** Automated GitHub Actions CI, Wilcoxon rank-sum statistical parity testing, and Hypothesis property-based testing.
- **Data:** Standardized JSON metadata export alongside MATLAB `.mat` artifacts.

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
python3 -m uav_benchmark.cli benchmark --project-root . --results-dir results/smoke_fleet1 --generations 5 --population 20 --runs 1 --fleet-size 1
```

Fleet smoke:

```bash
python3 -m uav_benchmark.cli benchmark --project-root . --results-dir results/smoke_fleet --protocol configs/smoke_fleet.yaml --gpu-mode auto
```

NMOPSO ablation (legacy single-UAV path study):

```bash
python3 -m uav_benchmark.cli ablation --project-root . --results-dir results/nmopso_ablation_smoke --generations 5 --population 20 --runs 1
python3 scripts/run_ablation.py
```

`benchmark` now always runs the full post-processing pipeline automatically after optimization:

- compute metrics
- statistical summary
- benchmark report CSV/JSON
- research plots

Default benchmark scope (without overrides):

- fleet sizes: `1,3`
- base problems: `c_100`, `m_100`, `s_120`

Cleanup generated artifacts/caches, or archive old runs:

```bash
python3 scripts/clean_workspace.py --results --caches
python3 -m uav_benchmark.cli archive --results-dir results --archive-dir archives
```

Additional run controls (`--extra-json`):

- `resumeExistingRuns: true|false`: skip completed `Run_*` folders and continue interrupted runs.
- `maxWorkers: N`: cap worker processes (default is `14`).
- `problemNames: ["c_100_uav3", ...]`: run only selected problems/scenarios.

## Fleet Paper Pipeline

Run the paper artifact pipeline:

```bash
python3 -m uav_benchmark.cli paper-artifacts --project-root . --results-dir results/paper_artifacts --protocol configs/paper_medium_fleet.yaml --gpu-mode auto
```

This runs benchmark + report + stats + fleet plots.

Helper scripts:

```bash
python3 scripts/run_mogwo_component_ablation.py
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
python3 -m uav_benchmark.cli benchmark --project-root . --results-dir results/bench
python3 -m uav_benchmark.cli ablation --project-root . --results-dir results/nmopso_ablation
python3 -m uav_benchmark.cli path-visualizer c_100 1 --algorithm NMOPSO --show
```

## Workspace Layout

- `uav_benchmark/`, `tests/`, `configs/`, `scripts/`, `problems/`, and `docs/` are the tracked source and contributor surfaces.
- `results/` and `logs/` are runtime-generated workspace directories. They stay in the repo root for stable CLI defaults, but their contents are ignored in Git.
- `research/` is the local research workspace. See `research/README.md` for where scratch notes, paper PDFs, and writing drafts now live.
- `scripts/legacy/` holds preserved legacy helpers, including the old problem-generation scripts.

## Directory Structure and Key Files

- `uav_benchmark/`: Core Python package (algorithms, evaluators, analysis, CLI).
- `problems/`: Terrain/problem definitions (`terrainStruct_*.mat`).
- `configs/`: Benchmark protocol YAMLs (e.g., smoke and paper settings).
- `scripts/`: Active helper scripts for benchmark/publication workflows.
- `scripts/legacy/`: Legacy migration/parity wrappers and historical utilities preserved for traceability.
- `tests/`: Unit and smoke tests.
- `docs/`: Stable contributor and reproducibility documentation.
- `research/`: Local-only research workspace for paper corpora, writing drafts, and scratch artifacts.
- `uav_benchmark/analysis/generate_research_plots.py`: Python launcher that runs embedded MATLAB plotting driver.
- `results/`: Generated benchmark outputs (ignored in Git except for the placeholder README).
- `logs/`: Generated logs and background-run metadata (ignored in Git except for the placeholder README).
- `requirements-python.txt`, `requirements-gpu.txt`: CPU/GPU dependency sets.
- `pyproject.toml`: Python project metadata.

Contributor docs:

- `docs/`: protocol, migration, and reproducibility notes.
- `research/README.md`: local research asset layout and policy.
- `scripts/legacy/`: legacy utilities retained outside the active source surface.

## Fleet Result Artifacts

Each run (`Run_*`) stores:

- `final_popobj.mat`
- `run_stats.mat`
- `mission_stats.mat`
- `fleet_paths.mat`
- `conflict_log.mat`
- `bp_*.mat` (compatibility path exports)

`conflict_log.mat` stores detailed per-conflict rows (`step`, `uav_i`, `uav_j`, `distance`, `violation`) plus per-candidate summary rates.

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
   - `./.venv/bin/python -m pytest -q`
4. For benchmark-affecting changes, include a short run summary (settings + key metrics).
5. Open a pull request with:
   - change summary,
   - affected modules/files,
   - validation evidence (tests and/or benchmark output).
