# Python UAV Benchmark (Single + Multi-UAV)

## Brief Project Overview

This repository provides a reproducible benchmark framework for constrained UAV path planning and mission optimization. It supports both single-UAV and multi-UAV experiments, compares multiple evolutionary baselines, and includes an RL-enhanced NMOPSO variant for learning-guided search.

Current scope includes:

- single-UAV workflows (backward compatible),
- multi-UAV mission benchmarking (homogeneous point-to-point),
- RL-enhanced NMOPSO (`RL-NMOPSO`) in multi-UAV mode,
- optional GPU acceleration (`--gpu-mode auto|off|force`).

## Project Goals

- Provide a reproducible benchmark framework for constrained UAV path planning.
- Preserve single-UAV compatibility while extending to multi-UAV mission-level optimization.
- Offer baseline evolutionary methods and RL-enhanced variants for comparative studies.
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

Single-UAV smoke:

```bash
python3 -m uav_benchmark.cli benchmark --project-root . --results-dir results/smoke_single --generations 5 --population 20 --runs 1 --mode single --compute-metrics
```

Multi-UAV smoke:

```bash
python3 -m uav_benchmark.cli benchmark-multi --project-root . --results-dir results/smoke_multi --protocol configs/smoke_multi.yaml --compute-metrics --gpu-mode auto
```

Cleanup generated artifacts/caches:

```bash
python3 scripts/clean_workspace.py --results --caches
```

RL-NMOPSO with stronger RL-GPU workload (separate from NMOPSO baseline):

```bash
python3 -m uav_benchmark.cli benchmark-multi \
  --project-root . \
  --results-dir results/rl_gpu \
  --generations 300 --population 80 --runs 10 \
  --gpu-mode auto \
  --extra-json '{"rlUseGpuPolicy":true,"rlControllerBackend":"auto","rlGpuHiddenDim":384,"rlGpuBatchSize":2048,"rlGpuTrainSteps":16,"rlGpuMinTrainSize":128,"rlGpuReplayCapacity":65536,"rlRewardNStep":5,"rlRewardGamma":0.9,"rlPhaseGating":true}'
```

Policy checkpoint modes (`--extra-json`):

- `rlPolicyMode: "train"`: train from scratch and save checkpoint.
- `rlPolicyMode: "warmstart"`: load checkpoint, continue training, then save.
- `rlPolicyMode: "freeze"`: load checkpoint and run inference-only (no policy updates).
- `rlPolicyMode: "online"`: train online within each run, no checkpoint load/save (direct baseline-style comparison).
- `rlPolicyCheckpointPath`: optional explicit checkpoint path; if omitted, a per-problem path is generated.

RL profile presets (`--extra-json`):

- `rlProfile: "lite"`: conservative RL defaults (linucb-first, low auxiliary budget, most auxiliary operators off).
- `rlProfile: "full"`: balanced default profile for RL-NMOPSO experiments.
- `rlProfile: "expert"`: same typed config path as `full`, intended for explicit per-component overrides.

Budget-aware RL controls (`--extra-json`):

- `rlAuxEvalBudgetFactor`: per-generation auxiliary evaluation budget as `factor * population` (default `1.0`).
- `rlRewardCostWeight`: reward penalty weight for auxiliary evaluation usage (default `0.08`).
- `useFRRMAB`: enable/disable FRRMAB arm scheduler.

Additional run controls (`--extra-json`):

- `resumeExistingRuns: true|false`: skip completed `Run_*` folders and continue interrupted runs.
- `maxWorkers: N`: cap worker processes (defaults to available CPU count).
- `problemNames: ["c_100_uav3", ...]`: run only selected problems/scenarios.
- `rlEliteRefine: true|false` plus:
  - `rlEliteRefineTopK`
  - `rlEliteRefineIters`
  - `rlEliteRefineSigmaStart`
  - `rlEliteRefineSigmaEnd`

## Multi-UAV Paper Pipeline

Run the paper artifact pipeline:

```bash
python3 -m uav_benchmark.cli paper-artifacts --project-root . --results-dir results/paper_artifacts --protocol configs/paper_medium_multi.yaml --gpu-mode auto
```

This runs benchmark + report + stats + multi-UAV plots.

Train/freeze helper scripts:

```bash
python3 scripts/run_paper_full_trainfreeze.py
python3 scripts/run_paper_first_problem_trainfreeze.py
python3 scripts/run_paper_first_problem_trainfreeze_refine.py
python3 scripts/run_rl_component_ablation.py --quick
python3 scripts/run_publication_suite.py \
  --results-root results/publication_suite \
  --attention-results-dir results/publication_suite/attention_ablation_strict \
  --benchmark-results-dir results/three_scenarios_30runs \
  --run-attention \
  --no-run-benchmark \
  --strict-audit
```

- `run_paper_full_trainfreeze.py`: full multi-scenario baseline + RL warmstart + RL freeze pipeline.
- `run_paper_first_problem_trainfreeze.py`: first-problem (`c_100_uav3`) benchmark with train/freeze protocol.
- `run_paper_first_problem_trainfreeze_refine.py`: first-problem train/freeze with RL elite-refinement enabled.
- `run_rl_component_ablation.py`: RL profile/component ablation matrix with per-case metric reports and aggregate CSV.
- `run_attention_ablation.py`: strict attention ablation matrix with run-manifest and quality gates.
- `publication_readiness_audit.py`: mandatory publication-gate audit over ablation + benchmark artifacts.
- `export_publication_tables.py`: export paper-ready tables in CSV/Markdown/LaTeX.
- `run_publication_suite.py`: end-to-end publication bundle orchestration (run/audit/tables/package).

Publication docs:

- `docs/rl_ablation_protocol.md`
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
- `matlabimplementation/`: Original MATLAB-side implementation assets.
- `results/`: Generated benchmark outputs (ignored in Git for large artifacts).
- `requirements-python.txt`, `requirements-gpu.txt`: CPU/GPU dependency sets.
- `pyproject.toml`: Python project metadata.

## Multi-UAV Result Artifacts

Each multi-UAV run (`Run_*`) stores:

- `final_popobj.mat`
- `run_stats.mat`
- `mission_stats.mat`
- `fleet_paths.mat`
- `conflict_log.mat`
- `bp_*.mat` (compatibility path exports)

`run_stats.mat` now includes GPU/RL split telemetry:

- `gpuBackend`, `gpuMemPeakBytes`, `gpuUpdateTimeSec`
- `rlPolicyBackend`, `rlPolicyGpuMemPeakBytes`, `rlControllerTimeSec`, `rlPolicyLossEma`

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
