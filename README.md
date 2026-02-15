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

## Quickstart

Single-UAV smoke:

```bash
python3 -m uav_benchmark.cli benchmark --project-root . --results-dir results/smoke_single --generations 5 --population 20 --runs 1 --mode single --compute-metrics
```

Multi-UAV smoke:

```bash
python3 -m uav_benchmark.cli benchmark-multi --project-root . --results-dir results/smoke_multi --protocol configs/smoke_multi.yaml --compute-metrics --gpu-mode auto
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
- `rlPolicyCheckpointPath`: optional explicit checkpoint path; if omitted, a per-problem path is generated.

Additional run controls (`--extra-json`):

- `resumeExistingRuns: true|false`: skip completed `Run_*` folders and continue interrupted runs.
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
```

- `run_paper_full_trainfreeze.py`: full multi-scenario baseline + RL warmstart + RL freeze pipeline.
- `run_paper_first_problem_trainfreeze.py`: first-problem (`c_100_uav3`) benchmark with train/freeze protocol.
- `run_paper_first_problem_trainfreeze_refine.py`: first-problem train/freeze with RL elite-refinement enabled.

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
- `scripts/`: Helper scripts for running benchmark workflows.
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
