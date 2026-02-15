# Python Port Migration Notes

## Scope
- Ported benchmark orchestration, algorithm runners, metric computation, analysis utilities, and terrain generation to Python.
- Kept MATLAB source files in place for traceability and parity checking.
- Left `reference_code/` unchanged.

## Module Mapping
- `scripts/run_benchmark.m` -> `scripts/run_benchmark.py` and `uav_benchmark/benchmark.py`
- `scripts/run_ablation.m` -> `scripts/run_ablation.py`
- `analysis/*.m` -> `uav_benchmark/analysis/*` via CLI entrypoints
- `core/evaluate_path.m` -> `uav_benchmark/core/evaluate_path.py`
- `core/metrics/*.m` -> `uav_benchmark/core/metrics.py`
- `algorithms/*/run_*.m` -> `uav_benchmark/algorithms/*.py`
- `Problem Generation/*.m` -> `Problem Generation/*.py` and `uav_benchmark/problem_generation/generate.py`

## Baseline Artifacts
- Script: `scripts/capture_baseline.py`
- Output: `docs/python_port/baseline_manifest.json`
- Purpose: deterministic hash manifest for MATLAB-produced outputs (`final_popobj.mat`, `gen_hv.mat`, `final_hv.mat`) used for strict parity checks.

## Parity Workflow
1. Capture MATLAB baseline hashes using `scripts/capture_baseline.py`.
2. Run Python benchmark (`scripts/run_benchmark.py`).
3. Recompute metrics (`python3 -m uav_benchmark.cli compute-metrics --results-dir results`).
4. Compare HV/PD distributions and baseline hashes for generated result files.

## Runtime Dependencies
- Python: `>=3.10`
- Required packages: `numpy`, `scipy`, `matplotlib`
- Install with `pip install -r requirements-python.txt`.
