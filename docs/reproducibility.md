# Reproducibility

## Environment

```bash
python3 -m pip install -r requirements-python.txt
```

Optional GPU backend:

```bash
# Apple Silicon MPS/CPU uses the default Torch wheel from requirements-python.txt.
# NVIDIA CUDA hosts should force-reinstall so the CUDA wheel replaces CPU Torch.
python3 -m pip install --upgrade --force-reinstall -r requirements-gpu.txt
```

## One-Command Paper Artifact Run

```bash
python3 -m uav_benchmark.cli paper-artifacts \
  --project-root . \
  --results-dir results/paper_artifacts \
  --protocol configs/full_benchmark.yaml \
  --gpu-mode auto
```

## Outputs

- `results/paper_artifacts/<ALGORITHM>/<PROBLEM>/Run_*/*`
- `results/paper_artifacts/metrics/benchmark_metrics_summary.csv`
- `results/paper_artifacts/metrics/pairwise_stats.csv`
- `results/paper_artifacts/metrics/win_tie_loss.csv`
- `results/paper_artifacts/metrics/benchmark_metrics_summary.json`
- `results/paper_artifacts/plots_fleet_uav/*.fig`

## Benchmark Test

```bash
python3 -m uav_benchmark.cli benchmark \
  --project-root . \
  --results-dir results/benchmark_test \
  --protocol configs/test_benchmark.yaml \
  --gpu-mode off
```

## Unit Tests

```bash
python3 -m pytest -q
```

## MOGWO Next Phase Helpers

Template protocol and plan:

- `configs/mogwo_component_ablation.yaml`
- `docs/mogwo_publication_plan.md`

Significance analysis (Wilcoxon/rank-sum + Holm + Cliff + Friedman):

```bash
python3 scripts/analyze_benchmark_significance.py \
  --results-dir results/full_benchmark_20260222 \
  --control-algorithm MOGWO \
  --pairwise-mode auto
```

Runtime complexity breakdown:

```bash
python3 scripts/analyze_runtime_breakdown.py \
  --results-dir results/full_benchmark_20260222
```
