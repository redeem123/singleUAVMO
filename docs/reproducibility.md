# Reproducibility

## Environment

```bash
python3 -m pip install -r requirements-python.txt
```

Optional GPU backend:

```bash
python3 -m pip install -r requirements-gpu.txt
```

## One-Command Paper Artifact Run

```bash
python3 -m uav_benchmark.cli paper-artifacts \
  --project-root . \
  --results-dir results/paper_artifacts \
  --protocol configs/paper_medium_fleet.yaml \
  --gpu-mode auto
```

## Outputs

- `results/paper_artifacts/<ALGORITHM>/<PROBLEM>/Run_*/*`
- `results/paper_artifacts/metrics/benchmark_metrics_summary.csv`
- `results/paper_artifacts/metrics/pairwise_stats.csv`
- `results/paper_artifacts/metrics/win_tie_loss.csv`
- `results/paper_artifacts/metrics/benchmark_metrics_summary.json`
- `results/paper_artifacts/plots_fleet_uav/*.fig`

## Smoke Validation

```bash
python3 -m uav_benchmark.cli benchmark-fleet \
  --project-root . \
  --results-dir results/smoke_fleet \
  --protocol configs/smoke_fleet.yaml \
  --compute-metrics \
  --gpu-mode off
```

## Unit Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## MOQGWO Next Phase Helpers

Template protocol and plan:

- `configs/moqgwo_component_ablation.yaml`
- `docs/moqgwo_publication_plan.md`

Significance analysis (Wilcoxon/rank-sum + Holm + Cliff + Friedman):

```bash
python3 scripts/analyze_benchmark_significance.py \
  --results-dir results/full_benchmark_20260222 \
  --control-algorithm MOQGWO \
  --pairwise-mode auto
```

Runtime complexity breakdown:

```bash
python3 scripts/analyze_runtime_breakdown.py \
  --results-dir results/full_benchmark_20260222
```
