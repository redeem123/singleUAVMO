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
  --protocol configs/paper_medium_multi.yaml \
  --gpu-mode auto
```

## Outputs

- `results/paper_artifacts/<ALGORITHM>/<PROBLEM>/Run_*/*`
- `results/paper_artifacts/metrics/benchmark_metrics_summary.csv`
- `results/paper_artifacts/metrics/pairwise_stats.csv`
- `results/paper_artifacts/metrics/win_tie_loss.csv`
- `results/paper_artifacts/metrics/benchmark_metrics_summary.json`
- `results/paper_artifacts/plots_multi_uav/*.png`

## Smoke Validation

```bash
python3 -m uav_benchmark.cli benchmark-multi \
  --project-root . \
  --results-dir results/smoke_multi \
  --protocol configs/smoke_multi.yaml \
  --compute-metrics \
  --gpu-mode off
```

## Unit Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
