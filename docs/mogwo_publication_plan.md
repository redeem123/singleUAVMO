# MOGWO Publication Preparation Plan

This document captures the exact next-phase protocol for fair component ablations, complexity analysis, and corrected statistical testing.

## 1) Component-Wise Ablation (Fairness Preserved)

Use the same mission evaluator, archive size, constraints, population, generations, and run count for all ablation cases.

Required cases:

- `full`: full A2-MOGWO.
- `no_attention`: same QGWO framework, but remove attention term only.
- `standard_gwo`: standard GWO core inside the same MO wrapper (same constraint handling, archive flow, and evaluation budget).

Fairness rules:

- Do not change evaluation budget between cases.
- Do not change problem set between cases.
- Keep random seed policy identical across cases.
- Keep repair/constraint enforcement policy identical unless explicitly part of the ablation target.

## 2) Complexity Analysis (Decomposition + Empirical Profiling)

Report complexity as:

`T_gen = T_eval + T_update + T_archive`

Interpretation target:

- Attention term should be `O(N * D)` because the number of leaders is constant.
- Evaluation usually dominates in this repository.

Empirical output requirement:

- Report `%T_eval`, `%T_update`, `%T_archive` per algorithm/problem.
- Also report total runtime per run.

Prepared helper:

- `scripts/analyze_runtime_breakdown.py`

## 3) Statistical Significance (Corrected)

Per-problem pairwise:

- Use Wilcoxon signed-rank when runs are paired by run index.
- Use rank-sum fallback when pairing is unavailable.
- Apply Holm correction across multiple comparisons.
- Report effect size (Cliff's delta) with corrected p-values.

Global multi-algorithm:

- Friedman test over paired blocks `(problem, run_id)`.
- Post-hoc control-vs-others with Holm-corrected p-values.

Prepared helper:

- `scripts/analyze_benchmark_significance.py`

## 4) Suggested Execution Order

1. Run component ablations with fixed protocol and fixed seeds.
2. Generate standard metric summaries.
3. Run runtime breakdown script for complexity evidence.
4. Run significance script for pairwise + global statistics.
5. Export publication tables from generated CSVs.

## 5) Prepared Commands

Component ablation runner template:

```bash
python3 scripts/run_mogwo_component_ablation.py \
  --protocol configs/full_benchmark.yaml \
  --results-root results/mogwo_component_ablation
```

Significance and multiple-comparison correction:

```bash
python3 scripts/analyze_benchmark_significance.py \
  --results-dir results/full_benchmark_20260222 \
  --control-algorithm MOGWO \
  --pairwise-mode auto
```

Complexity breakdown profiling:

```bash
python3 scripts/analyze_runtime_breakdown.py \
  --results-dir results/full_benchmark_20260222
```
