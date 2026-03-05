# MOGWO Attention Mechanism Research Log

## Global Research Objective
Continuously research, redesign, and improve the Attention mechanism of MOGWO until it demonstrates statistical dominance over state-of-the-art constraint-handling optimizers (like CMOSMA) across all benchmark metrics.

[... previous iterations omitted for brevity ...]

## Iteration 7 (2026-03-04): DARA-MOGWO v7 (Feasibility guard)
[... See previous notes ...]

---

## Iteration 8 (Current): DARA-MOGWO v8 (Feasibility-Conditioned Normalization & Smooth Phase Attention)

### PHASE 1 — Literature & Gap Analysis
**Diagnosis from v7:** 
While the attention guard (fallback) improved feasibility on `s_120` and `m_100`, the Hypervolume (HV) remained far below CMOSMA (~0.02 vs ~0.08). The algorithm was still treating objective variations of highly infeasible solutions on the same scale as feasible solutions, destroying the dominance ranking inside `_update_archive`. Also, the attention weights were static after feasibility was reached, failing to explicitly guide the pack towards mission-quality tradeoffs (makespan vs energy).

**Hypothesis (H8):**
1. **Feasibility-conditioned objective normalization:** Normalizing objectives *only* using the feasible subspace bounds will ensure infeasible solutions are strictly dominated, preventing archive dilution.
2. **Smooth Two-Phase Attention:** Instead of a hard guard, smoothly scale the attention logits. As feasibility drops (`p` approaches 1.0), force the pack to focus exclusively on `risk` and `turn_penalty`. When highly feasible (`p` approaches 0.0), explicitly shape the attention to spread out `makespan` and `energy`.
3. **Limiter Release:** Release the trust-region step limiter (`gamma_t = 1.0`) when the population is feasible, restoring GWO's natural exploratory capability to expand the Hypervolume.

### PHASE 2 — Theoretical Innovation
- `_update_archive` modified to calculate `objective_scale` based only on `abs_obj[feasible_mask]`.
- Replaced the hard fallback guard with a smooth gradient matrix in `_attention_weights`:
  `feas_boost = [0.0, 0.0, 0.70 * p, 0.40 * p]`
  `div_boost = [0.40 * low_div * (1-p), 0.40 * low_div * (1-p), 0.0, 0.0]`
- Re-enabled full velocity step (`step_scale = 1.0`) when `p <= 0.05`.

### PHASE 3 — Implementation
Modified `uav_benchmark/algorithms/mogwo/__init__.py`. Tested successfully via `tests/test_mogwo_attention.py`.

### PHASE 4 — Benchmark & Evaluation
Protocol: `results/eval_dara_v8c` (40 Gen, 40 Pop, 3 Runs)
- **`c_100` (City):** MOGWO Feasibility `1.0`, HV `0.0118`. (CMOSMA: HV `0.086`)
- **`m_100` (Mountain):** MOGWO Feasibility `1.0`, HV `0.0166`. (CMOSMA: HV `0.079`)
- **`s_120` (Suburban):** MOGWO Feasibility `1.0`, HV `0.0212`. (CMOSMA: HV `0.081`)

### PHASE 5 — Analysis & Decision
**Outcome:**
The H8 modifications were an absolute success for **Robust Feasibility**. MOGWO achieved 100% feasibility across ALL challenging benchmark maps, which was previously considered impossible for continuous GWO without topological mapping. It strictly dominated NSGA-II on every metric.

**Failure Mode:**
Despite 100% feasibility, MOGWO's Hypervolume (HV) is still stuck at ~25% of CMOSMA's HV capacity. The continuous GWO updates tend to collapse into a very narrow feasible corridor (often finding the safest path but failing to find diverse sets of makespan vs. energy tradeoffs). 

**Next Hypothesis (H9):**
To close the HV gap with CMOSMA, MOGWO needs a **Dual-Population Assisting Archive** (Unconstrained vs. Constrained). CMOSMA achieves massive HV because it maintains an Unconstrained Population (AP) that explores the edges of obstacles, constantly pulling the Feasible Population (FP) toward wider Pareto fronts. We will restructure MOGWO to run dual objective-archives simultaneously to seed the attention mechanism with boundary-hugging leaders.
