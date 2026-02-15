# Paper 1: TCOT-CTM-EA for UAV Path Planning

**Title**: Topology-Constrained Optimal Transport for Evolutionary Multitasking on Continuous UAV Task Manifolds

**Target Venue**: IEEE Transactions on Evolutionary Computation (TEVC) or GECCO

---

## 0. Pre-Submission Positioning Notes (Internal)

This section is an internal positioning memo and should be removed in the submission-ready manuscript.

### 0.1 Why Repositioning Was Necessary

Recent EMT literature (2023-2026) already covers:
1. Dynamic auxiliary-task design and transition for constrained multiobjective optimization.
2. Adaptive transfer-rate/policy learning.
3. EMT-based UAV planning variants.

Therefore, claiming novelty from "continuous task scheduling" alone would be weak.

### 0.2 Narrow Claim Used in This Draft

The defensible claim is:

> To the best of our knowledge, this work is the first EMT framework for UAV path planning that performs cross-task knowledge transfer by solving a topology-constrained optimal transport problem on a continuous environment manifold.

Paper-1 novelty is intentionally narrow:
1. Continuous environment manifold (`theta`-space) for task navigation.
2. Topology-constrained OT coupling for transfer flow.

### 0.3 Method Scope for Paper 1

Core modules:
1. Environment-native manifold parameterization.
2. Entropic OT transfer with topology mismatch penalty.
3. Counterfactual transfer credit for online `beta` adaptation.
4. Lightweight feasibility-aware scheduler.

Deferred to Paper 2:
1. Bridge-task synthesis.
2. Learned/geodesic manifold metric.
3. Advanced bandit/lookahead scheduler.

### 0.4 What Is Not Claimed as Novel

The following are treated as standard components:
1. Homotopy/signature extraction primitives.
2. Sinkhorn solver as an OT optimizer.
3. Basic feasibility-aware schedule logic.

Novelty is claimed at the **integration/operator-design level**, not at the primitive-tool level.

### 0.5 Readiness Criteria for a Strong Novelty Position

To justify a high-novelty claim in review:
1. Show operator-level gains versus hard-gate and non-topology OT baselines.
2. Provide causal evidence of reduced negative transfer (`NTI`, `TMTM`).
3. Demonstrate cross-scenario generalization with stable hyperparameters.
4. Align proposition trends with empirical `beta` sweeps.
5. Include transparent failure cases (signature aliasing, severe map shifts).

### 0.6 Implementation Roadmap in This Repository

1. Add `theta` metadata for each `c_*` scenario.
2. Implement topology signature extractor and mismatch computation.
3. Implement Sinkhorn OT transfer plugin and transfer-flow logging.
4. Add counterfactual control subgroup and online `beta` update.
5. Extend `run_benchmark.m` for `TGR`, `NTI`, `TMTM`, and reproducibility exports.

---

## Abstract

Evolutionary Multitasking (EMT) has shown strong performance when related tasks share transferable structure, but most methods still treat tasks as a fixed discrete set. For UAV path planning, this creates a mismatch: scenarios vary continuously (obstacle density, threat level, no-fly structure), while transfer is typically controlled by scalar rates or hand-crafted gates. This paper proposes **TCOT-CTM-EA** (Topology-Constrained Optimal-Transport Continuous Task Manifold Evolutionary Algorithm), which performs transfer on a continuous scenario manifold and solves cross-task knowledge assignment as an entropic optimal transport (OT) problem with explicit topology mismatch penalties. The method keeps the scheduler lightweight and focuses novelty on the transfer operator. We provide a formulation with practical propositions on mismatch mass control, an implementation protocol on the UAV benchmark in this repository, and an evaluation plan emphasizing negative-transfer diagnostics (NTI/TMTM) in addition to HV/IGD. The goal is a reproducible and defensible EMT contribution: not "continuous tasks" alone, but **topology-safe transfer flow optimization on a continuous task manifold**.

**Keywords**: Evolutionary Multitasking, UAV Path Planning, Optimal Transport, Homotopy-Aware Transfer, Continuous Task Manifold

---

## 1. Introduction

### 1.1 Motivation

UAV mission scenarios are naturally continuous in environment parameters (e.g., obstacle density, threat scaling, wind, and no-fly geometry). However, common EMT pipelines still transfer across a few discrete tasks, which often causes negative transfer when geometry changes induce different feasible path topologies.

The key practical issue is not only "task distance", but **topology compatibility** of paths being transferred. Two high-quality paths can be mutually harmful if they belong to different homotopy classes around obstacles. This motivates a transfer mechanism that is:
1. Continuous in scenario space.
2. Explicitly topology-aware.
3. Optimized globally (many-to-many), not pairwise by heuristic gates.

### 1.2 Problem Statement

Given a target UAV planning task `T(theta*)` and a budget of function evaluations, we seek to leverage auxiliary tasks on a continuous scenario manifold to maximize final target quality (HV/IGD and feasibility) while minimizing negative transfer.

Let `theta` denote normalized scenario parameters. The algorithm must decide:
1. Which nearby tasks to visit during evolution.
2. Which source individuals should transfer to which target individuals.
3. How strongly topology mismatch should be penalized over time.

### 1.3 Contributions

1. **Formulation**: Continuous environment-native task manifold for UAV scenarios (not linear interpolation of objective functions).
2. **Core operator**: Topology-constrained entropic OT transfer, producing a coupling matrix `pi*` that allocates cross-task knowledge flow.
3. **Control mechanism**: Counterfactual transfer credit for online adaptation of topology penalty, with a lightweight feasibility-aware scheduler.
4. **Evaluation protocol**: Reproducible benchmark design and ablations centered on negative-transfer diagnostics (`NTI`, `TMTM`) and transfer flow analysis.

### 1.4 Paper Organization

- Section 2 reviews EMT transfer methods, topology-aware path optimization, and OT relevance.
- Section 3 formalizes TCOT-CTM-EA and theoretical propositions.
- Section 4 details algorithmic workflow and complexity.
- Section 5 presents the experimental protocol and ablation design.
- Section 6 discusses strengths, limitations, and risks.
- Section 7 concludes and outlines follow-up work.

---

## 2. Background and Related Work

### 2.1 EMT Transfer: What Exists

Classic EMT (e.g., MFEA/MFEA-II) and newer transfer-learning variants adapt transfer rates, mappings, or auxiliary-task transitions. These methods significantly improve multitask optimization, but transfer is usually controlled by scalar probabilities, learned associations, or decomposition rules.

### 2.2 Topology in UAV Planning

Topology-aware planning and homotopy-based trajectory optimization are well-established in robotics. They show that topology class is a major determinant of path feasibility and quality in cluttered environments.

### 2.3 Why OT Here

OT offers a principled way to optimize many-to-many assignment under a custom cost matrix. In this work, OT is not the novelty by itself; the novelty lies in combining OT coupling with topology mismatch penalties on a continuous UAV task manifold for EMT transfer.

### 2.4 Gap This Paper Targets

Current evidence suggests a missing combination:
1. Continuous environment-parameter manifold for EMT task navigation.
2. Transfer flow optimized as a coupling problem.
3. Explicit topology penalty within the transfer optimization objective.

TCOT-CTM-EA addresses this combination.

---

## 3. Problem Formulation and Theory

### 3.1 Continuous UAV Task Manifold

Define normalized scenario parameters:
`theta = [rho_obs, threat_scale, nofly_ratio, wind_level, terrain_roughness, ...] in [0,1]^p`.

Each task is a simulator-backed problem instance `T(theta)` with common path encoding and objective set (e.g., length, threat exposure, energy, smoothness). Task proximity is measured by:
`d_M(theta_i, theta_j) = ||W(theta_i - theta_j)||_2`,
where `W` is a diagonal normalization/importance matrix.

### 3.2 Path Topology Signature

For each individual path `x`, compute topology signature `h(x)` (e.g., H-signature or winding-based feature). Let `Delta_h(i,j)` denote topology mismatch between source `x_i` and target `y_j`.

### 3.3 Topology-Constrained OT Transfer

At each transfer step, let source and target candidate sets be `S` and `T`, with marginals `a` and `b`.
We solve:

```
pi* = argmin_{pi in U(a,b)} <pi, C> + epsilon * H(pi)
```

with cost
`C_ij = alpha * d_M(theta_src(i), theta_tgt(j))^2 + beta * Delta_h(i,j) + gamma * Delta_q(i,j)`,
where `Delta_q` captures quality/rank/feasibility mismatch.

`pi*` defines transfer flow mass from source to target candidates.
Here
`U(a,b) = {pi >= 0 | pi * 1 = a, pi^T * 1 = b}`,
and `H(pi)` is the entropy regularizer.
Hyperparameters `alpha`, `beta`, `gamma`, and `epsilon` control manifold distance penalty, topology penalty, quality mismatch penalty, and entropy regularization, respectively.

### 3.4 Practical Propositions

**Proposition 1 (Mismatch-Mass Bound)**  
Given optimal coupling `pi*` and objective `J* = <pi*, C> + epsilon H(pi*)`, and `H(pi) >= 0`, topology mismatch mass is bounded by:
`<pi*, Delta_h> <= J* / beta`.
This gives a direct control knob: increasing `beta` upper-bounds mismatch mass.

**Proposition 2 (Topology-Safe Limit)**  
If a feasible coupling exists using only topology-compatible pairs, then as `beta -> +inf`, the total mass assigned to topology-mismatched pairs tends to zero.

Proof sketches are provided in Appendix A.

### 3.5 Counterfactual Credit for Online Penalty Update

Define transfer credit:
`tau = gain_transfer - gain_no_transfer`,
where `gain_no_transfer` is measured from a small control subgroup evolved without cross-task transfer.
Penalty update:
`beta <- clip(beta + eta * sign(-tau), [beta_min, beta_max])`.

---

## 4. TCOT-CTM-EA Algorithm

### 4.1 Overall Framework

```python
Algorithm: TCOT-CTM-EA

Input: 
  - Target task parameter theta*
  - Auxiliary task pool {theta_k} on manifold
  - Population size: N
  - Max generations: G
  - OT update interval: K_ot

Output: 
  - Final nondominated set on T(theta*)

1. Initialize:
   P <- random population
   theta_cur <- easiest nearby task
   beta <- beta0

2. For t = 1..G:
   a) Update theta_cur using feasibility-aware scheduler
   b) Generate same-task offspring (baseline EA operators)
   c) If mod(t, K_ot) == 0:
        - Build source set S and target set T
        - Compute topology signatures h(.)
        - Build cost matrix C_ij
        - Solve Sinkhorn OT to get coupling pi*
        - Sample cross-task transfer pairs from pi*
        - Generate transferred offspring
   d) Environmental selection on current task
   e) Evaluate small control subgroup (no-transfer) and update beta

3. Return nondominated solutions evaluated on T(theta*)
```

### 4.2 Lightweight Feasibility-Aware Scheduler

Scheduler state:
1. Feasibility rate.
2. Population diversity.
3. Recent target-HV trend.

Rule:
1. Increase task difficulty if feasibility is high and diversity is adequate.
2. Hold or step back if feasibility collapses.
3. Keep update magnitude small to avoid manifold jumps.

### 4.3 Complexity

Per generation (with OT every `K_ot` iterations):
1. Base MOEA cost: `O(N^2)` for nondominated sorting/selection (implementation-dependent).
2. OT step: `O(|S||T| * I_sinkhorn)` where `I_sinkhorn` is Sinkhorn iterations.
3. Topology signature extraction: typically linear in path discretization length.

Because OT is periodic, compute overhead can be bounded by controlling `K_ot` and subset sizes `|S|,|T|`.

---

## 5. Experimental Protocol

### 5.1 Benchmarks

Primary benchmark:
1. Repository UAV scenarios (`c_*`) with added `theta` metadata.

Controlled benchmark:
1. Synthetic manifold sweep over obstacle density, threat intensity, and wind.
2. Separate subsets with mild vs severe homotopy shifts.

### 5.2 Compared Algorithms

1. NMOPSO, MOPSO, NSGA-II, NSGA-III (repository baselines).
2. MFEA, MFEA-II (PlatEMO wrappers).
3. At least two recent transfer-learning EMT baselines (subject to available implementations): MFEA-ML, MTCS, PA-MTEA, or learnable cross-task association.
4. OT variants for ablation (see Section 5.4).

### 5.3 Metrics

Primary:
1. Hypervolume (HV) on target task.
2. IGD / IGD+ where applicable.
3. Feasibility rate.

Transfer diagnostics:
1. Transfer Gain Ratio (TGR).
2. Negative Transfer Incidence (NTI).
3. Homotopy-consistency rate.
4. Topology Mismatch Transfer Mass (TMTM) derived from `pi*`.

Efficiency:
1. Time-to-90%-target-HV.
2. Wall-clock overhead ratio vs non-OT transfer.

### 5.4 Critical Ablations

1. Remove topology term in OT (`beta = 0`).
2. Replace OT coupling with hard gate transfer.
3. Replace continuous manifold navigation with discrete task pair training only.
4. Freeze `beta` (remove counterfactual credit update).
5. Feasibility-aware scheduler vs fixed monotonic schedule.

### 5.5 Statistical and Reproducibility Protocol

1. 30 independent runs per setting (or the maximum feasible within budget).
2. Wilcoxon rank-sum with Holm-Bonferroni correction for multiple comparisons.
3. Report medians, interquartile ranges, and effect sizes (e.g., Cliff's delta).
4. Release run seeds, configuration files, and raw objective archives.
5. Include per-scenario transfer-flow visualizations derived from `pi*`.

### 5.6 Reporting Policy

This draft intentionally avoids fabricated quantitative claims. Numerical tables and plots should be added only after the full protocol is executed.

---

## 6. Discussion

### 6.1 Why the Design Should Work

1. Continuous manifold navigation keeps auxiliary tasks near target structure.
2. OT coupling optimizes transfer flow globally instead of local heuristics.
3. Topology penalty directly targets a major source of negative transfer in cluttered path planning.

### 6.2 Main Failure Modes

1. Topology signature aliasing: different paths map to similar signatures.
2. Excessive penalty `beta`: under-transfer and premature convergence.
3. Very sparse feasible regions: both OT and baseline transfer may fail.

### 6.3 Threats to Validity

1. Sensitivity to signature definition and path discretization.
2. Baseline implementation parity (parameter fairness across algorithms).
3. Additional OT overhead that may offset quality gains under strict runtime budgets.

### 6.4 Scope Boundaries

Paper 1 focuses on a minimal, defensible novelty package:
1. Continuous manifold formulation.
2. Topology-constrained OT transfer.
3. Lightweight scheduler and counterfactual credit.

Bridge-task synthesis, learned manifold metrics, and advanced bandit/lookahead controllers are deferred.

---

## 7. Conclusion

This paper frames EMT transfer for UAV path planning as a **topology-constrained OT coupling problem on a continuous task manifold**. The contribution is not a new low-level optimizer, but an explicit transfer-flow design that is measurable and controllable against topology mismatch.

Immediate deliverables for Paper 1:
1. Implement TCOT-CTM-EA in the repository benchmark stack.
2. Run the full ablation protocol with NTI/TMTM reporting.
3. Demonstrate when topology-constrained transport helps, and when it fails.

With these deliverables completed, the work should be technically defensible and ready for full paper submission.

---

## References

[1] Gupta et al. (2016). "Multifactorial Evolution: Toward Evolutionary Multitasking". IEEE TEVC.

[2] Bali et al. (2020). "Multifactorial Evolutionary Algorithm With Online Transfer Parameter Estimation: MFEA-II". IEEE TEVC.

[3] Liang et al. (2021). "Evolutionary Multitasking for Multi-objective Optimization with Multi-population". EMMOP.

[4] Bengio et al. (2009). "Curriculum Learning". ICML.

[5] Weinshall et al. (2018). "Curriculum Learning by Transfer Learning". ICML.

[6] Pan & Yang (2010). "A Survey on Transfer Learning". IEEE TKDE.

[7] Thrun & Pratt (1998). "Learning to Learn". Springer.

[8] Zhang et al. (2023). "Evolutionary multitasking for constrained multiobjective optimization with global and local auxiliary tasks". IEEE/CAA JAS. doi:10.1109/JAS.2023.123336.

[9] Ma et al. (2024). "Evolutionary multitasking based on constraints separation for constrained multi-objective optimization". IEEE/CAA JAS. doi:10.1109/JAS.2024.124545.

[10] Liu et al. (2024). "A dynamic-multi-task-assisted evolutionary algorithm for constrained multiobjective optimization". Swarm and Evolutionary Computation. doi:10.1016/j.swevo.2024.101683.

[11] Yang et al. (2025). "A dynamic adaptive task transitions multitasking for constrained multi-objective optimization". Complex & Intelligent Systems. doi:10.1007/s40747-025-02154-7.

[12] Zhang et al. (2025). "A coevolutionary multitasking optimization algorithm with dynamic constraint relaxation for constrained multiobjective optimization". Swarm and Evolutionary Computation. doi:10.1016/j.swevo.2025.101954.

[13] Feng et al. (2025). "Learning to transfer for evolutionary multitasking". IEEE Transactions on Cybernetics. doi:10.1109/TCYB.2025.3561518.

[14] Liang et al. (2025). "Multi-objective multi-UAV path planning via evolutionary multitasking optimization with adaptive operator selection and knowledge fusion". Swarm and Evolutionary Computation. doi:10.1016/j.swevo.2025.102145.

[15] Jiang et al. (2021). "A fast dynamic evolutionary multiobjective algorithm via manifold transfer learning". IEEE Transactions on Cybernetics. doi:10.1109/TCYB.2020.2989465.

[16] Yuan et al. (2022). "A Classification-Based Surrogate-Assisted Evolutionary Multitasking Method for Expensive Minimax Optimization Problems". IEEE Transactions on Cybernetics. doi:10.1109/TCYB.2022.3172348.

[17] Xu et al. (2024). "Evolutionary multitasking based on transfer strategy by budget online learning for constrained multiobjective optimization". Swarm and Evolutionary Computation. doi:10.1016/j.swevo.2024.101839.

[18] Li et al. (2025). "Evolutionary multitasking with transfer strategy based on multitask convergence state for constrained multiobjective optimization". IEEE Transactions on Evolutionary Computation. doi:10.1109/TEVC.2025.3598028.

[19] Tan et al. (2026). "Evolutionary multitasking with learnable cross-task association mapping". Expert Systems with Applications. doi:10.1016/j.eswa.2025.127535.

[20] Tan et al. (2026). "A transfer strategy based on probability adaptation and objective-space mapping in multifactorial evolutionary algorithm for expensive constrained optimization". Expert Systems with Applications. doi:10.1016/j.eswa.2025.128515.

[21] Bhattacharya et al. (2015). "Path planning under topological constraints using homology embeddings". Robotics and Autonomous Systems. doi:10.1016/j.robot.2014.10.021.

[22] Liu et al. (2025). "Homotopic trajectory optimization framework for cooperative UAV path planning using Gaussian probability field and grid maps". Machines. doi:10.3390/machines13030227.

[23] Cuturi (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport". NeurIPS. arXiv:1306.0895.

[24] Peyre and Cuturi (2019). "Computational Optimal Transport". Foundations and Trends in Machine Learning. doi:10.1561/2200000073.

[25] Liu et al. (2024). "Coevolutionary multitasking for constrained multiobjective optimization". Swarm and Evolutionary Computation. doi:10.1016/j.swevo.2024.101727.

[26] Dang et al. (2025). "Constrained multi-objective optimization assisted by convergence and diversity auxiliary tasks". Engineering Applications of Artificial Intelligence. doi:10.1016/j.engappai.2024.109546.

---

## Appendix A: Proofs

### A.1 Proof Sketch for Proposition 1 (Mismatch-Mass Bound)

Recall:
`J* = <pi*, C> + epsilon * H(pi*)`,
with
`C_ij = alpha d_M(theta_src(i), theta_tgt(j))^2 + beta Delta_h(i,j) + gamma Delta_q(i,j)`,
and `Delta_h(i,j) >= 0`.

Then:
`J* >= <pi*, C> >= beta * <pi*, Delta_h>`.

Therefore:
`<pi*, Delta_h> <= J* / beta`.

This bound is loose but operational: for fixed `J*`, increasing `beta` contracts an upper bound on topology mismatch transfer mass.

### A.2 Proof Sketch for Proposition 2 (Topology-Safe Limit)

Assume there exists at least one feasible coupling `pi_feas` whose support contains only topology-compatible pairs (`Delta_h = 0`).

For any coupling with nonzero mismatch mass `m > 0`, the OT objective contains additive penalty at least `beta * m`. As `beta -> +inf`, any optimizer minimizing the objective must drive `m -> 0`; otherwise objective diverges relative to feasible zero-mismatch alternatives.

Hence mismatch mass in optimal couplings converges to zero in the large-`beta` limit (under feasibility of compatible couplings).

### A.3 Notes on Practical Validity

1. The propositions describe trend guarantees, not finite-sample performance guarantees.
2. If topology signatures are noisy/aliased, effective `Delta_h` may be biased.
3. Empirical checks with TMTM and NTI are required to validate behavior in practice.

---

## Appendix B: Benchmark Instantiation Details

### B.1 Scenario Metadata Schema (`theta`)

Each scenario stores:
1. `rho_obs`: normalized obstacle density.
2. `threat_scale`: normalized threat intensity.
3. `nofly_ratio`: ratio of prohibited cells/regions.
4. `wind_level`: normalized wind disturbance level.
5. `terrain_roughness`: normalized terrain complexity.

All dimensions are normalized to `[0,1]` and serialized with scenario files for reproducible manifold distance computation.

### B.2 Topology Signature Interface

Given path polyline `P = [p1, ..., pL]`, signature extractor returns:
1. `h_id`: discrete class label (if available).
2. `h_vec`: continuous feature vector for soft mismatch computation.

Mismatch examples:
1. Hard: `Delta_h = 1[h_id_i != h_id_j]`.
2. Soft: cosine or Euclidean distance over `h_vec`.

### B.3 Transfer Log Format

Per OT update, log:
1. `theta_src`, `theta_tgt`.
2. Coupling matrix summary (`mass_diag`, `mass_offdiag`, entropy).
3. `TMTM` and `NTI`.
4. Current `beta`.
5. Delta target-HV after transfer step.

### B.4 Minimal Reproducibility Checklist

1. Fixed random seeds list.
2. Full algorithm config dump per run.
3. Raw objective archives and feasibility traces.
4. OT solver tolerance and max-iteration settings.
5. Postprocessing scripts for tables/figures.

---

**End of Paper 1 Draft (TCOT-CTM-EA version)**
