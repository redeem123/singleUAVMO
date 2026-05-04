# Proposal: CGPO for Constrained Multi-UAV Planning

## Position

The paper-facing method is **CGPO: Constraint-Graph Policy Optimizer**.

For a fair MOEA comparison, CGPO is presented as an evolutionary algorithm, not
as a hybrid planner with privileged route priors or an explicit repair stack:

> CGPO uses a constraint-interaction graph to shape parent selection and
> fleet-aware variation inside a population-based multi-objective evolutionary
> algorithm for constrained UAV path planning.

The active repository implementation has removed A*/ramp constructive seeds,
explicit repair, altitude relaxation, graph-guided feasibility projection,
CGPO-R, and matched CGPO hybrid baselines from the supported CGPO surface.

## Public Method Names

Paper-facing runs should use:

```text
CGPO
```

Mechanism and diagnostic ablations should use:

```text
CGPO_full
CGPO_random_only
CGPO_no_cig_edge_coupling
CGPO_no_ppf_pressure
CGPO_no_ovo_variation
CGPO_no_ovo_fleet_coordination
```

`CGPO_full` is the ablation label for the main paper method. Removed prototype
labels such as `with_repair`, `repair_only`, `gfp_only`, and
`constructive_only` should not appear in current paper configs.

## Core Method Story

### 1. Evolutionary Search

The optimizer uses the same benchmark-facing evolutionary budget as the MOEA
baselines: population size, generations, run count, problem set, constraints,
and metrics are shared. The main CGPO run starts from stochastic candidates
rather than A*/corridor or ramp candidates.

### 2. Constraint-Interaction Graph Guidance

CIG represents waypoint-level pressure from terrain clearance, obstacle
proximity, turn angle, route smoothness, and inter-UAV separation. In the
MOEA-style method, the graph is used to guide parent pressure and variation
instead of injecting domain-specific feasible routes.

### 3. Constraint Handling Without Privileged Repair

Explicit repair and graph-guided feasibility projection are not part of the
main method or current benchmark registration. Constraint handling is expressed
through selection pressure, variation, domain-bound clipping, and the shared
mission evaluator.

## Fair Comparison Requirement

Because the main claim is an evolutionary algorithm claim, the decisive
comparison is against MOEA baselines under the same population, generation, run,
problem, and constraint protocol. Hybrid baselines are no longer part of the
active paper protocol.

## Ablation Plan

Use the ablation config to separate mechanism contribution from dominance:

| Variant | Purpose |
| --- | --- |
| `full` | main paper method: CIG + PPF + OVO |
| `random_only` | disables all three graph-guided mechanisms |
| `no_cig_edge_coupling` | tests marginal value of CIG edge coupling |
| `no_ppf_pressure` | tests marginal value of pressure-based parent selection |
| `no_ovo_variation` | tests marginal value of OVO variation |
| `no_ovo_fleet_coordination` | tests marginal value of fleet-aware coordination |

Decision rule:

- If `CGPO_full` is competitive with MOEA baselines, CGPO is defensible
  as an evolutionary algorithm contribution.
- If `random_only` matches `CGPO_full`, the graph-guided mechanism story is not
  supported under the tested budget.
- If `no_ppf_pressure` or `no_ovo_fleet_coordination` matches `CGPO_full`, simplify
  the method and demote the unsupported component.

## Reviewer-Safe Contribution List

Use this contribution framing unless stronger paper-grade evidence appears:

1. A population-based constrained MOEA, CGPO, for multi-UAV path planning with
   hard terrain, turn, obstacle, and separation checks.
2. A constraint-interaction graph used to guide selection pressure and
   fleet-aware offspring generation.
3. A fair MOEA comparison under shared population, generation, run, problem,
   constraint, and metric settings.
4. A dominance-aware ablation protocol that separates CIG, PPF, OVO, and
   fleet-coordination effects.

## Claims To Avoid

Do not claim:

- every CIG/PPF/OVO component is essential before ablation evidence supports it;
- repair, graph-guided feasibility projection, constructive priors, CGPO-R, or
  matched hybrid baselines are part of the active method;
- CGPO is state of the art without paper-grade statistics.

## Required Paper Evidence

Before making a strong claim, run:

- `configs/paper_cgpo_swec_head2head.yaml` for the main MOEA-style CGPO
  comparison;
- `configs/paper_cgpo_swec_ablation.yaml` for mechanism ablation;
- `configs/cgpo_component_dominance_s120.yaml` before expensive paper runs to
  detect overengineered variants early;
- `scripts/analyze_cgpo_ablation.py` with `CGPO_full` as the ablation
  baseline.

Report feasibility, HV, IGD+, conflict rate, runtime, candidate evaluation
counts, initial feasible ratio, and CIG/PPF/OVO trace summaries. If the evidence
does not support a component, simplify the method rather than preserving the
term for appearance.
