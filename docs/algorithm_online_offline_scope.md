# Algorithm Online/Offline Scope

This file records the paper-facing algorithm scope. Here, `online` means the
method is part of the active comparison narrative or benchmark protocol.
`offline` means the method is retained only as background, a deferred comparator,
or a future wiring target. This is not the same as online replanning in a
dynamic flight controller.

## Online Algorithms

| Scope | Algorithm | Family | Core mechanism | Paper role |
| --- | --- | --- | --- | --- |
| Generic MOEA | `NSGA-II` | Evolutionary | Pareto sorting and crowding distance | Standard baseline for multi-objective optimization |
| Generic MOEA | `MOEAD` | Decomposition | Weight-vector decomposition | Strong convergence baseline in UAV benchmarks |
| UAV-specific MOEA | `NMOPSO` | PSO-based | Navigation-variable encoding with kinematic constraints | Widely used UAV path-planning baseline |
| UAV-specific MOEA | `MOEA-2DE` | Evolutionary | Dimension exploration and discrepancy evolution | Strong convergence/diversity in UAV scenarios |
| UAV-specific MOEA | `EMMOP` | Evolutionary multitasking | Knowledge transfer across tasks | Recent SOTA UAV multi-objective planner |
| Constraint handling | `NSGA-II` with Deb rules | Evolutionary | Feasibility-first constraint handling | Standard constrained-optimization baseline |
| Multi-UAV optimization | `CCEA-ADVS` | Co-evolutionary | Decomposition across UAV decision variables | Scalable multi-UAV path-planning method |
| Multi-UAV optimization | `TSKAC-NSGA-II` | Co-evolution + RL | Two-stage cooperative optimization | Handles multi-UAV safety and energy trade-offs |
| Separate MAPF / preflight baseline | `DTAPP-IICR` | MAPF-based | Delivery-time prioritized planning plus iterative conflict resolution | Represents realistic multi-UAV airspace coordination and must be reported separately from MOEA-style optimizers |

## Offline Algorithms

All methods not listed above are offline for the current paper scope. Existing
repository algorithms such as `L-SHADE-CDP`, `C-TAEA`, `GCNMOEA`,
`MO-MFEA-II`, `MOEAD-AWA-ASTAR`, `SAC-SMOPSO`, `RA-SMPSO`, `RA-NSGA-II`, and
the MOGWO variants should be discussed only as background, historical artifacts,
or future/deferred comparators unless the paper protocol is intentionally
expanded again.

## Wiring Notes

Several online paper-scope methods are already runnable in the repository:
`NSGA-II`, `MOEAD`, `NMOPSO`, `MOEA-2DE`, `EMMOP`, `CCEA-ADVS`, and
`TSKAC-NSGA-II`.
`DTAPP-IICR` is wired through the official C++ reference implementation in
`research/reference_code/uav_comparators/4DPlanning` and re-scored by the
Python benchmark evaluator. `CCEA-ADVS` is wired as a clean-room experimental
implementation because no official source code was available locally; its
second-stage Dubins refinement is labeled as a benchmark approximation in run
metadata.
