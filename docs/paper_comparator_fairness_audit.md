# Paper Comparator Fairness Audit

This audit covers the active `cgpo-swec` comparison profile.

## Selected Comparator Set

| Algorithm | Status | Fairness treatment |
| --- | --- | --- |
| `CGPO` | Experimental, repository implementation | Optimizes the shared Python benchmark objective during search. |
| `NSGA-II` | Benchmark-safe baseline | Optimizes the shared Python benchmark objective during search with feasibility-first constraint handling. |
| `MOEAD` | Benchmark-safe baseline | Optimizes the shared Python benchmark objective during search with weight-vector decomposition. |
| `NMOPSO` | Benchmark-safe baseline | Optimizes the shared Python benchmark objective during search through the shared fleet PSO engine. |
| `CCEA-ADVS` | Experimental, clean-room implementation | Optimizes the shared Python benchmark objective during search. Dubins refinement is labeled `benchmark_approximation`. |
| `TSKAC-NSGA-II` | Experimental, clean-room implementation | Optimizes the shared Python benchmark objective during the main task search; auxiliary-task transfer is reported as metadata. |
| `DTAPP-IICR` | Experimental official C++ reference adapter | Not an EA/MOEA optimizer comparator. It is a MAPF/preflight baseline whose final paths are re-scored by the shared Python benchmark evaluator. Report separately. |

## Protocol Consistency

The main paper profile is `cgpo-swec` and expands to:

`CGPO`, `NSGA-II`, `MOEAD`, `NMOPSO`, `CCEA-ADVS`, `TSKAC-NSGA-II`, `DTAPP-IICR`.

The head-to-head config uses:

- `population: 100`
- `generations: 500`
- `seed: 11`
- `fleetSizes: [1, 3, 5]`
- `problemNames: ["c_100", "m_100", "s_120"]`
- `computeMetrics: true`
- `hardCollisionConstraint: true`

`DTAPP-IICR` shares the scenario and final scoring protocol, but not the native
population/generation loop. Its run metadata records this explicitly.

## Metadata Requirements

Runnable optimizer comparators must save standard benchmark artifacts through
`_save_fleet_artifacts` and record:

- `algorithmName`
- `optimizerBackend`
- `pythonProblemEvaluation`
- `benchmarkObjectiveDuringSearch`
- `nativePopulationLoop`
- `nativeGenerationLoop`

For official/reference adapters, metadata must also disclose whether final paths
were re-scored by Python and whether a fair evaluation shim was used.

## Current Limitations

- `CCEA-ADVS` has no official source code available locally, so it is a
  clean-room implementation from the paper mechanics.
- `CCEA-ADVS` phase-2 Dubins refinement is an approximation until an exact 3D
  Dubins primitive is added.
- `DTAPP-IICR` is not an evolutionary or multi-objective optimizer. It should be
  reported as a separate airspace/planning baseline, not ranked as if it used the
  same optimizer budget.
