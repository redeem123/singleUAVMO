# MATLAB Reference Algorithm Bridge

Python is the benchmark driver for this repository. MATLAB reference algorithms are allowed only as algorithm-mechanics backends: they may keep their official initialization, variation, repair, selection, ranking, archiving, and stopping logic, but objective and constraint evaluation must go through the Python UAV benchmark evaluator during search.

## Contract

- Python owns benchmark configuration, terrain/problem loading, run folders, artifact format, metrics, and final reporting.
- MATLAB owns only the reference algorithm mechanics for MATLAB-based competitors.
- MATLAB must call `BridgeEvaluateInPython.m` for every optimizer-facing objective/constraint evaluation.
- Final-path rescoring is allowed for audit artifacts, but it does not make an algorithm fair unless the same Python benchmark objective vector was used during search.
- Official reference code should be preserved as much as practical. Prefer shimming the smallest evaluation method or problem adapter instead of rewriting operators.

## Supported Candidate Payloads

`BridgeEvaluateInPython.m` writes a request `.mat` file and calls `uav_benchmark.platemo_bridge.evaluator`.

- `PopDec`: decision matrix used by PlatEMO-style algorithms.
- `Decs`: legacy alias for decision matrices.
- `PathStack`: path tensor used by reference UAV codes that operate directly on paths.

All responses contain `PopObj` with the repository four-objective optimizer vector and `PopCon` with benchmark constraint violation.

Path-native MATLAB code should use `BridgeEvaluatePathPopulation.m` when it has a class array with `.path`, `.objs`, and `.cons` fields. It batches a whole population into one `PathStack` request.

## Current Patterns

- Modern PlatEMO algorithms use `PythonBridgeProblem`, which forwards `PopDec`.
- Older GLOBAL-style PlatEMO algorithms use `PythonBridgeLegacyProblem`, which also forwards `PopDec`.
- MOEA-2DE keeps the official MATLAB operators/selection and shadows the evaluation loop so initial populations, dimension probes, and offspring are forwarded as batched `PathStack` requests.
- EMMOP uses `PythonBridgeLegacyProblem` plus narrow DQN compatibility shims because the official DQN state assumed exactly two objectives and a large replay buffer. Selection and variation remain in the official MATLAB algorithm.

New MATLAB-based comparators should follow one of these patterns and should not call paper-specific objective functions for optimizer selection unless those functions have been replaced by `BridgeEvaluateInPython.m`.
