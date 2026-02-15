# Multi-UAV Protocol (Paper Medium)

## Mission Model

- Homogeneous point-to-point missions.
- Fleet sizes: `3`, `5`, `8`.
- Objectives (minimize):
  1. Mission makespan.
  2. Total energy surrogate.
  3. Aggregate risk penalty.
  4. Coordination conflict penalty.

## Constraints

- Terrain and map boundary constraints.
- Threat/no-fly safety constraints.
- Altitude bounds from terrain model (`zmin`, `zmax`).
- Turn-angle hard bound (`maxTurnDeg`).
- Inter-UAV minimum separation (`separationMin`).

## Scenario Set: `paper_medium`

Base terrains:

- `c_100`
- `c_150`
- `c_100_20_nofly`
- `c_70_40_nofly`
- `m_100`
- `m_200`
- `m_100_30c_nofly`
- `m_200_20c_nofly`
- `s_120`
- `s_180`
- `s_110_20_nofly`
- `s_80_40_nofly`

Generated multi-UAV problems are saved as:

- `terrainStruct_<base>_uav3.mat`
- `terrainStruct_<base>_uav5.mat`
- `terrainStruct_<base>_uav8.mat`

## Algorithms

- `RL-NMOPSO`
- `NMOPSO`
- `MOPSO`
- `NSGA-II`

## Statistical Protocol

- 10 runs per scenario/algorithm.
- Mann-Whitney U + Holm correction.
- Baseline for pairwise and win/tie/loss: `NMOPSO`.

## GPU

- `--gpu-mode auto`: use CuPy or PyTorch GPU backend if available.
- `--gpu-mode off`: force CPU.
- `--gpu-mode force`: require GPU backend; runtime metadata still records backend status in `run_stats.mat`.
