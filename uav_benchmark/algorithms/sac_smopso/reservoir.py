from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.algorithms.cmosma import _environmental_selection as _cmosma_environmental_selection
from uav_benchmark.algorithms.sac_smopso.geometry import _conflict_repair_step, _violation_value
from uav_benchmark.algorithms.sac_smopso.initialization import _model_constraint_values
from uav_benchmark.algorithms.sac_smopso.scoring import _reservoir_score
from uav_benchmark.algorithms.shared.nmopso_engine import NMOPSOEngine, _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate


def _sbx_crossover_pm(
    parents: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sbx_prob: float = 1.0,
    sbx_eta: float = 20.0,
    pm_prob: float | None = None,
    pm_eta: float = 20.0,
) -> np.ndarray:
    """Self-contained SBX crossover + polynomial mutation.

    Mirrors NSGA-II / CMOSMA defaults. We keep a local copy (instead of
    importing ``_sbx_mutation`` from the shared engine module) because
    that helper depends on constants that are no longer exported in the
    current repo layout. The implementation itself is standard Deb-style
    SBX + PM and does not introduce new hyper-parameters the paper has
    to justify.
    """
    parents = np.atleast_2d(np.asarray(parents, dtype=float))
    n, d = parents.shape
    if n < 2:
        return parents.copy()
    pairs = (n // 2) * 2
    p1 = parents[0:pairs:2]
    p2 = parents[1:pairs:2]
    u = np.random.rand(p1.shape[0], d)
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (sbx_eta + 1)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (sbx_eta + 1)),
    )
    cx_mask = np.random.rand(p1.shape[0], d) < sbx_prob
    c1 = np.where(cx_mask, 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2), p1)
    c2 = np.where(cx_mask, 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2), p2)
    offspring = np.vstack([c1, c2])
    if pm_prob is None:
        pm_prob = 1.0 / float(max(1, d))
    pm_mask = np.random.rand(*offspring.shape) < pm_prob
    if np.any(pm_mask):
        u_mut = np.random.rand(*offspring.shape)
        delta = np.where(
            u_mut < 0.5,
            (2.0 * u_mut) ** (1.0 / (pm_eta + 1)) - 1.0,
            1.0 - (2.0 * (1.0 - u_mut)) ** (1.0 / (pm_eta + 1)),
        )
        span = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
        span = np.where(span == 0, 1.0, span)
        offspring = np.where(pm_mask, offspring + delta * span, offspring)
    return np.clip(offspring, lower, upper)


def _refresh_unconstrained_population(
    aux_state: dict[str, Any],
    *,
    fresh_vectors: np.ndarray,
    fresh_candidates: list[Candidate],
    capacity: int,
    model: dict[str, Any],
) -> None:
    """Maintain a lean unconstrained companion population.

    This gives SAC-SMOPSO a stronger CMOSMA-style auxiliary search stream
    without introducing the full SOM machinery. The auxiliary population is
    updated with unconstrained environmental selection every generation.
    """
    if capacity <= 0:
        aux_state["vectors"] = np.zeros((0, 0), dtype=float)
        aux_state["candidates"] = []
        return
    current_vectors = np.asarray(aux_state.get("vectors", np.zeros((0, 0), dtype=float)), dtype=float)
    current_candidates = list(aux_state.get("candidates", []))

    merged_candidates: list[Candidate] = current_candidates + list(fresh_candidates)
    if not merged_candidates:
        aux_state["vectors"] = np.zeros((0, 0), dtype=float)
        aux_state["candidates"] = []
        return

    vector_blocks: list[np.ndarray] = []
    if current_vectors.ndim == 2 and current_vectors.size > 0:
        vector_blocks.append(current_vectors)
    fresh_matrix = np.asarray(fresh_vectors, dtype=float)
    if fresh_matrix.ndim == 2 and fresh_matrix.size > 0:
        vector_blocks.append(fresh_matrix)
    merged_vectors = np.vstack(vector_blocks) if vector_blocks else np.zeros((0, 0), dtype=float)

    seen: set[bytes] = set()
    unique_vectors: list[np.ndarray] = []
    unique_candidates: list[Candidate] = []
    for idx, candidate in enumerate(merged_candidates):
        vec = np.asarray(getattr(candidate, "vector", []), dtype=float)
        if vec.size == 0:
            if idx < merged_vectors.shape[0]:
                vec = np.asarray(merged_vectors[idx], dtype=float)
            else:
                continue
        key = vec.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique_vectors.append(vec.copy())
        unique_candidates.append(candidate)
    if not unique_candidates:
        aux_state["vectors"] = np.zeros((0, 0), dtype=float)
        aux_state["candidates"] = []
        return

    separation_min, drone_size, max_turn_deg = _model_constraint_values(model)
    if len(unique_candidates) <= int(capacity):
        order = np.argsort(
            np.asarray(
                [
                    _reservoir_score(
                        candidate,
                        separation_min=separation_min,
                        drone_size=drone_size,
                        max_turn_deg=max_turn_deg,
                    )
                    for candidate in unique_candidates
                ],
                dtype=float,
            )
        )
        aux_state["vectors"] = np.stack([unique_vectors[int(i)] for i in order], axis=0)
        aux_state["candidates"] = [unique_candidates[int(i)] for i in order]
        return

    next_vectors, next_candidates, _fitness = _cmosma_environmental_selection(
        np.stack(unique_vectors, axis=0),
        unique_candidates,
        int(capacity),
        use_constraints=False,
        model=model,
    )
    aux_state["vectors"] = np.asarray(next_vectors, dtype=float).copy()
    aux_state["candidates"] = list(next_candidates)


def _reservoir_sbx_injection(
    engine: NMOPSOEngine,
    aux_state: dict[str, Any],
    sbx_weight: float,
    repair_intensity: float,
    fleet_size: int,
    n_waypoints: int,
    lower: np.ndarray,
    upper: np.ndarray,
    aux_capacity: int,
) -> dict[str, float]:
    """Policy-scheduled SBX step mating population with the reservoir.

    This is the *only* multi-UAV generation operator when selected (see
    ``_run_fleet_sac_smopso``). It does three things in one pass:

      1. Draw parent pairs from (population, reservoir), mixing in a
         reservoir parent with probability ``sbx_weight``.
      2. SBX + polynomial-mutation crossover to produce offspring.
      3. Policy-scheduled Gaussian *conflict repair* over the raw
         offspring (``_conflict_repair_step``). Repaired children are
         kept if they reduce ``_violation_value``; otherwise the plain
         SBX child is used.

    Offspring are evaluated, pushed into the constrained archive, and
    the worst ``replace_count`` population slots (ranked by
    ``_reservoir_score``) are replaced by the best offspring.
    """
    stats = {"effectCount": 0.0, "evalCount": 0.0}
    if sbx_weight <= 1e-3 or engine.pop_size < 4:
        return stats

    ratio = 0.40 + 0.85 * float(sbx_weight)  # → up to ~1.25× pop
    inject_count = max(2, int(round(engine.pop_size * ratio)))
    if inject_count % 2 == 1:
        inject_count += 1
    inject_count = min(inject_count, engine.pop_size)

    pop = np.asarray(engine.population, dtype=float)
    first_parents = pop[np.random.randint(0, engine.pop_size, size=inject_count)]
    second_parents = pop[np.random.randint(0, engine.pop_size, size=inject_count)]
    aux_vectors = np.asarray(aux_state.get("vectors", np.zeros((0, 0), dtype=float)), dtype=float)
    if aux_vectors.ndim == 2 and aux_vectors.size > 0:
        use_res = np.random.rand(inject_count) < float(sbx_weight)
        picks = np.random.randint(0, aux_vectors.shape[0], size=inject_count)
        second_parents = np.where(
            use_res[:, None],
            aux_vectors[picks],
            second_parents,
        )

    parents = np.empty((inject_count * 2, pop.shape[1]), dtype=float)
    parents[0::2] = first_parents
    parents[1::2] = second_parents

    offspring = _sbx_crossover_pm(parents, lower, upper)
    if offspring.shape[0] > inject_count:
        offspring = offspring[:inject_count]

    # Policy-scheduled conflict-repair pass (Gaussian perturbation of
    # every offspring). Offspring that do not improve under the simple
    # violation metric fall back to the raw SBX child.
    repaired = _conflict_repair_step(
        engine=engine,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
        offspring=offspring,
        repair_intensity=repair_intensity,
    )
    if repaired is not offspring and repaired.shape == offspring.shape:
        # Evaluate both variants and keep whichever has lower violation.
        try:
            raw_cands = engine._evaluate(offspring)  # noqa: SLF001
            rep_cands = engine._evaluate(repaired)  # noqa: SLF001
        except (KeyError, IndexError, RuntimeError, TypeError, ValueError):
            raw_cands, rep_cands = [], []
        if raw_cands and rep_cands and len(raw_cands) == len(rep_cands):
            chosen: list[Candidate] = []
            chosen_vec = np.empty_like(offspring)
            for i, (rc, pc) in enumerate(zip(raw_cands, rep_cands, strict=False)):
                raw_v = _violation_value(rc.details if isinstance(rc.details, dict) else {})
                rep_v = _violation_value(pc.details if isinstance(pc.details, dict) else {})
                if rep_v + 1e-4 < raw_v:
                    chosen.append(pc)
                    chosen_vec[i] = repaired[i]
                else:
                    chosen.append(rc)
                    chosen_vec[i] = offspring[i]
            off_candidates = chosen
            offspring = chosen_vec
            stats["evalCount"] = float(2 * len(chosen))
        else:
            off_candidates = raw_cands
            stats["evalCount"] = float(len(off_candidates))
    else:
        try:
            off_candidates = engine._evaluate(offspring)  # noqa: SLF001 (stable internal)
        except (KeyError, IndexError, RuntimeError, TypeError, ValueError):
            return stats
        stats["evalCount"] = float(len(off_candidates))
    if not off_candidates:
        return stats

    engine.update_archive(off_candidates)
    if not engine.candidates:
        return stats
    previous_population = np.asarray(engine.population, dtype=float).copy()
    merged_vectors = np.vstack([engine.population, offspring])
    merged_candidates = list(engine.candidates) + list(off_candidates)
    next_vectors, next_candidates, _fitness = _cmosma_environmental_selection(
        merged_vectors,
        merged_candidates,
        engine.pop_size,
        use_constraints=True,
        model=engine.model,
    )
    if next_vectors.size == 0 or not next_candidates:
        return stats
    effect_count = 0
    for row in np.asarray(next_vectors, dtype=float):
        if not np.any(np.all(np.isclose(previous_population, row.reshape(1, -1), rtol=0.0, atol=1e-12), axis=1)):
            effect_count += 1
    engine.population = np.asarray(next_vectors, dtype=float).copy()
    engine.candidates = list(next_candidates)
    engine.current_obj = _candidate_matrix(engine.candidates)
    if engine.velocity.shape == engine.population.shape:
        engine.velocity[:] = 0.0
    else:
        engine.velocity = np.zeros_like(engine.population)
    engine.update_archive(engine.candidates)
    _refresh_unconstrained_population(
        aux_state,
        fresh_vectors=np.vstack([engine.population, offspring]),
        fresh_candidates=list(engine.candidates) + list(off_candidates),
        capacity=aux_capacity,
        model=engine.model,
    )
    stats["effectCount"] = float(effect_count)
    return stats
