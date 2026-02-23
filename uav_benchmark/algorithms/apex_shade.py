"""APEX-SHADE: Adaptive Pareto-Elite X-SHADE for Multi-UAV Path Planning.

Key innovations over baseline algorithms:
  1. L-SHADE adaptive parameter control (history-based F/CR via Lehmer mean)
  2. Constraint-Domination Principle (CDP) — proper hard constraint satisfaction
  3. Opposition-Based Learning (OBL) initialization for 2x better diversity
  4. Vectorized DE/current-to-pbest/1 mutation with external historical archive
  5. R2-indicator archive for spread-aware Pareto front maintenance
  6. Gaussian elite local search around top archive members
  7. Linear population reduction (L-SHADE) to focus budget on refinement
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.algorithms.multi_uav import (
    _build_bounds,
    _constraint_violation,
    _constraint_violation_vector,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _save_multi_artifacts,
    _should_write_final_hv,
    _ensure_multi_endpoints,
)
from uav_benchmark.algorithms.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.pso_types import Candidate
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import n_d_sort, crowding_distance
from uav_benchmark.core.nsga3_ops import uniform_point
from uav_benchmark.core.r2_archive import r2_archive_update, uniform_weight_vectors
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


# ────────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────────
_H = 10          # L-SHADE history size
_P_BEST = 0.11   # fraction for current-to-pbest mutation
_ARC_RATIO = 2.6 # external DE archive size ratio (relative to n_init)


# ────────────────────────────────────────────────────────────────────
# CDP Selection: Constraint-Domination Principle
# ────────────────────────────────────────────────────────────────────

def _cdp_sort_indices(obj: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """Sort population indices by CDP: feasible first (by NSGA-rank + crowding),
    then infeasible sorted by constraint violation ascending."""
    feas_mask = cv <= 0.0
    feas_idx  = np.where( feas_mask)[0]
    infeas_idx = np.where(~feas_mask)[0]

    # Sort infeasible by violation ascending
    if infeas_idx.size > 1:
        infeas_idx = infeas_idx[np.argsort(cv[infeas_idx])]

    # Sort feasible by non-dominated rank + crowding distance
    if feas_idx.size > 1:
        fobj = obj[feas_idx]
        # Guard against all-inf objectives
        finite_mask = np.all(np.isfinite(fobj), axis=1)
        if np.any(finite_mask):
            fronts, _ = n_d_sort(fobj.copy(), None, feas_idx.size)
            cd = crowding_distance(fobj, fronts)
            order = np.lexsort((-cd, fronts))
            feas_idx = feas_idx[order]

    if feas_idx.size == 0:
        return infeas_idx
    if infeas_idx.size == 0:
        return feas_idx
    return np.concatenate([feas_idx, infeas_idx])


def _cdp_wins(t_cv: float, p_cv: float, t_obj: np.ndarray, p_obj: np.ndarray) -> bool:
    """Return True if trial wins over parent under CDP."""
    t_feas = t_cv <= 0.0
    p_feas = p_cv <= 0.0
    if t_feas and p_feas:
        # Pareto dominance (trial must dominate)
        return bool(np.all(t_obj <= p_obj) and np.any(t_obj < p_obj))
    if t_feas and not p_feas:
        return True   # feasible beats infeasible always
    if p_feas and not t_feas:
        return False  # infeasible never beats feasible
    return t_cv < p_cv  # both infeasible: less violation wins


# ────────────────────────────────────────────────────────────────────
# SHADE Parameter Adaptation (Success-History based)
# ────────────────────────────────────────────────────────────────────

class SHADEMemory:
    """History memory for adaptive F and CR with Lehmer mean update."""

    def __init__(self, H: int = _H) -> None:
        self.H = H
        self.M_F  = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self._k = 0

    def sample(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample F ~ Cauchy(M_F, 0.1) clipped to (0,1], CR ~ N(M_CR, 0.1) clipped to [0,1]."""
        r = np.random.randint(0, self.H, size=size)
        # Vectorised Cauchy sampling — retry negatives component-wise
        F = np.zeros(size)
        for i in range(size):
            fi = -1.0
            trials = 0
            while fi <= 0.0 and trials < 50:
                fi = float(np.random.standard_cauchy() * 0.1 + self.M_F[r[i]])
                trials += 1
            F[i] = min(max(fi, 1e-4), 1.0)
        CR = np.clip(np.random.normal(self.M_CR[r], 0.1), 0.0, 1.0)
        return F, CR

    def update(self, S_F: list[float], S_CR: list[float],
               S_delta: list[float]) -> None:
        """Update with successful parameters, weighted by improvement magnitude."""
        if not S_F:
            return
        Fa  = np.array(S_F,     dtype=float)
        CRa = np.array(S_CR,    dtype=float)
        da  = np.array(S_delta, dtype=float)
        w   = da / (da.sum() + 1e-12)
        self.M_F[self._k]  = float((w * Fa**2).sum() / ((w * Fa).sum() + 1e-12))
        self.M_CR[self._k] = float((w * CRa).sum())
        self._k = (self._k + 1) % self.H


# ────────────────────────────────────────────────────────────────────
# Vectorised DE/current-to-pbest/1 mutation
# ────────────────────────────────────────────────────────────────────

def _de_mutation_vectorised(
    pop: np.ndarray,
    obj: np.ndarray,
    cv: np.ndarray,
    ext_arc: np.ndarray,
    F: np.ndarray,
    p_best_ratio: float,
) -> np.ndarray:
    """Fully vectorised DE/current-to-pbest/1 with external archive."""
    n, d = pop.shape

    # Build p-best pool from feasible, fall back to all by violation
    feas = np.where(cv <= 0.0)[0]
    if feas.size >= 2:
        scores = np.sum(np.where(np.isfinite(obj[feas]), obj[feas], 1e9), axis=1)
        ranked = feas[np.argsort(scores)]
    else:
        ranked = np.argsort(cv)  # all by violation asc
    p_size = max(2, int(p_best_ratio * len(ranked)))
    pbest_pool = ranked[:p_size]

    # Combined pop + external archive for r1, r2 donors
    combined = np.vstack([pop, ext_arc]) if ext_arc.shape[0] > 0 else pop
    nc = combined.shape[0]

    # Vectorised: sample pbest, r1, r2 for all individuals at once
    pb_idx = pbest_pool[np.random.randint(0, p_size, size=n)]

    # Sample r1, r2 unique from combined for each row (approx — avoid collision)
    r1 = np.random.randint(0, nc, size=n)
    r2 = np.random.randint(0, nc, size=n)

    # Fix collisions with parent index (for first n elements)
    parent_idx = np.arange(n)
    for it in range(3):  # 3 attempts to avoid trivial collisions
        bad1 = (r1 == parent_idx) | (r1 == pb_idx)
        bad2 = (r2 == parent_idx) | (r2 == r1) | (r2 == pb_idx)
        if np.any(bad1):
            r1[bad1] = np.random.randint(0, nc, size=int(bad1.sum()))
        if np.any(bad2):
            r2[bad2] = np.random.randint(0, nc, size=int(bad2.sum()))

    # Mutation: xi + F*(xpb - xi) + F*(xr1 - xr2)
    F_col = F[:, None]
    mutants = pop + F_col * (combined[pb_idx] - pop) + F_col * (combined[r1] - combined[r2])
    return mutants


def _binomial_crossover(
    pop: np.ndarray,
    mutants: np.ndarray,
    CR: np.ndarray,
) -> np.ndarray:
    """Binomial crossover — guaranteed at least one dimension crosses."""
    n, d = pop.shape
    mask = np.random.rand(n, d) < CR[:, None]
    j_rand = np.random.randint(0, d, size=n)
    mask[np.arange(n), j_rand] = True
    return np.where(mask, mutants, pop)


# ────────────────────────────────────────────────────────────────────
# R2 Pareto Archive Update
# ────────────────────────────────────────────────────────────────────

def _update_pareto_archive(
    archive: list[Candidate],
    new_cands: list[Candidate],
    max_size: int,
    r2_weights: np.ndarray,
    z_ideal: np.ndarray,
    model: dict,
) -> tuple[list[Candidate], np.ndarray]:
    """Merge new candidates into Pareto archive, pruned by R2 contribution.

    Only feasible individuals enter the archive. If none exist yet, keep the
    best-violation infeasible ones as placeholders (they're replaced as soon
    as feasible solutions are found).
    """
    feasible_new = [c for c in new_cands if float(c.details.get("feasible", 0.0)) > 0.5]

    all_cands = list(archive) + feasible_new
    if not all_cands:
        # No feasible anywhere — keep least-violating infeasible as placeholders
        combined = list(archive) + list(new_cands)
        if not combined:
            return [], z_ideal.copy()
        combined.sort(key=lambda c: _constraint_violation(c, model))
        return combined[:max_size], z_ideal.copy()

    obj_all = np.stack([c.objective for c in all_cands])
    vec_all = np.stack([c.vector   for c in all_cands])
    n_arch  = len(archive)

    new_obj, new_vec, z_out = r2_archive_update(
        archive_obj      = obj_all[:n_arch] if archive else np.zeros((0, obj_all.shape[1])),
        archive_vectors  = vec_all[:n_arch] if archive else np.zeros((0, vec_all.shape[1])),
        candidate_obj    = obj_all[n_arch:],
        candidate_vectors= vec_all[n_arch:],
        max_size=max_size,
        weights=r2_weights,
        z_ideal=z_ideal,
    )

    # Rebuild Candidate list by matching returned vectors to originals
    used: set[int] = set()
    kept: list[Candidate] = []
    for i in range(new_vec.shape[0]):
        matched = False
        for j, c in enumerate(all_cands):
            if j not in used and np.allclose(c.vector, new_vec[i], atol=1e-10):
                kept.append(c); used.add(j); matched = True; break
        if not matched:
            dists = np.linalg.norm(vec_all - new_vec[i], axis=1)
            for j_used in used:
                dists[j_used] = np.inf
            best = int(np.argmin(dists))
            kept.append(all_cands[best]); used.add(best)
    return kept, z_out


# ────────────────────────────────────────────────────────────────────
# Opposition-Based Learning
# ────────────────────────────────────────────────────────────────────

def _obl_population(pop: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Generate opposite population: x_obl = lb + ub - x, clipped."""
    return np.clip(lower + upper - pop, lower, upper)


# ────────────────────────────────────────────────────────────────────
# Gaussian Elite Local Search
# ────────────────────────────────────────────────────────────────────

def _elite_local_search(
    archive: list[Candidate],
    lower: np.ndarray,
    upper: np.ndarray,
    n_trials: int,
    sigma: float,
) -> np.ndarray:
    """Gaussian perturbation around top-k archive members."""
    if not archive or n_trials <= 0:
        return np.zeros((0, lower.size))
    obj = _candidate_matrix(archive)
    scores = np.sum(np.where(np.isfinite(obj), obj, 1e9), axis=1)
    top_k = max(1, min(5, len(archive)))
    top_idx = np.argsort(scores)[:top_k]
    span = upper - lower
    vectors: list[np.ndarray] = []
    per_elite = max(1, n_trials // top_k)
    for idx in top_idx:
        base = archive[int(idx)].vector.copy()
        noise = np.random.normal(0.0, sigma, size=(per_elite, base.size)) * span
        vectors.append(np.clip(base + noise, lower, upper))
    return np.vstack(vectors)


# ────────────────────────────────────────────────────────────────────
# L-SHADE: Linear Population Reduction
# ────────────────────────────────────────────────────────────────────

def _lshade_n(gen: int, max_gen: int, n_init: int, n_min: int) -> int:
    return max(n_min, int(round(n_init + (n_min - n_init) * gen / max_gen)))


# ────────────────────────────────────────────────────────────────────
# Main Runner
# ────────────────────────────────────────────────────────────────────

def run_multi_apex_shade(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    """APEX-SHADE multi-UAV runner.

    Dominant multi-objective constrained DE-based optimizer combining:
    L-SHADE + CDP + OBL + R2-archive + elite local search.
    """
    objective_count = 4
    model = dict(model)
    n_waypoints     = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value      = int(params.seed) if params.seed is not None else 0

    model, fleet_size = _ensure_multi_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["maxTurnDeg"]              = float(params.max_turn_deg)
    model["is_rl"]                   = False   # use hard-constraint evaluation
    model["hardCollisionConstraint"] = True

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dim = lower.size

    # Hyperparameters
    n_init   = max(8, int(params.population))
    n_min    = max(4, min(n_init // 4, 16))
    max_gen  = int(params.generations)
    arc_size = int(params.extra.get("nRep", max(n_init, 100)))
    metric_interval = int(params.extra.get("metricInterval", 20))
    ext_arc_cap = int(n_init * _ARC_RATIO)

    # NSGA-III reference points & R2 weights
    ref_points, _ = uniform_point(max(n_init, 60), objective_count)
    r2_weights    = uniform_weight_vectors(n_obj=objective_count, n_divisions=15)
    z_ideal_global = np.full(objective_count, np.inf)

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = (np.zeros((params.runs, 2), dtype=float)
                  if params.compute_metrics else np.zeros((0, 2), dtype=float))

    run_indices          = _resolve_run_indices(params)
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))

    for run_idx in run_indices:
        run_start = time.perf_counter()
        run_dir   = results_path / f"Run_{run_idx}"

        if resume_existing_runs:
            resumed = _resume_run_scores(
                run_dir=run_dir, problem_index=params.problem_index,
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resumed is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resumed
                continue

        # ── Seed RNG ─────────────────────────────────────────────────
        np.random.seed(seed_value * 1000 + run_idx)

        # ── OBL Initialisation ────────────────────────────────────────
        n   = n_init
        pop = np.random.uniform(lower, upper, size=(n, dim))
        obl = _obl_population(pop, lower, upper)
        init_pool = np.vstack([pop, obl])   # 2n candidates

        init_cands  = _evaluate_population(init_pool,  model, fleet_size, n_waypoints)
        init_obj    = _candidate_matrix(init_cands)
        init_cv     = _constraint_violation_vector(init_cands, model)

        # Keep best n by CDP ranking
        best_n = _cdp_sort_indices(init_obj, init_cv)[:n]
        pop        = init_pool[best_n]
        candidates = [init_cands[i] for i in best_n]
        obj        = init_obj[best_n]
        cv         = init_cv[best_n]

        # ── State ─────────────────────────────────────────────────────
        shade   = SHADEMemory(H=_H)
        ext_arc = np.zeros((0, dim))
        archive: list[Candidate] = []
        z_ideal = z_ideal_global.copy()

        # Seed Pareto archive with initial feasible solutions
        init_feasible = [c for c in candidates if float(c.details.get("feasible", 0.0)) > 0.5]
        if init_feasible:
            archive, z_ideal = _update_pareto_archive(
                [], init_feasible, arc_size, r2_weights, z_ideal, model
            )

        hv_hist = (np.zeros((max_gen, 2), dtype=float)
                   if params.compute_metrics else np.zeros((0, 2), dtype=float))

        # ── Generation Loop ───────────────────────────────────────────
        for gen in range(1, max_gen + 1):
            n_new = _lshade_n(gen, max_gen, n_init, n_min)

            # Sample adaptive F, CR
            F, CR = shade.sample(n)

            # Mutation + crossover
            mutants = _de_mutation_vectorised(pop, obj, cv, ext_arc, F, _P_BEST)
            mutants = np.clip(mutants, lower, upper)
            trials  = _binomial_crossover(pop, mutants, CR)

            # Elite local search injection (adaptive rate)
            n_elite = max(0, int(n * 0.06))
            if n_elite > 0 and archive:
                sigma_ls = max(0.02, 0.12 * (1.0 - gen / max_gen))
                ls_vecs  = _elite_local_search(archive, lower, upper, n_elite, sigma_ls)
                if ls_vecs.shape[0] > 0:
                    trials = np.vstack([trials, ls_vecs])

            # Evaluate
            trial_cands = _evaluate_population(trials, model, fleet_size, n_waypoints)
            trial_obj   = _candidate_matrix(trial_cands)
            trial_cv    = _constraint_violation_vector(trial_cands, model)

            # ── CDP Selection + SHADE record ──────────────────────────
            S_F: list[float] = []
            S_CR: list[float] = []
            S_delta: list[float] = []

            new_pop  = pop.copy()
            new_obj  = obj.copy()
            new_cv   = cv.copy()
            new_cands = list(candidates)

            for i in range(n):
                if _cdp_wins(float(trial_cv[i]), float(cv[i]), trial_obj[i], obj[i]):
                    # Save loser to external archive
                    if ext_arc.shape[0] < ext_arc_cap:
                        ext_arc = np.vstack([ext_arc, pop[i:i+1]])
                    else:
                        ext_arc[np.random.randint(ext_arc_cap)] = pop[i]
                    # Accept trial
                    new_pop[i]   = trials[i]
                    new_obj[i]   = trial_obj[i]
                    new_cv[i]    = float(trial_cv[i])
                    new_cands[i] = trial_cands[i]
                    # Record success
                    improvement = float(np.sum(np.maximum(0.0, obj[i] - trial_obj[i])))
                    S_F.append(float(F[i]))
                    S_CR.append(float(CR[i]))
                    S_delta.append(improvement + 1e-12)

            pop       = new_pop
            obj       = new_obj
            cv        = new_cv
            candidates = new_cands

            shade.update(S_F, S_CR, S_delta)

            # ── L-SHADE population reduction ──────────────────────────
            if n_new < n:
                keep = _cdp_sort_indices(obj, cv)[:n_new]
                pop        = pop[keep]
                obj        = obj[keep]
                cv         = cv[keep]
                candidates = [candidates[k] for k in keep]
                n = n_new

            # ── Archive update ────────────────────────────────────────
            archive_new = list(new_cands)
            # Also include elite-search offspring if any
            if trials.shape[0] > n:
                archive_new += trial_cands[n:]
            archive, z_ideal = _update_pareto_archive(
                archive, archive_new, arc_size, r2_weights, z_ideal, model
            )

            # ── Metrics ───────────────────────────────────────────────
            if params.compute_metrics and hv_hist.shape[0] > 0:
                if gen == 1 or gen == max_gen or gen % metric_interval == 0:
                    if archive:
                        aobj = _candidate_matrix(archive)
                        hv_hist[gen-1, 0] = cal_metric(1, aobj, params.problem_index, objective_count)
                        hv_hist[gen-1, 1] = cal_metric(2, aobj, params.problem_index, objective_count)
                elif gen > 1:
                    hv_hist[gen-1] = hv_hist[gen-2]

        # ── Finalize run ──────────────────────────────────────────────
        ensure_dir(run_dir)
        if params.compute_metrics and hv_hist.shape[0] > 0:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_hist})

        # Fallback: if archive is still empty (pathological), use least-violating pop
        if not archive:
            keep = _cdp_sort_indices(obj, cv)[:arc_size]
            archive = [candidates[k] for k in keep]

        _save_multi_artifacts(
            run_dir=run_dir,
            final_candidates=archive,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            rl_trace=None,
            run_metadata={
                "algorithmName": "APEX-SHADE",
                "representation": "cart",
                "requestedPopulation": float(params.population),
                "effectivePopulation": float(n_init),
                "archiveSize": float(arc_size),
            },
        )

        if params.compute_metrics:
            aobj = _candidate_matrix(archive)
            run_scores[run_idx - 1] = np.array([
                cal_metric(1, aobj, params.problem_index, objective_count),
                cal_metric(2, aobj, params.problem_index, objective_count),
            ], dtype=float)

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores
