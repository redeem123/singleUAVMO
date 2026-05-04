"""Self-contained NMOPSO engine for fleet path planning."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import uav_benchmark.algorithms.shared.pso_defaults as PSD
from uav_benchmark.algorithms.shared.nmopso_features import NMOPSOFeatureMixin
from uav_benchmark.algorithms.shared.nmopso_helpers import (
    _archive_front,
    _candidate_feasible_flags,
    _candidate_matrix,
    _de_current_to_pbest,
    _elite_refine_vectors,
    _finite_mean,
    _fixed_hv_reference,
    _gpu_velocity_update,
    _hypergrid_occupied_count,
    _leader_index,
    _normalize_feature_mode,
    _objective_score,
    _objective_spread_vectors,
    _r2_archive_from_candidates,
    _sample_hypergrid_leaders,
    _sbx_mutation,
)
from uav_benchmark.algorithms.shared.nmopso_helpers import (
    _finite_min as _finite_min,
)
from uav_benchmark.algorithms.shared.nmopso_helpers import (
    _hypergrid_cell_id as _hypergrid_cell_id,
)
from uav_benchmark.algorithms.shared.nmopso_helpers import (
    _stack_or_empty as _stack_or_empty,
)
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.r2_archive import uniform_weight_vectors

# ── Dataclasses ─────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Return value of :meth:`NMOPSOEngine.step`."""

    hv: float = 0.0
    diversity: float = 0.0
    feasible_ratio: float = 0.0
    conflict_rate: float = 0.0
    gpu_backend: str = "numpy:cpu"
    gpu_time_sec: float = 0.0


# ── Helpers (moved from fleet runner module) ───────────────────────


# ── Engine ──────────────────────────────────────────────────────────


class NMOPSOEngine(NMOPSOFeatureMixin):
    """Stateful NMOPSO engine for fleet path planning.

    Owns: population, velocity, pbest, archive, evaluation.
    Exposes ``step()`` + operator arms for adaptive search policies.
    """

    def __init__(
        self,
        model: dict[str, Any],
        pop_size: int,
        lower: np.ndarray,
        upper: np.ndarray,
        fleet_size: int,
        n_waypoints: int,
        representation: str,
        objective_count: int = 4,
        archive_size: int = 100,
        use_r2_archive: bool = True,
        paper_nmopso: bool = True,
        is_nmopso_family: bool = True,
        grid_cells: int = 10,
        grid_kappa: float = 1.0,
        use_grid_leader: bool = True,
        velocity_clamp_ratio: float = 0.5,
        gpu_mode: str = "off",
        feature_mode: str = "lite",
        evaluate_fn: Callable[[np.ndarray], list[Candidate]] | None = None,
        initial_population: np.ndarray | None = None,
        enable_fleet_repair: bool = True,
    ):
        self.model = model
        self.pop_size = pop_size
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.span = np.maximum(self.upper - self.lower, 1e-9)
        self.dimensions = int(self.lower.size)
        self.fleet_size = fleet_size
        self.n_waypoints = n_waypoints
        self.representation = representation
        self.objective_count = objective_count
        self.archive_size = archive_size
        self.use_r2_archive = use_r2_archive
        self.paper_nmopso = paper_nmopso
        self.is_nmopso_family = is_nmopso_family
        self.grid_cells = grid_cells
        self.grid_kappa = grid_kappa
        self.use_grid_leader = use_grid_leader
        self.velocity_limit_base = velocity_clamp_ratio * self.span
        self.gpu_mode = gpu_mode
        self.feature_mode = _normalize_feature_mode(feature_mode)
        self._evaluate_population = evaluate_fn
        self.initial_population = None if initial_population is None else np.asarray(initial_population, dtype=float)
        # When False, the built-in "Targeted Collision Repair (Fleet
        # Heuristic)" inside step() is skipped. Adaptive controllers (e.g.
        # SAC-SMOPSO) that own their own conflict-repair operator should set
        # this to False to avoid double-repairing the same particles.
        self.enable_fleet_repair = bool(enable_fleet_repair)

        # R2 archive setup
        self.r2_weights = (
            uniform_weight_vectors(n_obj=objective_count, n_divisions=15)
            if use_r2_archive
            else np.zeros((0, objective_count))
        )
        self.r2_z_ideal = np.full(objective_count, np.inf)

        # State (initialized in reset())
        self.population: np.ndarray = np.zeros((0, 0))
        self.velocity: np.ndarray = np.zeros((0, 0))
        self.pbest: np.ndarray = np.zeros((0, 0))
        self.pbest_obj: np.ndarray = np.zeros((0, 0))
        self.candidates: list[Candidate] = []
        self.archive: list[Candidate] = []
        self.current_obj: np.ndarray = np.zeros((0, 0))
        self.hv_ref_point: np.ndarray | None = None
        self.metric_rng: np.random.Generator = np.random.default_rng(0)
        self.generation: int = 0

        # Timing
        self.gpu_backend: str = "numpy:cpu"
        self.gpu_peak_bytes: float = 0.0
        self.gpu_update_time_sec: float = 0.0
        self.last_operator_evals: dict[str, int] = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}
        self.last_operator_proposed: dict[str, int] = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}
        self.last_operator_filtered: dict[str, int] = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}

    def reset(self) -> None:
        """Initialize/reset all PSO state for a new run."""
        if self.initial_population is not None and self.initial_population.shape == (self.pop_size, self.dimensions):
            self.population = np.clip(np.asarray(self.initial_population, dtype=float), self.lower, self.upper)
        else:
            self.population = np.random.uniform(self.lower, self.upper, size=(self.pop_size, self.dimensions))
        self.metric_rng = np.random.default_rng(0)
        self.velocity = np.zeros_like(self.population)
        self.candidates = self._evaluate(self.population)
        self._sync_population_from_candidates()
        self.pbest = self.population.copy()
        self.pbest_obj = _candidate_matrix(self.candidates).copy()
        self.current_obj = self.pbest_obj.copy()
        self.generation = 0
        self.hv_ref_point = None
        self.gpu_backend = "numpy:cpu"
        self.gpu_peak_bytes = 0.0
        self.gpu_update_time_sec = 0.0
        self.last_operator_evals = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}
        self.last_operator_proposed = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}
        self.last_operator_filtered = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}

        # Initial archive
        if self.use_r2_archive:
            self.r2_z_ideal = np.full(self.objective_count, np.inf)
            self.archive, self.r2_z_ideal = _r2_archive_from_candidates(
                [],
                self.candidates,
                max_size=self.archive_size,
                weights=self.r2_weights,
                z_ideal=self.r2_z_ideal,
            )
        else:
            self.archive = _archive_front(self.candidates, max_size=self.archive_size)

    def _evaluate(self, vectors: np.ndarray) -> list[Candidate]:
        """Evaluate a population matrix, returning Candidates."""
        if self._evaluate_population is None:
            raise RuntimeError("NMOPSOEngine requires an evaluation callback.")
        raw = self._evaluate_population(vectors)
        return [
            c if isinstance(c, Candidate) else Candidate(vector=c.vector, objective=c.objective, details=c.details)
            for c in raw
        ]

    def _sync_population_from_candidates(self) -> None:
        """Keep state vectors aligned with evaluators that project/repair decisions."""
        if len(self.candidates) != self.pop_size:
            return
        try:
            candidate_vectors = np.stack(
                [np.asarray(c.vector, dtype=float).reshape(-1) for c in self.candidates], axis=0
            )
        except (TypeError, ValueError):
            return
        if candidate_vectors.shape == self.population.shape:
            self.population = np.clip(candidate_vectors, self.lower, self.upper)

    def step(
        self,
        inertia: float,
        c1: float,
        c2: float,
        velocity_scale: float = 1.0,
        kappa_scale: float = 1.0,
        delta_scale: float = 1.0,
        region_scale: float = 1.0,
        leader_bias: float = 0.5,
        mutation_prob: float = 0.1,
        attention_weights: np.ndarray | None = None,
        repulsion_weight: float = 0.0,
    ) -> StepResult:
        """Execute one PSO generation: leader select → velocity → mutation → evaluate."""
        self.generation += 1
        result = StepResult()

        # --- Leader selection ---
        finite_archive_matrix = self._finite_archive_matrix()
        occupied_cells = 1

        if self.archive:
            leader_vectors = self.attention_leader_select(attention_weights)
            if leader_vectors is None:
                # Fallback to standard grid/random selection if no attention weights
                occupied_cells = (
                    max(1, _hypergrid_occupied_count(finite_archive_matrix, self.grid_cells))
                    if self.paper_nmopso
                    else 1
                )
                if self.paper_nmopso and finite_archive_matrix.size > 0:
                    finite_idx = np.where(np.all(np.isfinite(_candidate_matrix(self.archive)), axis=1))[0]
                    if finite_idx.size > 0:
                        picks, occupied_cells = _sample_hypergrid_leaders(
                            _candidate_matrix(self.archive)[finite_idx],
                            self.pop_size,
                            self.grid_cells,
                            self.grid_kappa * kappa_scale,
                        )
                        leader_vectors = np.stack([self.archive[finite_idx[p]].vector for p in picks], axis=0)
                if leader_vectors is None:
                    picks = [
                        _leader_index(self.archive, leader_bias, self.use_grid_leader, self.grid_cells)
                        for _ in range(self.pop_size)
                    ]
                    leader_vectors = np.stack([self.archive[p].vector for p in picks], axis=0)
        else:
            leader_vectors = self.pbest.copy()

        # --- Velocity & Position Update ---
        velocity_limit = self.velocity_limit_base * velocity_scale
        gpu_t0 = time.perf_counter()
        self.population, self.velocity, self.gpu_backend = _gpu_velocity_update(
            population=self.population,
            velocity=self.velocity,
            pbest=self.pbest,
            leaders=leader_vectors,
            inertia=inertia,
            c1=c1,
            c2=c2,
            lower=self.lower,
            upper=self.upper,
            velocity_limit=velocity_limit,
            gpu_mode=self.gpu_mode,
            fleet_size=self.fleet_size,
            n_waypoints=self.n_waypoints,
            repulsion_weight=repulsion_weight,
            safe_distance=float(self.model.get("separationMin", self.model.get("safeDist", 10.0))),
        )
        gpu_elapsed = float(time.perf_counter() - gpu_t0)
        self.gpu_update_time_sec += gpu_elapsed
        result.gpu_time_sec = gpu_elapsed
        result.gpu_backend = self.gpu_backend

        # --- Base Mutation ---
        if mutation_prob > 1e-9:
            delta_cells = (
                float(max(1, _hypergrid_occupied_count(finite_archive_matrix, self.grid_cells)))
                if self.paper_nmopso
                else 1.0
            )
            if self.paper_nmopso:
                gain = float(np.tanh((delta_cells * delta_scale * region_scale) / max(1.0, float(occupied_cells))))
                mutation_mask = np.random.rand(self.pop_size) < mutation_prob
                if np.any(mutation_mask):
                    noise = np.random.normal(0.0, 1.0, size=(int(np.sum(mutation_mask)), self.dimensions))
                    self.population[mutation_mask] = np.clip(
                        self.population[mutation_mask] + noise * gain * self.pbest[mutation_mask],
                        self.lower,
                        self.upper,
                    )
            else:
                mutation_mask = np.random.rand(self.pop_size) < mutation_prob
                if np.any(mutation_mask):
                    gen_progress = self.generation / max(1, self.generation + 100)  # placeholder; caller should set
                    sigma_scale = (
                        PSD.MUTATION_SIGMA_HIGH * (1.0 - gen_progress) + PSD.MUTATION_SIGMA_LOW
                        if self.is_nmopso_family
                        else PSD.MUTATION_SIGMA_GENERIC
                    )
                    mutation_sigma = sigma_scale * self.span
                    noise = (
                        np.random.normal(0.0, 1.0, size=(int(np.sum(mutation_mask)), self.dimensions)) * mutation_sigma
                    )
                    self.population[mutation_mask] = np.clip(
                        self.population[mutation_mask] + noise, self.lower, self.upper
                    )

        # --- Targeted Collision Repair (Fleet Heuristic) ---
        if self.enable_fleet_repair and self.fleet_size > 1 and hasattr(self, "candidates") and self.candidates:
            # Only apply proportional repair to highly constrained fleet spaces
            for i, cand in enumerate(self.candidates):
                if float(cand.details.get("conflictRate", 0.0)) > 0.0:
                    c_log = np.asarray(cand.details.get("conflictLog", []))
                    if c_log.size > 0 and c_log.ndim == 2:
                        bad_drones = np.unique(c_log[:, 1:3].astype(int))
                        for drone_idx in bad_drones:
                            if 0 <= drone_idx < self.fleet_size:
                                start_dim = (
                                    drone_idx
                                    * self.n_waypoints
                                    * (self.dimensions // (self.fleet_size * self.n_waypoints))
                                )
                                end_dim = (
                                    (drone_idx + 1)
                                    * self.n_waypoints
                                    * (self.dimensions // (self.fleet_size * self.n_waypoints))
                                )

                                global_mask = np.zeros(self.dimensions, dtype=bool)
                                global_mask[start_dim:end_dim] = True

                                # If Spherical, protect the radii `r` to preserve overall length efficiency
                                if self.paper_nmopso and (end_dim - start_dim) % 3 == 0:
                                    r_indices = np.arange(start_dim, end_dim, 3)
                                    global_mask[r_indices] = False

                                if np.any(global_mask):
                                    # Use a gentler aggressive sigma (2%) to break symmetry loops
                                    # without violently steering the drones 45 degrees into skyscrapers.
                                    agg_sigma = 0.02 * self.span[global_mask]
                                    noise = np.random.normal(0.0, 1.0, size=np.sum(global_mask)) * agg_sigma
                                    self.population[i, global_mask] = np.clip(
                                        self.population[i, global_mask] + noise,
                                        self.lower[global_mask],
                                        self.upper[global_mask],
                                    )

        # --- Evaluate ---
        self.candidates = self._evaluate(self.population)
        self._sync_population_from_candidates()
        self.current_obj = _candidate_matrix(self.candidates)

        # --- Update pbest ---
        pbest_matrix = np.asarray(self.pbest_obj, dtype=float)
        better = np.logical_and(
            np.all(self.current_obj <= pbest_matrix, axis=1),
            np.any(self.current_obj < pbest_matrix, axis=1),
        )
        ties = np.logical_and(
            np.all(self.current_obj == pbest_matrix, axis=1),
            np.random.rand(self.pop_size) < 0.5,
        )
        replace = np.logical_or(better, ties)
        if np.any(replace):
            self.pbest[replace] = self.population[replace]
            self.pbest_obj[replace] = self.current_obj[replace]

        # --- Archive update ---
        self.update_archive(self.candidates)

        # --- Compute metrics ---
        finite_archive = self._finite_archive_matrix()
        if self.hv_ref_point is None and finite_archive.size > 0:
            self.hv_ref_point = _fixed_hv_reference(finite_archive)
        if finite_archive.size > 0:
            result.hv = cal_metric(1, finite_archive, 0, self.objective_count, ref_point=self.hv_ref_point)
            result.diversity = float(np.mean(np.std(finite_archive, axis=0)))
        result.feasible_ratio = float(np.mean(_candidate_feasible_flags(self.candidates, self.current_obj)))
        result.conflict_rate = _finite_mean(
            [float(c.details.get("conflictRate", np.nan)) for c in self.candidates],
            default=0.0,
        )
        return result

    def update_archive(self, new_candidates: list[Candidate]) -> None:
        """Add candidates to the archive with Pareto/R2 pruning."""
        if self.use_r2_archive:
            self.archive, self.r2_z_ideal = _r2_archive_from_candidates(
                self.archive,
                new_candidates,
                max_size=self.archive_size,
                weights=self.r2_weights,
                z_ideal=self.r2_z_ideal,
            )
        else:
            self.archive = _archive_front(self.archive + new_candidates, max_size=self.archive_size)

    def _surrogate_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Build surrogate training tuples (decision vector -> objective score)."""
        train_x: list[np.ndarray] = []
        train_y: list[np.ndarray] = []

        if self.population.size > 0 and self.current_obj.size > 0:
            pop_y = _objective_score(np.asarray(self.current_obj, dtype=float))
            if pop_y.size == self.population.shape[0]:
                train_x.append(np.asarray(self.population, dtype=float))
                train_y.append(pop_y.reshape(-1))

        if self.archive:
            arch_x = np.stack([c.vector for c in self.archive], axis=0)
            arch_obj = _candidate_matrix(self.archive)
            arch_y = _objective_score(arch_obj)
            if arch_y.size == arch_x.shape[0]:
                train_x.append(np.asarray(arch_x, dtype=float))
                train_y.append(arch_y.reshape(-1))

        if not train_x or not train_y:
            return np.zeros((0, self.dimensions), dtype=float), np.zeros(0, dtype=float)

        x = np.vstack(train_x)
        y = np.concatenate(train_y)
        mask = np.logical_and(np.all(np.isfinite(x), axis=1), np.isfinite(y))
        if not np.any(mask):
            return np.zeros((0, self.dimensions), dtype=float), np.zeros(0, dtype=float)
        return x[mask], y[mask]

    def _surrogate_knn_predict(
        self, candidate_vectors: np.ndarray, train_x: np.ndarray, train_y: np.ndarray, k: int
    ) -> np.ndarray:
        """Predict objective score proxy via inverse-distance weighted KNN."""
        c = np.asarray(candidate_vectors, dtype=float)
        x = np.asarray(train_x, dtype=float)
        y = np.asarray(train_y, dtype=float).reshape(-1)
        if c.size == 0:
            return np.zeros(0, dtype=float)
        if x.size == 0 or y.size == 0:
            return np.full(c.shape[0], np.inf, dtype=float)

        k_eff = int(max(1, min(int(k), x.shape[0])))
        diff = c[:, None, :] - x[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
        nn_dist2 = np.take_along_axis(dist2, idx, axis=1)
        nn_y = y[idx]
        weights = 1.0 / np.sqrt(np.maximum(nn_dist2, 1e-12))
        numer = np.sum(weights * nn_y, axis=1)
        denom = np.sum(weights, axis=1)
        pred = numer / np.maximum(denom, 1e-12)
        return np.asarray(pred, dtype=float)

    def _prefilter_candidate_vectors(
        self,
        vectors: np.ndarray,
        max_evals: int | None,
        prefilter_enabled: bool,
        prefilter_ratio: float,
        prefilter_min_candidates: int,
        prefilter_k: int,
    ) -> tuple[np.ndarray, int, int]:
        """Surrogate prefilter to reduce expensive objective evaluations."""
        cand = np.asarray(vectors, dtype=float)
        if cand.ndim != 2 or cand.shape[0] == 0:
            return np.zeros((0, self.dimensions), dtype=float), 0, 0
        proposed = int(cand.shape[0])

        keep = proposed
        if max_evals is not None:
            keep = min(keep, max(0, int(max_evals)))
        if prefilter_enabled and prefilter_ratio < 1.0:
            ratio_keep = int(np.ceil(max(0.0, prefilter_ratio) * proposed))
            keep = min(keep, max(int(prefilter_min_candidates), ratio_keep))
        keep = int(np.clip(keep, 0, proposed))
        if keep <= 0:
            return np.zeros((0, self.dimensions), dtype=float), proposed, 0
        if keep >= proposed:
            return cand, proposed, proposed

        train_x, train_y = self._surrogate_training_data()
        if train_x.shape[0] >= max(4, int(prefilter_k)):
            pred = self._surrogate_knn_predict(cand, train_x, train_y, k=int(prefilter_k))
            order = np.argsort(pred)
        else:
            # Fallback heuristic when surrogate data is insufficient.
            center = np.mean(self.population, axis=0) if self.population.size > 0 else np.mean(cand, axis=0)
            order = np.argsort(np.linalg.norm(cand - center, axis=1))

        exploit_keep = int(max(1, round(0.8 * keep)))
        explore_keep = keep - exploit_keep
        chosen = list(order[:exploit_keep].tolist())
        if explore_keep > 0:
            tail = order[exploit_keep:]
            if tail.size > 0:
                picks = np.random.choice(tail, size=min(explore_keep, tail.size), replace=False)
                chosen.extend(np.asarray(picks, dtype=int).tolist())
        if not chosen:
            return np.zeros((0, self.dimensions), dtype=float), proposed, 0
        seen: set[int] = set()
        ordered_idx: list[int] = []
        for idx in chosen:
            i = int(idx)
            if i in seen:
                continue
            seen.add(i)
            ordered_idx.append(i)
            if len(ordered_idx) >= keep:
                break
        chosen_idx = np.asarray(ordered_idx, dtype=int)
        return cand[chosen_idx], proposed, int(chosen_idx.size)

    def inject_sbx(
        self,
        ratio: float,
        replace_ratio: float,
        mutation_scale: float = 1.0,
        max_evals: int | None = None,
        surrogate_prefilter_enabled: bool = False,
        surrogate_prefilter_ratio: float = 1.0,
        surrogate_prefilter_min_candidates: int = 1,
        surrogate_prefilter_k: int = 8,
    ) -> int:
        """Arm 1: SBX crossover injection. Returns number of replacements."""
        inject_count = int(max(2, round(self.pop_size * ratio)))
        if inject_count % 2 == 1:
            inject_count += 1
        if max_evals is not None:
            inject_count = min(inject_count, max(0, int(max_evals)))
            if inject_count % 2 == 1:
                inject_count -= 1
        if inject_count < 2:
            self.last_operator_evals["sbx"] = 0
            self.last_operator_proposed["sbx"] = 0
            self.last_operator_filtered["sbx"] = 0
            return 0
        parent_idx = np.random.randint(0, self.pop_size, size=inject_count)
        offspring_vectors = _sbx_mutation(self.population[parent_idx], self.lower, self.upper)
        if mutation_scale > 0.0 and mutation_scale != 1.0:
            base = self.population[np.random.randint(0, self.pop_size, size=inject_count)]
            offspring_vectors = np.clip(
                base + (offspring_vectors - base) * mutation_scale,
                self.lower,
                self.upper,
            )
        filtered_vectors, proposed_count, eval_count = self._prefilter_candidate_vectors(
            vectors=offspring_vectors,
            max_evals=max_evals,
            prefilter_enabled=surrogate_prefilter_enabled,
            prefilter_ratio=surrogate_prefilter_ratio,
            prefilter_min_candidates=surrogate_prefilter_min_candidates,
            prefilter_k=surrogate_prefilter_k,
        )
        self.last_operator_proposed["sbx"] = int(proposed_count)
        self.last_operator_evals["sbx"] = int(eval_count)
        self.last_operator_filtered["sbx"] = int(max(0, proposed_count - eval_count))
        if eval_count <= 0:
            return 0

        off_candidates = self._evaluate(filtered_vectors)
        off_obj = _candidate_matrix(off_candidates)
        self.update_archive(off_candidates)

        replaced = 0
        if replace_ratio > 0.0 and off_obj.size > 0 and self.current_obj.size > 0:
            replace_count = int(
                min(
                    max(1, round(self.pop_size * replace_ratio)),
                    self.pop_size,
                    off_obj.shape[0],
                )
            )
            cur_score = _objective_score(self.current_obj)
            off_score = _objective_score(off_obj)
            worst_idx = np.argsort(cur_score)[-replace_count:]
            best_off = np.argsort(off_score)[:replace_count]
            self.population[worst_idx] = filtered_vectors[best_off]
            self.velocity[worst_idx] = 0.0
            for li, oi in zip(worst_idx.tolist(), best_off.tolist(), strict=True):
                self.candidates[int(li)] = off_candidates[int(oi)]
            self.current_obj[worst_idx] = off_obj[best_off]
            replaced = int(replace_count)
        return replaced

    def inject_de(
        self,
        f_scale: float,
        cr_rate: float,
        ratio: float,
        replace_ratio: float,
        pbest_ratio: float,
        max_evals: int | None = None,
        surrogate_prefilter_enabled: bool = False,
        surrogate_prefilter_ratio: float = 1.0,
        surrogate_prefilter_min_candidates: int = 1,
        surrogate_prefilter_k: int = 8,
    ) -> int:
        """Arm 2: DE/current-to-pbest injection. Returns number of replacements."""
        if self.current_obj.size == 0:
            self.last_operator_evals["de"] = 0
            self.last_operator_proposed["de"] = 0
            self.last_operator_filtered["de"] = 0
            return 0
        de_count = int(min(max(4, round(self.pop_size * ratio)), self.pop_size))
        if max_evals is not None:
            de_count = min(de_count, max(0, int(max_evals)))
        if de_count < 4:
            self.last_operator_evals["de"] = 0
            self.last_operator_proposed["de"] = 0
            self.last_operator_filtered["de"] = 0
            return 0
        de_vectors = _de_current_to_pbest(
            population=self.population,
            objective=self.current_obj,
            lower=self.lower,
            upper=self.upper,
            de_count=de_count,
            f_scale=f_scale,
            cr_rate=cr_rate,
            pbest_ratio=pbest_ratio,
        )
        if de_vectors.size == 0:
            self.last_operator_evals["de"] = 0
            self.last_operator_proposed["de"] = 0
            self.last_operator_filtered["de"] = 0
            return 0
        filtered_vectors, proposed_count, eval_count = self._prefilter_candidate_vectors(
            vectors=de_vectors,
            max_evals=max_evals,
            prefilter_enabled=surrogate_prefilter_enabled,
            prefilter_ratio=surrogate_prefilter_ratio,
            prefilter_min_candidates=surrogate_prefilter_min_candidates,
            prefilter_k=surrogate_prefilter_k,
        )
        self.last_operator_proposed["de"] = int(proposed_count)
        self.last_operator_evals["de"] = int(eval_count)
        self.last_operator_filtered["de"] = int(max(0, proposed_count - eval_count))
        if eval_count <= 0:
            return 0

        de_candidates = self._evaluate(filtered_vectors)
        de_obj = _candidate_matrix(de_candidates)
        self.update_archive(de_candidates)

        replaced = 0
        if replace_ratio > 0.0 and de_obj.size > 0 and self.current_obj.size > 0:
            de_replace = int(min(max(1, round(self.pop_size * replace_ratio)), self.pop_size, de_obj.shape[0]))
            cur_score = _objective_score(self.current_obj)
            de_score = _objective_score(de_obj)
            worst_idx = np.argsort(cur_score)[-de_replace:]
            best_de = np.argsort(de_score)[:de_replace]
            self.population[worst_idx] = filtered_vectors[best_de]
            self.velocity[worst_idx] = 0.0
            for li, di in zip(worst_idx.tolist(), best_de.tolist(), strict=True):
                self.candidates[int(li)] = de_candidates[int(di)]
            self.current_obj[worst_idx] = de_obj[best_de]
            replaced = int(de_replace)
        return replaced

    def elite_refine(
        self,
        sigma: float,
        top_k: int,
        iters: int,
        max_evals: int | None = None,
        surrogate_prefilter_enabled: bool = False,
        surrogate_prefilter_ratio: float = 1.0,
        surrogate_prefilter_min_candidates: int = 1,
        surrogate_prefilter_k: int = 8,
    ) -> int:
        """Arm 3a: Elite refinement. Returns number of refined candidates."""
        if not self.archive or top_k <= 0 or iters <= 0:
            self.last_operator_evals["elite"] = 0
            self.last_operator_proposed["elite"] = 0
            self.last_operator_filtered["elite"] = 0
            return 0
        trial_vectors = _elite_refine_vectors(
            archive=self.archive,
            lower=self.lower,
            upper=self.upper,
            span=self.span,
            sigma=sigma,
            top_k=top_k,
            iters=iters,
            max_trials=max_evals,
        )
        filtered_vectors, proposed_count, eval_count = self._prefilter_candidate_vectors(
            vectors=trial_vectors,
            max_evals=max_evals,
            prefilter_enabled=surrogate_prefilter_enabled,
            prefilter_ratio=surrogate_prefilter_ratio,
            prefilter_min_candidates=surrogate_prefilter_min_candidates,
            prefilter_k=surrogate_prefilter_k,
        )
        self.last_operator_proposed["elite"] = int(proposed_count)
        self.last_operator_evals["elite"] = int(eval_count)
        self.last_operator_filtered["elite"] = int(max(0, proposed_count - eval_count))
        if eval_count <= 0:
            return 0

        refined = self._evaluate(filtered_vectors)
        if refined:
            self.update_archive(refined)
        return int(len(refined))

    def objective_spread(
        self,
        top_k_per_obj: int,
        trials_per_pick: int,
        sigma: float,
        replace_ratio: float = 0.0,
        max_evals: int | None = None,
        surrogate_prefilter_enabled: bool = False,
        surrogate_prefilter_ratio: float = 1.0,
        surrogate_prefilter_min_candidates: int = 1,
        surrogate_prefilter_k: int = 8,
    ) -> int:
        """Arm 3b: Objective spread. Returns number injected into population."""
        if not self.archive or top_k_per_obj <= 0 or trials_per_pick <= 0:
            self.last_operator_evals["spread"] = 0
            self.last_operator_proposed["spread"] = 0
            self.last_operator_filtered["spread"] = 0
            return 0
        spread_vectors = _objective_spread_vectors(
            archive=self.archive,
            lower=self.lower,
            upper=self.upper,
            span=self.span,
            top_k_per_obj=top_k_per_obj,
            trials_per_pick=trials_per_pick,
            sigma=sigma,
            max_trials=max_evals,
        )
        filtered_vectors, proposed_count, eval_count = self._prefilter_candidate_vectors(
            vectors=spread_vectors,
            max_evals=max_evals,
            prefilter_enabled=surrogate_prefilter_enabled,
            prefilter_ratio=surrogate_prefilter_ratio,
            prefilter_min_candidates=surrogate_prefilter_min_candidates,
            prefilter_k=surrogate_prefilter_k,
        )
        self.last_operator_proposed["spread"] = int(proposed_count)
        self.last_operator_evals["spread"] = int(eval_count)
        self.last_operator_filtered["spread"] = int(max(0, proposed_count - eval_count))
        if eval_count <= 0:
            self.last_operator_evals["spread"] = 0
            return 0
        spread = self._evaluate(filtered_vectors)
        if not spread:
            self.last_operator_evals["spread"] = 0
            return 0
        self.update_archive(spread)
        injected = 0
        if replace_ratio > 0.0 and self.current_obj.size > 0:
            spread_obj = _candidate_matrix(spread)
            evaluated_vectors = np.stack([c.vector for c in spread], axis=0)
            replace_count = int(min(max(1, round(self.pop_size * replace_ratio)), self.pop_size, spread_obj.shape[0]))
            cur_score = _objective_score(self.current_obj)
            sp_score = _objective_score(spread_obj)
            worst_idx = np.argsort(cur_score)[-replace_count:]
            best_idx = np.argsort(sp_score)[:replace_count]
            self.population[worst_idx] = evaluated_vectors[best_idx]
            self.velocity[worst_idx] = 0.0
            for li, si in zip(worst_idx.tolist(), best_idx.tolist(), strict=True):
                self.candidates[int(li)] = spread[int(si)]
            self.current_obj[worst_idx] = spread_obj[best_idx]
            injected = int(replace_count)
        return injected
