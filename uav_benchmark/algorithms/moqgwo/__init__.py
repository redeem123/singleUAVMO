"""MOQGWO family runner with attention fusion + Atlas archive."""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _should_write_final_hv,
    _ensure_fleet_endpoints,
)
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.mission_encoding import decision_to_paths
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import n_d_sort
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.algorithms.nmopso import (
    build_atlas_config,
    topology_signature,
    topology_bin_from_signature,
    robustness_from_cost,
    AtlasConfig,
)
from uav_benchmark.algorithms.moqgwo.gpu_strict_ops import (
    QGWOGPUStrictEngine,
    evaluate_population_gpu_strict,
    gpu_peak_bytes_for_device,
    require_torch_gpu_for_moqgwo,
)


_ATTN_TAU_OBJ = 0.20
_ATTN_TAU_ATLAS = 0.35
_ATTN_SAFE_GAIN = 0.55
_ATTN_ATLAS_GAIN = 0.30
_ATTN_BLEND_EPS = 0.03
_ATTN_ROW_DEGENERATE_EPS = 1e-6
_ATTN_EPS = 1e-12
_ATLAS_BIN_MULT = 1000
_RTCS_SAFE_RATIO = 0.40
_RTCS_BALANCED_RATIO = 0.40
_RTCS_CORRECT_RATIO = 0.15
_RTCS_ALTITUDE_GAIN = 0.12


def _fit_matrix(values: np.ndarray, rows: int, cols: int, fill: float) -> np.ndarray:
    out = np.full((rows, cols), float(fill), dtype=float)
    raw = np.asarray(values, dtype=float)
    if raw.size == 0:
        return out
    raw = raw.reshape(-1, cols) if raw.ndim != 2 else raw
    if raw.shape[1] != cols:
        return out
    use = min(rows, raw.shape[0])
    out[:use] = raw[:use]
    return out


def _fit_vector(values: np.ndarray, rows: int, fill: float) -> np.ndarray:
    out = np.full(rows, float(fill), dtype=float)
    raw = np.asarray(values, dtype=float).reshape(-1)
    if raw.size == 0:
        return out
    use = min(rows, raw.size)
    out[:use] = raw[:use]
    return out


def _fit_int_vector(values: np.ndarray, rows: int, fill: int) -> np.ndarray:
    out = np.full(rows, int(fill), dtype=int)
    raw = np.asarray(values, dtype=int).reshape(-1)
    if raw.size == 0:
        return out
    use = min(rows, raw.size)
    out[:use] = raw[:use]
    return out


def _rtcs_obstacle_discs(model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    threats = model.get("threats")
    if threats is not None:
        threat_arr = np.asarray(threats, dtype=float)
        if threat_arr.ndim == 2 and threat_arr.shape[1] >= 4 and threat_arr.shape[0] > 0:
            centers.append(threat_arr[:, :2])
            radii.append(np.maximum(0.0, threat_arr[:, 3]))
    nofly_c = model.get("nofly_c")
    nofly_r = model.get("nofly_r")
    if nofly_c is not None and nofly_r is not None:
        c = np.asarray(nofly_c, dtype=float)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if c.ndim == 2 and c.shape[1] >= 2 and c.shape[0] > 0:
            c = c[:, :2]
            r = np.asarray(nofly_r, dtype=float).reshape(-1)
            if r.size == 1:
                r = np.repeat(r, c.shape[0])
            elif r.size < c.shape[0]:
                r = np.pad(r, (0, c.shape[0] - r.size), mode="edge")
            centers.append(c)
            radii.append(np.maximum(0.0, r[: c.shape[0]]))
    if not centers:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    return np.vstack(centers), np.concatenate(radii)


def _rtcs_point_clearance(
    points: np.ndarray,
    *,
    model: dict[str, Any],
    obs_centers: np.ndarray,
    obs_radii: np.ndarray,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    z_abs = points[:, 2]
    xmax = int(float(model.get("xmax", 1)))
    ymax = int(float(model.get("ymax", 1)))
    x_idx = np.clip(np.rint(x).astype(int), 1, max(1, xmax)) - 1
    y_idx = np.clip(np.rint(y).astype(int), 1, max(1, ymax)) - 1
    ground = np.asarray(model["H"], dtype=float)[y_idx, x_idx]
    z_rel = z_abs - ground
    clearance = z_rel
    if obs_centers.size > 0:
        dx = x[:, None] - obs_centers[None, :, 0]
        dy = y[:, None] - obs_centers[None, :, 1]
        dist = np.sqrt(np.maximum(0.0, dx * dx + dy * dy)) - obs_radii[None, :]
        min_obs = np.min(dist, axis=1)
        clearance = np.minimum(clearance, min_obs)
    return clearance


def _rtcs_path_shape_features(path: np.ndarray) -> tuple[float, float]:
    if path.ndim != 2 or path.shape[0] < 2:
        return 1.0, 1.0
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))
    straight = float(np.linalg.norm(path[-1] - path[0]))
    if straight <= _ATTN_EPS or total <= _ATTN_EPS:
        len_proxy = 1.0
    else:
        len_proxy = float(np.clip((total / straight - 1.0) / 2.0, 0.0, 1.0))
    if path.shape[0] < 3:
        return len_proxy, 0.0
    v1 = path[1:-1] - path[:-2]
    v2 = path[2:] - path[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > _ATTN_EPS) & (n2 > _ATTN_EPS)
    if not np.any(valid):
        return len_proxy, 0.0
    cross = np.linalg.norm(np.cross(v1[valid], v2[valid]), axis=1)
    dots = np.sum(v1[valid] * v2[valid], axis=1)
    angles = np.abs(np.arctan2(cross, dots))
    turn_proxy = float(np.clip(np.mean(angles) / np.pi, 0.0, 1.0))
    return len_proxy, turn_proxy


def _rtcs_surrogate_features(
    population: np.ndarray,
    *,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    separation_min: float,
    atlas_config: AtlasConfig,
    use_topology: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pop = population.shape[0]
    quality = np.zeros(n_pop, dtype=float)
    top_bins = np.zeros(n_pop, dtype=int)
    severity = np.ones(n_pop, dtype=float)
    obs_centers, obs_radii = _rtcs_obstacle_discs(model)
    drone_size = float(model.get("droneSize", 1.0))
    safe_dist = float(model.get("safeDist", 10.0))

    for idx in range(n_pop):
        vector = np.asarray(population[idx], dtype=float).reshape(-1)
        try:
            paths = decision_to_paths(vector, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        except Exception:
            quality[idx] = 0.0
            top_bins[idx] = 0
            severity[idx] = 3.0
            continue

        min_clear = float("inf")
        min_sep = float("inf")
        len_vals: list[float] = []
        turn_vals: list[float] = []
        sigs: list[np.ndarray] = []
        for path in paths:
            p = np.asarray(path, dtype=float)
            sigs.append(topology_signature(p, model, atlas_config.max_obstacles))
            len_proxy, turn_proxy = _rtcs_path_shape_features(p)
            len_vals.append(len_proxy)
            turn_vals.append(turn_proxy)
            point_clear = _rtcs_point_clearance(
                p,
                model=model,
                obs_centers=obs_centers,
                obs_radii=obs_radii,
            )
            if p.shape[0] > 1:
                mids = 0.5 * (p[:-1] + p[1:])
                mid_clear = _rtcs_point_clearance(
                    mids,
                    model=model,
                    obs_centers=obs_centers,
                    obs_radii=obs_radii,
                )
                if mid_clear.size > 0:
                    point_clear = np.concatenate([point_clear, mid_clear])
            if point_clear.size > 0:
                min_clear = min(min_clear, float(np.min(point_clear)))

        if len(paths) > 1:
            ref = np.stack([np.asarray(path, dtype=float) for path in paths], axis=0)
            for i in range(ref.shape[0]):
                for j in range(i + 1, ref.shape[0]):
                    dist = np.linalg.norm(ref[i] - ref[j], axis=1)
                    if dist.size > 0:
                        min_sep = min(min_sep, float(np.min(dist)))

        if not np.isfinite(min_clear):
            min_clear = -safe_dist
        if not np.isfinite(min_sep):
            min_sep = 0.0
        clearance_def = max(0.0, drone_size - min_clear) / max(drone_size, _ATTN_EPS)
        safe_def = max(0.0, safe_dist - min_clear) / max(safe_dist, _ATTN_EPS)
        sep_def = max(0.0, separation_min - min_sep) / max(separation_min, _ATTN_EPS)
        clr_term = 1.0 / (1.0 + clearance_def + 0.5 * safe_def)
        sep_term = 1.0 / (1.0 + 2.0 * sep_def)
        len_proxy = float(np.mean(len_vals)) if len_vals else 1.0
        turn_proxy = float(np.mean(turn_vals)) if turn_vals else 1.0
        q = 0.45 * clr_term + 0.25 * sep_term + 0.20 * (1.0 - len_proxy) + 0.10 * (1.0 - turn_proxy)
        quality[idx] = float(np.clip(q, 0.0, 1.0))
        severity[idx] = float(np.clip(clearance_def + sep_def + 0.5 * len_proxy + 0.5 * turn_proxy, 0.0, 4.0))

        if use_topology and atlas_config.enabled and sigs:
            avg_sig = np.mean(np.stack(sigs, axis=0), axis=0)
            top_bins[idx] = int(topology_bin_from_signature(avg_sig, atlas_config))
        else:
            top_bins[idx] = 0
    return quality, top_bins, severity


def _rtcs_select_indices(
    quality: np.ndarray,
    top_bins: np.ndarray,
    *,
    use_topology: bool,
) -> np.ndarray:
    n = int(quality.size)
    if n <= 0:
        return np.zeros(0, dtype=int)
    n_safe = int(round(_RTCS_SAFE_RATIO * n))
    n_bal = int(round(_RTCS_BALANCED_RATIO * n))
    n_safe = int(np.clip(n_safe, 1, n))
    n_bal = int(np.clip(n_bal, 0, n - n_safe))
    n_exp = n - n_safe - n_bal

    order_desc = np.argsort(-quality, kind="stable")
    chosen = np.zeros(n, dtype=bool)
    safe_idx = order_desc[:n_safe]
    chosen[safe_idx] = True

    balanced: list[int] = []
    remaining = np.where(~chosen)[0]
    if n_bal > 0:
        if use_topology and remaining.size > 0:
            per_bin: dict[int, list[int]] = {}
            rem_order = remaining[np.argsort(-quality[remaining], kind="stable")]
            for idx in rem_order:
                key = int(top_bins[idx])
                per_bin.setdefault(key, []).append(int(idx))
            while len(balanced) < n_bal:
                progressed = False
                for key in sorted(per_bin):
                    bucket = per_bin[key]
                    if not bucket:
                        continue
                    pick = bucket.pop(0)
                    if not chosen[pick]:
                        balanced.append(pick)
                        chosen[pick] = True
                        progressed = True
                        if len(balanced) >= n_bal:
                            break
                if not progressed:
                    break
        if len(balanced) < n_bal:
            fill_candidates = np.where(~chosen)[0]
            fill = fill_candidates[np.argsort(-quality[fill_candidates], kind="stable")][: n_bal - len(balanced)]
            for idx in fill:
                balanced.append(int(idx))
                chosen[int(idx)] = True

    exploratory: list[int] = []
    if n_exp > 0:
        rem = np.where(~chosen)[0]
        if rem.size > 0:
            picks = rem[np.random.permutation(rem.size)[:n_exp]]
            for idx in picks:
                exploratory.append(int(idx))
                chosen[int(idx)] = True
    if np.sum(chosen) < n:
        tail = np.where(~chosen)[0]
        for idx in tail:
            exploratory.append(int(idx))
            chosen[int(idx)] = True
    merged = np.concatenate([
        np.asarray(safe_idx, dtype=int),
        np.asarray(balanced, dtype=int),
        np.asarray(exploratory, dtype=int),
    ])
    if merged.size != n:
        merged = np.arange(n, dtype=int)
    return merged


def _rtcs_apply_correction(
    population: np.ndarray,
    severity: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    fleet_size: int,
    n_waypoints: int,
    correction_weight: np.ndarray | None = None,
) -> np.ndarray:
    if population.size == 0:
        return population
    out = np.asarray(population, dtype=float).copy()
    n = out.shape[0]
    weight = np.ones(n, dtype=float)
    if correction_weight is not None:
        raw_weight = np.asarray(correction_weight, dtype=float).reshape(-1)
        use = min(n, raw_weight.size)
        weight[:use] = raw_weight[:use]
    weight = np.clip(weight, 0.10, 1.00)
    k = int(np.clip(np.ceil(_RTCS_CORRECT_RATIO * n), 1, n))
    priority = np.asarray(severity, dtype=float).reshape(-1)[:n] * weight
    pick = np.argsort(-priority, kind="stable")[:k]
    low_block = np.asarray(lower, dtype=float).reshape(fleet_size, n_waypoints, 3)
    up_block = np.asarray(upper, dtype=float).reshape(fleet_size, n_waypoints, 3)
    z_span = np.maximum(0.0, up_block[:, :, 2] - low_block[:, :, 2])
    for idx in pick:
        block = out[int(idx)].reshape(fleet_size, n_waypoints, 3).copy()
        gain = _RTCS_ALTITUDE_GAIN * float(np.clip(severity[int(idx)] / 2.0, 0.0, 1.0)) * float(weight[int(idx)])
        block[:, :, 2] = block[:, :, 2] + gain * z_span
        block = np.clip(block, low_block, up_block)
        out[int(idx)] = block.reshape(-1)
    return np.clip(out, lower, upper)


def _rtcs_initialize_population(
    population: np.ndarray,
    *,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    lower: np.ndarray,
    upper: np.ndarray,
    separation_min: float,
    atlas_config: AtlasConfig,
    use_topology: bool,
) -> np.ndarray:
    pop = np.asarray(population, dtype=float)
    if pop.ndim != 2 or pop.shape[0] <= 0:
        return pop
    quality, top_bins, severity = _rtcs_surrogate_features(
        pop,
        model=model,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
        separation_min=separation_min,
        atlas_config=atlas_config,
        use_topology=bool(use_topology),
    )
    order = _rtcs_select_indices(quality, top_bins, use_topology=bool(use_topology))
    n = pop.shape[0]
    n_safe = int(round(_RTCS_SAFE_RATIO * n))
    n_bal = int(round(_RTCS_BALANCED_RATIO * n))
    n_safe = int(np.clip(n_safe, 1, n))
    n_bal = int(np.clip(n_bal, 0, n - n_safe))
    correction_weight = np.ones(n, dtype=float)
    if order.size == n:
        safe_idx = order[:n_safe]
        bal_idx = order[n_safe : n_safe + n_bal]
        correction_weight[safe_idx] = 0.35
        correction_weight[bal_idx] = 0.70
    seeded = _rtcs_apply_correction(
        pop,
        severity,
        lower=lower,
        upper=upper,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
        correction_weight=correction_weight,
    )
    return np.clip(seeded, lower, upper)


# ─────────────────────────────────────────────────────────────────────
# GWO Engine
# ─────────────────────────────────────────────────────────────────────

class QGWO_Engine:
    """Grey Wolf Optimizer core with objective/atlas conditioned attention."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        pop_size: int,
        use_attention: bool = True,
    ) -> None:
        self.lower    = lower
        self.upper    = upper
        self.dim      = lower.size
        self.pop_size = pop_size
        self.use_attention = bool(use_attention)
        self.positions = np.random.uniform(lower, upper, size=(pop_size, self.dim))
        self.leaders   = np.zeros((3, self.dim))
        self._wolf_objectives = np.zeros((self.pop_size, 4), dtype=float)
        self._wolf_risk = np.zeros(self.pop_size, dtype=float)
        self._wolf_topology = np.zeros(self.pop_size, dtype=int)
        self._wolf_robust = np.ones(self.pop_size, dtype=int)
        self._leader_objectives = np.zeros((3, 4), dtype=float)
        self._leader_risk = np.zeros(3, dtype=float)
        self._leader_topology = np.zeros(3, dtype=int)
        self._leader_robust = np.ones(3, dtype=int)
        self._feasibility_pressure = 0.0
        self._atlas_enabled = False
        self._atlas_robust_bins = 2
        self.last_attention_stats: dict[str, float] = {
            "entropy_mean": 0.0,
            "lambda_safe": 0.0,
            "lambda_atlas": 0.0,
        }

    def set_attention_context(
        self,
        *,
        wolf_objectives: np.ndarray,
        wolf_risk: np.ndarray,
        feasibility_pressure: float,
        leader_objectives: np.ndarray,
        leader_risk: np.ndarray,
        wolf_topology: np.ndarray | None = None,
        wolf_robust: np.ndarray | None = None,
        leader_topology: np.ndarray | None = None,
        leader_robust: np.ndarray | None = None,
        atlas_enabled: bool = False,
        atlas_robust_bins: int = 2,
    ) -> None:
        self._wolf_objectives = np.clip(_fit_matrix(wolf_objectives, self.pop_size, 4, fill=1.0), 0.0, 1.0)
        self._wolf_risk = np.maximum(0.0, _fit_vector(wolf_risk, self.pop_size, fill=1.0))
        self._leader_objectives = np.clip(_fit_matrix(leader_objectives, 3, 4, fill=1.0), 0.0, 1.0)
        self._leader_risk = np.maximum(0.0, _fit_vector(leader_risk, 3, fill=1.0))
        self._feasibility_pressure = float(np.clip(feasibility_pressure, 0.0, 1.0))
        self._atlas_enabled = bool(atlas_enabled)
        self._atlas_robust_bins = max(2, int(atlas_robust_bins))
        self._wolf_topology = _fit_int_vector(
            np.asarray(wolf_topology if wolf_topology is not None else np.zeros(0, dtype=int), dtype=int),
            self.pop_size,
            fill=0,
        )
        self._wolf_robust = np.maximum(
            1,
            _fit_int_vector(
                np.asarray(wolf_robust if wolf_robust is not None else np.ones(0, dtype=int), dtype=int),
                self.pop_size,
                fill=1,
            ),
        )
        self._leader_topology = _fit_int_vector(
            np.asarray(leader_topology if leader_topology is not None else np.zeros(0, dtype=int), dtype=int),
            3,
            fill=0,
        )
        self._leader_robust = np.maximum(
            1,
            _fit_int_vector(
                np.asarray(leader_robust if leader_robust is not None else np.ones(0, dtype=int), dtype=int),
                3,
                fill=1,
            ),
        )

    @staticmethod
    def _normalize_channel_rows(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2:
            return np.zeros((0, 0), dtype=float)
        mean = np.mean(scores, axis=1, keepdims=True)
        std = np.std(scores, axis=1, keepdims=True)
        centered = scores - mean
        good = std > _ATTN_EPS
        out = np.zeros_like(scores)
        if np.any(good):
            out = np.divide(centered, np.where(good, std, 1.0))
        return out

    def _attention_weights(self) -> np.ndarray:
        p = float(np.clip(self._feasibility_pressure, 0.0, 1.0))
        w2 = 0.25 + 0.30 * p
        w_other = (1.0 - w2) / 3.0
        objective_weights = np.asarray([w_other, w2, w_other, w_other], dtype=float).reshape(1, 1, 4)

        diff = np.abs(self._wolf_objectives[:, None, :] - self._leader_objectives[None, :, :])
        score_obj = -np.sum(objective_weights * diff, axis=2) / _ATTN_TAU_OBJ
        score_obj = self._normalize_channel_rows(score_obj)

        leader_risk = np.maximum(0.0, np.asarray(self._leader_risk, dtype=float).reshape(3))
        risk_min = float(np.min(leader_risk))
        risk_max = float(np.max(leader_risk))
        risk_span = risk_max - risk_min
        if risk_span > _ATTN_EPS:
            safe_pref = 1.0 - ((leader_risk - risk_min) / risk_span)
            rank = np.empty(3, dtype=float)
            rank[np.argsort(leader_risk, kind="stable")] = np.arange(3, dtype=float)
            rank_pref = (2.0 - rank) / 2.0  # safest leader gets 1.0
            safe_pref = 0.75 * safe_pref + 0.25 * rank_pref
        else:
            j2 = np.clip(self._leader_objectives[:, 1], 0.0, 1.0)
            j2_min = float(np.min(j2))
            j2_span = float(np.max(j2) - j2_min)
            if j2_span > _ATTN_EPS:
                safe_pref = 1.0 - ((j2 - j2_min) / j2_span)
            else:
                safe_pref = np.zeros(3, dtype=float)
        safe_pref = safe_pref - float(np.mean(safe_pref))
        wolf_risk = np.maximum(0.0, np.asarray(self._wolf_risk, dtype=float).reshape(-1))
        wolf_scale = wolf_risk / (wolf_risk + 1.0)
        safe_scale = 0.25 + 0.75 * (0.60 * p + 0.40 * wolf_scale)
        score_safe = safe_scale[:, None] * safe_pref.reshape(1, 3)
        score_safe = self._normalize_channel_rows(score_safe)

        risk_span_scale = risk_span / (risk_span + 1.0)
        lambda_safe = 0.28 + 0.55 * p * (0.5 + 0.5 * risk_span_scale)
        lambda_safe = float(np.clip(lambda_safe, 0.28, 0.82))
        if self._atlas_enabled:
            lambda_atlas = _ATTN_ATLAS_GAIN * (1.0 - p) + 0.08 * risk_span_scale
            lambda_atlas = float(np.clip(lambda_atlas, 0.0, 0.33))
        else:
            lambda_atlas = 0.0
        lambda_obj = max(0.0, 1.0 - lambda_safe - lambda_atlas)
        norm = max(_ATTN_EPS, lambda_obj + lambda_safe + lambda_atlas)
        lambda_obj /= norm
        lambda_safe /= norm
        lambda_atlas /= norm

        score = lambda_obj * score_obj + lambda_safe * score_safe
        if self._atlas_enabled:
            denom = max(1.0, float(self._atlas_robust_bins - 1))
            top_match = (self._wolf_topology[:, None] == self._leader_topology[None, :]).astype(float)
            rob_gap = np.abs(self._wolf_robust[:, None].astype(float) - self._leader_robust[None, :].astype(float)) / denom
            leader_rob = np.asarray(self._leader_robust, dtype=float).reshape(3)
            rob_min = float(np.min(leader_rob))
            rob_span = float(np.max(leader_rob) - rob_min)
            if rob_span > _ATTN_EPS:
                rob_quality = (leader_rob - rob_min) / rob_span
            else:
                rob_quality = np.zeros(3, dtype=float)
            score_atlas = (1.2 * top_match - 0.8 * rob_gap + 0.6 * rob_quality.reshape(1, 3)) / _ATTN_TAU_ATLAS
            score_atlas = self._normalize_channel_rows(score_atlas)
            score = score + lambda_atlas * score_atlas

        tau_eff = 0.88 - 0.33 * p
        tau_eff = float(np.clip(tau_eff, 0.50, 0.88))
        score = score / tau_eff
        score = score - np.max(score, axis=1, keepdims=True)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            weights = np.exp(score)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights = np.divide(weights, np.where(weights_sum > _ATTN_EPS, weights_sum, 1.0))
        weights = (1.0 - _ATTN_BLEND_EPS) * weights + (_ATTN_BLEND_EPS / 3.0)
        weights = np.divide(weights, np.maximum(np.sum(weights, axis=1, keepdims=True), _ATTN_EPS))
        row_span = np.max(score, axis=1) - np.min(score, axis=1)
        invalid = ~np.isfinite(weights).all(axis=1)
        degenerate = invalid | (row_span <= _ATTN_ROW_DEGENERATE_EPS)
        if np.any(degenerate):
            weights[degenerate] = (1.0 / 3.0)
        weights = np.divide(weights, np.maximum(np.sum(weights, axis=1, keepdims=True), _ATTN_EPS))

        entropy = -np.sum(weights * np.log(np.clip(weights, _ATTN_EPS, 1.0)), axis=1)
        self.last_attention_stats = {
            "entropy_mean": float(np.mean(entropy)) if entropy.size > 0 else 0.0,
            "lambda_safe": float(lambda_safe),
            "lambda_atlas": float(lambda_atlas),
        }
        return weights

    # -- One generation step --------------------------------------------
    def step(self, generation: int, max_generations: int) -> np.ndarray:
        """Vectorised MOQGWO update with linear GWO and optional attention fusion."""
        # Paper-standard GWO schedule: linear decay only.
        a = 2.0 - generation * (2.0 / max_generations)

        # Standard GWO estimate from 3 leaders
        X_terms = np.zeros((3, self.pop_size, self.dim), dtype=float)
        for j in range(3):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            A  = 2.0 * a * r1 - a
            C  = 2.0 * r2
            D  = np.abs(C * self.leaders[j] - self.positions)
            X_terms[j] = self.leaders[j] - A * D
        X_GWO = np.mean(X_terms, axis=0)

        if self.use_attention:
            leader_weights = self._attention_weights()  # (pop, 3)
            terms_by_wolf = np.transpose(X_terms, (1, 0, 2))  # (pop, 3, dim)
            new_positions = np.sum(leader_weights[:, :, None] * terms_by_wolf, axis=1)
        else:
            self.last_attention_stats = {
                "entropy_mean": 0.0,
                "lambda_safe": 0.0,
                "lambda_atlas": 0.0,
            }
            new_positions = X_GWO

        # Sanitize and clip
        finite_mask = np.isfinite(new_positions)
        if not np.all(finite_mask):
            center = 0.5 * (self.lower + self.upper)
            new_positions = np.where(finite_mask, new_positions, center)

        self.positions = np.clip(new_positions, self.lower, self.upper)
        return self.positions


def _decode_atlas_context(atlas_indices: np.ndarray | None, count: int) -> tuple[np.ndarray, np.ndarray]:
    if atlas_indices is None or count <= 0:
        return np.zeros(max(0, count), dtype=int), np.ones(max(0, count), dtype=int)
    raw = np.asarray(atlas_indices, dtype=int).reshape(-1)
    if raw.size != count:
        return np.zeros(count, dtype=int), np.ones(count, dtype=int)
    robust = np.maximum(1, raw // _ATLAS_BIN_MULT).astype(int)
    topology = np.mod(raw, _ATLAS_BIN_MULT).astype(int)
    return topology, robust


def _candidate_objective_context(candidate: Candidate) -> np.ndarray:
    obj_raw = np.asarray(candidate.objective, dtype=float).reshape(-1)
    details = candidate.details if isinstance(candidate.details, dict) else {}
    detail_proxy = np.asarray([
        float(details.get("makespan", np.nan)),
        float(details.get("energy", np.nan)),
        float(details.get("risk", np.nan)),
        float(details.get("turnPenalty", np.nan)),
    ], dtype=float)
    out = np.ones(4, dtype=float)
    use = min(4, obj_raw.size)
    if use > 0:
        out[:use] = obj_raw[:use]
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = detail_proxy[bad]
    out[~np.isfinite(out)] = 1.0
    return np.clip(out, 0.0, 1.0)


def _candidate_risk_context(candidate: Candidate, separation_min: float) -> float:
    obj = _candidate_objective_context(candidate)
    details = candidate.details if isinstance(candidate.details, dict) else {}
    conflict_rate = float(details.get("conflictRate", 0.0))
    if not np.isfinite(conflict_rate):
        conflict_rate = 1.0
    min_separation = float(details.get("minSeparation", np.nan))
    if np.isfinite(min_separation):
        sep_sev = float(np.clip((separation_min - min_separation) / max(separation_min, _ATTN_EPS), 0.0, 1.0))
    else:
        sep_sev = 1.0
    collision_violation = float(details.get("collisionViolation", 0.0))
    if not np.isfinite(collision_violation):
        collision_violation = 1.0
    min_clearance = float(details.get("minClearance", np.nan))
    if np.isfinite(min_clearance):
        clearance_sev = max(0.0, -min_clearance) / max(separation_min, _ATTN_EPS)
    else:
        clearance_sev = 5.0
    risk = (
        0.10 * float(np.clip(obj[1], 0.0, 1.0))
        + 0.10 * float(np.clip(conflict_rate, 0.0, 1.0))
        + 0.35 * sep_sev
        + 0.20 * float(np.clip(collision_violation, 0.0, 1.0))
        + 0.25 * clearance_sev
    )
    return float(max(0.0, risk))


def _candidate_is_feasible(candidate: Candidate) -> bool:
    obj = np.asarray(candidate.objective, dtype=float).reshape(-1)
    if obj.size == 0 or np.any(~np.isfinite(obj)):
        return False
    details = candidate.details if isinstance(candidate.details, dict) else {}
    if "feasible" in details:
        return float(details.get("feasible", 0.0)) > 0.5
    collision = float(details.get("collisionViolation", 0.0)) > 0.5
    separation = float(details.get("separationViolation", 0.0)) > 0.5
    return not (collision or separation)


def _attention_context_from_candidates(
    candidates: list[Candidate],
    *,
    separation_min: float,
    atlas_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    count = len(candidates)
    objectives = np.ones((count, 4), dtype=float)
    risk = np.ones(count, dtype=float)
    feasible_mask = np.zeros(count, dtype=bool)
    for idx, cand in enumerate(candidates):
        objectives[idx] = _candidate_objective_context(cand)
        risk[idx] = _candidate_risk_context(cand, separation_min=separation_min)
        feasible_mask[idx] = _candidate_is_feasible(cand)
    topology, robust = _decode_atlas_context(atlas_indices, count)
    feasible_ratio = float(np.mean(feasible_mask.astype(float))) if count > 0 else 1.0
    return objectives, risk, topology, robust, feasible_ratio


def _attention_leader_context(
    archive: list[Candidate],
    leader_indices: np.ndarray,
    *,
    separation_min: float,
    atlas_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj = np.ones((3, 4), dtype=float)
    risk = np.ones(3, dtype=float)
    top = np.zeros(3, dtype=int)
    robust = np.ones(3, dtype=int)
    if len(archive) <= 0 or leader_indices.size <= 0:
        return obj, risk, top, robust

    top_all, robust_all = _decode_atlas_context(atlas_indices, len(archive))
    for slot, raw_idx in enumerate(np.asarray(leader_indices, dtype=int).reshape(-1)[:3]):
        idx = int(np.clip(raw_idx, 0, len(archive) - 1))
        cand = archive[idx]
        obj[slot] = _candidate_objective_context(cand)
        risk[slot] = _candidate_risk_context(cand, separation_min=separation_min)
        if top_all.size == len(archive):
            top[slot] = int(top_all[idx])
            robust[slot] = int(robust_all[idx])
    robust = np.maximum(1, robust)
    return obj, risk, top, robust


# ─────────────────────────────────────────────────────────────────────
# Grid Archive
# ─────────────────────────────────────────────────────────────────────

def _build_grid(
    obj_matrix: np.ndarray,
    divisions: int,
    inflation_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if obj_matrix.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, obj_matrix.shape[1]))
    obj_safe = np.asarray(obj_matrix, dtype=float).copy()
    n_obj = obj_safe.shape[1]
    min_vals = np.zeros(n_obj, dtype=float)
    max_vals = np.zeros(n_obj, dtype=float)
    for j in range(n_obj):
        col = obj_safe[:, j]
        finite = np.isfinite(col)
        if np.any(finite):
            min_vals[j] = float(np.min(col[finite]))
            max_vals[j] = float(np.max(col[finite]))
            col[~finite] = max_vals[j]
            obj_safe[:, j] = col
        else:
            min_vals[j] = 0.0
            max_vals[j] = 1.0
            obj_safe[:, j] = 0.0
    if inflation_alpha > 0.0:
        delta = max_vals - min_vals
        min_vals = min_vals - inflation_alpha * delta
        max_vals = max_vals + inflation_alpha * delta
    with np.errstate(divide="ignore", invalid="ignore"):
        step = (max_vals - min_vals) / divisions
        raw  = np.floor((obj_safe - min_vals) / step)
    raw  = np.nan_to_num(raw, nan=0.0, posinf=divisions - 1, neginf=0.0)
    cell = np.clip(raw.astype(int), 0, divisions - 1)
    basis  = divisions ** np.arange(obj_matrix.shape[1])
    linear = (cell * basis).sum(axis=1)
    _, unique, counts = np.unique(linear, return_inverse=True, return_counts=True)
    return linear, unique, counts


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float).reshape(-1)
    n = logits.size
    if n == 0:
        return logits
    finite_mask = np.isfinite(logits)
    if not np.any(finite_mask):
        return np.ones(n, dtype=float) / float(n)
    work = np.where(finite_mask, logits, -np.inf)
    max_logit = np.max(work)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exps = np.exp(work - max_logit)
    exps[~finite_mask] = 0.0
    total = float(np.sum(exps))
    if not np.isfinite(total) or total <= 0.0:
        return np.ones(n, dtype=float) / float(n)
    return exps / total


def _occupancy_per_solution(indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(0, dtype=float)
    _, inverse, counts = np.unique(indices, return_inverse=True, return_counts=True)
    return counts[inverse].astype(float)


def _weighted_occ_sample(
    grid_indices: np.ndarray,
    atlas_indices: np.ndarray | None,
    objective_weight: float,
    atlas_weight: float,
    scale: float,
    inverse: bool,
) -> int:
    obj_occ = _occupancy_per_solution(grid_indices)
    quality = np.zeros_like(obj_occ, dtype=float)
    atlas_occ = (
        _occupancy_per_solution(atlas_indices)
        if (atlas_indices is not None and atlas_indices.size == grid_indices.size)
        else np.ones_like(obj_occ)
    )
    if atlas_indices is not None and atlas_indices.size == grid_indices.size:
        robust = np.maximum(1, (np.asarray(atlas_indices, dtype=int).reshape(-1) // _ATLAS_BIN_MULT)).astype(float)
        robust_span = float(np.max(robust) - np.min(robust))
        if robust_span > _ATTN_EPS:
            quality = (robust - np.min(robust)) / robust_span
    occ = objective_weight * obj_occ + atlas_weight * atlas_occ
    quality_bias = 3.5 * quality
    if inverse:
        logits = (-scale * occ) + quality_bias
    else:
        logits = (scale * occ) - quality_bias
    probs = _stable_softmax(logits)
    return int(np.random.choice(probs.shape[0], p=probs))


def _paper_select_cell_member(grid_indices: np.ndarray) -> int:
    """Paper-style leader selection: roulette over cells with P(cell) ∝ 1/Ni."""
    if grid_indices.size == 0:
        return 0
    cells, inverse, counts = np.unique(grid_indices, return_inverse=True, return_counts=True)
    del cells
    cell_weights = 1.0 / np.maximum(1.0, counts.astype(float))
    cell_probs = cell_weights / np.sum(cell_weights)
    chosen_cell = int(np.random.choice(cell_probs.shape[0], p=cell_probs))
    members = np.where(inverse == chosen_cell)[0]
    return int(members[np.random.randint(0, members.size)])


def _paper_delete_cell_member(grid_indices: np.ndarray) -> int:
    """Paper-style archive delete: roulette over most crowded tendency with P(cell) ∝ Ni."""
    if grid_indices.size == 0:
        return 0
    cells, inverse, counts = np.unique(grid_indices, return_inverse=True, return_counts=True)
    del cells
    cell_weights = counts.astype(float)
    cell_probs = cell_weights / np.sum(cell_weights)
    chosen_cell = int(np.random.choice(cell_probs.shape[0], p=cell_probs))
    members = np.where(inverse == chosen_cell)[0]
    return int(members[np.random.randint(0, members.size)])


def _dominates_objective(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= (b + eps)) and np.any(a < (b - eps)))


def _paper_archive_update(
    archive: list[Candidate],
    new_cands: list[Candidate],
    max_size: int,
    divisions: int,
) -> list[Candidate]:
    """Foundation-paper style archive controller (objective-only)."""
    rep = list(archive)
    if rep:
        rep_obj = np.stack([c.objective for c in rep])
        fronts, _ = n_d_sort(rep_obj.copy(), None, rep_obj.shape[0])
        rep = [rep[i] for i in np.where(fronts == 1)[0]]
    while len(rep) > max_size:
        rep_obj = np.stack([c.objective for c in rep])
        grid, _, _ = _build_grid(rep_obj, divisions, inflation_alpha=0.1)
        kill_idx = _paper_delete_cell_member(grid)
        rep.pop(kill_idx)

    for cand in new_cands:
        if not rep:
            rep.append(cand)
            continue

        dominated_by_rep = False
        dominates_rep: list[int] = []
        for idx, member in enumerate(rep):
            if _dominates_objective(member.objective, cand.objective):
                dominated_by_rep = True
                break
            if _dominates_objective(cand.objective, member.objective):
                dominates_rep.append(idx)
        if dominated_by_rep:
            continue
        if dominates_rep:
            for idx in sorted(dominates_rep, reverse=True):
                rep.pop(idx)

        if len(rep) >= max_size:
            rep_obj = np.stack([c.objective for c in rep])
            grid, _, _ = _build_grid(rep_obj, divisions, inflation_alpha=0.1)
            kill_idx = _paper_delete_cell_member(grid)
            rep.pop(kill_idx)
        rep.append(cand)

    if not rep:
        return rep
    rep_obj = np.stack([c.objective for c in rep])
    fronts, _ = n_d_sort(rep_obj.copy(), None, rep_obj.shape[0])
    rep = [rep[i] for i in np.where(fronts == 1)[0]]
    while len(rep) > max_size:
        rep_obj = np.stack([c.objective for c in rep])
        grid, _, _ = _build_grid(rep_obj, divisions, inflation_alpha=0.1)
        kill_idx = _paper_delete_cell_member(grid)
        rep.pop(kill_idx)
    return rep


# ─────────────────────────────────────────────────────────────────────
# Archive Update
# ─────────────────────────────────────────────────────────────────────

def _update_archive(
    archive: list[Candidate],
    new_cands: list[Candidate],
    atlas_indices: np.ndarray | None,    # pre-computed for all (archive + new)
    max_size: int,
    divisions: int,
    atlas_config: AtlasConfig,
    paper_standard: bool = False,
) -> tuple[list[Candidate], np.ndarray | None]:
    """Merge + prune archive using non-dominated sorting + Atlas truncation.

    Returns (kept_candidates, kept_atlas_indices).
    """
    if paper_standard:
        kept = _paper_archive_update(archive, new_cands, max_size, divisions)
        return kept, None

    all_cands = list(archive) + list(new_cands)
    if not all_cands:
        return [], (np.zeros(0, dtype=int) if atlas_indices is not None else None)

    total      = len(all_cands)
    obj_all    = np.stack([c.objective for c in all_cands])
    atlas_all: np.ndarray | None = None
    if atlas_indices is not None and atlas_indices.size == total:
        atlas_all = atlas_indices

    # ── Phase 1: Non-dominated sorting (progressive fronts) ──────────
    fronts, _ = n_d_sort(obj_all.copy(), None, total)
    selected: list[int] = []
    rank = 1
    while len(selected) < max_size:
        local_front = np.where(fronts == rank)[0]
        if local_front.size == 0:
            break
        global_front = local_front
        remaining = max_size - len(selected)
        if global_front.size <= remaining:
            selected.extend(global_front.tolist())
            rank += 1
            continue

        # ── Phase 2: Atlas truncation on partial front ──────────────
        pool_global = np.concatenate([np.asarray(selected, dtype=int), global_front])
        pool_obj = obj_all[pool_global]
        pool_atl = atlas_all[pool_global] if atlas_all is not None else None
        grid, _, _ = _build_grid(pool_obj, divisions)

        delete_mask = np.zeros(pool_global.size, dtype=bool)
        while int((~delete_mask).sum()) > max_size:
            active = np.where(~delete_mask)[0]
            active_grid = grid[active]
            active_atl = pool_atl[active] if pool_atl is not None else None
            kill_local = _weighted_occ_sample(
                active_grid,
                active_atl if atlas_config.enabled else None,
                atlas_config.objective_weight,
                atlas_config.atlas_weight,
                scale=10.0,
                inverse=False,
            )
            delete_mask[active[kill_local]] = True

        keep_global = pool_global[np.where(~delete_mask)[0]]
        kept_atlas = atlas_all[keep_global] if atlas_all is not None else None
        return [all_cands[i] for i in keep_global], kept_atlas

    if not selected:
        selected = np.arange(min(max_size, total), dtype=int).tolist()
    keep_global = np.asarray(selected[:max_size], dtype=int)
    kept_atlas = atlas_all[keep_global] if atlas_all is not None else None
    return [all_cands[i] for i in keep_global], kept_atlas


# ─────────────────────────────────────────────────────────────────────
# Atlas index computation
# ─────────────────────────────────────────────────────────────────────

def _atlas_for_candidates(
    cands: list[Candidate],
    model: dict,
    atlas_config: AtlasConfig,
) -> np.ndarray:
    """Compute topology-robustness atlas indices for a list of Candidates."""
    indices = np.zeros(len(cands), dtype=int)
    for i, cand in enumerate(cands):
        paths = cand.details.get("paths", []) if isinstance(cand.details, dict) else []
        fleet_sigs = []
        for p in paths:
            sig = topology_signature(p, model, atlas_config.max_obstacles)
            fleet_sigs.append(sig)
        avg_sig = np.mean(fleet_sigs, axis=0) if fleet_sigs else np.zeros(1)
        robust_cost = _candidate_objective_context(cand)
        _, rob_bin = robustness_from_cost(robust_cost, atlas_config.n_robust_bins)
        top_bin    = topology_bin_from_signature(avg_sig, atlas_config)
        indices[i] = rob_bin * 1000 + top_bin
    return indices


# ─────────────────────────────────────────────────────────────────────
# Leader Selection — objective-grid + atlas-aware
# ─────────────────────────────────────────────────────────────────────

def _select_leaders(
    archive: list[Candidate],
    atlas_indices: np.ndarray | None,
    divisions: int,
    atlas_config: AtlasConfig,
    paper_standard: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Select 3 leaders (alpha, beta, delta) from archive grid occupancy."""
    n = len(archive)
    if n == 0:
        return np.zeros((3, 1), dtype=float), np.zeros(3, dtype=int)

    pool_idx = np.arange(n)

    pool_obj  = np.stack([archive[i].objective for i in pool_idx])
    pool_atl  = atlas_indices[pool_idx] if atlas_indices is not None else None
    grid, _, _ = _build_grid(pool_obj, divisions, inflation_alpha=(0.1 if paper_standard else 0.0))

    leaders = []
    leader_archive_indices: list[int] = []
    available_local = np.arange(pool_idx.size)
    for _ in range(3):
        if available_local.size == 0:
            available_local = np.arange(pool_idx.size)
        candidate_local = available_local
        active_grid = grid[candidate_local]
        active_atl = pool_atl[candidate_local] if pool_atl is not None else None
        if paper_standard:
            idx_in_active = _paper_select_cell_member(active_grid)
            chosen_local = candidate_local[idx_in_active]
        else:
            idx_in_active = _weighted_occ_sample(
                active_grid,
                active_atl if atlas_config.enabled else None,
                atlas_config.objective_weight,
                atlas_config.atlas_weight,
                scale=10.0,
                inverse=True,
            )
            chosen_local = candidate_local[idx_in_active]
        chosen_archive_idx = int(pool_idx[chosen_local])
        leaders.append(archive[chosen_archive_idx].vector.copy())
        leader_archive_indices.append(chosen_archive_idx)
        available_local = available_local[available_local != chosen_local]

    return np.stack(leaders), np.asarray(leader_archive_indices, dtype=int)


# ─────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────

def _resolve_variant(raw: Any) -> str:
    key = str(raw).strip().lower()
    if key in {"", "full", "a2", "a2moqgwo", "a2-moqgwo"}:
        return "full"
    if key in {"no_attention", "no-attention", "noattention"}:
        return "no_attention"
    if key in {"standard_gwo", "standard-gwo", "gwo", "standard"}:
        return "standard_gwo"
    if key in {"gpu_strict", "gpu-strict", "moqgwo-gpu-strict", "moqgwo_gpu_strict"}:
        return "gpu_strict"
    return "full"


def _apply_variant(params: BenchmarkParams, *, variant: str | None = None, use_atlas: bool | None = None) -> BenchmarkParams:
    merged_extra = dict(params.extra) if isinstance(params.extra, dict) else {}
    if variant is not None:
        merged_extra["moqgwoVariant"] = variant
    if use_atlas is not None:
        merged_extra["moqgwoUseAtlas"] = bool(use_atlas)
    return replace(params, extra=merged_extra)


def run_fleet_moqgwo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    """MOQGWO family runner with attention fusion and Atlas-aware archive."""
    objective_count = 4
    model = dict(model)
    n_waypoints     = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value      = int(params.seed) if params.seed is not None else 0

    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["maxTurnDeg"]              = float(params.max_turn_deg)
    model["is_rl"]                   = False
    model["hardCollisionConstraint"] = True

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    variant = _resolve_variant(params.extra.get("moqgwoVariant", "full"))
    paper_standard = variant == "standard_gwo"
    use_atlas = bool(params.extra.get("moqgwoUseAtlas", True))
    use_gpu_strict = variant == "gpu_strict"
    use_attention = (variant != "no_attention") and (not paper_standard)
    if paper_standard:
        # Keep paper-standard MOGWO untouched.
        use_atlas = False
    torch_module = None
    gpu_device = None
    gpu_backend = "numpy:cpu"
    if use_gpu_strict:
        torch_module, gpu_device, gpu_backend = require_torch_gpu_for_moqgwo(params.gpu_mode)

    archive_size   = int(params.extra.get("nRep", params.population))
    grid_divisions = int(params.extra.get("nGrid", 10))
    metric_interval = int(params.extra.get("metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = (np.zeros((params.runs, 2), dtype=float)
                  if params.compute_metrics else np.zeros((0, 2), dtype=float))

    atlas_config = build_atlas_config({
        "useTopologyRobustArchive": use_atlas,
        "atlasTopologyBins": int(params.extra.get("atlasTopologyBins", 24)),
        "atlasRobustBins": int(params.extra.get("atlasRobustBins", 4)),
        "atlasMaxObstacles": int(params.extra.get("atlasMaxObstacles", 3)),
        "atlasHashLevels": int(params.extra.get("atlasHashLevels", 6)),
        "atlasObjectiveWeight": float(params.extra.get("atlasObjectiveWeight", 0.5)),
        "atlasTopologyWeight": float(params.extra.get("atlasTopologyWeight", 0.5)),
    })
    if not atlas_config.enabled:
        atlas_config.objective_weight = 1.0
        atlas_config.atlas_weight = 0.0
    separation_min = float(params.separation_min)
    run_indices     = _resolve_run_indices(params)
    resume_existing = bool(params.extra.get("resumeExistingRuns", True))

    for run_idx in run_indices:
        run_start = time.perf_counter()
        run_dir   = results_path / f"Run_{run_idx}"

        if resume_existing:
            resumed = _resume_run_scores(
                run_dir=run_dir, problem_index=params.problem_index,
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resumed is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resumed
                continue

        np.random.seed(seed_value * 1000 + run_idx)
        if use_gpu_strict:
            torch_module.manual_seed(seed_value * 1000 + run_idx)
            if str(gpu_device).startswith("cuda"):
                torch_module.cuda.reset_peak_memory_stats(gpu_device)

        # ── Initialise ────────────────────────────────────────────────
        if use_gpu_strict:
            engine = QGWOGPUStrictEngine(
                lower,
                upper,
                params.population,
                torch_module=torch_module,
                device=gpu_device,
                use_attention=use_attention,
                use_quantum=False,
            )
        else:
            engine = QGWO_Engine(
                lower,
                upper,
                params.population,
                use_attention=use_attention,
            )
        use_rtcs_init = bool(params.extra.get("moqgwoInitBias", False))
        if use_rtcs_init and (not paper_standard) and (not use_gpu_strict):
            engine.positions = _rtcs_initialize_population(
                engine.positions,
                model=model,
                fleet_size=fleet_size,
                n_waypoints=n_waypoints,
                lower=lower,
                upper=upper,
                separation_min=separation_min,
                atlas_config=atlas_config,
                use_topology=bool(atlas_config.enabled),
            )
        attention_context_enabled = bool(use_attention and (not paper_standard) and hasattr(engine, "set_attention_context"))
        hv_hist = (np.zeros((params.generations, 2), dtype=float)
                   if params.compute_metrics else np.zeros((0, 2), dtype=float))
        run_gpu_peak_bytes = 0.0
        attention_entropy_sum = 0.0
        attention_lambda_safe_sum = 0.0
        attention_lambda_atlas_sum = 0.0
        attention_steps = 0

        # Initial evaluation
        if use_gpu_strict:
            init_cands = evaluate_population_gpu_strict(
                engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints,
                torch_module=torch_module, device=gpu_device,
            )
            run_gpu_peak_bytes = max(run_gpu_peak_bytes, gpu_peak_bytes_for_device(torch_module, gpu_device))
        else:
            init_cands = _evaluate_population(
                engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )
        init_atlas = _atlas_for_candidates(init_cands, model, atlas_config) if atlas_config.enabled else None
        if attention_context_enabled:
            init_obj_ctx, init_risk_ctx, init_top_ctx, init_rob_ctx, init_feasible_ratio = _attention_context_from_candidates(
                init_cands,
                separation_min=separation_min,
                atlas_indices=init_atlas,
            )
        else:
            init_obj_ctx = np.ones((params.population, 4), dtype=float)
            init_risk_ctx = np.ones(params.population, dtype=float)
            init_top_ctx = np.zeros(params.population, dtype=int)
            init_rob_ctx = np.ones(params.population, dtype=int)
            init_feasible_ratio = 1.0

        archive: list[Candidate] = []
        arc_atlas: np.ndarray | None = np.zeros(0, dtype=int) if atlas_config.enabled else None
        active_atlas_config = atlas_config

        # Bootstrap archive
        archive, arc_atlas = _update_archive(
            [], init_cands, init_atlas,
            archive_size, grid_divisions, active_atlas_config,
            paper_standard=paper_standard,
        )

        # Set initial leaders from archive
        if archive:
            selected_leaders, selected_indices = _select_leaders(
                archive,
                arc_atlas,
                grid_divisions,
                active_atlas_config,
                paper_standard=paper_standard,
            )
            if use_gpu_strict:
                engine.set_leaders(selected_leaders)
            else:
                engine.leaders = selected_leaders
            if attention_context_enabled:
                leader_obj_ctx, leader_risk_ctx, leader_top_ctx, leader_rob_ctx = _attention_leader_context(
                    archive,
                    selected_indices,
                    separation_min=separation_min,
                    atlas_indices=arc_atlas,
                )
                engine.set_attention_context(
                    wolf_objectives=init_obj_ctx,
                    wolf_risk=init_risk_ctx,
                    feasibility_pressure=float(np.clip(1.0 - init_feasible_ratio, 0.0, 1.0)),
                    leader_objectives=leader_obj_ctx,
                    leader_risk=leader_risk_ctx,
                    wolf_topology=init_top_ctx,
                    wolf_robust=init_rob_ctx,
                    leader_topology=leader_top_ctx,
                    leader_robust=leader_rob_ctx,
                    atlas_enabled=bool(active_atlas_config.enabled),
                    atlas_robust_bins=int(active_atlas_config.n_robust_bins),
                )

        # ── Generation Loop ───────────────────────────────────────────
        for gen in range(1, params.generations + 1):
            # Update positions
            new_positions = engine.step(gen, params.generations)
            if use_attention and (not paper_standard):
                stats = getattr(engine, "last_attention_stats", None)
                if isinstance(stats, dict):
                    attention_entropy_sum += float(stats.get("entropy_mean", 0.0))
                    attention_lambda_safe_sum += float(stats.get("lambda_safe", 0.0))
                    attention_lambda_atlas_sum += float(stats.get("lambda_atlas", 0.0))
                    attention_steps += 1

            # Evaluate new population
            if use_gpu_strict:
                new_cands = evaluate_population_gpu_strict(
                    new_positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints,
                    torch_module=torch_module, device=gpu_device,
                )
                run_gpu_peak_bytes = max(run_gpu_peak_bytes, gpu_peak_bytes_for_device(torch_module, gpu_device))
            else:
                new_cands = _evaluate_population(
                    new_positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
                )
            new_atlas = (
                _atlas_for_candidates(new_cands, model, active_atlas_config)
                if active_atlas_config.enabled
                else None
            )
            if attention_context_enabled:
                new_obj_ctx, new_risk_ctx, new_top_ctx, new_rob_ctx, new_feasible_ratio = _attention_context_from_candidates(
                    new_cands,
                    separation_min=separation_min,
                    atlas_indices=new_atlas,
                )
            else:
                new_obj_ctx = np.ones((params.population, 4), dtype=float)
                new_risk_ctx = np.ones(params.population, dtype=float)
                new_top_ctx = np.zeros(params.population, dtype=int)
                new_rob_ctx = np.ones(params.population, dtype=int)
                new_feasible_ratio = 1.0

            # Archive update
            combined_atlas = None
            if arc_atlas is not None and new_atlas is not None:
                combined_atlas = np.concatenate([arc_atlas, new_atlas])
            archive, arc_atlas = _update_archive(
                archive, new_cands, combined_atlas,
                archive_size, grid_divisions, active_atlas_config,
                paper_standard=paper_standard,
            )

            if archive:
                selected_leaders, selected_indices = _select_leaders(
                    archive,
                    arc_atlas,
                    grid_divisions,
                    active_atlas_config,
                    paper_standard=paper_standard,
                )
                if use_gpu_strict:
                    engine.set_leaders(selected_leaders)
                else:
                    engine.leaders = selected_leaders
                if attention_context_enabled:
                    leader_obj_ctx, leader_risk_ctx, leader_top_ctx, leader_rob_ctx = _attention_leader_context(
                        archive,
                        selected_indices,
                        separation_min=separation_min,
                        atlas_indices=arc_atlas,
                    )
                    engine.set_attention_context(
                        wolf_objectives=new_obj_ctx,
                        wolf_risk=new_risk_ctx,
                        feasibility_pressure=float(np.clip(1.0 - new_feasible_ratio, 0.0, 1.0)),
                        leader_objectives=leader_obj_ctx,
                        leader_risk=leader_risk_ctx,
                        wolf_topology=new_top_ctx,
                        wolf_robust=new_rob_ctx,
                        leader_topology=leader_top_ctx,
                        leader_robust=leader_rob_ctx,
                        atlas_enabled=bool(active_atlas_config.enabled),
                        atlas_robust_bins=int(active_atlas_config.n_robust_bins),
                    )

            # Metrics
            if params.compute_metrics and hv_hist.shape[0] > 0:
                if gen == 1 or gen == params.generations or gen % metric_interval == 0:
                    if archive:
                        arc_obj = np.stack([c.objective for c in archive])
                        hv_hist[gen-1, 0] = cal_metric(1, arc_obj, params.problem_index, objective_count)
                        hv_hist[gen-1, 1] = cal_metric(2, arc_obj, params.problem_index, objective_count)
                elif gen > 1:
                    hv_hist[gen-1] = hv_hist[gen-2]

        # ── Finalize ──────────────────────────────────────────────────
        ensure_dir(run_dir)
        if params.compute_metrics and hv_hist.shape[0] > 0:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_hist})

        if not archive:
            # Pathological fallback
            if use_gpu_strict:
                last_cands = evaluate_population_gpu_strict(
                    engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints,
                    torch_module=torch_module, device=gpu_device,
                )
            else:
                last_cands = _evaluate_population(
                    engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
                )
            if paper_standard:
                last_obj = np.stack([c.objective for c in last_cands]) if last_cands else np.zeros((0, objective_count))
                if last_obj.size > 0:
                    fronts, _ = n_d_sort(last_obj.copy(), None, last_obj.shape[0])
                    front1 = np.where(fronts == 1)[0]
                    archive = [last_cands[i] for i in front1[:archive_size]]
                else:
                    archive = []
            else:
                last_obj = np.stack([c.objective for c in last_cands]) if last_cands else np.zeros((0, objective_count))
                if last_obj.size > 0:
                    fronts, _ = n_d_sort(last_obj.copy(), None, last_obj.shape[0])
                    selected = np.where(fronts == 1)[0]
                    if selected.size == 0:
                        selected = np.arange(min(archive_size, last_obj.shape[0]), dtype=int)
                    archive = [last_cands[i] for i in selected[:archive_size]]
                else:
                    archive = []

        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=archive,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend=gpu_backend if use_gpu_strict else "numpy:cpu",
            gpu_peak_bytes=float(run_gpu_peak_bytes if use_gpu_strict else 0.0),
            run_metadata={
                "algorithmName": "MOQGWO-GPU-STRICT" if use_gpu_strict else "MOQGWO",
                "representation": "cart",
                "moqgwoVariant": str(variant),
                "moqgwoUseAtlas": float(1.0 if atlas_config.enabled else 0.0),
                "requestedPopulation": float(params.population),
                "effectivePopulation": float(params.population),
                "archiveSize": float(archive_size),
                "moqgwoAttentionEntropyMean": float(attention_entropy_sum / max(1, attention_steps)),
                "moqgwoLambdaSafeMean": float(attention_lambda_safe_sum / max(1, attention_steps)),
                "moqgwoLambdaAtlasMean": float(attention_lambda_atlas_sum / max(1, attention_steps)),
            },
        )

        if params.compute_metrics:
            arc_obj = np.stack([c.objective for c in archive])
            run_scores[run_idx - 1] = np.array([
                cal_metric(1, arc_obj, params.problem_index, objective_count),
                cal_metric(2, arc_obj, params.problem_index, objective_count),
            ], dtype=float)

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_fleet_moqgwo_no_attention(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_moqgwo(model, _apply_variant(params, variant="no_attention"))


def run_fleet_moqgwo_no_atlas(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_moqgwo(model, _apply_variant(params, use_atlas=False))


def run_fleet_moqgwo_standard_gwo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_moqgwo(model, _apply_variant(params, variant="standard_gwo"))


def run_fleet_moqgwo_gpu_strict(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_moqgwo(model, _apply_variant(params, variant="gpu_strict"))
