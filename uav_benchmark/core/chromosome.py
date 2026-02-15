from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uav_benchmark.core.evaluate_path import evaluate_path


def _sample_ground_floor(model: dict[str, Any], x_coord: float, y_coord: float) -> float:
    """Ground sample used for MATLAB-aligned start/end initialization."""
    height_map = np.asarray(model["H"], dtype=float)
    y_index = max(1, int(math.floor(y_coord)))
    x_index = max(1, int(math.floor(x_coord)))
    y_index = min(y_index, height_map.shape[0])
    x_index = min(x_index, height_map.shape[1])
    return float(height_map[y_index - 1, x_index - 1])


def _sample_ground_envelope(model: dict[str, Any], x_coord: float, y_coord: float) -> float:
    """Ground sample used for intermediate points (max of four neighbors)."""
    height_map = np.asarray(model["H"], dtype=float)
    xmax = int(model["xmax"])
    ymax = int(model["ymax"])
    x1 = max(1, int(math.floor(x_coord)))
    x2 = min(xmax, int(math.ceil(x_coord)))
    y1 = max(1, int(math.floor(y_coord)))
    y2 = min(ymax, int(math.ceil(y_coord)))

    if xmax == 20:
        # Keep parity with MATLAB's legacy small-map indexing branch.
        x_candidates = [x1 * 10, x2 * 10]
        y_candidates = [y1 * 10, y2 * 10]
    else:
        x_candidates = [x1, x2]
        y_candidates = [y1, y2]

    ground_values: list[float] = []
    for y_raw in y_candidates:
        for x_raw in x_candidates:
            yi = max(1, min(int(y_raw), height_map.shape[0])) - 1
            xi = max(1, min(int(x_raw), height_map.shape[1])) - 1
            ground_values.append(float(height_map[yi, xi]))
    return max(ground_values) if ground_values else 0.0


def _safe_height_start_end(model: dict[str, Any], x_coord: float, y_coord: float) -> float:
    return _sample_ground_floor(model, x_coord, y_coord) + float(model.get("safeH", 0.0))


def _safe_height_envelope(model: dict[str, Any], x_coord: float, y_coord: float) -> float:
    return _sample_ground_envelope(model, x_coord, y_coord) + float(model.get("safeH", 0.0))


@dataclass(slots=True)
class Chromosome:
    rnvec: np.ndarray
    path: np.ndarray
    objs: np.ndarray = field(default_factory=lambda: np.full(4, np.inf, dtype=float))
    front: int = 0
    vel: np.ndarray | None = None
    crowding_distance: float = 0.0
    rank: int = 0
    cons: float = 0.0
    domination_count: int = 0
    dominated_set: list[int] = field(default_factory=list)
    high_bound: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    @classmethod
    def new(cls, model: dict[str, Any]) -> "Chromosome":
        dim = int(model["n"])
        vector = np.zeros((dim, 3), dtype=float)
        vector[0] = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
        vector[-1] = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
        high_bound = np.zeros(dim, dtype=float)
        high_bound[0] = float(model["xmin"])
        bound = float(model["xmax"])
        for index in range(1, dim - 1):
            high_bound[index] = (index + 1) * bound / dim
        high_bound[-1] = bound
        return cls(rnvec=vector, path=vector.copy(), high_bound=high_bound)

    def copy(self) -> "Chromosome":
        cloned = Chromosome(
            rnvec=self.rnvec.copy(),
            path=self.path.copy(),
            objs=self.objs.copy(),
            front=self.front,
            vel=None if self.vel is None else self.vel.copy(),
            crowding_distance=self.crowding_distance,
            rank=self.rank,
            cons=self.cons,
            domination_count=self.domination_count,
            dominated_set=list(self.dominated_set),
            high_bound=self.high_bound.copy(),
        )
        return cloned

    def initialize(self, model: dict[str, Any]) -> "Chromosome":
        if int(model.get("ymax", 0)) == 200:
            variation = 20.0
        else:
            variation = 5.0
        n_points = int(model["n"])
        interpolation = np.linspace(0.0, 1.0, n_points)
        start = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
        end = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
        self.rnvec[:, 0] = start[0] + interpolation * (end[0] - start[0])
        self.rnvec[:, 1] = start[1] + interpolation * (end[1] - start[1])
        noise = (np.random.rand(n_points - 2, 2) - 0.5) * (2.0 * variation)
        self.rnvec[1:-1, 0:2] += noise
        self.rnvec[:, 0] = np.clip(self.rnvec[:, 0], float(model["xmin"]), float(model["xmax"]))
        self.rnvec[:, 1] = np.clip(self.rnvec[:, 1], float(model["ymin"]), float(model["ymax"]))
        self.path = self.rnvec.copy()
        self.adjust_constraint_turning_angle(model)
        return self

    def _horizontal_turn_violation(self, index: int) -> bool:
        alpha = self._horizontal_turn_angle_deg(index)
        if alpha is None:
            return False
        return alpha < 75.0

    def _horizontal_turn_angle_deg(self, index: int) -> float | None:
        segment_1 = float(np.linalg.norm(self.path[index, :2] - self.path[index - 1, :2]))
        segment_2 = float(np.linalg.norm(self.path[index - 1, :2] - self.path[index - 2, :2]))
        segment_3 = float(np.linalg.norm(self.path[index, :2] - self.path[index - 2, :2]))
        if segment_1 <= 1e-12 or segment_2 <= 1e-12:
            return None
        cosine_alpha = (segment_1 ** 2 + segment_2 ** 2 - segment_3 ** 2) / (2.0 * segment_1 * segment_2)
        cosine_alpha = float(np.clip(cosine_alpha, -1.0, 1.0))
        return float(math.degrees(math.acos(cosine_alpha)))

    def adjust_constraint_turning_angle(self, model: dict[str, Any]) -> "Chromosome":
        self.path = self.rnvec.copy()
        self.path[0, 0] = float(np.asarray(model["start"]).reshape(-1)[0])
        self.path[0, 1] = float(np.asarray(model["start"]).reshape(-1)[1])
        self.path[0, 2] = _safe_height_start_end(model, self.path[0, 0], self.path[0, 1])

        n_points = self.path.shape[0]
        if int(model.get("ymax", 0)) == 200:
            horizontal_min = -2.0
            horizontal_max = 10.0
        else:
            horizontal_min = -5.0
            horizontal_max = 5.0
        for point_index in range(1, n_points - 1):
            retries = 0
            while point_index > 2 and self._horizontal_turn_violation(point_index) and retries < 10:
                if np.random.rand() < 0.5:
                    self.path[point_index, 1] = self.path[point_index - 1, 1] + np.random.uniform(horizontal_min, horizontal_max)
                else:
                    self.path[point_index, 0] = self.path[point_index - 1, 0] + np.random.uniform(horizontal_min, horizontal_max)
                self.path[point_index, 0] = np.clip(self.path[point_index, 0], float(model["xmin"]), float(model["xmax"]))
                self.path[point_index, 1] = np.clip(self.path[point_index, 1], float(model["ymin"]), float(model["ymax"]))
                retries += 1

            horizontal_distance = float(np.linalg.norm(self.path[point_index, :2] - self.path[point_index - 1, :2]))
            max_delta_z = horizontal_distance * math.tan(math.radians(60.0))
            min_height = _safe_height_envelope(model, self.path[point_index, 0], self.path[point_index, 1])
            previous_height = self.path[point_index - 1, 2]
            if min_height < previous_height - max_delta_z:
                target_height = previous_height - max_delta_z
            else:
                target_height = min_height
            self.path[point_index, 2] = target_height

        end_point = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
        end_height = _safe_height_start_end(model, end_point[0], end_point[1])
        self.path[-1, 0] = end_point[0]
        self.path[-1, 1] = end_point[1]
        self.path[-1, 2] = end_height
        self.rnvec = self.path.copy()
        return self

    def evaluate(self, model: dict[str, Any]) -> "Chromosome":
        try:
            objectives = evaluate_path(self.path, model)
            if objectives.shape[0] != 4:
                objectives = np.full(4, np.inf, dtype=float)
        except (ValueError, IndexError, FloatingPointError):
            objectives = np.full(4, np.inf, dtype=float)
        self.objs = objectives
        self.compute_constraint_violation()
        return self

    def compute_constraint_violation(self) -> None:
        violation = 0.0
        for index in range(1, self.path.shape[0] - 1):
            horizontal_length = float(np.linalg.norm(self.path[index, :2] - self.path[index - 1, :2]))
            if horizontal_length > 0:
                slope = abs(self.path[index, 2] - self.path[index - 1, 2]) / horizontal_length
                vertical_angle = math.degrees(math.atan(slope))
                if vertical_angle > 60.0:
                    violation += vertical_angle - 60.0
            if index > 2:
                horizontal_angle = self._horizontal_turn_angle_deg(index)
                if horizontal_angle is not None and horizontal_angle < 75.0:
                    # Keep parity with MATLAB: penalize by angle deficit.
                    violation += abs(horizontal_angle - 75.0)
        self.cons = float(violation)
