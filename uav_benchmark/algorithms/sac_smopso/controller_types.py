from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _masked_mean_np(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    matrix = np.asarray(tokens, dtype=np.float32)
    weights = np.asarray(mask, dtype=np.float32).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or weights.size == 0:
        width = matrix.shape[1] if matrix.ndim == 2 else 0
        return np.zeros(width, dtype=np.float32)
    valid = weights > 0.5
    if not np.any(valid):
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return np.asarray(np.mean(matrix[valid], axis=0, dtype=np.float32), dtype=np.float32).reshape(-1)


@dataclass(slots=True)
class TemporalRelationalStateSpec:
    global_dim: int
    population_dim: int
    archive_dim: int
    topology_dim: int
    interaction_dim: int
    environment_dim: int
    temporal_dim: int


@dataclass(slots=True)
class TemporalRelationalState:
    global_features: np.ndarray
    population_tokens: np.ndarray
    population_mask: np.ndarray
    archive_tokens: np.ndarray
    archive_mask: np.ndarray
    topology_tokens: np.ndarray
    topology_mask: np.ndarray
    interaction_tokens: np.ndarray
    interaction_mask: np.ndarray
    environment_tokens: np.ndarray
    environment_mask: np.ndarray
    temporal_tokens: np.ndarray
    temporal_mask: np.ndarray

    def summary_vector(self) -> np.ndarray:
        pieces = [
            np.asarray(self.global_features, dtype=np.float32).reshape(-1),
            _masked_mean_np(self.population_tokens, self.population_mask),
            _masked_mean_np(self.archive_tokens, self.archive_mask),
            _masked_mean_np(self.topology_tokens, self.topology_mask),
            _masked_mean_np(self.interaction_tokens, self.interaction_mask),
            _masked_mean_np(self.environment_tokens, self.environment_mask),
            _masked_mean_np(self.temporal_tokens, self.temporal_mask),
        ]
        return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)

    def serialize(self) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(self.global_features, dtype=np.float32).copy(),
            np.asarray(self.population_tokens, dtype=np.float32).copy(),
            np.asarray(self.population_mask, dtype=np.float32).copy(),
            np.asarray(self.archive_tokens, dtype=np.float32).copy(),
            np.asarray(self.archive_mask, dtype=np.float32).copy(),
            np.asarray(self.topology_tokens, dtype=np.float32).copy(),
            np.asarray(self.topology_mask, dtype=np.float32).copy(),
            np.asarray(self.interaction_tokens, dtype=np.float32).copy(),
            np.asarray(self.interaction_mask, dtype=np.float32).copy(),
            np.asarray(self.environment_tokens, dtype=np.float32).copy(),
            np.asarray(self.environment_mask, dtype=np.float32).copy(),
            np.asarray(self.temporal_tokens, dtype=np.float32).copy(),
            np.asarray(self.temporal_mask, dtype=np.float32).copy(),
        )


@dataclass(slots=True)
class ControllerAction:
    operator_id: int
    continuous: np.ndarray
    operator_probs: np.ndarray
    log_prob: float
    source: str


@dataclass(slots=True)
class ControllerConfig:
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.98
    tau: float = 0.01
    replay_capacity: int = 2048
    batch_size: int = 32
    warmup_steps: int = 8
    updates_per_step: int = 1
    alpha_init: float = 0.08
    scratch_policy_mix_start: float = 0.35
    scratch_policy_mix_end: float = 0.90
    loaded_policy_mix_start: float = 0.60
    loaded_policy_mix_end: float = 1.00
    policy_mix_anneal_steps: int = 200
    use_operator_head: bool = True


@dataclass(slots=True)
class TensorStructuredStateBatch:
    global_features: Any
    population_tokens: Any
    population_mask: Any
    archive_tokens: Any
    archive_mask: Any
    topology_tokens: Any
    topology_mask: Any
    interaction_tokens: Any
    interaction_mask: Any
    environment_tokens: Any
    environment_mask: Any
    temporal_tokens: Any
    temporal_mask: Any
