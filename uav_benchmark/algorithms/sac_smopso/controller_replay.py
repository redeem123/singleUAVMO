from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from uav_benchmark.algorithms.sac_smopso.controller_types import (
    TemporalRelationalState,
    TensorStructuredStateBatch,
)
from uav_benchmark.algorithms.sac_smopso.torch_support import _TORCH_AVAILABLE, torch


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._data: deque[tuple[tuple[np.ndarray, ...], np.ndarray, int, float, tuple[np.ndarray, ...], float]] = deque(
            maxlen=max(32, int(capacity))
        )

    def __len__(self) -> int:
        return len(self._data)

    def serialize(self) -> list[tuple[tuple[np.ndarray, ...], np.ndarray, int, float, tuple[np.ndarray, ...], float]]:
        return [
            (
                _copy_serialized_state(item[0]),
                np.asarray(item[1], dtype=np.float32).copy(),
                int(item[2]),
                float(item[3]),
                _copy_serialized_state(item[4]),
                float(item[5]),
            )
            for item in self._data
        ]

    def load_serialized(
        self,
        payload: list[tuple[tuple[np.ndarray, ...], np.ndarray, int, float, tuple[np.ndarray, ...], float]] | None,
    ) -> None:
        self._data.clear()
        if not payload:
            return
        for state, continuous_action, operator_id, reward, next_state, done in payload:
            self._data.append(
                (
                    _copy_serialized_state(state),
                    np.asarray(continuous_action, dtype=np.float32).copy(),
                    int(operator_id),
                    float(reward),
                    _copy_serialized_state(next_state),
                    float(done),
                )
            )

    def add(
        self,
        state: TemporalRelationalState,
        continuous_action: np.ndarray,
        operator_id: int,
        reward: float,
        next_state: TemporalRelationalState,
        done: bool,
    ) -> None:
        self._data.append(
            (
                state.serialize(),
                np.asarray(continuous_action, dtype=np.float32).copy(),
                int(operator_id),
                float(reward),
                next_state.serialize(),
                1.0 if done else 0.0,
            )
        )

    def sample(
        self,
        batch_size: int,
        operator_count: int,
        device: Any,
        *,
        use_operator_head: bool = True,
    ) -> tuple[Any, ...]:
        if not _TORCH_AVAILABLE or torch is None:
            raise RuntimeError("Torch is required to sample replay tensors.")
        indices = np.random.choice(len(self._data), size=int(batch_size), replace=False)
        batch = [self._data[int(index)] for index in indices.tolist()]
        state_batch = _stack_serialized_states([item[0] for item in batch], device=device)
        cont_actions = np.stack([item[1] for item in batch], axis=0)
        operator_ids = np.asarray([item[2] for item in batch], dtype=np.int64)
        rewards = np.asarray([item[3] for item in batch], dtype=np.float32)
        next_state_batch = _stack_serialized_states([item[4] for item in batch], device=device)
        dones = np.asarray([item[5] for item in batch], dtype=np.float32)
        if use_operator_head and operator_count > 0:
            operator_one_hot = np.eye(operator_count, dtype=np.float32)[operator_ids]
        else:
            operator_one_hot = np.zeros((len(batch), max(0, int(operator_count))), dtype=np.float32)
        return (
            state_batch,
            torch.as_tensor(cont_actions, dtype=torch.float32, device=device),
            torch.as_tensor(operator_one_hot, dtype=torch.float32, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            next_state_batch,
            torch.as_tensor(dones, dtype=torch.float32, device=device),
        )


def _stack_serialized_states(
    serialized_states: list[tuple[np.ndarray, ...]], device: Any
) -> TensorStructuredStateBatch:
    if not _TORCH_AVAILABLE or torch is None:
        raise RuntimeError("Torch is required to stack structured states.")

    globals_ = np.stack([item[0] for item in serialized_states], axis=0)
    population_tokens = np.stack([item[1] for item in serialized_states], axis=0)
    population_mask = np.stack([item[2] for item in serialized_states], axis=0)
    archive_tokens = np.stack([item[3] for item in serialized_states], axis=0)
    archive_mask = np.stack([item[4] for item in serialized_states], axis=0)
    topology_tokens = np.stack([item[5] for item in serialized_states], axis=0)
    topology_mask = np.stack([item[6] for item in serialized_states], axis=0)
    interaction_tokens = np.stack([item[7] for item in serialized_states], axis=0)
    interaction_mask = np.stack([item[8] for item in serialized_states], axis=0)
    environment_tokens = np.stack([item[9] for item in serialized_states], axis=0)
    environment_mask = np.stack([item[10] for item in serialized_states], axis=0)
    temporal_tokens = np.stack([item[11] for item in serialized_states], axis=0)
    temporal_mask = np.stack([item[12] for item in serialized_states], axis=0)
    return TensorStructuredStateBatch(
        global_features=torch.as_tensor(globals_, dtype=torch.float32, device=device),
        population_tokens=torch.as_tensor(population_tokens, dtype=torch.float32, device=device),
        population_mask=torch.as_tensor(population_mask, dtype=torch.float32, device=device),
        archive_tokens=torch.as_tensor(archive_tokens, dtype=torch.float32, device=device),
        archive_mask=torch.as_tensor(archive_mask, dtype=torch.float32, device=device),
        topology_tokens=torch.as_tensor(topology_tokens, dtype=torch.float32, device=device),
        topology_mask=torch.as_tensor(topology_mask, dtype=torch.float32, device=device),
        interaction_tokens=torch.as_tensor(interaction_tokens, dtype=torch.float32, device=device),
        interaction_mask=torch.as_tensor(interaction_mask, dtype=torch.float32, device=device),
        environment_tokens=torch.as_tensor(environment_tokens, dtype=torch.float32, device=device),
        environment_mask=torch.as_tensor(environment_mask, dtype=torch.float32, device=device),
        temporal_tokens=torch.as_tensor(temporal_tokens, dtype=torch.float32, device=device),
        temporal_mask=torch.as_tensor(temporal_mask, dtype=torch.float32, device=device),
    )


def _copy_serialized_state(state: tuple[np.ndarray, ...] | list[Any]) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(item, dtype=np.float32).copy() for item in state)
