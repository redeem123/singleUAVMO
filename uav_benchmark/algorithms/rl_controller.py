from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class RLAction:
    inertia: float
    c1: float
    c2: float
    mutation_prob: float
    leader_bias: float
    velocity_scale: float = 1.0
    kappa_scale: float = 1.0
    region_scale: float = 1.0


def _normalize_allowed_actions(allowed_actions: np.ndarray | list[int] | tuple[int, ...] | None, n_actions: int) -> np.ndarray:
    if allowed_actions is None:
        return np.arange(n_actions, dtype=int)
    picks = np.asarray(allowed_actions, dtype=int).reshape(-1)
    picks = np.unique(picks[(picks >= 0) & (picks < n_actions)])
    if picks.size == 0:
        return np.arange(n_actions, dtype=int)
    return picks


@dataclass(slots=True)
class LinUCBController:
    n_features: int
    alpha: float = 1.0
    warmup_steps: int = 20
    actions: tuple[RLAction, ...] = field(
        default_factory=lambda: (
            RLAction(0.9, 1.8, 1.2, 0.1, 0.8),
            RLAction(0.9, 1.8, 1.2, 0.3, 0.8),
            RLAction(0.9, 1.2, 1.8, 0.1, 0.2),
            RLAction(0.9, 1.2, 1.8, 0.3, 0.2),
            RLAction(0.6, 1.8, 1.2, 0.1, 0.8),
            RLAction(0.6, 1.8, 1.2, 0.3, 0.8),
            RLAction(0.6, 1.2, 1.8, 0.1, 0.2),
            RLAction(0.6, 1.2, 1.8, 0.3, 0.2),
        )
    )
    _a: list[np.ndarray] = field(init=False, repr=False)
    _b: list[np.ndarray] = field(init=False, repr=False)
    _step: int = field(init=False, default=0, repr=False)
    _frozen: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._a = [np.eye(self.n_features, dtype=float) for _ in self.actions]
        self._b = [np.zeros(self.n_features, dtype=float) for _ in self.actions]

    def select_action(
        self,
        features: np.ndarray,
        allowed_actions: np.ndarray | list[int] | tuple[int, ...] | None = None,
    ) -> tuple[int, RLAction, np.ndarray]:
        x = np.asarray(features, dtype=float).reshape(-1)
        if x.size != self.n_features:
            raise ValueError(f"Feature size mismatch: got {x.size}, expected {self.n_features}")
        candidates = _normalize_allowed_actions(allowed_actions, len(self.actions))
        if (not self._frozen) and self._step < self.warmup_steps:
            idx = int(np.random.choice(candidates))
            theta = np.linalg.solve(self._a[idx], self._b[idx])
            self._step += 1
            return idx, self.actions[idx], theta

        scores = []
        for idx in candidates:
            a_inv = np.linalg.inv(self._a[idx])
            theta = a_inv @ self._b[idx]
            exploit = float(theta @ x)
            explore = 0.0 if self._frozen else self.alpha * float(np.sqrt(np.maximum(0.0, x @ a_inv @ x)))
            scores.append(exploit + explore)
        idx = int(candidates[int(np.argmax(np.asarray(scores, dtype=float)))])
        theta = np.linalg.solve(self._a[idx], self._b[idx])
        self._step += 1
        return idx, self.actions[idx], theta

    def update(self, action_idx: int, features: np.ndarray, reward: float) -> None:
        if self._frozen:
            return
        x = np.asarray(features, dtype=float).reshape(-1)
        self._a[action_idx] += np.outer(x, x)
        self._b[action_idx] += float(reward) * x

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = bool(frozen)

    def save(self, path: str | Path) -> None:
        payload = {
            "n_features": int(self.n_features),
            "alpha": float(self.alpha),
            "warmup_steps": int(self.warmup_steps),
            "step": int(self._step),
            "frozen": int(self._frozen),
            "a": np.asarray(self._a, dtype=float),
            "b": np.asarray(self._b, dtype=float),
        }
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            np.savez_compressed(handle, **payload)

    def load(self, path: str | Path, freeze: bool = False) -> bool:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return False
        payload = np.load(target, allow_pickle=False)
        a = np.asarray(payload["a"], dtype=float)
        b = np.asarray(payload["b"], dtype=float)
        if a.shape[0] != len(self.actions) or b.shape[0] != len(self.actions):
            payload.close()
            return False
        self._a = [a[idx].copy() for idx in range(a.shape[0])]
        self._b = [b[idx].copy() for idx in range(b.shape[0])]
        self._step = int(np.asarray(payload["step"]).reshape(-1)[0]) if "step" in payload else 0
        self._frozen = bool(freeze)
        payload.close()
        return True


class TorchBanditController:
    """GPU-native contextual bandit controller for RL-NMOPSO.

    This controller learns immediate reward prediction per action with a small
    MLP and replay training. The intent is twofold:
    1) Separate RL-NMOPSO from plain NMOPSO behavior.
    2) Move RL computation to GPU to increase RL-specific GPU utilization.
    """

    def __init__(
        self,
        n_features: int,
        actions: tuple[RLAction, ...],
        device: str,
        warmup_steps: int = 40,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        batch_size: int = 1024,
        train_steps: int = 12,
        min_train_size: int = 64,
        replay_capacity: int = 32768,
        epsilon_start: float = 0.20,
        epsilon_end: float = 0.02,
        epsilon_decay_steps: int = 2000,
        seed: int = 0,
        frozen: bool = False,
    ) -> None:
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency gate
            raise RuntimeError("TorchBanditController requires PyTorch.") from exc

        self._torch = torch
        self._nn = nn
        self.n_features = int(n_features)
        self.actions = tuple(actions)
        self.n_actions = len(self.actions)
        self.warmup_steps = int(max(0, warmup_steps))
        self.batch_size = int(max(8, batch_size))
        self.train_steps = int(max(1, train_steps))
        self.min_train_size = int(max(8, min_train_size))
        self.replay_capacity = int(max(self.batch_size, replay_capacity))
        self.epsilon_start = float(np.clip(epsilon_start, 0.0, 1.0))
        self.epsilon_end = float(np.clip(epsilon_end, 0.0, 1.0))
        self.epsilon_decay_steps = int(max(1, epsilon_decay_steps))
        self._step = 0
        self._rng = np.random.default_rng(seed)
        self._frozen = bool(frozen)

        self.device = torch.device(device)
        self.model = nn.Sequential(
            nn.Linear(self.n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self._feat = np.zeros((self.replay_capacity, self.n_features), dtype=np.float32)
        self._act = np.zeros(self.replay_capacity, dtype=np.int64)
        self._rew = np.zeros(self.replay_capacity, dtype=np.float32)
        self._size = 0
        self._ptr = 0
        self._loss_ema = 0.0

    @property
    def device_tag(self) -> str:
        return f"torch:{self.device.type}"

    @property
    def loss_ema(self) -> float:
        return float(self._loss_ema)

    @property
    def frozen(self) -> bool:
        return bool(self._frozen)

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = bool(frozen)

    def _epsilon(self) -> float:
        if self._frozen:
            return 0.0
        progress = min(1.0, max(0.0, self._step / self.epsilon_decay_steps))
        return float(self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress)

    def _append_replay(self, x: np.ndarray, action_idx: int, reward: float) -> None:
        self._feat[self._ptr] = x.astype(np.float32, copy=False)
        self._act[self._ptr] = int(action_idx)
        self._rew[self._ptr] = float(np.clip(reward, -1.0, 1.0))
        self._ptr = (self._ptr + 1) % self.replay_capacity
        self._size = min(self._size + 1, self.replay_capacity)

    def select_action(
        self,
        features: np.ndarray,
        allowed_actions: np.ndarray | list[int] | tuple[int, ...] | None = None,
    ) -> tuple[int, RLAction, np.ndarray]:
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        if x.size != self.n_features:
            raise ValueError(f"Feature size mismatch: got {x.size}, expected {self.n_features}")
        candidates = _normalize_allowed_actions(allowed_actions, self.n_actions)
        if (not self._frozen) and self._step < self.warmup_steps:
            idx = int(self._rng.choice(candidates))
            self._step += 1
            return idx, self.actions[idx], np.zeros(self.n_actions, dtype=float)

        eps = self._epsilon()
        if float(self._rng.random()) < eps:
            idx = int(self._rng.choice(candidates))
            q_np = np.zeros(self.n_actions, dtype=float)
        else:
            with self._torch.no_grad():
                x_t = self._torch.tensor(x, dtype=self._torch.float32, device=self.device).unsqueeze(0)
                q = self.model(x_t).squeeze(0)
                q_np = q.detach().cpu().numpy().astype(float, copy=False)
                local = q_np[candidates]
                idx = int(candidates[int(np.argmax(local))])
        self._step += 1
        return idx, self.actions[idx], q_np

    def update(self, action_idx: int, features: np.ndarray, reward: float) -> None:
        if self._frozen:
            return
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        self._append_replay(x, int(action_idx), float(reward))
        if self._size < self.min_train_size:
            return
        for _ in range(self.train_steps):
            take = min(self.batch_size, self._size)
            picks = self._rng.integers(0, self._size, size=take)
            x_b = self._torch.tensor(self._feat[picks], dtype=self._torch.float32, device=self.device)
            a_b = self._torch.tensor(self._act[picks], dtype=self._torch.long, device=self.device)
            r_b = self._torch.tensor(self._rew[picks], dtype=self._torch.float32, device=self.device)
            pred_all = self.model(x_b)
            pred = pred_all.gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = self.loss_fn(pred, r_b)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            value = float(loss.detach().item())
            self._loss_ema = value if self._loss_ema == 0.0 else (0.95 * self._loss_ema + 0.05 * value)

    def save(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(self._step),
            "loss_ema": float(self._loss_ema),
            "size": int(self._size),
            "ptr": int(self._ptr),
            "feat": self._feat[: self._size].copy(),
            "act": self._act[: self._size].copy(),
            "rew": self._rew[: self._size].copy(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        self._torch.save(payload, target)

    def load(self, path: str | Path, freeze: bool = False) -> bool:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return False
        payload = self._torch.load(target, map_location=self.device)
        model_payload = payload.get("model")
        if model_payload:
            self.model.load_state_dict(model_payload)
        optimizer_payload = payload.get("optimizer")
        if optimizer_payload:
            try:
                self.optimizer.load_state_dict(optimizer_payload)
            except Exception:
                pass
        self._step = int(payload.get("step", 0))
        self._loss_ema = float(payload.get("loss_ema", 0.0))
        size = int(payload.get("size", 0))
        size = int(np.clip(size, 0, self.replay_capacity))
        self._size = size
        self._ptr = int(payload.get("ptr", size % max(1, self.replay_capacity)))
        if size > 0:
            feat = np.asarray(payload.get("feat"), dtype=np.float32)
            act = np.asarray(payload.get("act"), dtype=np.int64)
            rew = np.asarray(payload.get("rew"), dtype=np.float32)
            use = min(size, feat.shape[0], act.shape[0], rew.shape[0])
            self._feat[:use] = feat[:use]
            self._act[:use] = act[:use]
            self._rew[:use] = rew[:use]
            self._size = use
            self._ptr = use % max(1, self.replay_capacity)
        self._frozen = bool(freeze)
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "device": self.device_tag,
            "replaySize": int(self._size),
            "lossEma": float(self._loss_ema),
            "steps": int(self._step),
            "frozen": bool(self._frozen),
        }
