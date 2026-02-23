"""Unified Adaptive Controller for RL-NMOPSO.

Single MLP that jointly outputs:
  - Continuous PSO parameters: w ∈ [0.4, 1.1], c1 ∈ [1.0, 2.5], c2 ∈ [1.0, 2.5]
  - Operator selection logits: 4 arms (noop / SBX / DE / elite-refine)

This replaces the previous dual LinUCB+FRRMAB two-level control with a single
unified decision-maker — the core novelty of RL-NMOPSO.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ── Action output ───────────────────────────────────────────────────

N_CONTINUOUS = 4    # (w, c1, c2, repulsion_weight)
N_OPERATORS = 4     # (noop, SBX, DE, elite-refine)
N_FEATURES = 6      # state vector dimensionality
N_OUTPUTS = N_CONTINUOUS + N_OPERATORS  # 8

# Continuous output ranges (applied via sigmoid scaling)
W_RANGE = (0.40, 1.10)
C1_RANGE = (1.00, 2.50)
C2_RANGE = (1.00, 2.50)
REP_WT_RANGE = (0.00, 5.00)

ATTENTION_MODE_COSINE = "cosine"
ATTENTION_MODE_LEARNED = "learned"


@dataclass(slots=True)
class ContinuousAction:
    """Decoded output from the unified controller."""
    w: float
    c1: float
    c2: float
    repulsion_weight: float
    operator: int       # 0=noop, 1=SBX, 2=DE, 3=elite-refine
    action_idx: int = 0  # raw index for logging


def _scale_continuous(raw: np.ndarray) -> tuple[float, float, float, float]:
    """Apply sigmoid + range scaling to raw continuous outputs."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(raw[:N_CONTINUOUS], -10, 10)))
    w = float(W_RANGE[0] + sig[0] * (W_RANGE[1] - W_RANGE[0]))
    c1 = float(C1_RANGE[0] + sig[1] * (C1_RANGE[1] - C1_RANGE[0]))
    c2 = float(C2_RANGE[0] + sig[2] * (C2_RANGE[1] - C2_RANGE[0]))
    rep = float(REP_WT_RANGE[0] + sig[3] * (REP_WT_RANGE[1] - REP_WT_RANGE[0]))
    return w, c1, c2, rep


def _select_operator(logits: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """ε-greedy operator selection from logits."""
    if float(rng.random()) < epsilon:
        return int(rng.integers(0, N_OPERATORS))
    return int(np.argmax(logits))


def _softmax_rows(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if scores.size == 0:
        return np.zeros_like(scores, dtype=float)
    temp = float(max(1e-6, temperature))
    logits = np.asarray(scores, dtype=float) / temp
    logits = np.where(np.isfinite(logits), logits, -1e9)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(logits, -60.0, 60.0))
    exp = np.where(np.isfinite(exp), exp, 0.0)
    denom = np.sum(exp, axis=1, keepdims=True)
    invalid = denom[:, 0] <= 1e-12
    if np.any(invalid):
        exp[invalid] = 1.0
        denom = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def _cosine_attention_weights(
    particle_features: np.ndarray,
    archive_features: np.ndarray,
    temperature: float = 0.35,
) -> np.ndarray:
    p = np.asarray(particle_features, dtype=float)
    a = np.asarray(archive_features, dtype=float)

    if p.ndim != 2 or a.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    if p.shape[0] == 0 or a.shape[0] == 0:
        return np.zeros((p.shape[0], a.shape[0]), dtype=float)
    if p.shape[1] != a.shape[1]:
        return np.zeros((p.shape[0], a.shape[0]), dtype=float)

    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    p_norm = np.linalg.norm(p, axis=1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    p_unit = p / np.maximum(p_norm, 1e-9)
    a_unit = a / np.maximum(a_norm, 1e-9)

    similarity = p_unit @ a_unit.T
    return _softmax_rows(similarity, temperature=temperature)


# ── Unified Controller (GPU, PyTorch) ──────────────────────────────

class UnifiedController:
    """Single MLP for joint continuous-param + operator selection.

    Architecture: 6 → hidden → hidden → 8
      outputs[0:4]  → sigmoid-scaled (w, c1, c2, repulsion_weight)
      outputs[4:8]  → softmax/argmax for operator selection

    Training: SmoothL1 loss on full 8-dim output via replay buffer.
    The continuous heads learn to predict optimal PSO params.
    The operator head learns which auxiliary operator maximizes reward.
    """

    def __init__(
        self,
        device: str = "cpu",
        hidden_dim: int = 64,
        lr: float = 1e-3,
        batch_size: int = 512,
        train_steps: int = 8,
        min_train_size: int = 64,
        replay_capacity: int = 16384,
        epsilon_start: float = 0.30,
        epsilon_end: float = 0.03,
        epsilon_decay_steps: int = 2000,
        warmup_steps: int = 30,
        attention_mode: str = ATTENTION_MODE_COSINE,
        attention_key_dim: int = 16,
        attention_lr: float = 5e-4,
        attention_batch_size: int = 16,
        attention_train_steps: int = 1,
        attention_min_train_size: int = 16,
        attention_replay_capacity: int = 1024,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("UnifiedController requires PyTorch.") from exc

        self._torch = torch
        self._nn = nn
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
        self._frozen = False
        mode = str(attention_mode).strip().lower()
        self.attention_mode = mode if mode in {ATTENTION_MODE_COSINE, ATTENTION_MODE_LEARNED} else ATTENTION_MODE_COSINE
        self.attention_key_dim = int(max(4, attention_key_dim))
        self.attention_lr = float(max(1e-6, attention_lr))
        self.attention_batch_size = int(max(1, attention_batch_size))
        self.attention_train_steps = int(max(1, attention_train_steps))
        self.attention_min_train_size = int(max(1, attention_min_train_size))
        self.attention_replay_capacity = int(max(8, attention_replay_capacity))

        self.device = torch.device(device)
        self.model = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_OUTPUTS),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        # Replay buffer
        self._feat = np.zeros((self.replay_capacity, N_FEATURES), dtype=np.float32)
        self._targets = np.zeros((self.replay_capacity, N_OUTPUTS), dtype=np.float32)
        self._masks = np.zeros((self.replay_capacity, N_OUTPUTS), dtype=np.float32)
        self._size = 0
        self._ptr = 0
        self._loss_ema = 0.0

        # Optional learned attention head (initialized lazily on first feature call)
        self._attention_q: Any | None = None
        self._attention_k: Any | None = None
        self._attention_optimizer: Any | None = None
        self._attention_input_dim: int = 0
        self._attention_replay: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self._attention_loss_ema: float = 0.0

    @property
    def device_tag(self) -> str:
        return f"unified:{self.device}"

    @property
    def loss_ema(self) -> float:
        return self._loss_ema

    @property
    def attention_loss_ema(self) -> float:
        return self._attention_loss_ema

    @property
    def frozen(self) -> bool:
        return self._frozen

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = bool(frozen)

    def _epsilon(self) -> float:
        progress = min(1.0, self._step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _ensure_attention_modules(self, feature_dim: int) -> bool:
        if self.attention_mode != ATTENTION_MODE_LEARNED:
            return False
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            return False
        if (
            self._attention_q is not None
            and self._attention_k is not None
            and self._attention_optimizer is not None
            and self._attention_input_dim == feature_dim
        ):
            return True

        self._attention_q = self._nn.Linear(feature_dim, self.attention_key_dim).to(self.device)
        self._attention_k = self._nn.Linear(feature_dim, self.attention_key_dim).to(self.device)
        params = list(self._attention_q.parameters()) + list(self._attention_k.parameters())
        self._attention_optimizer = self._torch.optim.Adam(params, lr=self.attention_lr)
        self._attention_input_dim = feature_dim
        self._attention_replay.clear()
        self._attention_loss_ema = 0.0
        return True

    def _attention_forward_torch(
        self,
        particle_features: Any,
        archive_features: Any,
        temperature: float,
    ) -> Any:
        if self._attention_q is None or self._attention_k is None:
            raise RuntimeError("Attention modules are not initialized.")
        q = self._attention_q(particle_features)
        k = self._attention_k(archive_features)
        scale = float(np.sqrt(max(1, int(q.shape[1]))))
        logits = (q @ k.T) / scale
        logits = logits / float(max(1e-6, temperature))
        return self._torch.softmax(logits, dim=1)

    def _train_attention_step(self, temperature: float) -> None:
        if (
            self.attention_mode != ATTENTION_MODE_LEARNED
            or self._attention_optimizer is None
            or self._attention_q is None
            or self._attention_k is None
        ):
            return
        replay_size = len(self._attention_replay)
        if replay_size < self.attention_min_train_size:
            return

        torch = self._torch
        for _ in range(self.attention_train_steps):
            batch_size = min(self.attention_batch_size, replay_size)
            indices = self._rng.integers(0, replay_size, size=batch_size)
            loss_accum = torch.zeros((), dtype=torch.float32, device=self.device)
            for idx in indices:
                part_np, arch_np, teacher_np, weight = self._attention_replay[int(idx)]
                p_t = torch.tensor(part_np, dtype=torch.float32, device=self.device)
                a_t = torch.tensor(arch_np, dtype=torch.float32, device=self.device)
                teacher_t = torch.tensor(teacher_np, dtype=torch.float32, device=self.device)
                pred = self._attention_forward_torch(p_t, a_t, temperature=temperature)
                pred = torch.clamp(pred, 1e-9, 1.0)
                ce = -torch.sum(teacher_t * torch.log(pred), dim=1).mean()
                entropy = -torch.sum(pred * torch.log(pred), dim=1).mean()
                sample_loss = float(weight) * ce - 0.20 * entropy
                loss_accum = loss_accum + sample_loss
            loss = loss_accum / float(max(1, batch_size))
            self._attention_optimizer.zero_grad()
            loss.backward()
            self._attention_optimizer.step()
            self._attention_loss_ema = 0.95 * self._attention_loss_ema + 0.05 * float(loss.item())

    def select_action(self, features: np.ndarray) -> ContinuousAction:
        """Select a continuous action from the state features."""
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        if x.size != N_FEATURES:
            raise ValueError(f"Feature size mismatch: got {x.size}, expected {N_FEATURES}")

        self._step += 1

        # Warmup: random actions
        if (not self._frozen) and self._step <= self.warmup_steps:
            w = float(self._rng.uniform(*W_RANGE))
            c1 = float(self._rng.uniform(*C1_RANGE))
            c2 = float(self._rng.uniform(*C2_RANGE))
            rep = float(self._rng.uniform(*REP_WT_RANGE))
            op = int(self._rng.integers(0, N_OPERATORS))
            return ContinuousAction(w=w, c1=c1, c2=c2, repulsion_weight=rep, operator=op, action_idx=-1)

        # Forward pass
        with self._torch.no_grad():
            x_t = self._torch.tensor(x, dtype=self._torch.float32, device=self.device).unsqueeze(0)
            raw = self.model(x_t).squeeze(0).cpu().numpy().astype(float, copy=False)

        w, c1, c2, rep = _scale_continuous(raw)
        eps = self._epsilon() if not self._frozen else 0.0

        # Add Gaussian noise to continuous params during exploration
        if not self._frozen and eps > 0.01:
            noise_scale = eps * 0.3  # small noise proportional to exploration
            w = float(np.clip(w + self._rng.normal(0, noise_scale * (W_RANGE[1] - W_RANGE[0])), *W_RANGE))
            c1 = float(np.clip(c1 + self._rng.normal(0, noise_scale * (C1_RANGE[1] - C1_RANGE[0])), *C1_RANGE))
            c2 = float(np.clip(c2 + self._rng.normal(0, noise_scale * (C2_RANGE[1] - C2_RANGE[0])), *C2_RANGE))
            rep = float(np.clip(rep + self._rng.normal(0, noise_scale * (REP_WT_RANGE[1] - REP_WT_RANGE[0])), *REP_WT_RANGE))

        op = _select_operator(raw[N_CONTINUOUS:], eps, self._rng)
        return ContinuousAction(w=w, c1=c1, c2=c2, repulsion_weight=rep, operator=op, action_idx=op)

    def update(self, features: np.ndarray, reward: float, action: ContinuousAction) -> None:
        """Store experience and train on replay buffer."""
        if self._frozen:
            return

        # Build target: for the continuous heads, store the actual values used
        # scaled back to raw space. For the operator head, store reward for the
        # selected operator.
        target = np.zeros(N_OUTPUTS, dtype=np.float32)

        # Inverse sigmoid for continuous targets
        def _inv_sigmoid_scaled(val: float, lo: float, hi: float) -> float:
            norm = np.clip((val - lo) / max(hi - lo, 1e-9), 1e-6, 1.0 - 1e-6)
            return float(np.log(norm / (1.0 - norm)))

        target[0] = _inv_sigmoid_scaled(action.w, *W_RANGE)
        target[1] = _inv_sigmoid_scaled(action.c1, *C1_RANGE)
        target[2] = _inv_sigmoid_scaled(action.c2, *C2_RANGE)
        target[3] = _inv_sigmoid_scaled(action.repulsion_weight, *REP_WT_RANGE)

        # Operator target: reward for selected operator, 0 for others
        target[N_CONTINUOUS + action.operator] = float(reward)

        # Mask: only apply loss to continuous heads and the selected operator
        mask = np.zeros(N_OUTPUTS, dtype=np.float32)
        mask[0:N_CONTINUOUS] = 1.0
        mask[N_CONTINUOUS + action.operator] = 1.0

        # Store in replay
        idx = self._ptr % self.replay_capacity
        self._feat[idx] = np.asarray(features, dtype=np.float32).reshape(-1)
        self._targets[idx] = target
        self._masks[idx] = mask
        self._ptr += 1
        self._size = min(self._size + 1, self.replay_capacity)

        # Train
        if self._size >= self.min_train_size:
            self._train_step()

    def _train_step(self) -> None:
        torch = self._torch
        indices = self._rng.integers(0, self._size, size=min(self.batch_size, self._size))
        x_batch = torch.tensor(self._feat[indices], dtype=torch.float32, device=self.device)
        y_batch = torch.tensor(self._targets[indices], dtype=torch.float32, device=self.device)
        m_batch = torch.tensor(self._masks[indices], dtype=torch.float32, device=self.device)

        for _ in range(self.train_steps):
            pred = self.model(x_batch)
            loss_components = self.loss_fn(pred, y_batch) * m_batch
            loss = loss_components.sum() / max(1.0, float(m_batch.sum()))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._loss_ema = 0.95 * self._loss_ema + 0.05 * float(loss.item())

    def save(self, path: str | Path) -> bool:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch = self._torch
        payload: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "loss_ema": self._loss_ema,
            "attentionMode": self.attention_mode,
            "attentionKeyDim": self.attention_key_dim,
            "attentionInputDim": self._attention_input_dim,
            "attentionLossEma": self._attention_loss_ema,
        }
        if self._attention_q is not None and self._attention_k is not None:
            payload["attentionQ"] = self._attention_q.state_dict()
            payload["attentionK"] = self._attention_k.state_dict()
            if self._attention_optimizer is not None:
                payload["attentionOptimizer"] = self._attention_optimizer.state_dict()
        torch.save(payload, str(path))
        return True

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            torch = self._torch
            state = torch.load(str(path), map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self._step = int(state.get("step", 0))
            self._loss_ema = float(state.get("loss_ema", 0.0))
            mode = str(state.get("attentionMode", self.attention_mode)).strip().lower()
            if mode in {ATTENTION_MODE_COSINE, ATTENTION_MODE_LEARNED}:
                self.attention_mode = mode
            self.attention_key_dim = int(max(4, int(state.get("attentionKeyDim", self.attention_key_dim))))
            self._attention_loss_ema = float(state.get("attentionLossEma", 0.0))
            attention_input_dim = int(state.get("attentionInputDim", 0))
            if (
                self.attention_mode == ATTENTION_MODE_LEARNED
                and attention_input_dim > 0
                and "attentionQ" in state
                and "attentionK" in state
            ):
                self._ensure_attention_modules(attention_input_dim)
                if self._attention_q is not None and self._attention_k is not None:
                    self._attention_q.load_state_dict(state["attentionQ"])
                    self._attention_k.load_state_dict(state["attentionK"])
                    if self._attention_optimizer is not None and "attentionOptimizer" in state:
                        try:
                            self._attention_optimizer.load_state_dict(state["attentionOptimizer"])
                        except Exception:
                            pass
            return True
        except Exception:
            return False

    def compute_attention_weights(
        self,
        particle_features: np.ndarray,
        archive_features: np.ndarray,
        temperature: float = 0.35,
    ) -> np.ndarray:
        """Compute attention weights from particle/archive feature matrices."""
        if self.attention_mode != ATTENTION_MODE_LEARNED:
            return _cosine_attention_weights(
                particle_features=particle_features,
                archive_features=archive_features,
                temperature=temperature,
            )

        p = np.asarray(particle_features, dtype=float)
        a = np.asarray(archive_features, dtype=float)
        if p.ndim != 2 or a.ndim != 2 or p.shape[0] == 0 or a.shape[0] == 0 or p.shape[1] != a.shape[1]:
            return _cosine_attention_weights(
                particle_features=particle_features,
                archive_features=archive_features,
                temperature=temperature,
            )
        if not self._ensure_attention_modules(int(p.shape[1])):
            return _cosine_attention_weights(
                particle_features=particle_features,
                archive_features=archive_features,
                temperature=temperature,
            )

        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        with self._torch.no_grad():
            p_t = self._torch.tensor(p, dtype=self._torch.float32, device=self.device)
            a_t = self._torch.tensor(a, dtype=self._torch.float32, device=self.device)
            weights_t = self._attention_forward_torch(
                particle_features=p_t,
                archive_features=a_t,
                temperature=temperature,
            )
            weights = weights_t.cpu().numpy().astype(float, copy=False)
        return _softmax_rows(weights, temperature=1.0)

    def update_attention(
        self,
        particle_features: np.ndarray,
        archive_features: np.ndarray,
        reward: float,
        temperature: float = 0.35,
    ) -> None:
        """Auxiliary learned-attention update using cosine pseudo-targets."""
        if self._frozen or self.attention_mode != ATTENTION_MODE_LEARNED:
            return
        p = np.asarray(particle_features, dtype=float)
        a = np.asarray(archive_features, dtype=float)
        if p.ndim != 2 or a.ndim != 2 or p.shape[0] == 0 or a.shape[0] == 0 or p.shape[1] != a.shape[1]:
            return
        if not self._ensure_attention_modules(int(p.shape[1])):
            return

        teacher = _cosine_attention_weights(p, a, temperature=temperature)
        if teacher.shape != (p.shape[0], a.shape[0]):
            return

        importance = float(np.clip(0.25 + 0.75 * ((float(reward) + 1.0) * 0.5), 0.05, 1.25))
        sample = (
            np.asarray(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32),
            np.asarray(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32),
            np.asarray(np.nan_to_num(teacher, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32),
            importance,
        )
        self._attention_replay.append(sample)
        if len(self._attention_replay) > self.attention_replay_capacity:
            overflow = len(self._attention_replay) - self.attention_replay_capacity
            if overflow > 0:
                del self._attention_replay[:overflow]

        self._train_attention_step(temperature=temperature)

    def summary(self) -> dict[str, Any]:
        return {
            "type": "UnifiedController",
            "device": str(self.device),
            "step": self._step,
            "frozen": self._frozen,
            "loss_ema": self._loss_ema,
            "replay_size": self._size,
            "attention_mode": self.attention_mode,
            "attention_loss_ema": self._attention_loss_ema,
            "attention_replay_size": len(self._attention_replay),
        }


# ── Fallback Controller (CPU, no PyTorch) ──────────────────────────

class FallbackController:
    """Lightweight CPU controller for environments without PyTorch.

    Uses running-mean Q-estimates per operator and simple linear regression
    for continuous parameters.
    """

    def __init__(
        self,
        warmup_steps: int = 20,
        alpha: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.warmup_steps = int(max(0, warmup_steps))
        self.alpha = float(max(0.01, alpha))  # learning rate for running mean
        self._step = 0
        self._rng = np.random.default_rng(seed)
        self._frozen = False

        # Q-estimates per operator
        self._q = np.zeros(N_OPERATORS, dtype=float)
        self._counts = np.zeros(N_OPERATORS, dtype=int)

        # Linear model for continuous params:
        # theta @ features -> (w, c1, c2, repulsion_weight)
        self._theta = np.zeros((N_CONTINUOUS, N_FEATURES), dtype=float)
        # Initialize with sensible defaults
        self._default_w = 0.75
        self._default_c1 = 1.5
        self._default_c2 = 1.5
        self._default_repulsion_weight = 2.5
        
        # Central baseline for advantage weighting
        self._v = 0.0

    @property
    def device_tag(self) -> str:
        return "fallback:cpu"

    @property
    def loss_ema(self) -> float:
        return 0.0

    @property
    def frozen(self) -> bool:
        return self._frozen

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = bool(frozen)

    def select_action(self, features: np.ndarray) -> ContinuousAction:
        x = np.asarray(features, dtype=float).reshape(-1)
        if x.size != N_FEATURES:
            raise ValueError(f"Feature size mismatch: got {x.size}, expected {N_FEATURES}")

        self._step += 1

        # Warmup: random
        if (not self._frozen) and self._step <= self.warmup_steps:
            w = float(self._rng.uniform(*W_RANGE))
            c1 = float(self._rng.uniform(*C1_RANGE))
            c2 = float(self._rng.uniform(*C2_RANGE))
            rep = float(self._rng.uniform(*REP_WT_RANGE))
            op = int(self._rng.integers(0, N_OPERATORS))
            return ContinuousAction(
                w=w,
                c1=c1,
                c2=c2,
                repulsion_weight=rep,
                operator=op,
                action_idx=-1,
            )

        raw = self._theta @ x
        w = float(np.clip(self._default_w + 0.1 * np.tanh(raw[0]), *W_RANGE))
        c1 = float(np.clip(self._default_c1 + 0.3 * np.tanh(raw[1]), *C1_RANGE))
        c2 = float(np.clip(self._default_c2 + 0.3 * np.tanh(raw[2]), *C2_RANGE))
        rep = float(
            np.clip(
                self._default_repulsion_weight + 1.0 * np.tanh(raw[3]),
                *REP_WT_RANGE,
            )
        )
        
        # Exploration noise disabled: Adding N(0, eps) to sensitive hyperparameters
        # like inertia (w) causes catastrophic convergence failure on City maps.
        eps = 0.0
        if eps > 0.01:
            w = float(np.clip(w + self._rng.normal(0, eps * 0.1), *W_RANGE))
            c1 = float(np.clip(c1 + self._rng.normal(0, eps * 0.3), *C1_RANGE))
            c2 = float(np.clip(c2 + self._rng.normal(0, eps * 0.3), *C2_RANGE))
            rep = float(np.clip(rep + self._rng.normal(0, eps * 1.0), *REP_WT_RANGE))

        # Operator: UCB-style selection
        untried = np.where(self._counts == 0)[0]
        if untried.size > 0 and not self._frozen:
            op = int(self._rng.choice(untried))
        else:
            ucb = self._q + 0.5 * np.sqrt(
                2.0 * np.log(max(1, self._step)) / np.maximum(self._counts, 1)
            )
            op = int(np.argmax(ucb))

        return ContinuousAction(
            w=w,
            c1=c1,
            c2=c2,
            repulsion_weight=rep,
            operator=op,
            action_idx=op,
        )

    def update(self, features: np.ndarray, reward: float, action: ContinuousAction) -> None:
        if self._frozen:
            return

        # Update operator Q-estimate
        op = action.operator
        self._counts[op] += 1
        self._q[op] += self.alpha * (reward - self._q[op])

        # Track mean reward as baseline V
        self._v += self.alpha * (reward - self._v)
        advantage = np.clip(reward - self._v, -1.0, 1.0)

        # Update linear model with gradient step
        x = np.asarray(features, dtype=float).reshape(-1)
        pred = self._theta @ x

        # Target: inverse-tanh of the deviation from default
        target = np.array([
            np.arctanh(np.clip((action.w - self._default_w) / 0.1, -0.99, 0.99)),
            np.arctanh(np.clip((action.c1 - self._default_c1) / 0.3, -0.99, 0.99)),
            np.arctanh(np.clip((action.c2 - self._default_c2) / 0.3, -0.99, 0.99)),
            np.arctanh(
                np.clip(
                    (action.repulsion_weight - self._default_repulsion_weight) / 1.0,
                    -0.99,
                    0.99,
                )
            ),
        ], dtype=float)
        error = target - pred
        
        # Advantage-Weighted Regression: Update towards actions that yielded 
        # better-than-average rewards. Ignore negative advantages to prevent
        # unstable divergence away from the baseline.
        lr = 0.005 * self.alpha
        if advantage > 0.0:
            self._theta += lr * advantage * np.outer(error, x)
            # Bound theta to prevent exploding gradients/NaNs
            self._theta = np.clip(self._theta, -10.0, 10.0)

    def save(self, path: str | Path) -> bool:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path),
                 theta=self._theta,
                 q=self._q,
                 counts=self._counts,
                 step=np.array([self._step]))
        return True

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            data = np.load(str(path))
            self._theta = data["theta"]
            self._q = data["q"]
            self._counts = data["counts"]
            self._step = int(data["step"][0])
            return True
        except Exception:
            return False

    def compute_attention_weights(
        self,
        particle_features: np.ndarray,
        archive_features: np.ndarray,
        temperature: float = 0.35,
    ) -> np.ndarray:
        """CPU fallback attention via cosine-similarity softmax."""
        return _cosine_attention_weights(
            particle_features=particle_features,
            archive_features=archive_features,
            temperature=temperature,
        )

    def update_attention(
        self,
        particle_features: np.ndarray,
        archive_features: np.ndarray,
        reward: float,
        temperature: float = 0.35,
    ) -> None:
        del particle_features, archive_features, reward, temperature

    def summary(self) -> dict[str, Any]:
        return {
            "type": "FallbackController",
            "device": "cpu",
            "step": self._step,
            "frozen": self._frozen,
            "q_estimates": self._q.tolist(),
            "attention_mode": ATTENTION_MODE_COSINE,
        }
