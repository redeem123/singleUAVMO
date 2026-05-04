from __future__ import annotations

from typing import Any

from uav_benchmark.algorithms.sac_smopso.torch_support import torch


def _migrate_action_dim_in_place(
    payload: dict[str, Any],
    *,
    old_dim: int,
    new_dim: int,
    operator_count: int,
) -> None:
    """Migrate a legacy SAC checkpoint saved with a smaller action dim.

    Pads the actor output heads (``mean_head`` and ``log_std_head``) with
    zero rows so the new action dimensions map to tanh(0)=0 (mid-range),
    and pads the first critic layer's input with zero columns inserted at
    the end of the original action slice. Also resets the replay buffer and
    strips optimizer states (which would reference the wrong shapes). The
    modification is made directly on ``payload`` so the caller can proceed
    with ``load_state_dict``.
    """
    if torch is None:
        raise RuntimeError("torch is required for checkpoint migration")
    assert new_dim > old_dim > 0
    extra = new_dim - old_dim

    def _pad_rows(weight: Any, rows: int) -> Any:
        zeros = torch.zeros((rows, weight.shape[1]), dtype=weight.dtype, device=weight.device)
        return torch.cat([weight, zeros], dim=0)

    def _pad_bias(bias: Any, rows: int) -> Any:
        zeros = torch.zeros((rows,), dtype=bias.dtype, device=bias.device)
        return torch.cat([bias, zeros], dim=0)

    actor_state = payload.get("actor", {})
    for head in ("mean_head", "log_std_head"):
        w_key = f"{head}.weight"
        b_key = f"{head}.bias"
        if w_key in actor_state and actor_state[w_key].shape[0] == old_dim:
            actor_state[w_key] = _pad_rows(actor_state[w_key], extra)
        if b_key in actor_state and actor_state[b_key].shape[0] == old_dim:
            actor_state[b_key] = _pad_bias(actor_state[b_key], extra)

    def _expand_critic_input(state: dict[str, Any]) -> None:
        key = "net.0.weight"
        if key not in state:
            return
        w = state[key]
        total_in = w.shape[1]
        expected_old_in = total_in
        hidden_dim = expected_old_in - old_dim - operator_count
        if hidden_dim <= 0:
            return
        # Insert `extra` zero columns at position (hidden_dim + old_dim).
        zeros = torch.zeros((w.shape[0], extra), dtype=w.dtype, device=w.device)
        state[key] = torch.cat(
            [w[:, : hidden_dim + old_dim], zeros, w[:, hidden_dim + old_dim :]],
            dim=1,
        )

    for critic_key in ("q1", "q2", "targetQ1", "targetQ2"):
        _expand_critic_input(payload.get(critic_key, {}))

    # Replay samples cached actions with `old_dim`; any future sample would
    # mismatch the new Q/actor shapes. Drop the replay buffer and force the
    # caller to rebuild optimizer states (their moment tensors reference the
    # old parameter shapes).
    payload["replayData"] = None
    for opt_key in ("actorOptim", "q1Optim", "q2Optim", "alphaOptim"):
        payload.pop(opt_key, None)
    payload["actionDim"] = new_dim


def _infer_checkpoint_operator_count(payload: dict[str, Any]) -> int:
    count = payload.get("operatorCount")
    if count is not None:
        try:
            return max(0, int(count))
        except (TypeError, ValueError):
            pass
    operator_names = payload.get("operatorNames", ())
    if isinstance(operator_names, (list, tuple)) and operator_names:
        return len(operator_names)
    actor_state = payload.get("actor", {})
    for key in ("logit_head.bias", "logit_head.weight"):
        value = actor_state.get(key)
        if value is not None:
            return int(value.shape[0])
    return 0


def _migrate_operator_count_in_place(
    payload: dict[str, Any],
    *,
    old_count: int,
    new_count: int,
    action_dim: int,
) -> None:
    """Migrate a checkpoint's unused discrete-operator branch to a new width.

    This is only safe when the live controller has ``use_operator_head=False``:
    the actor logits and critic operator columns are not consumed at runtime, so
    we can pad or trim them without changing the continuous policy.
    """
    if torch is None:
        raise RuntimeError("torch is required for checkpoint migration")
    old_count = max(0, int(old_count))
    new_count = max(0, int(new_count))
    if old_count == new_count:
        return

    actor_state = payload.get("actor", {})
    for key in ("logit_head.weight", "logit_head.bias"):
        value = actor_state.get(key)
        if value is None or int(value.shape[0]) != old_count:
            continue
        if new_count < old_count:
            actor_state[key] = value[:new_count].clone()
        else:
            pad_shape = (new_count - old_count, *value.shape[1:])
            padding = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            actor_state[key] = torch.cat([value, padding], dim=0)

    def _reshape_critic_input(state: dict[str, Any]) -> None:
        key = "net.0.weight"
        if key not in state:
            return
        weight = state[key]
        total_in = int(weight.shape[1])
        hidden_dim = total_in - int(action_dim) - old_count
        if hidden_dim <= 0:
            return
        cut = hidden_dim + int(action_dim)
        if new_count < old_count:
            state[key] = weight[:, : cut + new_count].clone()
            return
        padding = torch.zeros((weight.shape[0], new_count - old_count), dtype=weight.dtype, device=weight.device)
        state[key] = torch.cat([weight[:, : cut + old_count], padding], dim=1)

    for critic_key in ("q1", "q2", "targetQ1", "targetQ2"):
        _reshape_critic_input(payload.get(critic_key, {}))

    for opt_key in ("actorOptim", "q1Optim", "q2Optim", "alphaOptim"):
        payload.pop(opt_key, None)
    payload["operatorCount"] = new_count
