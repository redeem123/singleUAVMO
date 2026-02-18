"""RL configuration for RL-NMOPSO.

Single flat dataclass replaces the previous 4-class hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import uav_benchmark.algorithms.rl_defaults as RLD

_POLICY_MODES = ("train", "warmstart", "freeze", "online")


def normalize_policy_mode(raw: Any) -> str:
    mode = str(raw).strip().lower()
    if mode in _POLICY_MODES:
        return mode
    return "train"


@dataclass
class RLConfig:
    """Flat RL configuration — all knobs in one place."""
    # Controller
    controller_backend: str = "auto"   # "auto" | "unified" | "fallback"
    use_gpu_policy: bool = True
    warmup_steps: int = 30
    hidden_dim: int = 64
    lr: float = 1e-3
    policy_mode: str = "train"         # "train" | "warmstart" | "freeze" | "online"
    checkpoint_path: str = ""
    attention_enabled: bool = True
    attention_temperature: float = 0.35
    attention_mode: str = "cosine"  # "cosine" | "learned"
    attention_key_dim: int = 16
    attention_lr: float = 5e-4
    attention_batch_size: int = 16
    attention_train_steps: int = 1
    attention_min_train_size: int = 16
    attention_replay_capacity: int = 1024

    # Reward
    reward_hv_weight: float = RLD.DEFAULT_REWARD_HV_WEIGHT
    reward_feasible_weight: float = RLD.DEFAULT_REWARD_FEASIBLE_WEIGHT
    reward_diversity_weight: float = RLD.DEFAULT_REWARD_DIVERSITY_WEIGHT
    reward_hv_scale: float = RLD.DEFAULT_REWARD_HV_SCALE

    # Operators (use fixed configs from rl_defaults)
    aux_eval_budget_factor: float = RLD.AUX_EVAL_BUDGET_FACTOR
    aux_eval_budget_start_factor: float = RLD.AUX_EVAL_BUDGET_START_FACTOR
    aux_eval_budget_end_factor: float = RLD.AUX_EVAL_BUDGET_END_FACTOR
    operator_trigger_prob_start: float = RLD.OPERATOR_TRIGGER_PROB_START
    operator_trigger_prob_end: float = RLD.OPERATOR_TRIGGER_PROB_END
    operator_stagnation_boost: float = RLD.OPERATOR_STAGNATION_BOOST
    operator_stagnation_threshold: int = RLD.OPERATOR_STAGNATION_THRESHOLD
    reward_aux_cost_weight: float = RLD.REWARD_AUX_COST_WEIGHT
    surrogate_prefilter_enabled: bool = RLD.SURROGATE_PREFILTER_ENABLED
    surrogate_prefilter_ratio: float = RLD.SURROGATE_PREFILTER_RATIO
    surrogate_prefilter_min_candidates: int = RLD.SURROGATE_PREFILTER_MIN_CANDIDATES
    surrogate_prefilter_k: int = RLD.SURROGATE_PREFILTER_K


def parse_rl_config(extra: dict[str, Any], use_rl: bool) -> RLConfig:
    """Parse RL config from the params.extra dict."""
    if not use_rl:
        return RLConfig(
            controller_backend="fallback",
            use_gpu_policy=False,
            aux_eval_budget_factor=0.25,
        )

    attention_mode = str(extra.get("rlAttentionMode", "cosine")).strip().lower()
    if attention_mode not in {"cosine", "learned"}:
        attention_mode = "cosine"

    return RLConfig(
        controller_backend=str(extra.get("rlControllerBackend", "auto")).strip().lower(),
        use_gpu_policy=bool(extra.get("rlUseGpuPolicy", True)),
        warmup_steps=int(extra.get("rlWarmupSteps", 30)),
        hidden_dim=int(extra.get("rlHiddenDim", 64)),
        lr=float(extra.get("rlLr", 1e-3)),
        policy_mode=normalize_policy_mode(extra.get("rlPolicyMode", "train")),
        checkpoint_path=str(extra.get("rlPolicyCheckpointPath", "")).strip(),
        attention_enabled=bool(extra.get("rlAttentionEnabled", True)),
        attention_temperature=float(extra.get("rlAttentionTemperature", 0.35)),
        attention_mode=attention_mode,
        attention_key_dim=int(extra.get("rlAttentionKeyDim", 16)),
        attention_lr=float(extra.get("rlAttentionLr", 5e-4)),
        attention_batch_size=int(extra.get("rlAttentionBatchSize", 16)),
        attention_train_steps=int(extra.get("rlAttentionTrainSteps", 1)),
        attention_min_train_size=int(extra.get("rlAttentionMinTrainSize", 16)),
        attention_replay_capacity=int(extra.get("rlAttentionReplayCapacity", 1024)),
        reward_hv_weight=float(extra.get("rlRewardHvWeight", RLD.DEFAULT_REWARD_HV_WEIGHT)),
        reward_feasible_weight=float(extra.get("rlRewardFeasibleWeight", RLD.DEFAULT_REWARD_FEASIBLE_WEIGHT)),
        reward_diversity_weight=float(extra.get("rlRewardDiversityWeight", RLD.DEFAULT_REWARD_DIVERSITY_WEIGHT)),
        reward_hv_scale=float(extra.get("rlRewardHvScale", RLD.DEFAULT_REWARD_HV_SCALE)),
        aux_eval_budget_factor=float(extra.get("rlAuxEvalBudgetFactor", RLD.AUX_EVAL_BUDGET_FACTOR)),
        aux_eval_budget_start_factor=float(extra.get("rlAuxEvalBudgetStartFactor", RLD.AUX_EVAL_BUDGET_START_FACTOR)),
        aux_eval_budget_end_factor=float(extra.get("rlAuxEvalBudgetEndFactor", RLD.AUX_EVAL_BUDGET_END_FACTOR)),
        operator_trigger_prob_start=float(extra.get("rlOperatorTriggerProbStart", RLD.OPERATOR_TRIGGER_PROB_START)),
        operator_trigger_prob_end=float(extra.get("rlOperatorTriggerProbEnd", RLD.OPERATOR_TRIGGER_PROB_END)),
        operator_stagnation_boost=float(extra.get("rlOperatorStagnationBoost", RLD.OPERATOR_STAGNATION_BOOST)),
        operator_stagnation_threshold=int(extra.get("rlOperatorStagnationThreshold", RLD.OPERATOR_STAGNATION_THRESHOLD)),
        reward_aux_cost_weight=float(extra.get("rlRewardAuxCostWeight", RLD.REWARD_AUX_COST_WEIGHT)),
        surrogate_prefilter_enabled=bool(extra.get("rlSurrogatePrefilterEnabled", RLD.SURROGATE_PREFILTER_ENABLED)),
        surrogate_prefilter_ratio=float(extra.get("rlSurrogatePrefilterRatio", RLD.SURROGATE_PREFILTER_RATIO)),
        surrogate_prefilter_min_candidates=int(extra.get("rlSurrogatePrefilterMinCandidates", RLD.SURROGATE_PREFILTER_MIN_CANDIDATES)),
        surrogate_prefilter_k=int(extra.get("rlSurrogatePrefilterK", RLD.SURROGATE_PREFILTER_K)),
    )


def normalize_rl_extra(extra: dict[str, Any], use_rl: bool) -> dict[str, Any]:
    """Merge RL config defaults into extra dict for backward compat."""
    cfg = parse_rl_config(extra, use_rl)
    merged = dict(extra)
    merged.setdefault("rlControllerBackend", cfg.controller_backend)
    merged.setdefault("rlUseGpuPolicy", cfg.use_gpu_policy)
    merged.setdefault("rlPolicyMode", cfg.policy_mode)
    merged.setdefault("rlAttentionEnabled", cfg.attention_enabled)
    merged.setdefault("rlAttentionTemperature", cfg.attention_temperature)
    merged.setdefault("rlAttentionMode", cfg.attention_mode)
    merged.setdefault("rlAttentionKeyDim", cfg.attention_key_dim)
    merged.setdefault("rlAttentionLr", cfg.attention_lr)
    merged.setdefault("rlAttentionBatchSize", cfg.attention_batch_size)
    merged.setdefault("rlAttentionTrainSteps", cfg.attention_train_steps)
    merged.setdefault("rlAttentionMinTrainSize", cfg.attention_min_train_size)
    merged.setdefault("rlAttentionReplayCapacity", cfg.attention_replay_capacity)
    merged.setdefault("rlAuxEvalBudgetFactor", cfg.aux_eval_budget_factor)
    merged.setdefault("rlAuxEvalBudgetStartFactor", cfg.aux_eval_budget_start_factor)
    merged.setdefault("rlAuxEvalBudgetEndFactor", cfg.aux_eval_budget_end_factor)
    merged.setdefault("rlOperatorTriggerProbStart", cfg.operator_trigger_prob_start)
    merged.setdefault("rlOperatorTriggerProbEnd", cfg.operator_trigger_prob_end)
    merged.setdefault("rlOperatorStagnationBoost", cfg.operator_stagnation_boost)
    merged.setdefault("rlOperatorStagnationThreshold", cfg.operator_stagnation_threshold)
    merged.setdefault("rlRewardAuxCostWeight", cfg.reward_aux_cost_weight)
    merged.setdefault("rlSurrogatePrefilterEnabled", cfg.surrogate_prefilter_enabled)
    merged.setdefault("rlSurrogatePrefilterRatio", cfg.surrogate_prefilter_ratio)
    merged.setdefault("rlSurrogatePrefilterMinCandidates", cfg.surrogate_prefilter_min_candidates)
    merged.setdefault("rlSurrogatePrefilterK", cfg.surrogate_prefilter_k)
    return merged
