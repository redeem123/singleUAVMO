from __future__ import annotations

import unittest

from uav_benchmark.algorithms.rl_config import parse_rl_config, normalize_rl_extra, RLConfig


class RLConfigTest(unittest.TestCase):
    def test_non_rl_returns_fallback_defaults(self) -> None:
        cfg = parse_rl_config({}, use_rl=False)
        self.assertEqual(cfg.controller_backend, "fallback")
        self.assertFalse(cfg.use_gpu_policy)
        self.assertAlmostEqual(cfg.aux_eval_budget_factor, 0.25)

    def test_rl_returns_auto_defaults(self) -> None:
        cfg = parse_rl_config({}, use_rl=True)
        self.assertEqual(cfg.controller_backend, "auto")
        self.assertTrue(cfg.use_gpu_policy)
        self.assertTrue(cfg.attention_enabled)
        self.assertGreater(cfg.attention_temperature, 0.0)
        self.assertEqual(cfg.attention_mode, "cosine")
        self.assertGreater(cfg.attention_key_dim, 0)
        self.assertGreater(cfg.reward_hv_weight, 0.0)
        self.assertGreaterEqual(cfg.aux_eval_budget_start_factor, cfg.aux_eval_budget_end_factor)
        self.assertGreaterEqual(cfg.operator_trigger_prob_start, cfg.operator_trigger_prob_end)
        self.assertGreaterEqual(cfg.surrogate_prefilter_ratio, 0.0)
        self.assertLessEqual(cfg.surrogate_prefilter_ratio, 1.0)
        self.assertGreater(cfg.surrogate_prefilter_min_candidates, 0)
        self.assertGreater(cfg.surrogate_prefilter_k, 0)

    def test_custom_overrides(self) -> None:
        extra = {
            "rlHiddenDim": 128,
            "rlLr": 0.01,
            "rlPolicyMode": "freeze",
            "rlAttentionEnabled": False,
            "rlAttentionTemperature": 0.2,
            "rlAttentionMode": "learned",
            "rlAttentionKeyDim": 24,
            "rlAttentionLr": 1e-3,
            "rlAuxEvalBudgetStartFactor": 0.5,
            "rlAuxEvalBudgetEndFactor": 0.2,
            "rlOperatorTriggerProbStart": 0.9,
            "rlOperatorTriggerProbEnd": 0.4,
            "rlRewardAuxCostWeight": 0.2,
            "rlSurrogatePrefilterEnabled": True,
            "rlSurrogatePrefilterRatio": 0.6,
            "rlSurrogatePrefilterMinCandidates": 3,
            "rlSurrogatePrefilterK": 5,
        }
        cfg = parse_rl_config(extra, use_rl=True)
        self.assertEqual(cfg.hidden_dim, 128)
        self.assertAlmostEqual(cfg.lr, 0.01)
        self.assertEqual(cfg.policy_mode, "freeze")
        self.assertFalse(cfg.attention_enabled)
        self.assertAlmostEqual(cfg.attention_temperature, 0.2)
        self.assertEqual(cfg.attention_mode, "learned")
        self.assertEqual(cfg.attention_key_dim, 24)
        self.assertAlmostEqual(cfg.attention_lr, 1e-3)
        self.assertAlmostEqual(cfg.aux_eval_budget_start_factor, 0.5)
        self.assertAlmostEqual(cfg.aux_eval_budget_end_factor, 0.2)
        self.assertAlmostEqual(cfg.operator_trigger_prob_start, 0.9)
        self.assertAlmostEqual(cfg.operator_trigger_prob_end, 0.4)
        self.assertAlmostEqual(cfg.reward_aux_cost_weight, 0.2)
        self.assertTrue(cfg.surrogate_prefilter_enabled)
        self.assertAlmostEqual(cfg.surrogate_prefilter_ratio, 0.6)
        self.assertEqual(cfg.surrogate_prefilter_min_candidates, 3)
        self.assertEqual(cfg.surrogate_prefilter_k, 5)

    def test_policy_mode_online_supported(self) -> None:
        cfg = parse_rl_config({"rlPolicyMode": "online"}, use_rl=True)
        self.assertEqual(cfg.policy_mode, "online")

    def test_policy_mode_invalid_falls_back_to_train(self) -> None:
        cfg = parse_rl_config({"rlPolicyMode": "invalid-mode"}, use_rl=True)
        self.assertEqual(cfg.policy_mode, "train")

    def test_normalize_preserves_checkpoint(self) -> None:
        extra = {"rlPolicyCheckpointPath": "/tmp/policy.pt"}
        merged = normalize_rl_extra(extra, use_rl=True)
        self.assertEqual(str(merged.get("rlPolicyCheckpointPath")), "/tmp/policy.pt")

    def test_normalize_sets_defaults(self) -> None:
        merged = normalize_rl_extra({}, use_rl=True)
        self.assertIn("rlControllerBackend", merged)
        self.assertIn("rlPolicyMode", merged)
        self.assertIn("rlAttentionEnabled", merged)
        self.assertIn("rlAttentionTemperature", merged)
        self.assertIn("rlAttentionMode", merged)
        self.assertIn("rlAttentionKeyDim", merged)
        self.assertIn("rlAuxEvalBudgetStartFactor", merged)
        self.assertIn("rlAuxEvalBudgetEndFactor", merged)
        self.assertIn("rlOperatorTriggerProbStart", merged)
        self.assertIn("rlOperatorTriggerProbEnd", merged)
        self.assertIn("rlRewardAuxCostWeight", merged)
        self.assertIn("rlSurrogatePrefilterEnabled", merged)


if __name__ == "__main__":
    unittest.main()
