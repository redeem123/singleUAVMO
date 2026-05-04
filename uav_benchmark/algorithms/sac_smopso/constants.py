from __future__ import annotations

import numpy as np

# SAC-SMOPSO keeps the legacy operator-name tuple in checkpoints/metadata for
# compatibility, but the live policy acts through the continuous schedule only.
# The runtime controller disables the discrete operator head so the policy does
# not spend capacity on a branch that this runner does not consume.
_OPERATOR_NAMES = ("base", "sbx", "de", "elite", "spread")

# 12-d continuous action. Ten classic PSO / archive knobs plus
# ``repair_intensity`` (scales the in-SBX conflict repair pass) and
# ``sbx_weight`` (selects between a PSO velocity step and a full
# SBX+unconstrained-reservoir recombination step, per generation).
_ACTION_KEYS = (
    "inertia",
    "c1",
    "c2",
    "velocity_scale",
    "kappa_scale",
    "delta_scale",
    "leader_bias",
    "mutation_prob",
    "repulsion_weight",
    "archive_focus",
    "repair_intensity",
    "sbx_weight",
)
_ACTION_LOWER = np.array([0.25, 0.8, 0.8, 0.25, 0.6, 0.6, 0.1, 0.02, 0.0, 0.0, 0.0, 0.0], dtype=float)
_ACTION_UPPER = np.array([0.95, 2.8, 2.8, 1.3, 2.2, 2.0, 2.5, 0.50, 1.0, 1.0, 1.0, 0.9], dtype=float)
_OBJECTIVE_COUNT = 4
_GLOBAL_STATE_DIM = 24
_CANDIDATE_TOKEN_DIM = 14
_TOPOLOGY_TOKEN_DIM = 8
_INTERACTION_TOKEN_DIM = 7
_ENVIRONMENT_TOKEN_DIM = 8
_POPULATION_TOKEN_COUNT = 12
_ARCHIVE_TOKEN_COUNT = 16
_TOPOLOGY_TOKEN_COUNT = 8
_INTERACTION_TOKEN_COUNT = 12
_ENVIRONMENT_TOKEN_COUNT = 16
_TEMPORAL_WINDOW = 6
_POLICY_MODE_ALIASES = {
    "online": "online",
    "scratch": "online",
    "finetune": "finetune",
    "fine_tune": "finetune",
    "adapt": "finetune",
    "frozen": "frozen",
    "eval": "frozen",
    "evaluation": "frozen",
    "inference": "frozen",
}
_STATE_REPRESENTATION_ALIASES = {
    "flat": ("flat", "flat"),
    "trfts-hand": ("TRFTS-HAND", "handcrafted"),
    "trfts_hand": ("TRFTS-HAND", "handcrafted"),
    "handcrafted": ("TRFTS-HAND", "handcrafted"),
    "trfts": ("TRFTS", "learned"),
    "learned": ("TRFTS", "learned"),
    "trfts-cp": ("TRFTS-CP", "learned"),
    "trfts_cp": ("TRFTS-CP", "learned"),
    "constraint-pressure": ("TRFTS-CP", "learned"),
    "constraint_pressure": ("TRFTS-CP", "learned"),
}
