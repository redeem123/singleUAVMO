"""Default hyperparameters for RL-NMOPSO.

Centralized constants — reduced from ~50 to ~18 by eliminating
S-terrain overrides, per-operator scheduling, and phase gating.
"""
from __future__ import annotations

# ── PSO core ────────────────────────────────────────────────────────
DEFAULT_INERTIA = 1.0              # w (NMOPSO family initial weight)
DEFAULT_INERTIA_MOPSO = 0.7        # w (generic MOPSO initial weight)
DEFAULT_INERTIA_DAMP = 0.98        # wdamp (paper NMOPSO)
DEFAULT_INERTIA_DAMP_NMOPSO = 0.995  # wdamp (non-paper NMOPSO)
DEFAULT_INERTIA_REF = 0.40
C1_REF = 1.496
C2_REF = 1.496
REPULSION_REF = 2.5
DEFAULT_INERTIA_MIN = 0.40         # w_min (NMOPSO)
DEFAULT_INERTIA_MIN_MOPSO = 0.30   # w_min (generic)
DEFAULT_C1 = 1.5                   # cognitive coefficient
DEFAULT_C2 = 1.5                   # social coefficient
DEFAULT_MUTATION_PROB = 0.15       # mutation probability (NMOPSO)
DEFAULT_MUTATION_PROB_MOPSO = 0.1  # mutation probability (generic)
DEFAULT_VELOCITY_CLAMP_RATIO = 0.22  # v_max / span (NMOPSO)
DEFAULT_VELOCITY_CLAMP_RATIO_MOPSO = 0.30  # v_max / span (generic)
DEFAULT_GRID_CELLS = 7             # paper mode grid divisions
DEFAULT_GRID_CELLS_GENERIC = 8     # non-paper grid divisions
DEFAULT_GRID_KAPPA = 2.0           # κ for leader selection
DEFAULT_LEADER_BIAS = 0.5          # leader selection bias

# ── Archive ─────────────────────────────────────────────────────────
DEFAULT_R2_DIVISIONS = 15          # weight vector simplex divisions
DEFAULT_METRIC_INTERVAL = 20       # HV computation every N generations

# ── RL reward (simplified) ──────────────────────────────────────────
DEFAULT_REWARD_HV_WEIGHT = 0.70    # weight of ΔHV in reward
DEFAULT_REWARD_FEASIBLE_WEIGHT = 0.20  # weight of Δfeasibility
DEFAULT_REWARD_DIVERSITY_WEIGHT = 0.10  # weight of Δdiversity
DEFAULT_REWARD_HV_SCALE = 0.01     # HV delta scaling for tanh

# ── Operator fixed configs ──────────────────────────────────────────
SBX_INJECT_RATIO = 0.70           # fraction of pop used as SBX parents
SBX_REPLACE_RATIO = 0.45          # fraction of pop replaced by SBX offspring
DE_INJECT_RATIO = 0.75            # fraction of pop for DE donors
DE_REPLACE_RATIO = 0.45           # fraction of pop replaced by DE offspring
DE_F_SCALE = 0.65                 # DE scale factor (fixed, no scheduling)
DE_CR_RATE = 0.80                 # DE crossover rate (fixed)
DE_PBEST_RATIO = 0.25             # p-best fraction for DE
ELITE_REFINE_TOP_K = 6            # top-K archive members to perturb
ELITE_REFINE_ITERS = 3            # perturbation iterations per member
ELITE_SIGMA = 0.05                # perturbation σ (fixed)
# Runtime control defaults (for expensive objective evaluations)
AUX_EVAL_BUDGET_FACTOR = 0.25      # legacy flat per-gen aux eval cap = factor * pop
AUX_EVAL_BUDGET_START_FACTOR = 0.30  # scheduled start budget factor
AUX_EVAL_BUDGET_END_FACTOR = 0.10    # scheduled end budget factor
OPERATOR_TRIGGER_PROB_START = 0.85   # probability to execute non-noop operator (early)
OPERATOR_TRIGGER_PROB_END = 0.30     # probability to execute non-noop operator (late)
OPERATOR_STAGNATION_BOOST = 0.20     # temporary trigger boost under stagnation
OPERATOR_STAGNATION_THRESHOLD = 5    # generations without HV improvement before boost
REWARD_AUX_COST_WEIGHT = 0.10        # reward penalty weight for aux-eval usage

# Surrogate prefilter defaults (KNN proxy on decision space)
SURROGATE_PREFILTER_ENABLED = True
SURROGATE_PREFILTER_RATIO = 0.50      # evaluate only this fraction of proposed aux candidates
SURROGATE_PREFILTER_MIN_CANDIDATES = 4
SURROGATE_PREFILTER_K = 8

# ── Mutation noise ──────────────────────────────────────────────────
MUTATION_SIGMA_HIGH = 0.12         # max Gaussian mutation scale (NMOPSO)
MUTATION_SIGMA_LOW = 0.02          # min Gaussian mutation scale (NMOPSO)
MUTATION_SIGMA_GENERIC = 0.05      # Gaussian mutation scale (generic)
