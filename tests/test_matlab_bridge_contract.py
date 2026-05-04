from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATLAB_BRIDGE = ROOT / "uav_benchmark" / "platemo_bridge" / "matlab"


def test_shared_matlab_evaluation_helper_exists() -> None:
    helper = MATLAB_BRIDGE / "BridgeEvaluateInPython.m"
    text = helper.read_text(encoding="utf-8")
    assert "uav_benchmark.platemo_bridge.evaluator" in text
    assert "PopObj" in text
    assert "PopCon" in text


def test_path_population_batch_helper_exists() -> None:
    helper = MATLAB_BRIDGE / "BridgeEvaluatePathPopulation.m"
    text = helper.read_text(encoding="utf-8")
    assert "PathStack" in text
    assert "BridgeEvaluateInPython" in text
    assert ".objs" in text
    assert ".cons" in text


def test_matlab_problem_adapters_use_shared_helper() -> None:
    for relative in ("PythonBridgeProblem.m", "PythonBridgeLegacyProblem.m"):
        text = (MATLAB_BRIDGE / relative).read_text(encoding="utf-8")
        assert "BridgeEvaluateInPython" in text
        assert "-m uav_benchmark.platemo_bridge.evaluator" not in text


def test_moea2de_shim_uses_shared_helper_and_path_payload() -> None:
    shim = MATLAB_BRIDGE / "moea2de_fair_shims" / "Chromosome.m"
    text = shim.read_text(encoding="utf-8")
    assert "BridgeEvaluatePathPopulation" in text
    assert "-m uav_benchmark.platemo_bridge.evaluator" not in text


def test_moea2de_loop_batches_population_evaluation() -> None:
    shim = MATLAB_BRIDGE / "moea2de_fair_shims" / "MOEA_2DE.m"
    text = shim.read_text(encoding="utf-8")
    assert "BridgeEvaluatePathPopulation(population)" in text
    assert "BridgeEvaluatePathPopulation(testPopulation)" in text
    assert "BridgeEvaluatePathPopulation(child(1:childEvalCount))" in text


def test_moead_awa_astar_shim_uses_shared_helper_and_path_payload() -> None:
    shim = MATLAB_BRIDGE / "moead_awa_astar_fair_shims" / "Chromosome.m"
    text = shim.read_text(encoding="utf-8")
    assert "BridgeEvaluatePathPopulation" in text
    assert "J1" not in text
    assert "J2" not in text
    assert "-m uav_benchmark.platemo_bridge.evaluator" not in text


def test_moead_awa_astar_loop_batches_population_evaluation() -> None:
    shim = MATLAB_BRIDGE / "moead_awa_astar_fair_shims" / "MOEADAWA_Astar_fair.m"
    text = shim.read_text(encoding="utf-8")
    assert "BridgeEvaluatePathPopulation(population)" in text
    assert "F_operator" in text
    assert "aStar" not in text or "A*" in text


def test_emmop_has_four_objective_state_shim() -> None:
    shim = MATLAB_BRIDGE / "emmop_fair_shims" / "stateCal.m"
    text = shim.read_text(encoding="utf-8")
    assert "state = zeros(10, 1)" in text
    assert "Population.objs" in text
    assert "Four-objective-compatible" in text


def test_emmop_has_small_replay_buffer_training_shim() -> None:
    shim = MATLAB_BRIDGE / "emmop_fair_shims" / "trainDQN.m"
    text = shim.read_text(encoding="utf-8")
    assert "batch_size = min(batch_size, size(ERB, 1))" in text
    assert "forward(train_state, model)" in text
    assert "backprop(model, lr, error, batch_size, train_action)" in text


def test_emmop_algorithm_shim_removes_two_objective_reward_assumption() -> None:
    shim = MATLAB_BRIDGE / "emmop_fair_shims" / "EMMOP.m"
    text = shim.read_text(encoding="utf-8")
    assert "BenchmarkObjectiveReward" in text
    assert "d(1) * d(2)" not in text
    assert "round(0.2 * maxIter / 2)" in text


def test_emmop_main_shim_preserves_bridge_path_order() -> None:
    shim = MATLAB_BRIDGE / "emmop_fair_shims" / "main.m"
    text = shim.read_text(encoding="utf-8")
    assert "keeps" in text
    assert "addpath(genpath(cd))" not in text
    assert "Global.Start()" in text


def test_emmop_has_old_platemo_ga_compatibility_shim() -> None:
    shim = MATLAB_BRIDGE / "emmop_fair_shims" / "GA.m"
    text = shim.read_text(encoding="utf-8")
    assert "function Offspring = GA(Population)" in text
    assert "INDIVIDUAL" in text
    assert "Global.lower" in text
