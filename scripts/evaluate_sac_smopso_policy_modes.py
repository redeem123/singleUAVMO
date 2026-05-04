from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.algorithms.sac_smopso import controller as sac_controller_module
from uav_benchmark.algorithms.sac_smopso import run_sac_smopso
from uav_benchmark.algorithms.sac_smopso.workflow import (
    aggregate_records,
    configure_torch_runtime,
    float_field,
    load_protocol,
    load_rollout_summary,
    mean_field,
    policy_backend_tag,
    prepare_terrain,
    scenario_label,
    string_field,
    validate_multi_uav_fleet_sizes,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_mat
from uav_benchmark.utils.gpu import resolve_gpu
from uav_benchmark.utils.random import seed_everything


def _checkpoint_hidden_dim(checkpoint: Path | None, default: int = 128) -> int:
    if checkpoint is None or not checkpoint.exists():
        return int(default)
    if not bool(getattr(sac_controller_module, "_TORCH_AVAILABLE", False)):
        return int(default)
    torch_module = getattr(sac_controller_module, "torch", None)
    if torch_module is None:
        return int(default)
    try:
        payload = torch_module.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch versions do not support weights_only
        payload = torch_module.load(checkpoint, map_location="cpu")
    except Exception:
        return int(default)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    hidden_dim = config.get("hiddenDim", config.get("hidden_dim"))
    if hidden_dim is None:
        return int(default)
    try:
        return int(hidden_dim)
    except Exception:
        return int(default)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SAC-SMOPSO online, finetune, and frozen policy modes.")
    parser.add_argument("--protocol", type=Path, default=None, help="YAML protocol config path.")
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        help="Base problem files under problems/ used for evaluation.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[71, 83], help="Evaluation scenario seeds.")
    parser.add_argument("--fleet-sizes", nargs="+", type=int, default=[2, 3, 5], help="Fleet sizes to generate.")
    parser.add_argument("--generations", type=int, default=20, help="Generations per evaluation run.")
    parser.add_argument("--population", type=int, default=16, help="Population size per run.")
    parser.add_argument(
        "--separation-min", type=float, default=10.0, help="Minimum separation for generated scenarios."
    )
    parser.add_argument(
        "--gpu-mode", choices=["auto", "off", "force"], default="auto", help="GPU mode for evaluation runs."
    )
    parser.add_argument("--modes", nargs="+", default=["online", "finetune", "frozen"], help="Policy modes to compare.")
    parser.add_argument(
        "--state-representation",
        type=str,
        default="TRFTS",
        choices=["flat", "TRFTS-HAND", "TRFTS", "TRFTS-CP"],
        help="Controller state/encoder variant used during policy comparison.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Pretrained checkpoint path. Required for finetune/frozen comparisons.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "sac_smopso_policy_eval",
        help="Directory for evaluation artifacts and summary JSON.",
    )
    return parser.parse_args(argv)


def _apply_protocol(args: argparse.Namespace) -> argparse.Namespace:
    if args.protocol is None:
        return args
    payload = load_protocol(args.protocol.resolve())
    if "problems" in payload:
        args.problems = [str(value) for value in payload["problems"]]
    if "seeds" in payload:
        args.seeds = [int(value) for value in payload["seeds"]]
    if "fleetSizes" in payload:
        args.fleet_sizes = [int(value) for value in payload["fleetSizes"]]
    if "generations" in payload:
        args.generations = int(payload["generations"])
    if "population" in payload:
        args.population = int(payload["population"])
    if "separationMin" in payload:
        args.separation_min = float(payload["separationMin"])
    if "gpuMode" in payload:
        args.gpu_mode = str(payload["gpuMode"])
    if "policyModes" in payload:
        args.modes = [str(value) for value in payload["policyModes"]]
    if "checkpointPath" in payload:
        args.checkpoint = Path(str(payload["checkpointPath"]))
    if "stateRepresentation" in payload:
        args.state_representation = str(payload["stateRepresentation"])
    return args


def _run_mode(
    terrain: dict[str, Any],
    problem_name: str,
    problem_index: int,
    seed: int,
    mode: str,
    source_checkpoint: Path,
    generations: int,
    population: int,
    results_dir: Path,
    gpu_mode: str,
    state_representation: str,
) -> dict[str, Any]:
    checkpoint_arg = ""
    save_checkpoint = False
    deterministic = mode == "frozen"
    hidden_dim = _checkpoint_hidden_dim(source_checkpoint, default=128)
    if mode == "finetune":
        case_checkpoint = results_dir / "_case_checkpoints" / f"{problem_name}_{mode}.pt"
        case_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, case_checkpoint)
        checkpoint_arg = str(case_checkpoint)
        save_checkpoint = True
    elif mode == "frozen":
        checkpoint_arg = str(source_checkpoint)

    seed_everything(seed)
    params = BenchmarkParams(
        generations=int(generations),
        population=int(population),
        runs=1,
        compute_metrics=True,
        results_dir=results_dir / mode,
        problem_name=problem_name,
        problem_index=int(problem_index),
        mode="fleet",
        fleet_size=int(terrain.get("fleetSize", 1)),
        separation_min=float(terrain.get("separationMin", terrain.get("safeDist", 10.0))),
        max_turn_deg=float(terrain.get("maxTurnDeg", 75.0)),
        gpu_mode=str(gpu_mode),
        seed=int(seed),
        extra={
            "resumeExistingRuns": False,
            "metricInterval": max(1, generations // 4),
            "stateRepresentation": str(state_representation),
            "sacPolicyMode": mode,
            "sacCheckpointPath": checkpoint_arg,
            "sacSaveCheckpoint": save_checkpoint,
            "sacDeterministicPolicy": deterministic,
            "sacHiddenDim": int(hidden_dim),
            "sacWarmupSteps": min(6, max(3, generations)),
            "sacBatchSize": min(16, max(4, population)),
            "sacReplayCapacity": max(256, generations * population * 4),
        },
    )
    scores = run_sac_smopso(terrain, params)
    run_dir = (results_dir / mode / problem_name / "Run_1").resolve()
    metadata = load_mat(run_dir / "rl_metadata.mat")
    mission = load_mat(run_dir / "mission_stats.mat")
    record = {
        "mode": mode,
        "problem": problem_name,
        "seed": int(seed),
        "hypervolume": float(scores[0, 0]) if scores.size > 0 else 0.0,
        "pureDiversity": float(scores[0, 1]) if scores.size > 1 else 0.0,
        "feasibleMean": mean_field(mission, "feasible"),
        "conflictMean": mean_field(mission, "conflictRate"),
        "violationMean": (
            mean_field(mission, "turnViolation")
            + mean_field(mission, "separationViolation")
            + mean_field(mission, "collisionViolation")
        ),
        "policyMode": string_field(metadata.get("policyMode", mode)),
        "checkpointLoaded": float_field(metadata.get("checkpointLoaded", 0.0)),
    }
    record.update(load_rollout_summary(run_dir))
    return record


def _pairwise_summary(records: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    keyed: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (str(record["problem"]), int(record["seed"]))
        keyed.setdefault(key, {})[str(record["mode"])] = record
    metrics = {
        "hypervolume": True,
        "pureDiversity": True,
        "feasibleMean": True,
        "conflictMean": False,
        "violationMean": False,
        "firstFeasibleGeneration": False,
    }
    summary: dict[str, Any] = {}
    for metric, larger_is_better in metrics.items():
        deltas: list[float] = []
        for pair in keyed.values():
            if left not in pair or right not in pair:
                continue
            left_value = float(pair[left].get(metric, 0.0))
            right_value = float(pair[right].get(metric, 0.0))
            delta = left_value - right_value if larger_is_better else right_value - left_value
            deltas.append(delta)
        if not deltas:
            summary[metric] = {"count": 0, "meanDelta": 0.0, "wins": 0, "losses": 0, "ties": 0}
            continue
        wins = sum(delta > 1e-9 for delta in deltas)
        losses = sum(delta < -1e-9 for delta in deltas)
        ties = len(deltas) - wins - losses
        summary[metric] = {
            "count": len(deltas),
            "meanDelta": float(sum(deltas) / len(deltas)),
            "wins": int(wins),
            "losses": int(losses),
            "ties": int(ties),
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    args = _apply_protocol(_parse_args(argv))
    if not bool(getattr(sac_controller_module, "_TORCH_AVAILABLE", False)):
        raise RuntimeError("PyTorch is required to evaluate checkpointed SAC-SMOPSO policy modes.")
    configure_torch_runtime(sac_controller_module)
    gpu_info = resolve_gpu(args.gpu_mode)
    print(
        "runtime_backends "
        f"policy={policy_backend_tag(args.gpu_mode, controller_module=sac_controller_module, heuristic_fallback=True)} "
        f"evaluator={gpu_info.backend}:{gpu_info.device} "
        f"reason={gpu_info.reason}"
    )
    if args.checkpoint is None:
        raise ValueError(
            "A checkpoint is required for SAC policy evaluation. Pass --checkpoint or set checkpointPath in the protocol."
        )
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    args.fleet_sizes = validate_multi_uav_fleet_sizes(args.fleet_sizes)

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    problem_paths = [PROJECT_ROOT / "problems" / name for name in args.problems]

    records: list[dict[str, Any]] = []
    for fleet_size in args.fleet_sizes:
        for problem_index, problem_path in enumerate(problem_paths, start=1):
            for seed in args.seeds:
                terrain = prepare_terrain(
                    problem_path, fleet_size=int(fleet_size), seed=int(seed), separation_min=float(args.separation_min)
                )
                problem_name = scenario_label(problem_path, fleet_size=int(fleet_size), seed=int(seed))
                for mode in args.modes:
                    record = _run_mode(
                        terrain=terrain,
                        problem_name=problem_name,
                        problem_index=problem_index,
                        seed=int(seed),
                        mode=str(mode).strip().lower(),
                        source_checkpoint=checkpoint,
                        generations=int(args.generations),
                        population=int(args.population),
                        results_dir=results_dir,
                        gpu_mode=str(args.gpu_mode),
                        state_representation=str(args.state_representation),
                    )
                    records.append(record)
                    print(
                        f"{record['mode']:8s} {problem_name:32s} "
                        f"HV={record['hypervolume']:.4f} feas={record['feasibleMean']:.4f} "
                        f"conflict={record['conflictMean']:.4f} firstFeas={int(record['firstFeasibleGeneration'])}"
                    )

    mode_summary = {
        mode: aggregate_records(
            [record for record in records if record["mode"] == mode],
            (
                "hypervolume",
                "pureDiversity",
                "feasibleMean",
                "conflictMean",
                "violationMean",
                "firstFeasibleGeneration",
                "meanReward",
            ),
        )
        for mode in args.modes
    }
    pairwise = {}
    if "finetune" in args.modes and "frozen" in args.modes:
        pairwise["finetune_vs_frozen"] = _pairwise_summary(records, "finetune", "frozen")
    if "frozen" in args.modes and "online" in args.modes:
        pairwise["frozen_vs_online"] = _pairwise_summary(records, "frozen", "online")
    if "finetune" in args.modes and "online" in args.modes:
        pairwise["finetune_vs_online"] = _pairwise_summary(records, "finetune", "online")

    summary = {
        "config": {
            "problems": [path.name for path in problem_paths],
            "seeds": [int(seed) for seed in args.seeds],
            "fleetSizes": [int(size) for size in args.fleet_sizes],
            "generations": int(args.generations),
            "population": int(args.population),
            "separationMin": float(args.separation_min),
            "gpuMode": str(args.gpu_mode),
            "checkpoint": str(checkpoint),
            "modes": [str(mode) for mode in args.modes],
            "stateRepresentation": str(args.state_representation),
            "multiUavOnly": True,
            "scenarioSource": "generated_fleet_from_base_terrain",
        },
        "records": records,
        "modeSummary": mode_summary,
        "pairwise": pairwise,
    }
    summary_path = results_dir / "policy_mode_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
