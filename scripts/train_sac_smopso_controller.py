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
    STAGE_PRESETS,
    aggregate_records,
    configure_torch_runtime,
    float_field,
    load_rollout_summary,
    mean_field,
    policy_backend_tag,
    prepare_terrain,
    scenario_label,
    stage_preset,
    string_field,
    validate_multi_uav_fleet_sizes,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_mat
from uav_benchmark.utils.gpu import resolve_gpu
from uav_benchmark.utils.random import seed_everything


def _controller_extra(
    *,
    generations: int,
    population: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = {
        "sacHiddenDim": 128,
        "sacWarmupSteps": min(6, max(3, int(generations))),
        "sacBatchSize": min(16, max(4, int(population))),
        "sacReplayCapacity": max(256, int(generations) * int(population) * 4),
        "sacScratchPolicyMixStart": 0.35,
        "sacScratchPolicyMixEnd": 0.90,
        "sacLoadedPolicyMixStart": 0.60,
        "sacLoadedPolicyMixEnd": 1.00,
        "sacPolicyMixAnnealSteps": 200,
    }
    if overrides:
        extra.update(dict(overrides))
    return extra


def _summary_selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    # Select paper checkpoints by constrained-planning quality first. HV is
    # still used, but only after feasibility and safety metrics.
    return (
        float(summary.get("feasibleMeanMean", 0.0)),
        -float(summary.get("violationMeanMean", 0.0)),
        -float(summary.get("conflictMeanMean", 0.0)),
        float(summary.get("hypervolumeMean", 0.0)),
        float(summary.get("pureDiversityMean", 0.0)),
        -float(summary.get("firstFeasibleGenerationMean", 0.0)),
    )


_SELECTION_CRITERIA = (
    "feasibleMeanMean",
    "negativeViolationMeanMean",
    "negativeConflictMeanMean",
    "hypervolumeMean",
    "pureDiversityMean",
    "negativeFirstFeasibleGenerationMean",
)


def _update_best_checkpoint(
    *,
    checkpoint: Path,
    stage_results_dir: Path,
    stage_name: str,
    eval_summary: dict[str, Any],
    manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    if not checkpoint.exists():
        return {}
    stage_snapshot = stage_results_dir / "controller.pt"
    if stage_snapshot.resolve() != checkpoint.resolve():
        shutil.copy2(checkpoint, stage_snapshot)
    else:
        stage_snapshot = checkpoint
    best_checkpoint = checkpoint.parent / "best_controller.pt"
    best_manifest_path = checkpoint.parent / "best_checkpoint_manifest.json"
    selection_key = _summary_selection_key(eval_summary)
    selected_as_best = False
    previous_key = None
    if best_manifest_path.exists():
        try:
            previous_payload = json.loads(best_manifest_path.read_text(encoding="utf-8"))
            values = previous_payload.get("selectionKey", [])
            if isinstance(values, list):
                previous_key = tuple(float(value) for value in values)
        except Exception:
            previous_key = None
    if previous_key is None or selection_key > previous_key:
        shutil.copy2(stage_snapshot, best_checkpoint)
        best_manifest = {
            **manifest_payload,
            "checkpoint": str(best_checkpoint),
            "stageSnapshot": str(stage_snapshot),
            "selectedStage": str(stage_name),
            "selectionCriteria": list(_SELECTION_CRITERIA),
            "selectionKey": [float(value) for value in selection_key],
        }
        best_manifest_path.write_text(json.dumps(best_manifest, indent=2), encoding="utf-8")
        selected_as_best = True
    return {
        "stageSnapshot": str(stage_snapshot),
        "bestCheckpoint": str(best_checkpoint),
        "selectedAsBest": bool(selected_as_best),
        "selectionCriteria": list(_SELECTION_CRITERIA),
        "selectionKey": [float(value) for value in selection_key],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain and evaluate the SAC-SMOPSO controller.")
    parser.add_argument(
        "--train-problems",
        nargs="+",
        default=["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        help="Base problem files under problems/ used for training episodes.",
    )
    parser.add_argument(
        "--eval-problems",
        nargs="+",
        default=["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        help="Base problem files under problems/ used for frozen-policy evaluation.",
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[11, 23, 37, 53], help="Training scenario seeds.")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[71, 83], help="Held-out evaluation seeds.")
    parser.add_argument("--fleet-sizes", nargs="+", type=int, default=[2, 3, 5], help="Fleet sizes to generate.")
    parser.add_argument(
        "--stage", choices=sorted(STAGE_PRESETS.keys()), help="Apply a built-in curriculum stage preset."
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of passes over the training scenario list.")
    parser.add_argument("--generations", type=int, default=20, help="Generations per training/evaluation run.")
    parser.add_argument("--population", type=int, default=16, help="Population size per run.")
    parser.add_argument(
        "--separation-min", type=float, default=10.0, help="Minimum fleet separation for generated scenarios."
    )
    parser.add_argument(
        "--gpu-mode", choices=["auto", "off", "force"], default="auto", help="GPU mode for training/evaluation runs."
    )
    parser.add_argument(
        "--state-representation",
        default="TRFTS",
        choices=["flat", "TRFTS-HAND", "TRFTS", "TRFTS-CP"],
        help="Controller state/encoder variant to train and evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "sac_smopso_pretrain" / "controller.pt",
        help="Checkpoint path for the pretrained SAC controller.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "sac_smopso_pretrain",
        help="Directory for training/evaluation artifacts and summaries.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Only train; do not run frozen-policy evaluation.")
    parser.add_argument("--reset", action="store_true", help="Delete any existing checkpoint before training.")
    return parser.parse_args(argv)


def _run_episode(
    terrain: dict[str, Any],
    problem_name: str,
    problem_index: int,
    seed: int,
    checkpoint: Path,
    state_representation: str,
    policy_mode: str,
    generations: int,
    population: int,
    results_dir: Path,
    compute_metrics: bool,
    gpu_mode: str,
    controller_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    controller_extra = _controller_extra(
        generations=int(generations),
        population=int(population),
        overrides=controller_overrides,
    )
    params = BenchmarkParams(
        generations=int(generations),
        population=int(population),
        runs=1,
        compute_metrics=bool(compute_metrics),
        results_dir=results_dir,
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
            "sacPolicyMode": policy_mode,
            "sacCheckpointPath": str(checkpoint),
            "sacSaveCheckpoint": policy_mode != "frozen",
            "sacDeterministicPolicy": policy_mode == "frozen",
            **controller_extra,
        },
    )
    scores = run_sac_smopso(terrain, params)
    run_dir = results_dir / problem_name / "Run_1"
    metadata = load_mat(run_dir / "rl_metadata.mat")
    mission = load_mat(run_dir / "mission_stats.mat")
    run_stats_path = run_dir / "run_stats.mat"
    run_stats = load_mat(run_stats_path) if run_stats_path.exists() else {}
    record = {
        "problem": problem_name,
        "seed": int(seed),
        "policyMode": string_field(metadata.get("policyMode", policy_mode)),
        "checkpointLoaded": float_field(metadata.get("checkpointLoaded", 0.0)),
        "controllerReplaySize": float_field(metadata.get("controllerReplaySize", 0.0)),
        "controllerPolicyMix": float_field(metadata.get("controllerPolicyMix", 0.0)),
        "hypervolume": float(scores[0, 0]) if scores.size > 0 else 0.0,
        "pureDiversity": float(scores[0, 1]) if scores.size > 1 else 0.0,
        "feasibleMean": mean_field(mission, "feasible"),
        "conflictMean": mean_field(mission, "conflictRate"),
        "violationMean": (
            mean_field(mission, "turnViolation")
            + mean_field(mission, "separationViolation")
            + mean_field(mission, "collisionViolation")
        ),
        "runtimeSec": float_field(run_stats.get("runtimeSec", 0.0)),
        "rlControllerTimeSec": float_field(run_stats.get("rlControllerTimeSec", 0.0)),
        "gpuUpdateTimeSec": float_field(run_stats.get("gpuUpdateTimeSec", 0.0)),
    }
    record.update(load_rollout_summary(run_dir))
    return record


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not bool(getattr(sac_controller_module, "_TORCH_AVAILABLE", False)):
        raise RuntimeError("PyTorch is required to pretrain or evaluate a checkpointed SAC-SMOPSO controller.")
    configure_torch_runtime(sac_controller_module)
    gpu_info = resolve_gpu(args.gpu_mode)
    print(
        "runtime_backends "
        f"policy={policy_backend_tag(args.gpu_mode, controller_module=sac_controller_module, heuristic_fallback=True)} "
        f"evaluator={gpu_info.backend}:{gpu_info.device} "
        f"reason={gpu_info.reason}"
    )
    preset = stage_preset(args.stage)
    controller_overrides = dict(preset.get("controllerConfig", {})) if preset else {}
    if preset:
        args.train_problems = list(preset["trainProblems"])
        args.eval_problems = list(preset["evalProblems"])
        args.fleet_sizes = list(preset["fleetSizes"])
        args.train_seeds = list(preset["trainSeeds"])
        args.eval_seeds = list(preset["evalSeeds"])
        args.epochs = int(preset["epochs"])
        args.generations = int(preset["generations"])
        args.population = int(preset["population"])
        args.separation_min = float(preset["separationMin"])
    args.fleet_sizes = validate_multi_uav_fleet_sizes(args.fleet_sizes)
    results_dir = args.results_dir.resolve()
    stage_name = str(args.stage or "custom")
    stage_results_dir = results_dir / "stages" / stage_name if args.stage else results_dir
    train_results_dir = stage_results_dir / "train"
    eval_results_dir = stage_results_dir / "eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    stage_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if args.reset and checkpoint.exists():
        checkpoint.unlink()
    if args.reset:
        for stale_path in (
            checkpoint.parent / "best_controller.pt",
            checkpoint.parent / "best_checkpoint_manifest.json",
            results_dir / "latest_pretrain_summary.json",
            results_dir / "latest_checkpoint_manifest.json",
        ):
            if stale_path.exists():
                stale_path.unlink()

    train_problem_paths = [PROJECT_ROOT / "problems" / name for name in args.train_problems]
    eval_problem_paths = [PROJECT_ROOT / "problems" / name for name in args.eval_problems]

    train_records: list[dict[str, Any]] = []
    episode_counter = 0
    for epoch in range(int(args.epochs)):
        for fleet_size in args.fleet_sizes:
            for problem_index, problem_path in enumerate(train_problem_paths, start=1):
                for seed in args.train_seeds:
                    terrain = prepare_terrain(
                        problem_path,
                        fleet_size=int(fleet_size),
                        seed=int(seed),
                        separation_min=float(args.separation_min),
                    )
                    problem_name = scenario_label(problem_path, fleet_size=int(fleet_size), seed=int(seed))
                    record = _run_episode(
                        terrain=terrain,
                        problem_name=problem_name,
                        problem_index=problem_index,
                        seed=int(seed),
                        checkpoint=checkpoint,
                        state_representation=str(args.state_representation),
                        policy_mode="finetune",
                        generations=int(args.generations),
                        population=int(args.population),
                        results_dir=train_results_dir,
                        compute_metrics=False,
                        gpu_mode=str(args.gpu_mode),
                        controller_overrides=controller_overrides,
                    )
                    record["epoch"] = int(epoch + 1)
                    train_records.append(record)
                    episode_counter += 1
                    print(
                        f"train episode={episode_counter:03d} epoch={epoch + 1} "
                        f"{problem_name:32s} conflict={record['conflictMean']:.4f} "
                        f"violation={record['violationMean']:.4f}"
                    )

    eval_records: list[dict[str, Any]] = []
    if not args.skip_eval:
        for fleet_size in args.fleet_sizes:
            for problem_index, problem_path in enumerate(eval_problem_paths, start=1):
                for seed in args.eval_seeds:
                    terrain = prepare_terrain(
                        problem_path,
                        fleet_size=int(fleet_size),
                        seed=int(seed),
                        separation_min=float(args.separation_min),
                    )
                    problem_name = scenario_label(problem_path, fleet_size=int(fleet_size), seed=int(seed))
                    record = _run_episode(
                        terrain=terrain,
                        problem_name=problem_name,
                        problem_index=problem_index,
                        seed=int(seed),
                        checkpoint=checkpoint,
                        state_representation=str(args.state_representation),
                        policy_mode="frozen",
                        generations=int(args.generations),
                        population=int(args.population),
                        results_dir=eval_results_dir,
                        compute_metrics=True,
                        gpu_mode=str(args.gpu_mode),
                        controller_overrides=controller_overrides,
                    )
                    eval_records.append(record)
                    print(
                        f"eval  {problem_name:32s} "
                        f"HV={record['hypervolume']:.4f} PD={record['pureDiversity']:.4f} "
                        f"feas={record['feasibleMean']:.4f} conflict={record['conflictMean']:.4f}"
                    )

    summary = {
        "config": {
            "stage": str(args.stage or ""),
            "trainProblems": [path.name for path in train_problem_paths],
            "evalProblems": [path.name for path in eval_problem_paths],
            "trainSeeds": [int(seed) for seed in args.train_seeds],
            "evalSeeds": [int(seed) for seed in args.eval_seeds],
            "fleetSizes": [int(size) for size in args.fleet_sizes],
            "epochs": int(args.epochs),
            "generations": int(args.generations),
            "population": int(args.population),
            "separationMin": float(args.separation_min),
            "gpuMode": str(args.gpu_mode),
            "stateRepresentation": str(args.state_representation),
            "checkpoint": str(checkpoint),
            "stageResultsDir": str(stage_results_dir),
            "multiUavOnly": True,
            "scenarioSource": "generated_fleet_from_base_terrain",
            "controllerConfig": _controller_extra(
                generations=int(args.generations),
                population=int(args.population),
                overrides=controller_overrides,
            ),
        },
        "trainRecords": train_records,
        "evalRecords": eval_records,
        "trainSummary": aggregate_records(
            train_records,
            (
                "hypervolume",
                "pureDiversity",
                "feasibleMean",
                "conflictMean",
                "violationMean",
                "firstFeasibleGeneration",
                "meanReward",
            ),
        ),
        "evalSummary": aggregate_records(
            eval_records,
            (
                "hypervolume",
                "pureDiversity",
                "feasibleMean",
                "conflictMean",
                "violationMean",
                "firstFeasibleGeneration",
                "meanReward",
            ),
        ),
    }
    summary_path = stage_results_dir / "pretrain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_payload = {
        "checkpoint": str(checkpoint),
        "stage": str(args.stage or ""),
        "stageResultsDir": str(stage_results_dir),
        "multiUavOnly": True,
        "trainSummary": summary["trainSummary"],
        "evalSummary": summary["evalSummary"],
    }
    manifest_payload.update(
        _update_best_checkpoint(
            checkpoint=checkpoint,
            stage_results_dir=stage_results_dir,
            stage_name=stage_name,
            eval_summary=summary["evalSummary"],
            manifest_payload=manifest_payload,
        )
        if eval_records
        else {}
    )
    manifest_path = stage_results_dir / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    (results_dir / "latest_pretrain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "latest_checkpoint_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved checkpoint manifest to {manifest_path}")
    print(f"Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
