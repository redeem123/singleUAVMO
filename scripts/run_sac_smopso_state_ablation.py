from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.stats import wilcoxon  # type: ignore
except Exception:  # pragma: no cover - optional statistics helper
    wilcoxon = None  # type: ignore[assignment]

from uav_benchmark.algorithms.sac_smopso import controller as sac_controller_module
from uav_benchmark.algorithms.sac_smopso import run_sac_smopso
from uav_benchmark.algorithms.sac_smopso.workflow import (
    float_field,
    load_protocol,
    mean_field,
    policy_backend_tag,
    string_field,
    validate_multi_uav_fleet_sizes,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_mat, load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain
from uav_benchmark.utils.gpu import resolve_gpu
from uav_benchmark.utils.random import seed_everything

_DEFAULT_PAPER_SEEDS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAC-SMOPSO flat, handcrafted-structured, and learned-relational encoder variants."
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=[
            "terrainStruct_c_100.mat",
            "terrainStruct_m_100.mat",
            "terrainStruct_s_120.mat",
        ],
        help="Problem files under problems/ to evaluate.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 53], help="Random seeds.")
    parser.add_argument(
        "--fleet-sizes", nargs="+", type=int, default=None, help="Fleet sizes for generated fleet scenarios."
    )
    parser.add_argument("--generations", type=int, default=12, help="Generations per run.")
    parser.add_argument("--population", type=int, default=10, help="Population size.")
    parser.add_argument("--fleet-size", type=int, default=3, help="Fleet size for generated fleet scenarios.")
    parser.add_argument("--separation-min", type=float, default=10.0, help="Separation minimum for generated fleets.")
    parser.add_argument(
        "--gpu-mode", choices=["auto", "off", "force"], default="off", help="GPU mode for evaluation runs."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["flat", "TRFTS-HAND", "TRFTS"],
        help="State/encoder variants to compare.",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["online", "finetune", "frozen"],
        default="online",
        help="Controller policy mode used for the ablation.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Single checkpoint path shared by all modes.")
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default=None,
        help="Checkpoint template, e.g. results/sac_smopso_pretrain/{mode}/controller.pt.",
    )
    parser.add_argument("--protocol", type=Path, default=None, help="YAML protocol config path.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "sac_smopso_state_ablation",
        help="Output directory for ablation artifacts and summary JSON.",
    )
    return parser.parse_args(argv)


def _resolve_problem_names(raw: list[str] | tuple[str, ...]) -> list[str]:
    resolved: list[str] = []
    for item in raw:
        token = str(item).strip()
        if not token:
            continue
        if token.endswith(".mat"):
            resolved.append(token)
        elif token.startswith("terrainStruct_"):
            resolved.append(f"{token}.mat")
        else:
            resolved.append(f"terrainStruct_{token}.mat")
    return resolved


def _seed_list(run_count: int, base_seed: int) -> list[int]:
    if run_count <= len(_DEFAULT_PAPER_SEEDS):
        return _DEFAULT_PAPER_SEEDS[:run_count]
    return [int(base_seed) + 2 * index for index in range(run_count)]


def _apply_protocol(args: argparse.Namespace) -> argparse.Namespace:
    if args.protocol is None:
        return args
    mapping = load_protocol(args.protocol.resolve())
    params = BenchmarkParams.from_mapping(mapping)
    args.generations = int(params.generations)
    args.population = int(params.population)
    args.separation_min = float(params.separation_min)
    args.gpu_mode = str(params.gpu_mode)
    args.fleet_sizes = [int(size) for size in params.fleet_sizes] if params.fleet_sizes else [int(params.fleet_size)]
    args.fleet_size = int(args.fleet_sizes[0])
    args.policy_mode = str(mapping.get("policyMode", mapping.get("sacPolicyMode", args.policy_mode)))
    raw_modes = mapping.get("stateRepresentationModes", mapping.get("state_representation_modes"))
    if isinstance(raw_modes, (list, tuple)):
        args.modes = [str(item) for item in raw_modes]
    elif raw_modes is None:
        single_mode = mapping.get("stateRepresentation", mapping.get("state_representation"))
        if single_mode:
            args.modes = [str(single_mode)]
    raw_problems = mapping.get("problemNames", mapping.get("problem_names"))
    if isinstance(raw_problems, (list, tuple)):
        args.problems = _resolve_problem_names(tuple(str(item) for item in raw_problems))
    raw_seeds = mapping.get("seeds", mapping.get("Seeds"))
    if isinstance(raw_seeds, (list, tuple)):
        args.seeds = [int(seed) for seed in raw_seeds]
    else:
        args.seeds = _seed_list(int(params.runs), int(params.seed) if params.seed is not None else 11)
    checkpoint_template = mapping.get("checkpointTemplate", mapping.get("checkpoint_template"))
    if checkpoint_template is not None:
        args.checkpoint_template = str(checkpoint_template)
    checkpoint_path = mapping.get("checkpointPath", mapping.get("checkpoint_path"))
    if checkpoint_path is not None:
        args.checkpoint = Path(str(checkpoint_path))
    return args


def _resolve_checkpoint_path(
    state_representation: str,
    policy_mode: str,
    checkpoint: Path | None,
    checkpoint_template: str | None,
) -> Path | None:
    if policy_mode == "online":
        return None
    if checkpoint_template:
        return Path(checkpoint_template.format(mode=state_representation)).expanduser()
    return checkpoint


def _require_existing_checkpoint(
    checkpoint: Path | None,
    *,
    policy_mode: str,
    state_representation: str,
) -> Path | None:
    if policy_mode == "online":
        return None
    if checkpoint is None:
        raise ValueError(f"A checkpoint is required for {policy_mode} ablation runs.")
    resolved = checkpoint.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"SAC-SMOPSO {policy_mode} ablation for {state_representation} requires an existing checkpoint: {resolved}"
        )
    return resolved


def _checkpoint_hidden_dim(checkpoint: Path | None, default: int = 128) -> int:
    if checkpoint is None or not checkpoint.exists():
        return int(default)
    if not bool(getattr(sac_controller_module, "_TORCH_AVAILABLE", False)):
        return int(default)
    torch_module = getattr(sac_controller_module, "torch", None)
    if torch_module is None:
        return int(default)
    try:
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


def _load_completed_case(
    *,
    run_root: Path,
    problem_name: str,
    problem_label: str,
    seed: int,
    fleet_size: int,
    state_representation: str,
    policy_mode: str,
) -> dict[str, Any] | None:
    run_dir = run_root / problem_name / "Run_1"
    final_hv_path = run_root / problem_name / "final_hv.mat"
    mission_path = run_dir / "mission_stats.mat"
    metadata_path = run_dir / "rl_metadata.mat"
    if not (final_hv_path.exists() and mission_path.exists() and metadata_path.exists()):
        return None

    scores_payload = load_mat(final_hv_path)
    scores = np.asarray(scores_payload.get("bestScores", np.zeros((0, 2), dtype=float)), dtype=float)
    mission = load_mat(mission_path)
    metadata = load_mat(metadata_path)
    checkpoint_loaded = float_field(metadata.get("checkpointLoaded", 0.0))
    if policy_mode in {"finetune", "frozen"} and checkpoint_loaded < 0.5:
        raise RuntimeError(
            f"Existing SAC-SMOPSO {policy_mode} ablation artifact for {state_representation}/{problem_name} "
            "did not load a checkpoint."
        )
    return {
        "problem": problem_label,
        "seed": int(seed),
        "fleetSize": int(fleet_size),
        "stateRepresentation": string_field(metadata.get("stateRepresentation", state_representation)),
        "stateEncoderMode": string_field(metadata.get("stateEncoderMode", "")),
        "policyMode": string_field(metadata.get("policyMode", policy_mode)),
        "checkpointLoaded": checkpoint_loaded,
        "hypervolume": float(scores[0, 0]) if scores.size > 0 else 0.0,
        "pureDiversity": float(scores[0, 1]) if scores.size > 1 else 0.0,
        "feasibleMean": mean_field(mission, "feasible"),
        "conflictMean": mean_field(mission, "conflictRate"),
        "violationMean": (
            mean_field(mission, "turnViolation")
            + mean_field(mission, "separationViolation")
            + mean_field(mission, "collisionViolation")
        ),
    }


def _run_case(
    problem_file: Path,
    problem_index: int,
    seed: int,
    state_representation: str,
    generations: int,
    population: int,
    fleet_size: int,
    separation_min: float,
    policy_mode: str,
    checkpoint: Path | None,
    results_dir: Path,
    gpu_mode: str,
) -> dict[str, Any]:
    terrain = load_terrain_struct(problem_file)
    if int(terrain.get("fleetSize", 1)) <= 1:
        terrain = make_fleet_terrain(
            terrain=terrain,
            fleet_size=int(fleet_size),
            seed=1000 + int(seed),
            separation_min=float(separation_min),
            mission_prefix="sac_state_ablation",
        )
        problem_label = f"{problem_file.stem}_uav{int(fleet_size)}"
    else:
        problem_label = problem_file.stem
    problem_name = f"{problem_label}_seed{seed}"
    run_root = results_dir / state_representation
    completed = _load_completed_case(
        run_root=run_root,
        problem_name=problem_name,
        problem_label=problem_label,
        seed=seed,
        fleet_size=fleet_size,
        state_representation=state_representation,
        policy_mode=policy_mode,
    )
    if completed is not None:
        return completed
    checkpoint_arg = ""
    save_checkpoint = False
    deterministic_policy = policy_mode == "frozen"
    resolved_checkpoint = _require_existing_checkpoint(
        checkpoint,
        policy_mode=policy_mode,
        state_representation=state_representation,
    )
    hidden_dim = _checkpoint_hidden_dim(resolved_checkpoint, default=128)
    if policy_mode == "finetune":
        if resolved_checkpoint is None:
            raise ValueError("A checkpoint is required for finetune ablation runs.")
        case_checkpoint = results_dir / "_case_checkpoints" / state_representation / f"{problem_name}.pt"
        case_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved_checkpoint, case_checkpoint)
        checkpoint_arg = str(case_checkpoint)
        save_checkpoint = True
    elif policy_mode == "frozen":
        if resolved_checkpoint is None:
            raise ValueError("A checkpoint is required for frozen ablation runs.")
        checkpoint_arg = str(resolved_checkpoint)

    seed_everything(seed)
    params = BenchmarkParams(
        generations=generations,
        population=population,
        runs=1,
        compute_metrics=True,
        results_dir=run_root,
        problem_name=problem_name,
        problem_index=problem_index,
        mode="fleet",
        fleet_size=int(terrain.get("fleetSize", 1)),
        separation_min=float(terrain.get("separationMin", separation_min)),
        max_turn_deg=float(terrain.get("maxTurnDeg", 75.0)),
        gpu_mode=str(gpu_mode),
        seed=seed,
        extra={
            "metricInterval": max(1, generations // 3),
            "stateRepresentation": state_representation,
            "sacPolicyMode": policy_mode,
            "sacCheckpointPath": checkpoint_arg,
            "sacSaveCheckpoint": save_checkpoint,
            "sacDeterministicPolicy": deterministic_policy,
            "sacHiddenDim": hidden_dim,
            "sacWarmupSteps": min(4, max(2, generations)),
            "sacBatchSize": min(8, max(4, population)),
            "sacReplayCapacity": max(128, generations * population * 2),
        },
    )
    scores = run_sac_smopso(terrain, params)
    run_dir = run_root / problem_name / "Run_1"
    mission = load_mat(run_dir / "mission_stats.mat")
    metadata = load_mat(run_dir / "rl_metadata.mat")
    checkpoint_loaded = float_field(metadata.get("checkpointLoaded", 0.0))
    if policy_mode in {"finetune", "frozen"} and checkpoint_loaded < 0.5:
        raise RuntimeError(
            f"SAC-SMOPSO {policy_mode} ablation for {state_representation} did not load checkpoint "
            f"{resolved_checkpoint}."
        )

    return {
        "problem": problem_label,
        "seed": int(seed),
        "fleetSize": int(fleet_size),
        "stateRepresentation": string_field(metadata.get("stateRepresentation", state_representation)),
        "stateEncoderMode": string_field(metadata.get("stateEncoderMode", "")),
        "policyMode": string_field(metadata.get("policyMode", policy_mode)),
        "checkpointLoaded": checkpoint_loaded,
        "hypervolume": float(scores[0, 0]) if scores.size > 0 else 0.0,
        "pureDiversity": float(scores[0, 1]) if scores.size > 1 else 0.0,
        "feasibleMean": mean_field(mission, "feasible"),
        "conflictMean": mean_field(mission, "conflictRate"),
        "violationMean": (
            mean_field(mission, "turnViolation")
            + mean_field(mission, "separationViolation")
            + mean_field(mission, "collisionViolation")
        ),
    }


def _pairwise_deltas(records: list[dict[str, Any]], left_mode: str, right_mode: str) -> dict[str, list[float]]:
    keyed: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (str(record["problem"]), int(record["seed"]))
        keyed.setdefault(key, {})[str(record["stateRepresentation"])] = record

    deltas = {
        "hypervolume": [],
        "pureDiversity": [],
        "feasibleMean": [],
        "conflictMean": [],
        "violationMean": [],
    }
    for pair in keyed.values():
        if left_mode not in pair or right_mode not in pair:
            continue
        left = pair[left_mode]
        right = pair[right_mode]
        deltas["hypervolume"].append(float(left["hypervolume"]) - float(right["hypervolume"]))
        deltas["pureDiversity"].append(float(left["pureDiversity"]) - float(right["pureDiversity"]))
        deltas["feasibleMean"].append(float(left["feasibleMean"]) - float(right["feasibleMean"]))
        deltas["conflictMean"].append(float(right["conflictMean"]) - float(left["conflictMean"]))
        deltas["violationMean"].append(float(right["violationMean"]) - float(left["violationMean"]))
    return deltas


def _metric_summary(deltas: list[float]) -> dict[str, Any]:
    vector = np.asarray(deltas, dtype=float).reshape(-1)
    if vector.size == 0:
        return {"count": 0, "meanDelta": 0.0, "medianDelta": 0.0, "wins": 0, "losses": 0, "ties": 0}
    summary = {
        "count": int(vector.size),
        "meanDelta": float(np.mean(vector)),
        "medianDelta": float(np.median(vector)),
        "wins": int(np.sum(vector > 1e-9)),
        "losses": int(np.sum(vector < -1e-9)),
        "ties": int(np.sum(np.abs(vector) <= 1e-9)),
    }
    if wilcoxon is not None and vector.size >= 3 and np.any(np.abs(vector) > 1e-9):
        try:
            result = wilcoxon(vector, alternative="greater", zero_method="wilcox")
            summary["wilcoxonStatistic"] = float(getattr(result, "statistic", float("nan")))
            summary["wilcoxonPValue"] = float(getattr(result, "pvalue", float("nan")))
        except Exception:
            pass
    return summary


def main(argv: list[str] | None = None) -> None:
    args = _apply_protocol(_parse_args(argv))
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    gpu_info = resolve_gpu(args.gpu_mode)
    print(
        "runtime_backends "
        f"policy={policy_backend_tag(args.gpu_mode)} "
        f"evaluator={gpu_info.backend}:{gpu_info.device} "
        f"reason={gpu_info.reason}"
    )

    records: list[dict[str, Any]] = []
    problem_paths = [PROJECT_ROOT / "problems" / problem for problem in args.problems]
    fleet_sizes = validate_multi_uav_fleet_sizes(
        [int(size) for size in (args.fleet_sizes if args.fleet_sizes else [args.fleet_size])]
    )
    for fleet_size in fleet_sizes:
        for problem_index, problem_path in enumerate(problem_paths, start=1):
            for seed in args.seeds:
                for state_representation in args.modes:
                    checkpoint = _resolve_checkpoint_path(
                        state_representation=state_representation,
                        policy_mode=str(args.policy_mode),
                        checkpoint=args.checkpoint,
                        checkpoint_template=args.checkpoint_template,
                    )
                    record = _run_case(
                        problem_file=problem_path,
                        problem_index=problem_index,
                        seed=int(seed),
                        state_representation=state_representation,
                        generations=int(args.generations),
                        population=int(args.population),
                        fleet_size=int(fleet_size),
                        separation_min=float(args.separation_min),
                        policy_mode=str(args.policy_mode),
                        checkpoint=checkpoint,
                        results_dir=results_dir,
                        gpu_mode=str(args.gpu_mode),
                    )
                    records.append(record)
                    print(
                        f"{record['problem']:24s} seed={record['seed']:3d} "
                        f"mode={record['stateRepresentation']:10s} policy={record['policyMode']:8s} "
                        f"HV={record['hypervolume']:.4f} PD={record['pureDiversity']:.4f} "
                        f"feas={record['feasibleMean']:.3f} conflict={record['conflictMean']:.3f}"
                    )

    mode_summary: dict[str, dict[str, float]] = {}
    for mode in args.modes:
        mode_records = [record for record in records if record["stateRepresentation"] == mode]
        if not mode_records:
            continue
        mode_summary[mode] = {
            "hypervolumeMean": float(np.mean([float(record["hypervolume"]) for record in mode_records])),
            "pureDiversityMean": float(np.mean([float(record["pureDiversity"]) for record in mode_records])),
            "feasibleMean": float(np.mean([float(record["feasibleMean"]) for record in mode_records])),
            "conflictMean": float(np.mean([float(record["conflictMean"]) for record in mode_records])),
            "violationMean": float(np.mean([float(record["violationMean"]) for record in mode_records])),
        }
    comparisons = [
        ("TRFTS", "TRFTS-HAND"),
        ("TRFTS-HAND", "flat"),
        ("TRFTS", "flat"),
    ]
    pairwise = {}
    for left_mode, right_mode in comparisons:
        if left_mode in args.modes and right_mode in args.modes:
            deltas = _pairwise_deltas(records, left_mode=left_mode, right_mode=right_mode)
            pairwise[f"{left_mode}_vs_{right_mode}"] = {
                "orientedDeltas": {key: [float(value) for value in values] for key, values in deltas.items()},
                "summary": {key: _metric_summary(values) for key, values in deltas.items()},
            }
    summary = {
        "config": {
            "problems": [path.name for path in problem_paths],
            "seeds": [int(seed) for seed in args.seeds],
            "generations": int(args.generations),
            "population": int(args.population),
            "fleetSizes": fleet_sizes,
            "separationMin": float(args.separation_min),
            "gpuMode": str(args.gpu_mode),
            "modes": [str(mode) for mode in args.modes],
            "policyMode": str(args.policy_mode),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else "",
            "checkpointTemplate": str(args.checkpoint_template or ""),
        },
        "records": records,
        "modeSummary": mode_summary,
        "pairwise": pairwise,
    }

    seed_tag = "-".join(str(int(seed)) for seed in args.seeds)
    sep_tag = str(args.separation_min).replace(".", "p")
    fleet_tag = "-".join(str(size) for size in fleet_sizes)
    tag = f"fleet{fleet_tag}_sep{sep_tag}_g{int(args.generations)}_p{int(args.population)}_seeds{seed_tag}_{str(args.policy_mode)}"
    out_path = results_dir / f"summary_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_path = results_dir / "summary.json"
    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for pair_name, pair_payload in summary["pairwise"].items():
        print(f"\n{pair_name} summary (positive means left mode is better):")
        for metric_name, metric_summary in pair_payload["summary"].items():
            p_value = metric_summary.get("wilcoxonPValue")
            p_text = f", p={p_value:.4f}" if p_value is not None else ""
            print(
                f"- {metric_name}: mean={metric_summary['meanDelta']:.6f}, "
                f"median={metric_summary['medianDelta']:.6f}, "
                f"wins={metric_summary['wins']}, losses={metric_summary['losses']}, ties={metric_summary['ties']}{p_text}"
            )
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
