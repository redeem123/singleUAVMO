from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAC_PRETRAIN_CHECKPOINT_DEFAULT = "results/sac_smopso_pretrain/controller.pt"
SAC_PRETRAIN_RESULTS_DEFAULT = "results/sac_smopso_pretrain"
SAC_POLICY_RESULTS_DEFAULT = "results/sac_smopso_policy_eval"
SAC_ABLATION_RESULTS_DEFAULT = "results/sac_smopso_state_ablation"
TORCH_REQUIRED_COMMANDS = {"sac-pretrain", "sac-policy-eval", "sac-encoder-ablation"}


def _append_scalar_arg(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        argv.extend([flag, text])


def _append_multi_arg(argv: list[str], flag: str, values: list[object] | tuple[object, ...] | None) -> None:
    if not values:
        return
    argv.append(flag)
    argv.extend(str(value) for value in values)


def _ensure_project_root_on_path() -> None:
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _build_argv(
    args: argparse.Namespace,
    *,
    multi: tuple[tuple[str, str], ...] = (),
    scalar: tuple[tuple[str, str], ...] = (),
    flags: tuple[tuple[str, str], ...] = (),
) -> list[str]:
    argv: list[str] = []
    for flag, attr in multi:
        _append_multi_arg(argv, flag, getattr(args, attr, None))
    for flag, attr in scalar:
        _append_scalar_arg(argv, flag, getattr(args, attr, None))
    for flag, attr in flags:
        if getattr(args, attr, False):
            argv.append(flag)
    return argv


def _build_sac_pretrain_argv(args: argparse.Namespace) -> list[str]:
    return _build_argv(
        args,
        multi=(
            ("--train-problems", "train_problems"),
            ("--eval-problems", "eval_problems"),
            ("--train-seeds", "train_seeds"),
            ("--eval-seeds", "eval_seeds"),
            ("--fleet-sizes", "fleet_sizes"),
        ),
        scalar=(
            ("--stage", "stage"),
            ("--epochs", "epochs"),
            ("--generations", "generations"),
            ("--population", "population"),
            ("--separation-min", "separation_min"),
            ("--gpu-mode", "gpu_mode"),
            ("--state-representation", "state_representation"),
            ("--checkpoint", "checkpoint"),
            ("--results-dir", "results_dir"),
        ),
        flags=(
            ("--skip-eval", "skip_eval"),
            ("--reset", "reset"),
        ),
    )


def _build_sac_policy_eval_argv(args: argparse.Namespace) -> list[str]:
    argv = _build_argv(
        args,
        multi=(
            ("--problems", "problems"),
            ("--seeds", "seeds"),
            ("--fleet-sizes", "fleet_sizes"),
        ),
        scalar=(
            ("--generations", "generations"),
            ("--population", "population"),
            ("--separation-min", "separation_min"),
            ("--gpu-mode", "gpu_mode"),
        ),
    )
    _append_multi_arg(argv, "--modes", args.modes)
    argv.extend(
        _build_argv(
            args,
            scalar=(
                ("--state-representation", "state_representation"),
                ("--checkpoint", "checkpoint"),
                ("--protocol", "protocol"),
                ("--results-dir", "results_dir"),
            ),
        )
    )
    return argv


def _build_sac_encoder_ablation_argv(args: argparse.Namespace) -> list[str]:
    return _build_argv(
        args,
        multi=(
            ("--problems", "problems"),
            ("--seeds", "seeds"),
            ("--fleet-sizes", "fleet_sizes"),
            ("--modes", "modes"),
        ),
        scalar=(
            ("--generations", "generations"),
            ("--population", "population"),
            ("--fleet-size", "fleet_size"),
            ("--separation-min", "separation_min"),
            ("--gpu-mode", "gpu_mode"),
            ("--policy-mode", "policy_mode"),
            ("--checkpoint", "checkpoint"),
            ("--checkpoint-template", "checkpoint_template"),
            ("--protocol", "protocol"),
            ("--results-dir", "results_dir"),
        ),
    )


def handle_sac_pretrain(args: argparse.Namespace) -> None:
    _ensure_project_root_on_path()
    from scripts.train_sac_smopso_controller import main as sac_pretrain_main

    sac_pretrain_main(_build_sac_pretrain_argv(args))


def handle_sac_policy_eval(args: argparse.Namespace) -> None:
    _ensure_project_root_on_path()
    from scripts.evaluate_sac_smopso_policy_modes import main as sac_policy_eval_main

    sac_policy_eval_main(_build_sac_policy_eval_argv(args))


def handle_sac_encoder_ablation(args: argparse.Namespace) -> None:
    _ensure_project_root_on_path()
    from scripts.run_sac_smopso_state_ablation import main as sac_encoder_ablation_main

    sac_encoder_ablation_main(_build_sac_encoder_ablation_argv(args))


def register_sac_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    sac_pretrain_parser = subparsers.add_parser("sac-pretrain")
    sac_pretrain_parser.add_argument(
        "--train-problems",
        nargs="+",
        default=None,
        help="Base problem files under problems/ used for training episodes.",
    )
    sac_pretrain_parser.add_argument(
        "--eval-problems",
        nargs="+",
        default=None,
        help="Base problem files under problems/ used for frozen-policy evaluation.",
    )
    sac_pretrain_parser.add_argument(
        "--train-seeds", nargs="+", type=int, default=None, help="Training scenario seeds."
    )
    sac_pretrain_parser.add_argument(
        "--eval-seeds", nargs="+", type=int, default=None, help="Held-out evaluation seeds."
    )
    sac_pretrain_parser.add_argument(
        "--fleet-sizes", nargs="+", type=int, default=None, help="Fleet sizes to generate."
    )
    sac_pretrain_parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage3", "paper_stage1", "paper_stage2", "paper_stage3", "paper_stage4"],
        default=None,
        help="Apply a built-in curriculum stage preset.",
    )
    sac_pretrain_parser.add_argument(
        "--epochs", type=int, default=None, help="Number of passes over the training scenario list."
    )
    sac_pretrain_parser.add_argument(
        "--generations", type=int, default=None, help="Generations per training/evaluation run."
    )
    sac_pretrain_parser.add_argument("--population", type=int, default=None, help="Population size per run.")
    sac_pretrain_parser.add_argument(
        "--separation-min", type=float, default=None, help="Minimum fleet separation for generated scenarios."
    )
    sac_pretrain_parser.add_argument(
        "--gpu-mode", choices=("auto", "off", "force"), default="auto", help="GPU mode for training/evaluation runs."
    )
    sac_pretrain_parser.add_argument(
        "--state-representation",
        choices=("flat", "TRFTS-HAND", "TRFTS", "TRFTS-CP"),
        default=None,
        help="Controller state/encoder variant to train and evaluate.",
    )
    sac_pretrain_parser.add_argument(
        "--checkpoint",
        default=SAC_PRETRAIN_CHECKPOINT_DEFAULT,
        type=str,
        help="Checkpoint path for the pretrained SAC controller.",
    )
    sac_pretrain_parser.add_argument(
        "--results-dir",
        default=SAC_PRETRAIN_RESULTS_DEFAULT,
        type=str,
        help="Directory for training/evaluation artifacts and summaries.",
    )
    sac_pretrain_parser.add_argument(
        "--skip-eval", action="store_true", help="Only train; do not run frozen-policy evaluation."
    )
    sac_pretrain_parser.add_argument(
        "--reset", action="store_true", help="Delete any existing checkpoint before training."
    )

    sac_policy_eval_parser = subparsers.add_parser("sac-policy-eval")
    sac_policy_eval_parser.add_argument("--protocol", type=str, default=None, help="YAML protocol config path.")
    sac_policy_eval_parser.add_argument(
        "--problems", nargs="+", default=None, help="Base problem files under problems/ used for evaluation."
    )
    sac_policy_eval_parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Evaluation scenario seeds.")
    sac_policy_eval_parser.add_argument(
        "--fleet-sizes", nargs="+", type=int, default=None, help="Fleet sizes to generate."
    )
    sac_policy_eval_parser.add_argument("--generations", type=int, default=None, help="Generations per evaluation run.")
    sac_policy_eval_parser.add_argument("--population", type=int, default=None, help="Population size per run.")
    sac_policy_eval_parser.add_argument(
        "--separation-min", type=float, default=None, help="Minimum separation for generated scenarios."
    )
    sac_policy_eval_parser.add_argument(
        "--gpu-mode", choices=("auto", "off", "force"), default="auto", help="GPU mode for evaluation runs."
    )
    sac_policy_eval_parser.add_argument("--modes", nargs="+", default=None, help="Policy modes to compare.")
    sac_policy_eval_parser.add_argument(
        "--state-representation",
        default=None,
        choices=("flat", "TRFTS-HAND", "TRFTS", "TRFTS-CP"),
        help="Controller state/encoder variant used during policy comparison.",
    )
    sac_policy_eval_parser.add_argument("--checkpoint", required=False, type=str, help="Pretrained checkpoint path.")
    sac_policy_eval_parser.add_argument(
        "--results-dir",
        default=SAC_POLICY_RESULTS_DEFAULT,
        type=str,
        help="Directory for evaluation artifacts and summary JSON.",
    )

    sac_encoder_ablation_parser = subparsers.add_parser("sac-encoder-ablation")
    sac_encoder_ablation_parser.add_argument(
        "--problems", nargs="+", default=None, help="Problem files under problems/ to evaluate."
    )
    sac_encoder_ablation_parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds.")
    sac_encoder_ablation_parser.add_argument(
        "--fleet-sizes", nargs="+", type=int, default=None, help="Fleet sizes for generated fleet scenarios."
    )
    sac_encoder_ablation_parser.add_argument("--generations", type=int, default=None, help="Generations per run.")
    sac_encoder_ablation_parser.add_argument("--population", type=int, default=None, help="Population size.")
    sac_encoder_ablation_parser.add_argument(
        "--fleet-size", type=int, default=None, help="Fleet size for generated fleet scenarios."
    )
    sac_encoder_ablation_parser.add_argument(
        "--separation-min", type=float, default=None, help="Separation minimum for generated fleets."
    )
    sac_encoder_ablation_parser.add_argument(
        "--gpu-mode", choices=("auto", "off", "force"), default=None, help="GPU mode for evaluation runs."
    )
    sac_encoder_ablation_parser.add_argument(
        "--modes", nargs="+", default=None, help="State/encoder variants to compare."
    )
    sac_encoder_ablation_parser.add_argument(
        "--policy-mode",
        choices=("online", "finetune", "frozen"),
        default=None,
        help="Controller policy mode to evaluate.",
    )
    sac_encoder_ablation_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Single checkpoint path used for every mode."
    )
    sac_encoder_ablation_parser.add_argument(
        "--checkpoint-template",
        type=str,
        default=None,
        help="Checkpoint template, e.g. results/sac_smopso_pretrain/{mode}/controller.pt.",
    )
    sac_encoder_ablation_parser.add_argument("--protocol", type=str, default=None, help="YAML protocol config path.")
    sac_encoder_ablation_parser.add_argument(
        "--results-dir",
        default=SAC_ABLATION_RESULTS_DEFAULT,
        type=str,
        help="Output directory for ablation artifacts and summary JSON.",
    )
