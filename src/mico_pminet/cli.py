from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_protocol, repository_root


def _defaults() -> dict[str, Path]:
    root = repository_root()
    return {
        "root": root,
        "processed": root / "data" / "processed",
        "raw": root / "data" / "raw" / "all_data_acquire.csv.gz",
        "reference": root / "reference",
        "results": root / "results",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _defaults()
    parser = argparse.ArgumentParser(
        prog="mico-pminet",
        description="Reproduce the core MICO-PMInet computational workflow",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Rebuild processed data")
    preprocess.add_argument("--raw", type=Path, default=defaults["raw"])
    preprocess.add_argument("--output-dir", type=Path, default=defaults["results"] / "processed")
    preprocess.add_argument("--split-seed", type=int, default=22)

    pls = subparsers.add_parser("pls", help="Run eight single-organ PLS models")
    pls.add_argument("--processed-dir", type=Path, default=defaults["processed"])
    pls.add_argument("--output-dir", type=Path, default=defaults["results"] / "pls")

    train = subparsers.add_parser("train", help="Train MICO-PMInet models")
    train.add_argument("--processed-dir", type=Path, default=defaults["processed"])
    train.add_argument("--output-dir", type=Path, default=defaults["results"] / "deep")
    train.add_argument("--profile", choices=("smoke", "optimal", "paper"), default="smoke")
    train.add_argument(
        "--protocol",
        choices=("reported_results", "manuscript_protocol"),
        default="manuscript_protocol",
    )
    train.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    train.add_argument("--resume", action="store_true")

    anova = subparsers.add_parser("anova", help="Run Type II multifactor ANOVA")
    anova.add_argument(
        "--fold-results",
        type=Path,
        default=defaults["results"] / "deep" / "fold_metrics.csv",
    )
    anova.add_argument("--output-dir", type=Path, default=defaults["results"] / "anova")

    selection = subparsers.add_parser(
        "select", help="Select beta for the manuscript model using validation metrics"
    )
    selection.add_argument(
        "--fold-results",
        type=Path,
        default=defaults["reference"] / "published_beta_summary.csv",
    )
    selection.add_argument(
        "--output-dir", type=Path, default=defaults["results"] / "selection"
    )
    selection.add_argument("--criterion", default="Val_RMSE")
    selection.add_argument("--tie-breaker", default="Val_MAE")
    selection.add_argument("--model-name", default="OSB-WMHA-AWA-MORM")

    stats = subparsers.add_parser(
        "stats", help="Run split-correlated validation-only component analysis"
    )
    stats.add_argument(
        "--fold-results",
        type=Path,
        default=defaults["reference"] / "published_fold_metrics.csv",
    )
    stats.add_argument("--output-dir", type=Path, default=defaults["results"] / "stats")

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate one checkpoint by held-out cohort and organ retention"
    )
    evaluate.add_argument("--processed-dir", type=Path, default=defaults["processed"])
    evaluate.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults["reference"] / "mico_pminet_fold5.pt",
    )
    evaluate.add_argument("--model-name", default="OSB-WMHA-AWA-MORM")
    evaluate.add_argument(
        "--output-dir", type=Path, default=defaults["results"] / "evaluation"
    )
    evaluate.add_argument("--repeats", type=int, default=10)

    weights = subparsers.add_parser(
        "awa-weights", help="Export learned AWA weights separately from SHAP"
    )
    weights.add_argument("--processed-dir", type=Path, default=defaults["processed"])
    weights.add_argument("--checkpoint", type=Path, action="append")
    weights.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Optional directory containing fold*.pth checkpoints",
    )
    weights.add_argument("--model-name", default="OSB-WMHA-AWA-MORM")
    weights.add_argument(
        "--output-dir", type=Path, default=defaults["results"] / "awa_weights"
    )

    shap_parser = subparsers.add_parser("shap", help="Aggregate or recompute SHAP")
    shap_parser.add_argument("--processed-dir", type=Path, default=defaults["processed"])
    shap_parser.add_argument("--output-dir", type=Path, default=defaults["results"] / "shap")
    shap_parser.add_argument(
        "--cache", type=Path, default=defaults["reference"] / "published_shap_values.npz"
    )
    shap_parser.add_argument(
        "--checkpoint", type=Path, default=defaults["reference"] / "mico_pminet_fold5.pt"
    )
    shap_parser.add_argument("--recompute", action="store_true")
    shap_parser.add_argument("--seed", type=int, default=225)
    shap_parser.add_argument("--model-name", default="OSB-WMHA-AWA-MORM")

    verify = subparsers.add_parser("verify", help="Run fast release checks")
    verify.add_argument("--processed-dir", type=Path, default=defaults["processed"])
    verify.add_argument("--reference-dir", type=Path, default=defaults["reference"])
    verify.add_argument("--deep-results", type=Path)

    all_parser = subparsers.add_parser("all", help="Run the complete workflow")
    all_parser.add_argument("--profile", choices=("smoke", "optimal", "paper"), default="paper")
    all_parser.add_argument(
        "--protocol",
        choices=("reported_results", "manuscript_protocol"),
        default="manuscript_protocol",
    )
    all_parser.add_argument("--device", default="cpu")
    all_parser.add_argument("--resume", action="store_true")
    all_parser.add_argument("--recompute-shap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    defaults = _defaults()
    if args.command == "preprocess":
        from .preprocess import run_preprocessing

        run_preprocessing(args.raw, args.output_dir, args.split_seed)
    elif args.command == "pls":
        from .pls import run_pls

        run_pls(args.processed_dir, args.output_dir)
    elif args.command == "train":
        from .data import load_tensor_data
        from .training import run_experiment

        run_experiment(
            load_tensor_data(args.processed_dir),
            load_protocol(args.protocol),
            args.output_dir,
            args.profile,
            args.device,
            args.resume,
        )
    elif args.command == "anova":
        from .statistics import run_anova

        run_anova(args.fold_results, args.output_dir)
    elif args.command == "select":
        from .selection import validation_only_selection

        validation_only_selection(
            args.fold_results,
            args.output_dir,
            criterion=args.criterion,
            tie_breaker=args.tie_breaker,
            target_model=args.model_name,
        )
    elif args.command == "stats":
        from .statistics import run_correlated_analysis

        run_correlated_analysis(args.fold_results, args.output_dir)
    elif args.command == "evaluate":
        from .reporting import evaluate_checkpoint_by_cohort

        evaluate_checkpoint_by_cohort(
            args.checkpoint,
            args.model_name,
            args.processed_dir,
            args.output_dir,
            repeats=args.repeats,
        )
    elif args.command == "awa-weights":
        from .reporting import export_awa_weights

        checkpoints = list(args.checkpoint or [])
        if args.checkpoint_dir:
            checkpoints.extend(sorted(args.checkpoint_dir.glob("fold*.pth")))
        if not checkpoints:
            checkpoints = [defaults["reference"] / "mico_pminet_fold5.pt"]
        export_awa_weights(
            checkpoints, args.model_name, args.processed_dir, args.output_dir
        )
    elif args.command == "shap":
        from .interpret import recompute_shap, run_cached_shap

        if args.recompute:
            recompute_shap(
                args.checkpoint,
                args.processed_dir,
                args.output_dir,
                seed=args.seed,
                model_name=args.model_name,
            )
        else:
            run_cached_shap(args.cache, args.processed_dir, args.output_dir)
    elif args.command == "verify":
        from .verify import verify

        verify(args.processed_dir, args.reference_dir, args.deep_results)
    elif args.command == "all":
        from .data import load_tensor_data
        from .interpret import recompute_shap, run_cached_shap
        from .pls import run_pls
        from .preprocess import run_preprocessing
        from .selection import validation_only_selection
        from .statistics import run_correlated_analysis
        from .training import run_experiment
        from .verify import verify

        processed = defaults["results"] / "processed"
        run_preprocessing(defaults["raw"], processed, split_seed=22)
        run_pls(processed, defaults["results"] / "pls")
        run_experiment(
            load_tensor_data(processed),
            load_protocol(args.protocol),
            defaults["results"] / "deep",
            args.profile,
            args.device,
            args.resume,
        )
        validation_only_selection(
            defaults["results"] / "deep" / "fold_metrics_all_betas.csv",
            defaults["results"] / "selection",
            target_model="OSB-WMHA-AWA-MORM",
        )
        run_correlated_analysis(
            defaults["results"] / "deep" / "fold_metrics.csv",
            defaults["results"] / "stats",
        )
        if args.recompute_shap:
            checkpoint = defaults["results"] / "deep" / "mico_pminet_fold5.pt"
            recompute_shap(checkpoint, processed, defaults["results"] / "shap")
        else:
            run_cached_shap(
                defaults["reference"] / "published_shap_values.npz",
                processed,
                defaults["results"] / "shap",
            )
        verify(processed, defaults["reference"], defaults["results"] / "deep" / "model_summary.csv")


if __name__ == "__main__":
    main()
