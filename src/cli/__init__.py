"""Command-line entry points (SPEC §10).

Subcommands
-----------
- ``prepare-data``  : raw -> interim -> splits.
- ``build-topics``  : ``data/processed/topics.csv`` (SPEC §4.2.1).
- ``splits-version``: ``splits_version_legacy`` helper (TASK-002a).
- ``train-baseline``, ``train-sparse``, ``train-retrieval``,
  ``train-hybrid``: training entry points per family.
- ``eval``          : evaluation on a split (Recall@k, slices, manifest).
- ``prefix-eval``   : offline prefix-mode UX simulation + tau grid (TASK-012).
- ``build-manifest``: standalone manifest builder.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from src.cli import (
    build_manifest,
    build_topics,
    evaluate,
    eval_three_way,
    prefix_eval,
    predict_three_way,
    prepare_data,
    splits_version,
    train_baseline,
    train_hybrid,
    train_retrieval,
    train_sparse,
    train_three_way,
)


Handler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser with TASK-001 commands."""
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_prepare = subparsers.add_parser("prepare-data", help="Prepare raw/interim data (stub).")
    parser_prepare.add_argument("--config", required=True)
    parser_prepare.set_defaults(handler=prepare_data.main)

    parser_build_topics = subparsers.add_parser(
        "build-topics",
        help="Build deterministic topics.csv from clean.csv.",
    )
    parser_build_topics.add_argument("--config", required=True)
    parser_build_topics.set_defaults(handler=build_topics.main)

    parser_splits_version = subparsers.add_parser(
        "splits-version",
        help="Compute split hashes and write splits_version_legacy report.",
    )
    parser_splits_version.add_argument("--paths", required=False, default="")
    parser_splits_version.set_defaults(handler=splits_version.main)

    parser_train_baseline = subparsers.add_parser(
        "train-baseline",
        help="Train most-frequent baseline and write artifacts/metrics.",
    )
    parser_train_baseline.add_argument("--config", required=True)
    parser_train_baseline.set_defaults(handler=train_baseline.main)

    parser_train_sparse = subparsers.add_parser(
        "train-sparse",
        help="Train TF-IDF + LinearSVC/LogReg/SGD sparse model; metrics + manifest + leaderboard row.",
    )
    parser_train_sparse.add_argument("--config", required=True)
    parser_train_sparse.set_defaults(handler=train_sparse.main)

    parser_train_retrieval = subparsers.add_parser(
        "train-retrieval",
        help="Train E5 centroid + FAISS IP retrieval, manifest + leaderboard (TASK-005).",
    )
    parser_train_retrieval.add_argument("--config", required=True)
    parser_train_retrieval.set_defaults(handler=train_retrieval.main)

    parser_train_hybrid = subparsers.add_parser(
        "train-hybrid",
        help="TASK-006: hybrid sparse+retrieval (late fusion, val lambda, diagnostics).",
    )
    parser_train_hybrid.add_argument("--config", required=True)
    parser_train_hybrid.add_argument(
        "--skip-leaderboard",
        action="store_true",
        help="Do not append to reports/leaderboard.csv (grid search / TASK-007).",
    )
    parser_train_hybrid.set_defaults(handler=train_hybrid.main)

    parser_train_three_way = subparsers.add_parser(
        "train-three-way",
        help="TASK-040: write fusion artifacts under artifacts/three_way_hybrid/.",
    )
    parser_train_three_way.add_argument("--config", required=True)
    parser_train_three_way.set_defaults(handler=train_three_way.main)

    parser_predict_three_way = subparsers.add_parser(
        "predict-three-way",
        help="TASK-040: three-way metrics on val/test (precomputed embeddings).",
    )
    parser_predict_three_way.add_argument("--config", required=True)
    parser_predict_three_way.add_argument("--split", required=True, choices=["val", "test"])
    parser_predict_three_way.add_argument(
        "--out",
        required=True,
        help="Path to metrics.json (default run_id = parent directory name).",
    )
    parser_predict_three_way.add_argument("--run-id", default=None)
    parser_predict_three_way.add_argument("--device", default=None)
    parser_predict_three_way.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
    )
    parser_predict_three_way.set_defaults(handler=predict_three_way.main)

    parser_eval_three_way = subparsers.add_parser(
        "eval-three-way",
        help="TASK-040: three-way eval + manifest + leaderboard append.",
    )
    parser_eval_three_way.add_argument("--config", required=True)
    parser_eval_three_way.add_argument("--split", required=True, choices=["val", "test"])
    parser_eval_three_way.add_argument("--device", default=None)
    parser_eval_three_way.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
    )
    parser_eval_three_way.set_defaults(handler=eval_three_way.main)

    parser_evaluate = subparsers.add_parser(
        "evaluate",
        help="Eval from saved artifacts: baseline, sparse, retrieval, hybrid (see inference.artifacts_dir).",
    )
    parser_evaluate.add_argument("--config", required=True)
    parser_evaluate.add_argument("--split", required=True, choices=["val", "test"])
    parser_evaluate.add_argument(
        "--artifacts",
        default=None,
        help="Artifact directory for the model run (overrides inference.artifacts_dir in config).",
    )
    parser_evaluate.set_defaults(handler=evaluate.main)

    parser_eval_alias = subparsers.add_parser("eval", help="Alias for evaluate.")
    parser_eval_alias.add_argument("--config", required=True)
    parser_eval_alias.add_argument("--split", required=True, choices=["val", "test"])
    parser_eval_alias.add_argument("--artifacts", default=None)
    parser_eval_alias.set_defaults(handler=evaluate.main)

    parser_prefix_eval = subparsers.add_parser(
        "prefix-eval",
        help="Prefix UX offline simulation + tau grid (TASK-012).",
    )
    parser_prefix_eval.add_argument("--config", required=True)
    parser_prefix_eval.add_argument("--split", required=True, choices=["val", "test"])
    parser_prefix_eval.add_argument("--best", default=None, help="Hybrid artifacts directory.")
    parser_prefix_eval.add_argument("--calibrator", default=None, help="Path to isotonic.pkl.")
    parser_prefix_eval.add_argument("--out", default=None, help="reports/runs/<run_id>/ output directory.")
    parser_prefix_eval.add_argument("--ux-policy", dest="ux_policy", default=None)
    parser_prefix_eval.add_argument("--device", default=None)
    parser_prefix_eval.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Encoder batch size (HybridModel.load batch_size); default from device.",
    )
    parser_prefix_eval.add_argument(
        "--device-notes-file",
        dest="device_notes_file",
        default=None,
        help="UTF-8 text appended under 'Device selection / fallback' in prefix_decision_note.md.",
    )
    parser_prefix_eval.set_defaults(handler=prefix_eval.main)

    parser_build_manifest = subparsers.add_parser(
        "build-manifest",
        help="Build run manifest from config/run metadata (stub).",
    )
    parser_build_manifest.add_argument("--config", required=True)
    parser_build_manifest.set_defaults(handler=build_manifest.main)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by src.cli.__main__."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args))
