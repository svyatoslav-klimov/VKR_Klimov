from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.prepare import load_taxonomy_aliases, prepare_split, theme_to_topic_id_map
from src.eval.metrics import (
    accuracy_top1,
    head_mid_tail_recall,
    macro_f1_top1,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    time_slice_recall,
    weighted_f1_top1,
)
from src.models.baseline_most_frequent import fit, predict_topk, save
from src.utils.run_manifest import build_manifest, file_sha256, write_manifest


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def _build_metrics(
    run_id: str,
    config_path: str,
    config_hash: str,
    seed: int,
    family: str,
    test_df: pd.DataFrame,
    train_topic_counts: dict[str, int],
    y_pred_topk: list[list[str]],
    manifest_path: str,
    artifacts_dir: str,
    taxonomy_ver: str,
    split_meta: dict[str, Any],
) -> dict[str, Any]:
    y_true = test_df["topic_id"].astype(str).tolist()
    labels = sorted(set(train_topic_counts.keys()))

    recall = recall_at_k(y_true, y_pred_topk, k_list=(1, 3, 5, 10))
    practical = head_mid_tail_recall(
        y_true=y_true,
        y_pred_topk=y_pred_topk,
        topic_counts=train_topic_counts,
        mode="practical",
        head_min=50,
        mid_min=10,
    )
    strict = head_mid_tail_recall(
        y_true=y_true,
        y_pred_topk=y_pred_topk,
        topic_counts=train_topic_counts,
        mode="strict",
        head_min=500,
        mid_min=50,
    )
    by_time = time_slice_recall(
        y_true=y_true,
        y_pred_topk=y_pred_topk,
        created_at=test_df["created_at"].tolist(),
        n_buckets=4,
    )

    return {
        "run_id": run_id,
        "model_family": family,
        "config_path": config_path,
        "config_hash": config_hash,
        "seed": int(seed),
        "split": "test",
        "unseen_policy": "known_only",
        "k_list": [1, 3, 5, 10],
        "n_test_total": int(split_meta["n_test_total"]),
        "n_test_known": int(split_meta["n_test_known"]),
        "n_test_unseen": int(split_meta["n_test_unseen"]),
        "taxonomy_version": taxonomy_ver,
        "data_version": {
            "train_path": split_meta["train_path"],
            "val_path": split_meta["val_path"],
            "test_path": split_meta["test_path"],
            "train_hash": split_meta["train_hash"],
            "val_hash": split_meta["val_hash"],
            "test_hash": split_meta["test_hash"],
            "splits_version": split_meta["splits_version"],
        },
        "recall_at_k": recall,
        "accuracy_at_1": accuracy_top1(y_true, y_pred_topk),
        "macro_f1": macro_f1_top1(y_true, y_pred_topk, labels=labels),
        "weighted_f1": weighted_f1_top1(y_true, y_pred_topk, labels=labels),
        "mrr_at_10": mrr_at_k(y_true, y_pred_topk, k=10),
        "ndcg_at_10": ndcg_at_k(y_true, y_pred_topk, k=10),
        "head_mid_tail": practical,
        "head_mid_tail_strict": strict,
        "by_time_slice": by_time,
        "latency_ms": {"p50": 0.0, "p95": 0.0, "n": len(y_true), "device": "cpu"},
        "artifacts": {
            "artifacts_dir": artifacts_dir,
            "manifest_path": manifest_path,
        },
    }


def train_baseline(config_path: str) -> str:
    """Run end-to-end MostFrequent baseline training and evaluation."""
    config = _load_yaml(config_path)
    family = str(config.get("model_family", "baseline_most_frequent"))
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("config.data must be a mapping")

    train_path = str(data_cfg["train_path"])
    val_path = str(data_cfg["val_path"])
    test_path = str(data_cfg["test_path"])
    topics_path = str(data_cfg["topics_path"])
    taxonomy_yaml = str(data_cfg.get("taxonomy_yaml", "configs/taxonomy.yaml"))

    aliases = load_taxonomy_aliases(taxonomy_yaml)
    topics_map, taxonomy_ver = theme_to_topic_id_map(topics_path, taxonomy_yaml)

    train_raw = pd.read_parquet(train_path)
    val_raw = pd.read_parquet(val_path)
    test_raw = pd.read_parquet(test_path)

    train_df = prepare_split(train_raw, topics_map=topics_map, aliases=aliases, taxonomy_yaml=taxonomy_yaml)
    val_df = prepare_split(val_raw, topics_map=topics_map, aliases=aliases, taxonomy_yaml=taxonomy_yaml)
    test_df = prepare_split(test_raw, topics_map=topics_map, aliases=aliases, taxonomy_yaml=taxonomy_yaml)
    _ = val_df  # val reserved for future

    model = fit(train_df, topic_col="topic_id")
    train_topic_counts = {str(k): int(v) for k, v in model["topic_counts"].items()}

    test_pred = predict_topk(model, test_df["text"].astype(str).tolist(), k=10)

    all_train_topics = set(train_df["topic_id"].astype(str).tolist())
    test_total = int(len(test_raw))
    test_known_mask = test_df["topic_id"].astype(str).isin(all_train_topics)
    test_known = int(test_known_mask.sum())
    test_unseen = int(test_total - test_known)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_most_frequent"

    artifacts_dir = Path("artifacts") / "baseline_most_frequent" / run_id
    save(model, artifacts_dir)
    artifacts_paths = {
        "model": str(artifacts_dir / "model.pkl").replace("\\", "/"),
        "classes": str(artifacts_dir / "classes.json").replace("\\", "/"),
    }

    manifest = build_manifest(
        run_id=run_id,
        family=family,
        config_path=config_path,
        config=config,
        artifacts=artifacts_paths,
        retrieval_meta=None,
        legacy_splits=False,
    )
    manifest_path = Path("artifacts") / "manifests" / f"{run_id}.json"
    write_manifest(manifest, manifest_path)

    split_meta = {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "train_hash": manifest["data"]["train_hash"],
        "val_hash": manifest["data"]["val_hash"],
        "test_hash": manifest["data"]["test_hash"],
        "splits_version": manifest["data"]["splits_version"],
        "n_test_total": test_total,
        "n_test_known": test_known,
        "n_test_unseen": test_unseen,
    }

    metrics_payload = _build_metrics(
        run_id=run_id,
        config_path=config_path,
        config_hash=file_sha256(config_path),
        seed=int(config.get("seed", 42)),
        family=family,
        test_df=test_df,
        train_topic_counts=train_topic_counts,
        y_pred_topk=test_pred,
        manifest_path=str(manifest_path).replace("\\", "/"),
        artifacts_dir=str(artifacts_dir).replace("\\", "/"),
        taxonomy_ver=taxonomy_ver,
        split_meta=split_meta,
    )

    report_dir = Path("reports") / "runs" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metrics_payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    return run_id


def main(args: argparse.Namespace) -> int:
    """CLI entrypoint for TASK-003 baseline training."""
    run_id = train_baseline(config_path=args.config)
    print(run_id)
    return 0
