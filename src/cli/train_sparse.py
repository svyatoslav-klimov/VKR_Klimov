from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.prepare import (
    filter_by_preprocessed_text,
    load_taxonomy_aliases,
    prepare_split,
    preprocess_text_padded,
    theme_to_topic_id_map,
)
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
import src.models.sparse as sparse_model
from src.utils.leaderboard import append_row
from src.utils.run_manifest import build_manifest, file_sha256, write_manifest

def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def _apply_padded_text(df: pd.DataFrame, preproc: dict[str, Any] | None) -> pd.DataFrame:
    o = df.copy()
    o["text"] = [preprocess_text_padded(t, preproc) for t in o["text"].astype(str).tolist()]
    return o


def _build_metrics(
    run_id: str,
    config_path: str,
    config: dict[str, Any],
    config_hash: str,
    family: str,
    test_df: pd.DataFrame,
    train_topic_counts: dict[str, int],
    y_pred_topk: list[list[str]],
    latencies: list[float],
    manifest_path: str,
    artifacts_dir: str,
    taxonomy_ver: str,
    split_meta: dict[str, Any],
) -> dict[str, Any]:
    y_true = test_df["topic_id"].astype(str).tolist()
    labels = sorted(set(train_topic_counts.keys()))
    eval_cfg = config.get("eval", {})
    hmt = eval_cfg.get("head_mid_tail", {})
    p = hmt.get("practical", {}) or {}
    s = hmt.get("strict", {}) or {}
    head_p = int(p.get("head_min", 50))
    mid_p = int(p.get("mid_min", 10))
    head_s = int(s.get("head_min", 500))
    mid_s = int(s.get("mid_min", 50))
    n_time = int(eval_cfg.get("time_slices", 4))
    k_tup = tuple(int(x) for x in (eval_cfg.get("k_list") or [1, 3, 5, 10]))

    recall = recall_at_k(y_true, y_pred_topk, k_list=tuple(k_tup))
    practical = head_mid_tail_recall(
        y_true=y_true,
        y_pred_topk=y_pred_topk,
        topic_counts=train_topic_counts,
        mode="practical",
        head_min=head_p,
        mid_min=mid_p,
    )
    strict = head_mid_tail_recall(
        y_true=y_true,
        y_pred_topk=y_pred_topk,
        topic_counts=train_topic_counts,
        mode="strict",
        head_min=head_s,
        mid_min=mid_s,
    )
    by_time = time_slice_recall(
        y_true=y_true,
        y_pred_topk=y_pred_topk,
        created_at=test_df["created_at"].tolist(),
        n_buckets=n_time,
    )
    p50, p95 = 0.0, 0.0
    if latencies:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
    return {
        "run_id": run_id,
        "model_family": family,
        "config_path": config_path,
        "config_hash": config_hash,
        "seed": int(config.get("seed", 42)),
        "split": "test",
        "unseen_policy": str(eval_cfg.get("unseen_policy", "known_only")),
        "k_list": list(k_tup),
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
        "latency_ms": {"p50": p50, "p95": p95, "n": len(y_true), "device": "cpu"},
        "artifacts": {
            "artifacts_dir": artifacts_dir,
            "manifest_path": manifest_path,
        },
    }


def train_sparse(config_path: str) -> str:
    config = _load_yaml(config_path)
    family = str(config.get("model_family", "sparse"))
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("config.data must be a mapping")
    preproc = config.get("preprocessing")
    if not isinstance(preproc, dict) and preproc is not None:
        raise TypeError("config.preprocessing must be a dict or null")

    train_path = str(data_cfg["train_path"])
    val_path = str(data_cfg["val_path"])
    test_path = str(data_cfg["test_path"])
    topics_path = str(data_cfg["topics_path"])
    taxonomy_yaml = str(data_cfg.get("taxonomy_yaml", "configs/taxonomy.yaml"))

    mcfg = config.get("model", {})
    if not isinstance(mcfg, dict):
        mcfg = {}
    clf_key = str(mcfg.get("classifier", "linear_svc")).lower().replace("-", "_")
    if clf_key in ("linsvc", "linearsvc"):
        clf_key = "linear_svc"
    if clf_key in ("lr", "logistic"):
        clf_key = "logreg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_sparse_{clf_key}"

    aliases = load_taxonomy_aliases(taxonomy_yaml)
    topics_map, taxonomy_ver = theme_to_topic_id_map(topics_path, taxonomy_yaml)

    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    train_df0 = prepare_split(train_raw, topics_map, aliases, taxonomy_yaml)
    test_df0 = prepare_split(test_raw, topics_map, aliases, taxonomy_yaml)
    preproc_dict = preproc if isinstance(preproc, dict) else None
    if preproc_dict:
        train_df = filter_by_preprocessed_text(train_df0, preproc_dict)
    else:
        train_df = train_df0
    test_df = _apply_padded_text(test_df0, preproc_dict or {})

    model = sparse_model.fit(train_df, config, text_col="text", topic_col="topic_id")
    y_train = train_df["topic_id"].astype(str)
    train_topic_counts = {str(t): int(c) for t, c in y_train.value_counts().items()}

    k_list = (config.get("eval", {}) or {}).get("k_list", [1, 3, 5, 10])
    k = max(int(max(k_list)), 10)
    y_pred, _, latencies = sparse_model.predict_topk(
        model, test_df["text"].astype(str).tolist(), k=int(k)
    )

    all_train_topics = set(train_df["topic_id"].astype(str).tolist())
    test_total = int(len(test_raw))
    test_known_mask = test_df["topic_id"].astype(str).isin(all_train_topics)
    test_known = int(test_known_mask.sum())
    test_unseen = int(test_total - test_known)

    artifacts_dir = Path("artifacts") / "sparse_tfidf" / run_id
    sparse_model.save(model, artifacts_dir)
    artifacts_paths = {
        "model": str(artifacts_dir / "model.pkl").replace("\\", "/"),
        "vectorizer": str(artifacts_dir / "vectorizer.pkl").replace("\\", "/"),
        "classes": str(artifacts_dir / "classes.json").replace("\\", "/"),
    }

    cfg_path = str(Path(config_path).as_posix())
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
    ch = file_sha256(config_path)
    metrics_payload = _build_metrics(
        run_id=run_id,
        config_path=cfg_path,
        config=config,
        config_hash=ch,
        family=family,
        test_df=test_df,
        train_topic_counts=train_topic_counts,
        y_pred_topk=y_pred,
        latencies=latencies,
        manifest_path=str(manifest_path).replace("\\", "/"),
        artifacts_dir=str(artifacts_dir).replace("\\", "/"),
        taxonomy_ver=taxonomy_ver,
        split_meta=split_meta,
    )
    report_dir = Path("reports") / "runs" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    mpath = report_dir / "metrics.json"
    with mpath.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metrics_payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    append_row(
        Path("reports") / "leaderboard.csv",
        run_id=run_id,
        model_family=family,
        metrics=metrics_payload,
        manifest=manifest,
        config_path=cfg_path,
    )
    return run_id


def main(args: argparse.Namespace) -> int:
    run_id = train_sparse(config_path=args.config)
    print(run_id)
    return 0
