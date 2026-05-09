"""Smoke eval: three-way fusion metrics on val/test (TASK-040 A3)."""

from __future__ import annotations

import argparse
import json
import logging
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
from src.data.splits_version import splits_version
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
from src.models.three_way_hybrid import ThreeWayHybrid
from src.utils.run_manifest import file_sha256

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        out = yaml.safe_load(f)
    return out if isinstance(out, dict) else {}


def _prepare_split_df(
    config: dict[str, Any],
    split: str,
) -> tuple[pd.DataFrame, dict[str, int], str, dict[str, Any], int]:
    data_cfg = config["data"]
    train_path = Path(str(data_cfg["train_path"]))
    val_path = Path(str(data_cfg["val_path"]))
    test_path = Path(str(data_cfg["test_path"]))
    topics_path = Path(str(data_cfg["topics_path"]))
    tax = Path(str(data_cfg.get("taxonomy_yaml", "configs/taxonomy.yaml")))
    raw_train = pd.read_parquet(train_path)
    raw_eval = pd.read_parquet(val_path if split == "val" else test_path)

    aliases = load_taxonomy_aliases(str(tax))
    tmap, tax_ver = theme_to_topic_id_map(str(topics_path), str(tax))
    train_df0 = prepare_split(raw_train, tmap, aliases, str(tax))
    eval_df0 = prepare_split(raw_eval, tmap, aliases, str(tax))
    pre = config.get("preprocessing")
    pre_d = pre if isinstance(pre, dict) else None
    if pre_d:
        train_df = filter_by_preprocessed_text(train_df0, pre_d)
        eval_df = eval_df0.copy()
        eval_df["text"] = [
            preprocess_text_padded(str(t), pre_d) for t in eval_df0["text"].astype(str).tolist()
        ]
    else:
        train_df = train_df0
        eval_df = eval_df0

    y_train = train_df["topic_id"].astype(str)
    train_topic_counts = {str(t): int(c) for t, c in y_train.value_counts().items()}
    n_train_classes = int(len(set(y_train.tolist())))

    all_train_topics = set(y_train.tolist())
    n_total = int(len(raw_eval))
    known_mask = eval_df["topic_id"].astype(str).isin(all_train_topics)
    n_known = int(known_mask.sum())
    n_unseen = n_total - n_known

    sv = splits_version(str(train_path), str(val_path), str(test_path), legacy_random_val=True)
    split_meta = {
        "train_path": sv["train_path"],
        "val_path": sv["val_path"],
        "test_path": sv["test_path"],
        "train_hash": str(sv["train_hash"]),
        "val_hash": str(sv["val_hash"]),
        "test_hash": str(sv["test_hash"]),
        "splits_version": str(sv["splits_version"]),
        "n_test_total": n_total,
        "n_test_known": n_known,
        "n_test_unseen": n_unseen,
    }
    return eval_df, train_topic_counts, tax_ver, split_meta, n_train_classes


def _build_metrics(
    run_id: str,
    config: dict[str, Any],
    config_path: str,
    config_hash: str,
    family: str,
    split: str,
    test_df: pd.DataFrame,
    train_topic_counts: dict[str, int],
    y_pred_topk: list[list[str]],
    latencies: list[float],
    artifacts_dir: str,
    manifest_path: str,
    taxonomy_ver: str,
    split_meta: dict[str, Any],
    retrieval_meta: dict[str, Any] | None,
    latency_device: str,
    n_train_classes: int,
    notes: str = "",
) -> dict[str, Any]:
    y_true = test_df["topic_id"].astype(str).tolist()
    labels = sorted(set(train_topic_counts.keys()))
    eval_cfg = config.get("eval", {}) or {}
    hmt = eval_cfg.get("head_mid_tail", {}) or {}
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
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    out: dict[str, Any] = {
        "run_id": run_id,
        "model_family": family,
        "config_path": config_path,
        "config_hash": config_hash,
        "seed": int(config.get("seed", 42)),
        "split": split,
        "unseen_policy": str(eval_cfg.get("unseen_policy", "known_only")),
        "k_list": list(k_tup),
        "n_test_total": int(split_meta["n_test_total"]),
        "n_test_known": int(split_meta["n_test_known"]),
        "n_test_unseen": int(split_meta["n_test_unseen"]),
        "taxonomy_version": taxonomy_ver,
        "n_train_classes": int(n_train_classes),
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
        "latency_ms": {"p50": p50, "p95": p95, "n": len(y_true), "device": latency_device},
        "three_way_diagnostics": {
            "precomputed_embeddings": True,
            "artifacts_dir": artifacts_dir,
        },
        "artifacts": {
            "artifacts_dir": artifacts_dir,
            "manifest_path": manifest_path,
        },
        "notes": notes,
    }
    if retrieval_meta is not None:
        out["retrieval_meta"] = dict(retrieval_meta)
    return out


def run_predict_three_way(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg_path = Path(args.config).resolve()
    config = _load_yaml(cfg_path)
    split = str(args.split)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(args.run_id) if getattr(args, "run_id", None) else out_path.parent.name

    inf = config.get("inference") or {}
    if not isinstance(inf, dict) or not inf.get("artifacts_dir"):
        raise ValueError("config.inference.artifacts_dir required")
    art_dir = Path(str(inf["artifacts_dir"]).strip()).resolve()

    eval_df, train_topic_counts, tax_ver, split_meta, n_train_classes = _prepare_split_df(
        config, split
    )
    texts = eval_df["text"].astype(str).tolist()
    with (art_dir / "fusion_config.json").open("r", encoding="utf-8") as f:
        fc: dict[str, Any] = json.load(f)
    ret_dir = Path(str(fc.get("retrieval_artifacts_dir", "")))

    if split == "test":
        emb_p = ret_dir / "embeddings_test.npy"
    else:
        emb_p = ret_dir / "embeddings_val.npy"
    if not emb_p.is_file():
        raise FileNotFoundError(f"precomputed embeddings missing: {emb_p}")
    emb = np.ascontiguousarray(np.load(str(emb_p), allow_pickle=False).astype(np.float32))
    if emb.shape[0] != len(texts):
        raise ValueError(f"embeddings rows {emb.shape[0]} != texts {len(texts)}")

    device = getattr(args, "device", None) or "auto"
    batch_size = getattr(args, "batch_size", None)
    model = ThreeWayHybrid.load(art_dir, device=device, batch_size=batch_size)
    k_list = (config.get("eval", {}) or {}).get("k_list", [1, 3, 5, 10])
    k = max(int(max(k_list)), 10)

    y_pred, _, latencies = model.predict_topk_precomputed(texts, emb, k=k)

    source_run_id = art_dir.name
    manifest_p = Path("artifacts") / "manifests" / f"{source_run_id}.json"
    manifest_s = str(manifest_p).replace("\\", "/") if manifest_p.is_file() else ""
    ret_meta_path = ret_dir / "meta.json"
    with ret_meta_path.open("r", encoding="utf-8") as rf:
        retrieval_meta = json.load(rf)

    target = 0.8554707379
    r10 = float(recall_at_k(eval_df["topic_id"].astype(str).tolist(), y_pred, k_list=(10,))["10"])
    delta = abs(r10 - target)
    tol = 1e-2 if "cuda" in str(model.device).lower() else 1e-3
    notes = ""
    if delta > 1e-6:
        notes = f"bit_exact_target_delta={delta:.6e} tolerance={tol}"
    if delta > tol:
        notes = f"BLOCKER drift vs TASK-037 target {target}: delta={delta}"

    ch = file_sha256(cfg_path)
    metrics = _build_metrics(
        run_id=run_id,
        config=config,
        config_path=str(cfg_path.as_posix()),
        config_hash=ch,
        family="three_way_hybrid",
        split=split,
        test_df=eval_df,
        train_topic_counts=train_topic_counts,
        y_pred_topk=y_pred,
        latencies=latencies,
        artifacts_dir=str(art_dir).replace("\\", "/"),
        manifest_path=manifest_s,
        taxonomy_ver=tax_ver,
        split_meta=split_meta,
        retrieval_meta=retrieval_meta,
        latency_device="precomputed_embeddings",
        n_train_classes=n_train_classes,
        notes=notes,
    )
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    logger.info("recall@10=%s wrote %s", r10, out_path)
    print(f"recall@10: {r10}")
    print(f"wrote: {out_path.as_posix()}")
    return 0


def main(args: argparse.Namespace) -> int:
    return run_predict_three_way(args)
