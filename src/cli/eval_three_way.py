"""Full three-way eval: metrics + manifest + leaderboard (TASK-040 A10)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.cli.predict_three_way import _build_metrics, _prepare_split_df
from src.models.three_way_hybrid import ThreeWayHybrid
from src.utils.leaderboard import append_row
from src.utils.run_manifest import build_manifest, file_sha256, write_manifest

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        out = yaml.safe_load(f)
    return out if isinstance(out, dict) else {}


def run_eval_three_way(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg_path = Path(args.config).resolve()
    config = _load_yaml(cfg_path)
    split = str(args.split)

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

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_run_id = f"{run_ts}_eval_three_way_{split}"

    source_run_id = art_dir.name
    with (ret_dir / "meta.json").open("r", encoding="utf-8") as rf:
        retrieval_meta = json.load(rf)

    tr_sha = file_sha256(ret_dir / "embeddings_train.npy")
    te_sha = file_sha256(ret_dir / "embeddings_test.npy")
    rmeta_lb = {
        "encoder_name": str(retrieval_meta.get("model_name", "")),
        "encoder_revision": str(retrieval_meta.get("encoder_revision", "unknown")),
        "normalize_embeddings": bool(retrieval_meta.get("normalize_embeddings", True)),
        "faiss_metric": str(retrieval_meta.get("faiss_metric", "IP")),
        "embedding_dim": int(retrieval_meta.get("embedding_dim", 0)),
        "index_ntotal": int(retrieval_meta.get("index_ntotal", 0)),
        "representation": str(retrieval_meta.get("representation", "centroid")),
        "classes_path": (ret_dir / "classes.json").as_posix(),
        "e5_prefix_query": str(retrieval_meta.get("e5_prefix_query", "query: ")),
        "e5_prefix_passage": str(retrieval_meta.get("e5_prefix_passage", "passage: ")),
        "embeddings_train_sha256": tr_sha,
        "embeddings_test_sha256": te_sha,
    }

    ad = str(art_dir).replace("\\", "/")
    artifacts_paths = {
        "fusion_config": f"{ad}/fusion_config.json",
        "classes": f"{ad}/classes.json",
    }
    cfg_posix = str(cfg_path.as_posix())
    manifest = build_manifest(
        eval_run_id,
        "three_way_hybrid",
        cfg_posix,
        config,
        artifacts_paths,
        retrieval_meta=rmeta_lb,
        legacy_splits=False,
        splits_policy="time_based",
    )
    metrics_path = Path("reports") / "runs" / eval_run_id / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["metrics_path"] = str(metrics_path).replace("\\", "/")
    manifest_path = Path("artifacts") / "manifests" / f"{eval_run_id}.json"
    write_manifest(manifest, manifest_path)
    manifest_s = str(manifest_path).replace("\\", "/")

    ch = file_sha256(cfg_path)
    lb_notes = (
        "three_way_hybrid_promoted, fusion_mode=weighted_score_minmax, TASK-040 "
        f"source_artifacts={source_run_id}"
    )
    metrics = _build_metrics(
        run_id=eval_run_id,
        config=config,
        config_path=cfg_posix,
        config_hash=ch,
        family="three_way_hybrid",
        split=split,
        test_df=eval_df,
        train_topic_counts=train_topic_counts,
        y_pred_topk=y_pred,
        latencies=latencies,
        artifacts_dir=ad,
        manifest_path=manifest_s,
        taxonomy_ver=tax_ver,
        split_meta=split_meta,
        retrieval_meta=retrieval_meta,
        latency_device="precomputed_embeddings",
        n_train_classes=n_train_classes,
        notes=lb_notes,
    )
    with metrics_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    append_row(
        Path("reports") / "leaderboard.csv",
        run_id=eval_run_id,
        model_family="three_way_hybrid_promoted",
        metrics=metrics,
        manifest=manifest,
        config_path=cfg_posix,
        split=split,
        notes=lb_notes,
    )
    r10 = metrics.get("recall_at_k", {}).get("10")
    logger.info("eval_run_id=%s recall@10=%s", eval_run_id, r10)
    print(eval_run_id)
    print(f"recall@10: {r10}")
    return 0


def main(args: argparse.Namespace) -> int:
    return run_eval_three_way(args)
