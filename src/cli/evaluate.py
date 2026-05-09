from __future__ import annotations

import argparse
import json
import time
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
from src.models import baseline_most_frequent as bl
from src.models import retrieval as retr
from src.models import sparse as sparse_mod
from src.utils.run_manifest import file_sha256


def _apply_padded_text(df: pd.DataFrame, preproc: dict | None) -> pd.DataFrame:
    if not preproc:
        return df
    o = df.copy()
    o["text"] = [preprocess_text_padded(t, preproc) for t in o["text"].astype(str).tolist()]
    return o


def _pick_latest_artifact_dir(root: Path, marker: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / marker).is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No subdir with {marker} under {root}. Set inference.artifacts_dir or pass --artifacts."
        )
    return max(candidates, key=lambda p: p.name)


def _resolve_artifacts_dir(
    model_family: str,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> Path:
    override = getattr(args, "artifacts", None)
    if override:
        p = Path(override)
        if not p.is_dir():
            raise FileNotFoundError(f"--artifacts: not a directory: {p}")
        return p.resolve()

    inf = config.get("inference")
    if isinstance(inf, dict) and inf.get("artifacts_dir"):
        p = Path(str(inf["artifacts_dir"]).strip())
        if p.is_dir():
            return p.resolve()
        raise FileNotFoundError(f"inference.artifacts_dir is not a directory: {p}")

    mapping: dict[str, tuple[str, str]] = {
        "hybrid": ("artifacts/hybrid", "fusion_config.json"),
        "sparse": ("artifacts/sparse_tfidf", "model.pkl"),
        "retrieval": ("artifacts/retrieval_e5", "themes.faiss"),
        "baseline_most_frequent": ("artifacts/baseline_most_frequent", "model.pkl"),
    }
    if model_family not in mapping:
        raise ValueError(f"Unknown model_family for eval: {model_family}")
    root_rel, marker = mapping[model_family]
    return _pick_latest_artifact_dir(Path(root_rel), marker).resolve()


def _load_baseline_pickle(artifacts_dir: Path) -> dict[str, Any]:
    import pickle

    p = artifacts_dir / "model.pkl"
    with p.open("rb") as f:
        return pickle.load(f)


def _build_eval_metrics(
    eval_run_id: str,
    source_artifacts: str,
    source_run_id: str,
    config_path: str,
    config: dict[str, Any],
    config_hash: str,
    family: str,
    split: str,
    eval_df: pd.DataFrame,
    train_topic_counts: dict[str, int],
    y_pred_topk: list[list[str]],
    latencies: list[float],
    manifest_path: str,
    taxonomy_ver: str,
    split_meta: dict[str, Any],
    retrieval_meta: dict[str, Any] | None,
    latency_device: str,
    n_train_classes: int | None,
    hybrid_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    y_true = eval_df["topic_id"].astype(str).tolist()
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

    rec = recall_at_k(y_true, y_pred_topk, k_list=tuple(k_tup))
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
    by_t = time_slice_recall(
        y_true=y_true,
        y_pred_topk=y_pred_topk,
        created_at=eval_df["created_at"].tolist(),
        n_buckets=n_time,
    )
    p50, p95 = 0.0, 0.0
    if latencies:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
    out: dict[str, Any] = {
        "eval_run_id": eval_run_id,
        "source_run_id": source_run_id,
        "source_artifacts": source_artifacts.replace("\\", "/"),
        "run_id": eval_run_id,
        "model_family": family,
        "config_path": str(Path(config_path).as_posix()),
        "config_hash": config_hash,
        "seed": int(config.get("seed", 42)),
        "split": split,
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
        "recall_at_k": rec,
        "accuracy_at_1": accuracy_top1(y_true, y_pred_topk),
        "macro_f1": macro_f1_top1(y_true, y_pred_topk, labels=labels),
        "weighted_f1": weighted_f1_top1(y_true, y_pred_topk, labels=labels),
        "mrr_at_10": mrr_at_k(y_true, y_pred_topk, k=10),
        "ndcg_at_10": ndcg_at_k(y_true, y_pred_topk, k=10),
        "head_mid_tail": practical,
        "head_mid_tail_strict": strict,
        "by_time_slice": by_t,
        "latency_ms": {"p50": p50, "p95": p95, "n": len(y_true), "device": latency_device},
        "artifacts": {
            "artifacts_dir": source_artifacts.replace("\\", "/"),
            "manifest_path": manifest_path,
        },
    }
    if n_train_classes is not None:
        out["n_train_classes"] = int(n_train_classes)
    if retrieval_meta is not None:
        out["retrieval_meta"] = dict(retrieval_meta)
    if hybrid_diagnostics is not None:
        out["hybrid_diagnostics"] = dict(hybrid_diagnostics)
    return out


def _load_retrieval_meta(retrieval_dir: Path) -> dict[str, Any] | None:
    run_id = retrieval_dir.name
    man = Path("artifacts") / "manifests" / f"{run_id}.json"
    if not man.is_file():
        return None
    with man.open("r", encoding="utf-8") as f:
        m = json.load(f)
    return m.get("retrieval_meta")  # type: ignore[no-any-return]


def main(args: argparse.Namespace) -> int:
    config_path = str(args.config)
    cfg_p = Path(config_path)
    if not cfg_p.is_file():
        raise FileNotFoundError(config_path)
    with cfg_p.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}
    model_family = str(config.get("model_family", "retrieval"))
    split = str(args.split)
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("config.data must be a mapping")
    train_path = str(data_cfg["train_path"])
    val_path = str(data_cfg["val_path"])
    test_path = str(data_cfg["test_path"])
    split_path = test_path if split == "test" else val_path
    p_split = str(split_path)
    if not p_split:
        raise ValueError("data needs val_path / test_path")

    raw_train = pd.read_parquet(train_path)
    raw_eval = pd.read_parquet(p_split)
    tax = str(data_cfg.get("taxonomy_yaml", "configs/taxonomy.yaml"))
    topics = str(data_cfg["topics_path"])
    aliases = load_taxonomy_aliases(tax)
    tmap, tax_ver = theme_to_topic_id_map(topics, tax)
    train_df0 = prepare_split(raw_train, tmap, aliases, tax)
    eval_df0 = prepare_split(raw_eval, tmap, aliases, tax)
    pre = config.get("preprocessing")
    pre_d = pre if isinstance(pre, dict) else None
    if pre_d:
        train_df = filter_by_preprocessed_text(train_df0, pre_d)
    else:
        train_df = train_df0
    if pre_d and split in ("val", "test"):
        eval_df = _apply_padded_text(eval_df0, pre_d)
    else:
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
    split_meta: dict[str, Any] = {
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

    k_list = (config.get("eval", {}) or {}).get("k_list", [1, 3, 5, 10])
    k = max(int(max(k_list)), 10)
    texts = eval_df["text"].astype(str).tolist()
    ch = file_sha256(config_path)
    art_dir = _resolve_artifacts_dir(model_family, args, config)
    source_run_id = art_dir.name
    manifest_p = Path("artifacts") / "manifests" / f"{source_run_id}.json"
    manifest_s = str(manifest_p).replace("\\", "/") if manifest_p.is_file() else ""

    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_run_id = f"{eval_ts}_eval_{model_family}_{split}"

    y_pred: list[list[str]] = []
    latencies: list[float] = []
    ret_meta: dict[str, Any] | None = None
    hy_diag: dict[str, Any] | None = None
    lat_device = "cpu"

    if model_family == "retrieval":
        y_pred, _, latencies = retr.predict_topk(
            texts,
            str(art_dir),
            k=k,
        )
        ret_meta = _load_retrieval_meta(art_dir)
    elif model_family == "sparse":
        m = sparse_mod.load(art_dir)
        y_pred, _, latencies = sparse_mod.predict_topk(m, texts, k=k)
    elif model_family == "hybrid":
        from src.models import hybrid as hyb

        with (art_dir / "fusion_config.json").open("r", encoding="utf-8") as f:
            fc: dict[str, Any] = json.load(f)
        ret_dir = Path(str(fc.get("retrieval_artifacts_dir", "")))
        if split == "test":
            emb_p = ret_dir / "embeddings_test.npy"
        else:
            emb_p = art_dir / "embeddings_val.npy"
        if not emb_p.is_file():
            raise FileNotFoundError(
                f"R-001/R-005: precomputed embeddings not found: {emb_p}. Re-run train for val cache."
            )
        emb = np.ascontiguousarray(
            np.load(str(emb_p), allow_pickle=False).astype(np.float32, copy=False)
        )
        if emb.shape[0] != len(texts):
            raise ValueError(
                f"precomputed emb rows {emb.shape[0]} != n texts {len(texts)} for split {split}"
            )
        y_pred, _, latencies = hyb.predict_topk(
            texts,
            str(art_dir),
            k=k,
            precomputed_test_embeddings=emb,
        )
        ret_meta = _load_retrieval_meta(ret_dir)
        hy_diag = {
            "eval_fusion_mode": str(fc.get("mode", "")),
            "precomputed_path": str(emb_p).replace("\\", "/"),
        }
        lat_device = "precomputed_embeddings"
    elif model_family == "baseline_most_frequent":
        m = _load_baseline_pickle(art_dir)
        latencies = []
        for t in texts:
            t0 = time.perf_counter()
            p = bl.predict_topk(m, [t], k=k)[0]
            y_pred.append(p)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    else:
        raise ValueError(f"model_family not supported for eval: {model_family}")

    out_metrics = _build_eval_metrics(
        eval_run_id=eval_run_id,
        source_artifacts=str(art_dir).replace("\\", "/"),
        source_run_id=source_run_id,
        config_path=str(Path(config_path).as_posix()),
        config=config,
        config_hash=ch,
        family=model_family,
        split=split,
        eval_df=eval_df,
        train_topic_counts=train_topic_counts,
        y_pred_topk=y_pred,
        latencies=latencies,
        manifest_path=manifest_s,
        taxonomy_ver=tax_ver,
        split_meta=split_meta,
        retrieval_meta=ret_meta,
        latency_device=lat_device,
        n_train_classes=n_train_classes if model_family == "hybrid" else None,
        hybrid_diagnostics=hy_diag if model_family == "hybrid" else None,
    )

    out_dir = Path("reports") / "runs" / eval_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_metrics.json"
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(out_metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    r10 = out_metrics.get("recall_at_k", {}).get("10", None)
    print(f"eval_run_id: {eval_run_id}")
    print(f"source_artifacts: {out_metrics['source_artifacts']}")
    if r10 is not None:
        print(f"recall@10: {r10}")
    print(f"wrote: {out_path.as_posix()}")
    print(json.dumps({k: out_metrics[k] for k in out_metrics if k in ("recall_at_k", "accuracy_at_1", "mrr_at_10", "ndcg_at_10", "split")}, ensure_ascii=False, indent=2))
    return 0
