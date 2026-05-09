from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
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
from src.models import hybrid as hyb
from src.models import sparse as sparse_mod
from src.models.retrieval import (
    _batch_size,
    _resolve_device,
    encode_passages,
)
from src.utils.leaderboard import append_row
from src.utils.run_manifest import build_manifest, file_sha256, write_manifest
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


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


def _write_lambda_sweep_debug(
    report_dir: Path,
    ttexts: list[str],
    y_test: list[str],
    s_test: np.ndarray,
    r_test: np.ndarray,
    classes: list[str],
    candidates: int,
    fnorm: str,
    test_lambdas: list[float],
) -> None:
    """DEBUG log: 3-5 test rows (seed=42) for TASK-006a."""
    rng = np.random.RandomState(42)
    n = int(s_test.shape[0])
    k = min(5, max(3, n))
    idxs = sorted(rng.choice(n, size=k, replace=False).tolist())
    debug_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    rows_out: list[dict[str, Any]] = []
    for text_id in idxs:
        s_row = s_test[text_id]
        r_row = r_test[text_id]
        u = hyb.union_topk_indices(s_row, r_row, candidates)
        u_arr = np.asarray(u, dtype=np.int64)
        vs = s_row[u_arr].astype(np.float64, copy=False)
        vr = r_row[u_arr].astype(np.float64, copy=False)
        ns = hyb.normalize_per_query_row(vs, fnorm)
        nr = hyb.normalize_per_query_row(vr, fnorm)
        norm_sparse = [{"class": classes[u[jj]], "score": float(ns[jj])} for jj in range(len(u))]
        norm_ret = [{"class": classes[u[jj]], "score": float(nr[jj])} for jj in range(len(u))]
        sparse_top10 = [
            {"class": classes[j], "score": float(s_row[j])}
            for j in np.argsort(-s_row)[:10].tolist()
        ]
        retrieval_top10 = [
            {"class": classes[j], "score": float(r_row[j])}
            for j in np.argsort(-r_row)[:10].tolist()
        ]
        h10: dict[str, list[str]] = {}
        for dl in debug_lambdas:
            p = hyb.hybrid_topk_from_full(
                s_test[text_id : text_id + 1],
                r_test[text_id : text_id + 1],
                float(dl),
                "weighted_score",
                fnorm,
                candidates,
                classes,
                k_out=10,
            )[0]
            h10[str(float(dl))] = p
        y_true = y_test[text_id]
        hit_at: dict[str, bool] = {}
        for gl in test_lambdas:
            p = hyb.hybrid_topk_from_full(
                s_test[text_id : text_id + 1],
                r_test[text_id : text_id + 1],
                float(gl),
                "weighted_score",
                fnorm,
                candidates,
                classes,
                k_out=10,
            )[0]
            hit_at[str(float(gl))] = y_true in p[:10]
        rows_out.append(
            {
                "text_id": int(text_id),
                "text": (ttexts[text_id] if text_id < len(ttexts) else "")[:200],
                "topic_id_true": y_true,
                "sparse_top10": sparse_top10,
                "retrieval_top10": retrieval_top10,
                "norm_sparse_on_union": norm_sparse,
                "norm_retrieval_on_union": norm_ret,
                "hybrid_top10_per_lambda": h10,
                "hit_at_10_per_lambda": hit_at,
            }
        )
    dbg = report_dir / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    out_p = dbg / "lambda_sweep_examples.json"
    with out_p.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _build_metrics_hybrid(
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
    retrieval_meta: dict[str, Any] | None,
    latency_device: str,
    n_train_classes: int,
    hybrid_diagnostics: dict[str, Any],
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
    out: dict[str, Any] = {
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
        "hybrid_diagnostics": dict(hybrid_diagnostics),
        "artifacts": {
            "artifacts_dir": artifacts_dir,
            "manifest_path": manifest_path,
        },
    }
    if retrieval_meta is not None:
        out["retrieval_meta"] = dict(retrieval_meta)
    return out


def train_hybrid_cmd(config_path: str, *, skip_leaderboard: bool = False) -> str:
    """TASK-006: late-fusion hybrid eval + manifest + metrics (no sparse/retrieval retrain)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _set_seeds(42)
    config = _load_yaml(config_path)
    family = str(config.get("model_family", "hybrid"))
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("config.data must be a mapping")
    preproc = config.get("preprocessing")
    if not isinstance(preproc, dict) and preproc is not None:
        raise TypeError("config.preprocessing must be a dict or null")
    hcfg: dict[str, Any] = dict(config.get("hybrid", {}) or {})
    fusion = dict(hcfg.get("fusion", {}) or {})
    fmode = str(fusion.get("mode", "weighted_score"))
    fnorm = str(fusion.get("norm", "minmax"))
    candidates = int(fusion.get("candidates", 50))
    k_rrf = int(fusion.get("k_rrf", 60))
    rrf_k_grid = fusion.get("rrf_k_grid")
    if not isinstance(rrf_k_grid, list) or not rrf_k_grid:
        rrf_k_grid = [k_rrf]
    rrf_k_grid = [int(x) for x in rrf_k_grid]
    # DECISION-022: raw scores for intersection / full_row Pearson; fixed methodology.
    pearson_norm = "raw"
    sparse_dir = str(hcfg.get("sparse_artifacts_dir", "")).replace("\\", "/")
    ret_dir = str(hcfg.get("retrieval_artifacts_dir", "")).replace("\\", "/")
    ret_man_path = hcfg.get("retrieval_manifest_path")
    if not ret_man_path:
        ret_man_path = f"artifacts/manifests/{Path(ret_dir).name}.json"
    val_grid = fusion.get("calibrate_lambda_grid")
    if not isinstance(val_grid, list) or not val_grid:
        val_grid = hcfg.get("calibrate_lambda_grid")
    if not isinstance(val_grid, list) or not val_grid:
        val_grid = [round(i * 0.05, 2) for i in range(21)]
    test_lambda_grid = fusion.get("test_lambda_grid")
    if not isinstance(test_lambda_grid, list) or not test_lambda_grid:
        test_lambda_grid = hcfg.get("test_lambda_grid")
    if not isinstance(test_lambda_grid, list) or not test_lambda_grid:
        test_lambda_grid = list(val_grid)

    train_path = str(data_cfg["train_path"])
    val_path = str(data_cfg["val_path"])
    test_path = str(data_cfg["test_path"])
    topics_path = str(data_cfg["topics_path"])
    taxonomy_yaml = str(data_cfg.get("taxonomy_yaml", "configs/taxonomy.yaml"))

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sm = fmode.replace(" ", "_")
    sn = fnorm.replace(" ", "_")
    run_id = f"{run_ts}_hybrid_{sm}_{sn}"

    aliases = load_taxonomy_aliases(taxonomy_yaml)
    topics_map, taxonomy_ver = theme_to_topic_id_map(topics_path, taxonomy_yaml)
    train_raw = pd.read_parquet(train_path)
    val_raw = pd.read_parquet(val_path)
    test_raw = pd.read_parquet(test_path)
    train_df0 = prepare_split(train_raw, topics_map, aliases, taxonomy_yaml)
    val_df0 = prepare_split(val_raw, topics_map, aliases, taxonomy_yaml)
    test_df0 = prepare_split(test_raw, topics_map, aliases, taxonomy_yaml)
    preproc_dict = preproc if isinstance(preproc, dict) else None
    if preproc_dict:
        train_df = filter_by_preprocessed_text(train_df0, preproc_dict)
    else:
        train_df = train_df0
    val_df = _apply_padded_text(val_df0, preproc_dict or {})
    test_df = _apply_padded_text(test_df0, preproc_dict or {})

    y_train = train_df["topic_id"].astype(str)
    train_topic_counts = {str(t): int(c) for t, c in y_train.value_counts().items()}
    n_train_classes = int(len(set(y_train.tolist())))

    classes_s = hyb.load_json_classes(sparse_dir)
    index, meta, classes_r = hyb.load_retrieval_index_and_meta(ret_dir)
    hyb.assert_classes_r004(classes_s, classes_r)
    classes = classes_s

    m_sparse = sparse_mod.load(sparse_dir)
    if list(m_sparse["classes"]) != classes:
        raise RuntimeError("sparse model classes out of sync")

    enc_device = hcfg.get("encoding_device", "auto")
    enc_cfg: dict[str, Any] = {
        "retrieval": {
            "device": str(enc_device),
            "batch_size_gpu": int(hcfg.get("batch_size_gpu", 32)),
            "batch_size_cpu": int(hcfg.get("batch_size_cpu", 16)),
        }
    }
    dev = _resolve_device(enc_cfg)
    bsz = _batch_size(enc_cfg["retrieval"], dev)  # type: ignore[index]
    mname = str(meta["model_name"])
    p_q = str(meta.get("e5_prefix_query", "query: "))
    norm_e = bool(meta["normalize_embeddings"])

    _set_seeds(int(config.get("seed", 42)))
    enc = SentenceTransformer(mname, device=dev)
    msl = int(hcfg.get("max_seq_length", 512))
    try:
        for m in enc:
            m.max_seq_length = msl
    except Exception:
        pass
    vtexts = val_df["text"].astype(str).tolist()
    logger.info("Encoding val (%d) for hybrid (E5, meta.normalize_embeddings only).", len(vtexts))
    emb_val = encode_passages(
        enc,
        vtexts,
        p_q,
        norm_e,
        bsz,
    )
    del enc

    n_classes = len(classes)
    s_val = hyb.sparse_full_scores(m_sparse, vtexts, batch_size=int(hcfg.get("sparse_batch_size", 256)))
    r_val = hyb.retrieval_full_scores(index, n_classes, emb_val, k_search=n_classes)
    y_val = val_df["topic_id"].astype(str).tolist()

    if fmode == "rrf" and len(rrf_k_grid) > 1:
        br = k_rrf
        bv = -1.0
        for kv in rrf_k_grid:
            pv = hyb.hybrid_topk_from_full(
                s_val, r_val, 0.0, "rrf", fnorm, candidates, classes, k_out=10, k_rrf=int(kv)
            )
            rv = float(recall_at_k(y_val, pv, k_list=(10,))["10"])
            if rv > bv:
                bv, br = rv, int(kv)
        k_rrf = br

    best_lam = 0.5
    val_lam_sweep: dict[str, float] = {}
    if fmode == "weighted_score":
        best_lam, val_lam_sweep = hyb.calibrate_lambda_weighted(
            s_val,
            r_val,
            val_df["topic_id"].astype(str).tolist(),
            classes,
            [float(x) for x in val_grid],
            fnorm,
            candidates,
            k=10,
        )
    else:
        bl, _g, _m = hyb.calibrate_lambda(
            s_val,
            r_val,
            y_val,
            [float(x) for x in val_grid],
            fmode,
            fnorm,
            candidates,
            classes,
            k=10,
            k_rrf=k_rrf,
        )
        best_lam = bl
        val_lam_sweep = dict(_m) if isinstance(_m, dict) else {}

    rho_inter = hyb.pearson_sparse_retrieval_val(
        s_val, r_val, classes, pearson_norm, candidates, mode="intersection"
    )
    rho_full = hyb.pearson_sparse_retrieval_val(
        s_val, r_val, classes, pearson_norm, candidates, mode="full_row"
    )
    t1cr = hyb.top1_change_rate_val(
        s_val, r_val, best_lam, classes, fnorm, candidates, fmode, k_rrf=k_rrf
    )

    emb_path = Path("artifacts") / "hybrid" / run_id
    emb_path.mkdir(parents=True, exist_ok=True)
    val_emb_fp = emb_path / "embeddings_val.npy"
    np.save(str(val_emb_fp), emb_val, allow_pickle=False)

    etest_fp = Path(ret_dir) / "embeddings_test.npy"
    emb_test = np.ascontiguousarray(
        np.load(str(etest_fp), allow_pickle=False).astype(np.float32, copy=False)
    )

    k_list = (config.get("eval", {}) or {}).get("k_list", [1, 3, 5, 10])
    kmax = max(int(max(k_list)), 10, 10)
    ttexts = test_df["text"].astype(str).tolist()
    s_test = hyb.sparse_full_scores(m_sparse, ttexts, batch_size=int(hcfg.get("sparse_batch_size", 256)))
    r_test = hyb.retrieval_full_scores(
        index, n_classes, emb_test, k_search=n_classes
    )

    y_test = test_df["topic_id"].astype(str).tolist()
    lambda_grid_test: list[dict[str, Any]] = []
    for gl in test_lambda_grid:
        y_p = hyb.hybrid_topk_from_full(
            s_test,
            r_test,
            float(gl),
            "weighted_score",
            fnorm,
            candidates,
            classes,
            k_out=10,
            k_rrf=k_rrf,
        )
        r10t = float(recall_at_k(y_test, y_p, k_list=(10,))["10"])
        lambda_grid_test.append({"lambda": float(gl), "recall_at_10": r10t})

    sparse_run_id = Path(sparse_dir).name
    retrieval_run_id = Path(ret_dir).name
    sparse_metrics_path = Path("reports") / "runs" / sparse_run_id / "metrics.json"
    with sparse_metrics_path.open("r", encoding="utf-8") as f:
        sm = json.load(f)
    ref_sparse_r10 = float(sm["recall_at_k"]["10"])
    retrieval_metrics_path = Path("reports") / "runs" / retrieval_run_id / "metrics.json"
    with retrieval_metrics_path.open("r", encoding="utf-8") as f:
        rm = json.load(f)
    ref_ret_r10 = float(rm["recall_at_k"]["10"])
    l1 = next(
        (x for x in lambda_grid_test if abs(x["lambda"] - 1.0) < 1e-9),
        None,
    )
    l0 = next(
        (x for x in lambda_grid_test if abs(x["lambda"] - 0.0) < 1e-9),
        None,
    )
    if l1 is None or abs(l1["recall_at_10"] - ref_sparse_r10) > 0.005:
        raise RuntimeError(
            f"R-003/sanity: lambda=1 R@10 {l1!r} does not match sparse reference "
            f"{ref_sparse_r10:.4f} (tolerance 0.005). Zero-fill or fusion bug. "
            f"See DECISION-2026-04-27-023."
        )
    if l0 is None or abs(l0["recall_at_10"] - ref_ret_r10) > 0.005:
        raise RuntimeError(
            f"R-003/sanity: lambda=0 R@10 {l0!r} does not match retrieval reference "
            f"{ref_ret_r10:.4f} (tolerance 0.005). See DECISION-2026-04-27-023."
        )

    y_pred = hyb.hybrid_topk_from_full(
        s_test,
        r_test,
        best_lam,
        fmode,
        fnorm,
        candidates,
        classes,
        k_out=kmax,
        k_rrf=k_rrf,
    )
    pred_s = [
        [classes[j] for j in np.argsort(-s_test[i])[:10].tolist()]
        for i in range(s_test.shape[0])
    ]
    r_only = [
        [classes[j] for j in np.argsort(-r_test[i])[:10].tolist()]
        for i in range(r_test.shape[0])
    ]
    pred_h = y_pred
    pdiv = hyb.per_bucket_diversity(
        y_test,
        pred_s,
        r_only,
        [p[:10] for p in pred_h],
        train_topic_counts,
        head_min=int(fusion.get("diversity_head_min", 50)),
        mid_min=int(fusion.get("diversity_mid_min", 10)),
    )

    _write_lambda_sweep_debug(
        Path("reports") / "runs" / run_id,
        ttexts,
        y_test,
        s_test,
        r_test,
        classes,
        candidates,
        fnorm,
        [float(x) for x in test_lambda_grid],
    )

    latencies: list[float] = []
    n_test = len(ttexts)
    logger.info("Timing hybrid path per test row (sparse+faiss+fusion)")
    for i in range(n_test):
        t0 = time.perf_counter()
        s_one = hyb.sparse_full_scores(m_sparse, [ttexts[i]], batch_size=1)
        qv = np.ascontiguousarray(emb_test[i : i + 1])
        d, ind = index.search(qv, min(n_classes, int(index.ntotal)))
        r_one = hyb.scatter_retrieval_distances(ind, d, n_classes)
        _ = hyb.hybrid_topk_from_full(
            s_one, r_one, best_lam, fmode, fnorm, candidates, classes, k_out=10, k_rrf=k_rrf
        )[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)

    all_train_topics = set(train_df["topic_id"].astype(str).tolist())
    test_total = int(len(test_raw))
    test_known_mask = test_df["topic_id"].astype(str).isin(all_train_topics)
    test_known = int(test_known_mask.sum())
    test_unseen = int(test_total - test_known)

    ad = str(emb_path).replace("\\", "/")
    classes_hyb = emb_path / "classes.json"
    shutil.copy2(Path(sparse_dir) / "classes.json", classes_hyb)

    cls_sha = file_sha256(classes_hyb)
    fusion_payload: dict[str, Any] = {
        "mode": fmode,
        "norm": fnorm,
        "candidates": candidates,
        "best_lambda": float(best_lam),
        "k_rrf": k_rrf,
        "sparse_run_id": sparse_run_id,
        "retrieval_run_id": retrieval_run_id,
        "sparse_artifacts_dir": sparse_dir,
        "retrieval_artifacts_dir": ret_dir,
        "classes_sha256": cls_sha,
    }
    fusion_path = emb_path / "fusion_config.json"
    with fusion_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(fusion_payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    retrieval_manifest: dict[str, Any] | None = None
    rmp = Path(str(ret_man_path))
    if rmp.is_file():
        with rmp.open("r", encoding="utf-8") as rf:
            retrieval_manifest = json.load(rf)
    if retrieval_manifest and "retrieval_meta" in retrieval_manifest:
        rmeta = dict(retrieval_manifest["retrieval_meta"])
    else:
        tr_sha = file_sha256(Path(ret_dir) / "embeddings_train.npy")
        te_sha = file_sha256(Path(ret_dir) / "embeddings_test.npy")
        rmeta = {
            "encoder_name": str(meta.get("model_name", "")),
            "encoder_revision": str(meta.get("encoder_revision", "unknown")),
            "normalize_embeddings": bool(meta.get("normalize_embeddings", True)),
            "faiss_metric": str(meta.get("faiss_metric", "IP")),
            "embedding_dim": int(meta.get("embedding_dim", 0)),
            "index_ntotal": int(index.ntotal),
            "representation": str(meta.get("representation", "centroid")),
            "classes_path": f"{ret_dir}/classes.json",
            "e5_prefix_query": str(meta.get("e5_prefix_query", "query: ")),
            "e5_prefix_passage": str(meta.get("e5_prefix_passage", "passage: ")),
            "encoder_batch_size": int(hcfg.get("batch_size_gpu", 32)),
            "device": str(dev),
            "embeddings_train_sha256": tr_sha,
            "embeddings_test_sha256": te_sha,
        }

    artifacts_paths = {
        "fusion_config": f"{ad}/fusion_config.json",
        "classes": f"{ad}/classes.json",
        "embeddings_val": f"{ad}/embeddings_val.npy",
    }
    cfg_path = str(Path(config_path).as_posix())
    manifest = build_manifest(
        run_id=run_id,
        family=family,
        config_path=config_path,
        config=config,
        artifacts=artifacts_paths,
        retrieval_meta=rmeta,
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
    val_r10_best = float(val_lam_sweep.get(str(float(best_lam)), float("nan")))
    hybrid_diag: dict[str, Any] = {
        "best_lambda": float(best_lam),
        "val_recall_at_10": val_r10_best,
        "val_lambda_sweep": {k: float(v) for k, v in val_lam_sweep.items()},
        "lambda_grid": lambda_grid_test,
        "pearson_intersection_val": float(rho_inter),
        "pearson_full_row_val": float(rho_full),
        "pearson_sparse_dense_val": float(rho_inter),
        "top1_change_rate_val": float(t1cr),
        "per_bucket_diversity_test": pdiv,
        "fusion_mode": fmode,
        "norm": fnorm,
        "candidates": candidates,
    }
    if fmode == "rrf":
        hybrid_diag["k_rrf_used"] = int(k_rrf)
    metrics_payload = _build_metrics_hybrid(
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
        artifacts_dir=ad,
        taxonomy_ver=taxonomy_ver,
        split_meta=split_meta,
        retrieval_meta=rmeta,
        latency_device=str(dev),
        n_train_classes=n_train_classes,
        hybrid_diagnostics=hybrid_diag,
    )
    report_dir = Path("reports") / "runs" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    mpath = report_dir / "metrics.json"
    with mpath.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metrics_payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    if not skip_leaderboard:
        append_row(
            Path("reports") / "leaderboard.csv",
            run_id=run_id,
            model_family=family,
            metrics=metrics_payload,
            manifest=manifest,
            config_path=cfg_path,
        )
    print(run_id)
    return run_id


def main(args: argparse.Namespace) -> int:
    train_hybrid_cmd(
        config_path=args.config,
        skip_leaderboard=bool(getattr(args, "skip_leaderboard", False)),
    )
    return 0
