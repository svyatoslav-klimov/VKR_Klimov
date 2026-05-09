"""
E5 bi-encoder retrieval with FAISS IndexFlatIP (centroid representation).

Contract: meta.json next to themes.faiss (faiss-normalization-check skill).
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EPS = 1e-12


def encode_passages(
    encoder: SentenceTransformer,
    texts: list[str],
    prefix: str,
    normalize: bool,
    batch_size: int,
) -> np.ndarray:
    """Encode texts with a fixed string prefix; returns float32 (n, dim)."""
    prefixed = [f"{prefix}{t}" for t in texts]
    n = len(prefixed)
    if n == 0:
        d = int(encoder.get_sentence_embedding_dimension())
        return np.zeros((0, d), dtype=np.float32)
    logger.info("encode_passages: start 0 / %d", n)
    parts: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        if start > 0 and start % 5000 == 0:
            logger.info("encode_passages: progress %d / %d", start, n)
        chunk = prefixed[start : start + batch_size]
        emb = encoder.encode(
            chunk,
            batch_size=len(chunk),
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        parts.append(np.asarray(emb, dtype=np.float32))
    return np.vstack(parts).astype(np.float32, copy=False)


def build_centroids(
    per_item_emb: np.ndarray,
    topic_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Average per class, L2-normalize each centroid row. Classes sorted by topic_id."""
    if per_item_emb.shape[0] != len(topic_ids):
        raise ValueError("per_item_emb and topic_ids length mismatch")
    by_c: dict[str, list[np.ndarray]] = defaultdict(list)
    for tid, row in zip(topic_ids, per_item_emb, strict=True):
        by_c[str(tid)].append(np.asarray(row, dtype=np.float64))
    classes = sorted(by_c.keys())
    if not classes:
        raise ValueError("no classes in centroids build")
    dim = per_item_emb.shape[1]
    out = np.zeros((len(classes), dim), dtype=np.float32)
    for i, c in enumerate(classes):
        stack = np.stack(by_c[c], axis=0)
        m = np.mean(stack, axis=0)
        nrm = float(np.linalg.norm(m))
        out[i, :] = (m / max(nrm, _EPS)).astype(np.float32, copy=False)
    return out, classes


def build_index(centroids: np.ndarray) -> faiss.IndexFlatIP:
    if centroids.dtype != np.float32:
        raise ValueError(f"centroids must be float32, got {centroids.dtype}")
    dim = int(centroids.shape[1])
    index = faiss.IndexFlatIP(dim)
    x = centroids.astype(np.float32, copy=True)
    index.add(x)
    assert int(index.ntotal) == len(
        centroids
    ), f"index.ntotal {index.ntotal} != n centroids {len(centroids)}"
    return index


def write_meta(
    out_dir: Path,
    encoder_name: str,
    encoder_revision: str,
    embedding_dim: int,
    index_ntotal: int,
    classes_path: str,
    prefix_query: str,
    prefix_passage: str,
    *,
    normalize_embeddings: bool,
    faiss_metric: str,
    representation: str,
) -> dict[str, Any]:
    """Write ``meta.json`` and return the dict written (7+ keys; includes E5 prefixes)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "model_name": encoder_name,
        "normalize_embeddings": bool(normalize_embeddings),
        "faiss_metric": str(faiss_metric),
        "embedding_dim": int(embedding_dim),
        "index_ntotal": int(index_ntotal),
        "representation": str(representation),
        "classes_path": str(classes_path),
        "e5_prefix_query": str(prefix_query),
        "e5_prefix_passage": str(prefix_passage),
        "encoder_revision": str(encoder_revision),
    }
    dest = out_dir / "meta.json"
    with dest.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return meta


def _encoder_revision(model: SentenceTransformer) -> str:
    try:
        mod = model[0] if len(model) > 0 else None
        if mod is not None and hasattr(mod, "auto_model"):
            cfg = mod.auto_model.config
            h = getattr(cfg, "_commit_hash", None) or getattr(cfg, "commit_hash", None)
            if h is not None:
                return str(h)
    except Exception:
        pass
    return "unknown"


def _resolve_device(cfg: dict[str, Any]) -> str:
    import torch

    d = str(cfg.get("retrieval", {}).get("device", "auto")).lower()
    if d == "cpu":
        return "cpu"
    if d == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _batch_size(ret_cfg: dict[str, Any], device: str) -> int:
    if device == "cuda":
        return int(ret_cfg.get("batch_size_gpu", 32))
    return int(ret_cfg.get("batch_size_cpu", 16))


def fit_retrieval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    ret_cfg: dict[str, Any] = dict(config.get("retrieval", {}) or {})
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    import random

    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    encoder_name = str(
        ret_cfg.get("encoder_name")
        or ret_cfg.get("model_name")
        or "intfloat/multilingual-e5-base"
    )
    rep = str(ret_cfg.get("representation", "centroid"))
    norm = bool(ret_cfg.get("normalize_embeddings", True))
    p_q = str(ret_cfg.get("e5_prefix_query", "query: "))
    p_p = str(ret_cfg.get("e5_prefix_passage", "passage: "))
    faiss_metric = str(ret_cfg.get("faiss_metric", "IP"))
    if faiss_metric != "IP":
        raise ValueError("TASK-005 only supports faiss_metric=IP for IndexFlatIP")
    if rep != "centroid":
        raise NotImplementedError(f"representation {rep!r} not implemented yet (centroid only)")

    device = _resolve_device(config)
    if device == "cpu":
        print(
            "WARN: CUDA unavailable, falling back to CPU "
            "(expected ~30-60 min on 144k+6k texts)"
        )
    bsz = _batch_size(ret_cfg, device)

    logger.info("Loading encoder %s on %s (batch_size=%d)", encoder_name, device, bsz)
    encoder = SentenceTransformer(
        encoder_name,
        device=device,
    )
    msl = int(ret_cfg.get("max_seq_length", 512))
    try:
        for m in encoder:
            m.max_seq_length = msl
    except Exception:
        pass

    train_texts = train_df["text"].astype(str).tolist()
    train_topics = train_df["topic_id"].astype(str).tolist()
    n_train = len(train_texts)
    logger.info("Encoding %d train rows (passage prefix)", n_train)
    emb_train = encode_passages(
        encoder, train_texts, p_p, norm, bsz
    )
    if emb_train.shape[0] != n_train:
        raise RuntimeError("train embedding count mismatch")

    centroids, classes = build_centroids(emb_train, train_topics)
    index = build_index(centroids)
    assert int(index.ntotal) == len(classes)

    emb_test: np.ndarray | None = None
    if test_df is not None:
        n_test = len(test_df)
        test_texts = test_df["text"].astype(str).tolist()
        logger.info("Encoding %d test rows (query prefix) for cache + eval", n_test)
        emb_test = encode_passages(
            encoder, test_texts, p_q, norm, bsz
        )
        if emb_test.shape[0] != n_test:
            raise RuntimeError("test embedding count mismatch")

    enc_rev = _encoder_revision(encoder)
    dim = int(centroids.shape[1])
    file_meta: dict[str, Any] = {
        "model_name": encoder_name,
        "normalize_embeddings": norm,
        "faiss_metric": faiss_metric,
        "embedding_dim": dim,
        "index_ntotal": int(index.ntotal),
        "representation": rep,
        "classes_path": "classes.json",
        "e5_prefix_query": p_q,
        "e5_prefix_passage": p_p,
        "encoder_revision": enc_rev,
    }
    return {
        "encoder": encoder,
        "centroids": centroids,
        "classes": classes,
        "index": index,
        "embeddings_train": emb_train,
        "embeddings_test": emb_test,
        "meta_file": file_meta,
        "device": device,
        "encoder_name": encoder_name,
        "encoder_revision": enc_rev,
        "batch_size_used": bsz,
        "prefix_query": p_q,
        "prefix_passage": p_p,
        "normalize_embeddings": norm,
    }


def predict_topk(
    texts: list[str],
    artifacts_dir: str | Path,
    k: int = 10,
    *,
    encoder: SentenceTransformer | None = None,
) -> tuple[list[list[str]], list[list[float]], list[float]]:
    """
    Load index + meta from disk; encode queries using **only** meta.json
    (anti-R-001 / anti-R-005). If ``encoder`` is passed, reuse (train-time latency).
    """
    root = Path(artifacts_dir)
    with (root / "meta.json").open("r", encoding="utf-8") as f:
        meta: dict[str, Any] = json.load(f)
    model_name = meta["model_name"]
    norm = bool(meta["normalize_embeddings"])
    p_q = str(meta.get("e5_prefix_query", "query: "))
    classes_relp = str(meta.get("classes_path", "classes.json"))
    class_path = root / classes_relp if not Path(classes_relp).is_absolute() else Path(classes_relp)
    with class_path.open("r", encoding="utf-8") as f:
        classes: list[str] = json.load(f)
    index = faiss.read_index(str((root / "themes.faiss").resolve()))
    assert int(index.ntotal) == len(classes), f"{index.ntotal} != {len(classes)}"

    enc = encoder
    if enc is None:
        enc = SentenceTransformer(model_name, device="cpu")
        msl = 512
        try:
            for m in enc:
                m.max_seq_length = msl
        except Exception:
            pass

    k_eff = min(int(k), len(classes)) if classes else 0
    out_ids: list[list[str]] = []
    out_scores: list[list[float]] = []
    lats: list[float] = []
    for t in texts:
        t0 = time.perf_counter()
        q = enc.encode(
            [f"{p_q}{t}"],
            batch_size=1,
            normalize_embeddings=norm,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        qv = np.asarray(q, dtype=np.float32)
        d, ind = index.search(qv, k_eff)
        row_d = d[0]
        row_i = ind[0]
        tid = [classes[int(j)] for j in row_i]
        out_ids.append(tid)
        out_scores.append([float(x) for x in row_d])
        lats.append((time.perf_counter() - t0) * 1000.0)
    return out_ids, out_scores, lats


def search_with_query_embeddings(
    index: faiss.IndexFlatIP,
    classes: list[str],
    query_emb: np.ndarray,
    k: int,
) -> tuple[list[list[str]], list[list[float]]]:
    """Batch search; query_emb float32, L2-normalized (E5 with normalize=True)."""
    k_eff = min(int(k), len(classes)) if classes else 0
    if query_emb.dtype != np.float32:
        query_emb = query_emb.astype(np.float32, copy=False)
    d, ind = index.search(query_emb, k_eff)
    top_ids: list[list[str]] = []
    top_s: list[list[float]] = []
    for r in range(query_emb.shape[0]):
        top_ids.append([classes[int(j)] for j in ind[r]])
        top_s.append([float(x) for x in d[r]])
    return top_ids, top_s


def save_retrieval(artifacts_dir: str | Path, result: dict[str, Any]) -> None:
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes: list[str] = list(result["classes"])
    index: faiss.IndexFlatIP = result["index"]
    assert int(index.ntotal) == len(classes)

    with (out / "classes.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
        f.write("\n")

    faiss.write_index(index, str((out / "themes.faiss").resolve()))
    np.save(str(out / "embeddings_train.npy"), result["embeddings_train"], allow_pickle=False)
    et = result.get("embeddings_test")
    if et is not None:
        np.save(str(out / "embeddings_test.npy"), et, allow_pickle=False)

    m = dict(result["meta_file"])
    with (out / "meta.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
        f.write("\n")

