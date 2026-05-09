from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
import time
from scipy.stats import pearsonr

_FUSION = Literal["weighted_score", "rrf", "max"]
_NORM = Literal["minmax", "zscore", "softmax", "rank"]


def assert_classes_r004(classes_sparse: list[str], classes_retrieval: list[str]) -> None:
    if classes_sparse != classes_retrieval:
        raise RuntimeError("R-004: classes_sparse != classes_retrieval; hybrid cannot fuse different class sets.")


def load_json_classes(artifacts_dir: str | Path) -> list[str]:
    p = Path(artifacts_dir) / "classes.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def sparse_full_scores(
    model: dict[str, Any],
    texts: list[str] | np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Full per-row scores over all classes: decision_function or predict_proba (shape n, K)."""
    pipe = model["pipeline"]
    clf_name = str(model.get("classifier_name", "linear_svc"))
    texts_l = [str(t) for t in texts] if not isinstance(texts, list) else texts
    out: list[np.ndarray] = []
    use_proba = clf_name == "logreg"
    for i in range(0, len(texts_l), batch_size):
        ch = texts_l[i : i + batch_size]
        if use_proba:
            p = pipe.predict_proba(ch)
        else:
            p = pipe.decision_function(ch)
        a = np.asarray(p, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        out.append(a)
    if not out:
        k = len(model["classes"])
        return np.zeros((0, k), dtype=np.float32)
    return np.vstack(out).astype(np.float32, copy=False)


def scatter_retrieval_distances(
    ind: np.ndarray,
    dist: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Scatter FAISS (distances, labels) to dense rows: shape (n, n_classes). IP scores on diagonal positions."""
    n = int(ind.shape[0])
    r = np.zeros((n, n_classes), dtype=np.float32)
    for row in range(n):
        for p in range(ind.shape[1]):
            cid = int(ind[row, p])
            if 0 <= cid < n_classes:
                r[row, cid] = float(dist[row, p])
    return r


def retrieval_topk_to_full(
    index: faiss.Index,
    emb: np.ndarray,
    n_classes: int,
    *,
    k_search: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (D, I) for search with k = min(available). Caller builds full row via scatter or use retrieval_full_scores."""
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)
    k = int(k_search) if k_search is not None else n_classes
    k = max(1, min(k, n_classes, int(index.ntotal)))
    d, ind = index.search(emb, k)
    return d, ind


def retrieval_full_scores(
    index: faiss.Index,
    n_classes: int,
    emb: np.ndarray,
    *,
    k_search: int | None = None,
) -> np.ndarray:
    """
    R-005: FAISS search only (no re-encoding of test when ``emb`` is precomputed test embeddings).
    Reconstructs a dense score vector per row aligned to class indices (0..K-1).
    """
    d, ind = retrieval_topk_to_full(index, emb, n_classes, k_search=k_search)
    return scatter_retrieval_distances(ind, d, n_classes)


def load_retrieval_index_and_meta(
    retrieval_artifacts_dir: str | Path,
) -> tuple[faiss.Index, dict[str, Any], list[str]]:
    root = Path(retrieval_artifacts_dir).resolve()
    with (root / "meta.json").open("r", encoding="utf-8") as f:
        meta: dict[str, Any] = json.load(f)
    index = faiss.read_index(str((root / "themes.faiss").resolve()))
    classes_relp = str(meta.get("classes_path", "classes.json"))
    cpath = root / classes_relp if not Path(classes_relp).is_absolute() else Path(classes_relp)
    with cpath.open("r", encoding="utf-8") as f:
        classes: list[str] = json.load(f)
    if int(index.ntotal) != len(classes):
        raise ValueError(f"FAISS ntotal {index.ntotal} != len(classes) {len(classes)}")
    return index, meta, classes


def normalize_per_query_row(
    x: np.ndarray,
    mode: str,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Single-query vector to [0,1] (or unit interval target). ``x`` is 1D on candidate union."""
    v = np.asarray(x, dtype=np.float64)
    m = v.size
    if m == 0:
        return v.astype(np.float32)
    if mode == "minmax":
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi - lo < eps:
            return np.full(m, 0.5, dtype=np.float32)
        y = (v - lo) / (hi - lo + eps)
        return np.clip(y, 0.0, 1.0).astype(np.float32)
    if mode == "zscore":
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=0))
        if std < eps:
            z = np.zeros_like(v)
        else:
            z = (v - mean) / (std + eps)
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return s.astype(np.float32)
    if mode == "softmax":
        mx = float(np.max(v))
        e = np.exp(np.clip(v - mx, -50.0, 50.0))
        s = e / (float(np.sum(e)) + eps)
        return s.astype(np.float32)
    if mode == "rank":
        order = np.argsort(-v)
        ranks = np.empty(m, dtype=np.float64)
        for rnk, j in enumerate(order):
            ranks[j] = float(rnk)
        if m <= 1:
            return np.ones(m, dtype=np.float32)
        return (1.0 - ranks / float(m)).astype(np.float32)
    raise ValueError(f"Unknown norm mode: {mode!r}")


def normalize_per_query(
    scores: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Row-wise :func:`normalize_per_query_row` (for tests / batched n x m on same candidate sets)."""
    s = np.asarray(scores, dtype=np.float32)
    if s.ndim == 1:
        return normalize_per_query_row(s, mode)
    out = np.empty_like(s, dtype=np.float32)
    for i in range(s.shape[0]):
        out[i] = normalize_per_query_row(s[i], mode)
    return out


def _ranks_1based_desc(s_row: np.ndarray) -> np.ndarray:
    """1-based rank; rank 1 = largest score. Ties: stable order from argsort."""
    k = s_row.size
    order = np.argsort(-s_row)
    r = np.empty(k, dtype=np.int32)
    for pos, j in enumerate(order):
        r[j] = pos + 1
    return r


def _union_indices(s_row: np.ndarray, r_row: np.ndarray, candidates: int) -> list[int]:
    k = s_row.size
    n = min(candidates, k)
    is_ = set(np.argsort(-s_row)[:n].tolist())
    ir = set(np.argsort(-r_row)[:n].tolist())
    return sorted(is_ | ir)


def union_topk_indices(s_row: np.ndarray, r_row: np.ndarray, candidates: int) -> list[int]:
    """Public alias for the union of top-``candidates`` indices from sparse and retrieval rows."""
    return _union_indices(s_row, r_row, candidates)


def fuse(
    s_topk: list[list[str]],
    s_scores: list[list[float]],
    r_topk: list[list[str]],
    r_scores: list[list[float]],
    lam: float,
    mode: str,
    candidates: int,
    classes: list[str],
    s_full: np.ndarray,
    r_full: np.ndarray,
    *,
    norm: str = "minmax",
    k_rrf: int = 60,
    k_out: int = 10,
) -> list[list[str]]:
    """
    Late fusion on union of top-``candidates`` from sparse (``s_full``) and retrieval (``r_full``) rows.
    ``s_topk``/``r_topk`` optional for API compatibility; dense rows required.
    """
    fusion = str(mode)
    n = s_full.shape[0]
    if r_full.shape != s_full.shape:
        raise ValueError("s_full and r_full must have the same shape")
    k_eff = min(k_out, len(classes)) if classes else 0
    out: list[list[str]] = []
    for i in range(n):
        s_row = s_full[i]
        r_row = r_full[i]
        u = _union_indices(s_row, r_row, candidates)
        if not u or k_eff <= 0:
            out.append([])
            continue
        u_arr = np.asarray(u, dtype=np.int64)
        v_s = s_row[u_arr].astype(np.float64, copy=False)
        v_r = r_row[u_arr].astype(np.float64, copy=False)
        if fusion == "rrf":
            rs_s = _ranks_1based_desc(s_row)
            rs_r = _ranks_1based_desc(r_row)
            h = np.array(
                [
                    1.0 / (k_rrf + float(rs_s[j])) + 1.0 / (k_rrf + float(rs_r[j]))
                    for j in u
                ],
                dtype=np.float64,
            )
        else:
            ns = normalize_per_query_row(v_s, norm)
            nr = normalize_per_query_row(v_r, norm)
            if fusion == "weighted_score":
                h = float(lam) * ns.astype(np.float64) + (1.0 - float(lam)) * nr.astype(np.float64)
            elif fusion == "max":
                h = np.maximum(ns.astype(np.float64), nr.astype(np.float64))
            else:
                raise ValueError(f"Unknown fusion mode: {fusion!r}")
        order_u = np.argsort(-h)
        picked: list[str] = []
        for oi in order_u:
            j = u[int(oi)]
            picked.append(classes[j])
            if len(picked) >= k_eff:
                break
        out.append(picked)
    return out


def _recall_at_10(
    y_true: list[str],
    y_pred: list[list[str]],
) -> float:
    if not y_true:
        return 0.0
    hits = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t in p[:10]:
            hits += 1
    return hits / float(len(y_true))


def calibrate_lambda_weighted(
    s_val: np.ndarray,
    r_val: np.ndarray,
    y_val: list[str],
    classes: list[str],
    grid: list[float],
    norm: str,
    candidates: int,
    k: int = 10,
) -> tuple[float, dict[str, float]]:
    best_lam = float(grid[0])
    best = -1.0
    by_lam: dict[str, float] = {}
    for lam in grid:
        y_pred = fuse(
            [], [], [], [], float(lam), "weighted_score", candidates, classes, s_val, r_val, norm=norm, k_out=k
        )
        r10 = _recall_at_10(y_val, y_pred)
        by_lam[str(float(lam))] = float(r10)
        if r10 > best:
            best = r10
            best_lam = float(lam)
    return best_lam, by_lam


def calibrate_lambda(
    s_scores_val: np.ndarray,
    r_scores_val: np.ndarray,
    y_val: list[str],
    grid: list[float],
    mode: str,
    norm: str,
    candidates: int,
    classes: list[str],
    *,
    k: int = 10,
    k_rrf: int = 60,
) -> tuple[float, list[float], dict[str, float]]:
    """
    Val-only lambda search (weighted_score). For ``rrf`` / ``max`` returns dummy lambda
    and val Recall@10 for a single config (no ``grid`` effect on rrf).
    """
    m = str(mode)
    g = list(grid)
    if m == "weighted_score":
        bl, by = calibrate_lambda_weighted(
            s_scores_val, r_scores_val, y_val, classes, g, norm, candidates, k=k
        )
        out_map = {str(float(x)): float(by.get(str(float(x)), 0.0)) for x in g}
        return bl, g, out_map
    if m == "max":
        y_pred = hybrid_topk_from_full(
            s_scores_val, r_scores_val, 0.0, "max", norm, candidates, classes, k_out=k, k_rrf=k_rrf
        )
        r10 = _recall_at_10(y_val, y_pred)
        return 0.0, g, {str(float(x)): float(r10) for x in g}
    if m == "rrf":
        y_pred = hybrid_topk_from_full(
            s_scores_val, r_scores_val, 0.0, "rrf", norm, candidates, classes, k_out=k, k_rrf=k_rrf
        )
        r10 = _recall_at_10(y_val, y_pred)
        return 0.0, g, {str(float(x)): float(r10) for x in g}
    raise ValueError(f"Unknown mode: {mode!r}")


def hybrid_topk_from_full(
    s_full: np.ndarray,
    r_full: np.ndarray,
    lam: float,
    fusion_mode: str,
    norm: str,
    candidates: int,
    classes: list[str],
    k_out: int = 10,
    *,
    k_rrf: int = 60,
) -> list[list[str]]:
    return fuse(
        [], [], [], [],
        float(lam),
        fusion_mode,
        candidates,
        classes,
        s_full,
        r_full,
        norm=norm,
        k_rrf=k_rrf,
        k_out=k_out,
    )


def pearson_sparse_retrieval_val(
    s_val: np.ndarray,
    r_val: np.ndarray,
    _classes: list[str],
    norm: str,
    candidates: int,
    *,
    mode: str = "intersection",
) -> float:
    """
    Correlation of sparse vs dense (DECISION-022).

    - ``intersection`` (default): per-query, top-``candidates`` from each side; use class indices
      in **both** top sets; then ``raw`` scores or per-query row norm of those slices.
    - ``full_row``: flatten all (s_row, r_row) across queries (275-d per row).
    - ``union``: legacy union-of-top-``candidates`` (no zero-fill; full scores on ``u``).
    """
    xs: list[float] = []
    ys: list[float] = []
    n = s_val.shape[0]
    m = str(mode)
    for i in range(n):
        s_row = s_val[i]
        r_row = r_val[i]
        n_s = min(candidates, len(s_row))
        n_r = min(candidates, len(r_row))
        idx_s = set(np.argsort(-s_row)[:n_s].tolist())
        idx_r = set(np.argsort(-r_row)[:n_r].tolist())
        if m == "intersection":
            inter = sorted(idx_s & idx_r)
            if len(inter) < 2:
                continue
            inter_arr = np.asarray(inter, dtype=np.int64)
            a = s_row[inter_arr].astype(np.float64, copy=False)
            b = r_row[inter_arr].astype(np.float64, copy=False)
            if norm != "raw":
                a = normalize_per_query_row(a, norm).astype(np.float64)
                b = normalize_per_query_row(b, norm).astype(np.float64)
            xs.extend(a.tolist())
            ys.extend(b.tolist())
        elif m == "full_row":
            xs.extend(s_row.astype(np.float64, copy=False).tolist())
            ys.extend(r_row.astype(np.float64, copy=False).tolist())
        elif m == "union":
            u = _union_indices(s_row, r_row, candidates)
            if not u:
                continue
            u_arr = np.asarray(u, dtype=np.int64)
            a = s_row[u_arr].astype(np.float64, copy=False)
            b = r_row[u_arr].astype(np.float64, copy=False)
            if norm != "raw":
                a = normalize_per_query_row(a, norm).astype(np.float64)
                b = normalize_per_query_row(b, norm).astype(np.float64)
            xs.extend(a.tolist())
            ys.extend(b.tolist())
        else:
            raise ValueError(f"Unknown pearson mode: {mode!r}")
    if len(xs) < 2:
        return 0.0
    r, _ = pearsonr(np.asarray(xs), np.asarray(ys))
    if math.isnan(r):
        return 0.0
    return float(r)


def top1_change_rate_val(
    s_val: np.ndarray,
    r_val: np.ndarray,
    lam: float,
    classes: list[str],
    norm: str,
    candidates: int,
    fusion_mode: str = "weighted_score",
    *,
    k_rrf: int = 60,
) -> float:
    """Fraction of val queries where hybrid top-1 is neither sparse top-1 nor retrieval top-1."""
    n = s_val.shape[0]
    if n == 0:
        return 0.0
    changed = 0
    t1h = []
    t1s = [classes[int(j)] for j in np.argmax(s_val, axis=1).tolist()]
    t1r = [classes[int(j)] for j in np.argmax(r_val, axis=1).tolist()]
    pred = hybrid_topk_from_full(
        s_val, r_val, lam, fusion_mode, norm, candidates, classes, k_out=1, k_rrf=k_rrf
    )
    t1h = [p[0] if p else "" for p in pred]
    for hs, hf, hr in zip(t1h, t1s, t1r, strict=True):
        if hs and hs != hf and hs != hr:
            changed += 1
    return changed / float(n)


def per_bucket_diversity(
    y_true: list[str],
    pred_sparse: list[list[str]],
    pred_ret: list[list[str]],
    pred_hyb: list[list[str]],
    topic_counts: dict[str, int],
    *,
    head_min: int = 50,
    mid_min: int = 10,
) -> dict[str, float]:
    """Share of q per bucket where top10_hybrid is not a subset of sparse nor of retrieval top10."""
    buckets: dict[str, list[int]] = {"head": [], "mid": [], "tail": []}
    for idx, tid in enumerate(y_true):
        c = topic_counts.get(tid, 0)
        if c >= head_min:
            buckets["head"].append(idx)
        elif c >= mid_min:
            buckets["mid"].append(idx)
        else:
            buckets["tail"].append(idx)
    out: dict[str, float] = {}
    for name, inds in buckets.items():
        if not inds:
            out[name] = 0.0
            continue
        div = 0
        for i in inds:
            hs = set(pred_hyb[i][:10])
            ss = set(pred_sparse[i][:10])
            rs = set(pred_ret[i][:10])
            if not hs:
                continue
            if not (hs <= ss) and not (hs <= rs):
                div += 1
        out[name] = float(div) / float(len(inds))
    return out


class HybridModel:
    """Hybrid sparse + dense fusion with a single loaded encoder and batched encoding (TASK-011)."""

    def __init__(
        self,
        *,
        sparse_model: dict[str, Any],
        sparse_classes: list[str],
        retrieval_index: faiss.Index,
        retrieval_meta: dict[str, Any],
        classes: list[str],
        encoder: Any,
        fusion_config: dict[str, Any],
        device: str,
        batch_size: int,
        sparse_batch_size: int,
        e5_prefix_query: str,
        normalize_embeddings: bool,
        sparse_artifacts_dir: str,
        retrieval_artifacts_dir: str,
        artifacts_dir: str,
    ) -> None:
        self.sparse_model = sparse_model
        self.sparse_classes = sparse_classes
        self.retrieval_index = retrieval_index
        self.retrieval_meta = retrieval_meta
        self.classes = classes
        self.encoder = encoder
        self.fusion_config = fusion_config
        self.device = device
        self.batch_size = batch_size
        self.sparse_batch_size = sparse_batch_size
        self.e5_prefix_query = e5_prefix_query
        self.normalize_embeddings = normalize_embeddings
        self.sparse_artifacts_dir = sparse_artifacts_dir
        self.retrieval_artifacts_dir = retrieval_artifacts_dir
        self.artifacts_dir = artifacts_dir

    @classmethod
    def load(
        cls,
        artifacts_dir: str | Path,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        sparse_batch_size: int = 256,
    ) -> HybridModel:
        """
        Load fusion config, sparse model, FAISS index, meta, and encoder once.

        R-001: ``normalize_embeddings`` and E5 prefixes come only from retrieval ``meta.json``.
        Default encode batch size uses retrieval helpers: 64 on CUDA and 16 on CPU when
        ``batch_size`` is omitted (via ``batch_size_gpu``/``batch_size_cpu`` in internal stub).

        Args:
            artifacts_dir: Hybrid run directory containing ``fusion_config.json``.
            device: ``cpu``, ``cuda``, ``auto``, or ``None`` (same as ``auto``).
            batch_size: Encoder batch size; default from ``_batch_size`` with GPU 64 / CPU 16.
            sparse_batch_size: Batch size for sparse ``decision_function`` / ``predict_proba``.
        """
        from sentence_transformers import SentenceTransformer

        from src.models import sparse as sparse_mod
        from src.models.retrieval import _batch_size, _resolve_device

        root = Path(artifacts_dir).resolve()
        with (root / "fusion_config.json").open("r", encoding="utf-8") as f:
            fc: dict[str, Any] = json.load(f)
        sparse_dir = str(fc.get("sparse_artifacts_dir", "")).replace("\\", "/")
        ret_dir = str(fc.get("retrieval_artifacts_dir", "")).replace("\\", "/")
        m_sparse = sparse_mod.load(sparse_dir)
        sparse_classes = list(m_sparse["classes"])
        index, meta, classes_r = load_retrieval_index_and_meta(ret_dir)
        assert_classes_r004(sparse_classes, classes_r)
        classes = sparse_classes

        import torch

        if device is None or str(device).lower() == "auto":
            resolved_device = _resolve_device({"retrieval": {"device": "auto"}})
        else:
            d = str(device).lower()
            if d == "cpu":
                resolved_device = "cpu"
            elif d == "cuda":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                raise ValueError(f"device must be cpu|cuda|auto|None, got {device!r}")

        ret_bs_stub: dict[str, Any] = {"batch_size_gpu": 64, "batch_size_cpu": 16}
        enc_bsz = (
            int(batch_size)
            if batch_size is not None
            else _batch_size(ret_bs_stub, resolved_device)
        )
        mname = str(meta.get("model_name", "intfloat/multilingual-e5-base"))
        norm_e = bool(meta["normalize_embeddings"])
        p_q = str(meta.get("e5_prefix_query", "query: "))
        enc = SentenceTransformer(mname, device=resolved_device)
        msl = 512
        try:
            for m in enc:
                m.max_seq_length = msl
        except Exception:
            pass

        ad = str(root).replace("\\", "/")
        return cls(
            sparse_model=m_sparse,
            sparse_classes=sparse_classes,
            retrieval_index=index,
            retrieval_meta=dict(meta),
            classes=classes,
            encoder=enc,
            fusion_config=fc,
            device=resolved_device,
            batch_size=enc_bsz,
            sparse_batch_size=int(sparse_batch_size),
            e5_prefix_query=p_q,
            normalize_embeddings=norm_e,
            sparse_artifacts_dir=sparse_dir.replace("\\", "/"),
            retrieval_artifacts_dir=ret_dir.replace("\\", "/"),
            artifacts_dir=ad,
        )

    def _predict_core(
        self,
        texts: list[str],
        emb: np.ndarray,
        k: int,
    ) -> tuple[list[list[str]], list[list[float]], list[float]]:
        fc = self.fusion_config
        mode = str(fc.get("mode", "weighted_score"))
        norm = str(fc.get("norm", "minmax"))
        candidates = int(fc.get("candidates", 50))
        lam = float(fc.get("best_lambda", 0.5))
        k_rrf = int(fc.get("k_rrf", 60))
        n_classes = len(self.classes)
        s_full = sparse_full_scores(
            self.sparse_model, texts, batch_size=self.sparse_batch_size
        )
        if s_full.shape[1] != n_classes:
            raise ValueError("sparse n_classes != retrieval n_classes")
        r_full = retrieval_full_scores(
            self.retrieval_index, n_classes, emb, k_search=n_classes
        )
        pred = hybrid_topk_from_full(
            s_full,
            r_full,
            lam,
            mode,
            norm,
            candidates,
            self.classes,
            k_out=k,
            k_rrf=k_rrf,
        )
        out_scores: list[list[float]] = []
        lats: list[float] = []
        for row in pred:
            t0 = time.perf_counter()
            k_eff = min(k, len(row))
            out_scores.append([float(k_eff - j) for j in range(k_eff)] if row else [])
            lats.append((time.perf_counter() - t0) * 1000.0)
        return pred, out_scores, lats

    def predict_topk(
        self,
        texts: list[str],
        k: int = 10,
    ) -> tuple[list[list[str]], list[list[float]], list[float]]:
        """Encode with batched ``encode_passages``, fuse, return same triple as free ``predict_topk``."""
        from src.models.retrieval import encode_passages

        texts_l = [str(t) for t in texts]
        emb = encode_passages(
            self.encoder,
            texts_l,
            self.e5_prefix_query,
            self.normalize_embeddings,
            self.batch_size,
        )
        return self._predict_core(texts_l, emb, k)

    def _predict_topk_with_emb(
        self,
        texts: list[str],
        k: int,
        precomputed_test_embeddings: np.ndarray,
    ) -> tuple[list[list[str]], list[list[float]], list[float]]:
        """Skip encoder; use precomputed query embeddings (same shape as ``encode_passages`` output)."""
        emb = np.asarray(precomputed_test_embeddings, dtype=np.float32)
        texts_l = [str(t) for t in texts]
        return self._predict_core(texts_l, emb, k)


def predict_topk(
    texts: list[str],
    artifacts_dir: str | Path,
    k: int = 10,
    *,
    precomputed_test_embeddings: np.ndarray | None = None,
) -> tuple[list[list[str]], list[list[float]], list[float]]:
    """
    Load sparse + retrieval artifacts, fuse using ``fusion_config.json`` (R-001: encoding uses
    ``normalize_embeddings`` from retrieval ``meta.json`` only, not CLI).
    """
    model = HybridModel.load(artifacts_dir)
    if precomputed_test_embeddings is None:
        return model.predict_topk(texts, k)
    return model._predict_topk_with_emb(texts, k, precomputed_test_embeddings)
