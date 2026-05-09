"""Three-way late fusion: sparse (TF-IDF+SVC) + dense (E5 FAISS) + BM25 (TASK-040)."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

import faiss
import numpy as np

from src.models import hybrid as hyb
from src.models import sparse as sparse_mod

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

FusionMode = Literal[
    "weighted_score",
    "weighted_score_minmax",
    "weighted_score_zscore",
    "rrf",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_bm25_module_path() -> None:
    """Allow ``import bm25_index`` without ``import experiments`` (R-016.b)."""
    root = _repo_root()
    bm25_src = (root / "experiments" / "bm25_v1" / "src").resolve()
    if not bm25_src.is_dir():
        raise FileNotFoundError(f"BM25 sources not found: {bm25_src}")
    sp = str(bm25_src)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def load_bm25_index(bm25_dir: str | Path) -> Any:
    """Load BM25 index from canonical ``bm25_v1`` artifact directory."""
    _ensure_bm25_module_path()
    from bm25_index import BM25Index  # noqa: PLC0415

    return BM25Index.load(Path(bm25_dir))


def bm25_full_scores(bm25: Any, texts: list[str]) -> np.ndarray:
    """BM25 class scores, shape (n_queries, n_classes)."""
    _ensure_bm25_module_path()
    from bm25_index import tokenize_regex_unicode_lower  # noqa: PLC0415

    rows: list[np.ndarray] = []
    for t in texts:
        q = tokenize_regex_unicode_lower(str(t))
        rows.append(np.asarray(bm25._scores_per_class(q), dtype=np.float32))
    return np.vstack(rows).astype(np.float32, copy=False)


def normalize_per_query_row_three_way(
    x: np.ndarray,
    mode: str,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Match experiment 3-way normalization (minmax / zscore only)."""
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
    raise ValueError(f"Unknown norm mode for 3-way: {mode!r}")


def _ranks_1based_desc(s_row: np.ndarray) -> np.ndarray:
    k = s_row.size
    order = np.argsort(-s_row)
    r = np.empty(k, dtype=np.int32)
    for pos, j in enumerate(order):
        r[j] = pos + 1
    return r


def union_indices_two(s_row: np.ndarray, r_row: np.ndarray, candidates: int) -> list[int]:
    k = s_row.size
    n = min(candidates, k)
    is_ = set(np.argsort(-s_row)[:n].tolist())
    ir = set(np.argsort(-r_row)[:n].tolist())
    return sorted(is_ | ir)


def union_indices_three(
    s_row: np.ndarray,
    r_row: np.ndarray,
    b_row: np.ndarray,
    candidates: int,
) -> list[int]:
    k = s_row.size
    n = min(candidates, k)
    is_ = set(np.argsort(-s_row)[:n].tolist())
    ir = set(np.argsort(-r_row)[:n].tolist())
    ib = set(np.argsort(-b_row)[:n].tolist())
    return sorted(is_ | ir | ib)


def fuse_three_way_full(
    s_full: np.ndarray,
    r_full: np.ndarray,
    b_full: np.ndarray,
    *,
    lam_sparse: float,
    lam_dense: float,
    lam_bm25: float,
    fusion_mode: str,
    norm: str,
    candidates: int,
    classes: list[str],
    k_out: int = 10,
    k_rrf: int = 60,
) -> list[list[str]]:
    """Fuse three signals on the union of top-``candidates`` per signal (TASK-037 semantics)."""
    fm = str(fusion_mode)
    if fm == "weighted_score_zscore":
        norm_eff = "zscore"
    elif fm in ("weighted_score", "weighted_score_minmax"):
        norm_eff = str(norm)
    elif fm == "rrf":
        norm_eff = ""
    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")

    n = int(s_full.shape[0])
    if r_full.shape != s_full.shape or b_full.shape != s_full.shape:
        raise ValueError("s_full, r_full, b_full must share shape")
    k_eff = min(k_out, len(classes)) if classes else 0
    out: list[list[str]] = []

    lam_b = float(lam_bm25)
    for i in range(n):
        s_row = s_full[i]
        r_row = r_full[i]
        b_row = b_full[i]
        if fm != "rrf" and lam_b == 0.0:
            u = union_indices_two(s_row, r_row, candidates)
        else:
            u = union_indices_three(s_row, r_row, b_row, candidates)
        if not u or k_eff <= 0:
            out.append([])
            continue
        u_arr = np.asarray(u, dtype=np.int64)
        if fm == "rrf":
            rs_s = _ranks_1based_desc(s_row)
            rs_r = _ranks_1based_desc(r_row)
            rs_b = _ranks_1based_desc(b_row)
            h = np.array(
                [
                    1.0 / (k_rrf + float(rs_s[j]))
                    + 1.0 / (k_rrf + float(rs_r[j]))
                    + 1.0 / (k_rrf + float(rs_b[j]))
                    for j in u
                ],
                dtype=np.float64,
            )
        else:
            v_s = s_row[u_arr].astype(np.float64, copy=False)
            v_r = r_row[u_arr].astype(np.float64, copy=False)
            v_b = b_row[u_arr].astype(np.float64, copy=False)
            ns = normalize_per_query_row_three_way(v_s, norm_eff).astype(np.float64)
            nr = normalize_per_query_row_three_way(v_r, norm_eff).astype(np.float64)
            nb = normalize_per_query_row_three_way(v_b, norm_eff).astype(np.float64)
            h = (
                float(lam_sparse) * ns
                + float(lam_dense) * nr
                + float(lam_bm25) * nb
            )
        order_u = np.argsort(-h)
        picked: list[str] = []
        for oi in order_u:
            j = u[int(oi)]
            picked.append(classes[j])
            if len(picked) >= k_eff:
                break
        out.append(picked)
    return out


def fused_three_way_top10_scores_and_topics(
    s_full: np.ndarray,
    r_full: np.ndarray,
    b_full: np.ndarray,
    *,
    lam_sparse: float,
    lam_dense: float,
    lam_bm25: float,
    fusion_mode: str,
    norm: str,
    candidates: int,
    classes: list[str],
    k_out: int = 10,
    k_rrf: int = 60,
) -> tuple[np.ndarray, list[list[str]]]:
    """Top-``k_out`` fused scores (desc) and topic ids per row (prefix / calibration helpers)."""
    fm = str(fusion_mode)
    if fm == "weighted_score_zscore":
        norm_eff = "zscore"
    elif fm in ("weighted_score", "weighted_score_minmax"):
        norm_eff = str(norm)
    elif fm == "rrf":
        norm_eff = ""
    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")

    n = int(s_full.shape[0])
    k_eff = min(k_out, len(classes)) if classes else 0
    scores_out = np.zeros((n, k_eff), dtype=np.float64)
    topics_out: list[list[str]] = []

    lam_b = float(lam_bm25)
    for i in range(n):
        s_row = s_full[i]
        r_row = r_full[i]
        b_row = b_full[i]
        if fm != "rrf" and lam_b == 0.0:
            u = union_indices_two(s_row, r_row, candidates)
        else:
            u = union_indices_three(s_row, r_row, b_row, candidates)
        if not u or k_eff <= 0:
            topics_out.append([])
            continue
        u_arr = np.asarray(u, dtype=np.int64)
        if fm == "rrf":
            rs_s = _ranks_1based_desc(s_row)
            rs_r = _ranks_1based_desc(r_row)
            rs_b = _ranks_1based_desc(b_row)
            h = np.array(
                [
                    1.0 / (k_rrf + float(rs_s[j]))
                    + 1.0 / (k_rrf + float(rs_r[j]))
                    + 1.0 / (k_rrf + float(rs_b[j]))
                    for j in u
                ],
                dtype=np.float64,
            )
        else:
            v_s = s_row[u_arr].astype(np.float64, copy=False)
            v_r = r_row[u_arr].astype(np.float64, copy=False)
            v_b = b_row[u_arr].astype(np.float64, copy=False)
            ns = normalize_per_query_row_three_way(v_s, norm_eff).astype(np.float64)
            nr = normalize_per_query_row_three_way(v_r, norm_eff).astype(np.float64)
            nb = normalize_per_query_row_three_way(v_b, norm_eff).astype(np.float64)
            h = (
                float(lam_sparse) * ns
                + float(lam_dense) * nr
                + float(lam_bm25) * nb
            )
        order_u = np.argsort(-h)
        picked_scores: list[float] = []
        picked_topics: list[str] = []
        for oi in order_u:
            idx_u = int(oi)
            j = int(u[idx_u])
            picked_topics.append(classes[j])
            picked_scores.append(float(h[idx_u]))
            if len(picked_topics) >= k_eff:
                break
        pad = k_eff - len(picked_scores)
        if pad > 0:
            picked_scores.extend([0.0] * pad)
            picked_topics.extend([""] * pad)
        scores_out[i] = np.asarray(picked_scores[:k_eff], dtype=np.float64)
        topics_out.append(picked_topics[:k_eff])
    return scores_out, topics_out


def fuse_two_way_reference(
    s_full: np.ndarray,
    r_full: np.ndarray,
    *,
    lam_sparse: float,
    fusion_mode: str,
    norm: str,
    candidates: int,
    classes: list[str],
    k_out: int = 10,
    k_rrf: int = 60,
) -> list[list[str]]:
    """Two-way fusion (λ_bm25 = 0) matching :func:`hyb.hybrid_topk_from_full` semantics."""
    fm = str(fusion_mode)
    if fm == "weighted_score_minmax":
        mode_h = "weighted_score"
        norm_h = norm
    elif fm == "weighted_score_zscore":
        mode_h = "weighted_score"
        norm_h = "zscore"
    else:
        mode_h = fm
        norm_h = norm
    return hyb.hybrid_topk_from_full(
        s_full,
        r_full,
        float(lam_sparse),
        mode_h,
        norm_h,
        candidates,
        classes,
        k_out=k_out,
        k_rrf=k_rrf,
    )


def assert_three_way_classes_r004(
    sparse_c: list[str],
    retrieval_c: list[str],
    bm25_c: list[str],
) -> None:
    if not (sparse_c == retrieval_c == bm25_c):
        raise RuntimeError(
            "R-004: three_way requires sparse == retrieval == bm25 classes; mismatch."
        )


class ThreeWayHybrid:
    """Loaded 3-way fusion model: sparse + FAISS retrieval + BM25."""

    def __init__(
        self,
        *,
        sparse_model: dict[str, Any],
        sparse_classes: list[str],
        retrieval_index: faiss.Index,
        retrieval_meta: dict[str, Any],
        bm25: Any,
        bm25_classes: list[str],
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
        bm25_artifacts_dir: str,
        artifacts_dir: str,
    ) -> None:
        self.sparse_model = sparse_model
        self.sparse_classes = sparse_classes
        self.retrieval_index = retrieval_index
        self.retrieval_meta = retrieval_meta
        self.bm25 = bm25
        self.bm25_classes = bm25_classes
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
        self.bm25_artifacts_dir = bm25_artifacts_dir
        self.artifacts_dir = artifacts_dir

    def lam_sparse(self) -> float:
        return float(self.fusion_config.get("lambda_sparse", 0.1))

    def lam_dense(self) -> float:
        return float(self.fusion_config.get("lambda_dense", 0.7))

    def lam_bm25(self) -> float:
        return float(self.fusion_config.get("lambda_bm25", 0.2))

    def fusion_mode(self) -> str:
        return str(self.fusion_config.get("fusion_mode", "weighted_score_minmax"))

    def norm_mode(self) -> str:
        return str(self.fusion_config.get("norm", "minmax"))

    def candidates_n(self) -> int:
        return int(self.fusion_config.get("candidates", 30))

    def k_rrf(self) -> int:
        return int(self.fusion_config.get("k_rrf", 60))

    @classmethod
    def load(
        cls,
        artifacts_dir: str | Path,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        sparse_batch_size: int = 256,
    ) -> ThreeWayHybrid:
        """Load ``fusion_config.json`` and source models from disk."""
        from sentence_transformers import SentenceTransformer

        from src.models.retrieval import _batch_size, _resolve_device

        root = Path(artifacts_dir).resolve()
        with (root / "fusion_config.json").open("r", encoding="utf-8") as f:
            fc: dict[str, Any] = json.load(f)
        sparse_dir = str(fc.get("sparse_artifacts_dir", "")).replace("\\", "/")
        ret_dir = str(fc.get("retrieval_artifacts_dir", "")).replace("\\", "/")
        bm25_dir = str(fc.get("bm25_artifacts_dir", "")).replace("\\", "/")

        m_sparse = sparse_mod.load(sparse_dir)
        sparse_classes = list(m_sparse["classes"])
        index, meta, classes_r = hyb.load_retrieval_index_and_meta(ret_dir)
        hyb.assert_classes_r004(sparse_classes, classes_r)

        bm25 = load_bm25_index(bm25_dir)
        bm25_classes = list(getattr(bm25, "classes", []))
        assert_three_way_classes_r004(sparse_classes, classes_r, bm25_classes)
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
        logger.info(
            "ThreeWayHybrid loaded fusion_mode=%s device=%s",
            str(fc.get("fusion_mode", "")),
            resolved_device,
        )
        return cls(
            sparse_model=m_sparse,
            sparse_classes=sparse_classes,
            retrieval_index=index,
            retrieval_meta=dict(meta),
            bm25=bm25,
            bm25_classes=bm25_classes,
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
            bm25_artifacts_dir=bm25_dir.replace("\\", "/"),
            artifacts_dir=ad,
        )

    def _score_mats(
        self,
        texts: list[str],
        emb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_classes = len(self.classes)
        s_full = hyb.sparse_full_scores(
            self.sparse_model, texts, batch_size=self.sparse_batch_size
        )
        r_full = hyb.retrieval_full_scores(
            self.retrieval_index, n_classes, emb, k_search=n_classes
        )
        b_full = bm25_full_scores(self.bm25, texts)
        if s_full.shape[1] != n_classes or r_full.shape[1] != n_classes:
            raise ValueError("sparse/retrieval n_classes mismatch")
        if b_full.shape[1] != n_classes:
            raise ValueError("BM25 n_classes mismatch")
        return s_full, r_full, b_full

    def _predict_core(
        self,
        texts: list[str],
        emb: np.ndarray,
        k: int,
    ) -> tuple[list[list[str]], list[list[float]], list[float]]:
        s_full, r_full, b_full = self._score_mats(texts, emb)
        pred = fuse_three_way_full(
            s_full,
            r_full,
            b_full,
            lam_sparse=self.lam_sparse(),
            lam_dense=self.lam_dense(),
            lam_bm25=self.lam_bm25(),
            fusion_mode=self.fusion_mode(),
            norm=self.norm_mode(),
            candidates=self.candidates_n(),
            classes=self.classes,
            k_out=k,
            k_rrf=self.k_rrf(),
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
        """Encode queries, fuse, return (topk_ids, score proxies, row latencies ms)."""
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

    def predict_topk_precomputed(
        self,
        texts: list[str],
        query_embeddings: np.ndarray,
        k: int = 10,
    ) -> tuple[list[list[str]], list[list[float]], list[float]]:
        """Use precomputed query embeddings (e.g. ``embeddings_test.npy``)."""
        emb = np.asarray(query_embeddings, dtype=np.float32)
        texts_l = [str(t) for t in texts]
        return self._predict_core(texts_l, emb, k)

    def top1_scores_and_preds(
        self,
        texts: list[str],
        *,
        batch_rows: int = 64,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Per-row top-1 fusion score and predicted topic (calibration/TASK-020 style).

        Uses the same union + weighted fusion as :func:`fuse_three_way_full`.
        """
        from src.models.retrieval import encode_passages

        texts_l = [str(t) for t in texts]
        n = len(texts_l)
        scores_out = np.zeros(n, dtype=np.float64)
        preds: list[str] = []

        for start in range(0, n, batch_rows):
            chunk = texts_l[start : start + batch_rows]
            if precomputed_embeddings is None:
                emb = encode_passages(
                    self.encoder,
                    chunk,
                    self.e5_prefix_query,
                    self.normalize_embeddings,
                    self.batch_size,
                )
            else:
                pe = np.asarray(precomputed_embeddings, dtype=np.float32)
                emb = pe[start : start + batch_rows]
            s_full, r_full, b_full = self._score_mats(chunk, emb)
            sc, top = fused_three_way_top10_scores_and_topics(
                s_full,
                r_full,
                b_full,
                lam_sparse=self.lam_sparse(),
                lam_dense=self.lam_dense(),
                lam_bm25=self.lam_bm25(),
                fusion_mode=self.fusion_mode(),
                norm=self.norm_mode(),
                candidates=self.candidates_n(),
                classes=self.classes,
                k_out=1,
                k_rrf=self.k_rrf(),
            )
            for i in range(len(chunk)):
                if top[i] and top[i][0]:
                    scores_out[start + i] = float(sc[i, 0])
                    preds.append(top[i][0])
                else:
                    scores_out[start + i] = 0.0
                    preds.append("")
        return scores_out, preds


def predict_topk(
    texts: list[str],
    artifacts_dir: str | Path,
    k: int = 10,
    *,
    precomputed_test_embeddings: np.ndarray | None = None,
) -> tuple[list[list[str]], list[list[float]], list[float]]:
    """Stateful helper: load once via :class:`ThreeWayHybrid` then predict."""
    model = ThreeWayHybrid.load(artifacts_dir)
    if precomputed_test_embeddings is None:
        return model.predict_topk(texts, k)
    return model.predict_topk_precomputed(texts, precomputed_test_embeddings, k)
