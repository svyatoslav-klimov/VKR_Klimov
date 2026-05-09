from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.prepare import (
    filter_by_preprocessed_text,
    load_taxonomy_aliases,
    prepare_split,
    preprocess_text_padded,
    theme_to_topic_id_map,
)
from src.eval.metrics import recall_at_k
from src.models import hybrid as hyb

REPO_ROOT = Path(__file__).resolve().parents[1]
BEST_HYB_DIR = REPO_ROOT / "artifacts" / "hybrid" / "20260427_171719_hybrid_weighted_score_minmax"
GRID_HYB_YAML = REPO_ROOT / "configs" / "_grid" / "hybrid_30_minmax.yaml"
REF_R10 = 0.8161153519932146


def _grid_train_test_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align with ``train_hybrid_cmd`` preprocessing for the grid config that produced the ref run."""
    with GRID_HYB_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    data_cfg = cfg["data"]
    train_path = REPO_ROOT / data_cfg["train_path"]
    test_path = REPO_ROOT / data_cfg["test_path"]
    topics_path = REPO_ROOT / data_cfg["topics_path"]
    taxonomy_yaml = REPO_ROOT / data_cfg.get("taxonomy_yaml", "configs/taxonomy.yaml")
    preproc = cfg.get("preprocessing")
    pre_d = preproc if isinstance(preproc, dict) else None

    aliases = load_taxonomy_aliases(str(taxonomy_yaml))
    topics_map, _ = theme_to_topic_id_map(str(topics_path), str(taxonomy_yaml))
    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)
    train_df0 = prepare_split(train_raw, topics_map, aliases, str(taxonomy_yaml))
    test_df0 = prepare_split(test_raw, topics_map, aliases, str(taxonomy_yaml))
    if pre_d:
        train_df = filter_by_preprocessed_text(train_df0, pre_d)
    else:
        train_df = train_df0
    test_df = test_df0.copy()
    test_df["text"] = [
        preprocess_text_padded(t, pre_d or {}) for t in test_df0["text"].astype(str).tolist()
    ]
    return train_df, test_df


def test_normalize_per_query_modes():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(hyb.normalize_per_query_row(x, "minmax"), np.array([0.0, 0.5, 1.0]))
    z = hyb.normalize_per_query_row(x, "zscore")
    assert float(z.min()) >= 0.0 and float(z.max()) <= 1.0
    sm = hyb.normalize_per_query_row(x, "softmax")
    assert abs(float(np.sum(sm)) - 1.0) < 1e-5
    rk = hyb.normalize_per_query_row(x, "rank")
    assert len(rk) == 3


def test_normalize_minmax_constant():
    x = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    out = hyb.normalize_per_query_row(x, "minmax")
    assert np.allclose(out, 0.5)


def test_fuse_weighted_score_lambda_extremes():
    classes = ["a", "b", "c", "d"]
    s = np.array([[0.0, 0.1, 0.9, 0.0]], dtype=np.float32)
    r = np.array([[0.0, 0.8, 0.1, 0.0]], dtype=np.float32)
    out0 = hyb.hybrid_topk_from_full(s, r, 0.0, "weighted_score", "minmax", 3, classes, k_out=1)
    out1 = hyb.hybrid_topk_from_full(s, r, 1.0, "weighted_score", "minmax", 3, classes, k_out=1)
    out05 = hyb.hybrid_topk_from_full(s, r, 0.5, "weighted_score", "minmax", 3, classes, k_out=1)
    assert out0[0][0] == "b"  # retrieval
    assert out1[0][0] == "c"  # sparse
    assert len(out05[0]) == 1


def test_fuse_rrf_matches_formula():
    classes = [str(i) for i in range(4)]
    s = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    r = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    krrf = 60
    out = hyb.hybrid_topk_from_full(
        s, r, 0.0, "rrf", "minmax", 2, classes, k_out=2, k_rrf=krrf
    )
    assert len(out) == 1 and len(out[0]) >= 1
    rs = hyb._ranks_1based_desc(s[0])
    rr = hyb._ranks_1based_desc(r[0])
    u = set(np.argsort(-s[0])[:2].tolist()) | set(np.argsort(-r[0])[:2].tolist())
    scores: dict[str, float] = {}
    for j in u:
        scores[classes[j]] = 1.0 / (krrf + float(rs[j])) + 1.0 / (krrf + float(rr[j]))
    best = max(scores, key=scores.get)
    assert out[0][0] == best


def test_fuse_max_mode():
    classes = [str(i) for i in range(3)]
    s = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)
    r = np.array([[0.0, 0.2, 0.0]], dtype=np.float32)
    o = hyb.hybrid_topk_from_full(s, r, 0.0, "max", "minmax", 3, classes, k_out=1)
    assert o[0][0] in ("0", "1")


def test_calibrate_lambda_synthetic():
    # 2 classes, val always 0
    n = 20
    K = 2
    s = np.random.RandomState(42).rand(n, K).astype(np.float32)
    r = np.random.RandomState(1).rand(n, K).astype(np.float32)
    y_val = [str(int(i % 2)) for i in range(n)]
    classes = ["0", "1"]
    g = [0.0, 0.5, 1.0]
    bl, _grid, _m = hyb.calibrate_lambda(s, r, y_val, g, "weighted_score", "minmax", 10, classes, k=10)
    assert 0.0 <= bl <= 1.0


def test_r004_mismatch():
    with pytest.raises(RuntimeError, match="R-004"):
        hyb.assert_classes_r004(["1", "2"], ["2", "1"])


def test_weighted_score_lambda_1_matches_sparse_top10():
    n, k, cand = 6, 12, 50
    rng = np.random.RandomState(0)
    s = (rng.randn(n, k) * 1.5).astype(np.float32)
    r = (rng.randn(n, k) * 1.2).astype(np.float32)
    classes = [str(i) for i in range(k)]
    pred = hyb.hybrid_topk_from_full(
        s, r, 1.0, "weighted_score", "minmax", cand, classes, k_out=10
    )
    for i in range(n):
        want = [classes[j] for j in np.argsort(-s[i])[:10].tolist()]
        assert pred[i][:10] == want


def test_pearson_intersection_smoke():
    n, k, c = 50, 50, 20
    rng = np.random.RandomState(1)
    s = rng.randn(n, k).astype(np.float32)
    r = 0.3 * s + 0.7 * rng.randn(n, k).astype(np.float32)
    cl = [str(i) for i in range(k)]
    rho = hyb.pearson_sparse_retrieval_val(s, r, cl, "raw", c, mode="intersection")
    assert -1.0 <= rho <= 1.0
    rf = hyb.pearson_sparse_retrieval_val(s, r, cl, "raw", c, mode="full_row")
    assert -1.0 <= rf <= 1.0


def test_hybrid_diagnostics_synthetic_basics():
    n = 100
    K = 5
    rs = np.random.RandomState(0)
    s = rs.randn(n, K).astype(np.float32)
    r = rs.randn(n, K).astype(np.float32)
    y = [str(i % K) for i in range(n)]
    classes = [str(i) for i in range(K)]
    rho = hyb.pearson_sparse_retrieval_val(s, r, classes, "minmax", 20, mode="intersection")
    assert -1.0 <= rho <= 1.0
    best_lam, _ = hyb.calibrate_lambda_weighted(
        s, r, y, classes, [0.0, 0.5, 1.0], "minmax", 20, k=10
    )
    t1 = hyb.top1_change_rate_val(s, r, best_lam, classes, "minmax", 20, "weighted_score")
    assert 0.0 <= t1 <= 1.0
    pdiv = hyb.per_bucket_diversity(
        ["0", "0", "0"],
        [["0", "1"], ["0", "1"], ["0", "1"]],
        [["0", "2"], ["0", "2"], ["0", "2"]],
        [["1", "2"], ["1", "2"], ["1", "2"]],
        {"0": 100},
    )
    assert "head" in pdiv and pdiv["head"] >= 0.0


@pytest.mark.skipif(not BEST_HYB_DIR.is_dir(), reason="best hybrid artifacts dir missing")
def test_hybridmodel_load_smoke():
    m = hyb.HybridModel.load(BEST_HYB_DIR)
    assert len(m.classes) > 0
    assert m.encoder is not None
    assert int(m.retrieval_index.ntotal) == len(m.classes)


@pytest.mark.skipif(not BEST_HYB_DIR.is_dir(), reason="best hybrid artifacts dir missing")
def test_predict_topk_returns_only_known_classes():
    train_df, _ = _grid_train_test_frames()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(train_df), size=min(50, len(train_df)), replace=False)
    texts = train_df.iloc[idx]["text"].astype(str).tolist()
    allowed = set(hyb.load_json_classes(BEST_HYB_DIR))
    model = hyb.HybridModel.load(BEST_HYB_DIR)
    topk_ids, _, _ = model.predict_topk(texts, k=10)
    for row in topk_ids:
        for tid in row:
            assert tid in allowed


@pytest.mark.slow
@pytest.mark.skipif(not BEST_HYB_DIR.is_dir(), reason="best hybrid artifacts dir missing")
def test_predict_topk_bit_exact_with_train_metrics():
    _, test_df = _grid_train_test_frames()
    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["topic_id"].astype(str).tolist()
    model = hyb.HybridModel.load(BEST_HYB_DIR)
    pred, _, _ = model.predict_topk(texts, k=10)
    r10 = float(recall_at_k(y_true, pred, k_list=(10,))["10"])
    assert abs(r10 - REF_R10) <= 1e-10


@pytest.mark.skipif(not BEST_HYB_DIR.is_dir(), reason="best hybrid artifacts dir missing")
def test_predict_topk_batched_speedup():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    train_df, _ = _grid_train_test_frames()
    rng = np.random.RandomState(0)
    n = 200
    texts = train_df.iloc[rng.choice(len(train_df), size=n, replace=True)]["text"].astype(str).tolist()

    model_b = hyb.HybridModel.load(BEST_HYB_DIR, device="cuda", batch_size=64)
    t0 = time.perf_counter()
    model_b.predict_topk(texts, k=10)
    t_b = time.perf_counter() - t0

    model_p = hyb.HybridModel.load(BEST_HYB_DIR, device="cuda", batch_size=1)
    t1 = time.perf_counter()
    model_p.predict_topk(texts, k=10)
    t_p = time.perf_counter() - t1

    assert t_b < 0.6 * t_p

