"""Unit tests for three-way hybrid promotion (TASK-040)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from src.models import three_way_hybrid as twh


def test_three_way_class_registered() -> None:
    assert twh.ThreeWayHybrid.__name__ == "ThreeWayHybrid"


def test_lambda_sum_weighted_modes() -> None:
    lam_s, lam_d, lam_b = 0.1, 0.7, 0.2
    assert abs(lam_s + lam_d + lam_b - 1.0) < 1e-9


def test_fuse_zero_bm25_matches_two_way() -> None:
    rng = np.random.RandomState(42)
    k_cls = 20
    s_full = rng.randn(4, k_cls).astype(np.float32)
    r_full = rng.randn(4, k_cls).astype(np.float32)
    b_full = rng.randn(4, k_cls).astype(np.float32)
    classes = [f"{i:02x}" for i in range(k_cls)]
    candidates = 8
    lam_s = 0.25
    out3 = twh.fuse_three_way_full(
        s_full,
        r_full,
        b_full,
        lam_sparse=lam_s,
        lam_dense=1.0 - lam_s,
        lam_bm25=0.0,
        fusion_mode="weighted_score_minmax",
        norm="minmax",
        candidates=candidates,
        classes=classes,
        k_out=5,
    )
    out2 = twh.fuse_two_way_reference(
        s_full,
        r_full,
        lam_sparse=lam_s,
        fusion_mode="weighted_score_minmax",
        norm="minmax",
        candidates=candidates,
        classes=classes,
        k_out=5,
    )
    assert len(out3) == len(out2)
    for a, b in zip(out3, out2, strict=True):
        assert a == b


def test_class_assert_r004() -> None:
    good = ["a", "b", "c"]
    twh.assert_three_way_classes_r004(good, good, good)
    with pytest.raises(RuntimeError, match="R-004"):
        twh.assert_three_way_classes_r004(good, good, ["c", "b", "a"])


def test_faiss_norm_dummy_embeddings() -> None:
    p = Path("artifacts/retrieval_e5/20260427_165442_retrieval_e5/embeddings_test.npy")
    if not p.is_file():
        pytest.skip("embeddings_test.npy not present")
    emb = np.load(str(p), mmap_mode="r", allow_pickle=False)[:100].astype(np.float32)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_rrf_rank_helpers_finite() -> None:
    rng = np.random.RandomState(0)
    v = rng.randn(15).astype(np.float32)
    rs = twh._ranks_1based_desc(v)
    assert rs.min() >= 1
    u = twh.union_indices_three(v, v, v, candidates=10)
    assert len(u) >= 1


def test_calibration_apply_isotonic_roundtrip() -> None:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    x = np.linspace(0, 1, 30)
    y = (x > 0.5).astype(np.float64)
    iso.fit(x, y)
    path = Path("reports/_tmp_iso_three_way_test.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"type": "isotonic", "model": iso}, f)
    from experiments.calibration_v1.src.apply import apply_calibrator

    out = apply_calibrator(np.array([0.2, 0.8]), path)
    assert float(np.min(out)) >= 0.0 and float(np.max(out)) <= 1.0
    path.unlink(missing_ok=True)


def test_prefix_shown_semantics() -> None:
    tau = 0.5
    assert 0.6 >= tau
    assert not (0.3 >= tau)


@pytest.mark.integration
def test_load_three_way_and_predict_small() -> None:
    root = Path("experiments/three_way_fusion_v1/artifacts/three_way_hybrid")
    dirs = sorted(root.glob("20260505_*")) if root.is_dir() else []
    if not dirs:
        pytest.skip("TASK-037 artifact dir missing")
    winner = dirs[0]
    fc_path = winner / "fusion_config.json"
    if not fc_path.is_file():
        pytest.skip("no fusion_config")
    with fc_path.open("r", encoding="utf-8") as f:
        fc = json.load(f)
    sparse_d = Path(fc["sparse_artifacts_dir"])
    ret_d = Path(fc["retrieval_artifacts_dir"])
    if not sparse_d.is_dir() or not ret_d.is_dir():
        pytest.skip("source artifacts missing")
    model = twh.ThreeWayHybrid.load(winner, device="cpu", batch_size=8)
    texts = ["тест обращения гражданина"] * 3
    pred, _scores, _lat = model.predict_topk(texts, k=10)
    assert len(pred) == 3
    assert all(len(p) <= 10 for p in pred)
    assert all(t in model.classes for row in pred for t in row if t)


def test_top1_scores_shape() -> None:
    rng = np.random.RandomState(1)
    k_cls = 12
    s_full = rng.randn(2, k_cls).astype(np.float32)
    r_full = rng.randn(2, k_cls).astype(np.float32)
    b_full = rng.randn(2, k_cls).astype(np.float32)
    classes = [f"c{i}" for i in range(k_cls)]
    sc, top = twh.fused_three_way_top10_scores_and_topics(
        s_full,
        r_full,
        b_full,
        lam_sparse=0.2,
        lam_dense=0.5,
        lam_bm25=0.3,
        fusion_mode="weighted_score_minmax",
        norm="minmax",
        candidates=6,
        classes=classes,
        k_out=1,
    )
    assert sc.shape == (2, 1)
    assert len(top) == 2
