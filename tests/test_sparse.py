from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import src.models.sparse as sparse


def _synth_frame(n_rows: int, n_classes: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = [f"cid_{i}" for i in range(n_classes)]
    row_n = np.arange(n_rows)
    topics = [classes[i % n_classes] for i in row_n]
    texts = [
        f"uniqtoken{topics[i]} repeated " * 8 + f"noise {rng.integers(0, 10000)}" for i in row_n
    ]
    return pd.DataFrame({"text": texts, "topic_id": topics})


def test_fit_predict_synthetic_recall() -> None:
    train = _synth_frame(200, 10, seed=42)
    cfg = {
        "seed": 42,
        "model": {
            "classifier": "linear_svc",
            "max_features": 5000,
            "min_df": 1,
            "C": 1.0,
            "max_iter": 2000,
        },
    }
    model = sparse.fit(train, cfg, text_col="text", topic_col="topic_id")
    texts = train["text"].astype(str).tolist()
    topk, scores, _lat = sparse.predict_topk(model, texts, k=10)
    y_true = train["topic_id"].astype(str).tolist()
    hits1 = sum(1 for t, p in zip(y_true, topk) if t in p[:1])
    hits10 = sum(1 for t, p in zip(y_true, topk) if t in p[:10])
    assert hits1 / len(y_true) >= 0.5
    assert hits10 / len(y_true) >= 0.95


def test_predict_topk_shape_and_score_order() -> None:
    train = _synth_frame(80, 5, seed=0)
    cfg = {"seed": 42, "model": {"classifier": "linear_svc", "max_features": 2000, "min_df": 1, "max_iter": 1000}}
    model = sparse.fit(train, cfg, text_col="text", topic_col="topic_id")
    topk, scores, _ = sparse.predict_topk(model, ["class cid_0 token"], k=4)
    assert len(topk[0]) == 4
    s = scores[0]
    assert s == sorted(s, reverse=True)


def test_save_load_roundtrip() -> None:
    train = _synth_frame(60, 4, seed=1)
    cfg = {"seed": 42, "model": {"classifier": "linear_svc", "max_features": 1000, "min_df": 1, "max_iter": 500}}
    model = sparse.fit(train, cfg, text_col="text", topic_col="topic_id")
    out = Path("reports") / "tmp_sparse_rt"
    out.mkdir(parents=True, exist_ok=True)
    sparse.save(model, out)
    m2 = sparse.load(out)
    t = ["a b c class cid_0", "d e f class cid_1"]
    p1, _, _ = sparse.predict_topk(model, t, k=3)
    p2, _, _ = sparse.predict_topk(m2, t, k=3)
    assert p1 == p2
    for f in out.glob("*"):
        f.unlink()
    out.rmdir()


def test_logreg_and_linear_svc_smoke() -> None:
    train = _synth_frame(100, 6, seed=2)
    for clf in ("linear_svc", "logreg"):
        cfg = {"seed": 42, "model": {"classifier": clf, "max_features": 2000, "min_df": 1, "max_iter": 2000}}
        model = sparse.fit(train, cfg, text_col="text", topic_col="topic_id")
        pred, sc, _ = sparse.predict_topk(model, [train["text"].iloc[0]], k=5)
        assert len(pred[0]) == 5
        assert sc[0] == sorted(sc[0], reverse=True)


def test_model_pkl_determinism() -> None:
    train = _synth_frame(120, 8, seed=3)
    cfg = {"seed": 42, "model": {"classifier": "linear_svc", "max_features": 2000, "min_df": 1, "max_iter": 1500}}
    out1 = Path("reports") / "tmp_sp_d1"
    out2 = Path("reports") / "tmp_sp_d2"
    for p in (out1, out2):
        p.mkdir(parents=True, exist_ok=True)
    m1 = sparse.fit(train, cfg, text_col="text", topic_col="topic_id")
    m2 = sparse.fit(train, cfg, text_col="text", topic_col="topic_id")
    sparse.save(m1, out1)
    sparse.save(m2, out2)
    h1 = hashlib.sha256((out1 / "model.pkl").read_bytes()).hexdigest()
    h2 = hashlib.sha256((out2 / "model.pkl").read_bytes()).hexdigest()
    assert h1 == h2
    for p in (out1, out2):
        for f in p.glob("*"):
            f.unlink()
        p.rmdir()
