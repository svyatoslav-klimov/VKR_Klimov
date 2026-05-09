from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.models import retrieval as rr


def test_build_centroids_l2_normalized() -> None:
    dim = 4
    emb = np.eye(dim, dtype=np.float32)
    topics = ["b", "a", "a", "b"]
    c, classes = rr.build_centroids(emb, topics)
    assert classes == ["a", "b"]
    assert c.shape == (2, dim)
    for i in range(2):
        n = float(np.linalg.norm(c[i]))
        assert abs(n - 1.0) < 1e-5


def test_build_index_ntotal() -> None:
    x = np.random.default_rng(42).random((5, 8)).astype(np.float32)
    idx = rr.build_index(x)
    assert idx.ntotal == 5


def test_search_and_scores_desc() -> None:
    dim = 8
    rng = np.random.default_rng(0)
    cent = rng.random((3, dim)).astype(np.float32)
    cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    index = rr.build_index(cent)
    classes = ["t1", "t2", "t3"]
    q = rng.random((1, dim)).astype(np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    ids, scores = rr.search_with_query_embeddings(index, classes, q, k=3)
    assert len(ids[0]) == 3
    assert scores[0] == sorted(scores[0], reverse=True)


def test_save_roundtrip_predict_topk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dim = 6
    n_c = 3
    rng = np.random.default_rng(1)
    cent = rng.random((n_c, dim)).astype(np.float32)
    cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    index = rr.build_index(cent)
    classes = ["c1", "c2", "c3"]
    emb_tr = rng.random((10, dim)).astype(np.float32)
    emb_te = rng.random((4, dim)).astype(np.float32)
    meta = {
        "model_name": "fake-model",
        "normalize_embeddings": True,
        "faiss_metric": "IP",
        "embedding_dim": dim,
        "index_ntotal": n_c,
        "representation": "centroid",
        "classes_path": "classes.json",
        "e5_prefix_query": "query: ",
        "e5_prefix_passage": "passage: ",
        "encoder_revision": "unknown",
    }
    result = {
        "classes": classes,
        "index": index,
        "embeddings_train": emb_tr,
        "embeddings_test": emb_te,
        "meta_file": meta,
    }
    rr.save_retrieval(tmp_path, result)
    assert (tmp_path / "themes.faiss").is_file()
    with (tmp_path / "meta.json").open("r", encoding="utf-8") as f:
        on_disk = json.load(f)
    for k in (
        "model_name",
        "normalize_embeddings",
        "faiss_metric",
        "embedding_dim",
        "index_ntotal",
        "representation",
        "classes_path",
        "e5_prefix_query",
        "e5_prefix_passage",
    ):
        assert k in on_disk

    calls: list[dict[str, object]] = []

    class FakeEnc:
        def encode(self, texts, batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
            calls.append({"norm": normalize_embeddings, "n": len(texts)})
            return np.tile(
                np.arange(dim, dtype=np.float32) / float(dim),
                (len(texts), 1),
            )

    monkeypatch.setattr(rr, "SentenceTransformer", lambda *a, **k: FakeEnc())
    topk, sc, _ = rr.predict_topk(["hello", "world"], tmp_path, k=2)
    assert len(topk) == 2
    assert calls and calls[0]["norm"] is True
    assert sc[0] == sorted(sc[0], reverse=True)


def test_meta_write_keys(tmp_path: Path) -> None:
    rr.write_meta(
        tmp_path,
        encoder_name="m",
        encoder_revision="r",
        embedding_dim=8,
        index_ntotal=2,
        classes_path="classes.json",
        prefix_query="query: ",
        prefix_passage="passage: ",
        normalize_embeddings=True,
        faiss_metric="IP",
        representation="centroid",
    )
    with (tmp_path / "meta.json").open("r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["e5_prefix_query"] == "query: "
    assert m["e5_prefix_passage"] == "passage: "
    assert m["normalize_embeddings"] is True
