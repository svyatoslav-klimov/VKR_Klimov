"""Smoke tests for tools/error_analysis.py helpers."""

from __future__ import annotations

import pandas as pd

from tools.error_analysis import _per_class_recall


def test_per_class_recall_known_hits_only() -> None:
    df = pd.DataFrame(
        {
            "topic_id": ["a", "a", "b", "b"],
            "text": ["x", "yy", "z", "w"],
        }
    )
    topk = [["a", "c"], ["x"], ["b"], ["b", "a"]]
    train_counts = {"a": 100, "b": 200}
    out = _per_class_recall(df, topk, k_recall=2, train_topic_counts=train_counts)
    out_map = out.set_index("topic_id").to_dict("index")

    assert out_map["a"]["n_test_class"] == 2
    assert out_map["a"]["n_hits"] == 1
    assert abs(out_map["a"]["recall_at_10_test"] - 0.5) < 1e-9
    assert out_map["a"]["n_train_class"] == 100

    assert out_map["b"]["n_test_class"] == 2
    assert out_map["b"]["n_hits"] == 2
    assert abs(out_map["b"]["recall_at_10_test"] - 1.0) < 1e-9
    assert out_map["b"]["n_train_class"] == 200

