from __future__ import annotations

from src.eval import metrics


def test_recall_metrics_monotonic() -> None:
    y_true = ["a", "b", "c"]
    y_pred = [["a", "x", "y"], ["x", "b", "z"], ["z", "y", "c"]]
    result = metrics.recall_at_k(y_true, y_pred, k_list=(1, 2, 3))
    assert result["1"] <= result["2"] <= result["3"]


def test_f1_accuracy_mrr_ndcg_non_negative() -> None:
    y_true = ["a", "b", "c"]
    y_pred = [["a", "x", "y"], ["x", "b", "z"], ["z", "y", "c"]]
    labels = ["a", "b", "c", "x", "y", "z"]

    assert 0.0 <= metrics.accuracy_top1(y_true, y_pred) <= 1.0
    assert 0.0 <= metrics.macro_f1_top1(y_true, y_pred, labels=labels) <= 1.0
    assert 0.0 <= metrics.weighted_f1_top1(y_true, y_pred, labels=labels) <= 1.0
    assert 0.0 <= metrics.mrr_at_k(y_true, y_pred, k=3) <= 1.0
    assert 0.0 <= metrics.ndcg_at_k(y_true, y_pred, k=3) <= 1.0


def test_head_mid_tail_and_time_slice_blocks() -> None:
    y_true = ["a", "b", "c", "d"]
    y_pred = [["a", "b", "c"], ["b", "a", "c"], ["d", "c", "b"], ["d", "c", "a"]]
    topic_counts = {"a": 100, "b": 60, "c": 20, "d": 1}

    practical = metrics.head_mid_tail_recall(
        y_true=y_true,
        y_pred_topk=y_pred,
        topic_counts=topic_counts,
        mode="practical",
    )
    strict = metrics.head_mid_tail_recall(
        y_true=y_true,
        y_pred_topk=y_pred,
        topic_counts=topic_counts,
        mode="strict",
    )

    assert "head" in practical and "mid" in practical and "tail" in practical
    assert strict["thresholds"]["head_min"] == 500
    assert strict["thresholds"]["mid_min"] == 50

    slices = metrics.time_slice_recall(
        y_true=y_true,
        y_pred_topk=y_pred,
        created_at=[
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
            "2024-04-01",
        ],
        n_buckets=2,
    )
    assert len(slices) == 2
    assert all("recall@10" in row for row in slices)
