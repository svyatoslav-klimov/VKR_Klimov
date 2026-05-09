from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def recall_at_k(
    y_true: list[str],
    y_pred_topk: list[list[str]],
    k_list: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute Recall@k for each k in k_list."""
    if len(y_true) != len(y_pred_topk):
        raise ValueError("y_true and y_pred_topk must have equal length")
    if not y_true:
        return {str(k): 0.0 for k in k_list}

    result: dict[str, float] = {}
    for k in k_list:
        hits = 0
        for truth, preds in zip(y_true, y_pred_topk):
            if truth in preds[:k]:
                hits += 1
        result[str(k)] = float(hits / len(y_true))
    return result


def _top1(y_pred_topk: list[list[str]]) -> list[str]:
    return [preds[0] if preds else "" for preds in y_pred_topk]


def macro_f1_top1(y_true: list[str], y_pred_topk: list[list[str]], labels: list[str]) -> float:
    """Compute macro F1 on top-1 predictions."""
    if not y_true:
        return 0.0
    return float(
        f1_score(
            y_true,
            _top1(y_pred_topk),
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def weighted_f1_top1(y_true: list[str], y_pred_topk: list[list[str]], labels: list[str]) -> float:
    """Compute weighted F1 on top-1 predictions."""
    if not y_true:
        return 0.0
    return float(
        f1_score(
            y_true,
            _top1(y_pred_topk),
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )


def accuracy_top1(y_true: list[str], y_pred_topk: list[list[str]]) -> float:
    """Compute top-1 accuracy."""
    if not y_true:
        return 0.0
    return float(accuracy_score(y_true, _top1(y_pred_topk)))


def mrr_at_k(y_true: list[str], y_pred_topk: list[list[str]], k: int = 10) -> float:
    """Compute MRR@k for single relevant label per row."""
    if not y_true:
        return 0.0
    rr_sum = 0.0
    for truth, preds in zip(y_true, y_pred_topk):
        reciprocal = 0.0
        for rank, pred in enumerate(preds[:k], start=1):
            if pred == truth:
                reciprocal = 1.0 / rank
                break
        rr_sum += reciprocal
    return float(rr_sum / len(y_true))


def ndcg_at_k(y_true: list[str], y_pred_topk: list[list[str]], k: int = 10) -> float:
    """Compute nDCG@k with one relevant document per row."""
    if not y_true:
        return 0.0
    score = 0.0
    for truth, preds in zip(y_true, y_pred_topk):
        dcg = 0.0
        for rank, pred in enumerate(preds[:k], start=1):
            if pred == truth:
                dcg = 1.0 / np.log2(rank + 1)
                break
        score += dcg
    return float(score / len(y_true))


def head_mid_tail_recall(
    y_true: list[str],
    y_pred_topk: list[list[str]],
    topic_counts: dict[str, int],
    mode: str = "practical",
    head_min: int = 50,
    mid_min: int = 10,
) -> dict[str, object]:
    """Compute head/mid/tail slice recall@10 and macro_f1."""
    if mode == "strict":
        head_min, mid_min = 500, 50

    buckets = {"head": [], "mid": [], "tail": []}
    for idx, topic_id in enumerate(y_true):
        count = topic_counts.get(topic_id, 0)
        if count >= head_min:
            buckets["head"].append(idx)
        elif count >= mid_min:
            buckets["mid"].append(idx)
        else:
            buckets["tail"].append(idx)

    def _slice_metrics(indices: list[int]) -> dict[str, float | int]:
        if not indices:
            return {"n": 0, "recall@10": 0.0, "macro_f1": 0.0}
        truth = [y_true[i] for i in indices]
        preds = [y_pred_topk[i] for i in indices]
        labels = sorted(set(truth))
        return {
            "n": len(indices),
            "recall@10": recall_at_k(truth, preds, k_list=(10,))["10"],
            "macro_f1": macro_f1_top1(truth, preds, labels=labels),
        }

    return {
        "mode": mode,
        "thresholds": {"head_min": head_min, "mid_min": mid_min},
        "head": _slice_metrics(buckets["head"]),
        "mid": _slice_metrics(buckets["mid"]),
        "tail": _slice_metrics(buckets["tail"]),
    }


def time_slice_recall(
    y_true: list[str],
    y_pred_topk: list[list[str]],
    created_at: Iterable[object],
    n_buckets: int = 4,
) -> list[dict[str, object]]:
    """Compute Recall@10 across chronological buckets."""
    timestamps = pd.to_datetime(list(created_at), errors="coerce")
    if len(y_true) == 0 or timestamps.isna().all():
        return []

    frame = pd.DataFrame(
        {
            "idx": list(range(len(y_true))),
            "created_at": timestamps,
            "y_true": y_true,
        }
    ).dropna(subset=["created_at"])

    frame = frame.sort_values("created_at").reset_index(drop=True)
    bucket_ids = np.array_split(frame.index.to_numpy(), min(n_buckets, len(frame)))

    output: list[dict[str, object]] = []
    for bucket_no, bucket_idx in enumerate(bucket_ids, start=1):
        if len(bucket_idx) == 0:
            continue
        row_ids = frame.loc[bucket_idx, "idx"].tolist()
        truth = [y_true[i] for i in row_ids]
        preds = [y_pred_topk[i] for i in row_ids]
        output.append(
            {
                "slice": f"bucket_{bucket_no}",
                "n": len(row_ids),
                "recall@10": recall_at_k(truth, preds, k_list=(10,))["10"],
            }
        )
    return output


# Backward-compatible aliases from TASK-001 stubs.
def accuracy_at_1(topk_ids: list[list[str]], true_ids: list[str]) -> float:
    return accuracy_top1(true_ids, topk_ids)


def macro_f1(top1_ids: list[str], true_ids: list[str], labels: list[str]) -> float:
    preds = [[pred] for pred in top1_ids]
    return macro_f1_top1(true_ids, preds, labels)


def weighted_f1(top1_ids: list[str], true_ids: list[str], labels: list[str]) -> float:
    preds = [[pred] for pred in top1_ids]
    return weighted_f1_top1(true_ids, preds, labels)


def mrr_at_10(topk_ids: list[list[str]], true_ids: list[str]) -> float:
    return mrr_at_k(true_ids, topk_ids, k=10)


def ndcg_at_10(topk_ids: list[list[str]], true_ids: list[str]) -> float:
    return ndcg_at_k(true_ids, topk_ids, k=10)
