from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def predict_topk(
    model: dict[str, Any],
    texts: list[str],
    k: int = 10,
) -> list[list[str]]:
    """Return top-k topic IDs for each input text."""
    top_topics = [str(topic) for topic in model.get("top_topics", [])]
    if not top_topics:
        return [[] for _ in texts]
    k_eff = min(k, len(top_topics))
    topk = top_topics[:k_eff]
    return [list(topk) for _ in texts]


def fit(train_df: pd.DataFrame, topic_col: str = "topic_id") -> dict[str, Any]:
    """Fit MostFrequent baseline on train split."""
    if topic_col not in train_df.columns:
        raise KeyError(f"Missing topic column: {topic_col}")
    topic_series = train_df[topic_col].dropna().astype(str)
    counts = topic_series.value_counts()
    top_topics = sorted(counts.index.tolist(), key=lambda topic: (-int(counts[topic]), topic))
    topic_counts = {topic: int(counts[topic]) for topic in top_topics}
    return {"top_topics": top_topics, "topic_counts": topic_counts, "n_train": int(len(topic_series))}


def save(model: dict[str, Any], out_dir: str | Path) -> None:
    """Save baseline model artifacts to output directory."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "top_topics": list(model.get("top_topics", [])),
        "topic_counts": dict(model.get("topic_counts", {})),
        "n_train": int(model.get("n_train", 0)),
    }
    with (target / "model.pkl").open("wb") as stream:
        pickle.dump(model_payload, stream, protocol=4)

    with (target / "classes.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(model_payload["top_topics"], stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def train_baseline_most_frequent(config_path: str) -> str:
    """Train baseline model and persist artifacts directory path."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    train_path = config.get("data", {}).get("train_path")
    if not train_path:
        raise ValueError("config.data.train_path is required")

    train_df = pd.read_parquet(train_path)
    topic_col = "topic_id" if "topic_id" in train_df.columns else "theme"
    model = fit(train_df, topic_col=topic_col)

    out_dir = config.get("output", {}).get("artifacts_dir")
    if not out_dir:
        raise ValueError("config.output.artifacts_dir is required")
    save(model, out_dir)
    return str(out_dir)
