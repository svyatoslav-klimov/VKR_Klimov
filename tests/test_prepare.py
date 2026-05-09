from __future__ import annotations

import pandas as pd

from src.data.prepare import (
    load_taxonomy_aliases,
    prepare_split,
    theme_to_topic_id_map,
)


def test_theme_to_topic_id_map_stable_taxonomy_version() -> None:
    topics = "data/processed/topics.csv"
    _m1, v1 = theme_to_topic_id_map(topics, "configs/taxonomy.yaml")
    _m2, v2 = theme_to_topic_id_map(topics, "configs/taxonomy.yaml")
    assert v1 == v2
    assert v1.startswith("sha256:")
    assert len(_m1) > 0


def test_prepare_split_maps_topic_id_and_drops_nan() -> None:
    aliases = load_taxonomy_aliases("configs/taxonomy.yaml")
    mapping, _ver = theme_to_topic_id_map("data/processed/topics.csv", "configs/taxonomy.yaml")
    # Pick one known normalized name from topics (first row of topics.csv)
    tdf = pd.read_csv("data/processed/topics.csv", encoding="utf-8")
    first_name = str(tdf["topic_name"].iloc[0])
    from src.data.build_topics import normalize_topic

    key = normalize_topic(first_name)
    topic_id = mapping.get(key)
    assert topic_id is not None

    raw = pd.DataFrame(
        {
            "text": ["hello world " * 5, "short", "more text " * 10],
            "theme": [first_name, first_name, "___nonexistent_theme_xyz___"],
            "created_at_parsed": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    out = prepare_split(raw, mapping, aliases=aliases)
    assert "topic_id" in out.columns
    assert out["topic_id"].notna().all()
    assert len(out) < len(raw)
