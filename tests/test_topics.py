from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from src.data.build_topics import build_topics


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower().replace("ё", "е"))


def test_build_topics_is_deterministic_and_valid() -> None:
    config_path = "configs/data.yaml"
    topics_path, summary_path = build_topics(config_path=config_path)

    first_hash = hashlib.sha256(Path(topics_path).read_bytes()).hexdigest()
    second_topics_path, _ = build_topics(config_path=config_path)
    second_hash = hashlib.sha256(Path(second_topics_path).read_bytes()).hexdigest()
    assert first_hash == second_hash, "topics.csv should be byte-identical between runs"

    topics_df = pd.read_csv(topics_path, encoding="utf-8")
    assert len(topics_df) >= 200
    assert list(topics_df.columns) == ["topic_id", "topic_name", "count_total"]

    with Path(summary_path).open("r", encoding="utf-8") as stream:
        summary = json.load(stream)

    clean_df = pd.read_csv("data/interim/clean.csv", encoding="utf-8")
    raw = clean_df["тема"].dropna().astype(str)
    taxonomy_cfg = yaml.safe_load(Path("configs/taxonomy.yaml").read_text(encoding="utf-8")) or {}
    aliases_raw = taxonomy_cfg.get("aliases", {})
    aliases = {_norm(k): _norm(v) for k, v in aliases_raw.items()}
    mapped = [aliases.get(_norm(item), _norm(item)) for item in raw]
    unique_sorted = sorted(set(mapped))
    expected_taxonomy_version = "sha256:" + hashlib.sha256(
        "\n".join(unique_sorted).encode("utf-8")
    ).hexdigest()

    assert summary["taxonomy_version"] == expected_taxonomy_version
    assert summary["n_topics"] == len(unique_sorted)
