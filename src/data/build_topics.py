from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def normalize_topic(text: str) -> str:
    """Normalize topic string for deterministic taxonomy generation."""
    return re.sub(r"\s+", " ", str(text).strip().lower().replace("ё", "е"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure: {path}")
    return data


def _resolve_column_name(df: pd.DataFrame, configured_name: str) -> str:
    if configured_name in df.columns:
        return configured_name

    fallbacks = ("тема", "theme", "topic_id")
    for candidate in fallbacks:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"Topic column not found. Expected '{configured_name}' or one of {fallbacks}. "
        f"Available columns: {list(df.columns)}"
    )


def build_topics(config_path: str) -> tuple[Path, Path]:
    """Build deterministic topics.csv and summary JSON using SPEC §4.2.1."""
    config = _load_yaml(Path(config_path))

    paths_cfg = config.get("paths", {})
    columns_cfg = config.get("columns", {})
    taxonomy_cfg = config.get("taxonomy", {})
    reports_cfg = config.get("reports", {})

    clean_csv_path = Path(paths_cfg.get("clean_csv", "data/interim/clean.csv"))
    topics_csv_path = Path(paths_cfg.get("topics_csv", "data/processed/topics.csv"))
    aliases_yaml_path = Path(taxonomy_cfg.get("aliases_path", "configs/taxonomy.yaml"))
    summary_path = Path(reports_cfg.get("topics_summary", "reports/eda/topics_summary.json"))
    topic_column_name = str(columns_cfg.get("topic", "тема"))

    try:
        df = pd.read_csv(clean_csv_path, encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "BLOCKER FOR ROLE_01: data/interim/clean.csv is not UTF-8 encoded "
            "(R-007). Refusing implicit recoding."
        ) from exc

    raw_column = _resolve_column_name(df, topic_column_name)
    raw_topics = df[raw_column].dropna().astype(str)

    taxonomy_data = _load_yaml(aliases_yaml_path)
    aliases_raw = taxonomy_data.get("aliases", {})
    if not isinstance(aliases_raw, dict):
        raise ValueError(f"Expected mapping in taxonomy aliases: {aliases_yaml_path}")

    aliases = {
        normalize_topic(alias): normalize_topic(canonical)
        for alias, canonical in aliases_raw.items()
    }

    normalized_topics = [normalize_topic(topic) for topic in raw_topics]
    mapped_topics = [aliases.get(topic, topic) for topic in normalized_topics]

    counts = Counter(mapped_topics)
    unique_sorted = sorted(counts.keys())

    topics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    topics_rows = [
        {
            "topic_id": hashlib.sha1(name.encode("utf-8")).hexdigest()[:8],
            "topic_name": name,
            "count_total": int(counts[name]),
        }
        for name in unique_sorted
    ]
    topics_df = pd.DataFrame(topics_rows, columns=["topic_id", "topic_name", "count_total"])
    topics_df.to_csv(
        topics_csv_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    taxonomy_version = "sha256:" + hashlib.sha256(
        "\n".join(unique_sorted).encode("utf-8")
    ).hexdigest()
    n_aliases_applied = sum(1 for topic in normalized_topics if aliases.get(topic, topic) != topic)

    summary = {
        "taxonomy_version": taxonomy_version,
        "n_topics": len(unique_sorted),
        "n_aliases_applied": int(n_aliases_applied),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    return topics_csv_path, summary_path
