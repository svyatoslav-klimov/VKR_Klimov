from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.build_topics import normalize_topic
from src.utils.run_manifest import taxonomy_version


def prepare_data(config_path: str) -> str:
    """Build time-based train/val/test parquet from ``clean.csv`` (TASK-002b)."""
    from src.data.split import prepare_data_from_config

    prepare_data_from_config(config_path)
    return str(config_path)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def load_taxonomy_aliases(taxonomy_yaml: str | Path) -> dict[str, str]:
    """Load and normalize ``aliases`` from ``configs/taxonomy.yaml``-style file."""
    taxonomy = _load_yaml(taxonomy_yaml)
    aliases = taxonomy.get("aliases", {})
    if not isinstance(aliases, dict):
        return {}
    return {
        normalize_topic(alias): normalize_topic(canonical)
        for alias, canonical in aliases.items()
    }


def _resolve_topic_column(df: pd.DataFrame) -> str:
    for col in ("topic_id", "theme", "тема", "topic"):
        if col in df.columns:
            return col
    raise KeyError(f"No supported topic column in split dataframe: {list(df.columns)}")


def _resolve_created_at_column(df: pd.DataFrame) -> str:
    for col in ("created_at", "created_at_parsed", "created_at_legacy"):
        if col in df.columns:
            return col
    raise KeyError(f"No created_at column in split dataframe: {list(df.columns)}")


def theme_to_topic_id_map(
    topics_csv: str | Path,
    taxonomy_yaml: str = "configs/taxonomy.yaml",
) -> tuple[dict[str, str], str]:
    """
    Build normalized topic name -> ``topic_id`` from ``topics.csv``.

    ``taxonomy_yaml`` is reserved for API parity (DECISION-019); mapping is
    built from ``topics_csv`` only.     ``taxonomy_version`` matches
    ``run_manifest.taxonomy_version`` (sorted normalized names in CSV).
    """
    if not Path(taxonomy_yaml).is_file():
        raise FileNotFoundError(f"taxonomy yaml not found: {taxonomy_yaml}")
    topics = pd.read_csv(topics_csv, encoding="utf-8")
    if "topic_name" not in topics.columns or "topic_id" not in topics.columns:
        raise ValueError("topics.csv must contain topic_name and topic_id")
    mapping = {
        normalize_topic(name): str(topic_id)
        for name, topic_id in zip(topics["topic_name"].astype(str), topics["topic_id"].astype(str))
    }
    version = taxonomy_version(topics_csv)
    return mapping, version


def prepare_split(
    df: pd.DataFrame,
    topics_map: dict[str, str],
    aliases: dict[str, str] | None = None,
    taxonomy_yaml: str = "configs/taxonomy.yaml",
) -> pd.DataFrame:
    """
    Map legacy theme columns to ``topic_id``, attach ``created_at`` and
    ``topic_name_norm``. Drops rows with unknown topic or invalid date.

    Returns columns including ``text``, ``topic_id``, ``created_at``,
    ``topic_name_norm``.
    """
    alias_map = aliases if aliases is not None else load_taxonomy_aliases(taxonomy_yaml)
    topic_col = _resolve_topic_column(df)
    created_col = _resolve_created_at_column(df)
    if "text" not in df.columns:
        raise KeyError("Split dataframe must contain a 'text' column")

    if topic_col == "topic_id":
        out = df.copy()
        out["topic_id"] = out["topic_id"].dropna().astype(str)
        out["created_at"] = pd.to_datetime(out[created_col], errors="coerce")
        allowed_ids = set(str(v) for v in topics_map.values())
        ok = out["topic_id"].isin(allowed_ids)
        out = out.loc[ok].dropna(subset=["created_at"]).reset_index(drop=True)
        return out

    out = df.copy()
    normalized = out[topic_col].dropna().astype(str).map(normalize_topic)
    mapped_name = normalized.map(lambda x: alias_map.get(x, x))
    out = out.loc[normalized.index].copy()
    out["topic_name_norm"] = mapped_name
    out["topic_id"] = out["topic_name_norm"].map(topics_map)
    out["created_at"] = pd.to_datetime(out[created_col], errors="coerce")
    out = out.dropna(subset=["topic_id", "created_at"]).reset_index(drop=True)
    return out


def preprocess_text(raw: str, preproc: Mapping[str, Any] | None) -> str | None:
    """
    Apply SPEC-style preprocessing. Returns None if the text is below
    ``min_text_len`` after processing (row should be dropped for training/eval).
    """
    if not preproc:
        s = str(raw)
        if not s:
            return None
        return s
    s = str(raw)
    if preproc.get("normalize_unicode", True):
        s = unicodedata.normalize("NFKC", s)
    if preproc.get("to_lower", True):
        s = s.lower()
    s = s.replace("ё", "е")
    if preproc.get("collapse_spaces", True):
        s = re.sub(r"\s+", " ", s.strip())
    if preproc.get("mask_pii", True):
        s = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", s)
        s = re.sub(r"\b\+?\d[\d\-\s]{8,}\d\b", "[phone]", s)
    max_len = int(preproc.get("max_text_len", 2000))
    if len(s) > max_len:
        s = s[:max_len]
    min_len = int(preproc.get("min_text_len", 0))
    if len(s) < min_len:
        return None
    return s


def preprocess_text_padded(
    raw: str,
    preproc: Mapping[str, Any] | None,
) -> str:
    """Like ``preprocess_text`` but pad under ``min_text_len`` for eval (keeps n rows)."""
    r = preprocess_text(raw, preproc)
    if r is not None:
        return r
    if not preproc:
        return " "
    m = int(preproc.get("min_text_len", 10))
    return "x" * max(1, m)


def filter_by_preprocessed_text(
    frame: pd.DataFrame,
    preproc: Mapping[str, Any] | None,
    text_col: str = "text",
) -> pd.DataFrame:
    """Replace ``text`` with preprocessed strings; drop rows that fail ``min_text_len``."""
    texts = frame[text_col].astype(str).tolist()
    out_texts: list[str] = []
    keep: list[int] = []
    for i, t in enumerate(texts):
        pt = preprocess_text(t, preproc)
        if pt is None:
            continue
        out_texts.append(pt)
        keep.append(i)
    if not keep:
        return frame.iloc[0:0].copy()
    sub = frame.iloc[keep].copy()
    sub["text"] = out_texts
    return sub.reset_index(drop=True)
