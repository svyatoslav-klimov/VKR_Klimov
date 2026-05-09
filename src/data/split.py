from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.prepare import (
    filter_by_preprocessed_text,
    load_taxonomy_aliases,
    prepare_split,
    theme_to_topic_id_map,
)
from src.data.splits_version import file_sha256, splits_version, write_splits_version
from src.utils.run_manifest import taxonomy_version as manifest_taxonomy_version

logger = logging.getLogger(__name__)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def _read_clean_frame(clean_csv: Path, column_map: Mapping[str, str]) -> pd.DataFrame:
    text_col = column_map["text"]
    topic_col = column_map["topic"]
    df = pd.read_csv(clean_csv, encoding="utf-8")
    if text_col not in df.columns or topic_col not in df.columns:
        raise KeyError(f"clean.csv missing columns: need {text_col!r}, {topic_col!r}")
    created_key = column_map.get("created_at")
    legacy_key = column_map.get("created_at_legacy")
    if created_key and created_key in df.columns:
        ts_col = created_key
    elif legacy_key and legacy_key in df.columns:
        ts_col = legacy_key
    else:
        raise KeyError("clean.csv: no created_at or created_at_parsed column")
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "theme": df[topic_col],
        }
    )
    out["created_at"] = pd.to_datetime(df[ts_col], errors="coerce")
    return out.dropna(subset=["created_at"]).reset_index(drop=True)


def _enforce_text_disjoint(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_text = set(train["text"].astype(str))
    val_f = val[~val["text"].astype(str).isin(train_text)].copy()
    val_text = set(val_f["text"].astype(str))
    both_tv = train_text | val_text
    test_f = test[~test["text"].astype(str).isin(both_tv)].copy()
    return train, val_f.reset_index(drop=True), test_f.reset_index(drop=True)


def _slice_bucket_stats(
    frame: pd.DataFrame,
    train_topic_counts: dict[str, int],
    head_min: int,
    mid_min: int,
) -> dict[str, Any]:
    """Row counts and pseudo recall placeholder (N/A at split time) per bucket."""
    buckets = {"head": 0, "mid": 0, "tail": 0}
    for tid in frame["topic_id"].astype(str):
        c = train_topic_counts.get(tid, 0)
        if c >= head_min:
            buckets["head"] += 1
        elif c >= mid_min:
            buckets["mid"] += 1
        else:
            buckets["tail"] += 1
    return {"n_by_bucket": buckets, "n_rows": int(len(frame))}


def make_time_based_split(
    *,
    clean_csv: Path,
    out_dir: Path,
    topics_csv: Path,
    taxonomy_yaml: Path,
    column_map: Mapping[str, str],
    preprocessing: Mapping[str, Any] | None,
    test_start: str,
    val_window_months: int,
    test_window_months: int = 2,
    seed: int = 42,
    topics_summary_path: Path | None = None,
) -> dict[str, Any]:
    """
    Time-based split: train < val < test by ``created_at`` (no shuffle).

    - ``test``: rows with ``created_at >= test_start`` (last window of data).
    - ``val``: ``val_start <= created_at < test_start`` where
      ``val_start = test_start - val_window_months`` (MonthEnd-safe via pandas).
    - ``train``: ``created_at < val_start``.

    Output parquet columns: ``text``, ``topic_id``, ``created_at`` (+ optional ``theme`` dropped for canon).

    Enforces disjoint ``text`` across splits (drops val/test rows duplicated in earlier splits).
    """
    _ = seed  # reserved; split is deterministic from timestamps only

    aliases = load_taxonomy_aliases(taxonomy_yaml)
    topics_map, topics_ver = theme_to_topic_id_map(topics_csv, taxonomy_yaml)

    raw = _read_clean_frame(clean_csv, column_map)
    prepared = prepare_split(raw, topics_map=topics_map, aliases=aliases, taxonomy_yaml=str(taxonomy_yaml))
    prepared = prepared.sort_values("created_at").reset_index(drop=True)

    test_start_ts = pd.Timestamp(test_start)
    val_start_ts = test_start_ts - pd.DateOffset(months=int(val_window_months))

    train_mask = prepared["created_at"] < val_start_ts
    val_mask = (prepared["created_at"] >= val_start_ts) & (prepared["created_at"] < test_start_ts)
    test_mask = prepared["created_at"] >= test_start_ts

    train_df = prepared.loc[train_mask].copy()
    val_df = prepared.loc[val_mask].copy()
    test_df = prepared.loc[test_mask].copy()

    if preprocessing:
        train_df = filter_by_preprocessed_text(train_df, preprocessing, text_col="text")
        val_df = filter_by_preprocessed_text(val_df, preprocessing, text_col="text")
        test_df = filter_by_preprocessed_text(test_df, preprocessing, text_col="text")

    train_df, val_df, test_df = _enforce_text_disjoint(train_df, val_df, test_df)

    # Canonical columns; drop theme / topic_name_norm for SPEC 4.1 output
    keep_cols = ["text", "topic_id", "created_at"]
    train_df = train_df[keep_cols].sort_values("created_at").reset_index(drop=True)
    val_df = val_df[keep_cols].sort_values("created_at").reset_index(drop=True)
    test_df = test_df[keep_cols].sort_values("created_at").reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Time-based split produced an empty partition (train/val/test). "
            f"sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )
    assert train_df["created_at"].max() < val_df["created_at"].min(), "train/val time leak"
    assert val_df["created_at"].max() < test_df["created_at"].min(), "val/test time leak"

    train_classes = set(train_df["topic_id"].astype(str))
    n_train_classes = len(train_classes)
    train_topic_counts = train_df["topic_id"].astype(str).value_counts().to_dict()

    taxonomy_ref = manifest_taxonomy_version(topics_csv)
    if topics_summary_path and Path(topics_summary_path).is_file():
        with Path(topics_summary_path).open("r", encoding="utf-8") as f:
            summary = json.load(f)
        expected_tv = summary.get("taxonomy_version")
        if expected_tv and expected_tv != taxonomy_ref:
            raise ValueError(
                f"taxonomy_version mismatch vs topics_summary.json: {taxonomy_ref} != {expected_tv}"
            )

    # Optional: warn if test span exceeds test_window_months (informational)
    if len(test_df):
        test_span = test_df["created_at"].max() - test_df["created_at"].min()
        _ = test_span, test_window_months

    n_unseen_val = int((~val_df["topic_id"].astype(str).isin(train_classes)).sum())
    n_unseen_test = int((~test_df["topic_id"].astype(str).isin(train_classes)).sum())

    practical_train = _slice_bucket_stats(train_df, train_topic_counts, 50, 10)
    practical_val = _slice_bucket_stats(val_df, train_topic_counts, 50, 10)
    practical_test = _slice_bucket_stats(test_df, train_topic_counts, 50, 10)
    strict_train = _slice_bucket_stats(train_df, train_topic_counts, 500, 50)
    strict_val = _slice_bucket_stats(val_df, train_topic_counts, 500, 50)
    strict_test = _slice_bucket_stats(test_df, train_topic_counts, 500, 50)

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clean_csv": str(clean_csv).replace("\\", "/"),
        "taxonomy_version": taxonomy_ref,
        "time_windows": {
            "test_start_inclusive": str(test_start_ts),
            "val_start_inclusive": str(val_start_ts),
            "train_end_exclusive": str(val_start_ts),
            "train_created_at_max": str(train_df["created_at"].max()),
            "val_created_at_min": str(val_df["created_at"].min()),
            "val_created_at_max": str(val_df["created_at"].max()),
            "test_created_at_min": str(test_df["created_at"].min()) if len(test_df) else None,
            "test_created_at_max": str(test_df["created_at"].max()) if len(test_df) else None,
        },
        "counts": {
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "n_train_classes": n_train_classes,
            "n_unseen_val": n_unseen_val,
            "n_unseen_test": n_unseen_test,
        },
        "head_mid_tail_practical": {
            "thresholds": {"head_min": 50, "mid_min": 10},
            "train": practical_train,
            "val": practical_val,
            "test": practical_test,
        },
        "head_mid_tail_strict": {
            "thresholds": {"head_min": 500, "mid_min": 50},
            "train": strict_train,
            "val": strict_val,
            "test": strict_test,
        },
        "validation": {
            "train_max_lt_val_min": True,
            "val_max_lt_test_min": True,
            "text_overlap_train_val": 0,
            "text_overlap_train_test": 0,
            "text_overlap_val_test": 0,
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    report["parquet_sha256"] = {
        "train": file_sha256(train_path),
        "val": file_sha256(val_path),
        "test": file_sha256(test_path),
    }

    return report


def _split_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Split report (time-based)",
        "",
        f"Generated: {report['generated_at']}",
        f"taxonomy_version: {report['taxonomy_version']}",
        "",
        "## Time windows",
        "",
        "```",
        json.dumps(report["time_windows"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Counts",
        "",
        "```",
        json.dumps(report["counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Head/mid/tail (practical)",
        "",
        "```",
        json.dumps(report["head_mid_tail_practical"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Head/mid/tail (strict)",
        "",
        "```",
        json.dumps(report["head_mid_tail_strict"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def prepare_data_from_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load ``configs/data.yaml``, build splits under ``paths.train_path`` directory,
    write ``reports/split/*`` and return the split report dict.
    """
    cfg = _load_yaml(config_path)
    paths = cfg.get("paths") or {}
    columns = cfg.get("columns") or {}
    split_cfg = cfg.get("split") or {}
    preproc = cfg.get("preprocessing")

    clean_csv = Path(paths["clean_csv"])
    topics_csv = Path(paths["topics_csv"])
    train_path = Path(paths["train_path"])
    out_dir = train_path.parent
    taxonomy_yaml = Path(cfg.get("taxonomy", {}).get("aliases_path", "configs/taxonomy.yaml"))

    test_start = str(split_cfg.get("test_start", "2024-07-01"))
    val_window_months = int(split_cfg.get("val_window_months", 3))
    test_window_months = int(split_cfg.get("test_window_months", 2))
    seed = int(split_cfg.get("seed", cfg.get("seed", 42)))

    topics_summary = split_cfg.get("topics_summary_path", "reports/eda/topics_summary.json")

    report = make_time_based_split(
        clean_csv=clean_csv,
        out_dir=out_dir,
        topics_csv=topics_csv,
        taxonomy_yaml=taxonomy_yaml,
        column_map=columns,
        preprocessing=preproc,
        test_start=test_start,
        val_window_months=val_window_months,
        test_window_months=test_window_months,
        seed=seed,
        topics_summary_path=Path(topics_summary) if topics_summary else None,
    )

    n_train_classes = report["counts"]["n_train_classes"]
    if n_train_classes < 261:
        raise RuntimeError(
            "BLOCKER TASK-002b: n_train_classes=%s < 261 after filter "
            "(tail classes dropped excessively; escalate ROLE_01)." % n_train_classes
        )

    report_dir = Path(split_cfg.get("report_dir", "reports/split"))
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "split_report.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    md_path = report_dir / "split_report.md"
    with md_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(_split_report_markdown(report))

    split_hashes = splits_version(
        str(train_path),
        str(Path(paths["val_path"])),
        str(Path(paths["test_path"])),
        legacy_random_val=False,
    )
    split_hashes.pop("splits_legacy_random_val", None)
    ver_path = report_dir / "splits_version.json"
    write_splits_version(ver_path, split_hashes)

    logger.info(
        "prepare-data: wrote train/val/test to %s (n_train_classes=%s)",
        out_dir,
        n_train_classes,
    )
    return report


def build_time_based_splits(config_path: str) -> tuple[str, str, str]:
    """Backward-compatible name: build splits from YAML; return train, val, test paths."""
    cfg = _load_yaml(config_path)
    paths = cfg["paths"]
    prepare_data_from_config(config_path)
    return str(paths["train_path"]), str(paths["val_path"]), str(paths["test_path"])
