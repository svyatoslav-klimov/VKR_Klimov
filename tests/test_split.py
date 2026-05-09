from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import split


def _validate_parquet_triplet(train_path: Path, val_path: Path, test_path: Path) -> None:
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    for df, name in [(train, "train"), (val, "val"), (test, "test")]:
        assert "text" in df.columns and "topic_id" in df.columns, name
        assert "created_at" in df.columns, name
        assert df["created_at"].notna().all(), name

    assert train["created_at"].max() < val["created_at"].min()
    assert val["created_at"].max() < test["created_at"].min()

    overlap_tv = set(train["text"]) & set(val["text"])
    overlap_vt = set(val["text"]) & set(test["text"])
    overlap_tt = set(train["text"]) & set(test["text"])
    assert not overlap_tv, overlap_tv
    assert not overlap_vt, overlap_vt
    assert not overlap_tt, overlap_tt


def test_build_time_based_splits_on_disk() -> None:
    """Project splits must satisfy strict time order and disjoint texts."""
    root = Path(__file__).resolve().parents[1]
    train_p = root / "data" / "splits" / "train.parquet"
    val_p = root / "data" / "splits" / "val.parquet"
    test_p = root / "data" / "splits" / "test.parquet"
    if not train_p.is_file():
        pytest.skip("split parquet not present")
    _validate_parquet_triplet(train_p, val_p, test_p)


def test_make_time_based_split_synthetic(tmp_path: Path) -> None:
    """Small CSV: train Apr-, val Apr-Jun before Jul, test Jul+."""
    root = Path(__file__).resolve().parents[1]
    topics_csv = root / "data" / "processed" / "topics.csv"
    if not topics_csv.is_file():
        pytest.skip("topics.csv missing")

    topics = pd.read_csv(topics_csv, encoding="utf-8")
    tname = str(topics.loc[0, "topic_name"])
    tname2 = str(topics.loc[1, "topic_name"])

    rows = []
    for i in range(20):
        rows.append(
            {
                "обращение": f"msg train {i}",
                "тема": tname,
                "created_at_parsed": f"2024-03-{10+i:02d} 12:00:00",
            }
        )
    for i in range(15):
        rows.append(
            {
                "обращение": f"msg val {i}",
                "тема": tname2,
                "created_at_parsed": f"2024-05-{i+1:02d} 12:00:00",
            }
        )
    for i in range(10):
        rows.append(
            {
                "обращение": f"msg test {i}",
                "тема": tname,
                "created_at_parsed": f"2024-07-{i+1:02d} 12:00:00",
            }
        )

    clean = tmp_path / "clean.csv"
    pd.DataFrame(rows).to_csv(clean, index=False, encoding="utf-8")

    column_map = {
        "text": "обращение",
        "topic": "тема",
        "created_at": "created_at_parsed",
        "created_at_legacy": "created_at_parsed",
    }
    out_dir = tmp_path / "splits"
    report = split.make_time_based_split(
        clean_csv=clean,
        out_dir=out_dir,
        topics_csv=topics_csv,
        taxonomy_yaml=root / "configs" / "taxonomy.yaml",
        column_map=column_map,
        preprocessing=None,
        test_start="2024-07-01",
        val_window_months=3,
        test_window_months=2,
        seed=42,
        topics_summary_path=None,
    )
    assert report["counts"]["n_train"] >= 1
    assert report["counts"]["n_val"] >= 1
    assert report["counts"]["n_test"] >= 1
    _validate_parquet_triplet(out_dir / "train.parquet", out_dir / "val.parquet", out_dir / "test.parquet")


def test_splits_version_report_has_no_legacy_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """splits_version.json written by prepare_data must omit splits_legacy_random_val."""
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports" / "split" / "splits_version.json"
    if not rep.is_file():
        pytest.skip("splits_version.json not present")
    data = json.loads(rep.read_text(encoding="utf-8"))
    assert "splits_legacy_random_val" not in data
    assert data["splits_version"].startswith("sha256:")
