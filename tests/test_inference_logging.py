"""TASK-032: inference Parquet logging (env-driven, no model load)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.logging import (
    LOG_COLUMNS,
    compute_request_hash,
    floor_timestamp_minute_utc,
    reset_inference_log_writer_for_tests,
)
from src.api.server import create_app_for_test
from tests.test_api import FakePredictor


@pytest.fixture(autouse=True)
def _reset_writer_singleton() -> None:
    reset_inference_log_writer_for_tests()
    yield
    reset_inference_log_writer_for_tests()


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "inf_logs"
    d.mkdir()
    return d


def _schema_doc_columns(repo_root: Path) -> list[str]:
    path = repo_root / "docs/inference_logging_schema.json"
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    cols = doc.get("columns")
    assert isinstance(cols, list)
    names = [c["name"] for c in cols]
    return names


def test_request_hash_format_and_stability() -> None:
    floor = "2026-05-08T11:00:00Z"
    h1 = compute_request_hash("same text", floor)
    h2 = compute_request_hash("same text", floor)
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert compute_request_hash("same text", "2026-05-08T11:01:00Z") != h1


def test_floor_timestamp_minute_utc() -> None:
    from datetime import datetime, timezone

    dt = datetime(2026, 5, 8, 11, 34, 56, 789000, tzinfo=timezone.utc)
    assert floor_timestamp_minute_utc(dt) == "2026-05-08T11:34:00Z"


def test_schema_json_matches_logging_module() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert _schema_doc_columns(repo_root) == list(LOG_COLUMNS)


def test_predict_writes_parquet_without_raw_text(log_dir: Path) -> None:
    os.environ["INFERENCE_LOG_ENABLED"] = "1"
    os.environ["INFERENCE_LOG_DIR"] = str(log_dir)
    os.environ["INFERENCE_LOG_RAW_TEXT"] = "0"
    reset_inference_log_writer_for_tests()

    fixed = pd.Timestamp("2026-05-08T12:00:00", tz="UTC").to_pydatetime()
    with patch("src.api.server._utc_now", return_value=fixed):
        c = TestClient(create_app_for_test({"predictor": FakePredictor()}))
        text = "z" * 100
        r = c.post("/predict", json={"text": text})
    assert r.status_code == 200

    day = "2026-05-08"
    pq = log_dir / f"{day}.parquet"
    assert pq.is_file()
    df = pd.read_parquet(pq, engine="pyarrow")
    assert list(df.columns) == list(LOG_COLUMNS)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["event_type"] == "predict"
    assert row["model_version"] == "fake_mv"
    assert pd.isna(row["raw_text"]) or row["raw_text"] is None
    assert row["text_len"] == 100
    assert row["request_hash"] == compute_request_hash(text, "2026-05-08T12:00:00Z")
    assert isinstance(row["top_k_ids"], (list, type(None))) or hasattr(row["top_k_ids"], "__len__")


def test_predict_raw_text_when_env_enabled(log_dir: Path) -> None:
    os.environ["INFERENCE_LOG_ENABLED"] = "1"
    os.environ["INFERENCE_LOG_DIR"] = str(log_dir)
    os.environ["INFERENCE_LOG_RAW_TEXT"] = "1"
    reset_inference_log_writer_for_tests()

    fixed = pd.Timestamp("2026-05-08T12:01:00", tz="UTC").to_pydatetime()
    with patch("src.api.server._utc_now", return_value=fixed):
        c = TestClient(create_app_for_test({"predictor": FakePredictor()}))
        text = "w" * 100
        r = c.post("/predict", json={"text": text})
    assert r.status_code == 200

    df = pd.read_parquet(log_dir / "2026-05-08.parquet", engine="pyarrow")
    row = df.iloc[-1]
    assert row["raw_text"] == text


def test_feedback_writes_row(log_dir: Path) -> None:
    os.environ["INFERENCE_LOG_ENABLED"] = "1"
    os.environ["INFERENCE_LOG_DIR"] = str(log_dir)
    os.environ["INFERENCE_LOG_RAW_TEXT"] = "0"
    reset_inference_log_writer_for_tests()

    fixed = pd.Timestamp("2026-05-09T08:00:00", tz="UTC").to_pydatetime()
    with patch("src.api.server._utc_now", return_value=fixed):
        c = TestClient(create_app_for_test({"predictor": FakePredictor()}))
        r = c.post(
            "/feedback",
            json={"request_hash": "sha256:abc123", "chosen_topic_id": "topic_z"},
        )
    assert r.status_code == 200

    df = pd.read_parquet(log_dir / "2026-05-09.parquet", engine="pyarrow")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["event_type"] == "feedback"
    assert row["chosen_topic"] == "topic_z"
    assert row["request_hash"] == "sha256:abc123"
    assert row["was_in_top_k"] is None or pd.isna(row["was_in_top_k"])


def test_logging_disabled_no_parquet(log_dir: Path) -> None:
    os.environ["INFERENCE_LOG_ENABLED"] = "0"
    os.environ["INFERENCE_LOG_DIR"] = str(log_dir)
    reset_inference_log_writer_for_tests()

    c = TestClient(create_app_for_test({"predictor": FakePredictor()}))
    r = c.post("/predict", json={"text": "q" * 100})
    assert r.status_code == 200
    assert not list(log_dir.glob("*.parquet"))


def test_validation_error_logs_when_enabled(log_dir: Path) -> None:
    os.environ["INFERENCE_LOG_ENABLED"] = "1"
    os.environ["INFERENCE_LOG_DIR"] = str(log_dir)
    reset_inference_log_writer_for_tests()

    fixed = pd.Timestamp("2026-05-10T09:00:00", tz="UTC").to_pydatetime()
    with patch("src.api.server._utc_now", return_value=fixed):
        c = TestClient(create_app_for_test({"predictor": FakePredictor()}))
        r = c.post("/predict", json={"text": ""})
    assert r.status_code == 400

    df = pd.read_parquet(log_dir / "2026-05-10.parquet", engine="pyarrow")
    assert len(df) == 1
    assert df.iloc[0]["event_type"] == "predict"
    assert df.iloc[0]["text_len"] == 0


def test_closed_set_error_logs(log_dir: Path) -> None:
    os.environ["INFERENCE_LOG_ENABLED"] = "1"
    os.environ["INFERENCE_LOG_DIR"] = str(log_dir)
    reset_inference_log_writer_for_tests()

    fixed = pd.Timestamp("2026-05-11T10:00:00", tz="UTC").to_pydatetime()
    with patch("src.api.server._utc_now", return_value=fixed):
        c = TestClient(create_app_for_test({"predictor": FakePredictor()}))
        r = c.post("/predict", json={"text": "__closed_set__" + "x" * 100})
    assert r.status_code == 500

    df = pd.read_parquet(log_dir / "2026-05-11.parquet", engine="pyarrow")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["event_type"] == "predict"
    assert row["top_k_ids"] is None or len(row["top_k_ids"]) == 0


def test_hidden_ux_still_logs_full_topk(log_dir: Path) -> None:
    """When UX hides suggestions, HTTP items are empty but logs keep sink top-k."""
    os.environ["INFERENCE_LOG_ENABLED"] = "1"
    os.environ["INFERENCE_LOG_DIR"] = str(log_dir)
    reset_inference_log_writer_for_tests()

    fixed = pd.Timestamp("2026-05-12T11:00:00", tz="UTC").to_pydatetime()
    with patch("src.api.server._utc_now", return_value=fixed):
        c = TestClient(create_app_for_test({"predictor": FakePredictor()}))
        r = c.post("/predict", json={"text": "y" * 50})
    assert r.status_code == 200
    assert r.json()["shown"] is False

    df = pd.read_parquet(log_dir / "2026-05-12.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["ux_shown"] == False
    assert len(row["top_k_ids"]) == 2