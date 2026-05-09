"""TASK-033: active-learning queue from inference Parquet logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.api.active_learning import (
    QUEUE_TITLE,
    TASK_ID,
    _seed_request_hash,
    build_queue_for_date,
    build_queue_from_frames,
    load_tail_topic_ids,
    main,
    read_day_log,
    repo_root_from_here,
    write_queue_json,
)
from src.api.logging import LOG_COLUMNS

REPO = Path(__file__).resolve().parents[1]


def _schema_meta() -> tuple[list[str], list[str], list[str]]:
    path = REPO / "docs/active_learning_queue_schema.json"
    with path.open(encoding="utf-8") as f:
        schema = json.load(f)
    return (
        schema["document"]["required"],
        schema["item"]["required"],
        schema["item"]["forbidden_keys"],
    )


def _assert_queue_schema(doc: dict) -> None:
    doc_req, item_req, forbidden = _schema_meta()
    for k in doc_req:
        assert k in doc
    assert doc["title"] == QUEUE_TITLE
    assert doc["task_id"] == TASK_ID
    blob = json.dumps(doc, ensure_ascii=False)
    assert "raw_text" not in blob
    for it in doc["items"]:
        for k in item_req:
            assert k in it, f"missing {k}"
        for fk in forbidden:
            assert fk not in it


def _pred_row(
    *,
    i: int,
    request_hash: str,
    timestamp_utc: str,
    top_k_ids: list[str],
    top_k_scores: list[float],
    top_k_cal: list[float | None],
    date_utc: str = "2026-06-15",
) -> dict:
    return {
        "request_hash": request_hash,
        "timestamp_utc": timestamp_utc,
        "date_utc": date_utc,
        "text_len": 20,
        "raw_text": None,
        "top_k_ids": top_k_ids,
        "top_k_scores": top_k_scores,
        "top_k_scores_calibrated": top_k_cal,
        "chosen_topic": None,
        "was_in_top_k": None,
        "latency_ms": 1.0,
        "model_version": "mv_test",
        "taxonomy_version": "tax_test",
        "prefix_len_used": 20,
        "ux_shown": True,
        "conf_channel": "c",
        "conf_value": 0.5,
        "event_type": "predict",
    }


def _fb_row(*, request_hash: str, chosen: str, timestamp_utc: str, date_utc: str) -> dict:
    return {
        "request_hash": request_hash,
        "timestamp_utc": timestamp_utc,
        "date_utc": date_utc,
        "text_len": 0,
        "raw_text": None,
        "top_k_ids": [],
        "top_k_scores": [],
        "top_k_scores_calibrated": [],
        "chosen_topic": chosen,
        "was_in_top_k": None,
        "latency_ms": None,
        "model_version": "",
        "taxonomy_version": "",
        "prefix_len_used": 0,
        "ux_shown": False,
        "conf_channel": "",
        "conf_value": None,
        "event_type": "feedback",
    }


def _tail_and_non_tail() -> tuple[str, str]:
    topics = REPO / "data/processed/topics.csv"
    df = pd.read_csv(topics, encoding="utf-8")
    tail = str(df[df["count_total"] < 10]["topic_id"].iloc[0])
    non_tail = str(df[df["count_total"] >= 10]["topic_id"].iloc[0])
    return tail, non_tail


def _parse_ts(s: str) -> float:
    t = s.replace("Z", "+00:00")
    return datetime.fromisoformat(t).replace(tzinfo=timezone.utc).timestamp()


def test_queue_schema_and_category_coverage(tmp_path: Path) -> None:
    tail_id, non_tail_id = _tail_and_non_tail()
    tail_set = load_tail_topic_ids(REPO / "data/processed/topics.csv")
    assert tail_id in tail_set
    assert non_tail_id not in tail_set

    rows: list[dict] = []
    for i in range(10):
        h = f"sha256:outside{i:03d}"
        rows.append(
            _pred_row(
                i=i,
                request_hash=h,
                timestamp_utc="2026-06-14T01:00:00.000000Z",
                top_k_ids=[non_tail_id, tail_id],
                top_k_scores=[0.8, 0.2],
                top_k_cal=[0.9, None],
            )
        )
        rows.append(
            _fb_row(
                request_hash=h,
                chosen="zzzzzzzz",
                timestamp_utc="2026-06-14T01:01:00.000000Z",
                date_utc="2026-06-15",
            )
        )

    for i in range(10, 20):
        rows.append(
            _pred_row(
                i=i,
                request_hash=f"sha256:margin{i:03d}",
                timestamp_utc="2026-06-14T02:00:00.000000Z",
                top_k_ids=[non_tail_id, tail_id],
                top_k_scores=[0.5, 0.46],
                top_k_cal=[0.9, None],
            )
        )

    for i in range(20, 30):
        rows.append(
            _pred_row(
                i=i,
                request_hash=f"sha256:lowconf{i:03d}",
                timestamp_utc="2026-06-14T03:00:00.000000Z",
                top_k_ids=[non_tail_id, tail_id],
                top_k_scores=[0.9, 0.1],
                top_k_cal=[0.1, None],
            )
        )

    for i in range(30, 40):
        rows.append(
            _pred_row(
                i=i,
                request_hash=f"sha256:tail{i:03d}",
                timestamp_utc="2026-06-14T04:00:00.000000Z",
                top_k_ids=[tail_id, non_tail_id],
                top_k_scores=[0.9, 0.1],
                top_k_cal=[0.9, None],
            )
        )

    for i in range(40, 50):
        rows.append(
            _pred_row(
                i=i,
                request_hash=f"sha256:late{i:03d}",
                timestamp_utc="2026-06-15T05:00:00.000000Z",
                top_k_ids=[non_tail_id, tail_id],
                top_k_scores=[0.9, 0.1],
                top_k_cal=[0.9, None],
            )
        )

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    day = "2026-06-15"
    pq = log_dir / f"{day}.parquet"
    df = pd.DataFrame([{k: r[k] for k in LOG_COLUMNS} for r in rows])
    df.to_parquet(pq, engine="pyarrow", index=False)

    root = repo_root_from_here()
    out = tmp_path / "out"
    doc = build_queue_for_date(
        repo_root=root,
        log_dir=log_dir,
        date_utc=day,
        topics_path=REPO / "data/processed/topics.csv",
        tau_low_conf_cli=0.5,
        tau_low_margin=0.05,
        late_period_start_iso="2026-06-15",
        include_seed=False,
        seed_parquet=None,
    )
    _assert_queue_schema(doc)
    write_queue_json(out / f"{day}.json", doc)

    def count_cat(code: str) -> int:
        return sum(1 for it in doc["items"] if code in it["categories"])

    assert count_cat("outside_top_k") >= 10
    assert count_cat("low_margin") >= 10
    assert count_cat("low_conf") >= 10
    assert count_cat("tail_predicted") >= 10
    assert count_cat("late_period") >= 10

    priorities = [float(it["priority"]) for it in doc["items"]]
    assert priorities == sorted(priorities, reverse=True)

    for p in sorted(set(priorities), reverse=True):
        sub = [
            (_parse_ts(it["timestamp_utc"]), it["request_hash"])
            for it in doc["items"]
            if float(it["priority"]) == p
        ]
        if not sub:
            continue
        ts_vals = [x[0] for x in sub]
        assert ts_vals == sorted(ts_vals)


def test_feedback_predict_join_was_in_top_k() -> None:
    _, non_tail = _tail_and_non_tail()
    other = "aaaaaaaa"
    t1, t2 = non_tail, other
    if t1 == t2:
        t2 = "bbbbbbbb"
    h = "sha256:join_one"
    pred = _pred_row(
        i=0,
        request_hash=h,
        timestamp_utc="2026-06-14T10:00:00.000000Z",
        top_k_ids=[t1, t2],
        top_k_scores=[0.5, 0.46],
        top_k_cal=[0.9, None],
    )
    fb = _fb_row(
        request_hash=h,
        chosen=t1,
        timestamp_utc="2026-06-14T10:05:00.000000Z",
        date_utc="2026-06-15",
    )
    df = pd.DataFrame([{k: pred[k] for k in LOG_COLUMNS}, {k: fb[k] for k in LOG_COLUMNS}])
    tail_set = load_tail_topic_ids(REPO / "data/processed/topics.csv")
    doc = build_queue_from_frames(
        df,
        date_utc="2026-06-15",
        tail_set=tail_set,
        tau_low_conf=0.5,
        tau_low_margin=0.05,
        late_start=None,
        include_seed=False,
        seed_path=None,
    )
    it = next(x for x in doc["items"] if x["request_hash"] == h)
    assert it["was_in_top_k"] is True
    assert "outside_top_k" not in it["categories"]
    assert "low_margin" in it["categories"]


def test_empty_log_file_yields_empty_items(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    root = repo_root_from_here()
    doc = build_queue_for_date(
        repo_root=root,
        log_dir=log_dir,
        date_utc="2026-01-01",
        topics_path=REPO / "data/processed/topics.csv",
        tau_low_conf_cli=0.5,
        tau_low_margin=0.05,
        late_period_start_iso=None,
        include_seed=False,
        seed_parquet=None,
    )
    assert doc["items"] == []
    _assert_queue_schema(doc)


def test_include_seed_synthetic_parquet(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    seed_df = pd.DataFrame(
        [
            {
                "test_idx": 100,
                "true_topic_id": "deadbeef",
                "top1_pred": "cafebabe",
                "top1_score_raw": 0.99,
                "conf_calibrated": 0.4,
                "true_in_top10": False,
                "text_excerpt": "hello",
            },
            {
                "test_idx": 101,
                "true_topic_id": "11111111",
                "top1_pred": "22222222",
                "top1_score_raw": 0.88,
                "conf_calibrated": None,
                "true_in_top10": False,
                "text_excerpt": None,
            },
        ]
    )
    seed_df.to_parquet(seed_path, engine="pyarrow", index=False)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    root = repo_root_from_here()
    doc = build_queue_for_date(
        repo_root=root,
        log_dir=log_dir,
        date_utc="2026-06-20",
        topics_path=REPO / "data/processed/topics.csv",
        tau_low_conf_cli=0.5,
        tau_low_margin=0.05,
        late_period_start_iso=None,
        include_seed=True,
        seed_parquet=seed_path,
    )
    seed_items = [it for it in doc["items"] if it["source"] == "seed_iter2_confident_error"]
    assert len(seed_items) == 2
    for it in seed_items:
        assert it["request_hash"].startswith("seed:")
        assert "seed_iter2_confident_error" in it["categories"]
        assert "outside_top_k" in it["categories"]
        assert it["priority"] == 0.4
        assert it["was_in_top_k"] is False
    h0 = _seed_request_hash("100", "deadbeef")
    assert seed_items[0]["request_hash"] == h0
    _assert_queue_schema(doc)


def test_seed_request_hash_stable() -> None:
    assert _seed_request_hash("0", "ab") == _seed_request_hash("0", "ab")
    assert _seed_request_hash("0", "ab") != _seed_request_hash("1", "ab")


def test_cli_smoke_writes_json(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    out = tmp_path / "out"
    argv = [
        "--logs",
        str(log_dir),
        "--out",
        str(out),
        "--date",
        "2026-07-01",
        "--topics",
        str(REPO / "data/processed/topics.csv"),
        "--tau-low-conf",
        "0.5",
    ]
    assert main(argv) == 0
    j = out / "2026-07-01.json"
    assert j.is_file()
    with j.open(encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["items"] == []


def test_read_day_log_missing_returns_empty_columns(tmp_path: Path) -> None:
    missing = tmp_path / "nope.parquet"
    df = read_day_log(missing)
    assert list(df.columns) == list(LOG_COLUMNS)
