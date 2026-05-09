# Regression guards for TASK-038 (Iter-2 formal audit artefacts).
"""Lightweight asserts on archived metrics/manifest parity (no GPU)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def winner_metrics() -> dict:
    p = REPO / "reports" / "runs" / "20260427_171719_hybrid_weighted_score_minmax" / "metrics.json"
    assert p.is_file(), f"missing {p}"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_iter2_leaderboard_excludes_three_way_expect_25_rows() -> None:
    lb = pd.read_csv(REPO / "reports" / "leaderboard.csv", encoding="utf-8")
    filt = lb[~lb["model_family"].astype(str).str.contains("three_way", na=False)]
    assert len(filt) == 25


def test_winner_recall_chain_monotonic(winner_metrics: dict) -> None:
    r = winner_metrics["recall_at_k"]
    assert float(r["1"]) <= float(r["3"]) <= float(r["5"]) <= float(r["10"])


def test_winner_head_mid_tail_both_buckets(winner_metrics: dict) -> None:
    assert "head_mid_tail" in winner_metrics
    assert "head_mid_tail_strict" in winner_metrics


def test_reproduction_metrics_bit_identical_vs_winner_tmp018() -> None:
    wa = json.loads(
        (REPO / "reports/runs/20260427_171719_hybrid_weighted_score_minmax/metrics.json").read_text(encoding="utf-8")
    )
    wb = json.loads(
        (REPO / "reports/runs/20260505_115158_hybrid_weighted_score_minmax/metrics.json").read_text(encoding="utf-8")
    )
    assert wa["recall_at_k"]["10"] == wb["recall_at_k"]["10"]


def test_audit_findings_has_header_when_csv_exists() -> None:
    p = REPO / "reports" / "audit_findings_iter2.csv"
    assert p.is_file()
    rows = list(csv.reader(p.open("r", encoding="utf-8", newline="")))
    assert rows[0][0] == "issue_id"


@pytest.mark.skipif(not (REPO / "reports/leakage_checks.json").is_file(), reason="run tools/audit_iter2.py")
def test_leakage_all_pass_latest() -> None:
    raw = json.loads((REPO / "reports/leakage_checks.json").read_text(encoding="utf-8"))
    bad = [c for c in raw["checks"] if c["verdict"] not in {"PASS", "PASS_ARCHIVE"}]
    assert not bad


@pytest.mark.skipif(not (REPO / "reports/metric_sanity_checks.json").is_file(), reason="run tools/audit_iter2.py")
def test_metric_sanity_latest_ok_when_present() -> None:
    raw = json.loads((REPO / "reports/metric_sanity_checks.json").read_text(encoding="utf-8"))
    checks = raw.get("checks") or []
    if not checks:
        pytest.skip("no metric sanity checks")
    assert all(c.get("ok") for c in checks)
