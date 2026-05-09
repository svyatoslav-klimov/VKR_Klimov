"""Unit tests for TASK-012 prefix UX evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.cli.prefix_eval import require_test_policy
from src.eval.prefix_eval import (
    GridCell,
    build_tau_grid,
    compute_confidence_channels,
    pick_optimal_ux_policy,
    truncate_text,
    verify_cross_prefix_monotonicity,
)


def test_truncate_text_full_short_input():
    t, tr = truncate_text("hi", 100)
    assert t == "hi" and tr is False


def test_truncate_text_truncated():
    t, tr = truncate_text("abcde", 3)
    assert t == "abc" and tr is True


def test_confidence_channels_shape_and_dtype():
    x = np.random.RandomState(0).rand(100, 10).astype(np.float64)
    ch = compute_confidence_channels(x, calibrator_path=None)
    assert set(ch.keys()) == {"max_score", "margin", "entropy", "normalized_gap"}
    assert ch["max_score"].shape == (100,)
    assert ch["margin"].shape == (100,)


def test_entropy_near_zero_for_peaked_scores():
    """Softmax(top10): sharp logit gap -> entropy ~ 0."""
    x = np.full((3, 10), -50.0, dtype=np.float64)
    x[:, 0] = 0.0
    ch = compute_confidence_channels(x, None)
    assert np.all(ch["entropy"] < 1e-6)


def test_calibrated_channel_optional(tmp_path):
    x = np.ones((4, 10), dtype=np.float64) * 0.5
    no_cal = compute_confidence_channels(x, None)
    assert "max_score_calibrated" not in no_cal
    pkl = tmp_path / "iso.pkl"
    from sklearn.isotonic import IsotonicRegression
    import pickle

    m = IsotonicRegression(out_of_bounds="clip")
    m.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pkl.open("wb") as f:
        pickle.dump({"type": "isotonic", "model": m}, f)
    with_cal = compute_confidence_channels(x, pkl)
    assert "max_score_calibrated" in with_cal


def test_bootstrap_ci_lower_bound_le_mean():
    rng = np.random.default_rng(42)
    per_text_records = []
    for ti in range(30):
        for pl in (10, 20):
            per_text_records.append(
                {
                    "text_id": str(ti),
                    "prefix_len": pl,
                    "char_len": pl,
                    "prefix_truncated": False,
                    "channels": {"max_score": float(rng.uniform(0.2, 0.9))},
                    "top10_topics": ["a", "b"],
                    "y_topic": "a",
                }
            )
    grid = build_tau_grid(
        per_text_records,
        [10],
        [0.5],
        ["max_score"],
        bootstrap_n=200,
        ci_level=0.95,
        rng=rng,
    )
    c = grid[0]
    assert c.ci95_low <= c.recall_at_10_given_shown_val + 1e-9


def test_pick_respects_show_rate_floor():
    g = [
        GridCell(10, "max_score", 0.5, 0.1, 0.4, 0.9, 0.9, 0.85, 0.95),
        GridCell(20, "max_score", 0.5, 0.1, 0.55, 0.5, 0.5, 0.5, 0.6),
    ]
    best = pick_optimal_ux_policy(g, show_rate_floor=0.50)
    assert best.L_min == 20


def test_test_split_requires_policy_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="R-007"):
        require_test_policy("test", None)


def test_cross_prefix_monotonicity_violation():
    assert verify_cross_prefix_monotonicity([0.5, 0.4, 0.6], tolerance=0.02) is False


def test_require_test_policy_val_returns_none():
    assert require_test_policy("val", None) is None
