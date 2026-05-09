"""Tests for TASK-013 tools/diagnose_tail.py."""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_diagnose_tail():
    path = REPO_ROOT / "tools" / "diagnose_tail.py"
    spec = importlib.util.spec_from_file_location("diagnose_tail_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dt():
    return _load_diagnose_tail()


def test_assign_failure_mode_priority(dt):
    """Priority a > b > c > d on synthetic inputs."""
    assert (
        dt.assign_failure_mode(
            5,
            100,
            0.5,
            0.99,
            n_sparse_class_min=10,
            l_short=30,
            term_overlap_threshold=0.10,
            top2_ambiguity_threshold=0.85,
        )
        == "data_sparsity"
    )
    assert (
        dt.assign_failure_mode(
            50,
            10,
            0.5,
            0.99,
            n_sparse_class_min=10,
            l_short=30,
            term_overlap_threshold=0.10,
            top2_ambiguity_threshold=0.85,
        )
        == "short_text"
    )
    assert (
        dt.assign_failure_mode(
            50,
            100,
            0.05,
            0.99,
            n_sparse_class_min=10,
            l_short=30,
            term_overlap_threshold=0.10,
            top2_ambiguity_threshold=0.85,
        )
        == "concept_drift"
    )
    assert (
        dt.assign_failure_mode(
            50,
            100,
            0.5,
            0.90,
            n_sparse_class_min=10,
            l_short=30,
            term_overlap_threshold=0.10,
            top2_ambiguity_threshold=0.85,
        )
        == "label_ambiguity"
    )


def test_decision_tree_gate_branches(dt):
    """Five distributions map to expected gate verdict labels."""
    g = dt.gate_verdict_from_metrics
    t_ds, t_st, t_la, t_u = 0.50, 0.40, 0.30, 0.50
    assert (
        g(0.51, 0.0, 0.0, 0.0, frac_data_sparsity_threshold=t_ds, frac_short_text_threshold=t_st,
          frac_label_ambiguity_threshold=t_la, frac_tail_truly_unlearnable_threshold=t_u)
        == "synthetic-only-option"
    )
    assert (
        g(0.2, 0.41, 0.0, 0.0, frac_data_sparsity_threshold=t_ds, frac_short_text_threshold=t_st,
          frac_label_ambiguity_threshold=t_la, frac_tail_truly_unlearnable_threshold=t_u)
        == "label-aware-needed"
    )
    assert (
        g(0.2, 0.3, 0.31, 0.0, frac_data_sparsity_threshold=t_ds, frac_short_text_threshold=t_st,
          frac_label_ambiguity_threshold=t_la, frac_tail_truly_unlearnable_threshold=t_u)
        == "multi-prototype-helps"
    )
    assert (
        g(0.2, 0.3, 0.2, 0.51, frac_data_sparsity_threshold=t_ds, frac_short_text_threshold=t_st,
          frac_label_ambiguity_threshold=t_la, frac_tail_truly_unlearnable_threshold=t_u)
        == "truly-unlearnable"
    )
    assert (
        g(0.2, 0.3, 0.2, 0.2, frac_data_sparsity_threshold=t_ds, frac_short_text_threshold=t_st,
          frac_label_ambiguity_threshold=t_la, frac_tail_truly_unlearnable_threshold=t_u)
        == "no-tail-action-needed"
    )


def test_term_overlap_jaccard_manual(dt):
    """Hand-checked Jaccard on toy IDF / token sets."""
    idf = {"a": 1.0, "b": 0.9, "c": 0.5, "x": 0.4}
    train_class_texts = ["a b c"]
    test_text = "a b x"
    j = dt.term_overlap_jaccard(train_class_texts, test_text, idf, top_n=100)
    sa = {"a", "b", "c"}
    sb = {"a", "b", "x"}
    expected = len(sa & sb) / len(sa | sb)
    assert abs(j - expected) < 1e-9


def test_smoke_diagnose_tail(dt, tmp_path):
    """End-to-end core logic on ~50 synthetic rows without Hybrid artifacts."""
    rng = np.random.default_rng(42)
    classes = [f"c{i:02d}" for i in range(8)]
    train_rows = []
    for _ in range(80):
        tid = rng.choice(classes)
        train_rows.append({"topic_id": tid, "text": f"train token {tid} word " * 5})
    train_df = pd.DataFrame(train_rows)
    test_rows = []
    for i in range(50):
        tid = classes[i % len(classes)]
        test_rows.append({"topic_id": tid, "text": f"test query {i} " + ("x " * 20)})
    test_df = pd.DataFrame(test_rows)

    pred_top100 = []
    sparse_top10 = []
    for i in range(50):
        true_tid = str(test_df.iloc[i]["topic_id"])
        others = [c for c in classes if c != true_tid]
        if i < 12:
            preds = others[:100]
        else:
            preds = [true_tid] + others[:99]
        pred_top100.append(preds)
        sparse_top10.append(preds[:10])

    topic_names = {c: f"name {c}" for c in classes}
    args_ns = Namespace(
        n_sparse_class_min=10,
        l_short=30,
        term_overlap_threshold=0.10,
        top2_ambiguity_threshold=0.85,
        frac_data_sparsity=0.50,
        frac_short_text=0.40,
        frac_label_ambiguity=0.30,
        frac_tail_truly_unlearnable=0.50,
    )
    out = dt.run_diagnosis_core(
        train_df=train_df,
        test_df=test_df,
        topic_names=topic_names,
        pred_top100=pred_top100,
        classes_sparse=classes,
        sparse_preds_top10=sparse_top10,
        args_ns=args_ns,
    )
    assert "gate_verdict" in out
    assert out["mcnemar_p_value"] is None or isinstance(out["mcnemar_p_value"], float)


def test_fusion_config_sha256_stable(dt, tmp_path):
    """Hash helper is deterministic."""
    p = tmp_path / "fusion_config.json"
    p.write_text(json.dumps({"best_lambda": 0.14}, ensure_ascii=False), encoding="utf-8")
    h1 = dt.fusion_config_sha256(p)
    h2 = dt.fusion_config_sha256(p)
    assert len(h1) == 64 and h1 == h2
