"""Append-only leaderboard (DECISION-2026-04-27-020)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Canonical SPEC §8 column order (22 columns including notes).
LEADERBOARD_COLUMNS = [
    "run_id",
    "model_family",
    "config_path",
    "split",
    "unseen_policy",
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "macro_f1",
    "weighted_f1",
    "mrr@10",
    "ndcg@10",
    "head_recall@10",
    "mid_recall@10",
    "tail_recall@10",
    "latency_p50_ms",
    "latency_p95_ms",
    "taxonomy_version",
    "splits_version",
    "splits_legacy_random_val",
    "notes",
]


def _fmt_opt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _head_mid_tail_at_10(metrics: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    hmt = metrics.get("head_mid_tail") or {}
    out: list[float | None] = []
    for bucket in ("head", "mid", "tail"):
        block = hmt.get(bucket)
        if not isinstance(block, dict):
            out.append(None)
            continue
        r = block.get("recall@10")
        out.append(float(r) if r is not None else None)
    return out[0], out[1], out[2]


def _latency_p50_p95(metrics: dict[str, Any]) -> tuple[float | None, float | None]:
    lm = metrics.get("latency_ms") or {}
    if not isinstance(lm, dict):
        return None, None
    p50 = lm.get("p50")
    p95 = lm.get("p95")
    return (float(p50) if p50 is not None else None, float(p95) if p95 is not None else None)


def append_row(
    path: str | Path,
    run_id: str,
    model_family: str,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    config_path: str,
    *,
    split: str = "test",
    unseen_policy: str = "known_only",
    head_recall_at_10: float | None = None,
    mid_recall_at_10: float | None = None,
    tail_recall_at_10: float | None = None,
    latency_p50_ms: float | None = None,
    latency_p95_ms: float | None = None,
    notes: str = "",
) -> None:
    """Append one row; raise if run_id already present."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"leaderboard missing: {p}")
    with p.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != LEADERBOARD_COLUMNS:
            raise ValueError(
                "leaderboard.csv header mismatch; run migrate_leaderboard_schema once"
            )
        for rec in reader:
            if rec and rec[0] == run_id:
                raise ValueError(f"run_id already in leaderboard: {run_id}")

    h_def, m_def, t_def = _head_mid_tail_at_10(metrics)
    if head_recall_at_10 is None:
        head_recall_at_10 = h_def
    if mid_recall_at_10 is None:
        mid_recall_at_10 = m_def
    if tail_recall_at_10 is None:
        tail_recall_at_10 = t_def

    lp50, lp95 = _latency_p50_p95(metrics)
    if latency_p50_ms is None:
        latency_p50_ms = lp50
    if latency_p95_ms is None:
        latency_p95_ms = lp95

    mrec = metrics["recall_at_k"]
    d = manifest["data"]
    legacy = d.get("splits_legacy_random_val", False)
    legacy_cell = "true" if legacy else "false"

    out_row = [
        run_id,
        model_family,
        str(config_path).replace("\\", "/"),
        split,
        unseen_policy,
        f"{mrec['1']:.4f}",
        f"{mrec['3']:.4f}",
        f"{mrec['5']:.4f}",
        f"{mrec['10']:.4f}",
        f"{metrics['macro_f1']:.4f}",
        f"{metrics['weighted_f1']:.4f}",
        f"{metrics['mrr_at_10']:.4f}",
        f"{metrics['ndcg_at_10']:.4f}",
        _fmt_opt_float(head_recall_at_10),
        _fmt_opt_float(mid_recall_at_10),
        _fmt_opt_float(tail_recall_at_10),
        _fmt_opt_float(latency_p50_ms),
        _fmt_opt_float(latency_p95_ms),
        str(d.get("taxonomy_version", "")),
        str(d.get("splits_version", "")),
        legacy_cell,
        notes,
    ]
    with p.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(out_row)


def migrate_leaderboard_schema(
    path: str | Path,
    *,
    dry_run: bool = True,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """One-shot migration from legacy 17-column CSV to SPEC §8 (22 columns).

    Rewrites header and existing rows with extra columns filled from metrics.json
    where available. Does not alter legacy numeric cells for recall/F1 columns.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"leaderboard missing: {p}")

    reports_root = Path(reports_dir)

    with p.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"empty leaderboard: {p}")
        rows = list(reader)

    if tuple(header) == tuple(LEADERBOARD_COLUMNS):
        return {"status": "noop", "reason": "already_spec_v8_schema"}

    old_labels = [
        "run_id",
        "model_family",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "macro_f1",
        "weighted_f1",
        "mrr_at_10",
        "ndcg_at_10",
        "n_test_total",
        "n_test_known",
        "splits_version",
        "taxonomy_version",
        "splits_legacy_random_val",
        "created_at",
        "config_path",
    ]
    if tuple(header) != tuple(old_labels):
        raise ValueError(
            f"unexpected leaderboard header ({len(header)} cols); "
            f"expected legacy {len(old_labels)} cols or migrated "
            f"{len(LEADERBOARD_COLUMNS)} cols"
        )

    idx = {name: i for i, name in enumerate(header)}
    new_rows: list[list[str]] = []

    for row in rows:
        if not row:
            continue
        run_id = row[idx["run_id"]]
        metrics: dict[str, Any] = {}
        mpath = reports_root / "runs" / run_id / "metrics.json"
        if mpath.is_file():
            with mpath.open("r", encoding="utf-8") as fh:
                metrics = json.load(fh)

        h_v, m_v, t_v = _head_mid_tail_at_10(metrics)
        lp50, lp95 = _latency_p50_p95(metrics)

        split_val = metrics.get("split") if isinstance(metrics.get("split"), str) else "test"
        unseen_val = (
            metrics.get("unseen_policy")
            if isinstance(metrics.get("unseen_policy"), str)
            else "known_only"
        )

        new_rows.append(
            [
                row[idx["run_id"]],
                row[idx["model_family"]],
                row[idx["config_path"]],
                split_val,
                unseen_val,
                row[idx["recall_at_1"]],
                row[idx["recall_at_3"]],
                row[idx["recall_at_5"]],
                row[idx["recall_at_10"]],
                row[idx["macro_f1"]],
                row[idx["weighted_f1"]],
                row[idx["mrr_at_10"]],
                row[idx["ndcg_at_10"]],
                _fmt_opt_float(h_v),
                _fmt_opt_float(m_v),
                _fmt_opt_float(t_v),
                _fmt_opt_float(lp50),
                _fmt_opt_float(lp95),
                row[idx["taxonomy_version"]],
                row[idx["splits_version"]],
                row[idx["splits_legacy_random_val"]],
                "",
            ]
        )

    summary = {
        "status": "would_migrate" if dry_run else "migrated",
        "path": str(p),
        "rows": len(new_rows),
        "dry_run": dry_run,
        "columns_before": len(header),
        "columns_after": len(LEADERBOARD_COLUMNS),
    }

    if not dry_run:
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8", newline="\n") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(LEADERBOARD_COLUMNS)
                for nr in new_rows:
                    writer.writerow(nr)
            tmp_path.replace(p)
        except BaseException:
            if tmp_path.is_file():
                tmp_path.unlink(missing_ok=True)
            raise

    return summary
