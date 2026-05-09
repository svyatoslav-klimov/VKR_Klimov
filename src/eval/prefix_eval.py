"""Prefix-mode UX simulation: offline truncation, confidence channels, tau-grid (TASK-012)."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.eval.metrics import recall_at_k
from src.models.hybrid import (
    HybridModel,
    normalize_per_query_row,
    retrieval_full_scores,
    sparse_full_scores,
)
from src.models.hybrid import _ranks_1based_desc
from src.models.hybrid import _union_indices as union_indices
from src.models.retrieval import encode_passages

logger = logging.getLogger(__name__)

PrefixLen = int | Literal["full"]

_CHANNEL_PRIORITY: dict[str, int] = {
    "max_score": 0,
    "max_score_calibrated": 1,
    "margin": 2,
    "normalized_gap": 3,
    "entropy": 4,
}


def normalize_prefix_len_tag(pl: PrefixLen | str | int) -> str:
    """Stable string tag for parquet/group keys."""
    if isinstance(pl, str):
        s = pl.strip().lower()
        return "full" if s == "full" else str(int(pl))
    if pl == "full":
        return "full"
    return str(int(pl))


def truncate_text(text: str, prefix_len: PrefixLen | str) -> tuple[str, bool]:
    """Return truncated prefix and whether truncation was applied."""
    s = str(text)
    pl = prefix_len
    if pl == "full" or pl is None:
        return s, False
    target = int(pl)
    if len(s) < target:
        return s, False
    return s[:target], True


def _softmax_entropy(top10: np.ndarray) -> np.ndarray:
    """Entropy over softmax(top10, T=1), vectorized batch."""
    x = np.asarray(top10, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mx = np.max(x, axis=1, keepdims=True)
    ex = np.exp(np.clip(x - mx, -50.0, 50.0))
    p = ex / (np.sum(ex, axis=1, keepdims=True) + 1e-15)
    logp = np.log(p + 1e-15)
    ent = -np.sum(p * logp, axis=1)
    return ent.astype(np.float64)


def fused_union_top10_scores_and_topics(
    s_full: np.ndarray,
    r_full: np.ndarray,
    *,
    lam: float,
    fusion_mode: str,
    norm: str,
    candidates: int,
    classes: list[str],
    k_out: int,
    k_rrf: int,
) -> tuple[np.ndarray, list[list[str]]]:
    """
    Hybrid fused scores on union-of-top-candidates (same semantics as ``fuse``).

    Returns:
        scores: (n, k_eff) fused scores sorted descending per row.
        topics: parallel topic names (length k_eff each row).
    """
    fusion = str(fusion_mode)
    n_classes = len(classes)
    k_eff = min(k_out, n_classes) if classes else 0
    n = int(s_full.shape[0])
    scores_out = np.zeros((n, k_eff), dtype=np.float64)
    topics_out: list[list[str]] = []

    for i in range(n):
        s_row = s_full[i]
        r_row = r_full[i]
        u = union_indices(s_row, r_row, candidates)
        if not u or k_eff <= 0:
            topics_out.append([])
            continue
        u_arr = np.asarray(u, dtype=np.int64)
        v_s = s_row[u_arr].astype(np.float64, copy=False)
        v_r = r_row[u_arr].astype(np.float64, copy=False)

        if fusion == "rrf":
            rs_s = _ranks_1based_desc(s_row)
            rs_r = _ranks_1based_desc(r_row)
            h = np.array(
                [
                    1.0 / (k_rrf + float(rs_s[j])) + 1.0 / (k_rrf + float(rs_r[j]))
                    for j in u
                ],
                dtype=np.float64,
            )
        else:
            ns = normalize_per_query_row(v_s.astype(np.float32, copy=False), norm).astype(np.float64)
            nr = normalize_per_query_row(v_r.astype(np.float32, copy=False), norm).astype(np.float64)
            if fusion == "weighted_score":
                h = float(lam) * ns + (1.0 - float(lam)) * nr
            elif fusion == "max":
                h = np.maximum(ns, nr)
            else:
                raise ValueError(f"Unknown fusion mode: {fusion!r}")

        order_u = np.argsort(-h)
        picked_scores: list[float] = []
        picked_topics: list[str] = []
        for oi in order_u:
            j = int(u[int(oi)])
            picked_topics.append(classes[j])
            picked_scores.append(float(h[int(oi)]))
            if len(picked_topics) >= k_eff:
                break
        pad = k_eff - len(picked_scores)
        if pad > 0:
            picked_scores.extend([0.0] * pad)
            picked_topics.extend([""] * pad)
        scores_out[i] = np.asarray(picked_scores[:k_eff], dtype=np.float64)
        topics_out.append(picked_topics[:k_eff])

    return scores_out, topics_out


def compute_confidence_channels(
    top10_scores: np.ndarray,
    calibrator_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Five confidence channels from fused hybrid top-10 scores.

    Optional fifth channel ``max_score_calibrated`` only when ``calibrator_path``
    is set (DECISION-040).
    """
    t = np.asarray(top10_scores, dtype=np.float64)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    top10 = t
    max_score = top10[:, 0].copy()
    margin = top10[:, 0] - top10[:, 1]
    entropy = _softmax_entropy(top10)
    normalized_gap = (top10[:, 0] - top10[:, 1]) / (top10[:, 0] + 1e-9)

    out: dict[str, np.ndarray] = {
        "max_score": max_score,
        "margin": margin,
        "entropy": entropy,
        "normalized_gap": normalized_gap,
    }
    if calibrator_path is not None:
        from experiments.calibration_v1.src.apply import apply_calibrator

        out["max_score_calibrated"] = apply_calibrator(max_score, calibrator_path)
    return out


def _records_for_split(
    model: HybridModel,
    df: pd.DataFrame,
    prefix_lengths: list[PrefixLen | str],
    calibrator_path: str | Path | None,
    *,
    text_col: str = "text",
    label_col: str = "topic_id",
    batch_encode_size: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run hybrid fusion on truncated texts per prefix length; build flat records."""
    fc = model.fusion_config
    lam = float(fc.get("best_lambda", 0.5))
    mode = str(fc.get("mode", "weighted_score"))
    norm = str(fc.get("norm", "minmax"))
    candidates = int(fc.get("candidates", 50))
    k_rrf = int(fc.get("k_rrf", 60))
    classes = model.classes
    ids = df.index.astype(str).tolist() if hasattr(df.index, "astype") else [str(i) for i in range(len(df))]
    raw_texts = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(str).tolist()

    bsz = max(int(batch_encode_size or model.batch_size), 16)
    flat_records: list[dict[str, Any]] = []

    prev_top10_topics: dict[str, list[str]] = {}

    for pl in prefix_lengths:
        pairs = [truncate_text(t, pl) for t in raw_texts]
        trunc_col = [p[0] for p in pairs]
        truncated_flags = [p[1] for p in pairs]

        n = len(trunc_col)
        scores_blk = np.zeros((n, 10), dtype=np.float64)
        topics_blk: list[list[str]] = [[] for _ in range(n)]

        for start in range(0, n, bsz):
            chunk = trunc_col[start : start + bsz]
            emb = encode_passages(
                model.encoder,
                chunk,
                model.e5_prefix_query,
                model.normalize_embeddings,
                model.batch_size,
            )
            s_full = sparse_full_scores(model.sparse_model, chunk, batch_size=model.sparse_batch_size)
            r_full = retrieval_full_scores(
                model.retrieval_index,
                len(classes),
                emb,
                k_search=len(classes),
            )
            sc, top = fused_union_top10_scores_and_topics(
                s_full,
                r_full,
                lam=lam,
                fusion_mode=mode,
                norm=norm,
                candidates=candidates,
                classes=classes,
                k_out=10,
                k_rrf=k_rrf,
            )
            end = start + len(chunk)
            scores_blk[start:end] = sc
            for j, row in enumerate(top):
                topics_blk[start + j] = row

        chans = compute_confidence_channels(scores_blk, calibrator_path)

        pl_key = normalize_prefix_len_tag(pl)

        for i in range(n):
            tid = ids[i]
            char_len = len(trunc_col[i])
            top10_t = topics_blk[i][:10] if topics_blk[i] else []
            row_prev = prev_top10_topics.get(tid, [])
            jac = _jaccard_top10(row_prev, top10_t) if row_prev else None
            prev_top10_topics[tid] = list(top10_t)

            ch_dict = {k: float(chans[k][i]) for k in chans}
            rec = {
                "text_id": tid,
                "prefix_len": pl_key,
                "char_len": char_len,
                "prefix_truncated": bool(truncated_flags[i]),
                "channels": ch_dict,
                "top10_topics": list(top10_t),
                "y_topic": y_true[i],
                "top10_scores": scores_blk[i].tolist(),
                "jaccard_top10_with_prev": jac,
            }
            flat_records.append(rec)

    per_prefix: dict[str, Any] = {}
    by_pl: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in flat_records:
        by_pl[r["prefix_len"]].append(r)

    for pl_key, rows in by_pl.items():
        preds = [x["top10_topics"] for x in rows]
        yt = [x["y_topic"] for x in rows]
        rk = recall_at_k(yt, preds, k_list=(1, 3, 5, 10))
        per_prefix[str(pl_key)] = {
            "recall_at_1": rk["1"],
            "recall_at_3": rk["3"],
            "recall_at_5": rk["5"],
            "recall_at_10": rk["10"],
            "n": len(rows),
        }

    return flat_records, per_prefix


def _jaccard_top10(a: list[str], b: list[str]) -> float:
    sa = set(a[:10])
    sb = set(b[:10])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union else 0.0


def evaluate_prefix(
    model: HybridModel,
    df: pd.DataFrame,
    prefix_lengths: list[PrefixLen | str],
    channels: list[str],
    calibrator_path: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run prefix simulation and per-prefix aggregate recalls."""
    _ = channels  # retained for API parity with plan
    return _records_for_split(model, df, prefix_lengths, calibrator_path)


@dataclass
class GridCell:
    """Single UX-policy grid cell on val."""

    L_min: int
    channel: str
    tau_quantile: float
    tau_value: float
    show_rate_val: float
    recall_at_10_given_shown_val: float
    recall_at_10_overall_val: float
    ci95_low: float
    ci95_high: float


def _eligible_records(records: list[dict[str, Any]], L_min: int) -> list[dict[str, Any]]:
    return [r for r in records if int(r["char_len"]) >= L_min]


def _metrics_for_tau(
    eligible: list[dict[str, Any]],
    channel: str,
    tau_value: float,
    y_key: str = "y_topic",
    top_key: str = "top10_topics",
) -> tuple[float, float, float]:
    """show_rate, recall_given_shown, recall_overall_eligible."""
    if not eligible:
        return 0.0, 0.0, 0.0
    shown = []
    hit_shown = []
    hit_all = []
    for r in eligible:
        conf = float(r["channels"][channel])
        top10 = r[top_key]
        y = str(r[y_key])
        hit = y in (top10[:10] if top10 else [])
        hit_all.append(hit)
        shown.append(conf >= tau_value)
        if conf >= tau_value:
            hit_shown.append(hit)

    show_rate = float(np.mean(shown)) if shown else 0.0
    if sum(shown) == 0:
        r_given = 0.0
    else:
        r_given = float(np.mean(hit_shown))
    r_overall = float(np.mean(hit_all))
    return show_rate, r_given, r_overall


def build_tau_grid(
    per_text_records: list[dict[str, Any]],
    l_min_grid: list[int],
    tau_quantile_grid: list[float],
    channels: list[str],
    bootstrap_n: int = 1000,
    ci_level: float = 0.95,
    *,
    rng: np.random.Generator | None = None,
) -> list[GridCell]:
    """Bootstrap CI for Recall@10|shown per grid cell; tau quantiles on full val."""
    rng = rng or np.random.default_rng(42)
    alpha = (1.0 - ci_level) / 2.0
    q_low = alpha * 100.0
    q_high = (1.0 - alpha) * 100.0

    grid: list[GridCell] = []

    for L_min in l_min_grid:
        eligible = _eligible_records(per_text_records, L_min)
        if not eligible:
            continue
        n_elig = len(eligible)
        tid_list = [str(r["text_id"]) for r in eligible]
        tid_to_rows: dict[str, list[int]] = defaultdict(list)
        for i, tid in enumerate(tid_list):
            tid_to_rows[tid].append(i)
        uids = np.array(sorted(tid_to_rows.keys()), dtype=object)

        vals_by_ch = {
            ch: np.asarray([float(r["channels"][ch]) for r in eligible], dtype=np.float64)
            for ch in channels
        }
        hit_vec = np.asarray(
            [
                str(r["y_topic"]) in (r["top10_topics"][:10] if r["top10_topics"] else [])
                for r in eligible
            ],
            dtype=np.bool_,
        )

        for channel in channels:
            vals = vals_by_ch[channel]
            for tau_q in tau_quantile_grid:
                tau_value = float(np.quantile(vals, tau_q))
                shown_full = vals >= tau_value
                show_rate = float(np.mean(shown_full))
                if np.any(shown_full):
                    r_gs = float(np.dot(shown_full.astype(np.float64), hit_vec.astype(np.float64)) / shown_full.sum())
                else:
                    r_gs = 0.0
                r_ov = float(np.mean(hit_vec))

                boot_stats = np.empty(bootstrap_n, dtype=np.float64)
                n_uid = len(uids)
                for b in range(bootstrap_n):
                    samp_tids = rng.choice(uids, size=n_uid, replace=True)
                    parts: list[np.ndarray] = []
                    for t in samp_tids:
                        parts.append(np.asarray(tid_to_rows[str(t)], dtype=np.int64))
                    if not parts:
                        ix = np.array([], dtype=np.int64)
                    else:
                        ix = np.concatenate(parts)
                    sv = vals[ix]
                    hv = hit_vec[ix]
                    sh = sv >= tau_value
                    if sh.sum() == 0:
                        boot_stats[b] = 0.0
                    else:
                        boot_stats[b] = float(np.dot(sh.astype(np.float64), hv.astype(np.float64)) / sh.sum())

                lo = float(np.percentile(boot_stats, q_low))
                hi = float(np.percentile(boot_stats, q_high))

                grid.append(
                    GridCell(
                        L_min=int(L_min),
                        channel=channel,
                        tau_quantile=float(tau_q),
                        tau_value=tau_value,
                        show_rate_val=show_rate,
                        recall_at_10_given_shown_val=r_gs,
                        recall_at_10_overall_val=r_ov,
                        ci95_low=lo,
                        ci95_high=hi,
                    )
                )

    return grid


def pick_optimal_ux_policy(
    grid: list[GridCell],
    show_rate_floor: float = 0.50,
) -> GridCell:
    """Max lower CI95(R@10|shown); ties -> smaller L_min, simpler channel."""
    valid = [c for c in grid if c.show_rate_val >= show_rate_floor - 1e-12]
    if not valid:
        raise RuntimeError(
            f"no grid cells meet show_rate_floor={show_rate_floor}; relax floor or grid."
        )

    def sort_key(c: GridCell) -> tuple[float, int, int]:
        pri = _CHANNEL_PRIORITY.get(c.channel, 99)
        return (-c.ci95_low, c.L_min, pri)

    return sorted(valid, key=sort_key)[0]


def verify_cross_prefix_monotonicity(
    show_rates_in_prefix_order: list[float],
    tolerance: float = 0.02,
) -> bool:
    """Show rates along increasing prefix lengths must be (approximately) non-decreasing."""
    rates = list(show_rates_in_prefix_order)
    for a, b in zip(rates, rates[1:], strict=False):
        if float(a) - tolerance > float(b):
            return False
    return True


def compute_val_test_parity(
    picked_policy: GridCell,
    val_data: list[dict[str, Any]],
    test_data: list[dict[str, Any]],
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Compare R@10|shown val vs test under fixed tau/channel/L_min."""
    L_min = picked_policy.L_min
    ch = picked_policy.channel
    tau = picked_policy.tau_value

    def _apply(rs: list[dict[str, Any]]) -> float:
        el = _eligible_records(rs, L_min)
        _, rgs, _ = _metrics_for_tau(el, ch, tau)
        return rgs

    rv = _apply(val_data)
    rt = _apply(test_data)
    delta = abs(rv - rt)
    return {
        "recall_val": rv,
        "recall_test": rt,
        "delta_r10_given_shown": float(delta),
        "parity_passed": bool(delta <= tolerance + 1e-12),
    }


def build_prefix_show_rates_for_policy(
    records: list[dict[str, Any]],
    policy: GridCell,
    prefix_order: list[PrefixLen | str],
) -> dict[str | int, float]:
    """Mean show indicator per prefix_len for chosen policy."""
    L_min = policy.L_min
    ch = policy.channel
    tau = policy.tau_value
    by_pl: dict[Any, list[bool]] = defaultdict(list)
    for r in records:
        if int(r["char_len"]) < L_min:
            continue
        pl = r["prefix_len"]
        conf = float(r["channels"][ch])
        by_pl[pl].append(conf >= tau)
    out: dict[str, float] = {}
    for pl in prefix_order:
        kk = normalize_prefix_len_tag(pl)
        lst = by_pl.get(kk, [])
        out[kk] = float(np.mean(lst)) if lst else 0.0
    return out


def plot_prefix_rocs(
    records_val: list[dict[str, Any]],
    channels: list[str],
    out_paths: dict[str, Path],
    *,
    L_min: int = 10,
) -> None:
    """ROC-like curves (show_rate vs R@10|shown) per channel (matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping prefix ROC figures.")
        return

    eligible = _eligible_records(records_val, L_min)
    if not eligible:
        return

    for channel in channels:
        vals = sorted({float(r["channels"][channel]) for r in eligible})
        xs: list[float] = []
        ys: list[float] = []
        for tau in vals:
            sr, rgs, _ = _metrics_for_tau(eligible, channel, tau)
            xs.append(sr)
            ys.append(rgs)
        outp = out_paths.get(channel)
        if outp is None:
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o", markersize=2)
        plt.xlabel("show_rate")
        plt.ylabel("Recall@10 | shown")
        plt.title(f"prefix UX sweep {channel}")
        plt.grid(True, alpha=0.3)
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outp, dpi=120, bbox_inches="tight")
        plt.close()


def records_to_wide_parquet_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (text_id, prefix_len) with channel columns for reload."""
    import json as _json

    rows: list[dict[str, Any]] = []
    for r in records:
        row: dict[str, Any] = {
            "text_id": str(r["text_id"]),
            "prefix_len": r["prefix_len"],
            "char_len": int(r["char_len"]),
            "prefix_truncated": bool(r["prefix_truncated"]),
            "y_topic": str(r["y_topic"]),
            "top10_topics_json": _json.dumps(r["top10_topics"], ensure_ascii=False),
        }
        for k, v in r["channels"].items():
            row[f"ch_{k}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def records_from_wide_parquet_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Inverse of :func:`records_to_wide_parquet_df`."""
    import json as _json

    out: list[dict[str, Any]] = []
    ch_cols = [c for c in df.columns if c.startswith("ch_")]
    for _, row in df.iterrows():
        chans = {c[3:]: float(row[c]) for c in ch_cols}
        tops = _json.loads(str(row["top10_topics_json"]))
        out.append(
            {
                "text_id": str(row["text_id"]),
                "prefix_len": row["prefix_len"],
                "char_len": int(row["char_len"]),
                "prefix_truncated": bool(row["prefix_truncated"]),
                "channels": chans,
                "top10_topics": list(tops),
                "y_topic": str(row["y_topic"]),
            }
        )
    return out


def records_to_parquet_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Long-form table for TASK_QUEUE schema."""
    rows = []
    for r in records:
        tid = r["text_id"]
        pl = r["prefix_len"]
        for name, val in r["channels"].items():
            hit_cols = {f"hit_at_{k}": int(r["y_topic"] in r["top10_topics"][:k]) for k in (1, 3, 5, 10)}
            rows.append(
                {
                    "text_id": tid,
                    "prefix_len": pl,
                    "channel": name,
                    "conf": float(val),
                    "tau_q": np.nan,
                    "shown": np.nan,
                    **hit_cols,
                    "top10_jaccard_with_prev": r.get("jaccard_top10_with_prev"),
                }
            )
    return pd.DataFrame(rows)
