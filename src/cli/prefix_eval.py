"""CLI: prefix-eval (TASK-012)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.prepare import (
    filter_by_preprocessed_text,
    load_taxonomy_aliases,
    prepare_split,
    preprocess_text_padded,
    theme_to_topic_id_map,
)
from src.eval.prefix_eval import (
    GridCell,
    build_prefix_show_rates_for_policy,
    build_tau_grid,
    compute_val_test_parity,
    evaluate_prefix,
    normalize_prefix_len_tag,
    pick_optimal_ux_policy,
    plot_prefix_rocs,
    records_from_wide_parquet_df,
    records_to_parquet_df,
    records_to_wide_parquet_df,
    verify_cross_prefix_monotonicity,
)
from src.models.hybrid import HybridModel
from src.utils.leaderboard import append_row
from src.utils.run_manifest import build_manifest, write_manifest

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        out = yaml.safe_load(f)
    return out if isinstance(out, dict) else {}


def _prepare_eval_df(config: dict[str, Any], split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_cfg = config["data"]
    train_path = Path(str(data_cfg["train_path"]))
    val_path = Path(str(data_cfg["val_path"]))
    test_path = Path(str(data_cfg["test_path"]))
    topics_path = Path(str(data_cfg["topics_path"]))
    tax = Path(str(data_cfg.get("taxonomy_yaml", "configs/taxonomy.yaml")))

    split_path = val_path if split == "val" else test_path
    raw_train = pd.read_parquet(train_path)
    raw_eval = pd.read_parquet(split_path)

    aliases = load_taxonomy_aliases(str(tax))
    tmap, tax_ver = theme_to_topic_id_map(str(topics_path), str(tax))
    train_df0 = prepare_split(raw_train, tmap, aliases, str(tax))
    eval_df0 = prepare_split(raw_eval, tmap, aliases, str(tax))
    pre = config.get("preprocessing")
    pre_d = pre if isinstance(pre, dict) else None
    if pre_d:
        train_df = filter_by_preprocessed_text(train_df0, pre_d)
    else:
        train_df = train_df0
    if pre_d and split in ("val", "test"):
        eval_df = eval_df0.copy()
        eval_df["text"] = [
            preprocess_text_padded(str(t), pre_d) for t in eval_df0["text"].astype(str).tolist()
        ]
    else:
        eval_df = eval_df0

    y_train = train_df["topic_id"].astype(str)
    all_train_topics = set(y_train.tolist())
    known_mask = eval_df["topic_id"].astype(str).isin(all_train_topics)
    eval_df = eval_df.loc[known_mask].reset_index(drop=True)
    meta = {"taxonomy_ver": str(tax_ver), "n_rows": len(eval_df)}
    return eval_df, meta


def _metrics_for_leaderboard(recall10: float, split: str) -> dict[str, Any]:
    r = float(recall10)
    return {
        "split": split,
        "unseen_policy": "known_only",
        "recall_at_k": {"1": r, "3": r, "5": r, "10": r},
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "mrr_at_10": r,
        "ndcg_at_10": r,
        "head_mid_tail": {},
        "latency_ms": {},
    }


def require_test_policy(split: str, ux_policy: str | None) -> Path | None:
    """R-007 guard: test split must reuse val-picked tau (yaml), never re-pick."""
    if split != "test":
        return None
    pb = Path("configs") / "prefix_best.yaml"
    if ux_policy:
        return Path(ux_policy).resolve()
    if pb.is_file():
        return pb
    raise RuntimeError(
        "R-007: tau was picked on val only; provide configs/prefix_best.yaml "
        "or --ux-policy for split=test (no re-pick on test)."
    )


def run_prefix_cli(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg_path = Path(args.config).resolve()
    config = _load_yaml(cfg_path)
    split = str(args.split)

    prefix_cfg = config.get("prefix_eval") or {}
    if not isinstance(prefix_cfg, dict):
        prefix_cfg = {}

    prefix_lengths = prefix_cfg.get("prefix_lengths") or [
        10,
        20,
        30,
        50,
        100,
        "full",
    ]
    channels = prefix_cfg.get("channels") or [
        "max_score",
        "margin",
        "entropy",
        "normalized_gap",
        "max_score_calibrated",
    ]
    l_min_grid = list(prefix_cfg.get("l_min_grid") or [10, 20, 30, 50, 100])
    tau_q_grid = list(prefix_cfg.get("tau_quantile_grid") or [0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    bootstrap_n = int(prefix_cfg.get("bootstrap_n", 1000))
    ci_level = float(prefix_cfg.get("ci_level", 0.95))
    show_floor = float(prefix_cfg.get("show_rate_floor", 0.50))
    parity_tol = float(prefix_cfg.get("val_test_parity_tolerance", 0.02))
    mono_tol = float(prefix_cfg.get("cross_prefix_monotonicity_tolerance", 0.02))

    best_dir = Path(args.best or prefix_cfg.get("best_hybrid_run_dir") or "").resolve()
    if not best_dir.is_dir():
        raise FileNotFoundError(f"hybrid artifacts dir not found: {best_dir}")

    cal_raw = args.calibrator or prefix_cfg.get("calibrator_path")
    calibrator_path: Path | None = Path(cal_raw).resolve() if cal_raw else None
    if calibrator_path is not None and not calibrator_path.is_file():
        raise FileNotFoundError(f"calibrator not found: {calibrator_path}")

    calibration_run_id = str(prefix_cfg.get("calibration_run_id", "20260429_full_cal_v1"))
    parent_run_id = str(prefix_cfg.get("parent_run_id", "20260427_171719_hybrid_weighted_score_minmax"))

    policy_path = require_test_policy(split, getattr(args, "ux_policy", None))

    out_arg = getattr(args, "out", None)
    if out_arg:
        out_root = Path(out_arg).resolve()
        run_id = out_root.name
        out_dir = out_root
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_prefix_{split}"
        out_dir = Path("reports") / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = getattr(args, "device", None) or prefix_cfg.get("device", "auto")
    bs_arg = getattr(args, "batch_size", None)
    batch_size_opt = int(bs_arg) if bs_arg is not None else None

    notes_file = getattr(args, "device_notes_file", None)
    device_notes_extra = ""
    if notes_file:
        nfp = Path(str(notes_file))
        if nfp.is_file():
            device_notes_extra = nfp.read_text(encoding="utf-8").strip()

    eval_df, prep_meta = _prepare_eval_df(config, split)
    logger.info("split %s rows=%s", split, len(eval_df))

    model = HybridModel.load(best_dir, device=device, batch_size=batch_size_opt)
    records, per_prefix = evaluate_prefix(
        model,
        eval_df,
        prefix_lengths,
        channels,
        str(calibrator_path) if calibrator_path else None,
    )

    picked: GridCell | None = None
    grid_json: list[dict[str, Any]] = []
    parity_block: dict[str, Any] | None = None
    mono_ok = True

    wide_path = out_dir / "prefix_records_wide.parquet"
    records_to_wide_parquet_df(records).to_parquet(wide_path, index=False)

    pq_path = out_dir / "prefix_curves.parquet"
    records_to_parquet_df(records).to_parquet(pq_path, index=False)

    if split == "val":
        grid = build_tau_grid(
            records,
            l_min_grid,
            tau_q_grid,
            channels,
            bootstrap_n=bootstrap_n,
            ci_level=ci_level,
        )
        picked = pick_optimal_ux_policy(grid, show_rate_floor=show_floor)
        grid_json = [
            {
                "L_min": c.L_min,
                "channel": c.channel,
                "tau_quantile": c.tau_quantile,
                "tau_value": c.tau_value,
                "show_rate_val": c.show_rate_val,
                "recall_at_10_given_shown_val": c.recall_at_10_given_shown_val,
                "recall_at_10_overall_val": c.recall_at_10_overall_val,
                "ci95_low": c.ci95_low,
                "ci95_high": c.ci95_high,
            }
            for c in grid
        ]

        sr_by_pl = build_prefix_show_rates_for_policy(records, picked, prefix_lengths)
        mono_rates = []
        for pl in prefix_lengths:
            kk = normalize_prefix_len_tag(pl)
            mono_rates.append(float(sr_by_pl.get(kk, 0.0)))
        mono_ok = verify_cross_prefix_monotonicity(mono_rates, tolerance=mono_tol)

        pb_out = Path("configs") / "prefix_best.yaml"
        best_payload = {
            "picked": {
                "L_min": picked.L_min,
                "channel": picked.channel,
                "tau_value": picked.tau_value,
                "tau_quantile": picked.tau_quantile,
                "show_rate_val": picked.show_rate_val,
                "recall_at_10_given_shown_val": picked.recall_at_10_given_shown_val,
                "ci95_low_val": picked.ci95_low,
                "ci95_high_val": picked.ci95_high,
            },
            "calibration_run_id": calibration_run_id,
            "parent_run_id": parent_run_id,
            "val_run_id": run_id,
            "val_records_wide_parquet": str(wide_path.resolve()).replace("\\", "/"),
        }
        with pb_out.open("w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(best_payload, f, allow_unicode=True, sort_keys=False)

        plot_paths = {
            ch: Path("reports") / "figures" / f"prefix_roc_{ch}.png" for ch in channels
        }
        plot_prefix_rocs(records, channels, plot_paths, L_min=picked.L_min)

    else:
        assert policy_path is not None
        pol = _load_yaml(policy_path)
        pp = pol.get("picked") if isinstance(pol.get("picked"), dict) else pol
        picked = GridCell(
            L_min=int(pp["L_min"]),
            channel=str(pp["channel"]),
            tau_quantile=float(pp.get("tau_quantile", 0.0)),
            tau_value=float(pp["tau_value"]),
            show_rate_val=float(pp.get("show_rate_val", 0.0)),
            recall_at_10_given_shown_val=float(pp.get("recall_at_10_given_shown_val", 0.0)),
            recall_at_10_overall_val=float(pp.get("recall_at_10_overall_val", 0.0)),
            ci95_low=float(pp.get("ci95_low_val", pp.get("ci95_low", 0.0))),
            ci95_high=float(pp.get("ci95_high_val", pp.get("ci95_high", 0.0))),
        )

        val_wide_s = pol.get("val_records_wide_parquet") or pp.get("val_records_wide_parquet")
        if not val_wide_s:
            raise RuntimeError("prefix_best.yaml must contain val_records_wide_parquet for parity.")
        vw = Path(str(val_wide_s))
        if not vw.is_file():
            raise FileNotFoundError(f"val wide parquet missing: {vw}")
        val_df = pd.read_parquet(vw)
        val_data = records_from_wide_parquet_df(val_df)
        parity_block = compute_val_test_parity(picked, val_data, records, tolerance=parity_tol)

        sr_by_pl = build_prefix_show_rates_for_policy(records, picked, prefix_lengths)
        mono_rates = []
        for pl in prefix_lengths:
            kk = normalize_prefix_len_tag(pl)
            mono_rates.append(float(sr_by_pl.get(kk, 0.0)))
        mono_ok = verify_cross_prefix_monotonicity(mono_rates, tolerance=mono_tol)

    assert picked is not None

    ux_policy_json: dict[str, Any] = {
        "L_min": picked.L_min,
        "channel": picked.channel,
        "tau_value": picked.tau_value,
        "tau_quantile": picked.tau_quantile,
        "show_rate_val": picked.show_rate_val,
        "recall_at_10_given_shown_val": picked.recall_at_10_given_shown_val,
        "show_rate_test": None,
        "recall_at_10_given_shown_test": None,
        "ci95_low_val": picked.ci95_low,
        "ci95_high_val": picked.ci95_high,
        "calibration_run_id": calibration_run_id,
    }

    if split == "test":
        from src.eval.prefix_eval import _eligible_records, _metrics_for_tau

        el_t = _eligible_records(records, picked.L_min)
        sr_t, r_gs_t, _ = _metrics_for_tau(el_t, picked.channel, picked.tau_value)
        ux_policy_json["show_rate_test"] = sr_t
        ux_policy_json["recall_at_10_given_shown_test"] = r_gs_t

    metrics_prefix: dict[str, Any] = {
        "run_id": run_id,
        "split": split,
        "prefix_lengths": prefix_lengths,
        "channels": channels,
        "ux_policy": ux_policy_json,
        "per_prefix": [
            {"len": k, **per_prefix[k]}
            for k in sorted(per_prefix.keys(), key=lambda z: (str(z) == "full", str(z)))
        ],
        "tau_grid_val": grid_json,
        "parity": parity_block,
        "cross_prefix_monotonicity_passed": mono_ok,
        "prep": prep_meta,
        "records_parquet_path": str(pq_path).replace("\\", "/"),
        "records_wide_parquet_path": str(wide_path).replace("\\", "/"),
        "runtime": {
            "encoder_device": str(model.device),
            "encoder_batch_size": int(model.batch_size),
            "sparse_batch_size": int(model.sparse_batch_size),
        },
    }

    mp_path = out_dir / "metrics_prefix.json"
    with mp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics_prefix, f, ensure_ascii=False, indent=2)
        f.write("\n")

    note_path = out_dir / "prefix_decision_note.md"
    _write_decision_note(
        note_path,
        split=split,
        picked=picked,
        mono_ok=mono_ok,
        parity=parity_block,
        encoder_device=str(model.device),
        encoder_batch_size=int(model.batch_size),
        device_notes_extra=device_notes_extra,
    )

    merged_cfg = dict(config)
    merged_cfg.setdefault("data", {})
    manifest = build_manifest(
        run_id,
        "prefix_eval",
        str(cfg_path),
        merged_cfg,
        artifacts={
            "metrics_prefix": str(mp_path).replace("\\", "/"),
            "prefix_curves": str(pq_path).replace("\\", "/"),
            "prefix_records_wide": str(wide_path).replace("\\", "/"),
            "hybrid_dir": str(best_dir).replace("\\", "/"),
        },
        retrieval_meta=None,
        legacy_splits=True,
        splits_policy="time_based",
    )
    manifest["parent_run_id"] = parent_run_id
    manifest["calibration_run_id"] = calibration_run_id
    manifest["prefix_eval"] = {
        "split": split,
        "picked_channel": picked.channel,
        "picked_L_min": picked.L_min,
    }
    mpath = Path("artifacts") / "manifests" / f"{run_id}.json"
    write_manifest(manifest, mpath)

    r_lb = float(picked.recall_at_10_given_shown_val)
    if split == "test" and ux_policy_json.get("recall_at_10_given_shown_test") is not None:
        r_lb = float(ux_policy_json["recall_at_10_given_shown_test"])
    lb_metrics = _metrics_for_leaderboard(r_lb, split)
    notes = (
        f"prefix_eval, picked (L_min={picked.L_min}, channel={picked.channel}, "
        f"tau={picked.tau_value:.6f})"
    )
    append_row(
        Path("reports") / "leaderboard.csv",
        run_id,
        "prefix_eval",
        lb_metrics,
        manifest,
        str(cfg_path).replace("\\", "/"),
        split=split,
        unseen_policy="known_only",
        notes=notes,
    )

    logger.info("wrote %s", mp_path)
    return 0


def _write_decision_note(
    path: Path,
    *,
    split: str,
    picked: GridCell,
    mono_ok: bool,
    parity: dict[str, Any] | None,
    encoder_device: str = "",
    encoder_batch_size: int = 0,
    device_notes_extra: str = "",
) -> None:
    lines = [
        f"# Prefix UX decision ({split})",
        "",
        f"- encoder_device={encoder_device}, encoder_batch_size={encoder_batch_size}",
        f"- L_min={picked.L_min}, channel={picked.channel}, tau={picked.tau_value:.6f} (q={picked.tau_quantile})",
        f"- val show_rate={picked.show_rate_val:.4f}, R@10|shown={picked.recall_at_10_given_shown_val:.4f}",
        f"- bootstrap CI95 [{picked.ci95_low:.4f}, {picked.ci95_high:.4f}]",
        f"- cross_prefix_monotonicity_passed={mono_ok}",
    ]
    if parity:
        lines.append(
            f"- val/test parity delta R@10|shown={parity.get('delta_r10_given_shown')} "
            f"passed={parity.get('parity_passed')}"
        )
    if device_notes_extra:
        lines.extend(["", "## Device selection / fallback", "", device_notes_extra])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    """Entrypoint used by ``python -m src.cli prefix-eval``."""
    return run_prefix_cli(args)
