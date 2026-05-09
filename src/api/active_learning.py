"""Build daily active-learning candidate queues from inference Parquet logs (TASK-033)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.api.logging import LOG_COLUMNS

logger = logging.getLogger(__name__)

TASK_ID = "TASK-033"
QUEUE_TITLE = "ORLLM active-learning queue (TASK-033)"


def repo_root_from_here() -> Path:
    """Repository root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def _parse_utc_timestamp(iso: str) -> datetime:
    """Parse ISO-8601 UTC timestamps as used in inference logs."""
    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_tail_topic_ids(topics_path: Path) -> set[str]:
    """Topic IDs with count_total < 10 (tail bucket for TASK-033)."""
    import pandas as pd

    df = pd.read_csv(topics_path, encoding="utf-8")
    if "topic_id" not in df.columns or "count_total" not in df.columns:
        raise ValueError(f"topics.csv missing columns: {topics_path}")
    tail = df[df["count_total"] < 10]
    return set(str(x) for x in tail["topic_id"].tolist())


def resolve_tau_low_conf(
    repo_root: Path,
    cli_value: float | None,
) -> float:
    """τ for low_conf: CLI > prefix_best/final picked.tau_value > 0.5."""
    if cli_value is not None:
        return float(cli_value)
    for rel in ("configs/prefix_best.yaml", "configs/final.yaml"):
        p = repo_root / rel
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        picked = cfg.get("picked") or {}
        if "tau_value" in picked:
            return float(picked["tau_value"])
    return 0.5


def _cell_to_list(val: Any) -> list[Any]:
    if val is None:
        return []
    try:
        import numpy as np

        if isinstance(val, float) and np.isnan(val):
            return []
    except ImportError:  # pragma: no cover
        pass
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, list):
        return val
    return list(val) if val else []


def _first_calibrated(cal: list[Any]) -> float | None:
    if not cal:
        return None
    v = cal[0]
    if v is None:
        return None
    try:
        import pandas as pd

        if pd.isna(v):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    try:
        import numpy as np

        if isinstance(v, float) and np.isnan(v):
            return None
    except ImportError:  # pragma: no cover
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_category_flags(
    *,
    top_k_ids: list[str],
    top_k_scores: list[float],
    top_k_scores_calibrated: list[Any],
    chosen_topic: str | None,
    has_feedback: bool,
    tau_low_margin: float,
    tau_low_conf: float,
    tail_set: set[str],
    late_start: datetime | None,
    timestamp_utc: str,
) -> tuple[list[str], dict[str, bool]]:
    """Return (category_codes, bool flags)."""
    is_outside_top_k = False
    if has_feedback and chosen_topic is not None:
        is_outside_top_k = chosen_topic not in top_k_ids

    is_low_margin = False
    if len(top_k_scores) >= 2:
        gap = float(top_k_scores[0]) - float(top_k_scores[1])
        is_low_margin = gap < tau_low_margin

    is_low_conf = False
    cal0 = _first_calibrated(top_k_scores_calibrated)
    if cal0 is not None:
        is_low_conf = cal0 < tau_low_conf

    is_tail = False
    if top_k_ids:
        is_tail = top_k_ids[0] in tail_set

    is_late_period = False
    if late_start is not None:
        is_late_period = _parse_utc_timestamp(timestamp_utc) >= late_start

    cats: list[str] = []
    if is_outside_top_k:
        cats.append("outside_top_k")
    if is_low_margin:
        cats.append("low_margin")
    if is_low_conf:
        cats.append("low_conf")
    if is_tail:
        cats.append("tail_predicted")
    if is_late_period:
        cats.append("late_period")

    flags = {
        "is_outside_top_k": is_outside_top_k,
        "is_low_margin": is_low_margin,
        "is_low_conf": is_low_conf,
        "is_tail": is_tail,
        "is_late_period": is_late_period,
    }
    return cats, flags


def compute_priority(flags: dict[str, bool]) -> float:
    """Weighted priority (low_margin excluded from weighting)."""
    return (
        0.4 * float(flags["is_outside_top_k"])
        + 0.2 * float(flags["is_low_conf"])
        + 0.2 * float(flags["is_tail"])
        + 0.2 * float(flags["is_late_period"])
    )


def _seed_request_hash(text_key: str, topic_truth: str) -> str:
    payload = f"{text_key}{topic_truth}".encode("utf-8")
    return "seed:" + hashlib.sha256(payload).hexdigest()


def seed_items_from_parquet(
    seed_path: Path,
    *,
    tail_set: set[str],
) -> list[dict[str, Any]]:
    """Map TASK-021 confident_errors.parquet rows to queue items."""
    import pandas as pd

    df = pd.read_parquet(seed_path, engine="pyarrow")
    items: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        true_tid = str(row["true_topic_id"])
        test_idx = int(row["test_idx"])
        top1 = str(row["top1_pred"])
        raw_sc = float(row["top1_score_raw"])
        cal = row.get("conf_calibrated")
        cal_f = _first_calibrated([cal])
        cal_list: list[float | None] = [cal_f, None]
        text_key = f"{test_idx}"
        rqh = _seed_request_hash(text_key, true_tid)
        excerpt = row.get("text_excerpt")
        if excerpt is None or pd.isna(excerpt):
            text_len = 0
        else:
            text_len = len(str(excerpt))
        flags = {
            "is_outside_top_k": True,
            "is_low_margin": False,
            "is_low_conf": False,
            "is_tail": top1 in tail_set,
            "is_late_period": False,
        }
        items.append(
            {
                "request_hash": rqh,
                "timestamp_utc": "1970-01-01T00:00:00.000000Z",
                "text_len": text_len,
                "top_k_ids": [top1],
                "top_k_scores": [raw_sc],
                "top_k_scores_calibrated": cal_list,
                "chosen_topic": true_tid,
                "was_in_top_k": False,
                "categories": ["outside_top_k", "seed_iter2_confident_error"],
                "priority": 0.4,
                "source": "seed_iter2_confident_error",
                "model_version": "",
                "taxonomy_version": "",
                **flags,
            }
        )
    return items


def read_day_log(log_path: Path) -> Any:
    """Load Parquet for one UTC day; return empty DataFrame if missing."""
    import pandas as pd

    if not log_path.is_file():
        return pd.DataFrame(columns=list(LOG_COLUMNS))
    df = pd.read_parquet(log_path, engine="pyarrow")
    return df


def build_queue_from_frames(
    df: Any,
    *,
    date_utc: str,
    tail_set: set[str],
    tau_low_conf: float,
    tau_low_margin: float,
    late_start: datetime | None,
    include_seed: bool,
    seed_path: Path | None,
) -> dict[str, Any]:
    """Core builder: predict rows -> queue document (no seed path I/O)."""
    items: list[dict[str, Any]] = []
    if len(df) == 0:
        pass
    else:
        feedback_rows = df[df["event_type"] == "feedback"]
        predict_rows = df[df["event_type"] == "predict"]

        fb_map: dict[str, dict[str, Any]] = {}
        for _, r in feedback_rows.iterrows():
            h = str(r["request_hash"])
            fb_map[h] = {
                "chosen_topic": None if r["chosen_topic"] is None else str(r["chosen_topic"]),
                "timestamp_utc": str(r["timestamp_utc"]),
            }

        for _, r in predict_rows.iterrows():
            req_hash = str(r["request_hash"])
            ts = str(r["timestamp_utc"])
            fb = fb_map.get(req_hash)
            has_feedback = fb is not None
            chosen = fb["chosen_topic"] if fb else None

            top_k_ids = [str(x) for x in _cell_to_list(r["top_k_ids"])]
            top_k_scores = []
            for x in _cell_to_list(r["top_k_scores"]):
                try:
                    top_k_scores.append(float(x))
                except (TypeError, ValueError):
                    top_k_scores.append(0.0)
            top_k_cal_raw = _cell_to_list(r["top_k_scores_calibrated"])

            was_in_top_k: bool | None
            if not has_feedback:
                was_in_top_k = None
            else:
                was_in_top_k = chosen in top_k_ids if chosen is not None else None

            cats, flags = compute_category_flags(
                top_k_ids=top_k_ids,
                top_k_scores=top_k_scores,
                top_k_scores_calibrated=top_k_cal_raw,
                chosen_topic=chosen,
                has_feedback=has_feedback,
                tau_low_margin=tau_low_margin,
                tau_low_conf=tau_low_conf,
                tail_set=tail_set,
                late_start=late_start,
                timestamp_utc=ts,
            )
            if not cats:
                continue

            text_len = int(r["text_len"]) if r["text_len"] == r["text_len"] else 0
            cal_out: list[float | None] = []
            for x in top_k_cal_raw:
                if x is None:
                    cal_out.append(None)
                else:
                    try:
                        v = float(x)
                        if v != v:  # NaN
                            cal_out.append(None)
                        else:
                            cal_out.append(v)
                    except (TypeError, ValueError):
                        cal_out.append(None)

            items.append(
                {
                    "request_hash": req_hash,
                    "timestamp_utc": ts,
                    "text_len": text_len,
                    "top_k_ids": top_k_ids,
                    "top_k_scores": top_k_scores,
                    "top_k_scores_calibrated": cal_out,
                    "chosen_topic": chosen,
                    "was_in_top_k": was_in_top_k,
                    "categories": cats,
                    "priority": compute_priority(flags),
                    "source": "predict_log",
                    "model_version": str(r["model_version"] or ""),
                    "taxonomy_version": str(r["taxonomy_version"] or ""),
                    **flags,
                }
            )

    if include_seed and seed_path is not None and seed_path.is_file():
        items.extend(seed_items_from_parquet(seed_path, tail_set=tail_set))

    def sort_key(it: dict[str, Any]) -> tuple[float, float, str]:
        ts = _parse_utc_timestamp(str(it["timestamp_utc"])).timestamp()
        return (-float(it["priority"]), ts, str(it["request_hash"]))

    items.sort(key=sort_key)

    return {
        "title": QUEUE_TITLE,
        "task_id": TASK_ID,
        "date_utc": date_utc,
        "items": items,
    }


def build_queue_for_date(
    *,
    repo_root: Path,
    log_dir: Path,
    date_utc: str,
    topics_path: Path,
    tau_low_conf_cli: float | None,
    tau_low_margin: float,
    late_period_start_iso: str | None,
    include_seed: bool,
    seed_parquet: Path | None,
) -> dict[str, Any]:
    """Load logs + topics, build queue dict (caller writes JSON)."""
    tail_set = load_tail_topic_ids(topics_path)
    tau_low_conf = resolve_tau_low_conf(repo_root, tau_low_conf_cli)
    late_start: datetime | None = None
    if late_period_start_iso:
        late_start = _parse_utc_timestamp(late_period_start_iso + "T00:00:00Z")

    log_path = log_dir / f"{date_utc}.parquet"
    df = read_day_log(log_path)

    default_seed = repo_root / "reports/error_analysis/confident_errors.parquet"
    seed_path = seed_parquet if seed_parquet is not None else default_seed

    return build_queue_from_frames(
        df,
        date_utc=date_utc,
        tail_set=tail_set,
        tau_low_conf=tau_low_conf,
        tau_low_margin=tau_low_margin,
        late_start=late_start,
        include_seed=include_seed,
        seed_path=seed_path if include_seed else None,
    )


def assert_no_raw_text(obj: Any) -> None:
    """Recursively ensure no raw_text key (privacy)."""
    if isinstance(obj, dict):
        if "raw_text" in obj:
            raise ValueError("queue must not contain raw_text")
        for v in obj.values():
            assert_no_raw_text(v)
    elif isinstance(obj, list):
        for x in obj:
            assert_no_raw_text(x)


def write_queue_json(path: Path, doc: dict[str, Any]) -> None:
    """Write UTF-8 JSON with stable formatting."""
    assert_no_raw_text(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=QUEUE_TITLE)
    p.add_argument("--logs", type=Path, required=True, help="Directory with YYYY-MM-DD.parquet")
    p.add_argument("--out", type=Path, required=True, help="Output directory for queue JSON")
    p.add_argument("--date", type=str, required=True, help="UTC date YYYY-MM-DD")
    p.add_argument(
        "--topics",
        type=Path,
        default=None,
        help="Override path to topics.csv (default: <repo>/data/processed/topics.csv)",
    )
    p.add_argument(
        "--seed-parquet",
        type=Path,
        default=None,
        help="TASK-021 confident_errors.parquet (default: reports/error_analysis/...)",
    )
    p.add_argument(
        "--late-period-start",
        type=str,
        default=None,
        help="YYYY-MM-DD; timestamps on/after this day UTC are late_period "
        "(omit: no late_period flags)",
    )
    p.add_argument(
        "--tau-low-conf",
        type=float,
        default=None,
        help="Override low_conf threshold (default: prefix_best/final or 0.5)",
    )
    p.add_argument(
        "--tau-low-margin",
        type=float,
        default=0.05,
        help="Margin threshold for low_margin (default 0.05)",
    )
    p.add_argument(
        "--include-seed",
        action="store_true",
        help="Append TASK-021 confident errors as seed_iter2_confident_error",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    root = repo_root_from_here()
    topics = args.topics if args.topics is not None else root / "data/processed/topics.csv"
    out_path = args.out / f"{args.date}.json"
    try:
        doc = build_queue_for_date(
            repo_root=root,
            log_dir=args.logs,
            date_utc=args.date,
            topics_path=topics,
            tau_low_conf_cli=args.tau_low_conf,
            tau_low_margin=float(args.tau_low_margin),
            late_period_start_iso=args.late_period_start,
            include_seed=args.include_seed,
            seed_parquet=args.seed_parquet,
        )
        write_queue_json(out_path, doc)
    except Exception as e:  # pragma: no cover - CLI returns non-zero
        print(f"active_learning: error: {e}", file=sys.stderr)
        return 1
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
