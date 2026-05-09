"""Inference telemetry: daily Parquet rows for /predict and /feedback (TASK-032)."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical column names / types documented in docs/inference_logging.md and
# docs/inference_logging_schema.json

LOG_COLUMNS: tuple[str, ...] = (
    "request_hash",
    "timestamp_utc",
    "date_utc",
    "text_len",
    "raw_text",
    "top_k_ids",
    "top_k_scores",
    "top_k_scores_calibrated",
    "chosen_topic",
    "was_in_top_k",
    "latency_ms",
    "model_version",
    "taxonomy_version",
    "prefix_len_used",
    "ux_shown",
    "conf_channel",
    "conf_value",
    "event_type",
)

_ENV_ENABLED = "INFERENCE_LOG_ENABLED"
_ENV_DIR = "INFERENCE_LOG_DIR"
_ENV_RAW_TEXT = "INFERENCE_LOG_RAW_TEXT"


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() in {"1", "true", "True", "yes", "YES", "on", "ON"}


def floor_timestamp_minute_utc(when: datetime) -> str:
    """UTC instant truncated to minute, used in request_hash stability."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    else:
        when = when.astimezone(timezone.utc)
    floored = when.replace(second=0, microsecond=0)
    return floored.strftime("%Y-%m-%dT%H:%M:00Z")


def compute_request_hash(text: str, timestamp_floor_minute: str) -> str:
    """sha256(text + newline + timestamp_floor_minute), prefixed with sha256:."""
    payload = f"{text}\n{timestamp_floor_minute}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def ensure_parquet_engine() -> str:
    """Return pandas parquet engine name or raise with TASK-042 hygiene hint."""
    import pandas as pd

    try:
        eng = pd.io.parquet.get_engine("pyarrow")
    except ImportError as e:  # pragma: no cover - env-specific
        raise RuntimeError(
            "Inference logging requires a Parquet engine (pyarrow via pandas). "
            "Install pyarrow or align deps per TASK-042 dependency manifest hygiene."
        ) from e
    _ = eng  # use engine side-effect
    return "pyarrow"


@dataclass
class InferenceLogConfig:
    enabled: bool
    log_dir: Path
    store_raw_text: bool


def load_log_config(repo_root: Path | None = None) -> InferenceLogConfig:
    enabled = env_flag(_ENV_ENABLED, default=True)
    raw = env_flag(_ENV_RAW_TEXT, default=False)
    dir_raw = os.environ.get(_ENV_DIR, "reports/inference_logs").strip()
    base = Path(dir_raw)
    if not base.is_absolute():
        root = repo_root or Path(__file__).resolve().parents[2]
        base = (root / base).resolve()
    return InferenceLogConfig(enabled=enabled, log_dir=base, store_raw_text=raw)


class InferenceLogWriter:
    """Append one row per event by reading/writing the daily Parquet file."""

    def __init__(self, cfg: InferenceLogConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._engine: str | None = None

    @property
    def config(self) -> InferenceLogConfig:
        return self._cfg

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def _engine_name(self) -> str:
        if self._engine is None:
            self._engine = ensure_parquet_engine()
        return self._engine

    def daily_path(self, date_utc: str) -> Path:
        """YYYY-MM-DD.parquet under configured log directory."""
        return self._cfg.log_dir / f"{date_utc}.parquet"

    def append_row(self, row: dict[str, Any]) -> None:
        if not self._cfg.enabled:
            return
        missing = [k for k in LOG_COLUMNS if k not in row]
        if missing:
            raise ValueError(f"Inference log row missing keys: {missing}")
        extra = [k for k in row if k not in LOG_COLUMNS]
        if extra:
            raise ValueError(f"Inference log row has unknown keys: {extra}")

        date_utc = str(row["date_utc"])
        path = self.daily_path(date_utc)
        self._cfg.log_dir.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        new_df = pd.DataFrame([{k: row[k] for k in LOG_COLUMNS}])

        with self._lock:
            if path.is_file():
                old = pd.read_parquet(path, engine=self._engine_name())
                out = pd.concat([old, new_df], ignore_index=True)
                out = out[list(LOG_COLUMNS)]
            else:
                out = new_df[list(LOG_COLUMNS)]
            out.to_parquet(path, engine=self._engine_name(), index=False)

        logger.debug("Wrote inference log row to %s", path)


_writer_lock = threading.Lock()
_writer_singleton: InferenceLogWriter | None = None


def get_inference_log_writer(repo_root: Path | None = None) -> InferenceLogWriter:
    """Process-wide writer (config from env at first call)."""
    global _writer_singleton
    with _writer_lock:
        if _writer_singleton is None:
            _writer_singleton = InferenceLogWriter(load_log_config(repo_root))
        return _writer_singleton


def reset_inference_log_writer_for_tests() -> None:
    """Test helper: clear singleton so the next get_* picks up fresh env."""
    global _writer_singleton
    with _writer_lock:
        _writer_singleton = None
