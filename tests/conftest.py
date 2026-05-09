"""Shared pytest fixtures for production tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_inference_logging_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid writing Parquet under reports/inference_logs during unrelated tests."""
    monkeypatch.setenv("INFERENCE_LOG_ENABLED", "0")
