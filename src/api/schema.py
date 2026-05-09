"""Pydantic schemas for the inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthzResponse(BaseModel):
    status: str
    service: str
    task_id: str


class ReadyzResponse(BaseModel):
    status: str
    model_version: str
    taxonomy_version: str
    classes_count: int
    device: str


class PredictRequest(BaseModel):
    """POST /predict body."""

    text: str
    return_calibrated: bool = True
    return_meta: bool = False

    model_config = ConfigDict(extra="forbid")


class PredictItem(BaseModel):
    topic_id: str
    topic_name: str
    rank: int
    score_raw: float
    score_calibrated: float | None = None


class PredictMeta(BaseModel):
    run_id: str
    calibration_run_id: str | None = None
    prefix_policy_run_id: str | None = None
    prefix_policy: dict[str, Any]


class PredictResponse(BaseModel):
    items: list[PredictItem]
    top10: list[PredictItem]
    shown: bool
    shown_topic_ids: list[str]
    model_version: str
    taxonomy_version: str
    latency_ms: float
    meta: PredictMeta | None = None
    conf_channel: str | None = None
    conf_value: float | None = None


class FeedbackRequest(BaseModel):
    request_hash: str = Field(min_length=1)
    chosen_topic_id: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class FeedbackResponse(BaseModel):
    accepted: bool = True
