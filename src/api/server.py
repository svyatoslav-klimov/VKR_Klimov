"""FastAPI application: health, readiness, predict, feedback stub."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.logging import (
    compute_request_hash,
    floor_timestamp_minute_utc,
    get_inference_log_writer,
    InferenceLogWriter,
)
from src.api.predictor import OrllmPredictor, build_predictor_from_env
from src.api.schema import (
    FeedbackRequest,
    FeedbackResponse,
    HealthzResponse,
    PredictRequest,
    PredictResponse,
    ReadyzResponse,
)

logger = logging.getLogger(__name__)


class AppState:
    predictor: OrllmPredictor | None = None
    startup_error: str | None = None


def get_state(request: Request) -> AppState:
    return request.app.state.orllm


def get_predictor(request: Request) -> OrllmPredictor:
    st = get_state(request)
    if st.predictor is None:
        detail = st.startup_error or "model not loaded"
        raise HTTPException(status_code=503, detail=detail)
    return st.predictor


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_timestamp_utc(when: datetime) -> str:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    else:
        when = when.astimezone(timezone.utc)
    return when.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _predict_log_row(
    *,
    writer: InferenceLogWriter,
    request_hash: str,
    when: datetime,
    text: str,
    text_len: int,
    resp: PredictResponse | None,
    log_sink: dict[str, Any],
    event_type: str,
    latency_ms: float | None,
) -> dict[str, Any]:
    cfg = writer.config
    raw_val: str | None = text if cfg.store_raw_text else None

    top_ids = list(log_sink.get("top_k_ids", []))
    top_scores = list(log_sink.get("top_k_scores", []))
    top_cal = list(log_sink.get("top_k_scores_calibrated", []))
    prefix_len = int(log_sink.get("prefix_len_used", text_len))
    conf_ch: str | None = log_sink.get("conf_channel")
    conf_val: float | None = log_sink.get("conf_value")
    if isinstance(conf_val, (int, float)):
        conf_val = float(conf_val)
    else:
        conf_val = None

    mv = ""
    tv = ""
    ux = False
    if resp is not None:
        mv = resp.model_version
        tv = resp.taxonomy_version
        ux = bool(resp.shown)
        if resp.conf_channel is not None:
            conf_ch = resp.conf_channel
        if resp.conf_value is not None:
            conf_val = float(resp.conf_value)
        if latency_ms is None:
            latency_ms = float(resp.latency_ms)

    date_utc = when.strftime("%Y-%m-%d")
    return {
        "request_hash": request_hash,
        "timestamp_utc": _iso_timestamp_utc(when),
        "date_utc": date_utc,
        "text_len": int(text_len),
        "raw_text": raw_val,
        "top_k_ids": top_ids,
        "top_k_scores": top_scores,
        "top_k_scores_calibrated": top_cal,
        "chosen_topic": None,
        "was_in_top_k": None,
        "latency_ms": latency_ms,
        "model_version": mv,
        "taxonomy_version": tv,
        "prefix_len_used": prefix_len,
        "ux_shown": ux,
        "conf_channel": conf_ch if conf_ch is not None else "",
        "conf_value": conf_val,
        "event_type": event_type,
    }


def _feedback_log_row(
    *,
    writer: InferenceLogWriter,
    body: FeedbackRequest,
    when: datetime,
) -> dict[str, Any]:
    date_utc = when.strftime("%Y-%m-%d")
    return {
        "request_hash": body.request_hash.strip(),
        "timestamp_utc": _iso_timestamp_utc(when),
        "date_utc": date_utc,
        "text_len": 0,
        "raw_text": None,
        "top_k_ids": [],
        "top_k_scores": [],
        "top_k_scores_calibrated": [],
        "chosen_topic": body.chosen_topic_id.strip(),
        "was_in_top_k": None,
        "latency_ms": None,
        "model_version": "",
        "taxonomy_version": "",
        "prefix_len_used": 0,
        "ux_shown": False,
        "conf_channel": "",
        "conf_value": None,
        "event_type": "feedback",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    st = AppState()
    app.state.orllm = st
    try:
        pred = build_predictor_from_env()
        pred.load_model()
        st.predictor = pred
        logger.info("Inference model ready: %s", pred.model_version)
    except Exception as e:
        st.startup_error = str(e)
        logger.exception("Model startup failed: %s", e)
    yield


app = FastAPI(
    title="ORLLM Inference",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthzResponse)
def healthz() -> HealthzResponse:
    return HealthzResponse(
        status="ok",
        service="orllm-inference",
        task_id="TASK-031",
    )


@app.get("/readyz", response_model=ReadyzResponse)
def readyz(request: Request) -> ReadyzResponse:
    st = get_state(request)
    p = st.predictor
    if p is None:
        raise HTTPException(
            status_code=503,
            detail=st.startup_error or "model not loaded",
        )
    return ReadyzResponse(
        status="ready",
        model_version=p.model_version,
        taxonomy_version=p.taxonomy_version,
        classes_count=p.classes_count,
        device=p.device,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    predictor: Annotated[OrllmPredictor, Depends(get_predictor)],
) -> PredictResponse:
    writer = get_inference_log_writer()
    log_sink: dict[str, Any] = {}
    text = body.text.strip()
    when = _utc_now()
    minute_floor = floor_timestamp_minute_utc(when)
    req_hash = compute_request_hash(text, minute_floor)

    def log_predict(resp: PredictResponse | None, *, lat_ms: float | None = None) -> None:
        try:
            row = _predict_log_row(
                writer=writer,
                request_hash=req_hash,
                when=when,
                text=text,
                text_len=len(text),
                resp=resp,
                log_sink=log_sink,
                event_type="predict",
                latency_ms=lat_ms,
            )
            writer.append_row(row)
        except Exception:
            logger.exception("Inference logging failed (predict path)")

    if not text:
        if writer.enabled:
            try:
                row = _predict_log_row(
                    writer=writer,
                    request_hash=req_hash,
                    when=when,
                    text=text,
                    text_len=0,
                    resp=None,
                    log_sink=log_sink,
                    event_type="predict",
                    latency_ms=None,
                )
                row["model_version"] = predictor.model_version
                row["taxonomy_version"] = predictor.taxonomy_version
                writer.append_row(row)
            except Exception:
                logger.exception("Inference logging failed (empty text)")
        raise HTTPException(status_code=400, detail="text must be non-empty")

    max_len = predictor.max_text_len
    if len(text) > max_len:
        if writer.enabled:
            try:
                row = _predict_log_row(
                    writer=writer,
                    request_hash=req_hash,
                    when=when,
                    text=text,
                    text_len=len(text),
                    resp=None,
                    log_sink=log_sink,
                    event_type="predict",
                    latency_ms=None,
                )
                row["model_version"] = predictor.model_version
                row["taxonomy_version"] = predictor.taxonomy_version
                writer.append_row(row)
            except Exception:
                logger.exception("Inference logging failed (too long text)")
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds max_text_len={max_len}",
        )

    try:
        resp = predictor.predict(
            text,
            return_calibrated=body.return_calibrated,
            return_meta=body.return_meta,
            _log_sink=log_sink,
        )
        log_predict(resp)
        return resp
    except RuntimeError as e:
        msg = str(e)
        if writer.enabled:
            try:
                row = _predict_log_row(
                    writer=writer,
                    request_hash=req_hash,
                    when=when,
                    text=text,
                    text_len=len(text),
                    resp=None,
                    log_sink=log_sink,
                    event_type="predict",
                    latency_ms=None,
                )
                row["model_version"] = predictor.model_version
                row["taxonomy_version"] = predictor.taxonomy_version
                writer.append_row(row)
            except Exception:
                logger.exception("Inference logging failed (runtime error)")
        if "closed-set violation" in msg:
            raise HTTPException(status_code=500, detail=msg) from e
        raise HTTPException(status_code=500, detail=msg) from e


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(body: FeedbackRequest) -> FeedbackResponse:
    writer = get_inference_log_writer()
    if writer.enabled:
        try:
            row = _feedback_log_row(writer=writer, body=body, when=_utc_now())
            writer.append_row(row)
        except Exception:
            logger.exception("Inference logging failed (feedback)")
    return FeedbackResponse(accepted=True)


def create_app_for_test(overrides: dict[str, Any] | None = None) -> FastAPI:
    """Test helper: build app without heavy lifespan (inject predictor via overrides)."""
    overrides = overrides or {}
    app_test = FastAPI(title="ORLLM Inference (test)")

    st = AppState()
    st.predictor = overrides.get("predictor")  # type: ignore[assignment]
    st.startup_error = overrides.get("startup_error")
    app_test.state.orllm = st

    app_test.add_api_route("/healthz", healthz, methods=["GET"])
    app_test.add_api_route("/readyz", readyz, methods=["GET"])
    app_test.add_api_route("/predict", predict, methods=["POST"])
    app_test.add_api_route("/feedback", feedback, methods=["POST"])
    return app_test
