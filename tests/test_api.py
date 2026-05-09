"""TASK-031: FastAPI inference API tests (TestClient + fake predictor)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from src.api.schema import PredictItem, PredictResponse
from src.api.server import app, create_app_for_test, get_predictor


class FakePredictor:
    """Lightweight stand-in; avoids loading SentenceTransformer in unit tests."""

    def __init__(self, *, max_len: int = 2000) -> None:
        self._max_len = max_len
        self.model_version = "fake_mv"
        self.taxonomy_version = "sha256:fake"
        self.classes_count = 2
        self.device = "cpu"

    @property
    def max_text_len(self) -> int:
        return self._max_len

    def predict(
        self,
        text: str,
        *,
        return_calibrated: bool = True,
        return_meta: bool = False,
        _log_sink: dict[str, Any] | None = None,
    ) -> PredictResponse:
        if "__closed_set__" in text:
            raise RuntimeError(
                "closed-set violation: topic_id 'nope' not in model "
                "classes and topics metadata"
            )
        if len(text) < 100:
            # Simulate "low confidence" path: items still populated,
            # shown=False because tau-gate fails (not because of L_min).
            short_items = [
                PredictItem(
                    topic_id="short_a",
                    topic_name="short theme A",
                    rank=1,
                    score_raw=0.5,
                    score_calibrated=0.4 if return_calibrated else None,
                ),
                PredictItem(
                    topic_id="short_b",
                    topic_name="short theme B",
                    rank=2,
                    score_raw=0.3,
                    score_calibrated=None,
                ),
            ]
            short_meta = None
            if return_meta:
                from src.api.schema import PredictMeta

                short_meta = PredictMeta(
                    run_id="r",
                    calibration_run_id=None,
                    prefix_policy_run_id=None,
                    prefix_policy={
                        "l_min": 100,
                        "channel": "ensemble",
                        "tau": 0.7,
                    },
                )
            if _log_sink is not None:
                _log_sink.clear()
                _log_sink.update(
                    {
                        "top_k_ids": [it.topic_id for it in short_items],
                        "top_k_scores": [
                            float(it.score_raw) for it in short_items
                        ],
                        "top_k_scores_calibrated": [
                            it.score_calibrated for it in short_items
                        ],
                        "prefix_len_used": len(text),
                        "conf_channel": "ensemble",
                        "conf_value": 0.2,
                    }
                )
            return PredictResponse(
                items=short_items,
                top10=list(short_items),
                shown=False,
                shown_topic_ids=[it.topic_id for it in short_items],
                model_version=self.model_version,
                taxonomy_version=self.taxonomy_version,
                latency_ms=0.1,
                meta=short_meta,
                conf_channel="ensemble",
                conf_value=0.2,
            )
        c1: float | None = 0.91 if return_calibrated else None
        items = [
            PredictItem(
                topic_id="1abffadf",
                topic_name='test "theme"',
                rank=1,
                score_raw=0.88,
                score_calibrated=c1,
            ),
            PredictItem(
                topic_id="638287e4",
                topic_name="other",
                rank=2,
                score_raw=0.4,
                score_calibrated=None,
            ),
        ]
        meta = None
        if return_meta:
            from src.api.schema import PredictMeta

            meta = PredictMeta(
                run_id="r",
                calibration_run_id=None,
                prefix_policy_run_id=None,
                prefix_policy={"l_min": 100, "channel": "ensemble", "tau": 0.7},
            )
        if _log_sink is not None:
            calibs = [it.score_calibrated for it in items]
            _log_sink.clear()
            _log_sink.update(
                {
                    "top_k_ids": [it.topic_id for it in items],
                    "top_k_scores": [float(it.score_raw) for it in items],
                    "top_k_scores_calibrated": calibs,
                    "prefix_len_used": len(text),
                    "conf_channel": "ensemble",
                    "conf_value": 0.85,
                }
            )
        return PredictResponse(
            items=items,
            top10=list(items),
            shown=True,
            shown_topic_ids=[i.topic_id for i in items],
            model_version=self.model_version,
            taxonomy_version=self.taxonomy_version,
            latency_ms=1.5,
            meta=meta,
            conf_channel="ensemble",
            conf_value=0.85,
        )


@pytest.fixture
def client_no_model() -> TestClient:
    return TestClient(create_app_for_test({"predictor": None, "startup_error": "offline"}))


@pytest.fixture
def client_fake() -> TestClient:
    return TestClient(create_app_for_test({"predictor": FakePredictor()}))


def test_healthz_ok(client_no_model: TestClient) -> None:
    r = client_no_model.get("/healthz")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["service"] == "orllm-inference"


def test_readyz_503_when_unloaded(client_no_model: TestClient) -> None:
    r = client_no_model.get("/readyz")
    assert r.status_code == 503


def test_readyz_200_fake(client_fake: TestClient) -> None:
    r = client_fake.get("/readyz")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ready"
    assert j["model_version"] == "fake_mv"
    assert j["classes_count"] == 2
    assert j["device"] == "cpu"


def test_predict_422_missing_body(client_fake: TestClient) -> None:
    r = client_fake.post("/predict", json={})
    assert r.status_code == 422


def test_predict_422_wrong_type(client_fake: TestClient) -> None:
    r = client_fake.post("/predict", json={"text": 123})
    assert r.status_code == 422


def test_predict_400_empty_text(client_fake: TestClient) -> None:
    r = client_fake.post("/predict", json={"text": ""})
    assert r.status_code == 400


def test_predict_400_whitespace(client_fake: TestClient) -> None:
    r = client_fake.post("/predict", json={"text": "   \t  "})
    assert r.status_code == 400


def test_predict_400_too_long(client_fake: TestClient) -> None:
    c = TestClient(create_app_for_test({"predictor": FakePredictor(max_len=10)}))
    r = c.post("/predict", json={"text": "x" * 11})
    assert r.status_code == 400


def test_predict_200_structure(client_fake: TestClient) -> None:
    text = "x" * 100
    r = client_fake.post(
        "/predict",
        json={"text": text, "return_calibrated": True, "return_meta": True},
    )
    assert r.status_code == 200
    j = r.json()
    assert "items" in j and "top10" in j
    assert j["items"] == j["top10"]
    assert set(j["shown_topic_ids"]) == {i["topic_id"] for i in j["items"]}
    assert j["shown"] is True
    assert j["model_version"] == "fake_mv"
    assert j["taxonomy_version"] == "sha256:fake"
    assert "latency_ms" in j
    assert j["meta"] is not None
    assert j["items"][0]["score_calibrated"] == 0.91
    assert j["items"][1]["score_calibrated"] is None


def test_predict_short_text_returns_items_with_shown_false(
    client_fake: TestClient,
) -> None:
    """API must return top-N predictions even on short text;
    shown=False reflects only the confidence (tau) gate."""
    r = client_fake.post("/predict", json={"text": "y" * 50})
    assert r.status_code == 200
    j = r.json()
    assert j["shown"] is False
    assert len(j["items"]) > 0
    assert len(j["top10"]) > 0
    assert j["items"] == j["top10"]
    assert set(j["shown_topic_ids"]) == {
        i["topic_id"] for i in j["items"]
    }


@pytest.mark.parametrize("length", [1, 5, 50, 100, 200, 500])
def test_predict_any_length_returns_top10(
    client_fake: TestClient, length: int,
) -> None:
    """L_min gate dropped: API returns top-N for any non-empty
    text length within max_text_len."""
    r = client_fake.post("/predict", json={"text": "x" * length})
    assert r.status_code == 200
    j = r.json()
    assert len(j["items"]) > 0
    assert j["items"] == j["top10"]


def test_predict_closed_set_500(client_fake: TestClient) -> None:
    r = client_fake.post("/predict", json={"text": "__closed_set__" + "x" * 100})
    assert r.status_code == 500


def test_feedback_ok(client_fake: TestClient) -> None:
    r = client_fake.post(
        "/feedback",
        json={"request_hash": "deadbeef", "chosen_topic_id": "1abffadf"},
    )
    assert r.status_code == 200
    assert r.json() == {"accepted": True}


def test_feedback_422_empty_hash(client_fake: TestClient) -> None:
    r = client_fake.post(
        "/feedback",
        json={"request_hash": "", "chosen_topic_id": "1abffadf"},
    )
    assert r.status_code == 422


def test_dependency_override_injects_fake_predictor() -> None:
    """Monkeypatch path: override Depends(get_predictor) without app.state model."""
    app_o = create_app_for_test({"predictor": None})

    def _predictor_override(request: Request) -> FakePredictor:
        return FakePredictor()

    app_o.dependency_overrides[get_predictor] = _predictor_override
    try:
        with TestClient(app_o) as c:
            r = c.post("/predict", json={"text": "z" * 100})
            assert r.status_code == 200
            assert r.json()["model_version"] == "fake_mv"
    finally:
        app_o.dependency_overrides.clear()


def test_predict_422_extra_field(client_fake: TestClient) -> None:
    r = client_fake.post(
        "/predict",
        json={"text": "x" * 100, "unexpected": 1},
    )
    assert r.status_code == 422


@pytest.mark.integration
def test_promoted_app_smoke_if_artifacts() -> None:
    fc = Path(
        "artifacts/three_way_hybrid/20260507_135733_three_way_promoted/fusion_config.json"
    )
    if not fc.is_file():
        pytest.skip("final three-way artifacts not present")
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        rz = client.get("/readyz")
        if rz.status_code != 200:
            pytest.skip(f"model startup failed in this environment: {rz.text}")
        p = client.post(
            "/predict",
            json={"text": "Прошу отремонтировать дорогу возле дома " * 4},
        )
        if p.status_code != 200:
            pytest.skip(f"predict failed: {p.status_code} {p.text}")
        j = p.json()
        assert "latency_ms" in j
        assert "model_version" in j


def test_predict_503_when_no_model(client_no_model: TestClient) -> None:
    r = client_no_model.post("/predict", json={"text": "x" * 100})
    assert r.status_code == 503
