"""Load final model, apply prefix UX policy, build /predict payloads (TASK-031)."""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.api.schema import PredictItem, PredictMeta, PredictResponse
from src.eval.prefix_eval import compute_confidence_channels
from src.models.retrieval import encode_passages
from src.models.three_way_hybrid import ThreeWayHybrid, fused_three_way_top10_scores_and_topics

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/final.yaml")
DEFAULT_MANIFEST = Path(
    "artifacts/manifests/20260507_135733_three_way_promoted.json"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        o = yaml.safe_load(f)
    return o if isinstance(o, dict) else {}


def _apply_calibrator_pickle(scores: np.ndarray, calibrator_path: Path) -> np.ndarray:
    """Apply TASK-020-style calibrator without importing experiments.*."""
    with calibrator_path.open("rb") as f:
        payload: dict[str, Any] = pickle.load(f)
    kind = str(payload.get("type", ""))
    model = payload.get("model")
    x = np.asarray(scores, dtype=np.float64)
    flat = x.ravel()
    if kind == "platt":
        if not isinstance(model, LogisticRegression):
            raise TypeError("platt payload must wrap LogisticRegression")
        proba = model.predict_proba(flat.reshape(-1, 1))[:, 1]
        out = proba.reshape(x.shape)
        return np.asarray(out, dtype=np.float64)
    if kind == "isotonic":
        if not isinstance(model, IsotonicRegression):
            raise TypeError("isotonic payload must wrap IsotonicRegression")
        tr = model.transform(flat)
        out = np.asarray(tr, dtype=np.float64).reshape(x.shape)
        return out
    raise ValueError(f"Unknown calibrator type: {kind!r}")


def _resolve_calibrator_path(repo: Path, calibration_run_id: str) -> Path | None:
    cal_dir = repo / "experiments/calibration_v1/artifacts/calibration" / calibration_run_id
    metrics_path = (
        repo / "experiments/calibration_v1/reports/runs" / calibration_run_id / "metrics_calibration.json"
    )
    chosen = "isotonic"
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as f:
            mj = json.load(f)
        chosen = str(mj.get("chosen_calibrator", "isotonic"))
    name = "platt.pkl" if chosen == "platt" else "isotonic.pkl"
    pkl = cal_dir / name
    if pkl.is_file():
        return pkl
    logger.warning("Calibrator artifact missing at %s; prefix channel falls back to raw max_score.", pkl)
    return None


def _prefix_policy_path(repo: Path, prefix_policy_run_id: str) -> Path:
    return repo / "experiments/prefix_v1" / prefix_policy_run_id / "picked_policy.json"


class OrllmPredictor:
    """Inference wrapper: ThreeWayHybrid + topics + prefix UX + closed-set checks."""

    def __init__(
        self,
        *,
        config_path: Path,
        manifest_path: Path,
        repo_root: Path | None = None,
    ) -> None:
        self._root = repo_root or _repo_root()
        self._config_path = config_path.resolve()
        self._config = _load_yaml(self._config_path)
        inf = self._config.get("inference") or {}
        if not isinstance(inf, dict) or not inf.get("artifacts_dir"):
            raise ValueError("config.inference.artifacts_dir required")
        self._artifacts_dir = (self._root / str(inf["artifacts_dir"]).strip()).resolve()
        self.model_version = self._artifacts_dir.name

        self._manifest_path = manifest_path.resolve() if manifest_path.is_absolute() else (self._root / manifest_path).resolve()
        with self._manifest_path.open("r", encoding="utf-8") as f:
            self._manifest: dict[str, Any] = json.load(f)
        data = self._manifest.get("data") or {}
        self.taxonomy_version = str(data.get("taxonomy_version", ""))

        topics_p = self._config.get("data", {}).get("topics_path", "data/processed/topics.csv")
        topics_path = (self._root / str(topics_p)).resolve()
        df = pd.read_csv(topics_path, encoding="utf-8", dtype={"topic_id": str})
        self._topic_name: dict[str, str] = {
            str(row.topic_id): str(row.topic_name)
            for row in df.itertuples(index=False)
        }
        self._topics_ids = set(self._topic_name.keys())

        cal_id = str(self._config.get("calibration_run_id", ""))
        self._calibration_run_id = cal_id or None
        self._calibrator_path: Path | None = None
        if cal_id:
            self._calibrator_path = _resolve_calibrator_path(self._root, cal_id)

        prefix_run = str(self._config.get("prefix_policy_run_id", ""))
        self._prefix_policy_run_id = prefix_run or None
        pol_path = _prefix_policy_path(self._root, prefix_run) if prefix_run else None
        if pol_path is None or not pol_path.is_file():
            raise FileNotFoundError(f"prefix policy missing: {pol_path}")
        with pol_path.open("r", encoding="utf-8") as f:
            self._prefix_policy: dict[str, Any] = json.load(f)

        self._l_min = int(self._prefix_policy["L_min"])
        self._prefix_channel = str(self._prefix_policy["channel"])
        self._tau = float(self._prefix_policy["tau_value"])

        pp = self._config.get("preprocessing") or {}
        self._max_text_len = int(pp.get("max_text_len", 2000))

        self._model: ThreeWayHybrid | None = None

    @property
    def classes_count(self) -> int:
        if self._model is None:
            p = self._artifacts_dir / "classes.json"
            with p.open("r", encoding="utf-8") as f:
                classes = json.load(f)
            return len(classes) if isinstance(classes, list) else 0
        return len(self._model.classes)

    @property
    def max_text_len(self) -> int:
        return self._max_text_len

    @property
    def device(self) -> str:
        if self._model is None:
            return "unknown"
        return str(self._model.device)

    def load_model(self) -> None:
        """Load heavy weights once (GPU-first via ThreeWayHybrid.load)."""
        self._model = ThreeWayHybrid.load(self._artifacts_dir, device="auto")

    def ensure_model(self) -> ThreeWayHybrid:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model

    def _closed_set_validate(self, topic_ids: list[str], classes_set: set[str]) -> None:
        for tid in topic_ids:
            if tid not in classes_set or tid not in self._topics_ids:
                raise RuntimeError(
                    f"closed-set violation: topic_id {tid!r} not in model classes and topics metadata"
                )

    def _ux_confidence_scalar(self, top10_scores: np.ndarray, *, return_calibrated: bool) -> float:
        """Channel value for UX gate (same construction as retune_prefix_tau_three_way)."""
        t = np.asarray(top10_scores, dtype=np.float64).reshape(1, -1)
        chans = compute_confidence_channels(t, calibrator_path=None)
        if return_calibrated and self._calibrator_path is not None:
            m = np.asarray(chans["max_score"], dtype=np.float64)
            chans["max_score_calibrated"] = _apply_calibrator_pickle(m, self._calibrator_path)
        if "ensemble" not in chans:
            m = np.asarray(chans["max_score"], dtype=np.float64)
            mc = np.asarray(chans.get("max_score_calibrated", m), dtype=np.float64)
            chans["ensemble"] = 0.5 * (m + mc)
        ch_name = self._prefix_channel
        if ch_name not in chans:
            raise ValueError(f"Unknown prefix UX channel: {ch_name!r}")
        return float(np.asarray(chans[ch_name], dtype=np.float64).reshape(-1)[0])

    def predict(
        self,
        text: str,
        *,
        return_calibrated: bool = True,
        return_meta: bool = False,
        _log_sink: dict[str, Any] | None = None,
    ) -> PredictResponse:
        """Run fusion inference and apply prefix show policy."""
        text = str(text).strip()
        t0 = time.perf_counter()
        model = self.ensure_model()
        classes_set = set(model.classes)

        texts_l = [text]
        emb = encode_passages(
            model.encoder,
            texts_l,
            model.e5_prefix_query,
            model.normalize_embeddings,
            model.batch_size,
        )
        s_full, r_full, b_full = model._score_mats(texts_l, emb)
        sc, top = fused_three_way_top10_scores_and_topics(
            s_full,
            r_full,
            b_full,
            lam_sparse=model.lam_sparse(),
            lam_dense=model.lam_dense(),
            lam_bm25=model.lam_bm25(),
            fusion_mode=model.fusion_mode(),
            norm=model.norm_mode(),
            candidates=model.candidates_n(),
            classes=model.classes,
            k_out=10,
            k_rrf=model.k_rrf(),
        )
        row_scores = np.asarray(sc[0], dtype=np.float64)
        row_topics = top[0] if top else []
        row_topics = [str(t) for t in row_topics if t]

        cal_top1: float | None = None
        if return_calibrated and self._calibrator_path is not None:
            raw_top1 = np.array([row_scores[0]], dtype=np.float64)
            cal_top1 = float(_apply_calibrator_pickle(raw_top1, self._calibrator_path)[0])

        top_ids_full = [str(t) for t in row_topics[:10]]
        top_scores_full = [float(row_scores[i]) for i in range(min(10, len(row_scores)))]
        while len(top_scores_full) < len(top_ids_full):
            top_scores_full.append(0.0)
        cal_full: list[float | None] = []
        for rank in range(1, len(top_ids_full) + 1):
            if rank == 1 and cal_top1 is not None:
                cal_full.append(cal_top1)
            elif rank == 1 and not return_calibrated:
                cal_full.append(None)
            else:
                cal_full.append(None)

        items: list[PredictItem] = []
        for rank, tid in enumerate(row_topics[:10], start=1):
            name = self._topic_name.get(tid, "")
            raw = float(row_scores[rank - 1]) if rank - 1 < len(row_scores) else 0.0
            s_cal: float | None = cal_top1 if rank == 1 else None
            if not return_calibrated:
                s_cal = None
            items.append(
                PredictItem(
                    topic_id=tid,
                    topic_name=name,
                    rank=rank,
                    score_raw=raw,
                    score_calibrated=s_cal,
                )
            )
        ids = [it.topic_id for it in items]
        self._closed_set_validate(ids, classes_set)

        ux_conf = self._ux_confidence_scalar(row_scores, return_calibrated=return_calibrated)
        # DECISION-2026-05-09-056: L_min length-gate moved to frontend.
        # API returns top-10 for any non-empty text; `shown` reflects only
        # the confidence (tau) gate. `meta.prefix_policy.l_min` stays
        # exposed as an informational hint for the frontend UX policy.
        shown = bool(ux_conf >= self._tau)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if _log_sink is not None:
            _log_sink.clear()
            _log_sink.update(
                {
                    "top_k_ids": list(top_ids_full),
                    "top_k_scores": list(top_scores_full),
                    "top_k_scores_calibrated": list(cal_full),
                    "prefix_len_used": int(len(text)),
                    "conf_channel": str(self._prefix_channel),
                    "conf_value": float(ux_conf),
                }
            )

        meta: PredictMeta | None = None
        if return_meta:
            meta = PredictMeta(
                run_id=self.model_version,
                calibration_run_id=self._calibration_run_id,
                prefix_policy_run_id=self._prefix_policy_run_id,
                prefix_policy={
                    "l_min": self._l_min,
                    "channel": self._prefix_channel,
                    "tau": self._tau,
                },
            )

        return PredictResponse(
            items=list(items),
            top10=list(items),
            shown=shown,
            shown_topic_ids=list(ids),
            model_version=self.model_version,
            taxonomy_version=self.taxonomy_version,
            latency_ms=float(latency_ms),
            meta=meta,
            conf_channel=str(self._prefix_channel),
            conf_value=float(ux_conf),
        )


def build_predictor_from_env(
    config_path: Path | None = None,
    manifest_path: Path | None = None,
) -> OrllmPredictor:
    """Default factory: final.yaml + promoted manifest."""
    root = _repo_root()
    cfg = config_path or (root / DEFAULT_CONFIG)
    man = manifest_path or (root / DEFAULT_MANIFEST)
    return OrllmPredictor(config_path=cfg, manifest_path=man, repo_root=root)
