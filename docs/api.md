# ORLLM inference HTTP API (TASK-031)

Production FastAPI service over the Iter-3 final three-way hybrid model
(`configs/final.yaml`).

## Security

There is **no authentication or authorization** in TASK-031. The service is
intended for private network deployment or behind an API gateway.

**TASK-032:** Optional inference telemetry writes daily Parquet under
`reports/inference_logs/` (env-configurable). See
[`inference_logging.md`](inference_logging.md). Raw request text is **off** unless
`INFERENCE_LOG_RAW_TEXT=1`.

## Run locally

From the repository root, using the project virtualenv:

```powershell
Set-Location E:\Python\orLLM
.\.venv\Scripts\python.exe -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

Python example:

```python
import requests

r = requests.post(
    "http://127.0.0.1:8080/predict",
    json={"text": "Прошу отремонтировать дорогу возле дома", "return_meta": True},
    timeout=60,
)
print(r.status_code, r.json())
```

## Endpoints

### GET /healthz

Liveness probe. Does **not** load the model.

**200** body:

```json
{
  "status": "ok",
  "service": "orllm-inference",
  "task_id": "TASK-031"
}
```

### GET /readyz

Readiness probe after application startup (model load).

**200** when the model is loaded:

```json
{
  "status": "ready",
  "model_version": "20260507_135733_three_way_promoted",
  "taxonomy_version": "sha256:...",
  "classes_count": 303,
  "device": "cuda"
}
```

**503** if `ThreeWayHybrid.load` failed during startup (disk, deps, CUDA, etc.):
response body uses FastAPI `{"detail": "..."}`.

### POST /predict

**Request JSON**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | User utterance (after trim must be non-empty). |
| `return_calibrated` | bool | `true` | If true, apply top-1 calibrated confidence when calibrator `.pkl` is present. |
| `return_meta` | bool | `false` | If true, include `meta` block with run ids and prefix policy snapshot. |

**200** body (required fields):

- `items`, `top10`: same list of ranked suggestions (closed-set `topic_id` from
  `classes.json`). When UX policy hides suggestions, both are empty arrays.
- `shown`: whether suggestions are shown after server-side UX gating.
- `shown_topic_ids`: list of `topic_id` in rank order when `shown` is true;
  empty when hidden.
- `model_version`: promoted run id (artifact directory name).
- `taxonomy_version`: from training manifest (`artifacts/manifests/...json`).
- `latency_ms`: server-side wall time for the request (encode + fusion + checks).
- `conf_channel`, `conf_value`: prefix-policy scalar gate diagnostics (TASK-032;
  same construction as offline prefix evaluation channels).

Each element of `items` / `top10`:

- `topic_id`, `topic_name` (from `data/processed/topics.csv`), `rank` (1-based),
  `score_raw` (fused score proxy after per-query normalization inside the union
  of candidates, same family as offline prefix evaluation),
  `score_calibrated`: see Calibration below.

Optional **`meta`** (when `return_meta: true`):

- `run_id`, `calibration_run_id`, `prefix_policy_run_id`,
  `prefix_policy`: `{ "l_min", "channel", "tau" }` from
  `experiments/prefix_v1/<prefix_policy_run_id>/picked_policy.json`.

**Errors**

| Code | When |
|------|------|
| 400 | Empty or whitespace-only `text`; or `len(text)` over `preprocessing.max_text_len` from `configs/final.yaml` (default 2000). |
| 422 | Missing `text`, wrong types, or extra JSON keys (`extra inputs forbidden`). |
| 503 | Model not loaded (`GET /readyz` not ready). |
| 500 | Closed-set violation (unexpected `topic_id` vs `classes.json` / topics table) or internal failure. |

**UX policy (server-side)**

Loaded from the prefix policy run in `configs/final.yaml`
(`experiments/prefix_v1/<prefix_policy_run_id>/picked_policy.json`). For the
current final model this is `L_min=100`, `channel=ensemble`,
`tau≈0.7087` on the ensemble confidence.

If `len(text) < L_min` **or** the selected confidence channel is below `tau`,
the API returns `shown=false`, `items=[]`, `top10=[]`, `shown_topic_ids=[]`.
Lengths are Python string lengths (Unicode code points) on the trimmed text.

### POST /feedback

Active-learning compatible endpoint (TASK-031 surface; TASK-032 persistence).

**Request:** `{ "request_hash": "string", "chosen_topic_id": "string" }`  
**200:** `{ "accepted": true }`  

When `INFERENCE_LOG_ENABLED=1`, a feedback row is appended to the daily Parquet log
(`chosen_topic`, client `request_hash`, `was_in_top_k` reserved null until TASK-033).

**422** if fields are missing, wrong type, or empty strings.

## Calibration honesty

Offline calibration (TASK-040 / calibration_v1) fits a **top-1** score-to-probability
mapping on fused `max_score` from the top-10 vector. The API therefore:

- Applies the calibrator **only to rank 1** and exposes that value as
  `items[0].score_calibrated` when `return_calibrated` is true and
  `isotonic.pkl` / `platt.pkl` exists under
  `experiments/calibration_v1/artifacts/calibration/<calibration_run_id>/`.
- Sets `score_calibrated` to `null` for ranks 2–10. These ranks are **not**
  claimed to be calibrated marginal probabilities.
- If `return_calibrated` is false, all `score_calibrated` fields are `null`.
- If calibrator files are absent, raw `score_raw` is still returned; ensemble
  UX gating falls back to uncalibrated `max_score` in the ensemble mix
  (matches tooling behavior when `max_score_calibrated` is missing).

## Implementation notes

- Model loads once at FastAPI startup via `ThreeWayHybrid.load(..., device="auto")`.
- Inference uses the same fused top-10 scoring path as prefix evaluation
  (`fused_three_way_top10_scores_and_topics`), not the rank-placeholder scores
  from `predict_topk` list positions.
- Default manifest:
  `artifacts/manifests/20260507_135733_three_way_promoted.json`.
