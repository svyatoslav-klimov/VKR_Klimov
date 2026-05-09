# Inference logging (TASK-032)

Structured telemetry for the FastAPI service (`src/api/server.py`): each successful
`/predict` (and optional error paths with metadata), plus `/feedback`, append one row
to a **daily Parquet** file for downstream analytics and TASK-033 active-learning
joins.

## Paths and format

- Default directory: `reports/inference_logs/`
- Daily file: `{INFERENCE_LOG_DIR}/{YYYY-MM-DD}.parquet`
- Append semantics: existing rows for that day are read, concatenated with the new
  row, column order normalized, and the file rewritten (acceptable volume for Iter-3).

Canonical column names and types are listed in
[`inference_logging_schema.json`](inference_logging_schema.json). They match the
tuple `LOG_COLUMNS` in `src/api/logging.py`.

## Environment variables

| Variable | Values | Default | Meaning |
|----------|--------|---------|---------|
| `INFERENCE_LOG_ENABLED` | `1` / `0` (also `true`, `yes`, `on`) | `1` | Master switch; when disabled, no rows are written. |
| `INFERENCE_LOG_DIR` | filesystem path | `reports/inference_logs` | Output directory; relative paths resolve from repo root. |
| `INFERENCE_LOG_RAW_TEXT` | `1` / `0` | `0` | When `1`, store full trimmed text in `raw_text`; otherwise `raw_text` is null. |

## Privacy and PII policy

- **Default:** `raw_text` is **not** stored (`INFERENCE_LOG_RAW_TEXT` unset or `0`).
  Operational signals (`text_len`, scores, topic ids, latency, model versions) are
  retained.
- **`request_hash`:** correlates feedback with predictions without storing raw text:
  `sha256:` + hex digest of UTF-8 `text + "\\n" + timestamp_floor_minute_utc`, where
  the floor is the UTC instant truncated to minute (`...THH:MM:00Z`). Same text and
  same minute yields the same hash.
- **Explicit opt-in:** enable `INFERENCE_LOG_RAW_TEXT=1` only where policy allows
  storing full utterances (e.g. controlled staging).

## Request correlation

- **`/predict`:** server computes `request_hash` from body text and clock (minute
  granularity). Clients may log this hash client-side if exposed in future API
  versions; TASK-032 keeps it internal to the written row.
- **`/feedback`:** uses `request_hash` from the JSON body (TASK-031 contract). TASK-033
  will join feedback rows to predict rows on this key and fill `was_in_top_k`.

## Retention

Retention is **not** enforced in code (TASK-032). Operators should mirror standard
`reports/` hygiene: rotate or archive daily Parquet files per organizational policy.
Suggested default: treat logs like application telemetry with TTL aligned to the
smallest of legal retention and disk budgets.

## Dependencies

Writing Parquet requires **pandas with a Parquet engine** (this repo resolves it via
**pyarrow**). If neither engine is available, startup/logging raises a clear error
pointing to TASK-042 dependency manifest hygiene.

## TASK-033 usage

Consumers should:

1. Load daily Parquet under `INFERENCE_LOG_DIR`.
2. Filter `event_type == "predict"` for ranking telemetry; `event_type == "feedback"`
   for explicit user choices.
3. Join `feedback.request_hash` to `predict.request_hash` to derive labels and
   populate `was_in_top_k` (outside TASK-032 scope).

## Related docs

- HTTP API: [`api.md`](api.md)
- Machine-readable column manifest: [`inference_logging_schema.json`](inference_logging_schema.json)
