# Active-learning queue (TASK-033)

Batch helper over TASK-032 inference Parquet logs. Produces a daily JSON list of rows worth manual review before any retraining (no training loop in Iter-3).

## Privacy

Queue JSON **must not** contain `raw_text`. Only `text_len` (and model/taxonomy metadata) appear, matching the default TASK-032 logging stance when `INFERENCE_LOG_RAW_TEXT=0`.

## Tail bucket

Tail topics are taken from `data/processed/topics.csv`: any `topic_id` with `count_total < 10` is treated as **tail** for the `tail_predicted` flag (practical strict threshold aligned with head/mid/tail discussion in SPEC).

## Thresholds

| Signal | Source | Default |
|--------|--------|---------|
| `tau_low_conf` | `configs/prefix_best.yaml` `picked.tau_value`, else `configs/final.yaml` `picked.tau_value`, else **0.5** | CLI `--tau-low-conf` overrides |
| `tau_low_margin` | fixed in TASK-033 | **0.05**; CLI `--tau-low-margin` |
| Late period | optional | CLI `--late-period-start YYYY-MM-DD` (UTC start-of-day). If omitted, **no** rows receive `late_period`. |

## Categories

Each predict row may match zero or more of:

| Code | Rule |
|------|------|
| `outside_top_k` | After join, feedback exists and `chosen_topic` is **not** in `top_k_ids`. |
| `low_margin` | `len(top_k_scores) >= 2` and `top_k_scores[0] - top_k_scores[1] < tau_low_margin`. |
| `low_conf` | First finite `top_k_scores_calibrated[0] < tau_low_conf`. |
| `tail_predicted` | `top_k_ids[0]` is in the tail set (`count_total < 10`). |
| `late_period` | `timestamp_utc >=` start of `--late-period-start` (UTC). |

Rows with **no** matching category are dropped. Seed rows (see below) always carry `outside_top_k` and `seed_iter2_confident_error`.

## Priority

Only four booleans contribute (not `low_margin`):

```text
priority = 0.4 * is_outside_top_k + 0.2 * is_low_conf + 0.2 * is_tail + 0.2 * is_late_period
```

Sort: **priority descending**, ties **timestamp_utc ascending** (then `request_hash` for stability).

Seed rows from TASK-021 use **priority = 0.4** (outside-only weighting per plan).

## TASK-021 seed merge

`--include-seed` appends rows from `reports/error_analysis/confident_errors.parquet` (override with `--seed-parquet`). Mapping:

- `request_hash`: `seed:` + hex sha256 over `f\"{test_idx}{true_topic_id}\"` (UTF-8).
- `chosen_topic`: `true_topic_id`.
- `was_in_top_k`: `false`.
- `top_k_ids` / `top_k_scores`: `[top1_pred]`, `[top1_score_raw]`.
- `top_k_scores_calibrated`: `[conf_calibrated, null]` when defined.
- `categories`: `["outside_top_k", "seed_iter2_confident_error"]`.
- `source`: `seed_iter2_confident_error`.

## Same-day join

Feedback is joined to predict on `request_hash` using rows from the **same** Parquet file (same UTC day). Cross-day feedback is out of scope (future task).

## CLI

From repo root (Windows venv):

```powershell
.\.venv\Scripts\python.exe -m src.api.active_learning `
  --logs reports/inference_logs `
  --out reports/active_learning `
  --date 2026-05-08 `
  [--include-seed] `
  [--seed-parquet path\to\confident_errors.parquet] `
  [--topics data\processed\topics.csv] `
  [--late-period-start YYYY-MM-DD] `
  [--tau-low-conf 0.5] `
  [--tau-low-margin 0.05]
```

Writes `out/YYYY-MM-DD.json`. Missing log file yields `items: []`.

## Machine-readable schema

See `docs/active_learning_queue_schema.json`.

## References

- TASK-032: `docs/inference_logging.md`, `docs/inference_logging_schema.json`
- Plan: `.ai/plans/2026-05-08_TASK-033_active-learning-queue-stub.md`
