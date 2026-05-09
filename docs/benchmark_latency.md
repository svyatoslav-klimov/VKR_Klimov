# Latency & throughput benchmark (`tools/benchmark_latency.py`, TASK-034)

Production diagnostic grid: p50/p95/p99 per batch size and device, throughput, cold vs warm load, hardware fingerprint, and Pareto summary (R@10 vs latency).

## CLI (summary)

| Option | Default | Description |
|--------|---------|-------------|
| `--families` | all four | `sparse`, `retrieval`, `hybrid`, `three_way_hybrid` |
| `--devices` | `cpu cuda` | CUDA cells run first internally |
| `--batch-sizes` | `1 8 32 64 128` | Per-cell batch size |
| `--n-bench` | `1000` | Target iterations per cell (`n_bench_target`) |
| `--n-warmup` | `100` | Warmup before bench (20 if batch ≥ 64) |
| `--max-cell-min` | `30` | Per-cell budget hint for adaptive `n_bench` (see below) |
| `--total-budget-h` | `6` | Total wall-time budget for the whole grid |
| `--seed` | `42` | RNG for sampling texts from pool |
| `--out` | `reports/benchmarks/` | Output directory for CSV + MD |
| `--figures-out` | `reports/figures/` | Directory for `latency_pareto.png` |
| `--texts-pool` | `data/splits/train.parquet` | Text pool for synthetic batches |
| `--texts-pool-rows` | `5000` | Sample size |
| `--skip-existing` | off | Skip cells already present in `latency_full.csv` |

Full help: `python tools/benchmark_latency.py -h`.

## Output files

- `latency_full.csv` — long-form, one row per `(family, run_id, device, batch_size)` cell.
- `cold_vs_warm.csv` — one row per `(family, run_id, device)` with cold load, warm reload, first inference, post-warmup p50.
- `latency_full.md` — narrative: hardware string, compliance summary, Pareto table (batch=32, CUDA), link to figure.
- `latency_pareto.png` — scatter R@10 (test) vs p50 ms (batch=32, CUDA).

## CSV schemas (verbatim columns)

**Latency** (`latency_full.csv`):  
`family`, `run_id`, `device`, `batch_size`, `n_warmup`, `n_bench_actual`, `n_bench_target`, `p50_ms`, `p95_ms`, `p99_ms`, `throughput_texts_per_sec`, `model_load_seconds_cold`, `hardware_id`, `recall_at_10_test`, `timestamp_utc`, `notes`.

**Cold vs warm** (`cold_vs_warm.csv`):  
`family`, `run_id`, `device`, `model_load_seconds_cold`, `model_load_seconds_warm`, `first_inference_ms`, `post_warmup_p50_ms`, `hardware_id`, `timestamp_utc`.

`recall_at_10_test` is read from `reports/runs/<run_id>/metrics.json` → `recall_at_k["10"]`.

## Hardware fingerprint

`hardware_id` is the first 16 hex chars of SHA256 of a pipe-joined string:

`cpu_model | ram_gb=N | gpu_name | vram_gb=N | cuda=... | torch=... | py=...`

If `psutil` is missing, RAM is approximated from `os.cpu_count()` and the string carries an estimated marker (see TASK-042 dependency hygiene).

The markdown report includes the **expanded** fingerprint line (not only the hash).

## Time budget and `n_bench` (DECISION-054)

- **CUDA:** `n_bench_actual` aims for the full `--n-bench` target (typically 1000).
- **CPU, batch ≤ 32:** same target; if the probe suggests the cell may exceed `--max-cell-min`, a `may_exceed_max_cell_min` note may appear.
- **CPU, batch 64 or 128:** if 1000 iterations would exceed the per-cell budget estimate, reduce to at least 200 with `reduced_n_bench_due_to_cpu_budget` in `notes`.

Total grid runtime is capped by `--total-budget-h` (default 6 hours).

## Hugging Face encoder load

The tool wraps `sentence_transformers.SentenceTransformer` so that a **cached
snapshot directory** under `%USERPROFILE%\\.cache\\huggingface\\hub\\` is used
when available. This avoids intermittent API errors (HTTP 500) and works
without network when the model is already cached. If no snapshot exists, the
original model id is passed through (Hub fetch as usual).

Do **not** combine this with `HF_HUB_OFFLINE=1` unless your `transformers`
version fully supports offline tokenizer init (some versions still call the Hub
for mistral-related checks).

All CUDA `(family, batch_size)` cells are scheduled before CPU. On `torch.cuda.OutOfMemoryError`, the tool retries with batch size halved down to 8; otherwise the cell is marked `oom_skipped` or `oom_fallback_to_bs<N>` in `notes`.

## Canonical artifact paths

See `CANONICAL_RUNS` in `tools/benchmark_latency.py` (TASK-034 plan §3).
