# ORLLM — Topic suggestion service

Top-10 thematic-tag ranking over a closed-set Russian taxonomy
(294 production classes, head/mid/tail balanced) for citizen-appeal
texts. Production deployment of the **Iter-3 final three-way hybrid**
model (TF-IDF SVC × multilingual-E5 dense × BM25 → weighted_score
min-max fusion → calibrated top-1 → prefix-policy UX gate) wrapped in
a FastAPI service with daily Parquet telemetry and an active-learning
queue stub.

## Final model

| | |
|---|---|
| **Family** | three_way_hybrid (sparse + dense + BM25) |
| **Run id** | `20260507_135733_three_way_promoted` |
| **Recall@10 test (known_only)** | **0.8555** |
| **Recall@1 / @5** | 0.4419 / 0.7259 |
| **MRR@10 / nDCG@10** | 0.5656 / 0.6345 |
| **head / mid / tail (practical)** | 0.875 / 0.637 / 0.154 |
| **Macro-F1 / Weighted-F1** | 0.1590 / 0.4250 |
| **Encoder** | `intfloat/multilingual-e5-base` (rev d128750…d128750) |
| **Fusion λ** | sparse=0.10, dense=0.70, bm25=0.20 (min-max norm, candidates=30) |
| **Calibrator (top-1)** | isotonic (Platt available) |
| **UX policy** | `L_min=100`, `channel=ensemble`, `tau≈0.7087` |
| **Taxonomy version** | `sha256:85f23a78…3840fe90` |
| **Splits version** | `sha256:819b765d…adfd16dc` |

Full eval bundle: `reports/runs/20260507_135733_three_way_promoted/metrics.json`.

## Repository layout

```
src/api/                  FastAPI service (server, predictor, schema, logging, AL)
src/models/               three_way_hybrid + components (sparse, retrieval, hybrid)
src/eval/                 Recall/MRR/nDCG metrics, prefix-policy channels
src/cli/                  train / eval / predict CLI (re-train scenario)
src/data/, src/utils/     pipeline plumbing
configs/final.yaml        Iter-3 production config (frozen)
configs/taxonomy.yaml     Closed-set taxonomy schema
configs/data.yaml         Splits generation config (re-train scenario)
artifacts/                Component artifacts (sparse SVC, E5+FAISS, three-way fusion)
experiments/bm25_v1/...   BM25 component (TECH-DEBT TD-001, see docs/production_package.md §6)
experiments/calibration_v1/...   Calibrators (TECH-DEBT TD-002)
experiments/prefix_v1/... UX-gate policy (TECH-DEBT TD-003)
data/processed/topics.csv Source of truth for topic_id (R-004)
data/splits/*.parquet     Time-based train/val/test splits
tests/                    pytest suite (smoke + critical)
tools/benchmark_latency.py  Capacity-planning latency grid
docs/                     SPEC §12 documentation set
reports/                  Final metrics, leaderboard, latency benchmarks
```

Detailed architecture and the rationale for the `experiments/` runtime paths:
[`docs/production_package.md`](docs/production_package.md).

## Prerequisites

- Python **3.10.11** (other 3.10/3.11 should work; not tested elsewhere).
- Optional: NVIDIA CUDA 12.1 + an Ampere-class GPU (RTX 30xx) for ~10×
  faster encoding. CPU-only fully supported, only inference latency
  scales with `batch_size`.
- ~1.1 GB of free disk for the `intfloat/multilingual-e5-base` HF cache
  (auto-downloaded on first launch, or pre-warm offline — see below).
- Git LFS for cloning binary artifacts (`*.pkl`, `*.npy`, `*.faiss`,
  `*.parquet`) — install once: `git lfs install`.

## Install

```powershell
git lfs install
git clone <new-repo-url> ORLLM
cd ORLLM

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

CPU-only PyTorch: drop the `--extra-index-url` and `pip install`
will pull `torch` from PyPI (CPU build). Edit `requirements.txt`:
change `torch==2.5.1+cu121` → `torch==2.5.1`.

## Pre-warm the HF encoder cache (offline-friendly)

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"
```

The model lands in `%USERPROFILE%\.cache\huggingface\hub\models--intfloat--multilingual-e5-base\`.
Subsequent loads are offline.

## Smoke test (3 commands)

```powershell
$env:PYTHONIOENCODING = "utf-8"

# 1. Pytest smoke
python -m pytest tests/test_three_way_hybrid.py tests/test_api.py -v

# 2. Re-evaluate the frozen final model (parity check vs published Recall@10=0.8555)
python -m src.cli eval --config configs/final.yaml --split test

# 3. Run service locally
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

## Service

```http
GET  /healthz   → 200 {status, service, task_id}
GET  /readyz    → 200 {model_version, taxonomy_version, classes_count, device}
POST /predict   → 200 {items[], top10[], shown, shown_topic_ids[],
                       model_version, taxonomy_version, latency_ms,
                       conf_channel, conf_value}
POST /feedback  → 200 {accepted}
```

Full schema and UX-gating semantics: [`docs/api.md`](docs/api.md).

Sample probe:

```powershell
$body = '{"text":"Прошу отремонтировать дорогу возле дома 5","return_meta":true}'
Invoke-WebRequest -Method Post `
  -Uri http://127.0.0.1:8080/predict `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Sample response (truncated):

```json
{
  "items": [
    {"topic_id": "12.5", "topic_name": "Содержание автомобильных дорог общего пользования местного значения",
     "rank": 1, "score_raw": 0.91, "score_calibrated": 0.83},
    {"topic_id": "12.7", "topic_name": "Уборка территории общего пользования",
     "rank": 2, "score_raw": 0.74, "score_calibrated": null}
  ],
  "shown": true,
  "shown_topic_ids": ["12.5", "12.7", "..."],
  "model_version": "20260507_135733_three_way_promoted",
  "taxonomy_version": "sha256:85f23a78...3840fe90",
  "latency_ms": 312.4,
  "conf_channel": "ensemble",
  "conf_value": 0.81
}
```

## Telemetry (optional, off by default)

Set environment variables before launch:

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_LOG_ENABLED` | `0` | `1` enables daily Parquet writer |
| `INFERENCE_LOG_DIR` | `reports/inference_logs/` | Output directory |
| `INFERENCE_LOG_RAW_TEXT` | `0` | `1` stores raw request text (PII!) |
| `ORLLM_FINAL_CONFIG` | `configs/final.yaml` | Override config path |

Schema and retention: [`docs/inference_logging.md`](docs/inference_logging.md).
Active-learning queue (built from logs + confident-error seed):
[`docs/active_learning.md`](docs/active_learning.md).

## Re-train / re-evaluate

The repository ships the same `time-based` splits used during Iter-3
training (`data/splits/{train,val,test}.parquet`) and all component
manifests under `artifacts/manifests/`.

```powershell
# Rebuild splits from a fresh raw dataset (only when refreshing data)
python -m src.cli prepare-data    --config configs/data.yaml

# Train a single component (re-baseline)
python -m src.cli train-sparse    --config configs/sparse.yaml      # NOT shipped: ROLE_02 must add
python -m src.cli train-retrieval --config configs/retrieval.yaml   # NOT shipped: ROLE_02 must add
python -m src.cli train-hybrid    --config configs/hybrid.yaml      # NOT shipped: ROLE_02 must add

# Train + freeze the three-way fusion (Iter-3 final)
python -m src.cli train-three-way --config configs/final.yaml

# Re-evaluate
python -m src.cli eval            --config configs/final.yaml --split test
```

Component-level training configs (`sparse.yaml`, `retrieval.yaml`,
`hybrid.yaml`) are **NOT** shipped in the production package — they
are research-time configs and live in the upstream development repo.
For re-training individual components in this repo, copy the matching
config from the dev repo or construct one from `configs/final.yaml`.

Re-train end-to-end protocol: [`docs/reproduce.md`](docs/reproduce.md).
Component contracts and metric definitions:
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Capacity planning (latency)

```powershell
python tools/benchmark_latency.py --help
```

Latest grid (partial — see `reports/benchmarks/latency_full.md`):

| family | bs=1 (p50 ms) | bs=8 | bs=32 | bs=64 | bs=128 |
|---|---:|---:|---:|---:|---:|
| sparse | 1.15 | 10.21 | 44.57 | 93.68 | 180.13 |
| retrieval (E5) | 13.70 | 72.37 | 291.39 | 1415.10 | 2532.77\* |
| hybrid (sparse+dense) | 15.59 | 79.99 | 314.84 | 2296.69\* | — |
| three_way (final prod) | — | — | — | — | — |

\* p99 unreliable due to per-cell wall-budget truncation. CPU rows and
three_way coverage missing — see [`docs/benchmark_latency.md`](docs/benchmark_latency.md)
and `reports/benchmarks/`.

## Documentation map

| File | Contents |
|---|---|
| [`docs/production_package.md`](docs/production_package.md) | Production layout, TECH-DEBT, scenarios A/B |
| [`docs/api.md`](docs/api.md) | HTTP API contract |
| [`docs/inference_logging.md`](docs/inference_logging.md) | Telemetry schema, retention, PII |
| [`docs/inference_logging_schema.json`](docs/inference_logging_schema.json) | Machine-readable column schema |
| [`docs/active_learning.md`](docs/active_learning.md) | Active-learning queue producer |
| [`docs/benchmark_latency.md`](docs/benchmark_latency.md) | Latency tool CLI + outputs |
| [`docs/reproduce.md`](docs/reproduce.md) | Reproducibility protocol |
| [`docs/experiment_protocol.md`](docs/experiment_protocol.md) | Splits, metrics, anti-leakage, anti-mixing |
| [`docs/audit_report.md`](docs/audit_report.md) | Iter-1 QA closure |
| [`docs/audit_report_iter2.md`](docs/audit_report_iter2.md) | Iter-2 QA closure |
| [`docs/implementation_plan.md`](docs/implementation_plan.md) | Phase plan and history |

## Versioning and provenance

This repository is a **frozen production snapshot** of the Iter-3
deliverable. Each artifact is reproducible from the corresponding
`artifacts/manifests/<run_id>.json` (canonical SPEC §4.6 manifest:
input file SHA-256, taxonomy version, splits version, package
versions, FAISS metadata).

Updates ship as new commits with a fresh `run_id`; previous runs are
preserved in git history and can be restored verbatim. There is no
in-place mutation of accepted artifacts (see SPEC §3 R-002 risk
glossary).
