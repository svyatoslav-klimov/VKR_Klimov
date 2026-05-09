# ORLLM production-relevant package layout

> **Назначение документа.** Зафиксировать минимальный набор файлов и
> артефактов, который требуется для запуска итогового inference/service
> Iter-3 ("three-way hybrid": TF-IDF SVC × E5 dense × BM25 →
> weighted_score_minmax → calibration → prefix UX-gating → FastAPI).
> Документ не удаляет и не переписывает существующие файлы — это
> contract для дальнейшей упаковки/деплоя (отдельный TASK по
> промоушену артефактов из `experiments/` в `artifacts/`).
>
> **Принцип.** Сначала рабочее, потом чистое. Если артефакт нужен
> рантайму, он остаётся в текущем местоположении даже если оно
> «неидеальное» (например, под `experiments/`). Эти случаи явно
> помечены как **TECH-DEBT** и собраны в §6.
>
> **Связано:** SPEC §3 (структура), §5 (интерфейсы), §9 (best.yaml),
> §10 (CLI/API), §12 (документация); DECISION-2026-04-28-036
> (изоляция кода экспериментов); TASK-031..034 (production deployment).

## 1. Scope: два сценария

| Сценарий | Что нужно | Объём |
|---|---|---|
| **A. Pure inference / serve** | модель + UX-gating + telemetry; запускается `uvicorn src.api.server:app` | минимум |
| **B. Inference + re-train/re-eval** | A + `src/cli.py` train/eval pipeline + splits parquet | + ~50 MB данных |

Pure inference — основной путь для production-deployment. Re-train —
сценарий обновления модели на новых данных (ежеквартальный refresh,
SPEC §6.1).

---

## 2. Inclusion list (что ОБЯЗАТЕЛЬНО)

### 2.1. Точка входа и сервис

```
src/api/
├── __init__.py
├── server.py             # FastAPI app, /healthz /readyz /predict /feedback
├── predictor.py          # OrllmPredictor: ThreeWayHybrid + UX gate + closed-set
├── schema.py             # Pydantic v2 модели запросов/ответов
├── logging.py            # InferenceLogWriter: daily Parquet (TASK-032)
└── active_learning.py    # AL queue stub (TASK-033)
```

### 2.2. Production-код модели и eval

```
src/
├── __init__.py
├── cli.py                                  # SPEC §10 (train-*/eval/predict)
├── models/
│   ├── __init__.py
│   ├── three_way_hybrid.py                 # Final fusion model (Iter-3)
│   ├── hybrid.py                           # Iter-2 hybrid (импортируется three_way)
│   ├── sparse.py                           # TF-IDF + LinearSVC + Platt-сlf
│   └── retrieval.py                        # E5 + FAISS centroids loader
├── eval/
│   ├── __init__.py
│   ├── metrics.py                          # Recall@k / MRR / nDCG / head-mid-tail
│   ├── prefix_eval.py                      # compute_confidence_channels (UX gate)
│   ├── split_strict.py                     # known_only enforcement
│   └── error_analysis.py                   # confident-error seed (TASK-021/033)
├── data/
│   ├── __init__.py
│   ├── build_topics.py                     # normalize_topic + classes loader
│   ├── splits_version.py                   # SPEC §6.1 splits_version hash
│   └── prepare.py                          # prepare-data CLI (только сценарий B)
└── utils/
    ├── __init__.py
    ├── run_manifest.py                     # SPEC §4.6 build_manifest
    └── leaderboard.py                      # SPEC §8 append_row
```

> **Не включаются** из `src/`: `src/_legacy/**` (DECISION-2026-04-26-017,
> уже в `.gitignore`).

### 2.3. Конфиги

```
configs/
├── final.yaml                              # Iter-3 финальный (TASK-040)
└── taxonomy.yaml                           # ссылка из final.yaml.data.taxonomy_yaml
```

Для сценария B (re-train) дополнительно:

```
configs/
└── data.yaml                               # split-генерация (если нужен полный rebuild)
```

> **Исключаются**: `best.yaml`, `sparse.yaml`, `retrieval.yaml`,
> `hybrid.yaml`, `prefix.yaml`, `prefix_best.yaml`, `baseline.yaml`,
> `base.yaml`, `_grid/`, `encoder_ablation/`, `123taxonomy — копия.yaml`.

### 2.4. Артефакты модели (frozen run-id'ы)

Финальная модель three-way — это **fusion-конфиг** поверх трёх
обученных компонент. Все четыре директории нужны рантайму одновременно.

```
artifacts/
├── three_way_hybrid/20260507_135733_three_way_promoted/   # конфиг fusion
│   ├── classes.json
│   ├── fusion_config.json                  # λ_sparse=0.1, λ_dense=0.7, λ_bm25=0.2
│   └── meta.json
├── sparse_tfidf/20260427_143358_sparse_linear_svc/        # компонент 1
│   ├── classes.json
│   ├── model.pkl                           # LinearSVC + Platt-prob
│   └── vectorizer.pkl                      # ~9 MB
├── retrieval_e5/20260427_165442_retrieval_e5/             # компонент 2
│   ├── classes.json
│   ├── meta.json                           # encoder_revision, normalize, faiss_metric
│   ├── themes.faiss                        # ~880 KB centroids index
│   ├── embeddings_train.npy                # требуется retrieval-loader
│   ├── embeddings_val.npy
│   └── embeddings_test.npy
└── manifests/
    └── 20260507_135733_three_way_promoted.json   # SPEC §4.6 (читается predictor)
```

Дополнительные production-runtime артефакты (см. §6 TECH-DEBT):

```
experiments/bm25_v1/artifacts/sparse/20260429_023849_bm25_aggtext/        # компонент 3
├── classes.json
├── meta.json
└── model.pkl                               # BM25Okapi index pickled

experiments/calibration_v1/artifacts/calibration/20260505_234500_three_way/   # калибровка top-1
├── isotonic.pkl                            # текущий выбранный калибратор
├── platt.pkl                               # альтернативный (для A/B)
└── (опционально) calibration_report.md, val_*split.json — read-only

experiments/prefix_v1/20260507_131143_three_way_full/                    # UX-policy
└── picked_policy.json                      # L_min=100, channel=ensemble, tau≈0.7087
```

> Эти три каталога **не являются** "research artifacts" — это **runtime
> dependencies**, на которые жёстко ссылаются `final.yaml.three_way.bm25_artifacts_dir`,
> `final.yaml.calibration_run_id`, `final.yaml.prefix_policy_run_id` и
> `src/api/predictor.py:_resolve_calibrator_path`/`_prefix_policy_path`.

### 2.5. Данные

```
data/
├── processed/
│   └── topics.csv                          # ★ единственный источник topic_id (R-004, runtime)
└── splits/                                 # только для сценария B (re-train/eval)
    ├── train.parquet
    ├── val.parquet
    └── test.parquet
```

> **Исключаются**: `data/raw/`, `data/interim/clean.csv`,
> `data/processed/dataset.parquet` (intermediate; для прода достаточно
> `topics.csv` + готовых splits), `experiments/head_demo_e02/data/splits/*`.

### 2.6. Тесты

Production-package поставляется с минимальным smoke-набором, чтобы при
развёртывании можно было `pytest -k smoke` подтвердить рабочее состояние:

```
tests/
├── conftest.py
├── test_api.py                             # FastAPI integration smoke
├── test_inference_logging.py               # TASK-032 contract
├── test_active_learning.py                 # TASK-033 priority logic
├── test_three_way_hybrid.py                # final model smoke
├── test_metrics.py                         # Recall@k / MRR / nDCG
├── test_run_manifest.py                    # SPEC §4.6
├── test_split.py                           # SPEC §6.1
├── test_topics.py                          # taxonomy invariants
├── test_prefix_eval.py                     # UX channels
├── test_artifacts.py                       # SPEC §11 — meta consistency
└── test_reproducibility.py                 # SPEC §11 — best.yaml rerun parity
```

> Опциональные: `test_sparse.py`, `test_retrieval.py`, `test_hybrid.py`,
> `test_error_analysis.py`, `test_audit_iter2.py`, `test_prepare.py`,
> `test_splits_version.py`, `test_tail_diagnostics.py`,
> `test_benchmark_latency.py` — нужны для full QA, можно опустить
> для slim-deployment.

### 2.7. Инструменты (operational)

```
tools/
└── benchmark_latency.py                    # capacity planning (TASK-034, частичный)
```

> `tools/finalize_latency_partial.py` — salvage-script,
> **не нужен в production**.
>
> Остальные `tools/grid_*.py`, `tools/bench_*.py`, `tools/promote_*.py` —
> исследовательские, исключаются.

### 2.8. Документация

```
README.md                                   # quickstart + ссылка на этот файл
requirements.txt                            # pinned deps
pyproject.toml / pytest.ini                 # если используются
.gitignore

docs/
├── api.md                                  # ★ HTTP API контракт (TASK-031)
├── inference_logging.md                    # ★ telemetry (TASK-032)
├── inference_logging_schema.json
├── active_learning.md                      # AL queue (TASK-033)
├── benchmark_latency.md                    # latency tool (TASK-034)
├── reproduce.md                            # SPEC §12 reproducibility
├── experiment_protocol.md                  # SPEC §6 в полной форме
├── metrics_spec.md                         # SPEC §4.4 формулы метрик
├── audit_report.md                         # ROLE_04 итоговый QA (Iter-1/2)
├── audit_report_iter2.md                   # ROLE_04 Iter-2 closure
├── implementation_plan.md
└── production_package.md                   # ← этот файл
```

### 2.9. Финальные отчёты прогона

```
reports/
├── runs/20260507_135733_three_way_promoted/metrics.json   # ★ финальные метрики
├── leaderboard.csv                                        # SPEC §8
├── decision_note.md                                       # SPEC §9 обоснование
└── benchmarks/                                            # TASK-034 partial
    ├── latency_full.csv                                   # 14 strict CUDA cells
    ├── latency_full.md
    └── cold_vs_warm.csv                                   # 4 family × 2 device
└── figures/latency_pareto.png                             # bs=32 CUDA Pareto
```

> Runtime-генерируемые `reports/inference_logs/`, `reports/active_learning/`
> в gitignore — заводятся пустыми и наполняются при работе сервиса.
>
> **Исключаются**: `reports/runs/<все остальные run-id>/`, `reports/audit/`,
> `reports/eda/`, `reports/error_analysis/`, `reports/calibration/`,
> `reports/prefix/`, `reports/grid/` (research-history).

---

## 3. Exclusion list (полная сверка)

Подтверждённый список того, что **НЕ входит** в production-relevant
package, с пояснениями.

| Путь | Причина исключения |
|---|---|
| `experiments/` (≠ runtime-артефакты §2.4) | research-ветки; `bm25_v1/`, `calibration_v1/`, `prefix_v1/` остаются как §6 TECH-DEBT |
| `.cursor/`, `.cursor/skills/`, `.cursor/agents/`, `.cursor/rules/` | dev infrastructure для агентной разработки |
| `agent-transcripts/`, `terminals/` | runtime контекст агентов |
| `.ai/plans/` | immutable architect plans (SPEC R-013) |
| `TASK_QUEUE.md` | очередь задач + DECISION log |
| `src/_legacy/**` | frozen Iter-0 snapshot (DECISION-2026-04-26-017, gitignore) |
| `exp/` | frozen legacy R/O (R-008) |
| `configs/best.yaml` | Iter-2 winner; для прода используется `final.yaml` (Iter-3) |
| `configs/sparse.yaml`, `retrieval.yaml`, `hybrid.yaml` | компонентные тренировочные конфиги; используются только при re-train компонент |
| `configs/prefix.yaml`, `prefix_best.yaml` | offline prefix-tuning; рантайм читает picked_policy.json напрямую |
| `configs/baseline.yaml`, `base.yaml` | research-baselines |
| `configs/_grid/`, `configs/encoder_ablation/` | grid configs Iter-1/2 |
| `configs/123taxonomy — копия.yaml` | артефакт ручной правки |
| `data/raw/`, `data/interim/clean.csv` | сырые данные не нужны рантайму; есть готовые `data/splits/*.parquet` |
| `data/processed/dataset.parquet` | intermediate full-dataset; перекрывается splits |
| `reports/runs/<кроме промоушена>/` | history исследования |
| `reports/audit/`, `reports/eda/`, `reports/error_analysis/`, `reports/calibration/`, `reports/prefix/`, `reports/grid/` | research/QA bundles |
| `tools/grid_*.py`, `tools/bench_*.py`, `tools/promote_*.py` | research instruments |
| `tools/finalize_latency_partial.py` | one-off salvage script |

---

## 4. Runtime entrypoints (acceptance smoke)

```powershell
# Активировать venv
Set-Location E:\Python\orLLM
$env:PYTHONIOENCODING = "utf-8"

# 1. Smoke health
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -k smoke -v

# 2. Запуск сервиса
.\.venv\Scripts\python.exe -m uvicorn src.api.server:app --host 0.0.0.0 --port 8080

# 3. Manual probe
Invoke-WebRequest -Method Post -Uri http://127.0.0.1:8080/predict `
  -ContentType "application/json; charset=utf-8" `
  -Body '{"text":"Прошу отремонтировать дорогу возле дома 5","return_meta":true}'

# 4. Прогон финального metrics (re-eval)
.\.venv\Scripts\python.exe -m src.cli eval `
  --config configs/final.yaml --split test
# → должно дать Recall@10 ≈ 0.8555 (parity ±1e-3 CPU / ±1e-2 GPU per SPEC §11)
```

Acceptance:

- `GET /healthz` → 200, `{"status":"ok"}`.
- `GET /readyz` → 200, `model_version=20260507_135733_three_way_promoted`.
- `POST /predict` → 200, `items[]` ≤ 10, `model_version`/`taxonomy_version`/`latency_ms` присутствуют.
- `pytest tests/test_three_way_hybrid.py tests/test_api.py` — green.

---

## 5. Внешние зависимости (runtime)

Из `requirements.txt` для production-инференса нужны:

| Пакет | Назначение |
|---|---|
| `fastapi`, `uvicorn[standard]` | HTTP сервис |
| `pydantic>=2` | request/response schema |
| `pandas`, `numpy`, `scipy` | базовые вычисления |
| `scikit-learn` | LinearSVC, LogisticRegression (Platt), IsotonicRegression |
| `sentence-transformers`, `torch`, `transformers`, `tokenizers` | E5 encoder |
| `faiss-cpu` (или `faiss-gpu`) | centroid retrieval |
| `pyarrow` | Parquet (telemetry + splits) |
| `pyyaml` | configs |
| `rank_bm25` | BM25 компонент |

Опционально:
- `psutil`, `matplotlib` — только для `tools/benchmark_latency.py`.
- `pytest`, `pytest-asyncio`, `httpx` — testing.

Веса E5 кэшируются в `%USERPROFILE%\.cache\huggingface\hub\models--intfloat--multilingual-e5-base\`.
В offline-окружении кэш должен быть пред-загружен (см. `docs/reproduce.md`).

---

## 6. TECH-DEBT (известные нарушения чистоты)

Эти пункты **не блокируют запуск** прод-сервиса, но должны быть закрыты
отдельным TASK `production_promotion` по правилу
`85-experiment-code-isolation.mdc` для долгосрочной сопровождаемости.

| # | Описание | План закрытия |
|---|---|---|
| TD-001 | BM25 модель лежит в `experiments/bm25_v1/artifacts/sparse/20260429_023849_bm25_aggtext/`. По SPEC §3.1 каноничный путь — `artifacts/bm25/<run_id>/`. | TASK `production_promotion`: `git mv experiments/bm25_v1/artifacts/sparse/<run_id>/ artifacts/bm25/<run_id>/`, обновить `final.yaml.three_way.bm25_artifacts_dir` и `meta.json` промоушена. |
| TD-002 | Калибраторы (`isotonic.pkl`, `platt.pkl`) лежат в `experiments/calibration_v1/artifacts/calibration/<run_id>/`. Hard-coded путь в `src/api/predictor.py:_resolve_calibrator_path`. | Перенести в `artifacts/calibration/<run_id>/`; параметризовать путь через `final.yaml.calibration_artifacts_dir` (вместо `calibration_run_id`). |
| TD-003 | UX-policy (`picked_policy.json`) лежит в `experiments/prefix_v1/<run_id>/`. Hard-coded путь в `src/api/predictor.py:_prefix_policy_path`. | Перенести в `artifacts/prefix_policy/<run_id>/`; параметризовать через `final.yaml.prefix_policy_artifacts_dir`. |
| TD-004 | `tools/benchmark_latency.py` имеет баг финализации (`KeyError: 'device'` если CSV без header'а — race с не-удалённым stale-файлом). Latency grid не покрывает `three_way_hybrid` и CPU. | Багфикс + повторный grid (или DECISION архитектора зафиксировать текущие 14 cells как достаточные с явным caveat). |
| TD-005 | `final.yaml` ссылается на пути с `experiments/` — формально нарушает §3.1 Pure-production-isolation. | Закрывается совместно TD-001..TD-003 (новый `final_prod.yaml`/обновление). |

Каждый TD — отдельная карточка в `TASK_QUEUE.md`, не блокирует Iter-3 sign-off.

---

## 7. Размер пакета (оценка)

Без HF-кэша и без `data/raw/`:

| Группа | ~Размер |
|---|---:|
| `src/` (всё кроме `_legacy/`) | ~250 KB |
| `tests/` (выбранный subset) | ~80 KB |
| `configs/{final,taxonomy}.yaml` (+ data.yaml для B) | ~5 KB |
| `artifacts/sparse_tfidf/...` (`model.pkl` + `vectorizer.pkl`) | ~9 MB |
| `artifacts/retrieval_e5/...` (faiss + 3 .npy) | ~30 MB |
| `artifacts/three_way_hybrid/...` | ~10 KB |
| `experiments/bm25_v1/.../<bm25 model>` | ~1.5 MB |
| `experiments/calibration_v1/.../<isotonic+platt>.pkl` | ~2 KB |
| `experiments/prefix_v1/.../picked_policy.json` | ~0.5 KB |
| `data/processed/topics.csv` | ~30 KB |
| `data/splits/*.parquet` (только B) | ~50 MB |
| `docs/`, `README.md`, `requirements.txt` | ~200 KB |
| **A. Pure inference** | **~41 MB** + HF cache |
| **B. + Re-train** | **~91 MB** + HF cache |

HF-кэш `multilingual-e5-base` снаружи: ~1.1 GB.

---

## 8. Что НЕ переносится при упаковке (даже если нужно для разработки)

- Все `.cursor/**` (агенты, rules, skills, transcripts).
- `agent-transcripts/`, `terminals/`.
- `.ai/plans/` (immutable architect plans).
- `TASK_QUEUE.md` (development backlog).
- `src/_legacy/**`, `exp/**`.
- `experiments/<*>/src/**` (research code; production уже промоушен).
- Любые файлы с расширениями из `.gitignore` под секцией Binary artifacts,
  если они не входят в whitelist §2.4.
