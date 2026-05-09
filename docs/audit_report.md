# QA Audit Report — Iteration 1

**Date:** 2026-04-27  
**Auditor:** ROLE_04  
**Scope:** TASK-001 ... TASK-007 (5 accepted runs in leaderboard)

## 1. Executive summary

- **Blockers: 0**
- **Major: 0**
- **Minor: 2** (AUD-20260427-01, AUD-20260427-02)
- **Verdict: ITERATION CLOSED**
- **Best config:** N=30, norm=minmax, lambda=0.14, R@10_test reproduced = 0.8161153519932146 (bit-identical to archived metrics.json; |Δ| = 0).

## 2. Scope

- 5 runs, architectural DECISIONs 021 through 029 (per project log).
- splits_version: sha256:819b765d… (time-based, DECISION-028).
- taxonomy_version: sha256:85f23a78… (302 topics in topics.csv).
- n_train_classes (FAISS index_ntotal): 294.
- test: 5895 total, 5892 known, 3 unseen (known_only).

## 3. Per-check results

| Check | Description | Result | Evidence |
| --- | --- | --- | --- |
| A | anti-leakage (text intersection + time monotonicity) | PASS | reports/leakage_checks.json |
| B | FAISS R-001 (E5 prefixes, normalize, IP) | PASS | retrieval meta.json (static review) |
| C | антимиксинг R-003 (Pearson, top1 change, diversity) | PASS | hybrid metrics.json hybrid_diagnostics |
| D | R-008 (no exp/ in active src/) | PASS | rg scan (0 matches outside _legacy) |
| E | R-004 (classes consistency sparse==retrieval, n=294) | PASS | reports/leakage_checks.json |
| F | reproducibility best.yaml | PASS | reports/eval_pipeline_check.json |
| G | eval-pipeline carry-over (5 configs x R@10) | PASS | reports/eval_pipeline_check.json |
| H | sanity 10 examples (true in top10 vs population ~82%) | PASS | reports/sanity_examples.json |
| I | encoding (UTF-8, LF) | PASS (manual) | QA artifacts written with UTF-8 |
| J | metrics monotonicity | PASS | reports/metric_sanity_checks.json |

## 4. Issues

### AUD-20260427-01 (Minor) — splits_legacy_random_val отсутствует в manifest

- **R-код:** R-009
- **Task ID:** TASK-003 (origin), TASK-002b (re-issue)
- **Severity:** Minor
- **Description:** Builder manifest в `src/utils/run_manifest.py` не записывает поле `splits_legacy_random_val` (или эквивалент в секции splits), тогда как leaderboard фиксирует `false`. Прозрачность воспроизводимости сплита слабее, чем могла бы быть.
- **Recommendation:** В `build_manifest` добавить явный флаг, согласованный с `splits_version` / политикой сплита (например `manifest["splits"]["legacy_random_val"] = False`).
- **Verification step:** После фикса открыть новый `artifacts/manifests/<run_id>.json` и сравнить с leaderboard.

### AUD-20260427-02 (Minor) — config_path в best run указывает на grid YAML

- **R-код:** —
- **Task ID:** TASK-007
- **Severity:** Minor
- **Description:** В `reports/runs/20260427_171719_hybrid_weighted_score_minmax/metrics.json` поле `config_path` указывает на `configs/_grid/hybrid_30_minmax.yaml` (run-time grid config). Канонический пост-hoc конфиг зафиксирован как `configs/best.yaml` (DECISION-029).
- **Recommendation:** Уже задокументировано в DECISION-029; при необходимости дублировать в операторском чек-листе, что отчёты grid-run хранят исходный путь конфига.

### New issues from F/G/H

- Нет. Воспроизведение R@10 и carry-over совпали с архивными `metrics.json` с нулевой численной дельтой; sanity sample 9/10 в top-10 согласуется с ожиданием для n=10.

## 5. Reproducibility report

- **best.yaml** R@10_test: ожидаемое 0.8161153519932146, фактическое 0.8161153519932146, |Δ| = 0.0 (порог 1e-6: PASS).
- **eval_run_id** свежего прогона: `20260427_180015_eval_hybrid_test` (см. reports/runs/.../eval_metrics.json).
- Остальные четыре конфига: все `ok: true` в `reports/eval_pipeline_check.json`.

## 6. Sanity examples

См. `reports/sanity_examples.json`. **fraction_true_in_top10 = 0.9** (9 из 10). Для популяции ~81.6% Wilson 95% CI на n=10 широкий; любой исход в диапазоне, совместимом с биномиальной вариативностью, приемлем. Здесь 9/10 попадает в разумные ожидания.

## 7. Recommendations for iteration 2

1. **TASK-008 (DECISION-016):** prefix simulation + threshold tau tuning.
2. **Tail bucket improvement:** synthetic augmentation, label-aware representations, few-shot exemplars per topic.
3. **Encoder fine-tuning:** in-domain sentence encoder on citizen appeals.
4. **AUD-01 fix:** `splits_legacy_random_val` (или аналог) в manifest builder.
5. **TASK-006b carry-over:** batched-eval API в predict_topk для retrieval (per-query encoding медленный на full test).
