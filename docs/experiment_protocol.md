# Experiment Protocol (Draft)

This draft follows `ai_docs/SPEC.md` sections 6 and 11.

## 1. Time-based split and anti-leakage

- Split order must be strict: `train < val < test` by `created_at`.
- Random shuffle is forbidden in production split generation.
- All corpus statistics (TF-IDF, centroids, normalization stats) are fit on train only.
- Hyperparameter tuning uses val only.
- Test is used only for final reporting.

## 2. Metrics policy

- Primary metric: `Recall@10` on test with `unseen_policy=known_only`.
- Control metrics: `Recall@1/3/5`, `Macro-F1`, `Weighted-F1`, `Accuracy@1`, `MRR@10`, `nDCG@10`.
- Both slice modes are reported:
  - practical: `head>=50`, `mid>=10`, `tail<10`
  - strict: `head>=500`, `mid>=50`, `tail<50`

## 3. Reliability constraints

- R-001: retrieval query normalization must match index build normalization.
- R-002: all run artifacts are isolated by `run_id`.
- R-003: hybrid must combine sparse and dense from different sources.
- R-004: class lists must match across sparse and retrieval artifacts.
- R-005: FAISS index must be loaded via `faiss.read_index(str(path))`.
- R-006: subprocess calls must use `sys.executable`.
- R-007: all data IO must use UTF-8.
- R-008: writing to `exp/**` is forbidden.
- R-009: random-stratified val split is legacy-only and must be replaced in TASK-002b.

## 4. Minimum tests (SPEC §11)

- `tests/test_metrics.py` for metric contract and corner cases.
- `tests/test_split.py` for split ordering and overlap checks.
- `tests/test_inference.py` for predict contract (later tasks).
- `tests/test_artifacts.py` for class/index consistency (later tasks).
- `tests/test_reproducibility.py` for rerun stability (later tasks).
