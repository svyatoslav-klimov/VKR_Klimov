# TASK-037 decision note (three_way_fusion_v1)

## Gate verdict: **PASS**

**RATIONALE (plan §4.3):** Val-affinity winner `20260505_211952_three_way_w_0p10_0p70_0p20` has
`val_R@10=0.8574200702` (> `0.8226` = Iter-2 val ~0.81759 + 0.005) and
`test_R@10=0.8554707379` (>= `0.8111` = Iter-2 test 0.816115… - 0.005). Global degradation tolerance
is not violated.

**R-003 orthogonality:** `analyze_orthogonality.py` now matches **TASK-017 methodology** for gated
Pearsons (surrogate TF-IDF + E5 centroids, k=10 mean on val known-only). Fresh run:
`pearson_bm25_e5=0.26063`, `pearson_bm25_tfidf=0.55983`, `|drift|` vs TASK-017 cached `< 1e-4`,
`blocker_drift_gt_0_1=false`. Supplemental **fusion-signal** Pearsons (production sparse + FAISS IP +
BM25 raw query text) are logged separately in `orthogonality.json` under `fusion_pearson_*`.

**Val embeddings:** Missing `embeddings_val.npy` resolved via GPU-first encode
(Rule 95 ladder 64->32->16->8 CUDA, then CPU 64 in helper), with cache written to
`artifacts/retrieval_e5/20260427_165442_retrieval_e5/embeddings_val.npy`. Grid `val` row count
(10822) differs from orthogonality known-only path (10808); encoder helper invalidates cache on
row-count mismatch and re-encodes.

## Top-5 by val R@10

1. `20260505_211952_three_way_w_0p10_0p70_0p20` mode=weighted_score_minmax lam=(0.10,0.70,0.20) val_R@10=0.8574200702 test_R@10=0.8554707379 val_tail_R@10=0.1707317073
2. `20260505_211915_three_way_w_0p10_0p60_0p30` mode=weighted_score_minmax lam=(0.10,0.60,0.30) val_R@10=0.8572352615 test_R@10=0.8529262087 val_tail_R@10=0.1219512195
3. `20260505_213049_three_way_w_0p10_0p60_0p30` mode=weighted_score_zscore lam=(0.10,0.60,0.30) val_R@10=0.8565884310 test_R@10=0.8525869381 val_tail_R@10=0.1138211382
4. `20260505_213128_three_way_w_0p10_0p70_0p20` mode=weighted_score_zscore lam=(0.10,0.70,0.20) val_R@10=0.8562188135 test_R@10=0.8564885496 val_tail_R@10=0.1788617886
5. `20260505_212146_three_way_w_0p20_0p60_0p20` mode=weighted_score_minmax lam=(0.20,0.60,0.20) val_R@10=0.8545555350 test_R@10=0.8502120441 val_tail_R@10=0.1707317073

## Winner summary (val-primary)

- **run_id:** `20260505_211952_three_way_w_0p10_0p70_0p20`
- **fusion_mode:** `weighted_score_minmax`
- **lambda (sparse,dense,bm25):** (0.10, 0.70, 0.20)
- **val_R@10:** 0.8574200702
- **test_R@10:** 0.8554707379

## Best test R@10 (secondary pick)

- **run_id:** `20260505_213128_three_way_w_0p10_0p70_0p20`
- **fusion_mode:** `weighted_score_zscore`
- **lambda (sparse,dense,bm25):** (0.10, 0.70, 0.20)
- **test_R@10:** 0.8564885496 (val_R@10=0.8562188135)

## Grid execution summary

- **CSV:** `experiments/three_way_fusion_v1/reports/grid/three_way_grid.csv` (38 rows including one smoke; 37 production grid points)
- **Modes:** `weighted_score_minmax` (18), `weighted_score_zscore` (18), `rrf` (1)
- **RRF baseline:** `20260505_214041_three_way_rrf_k60` val_R@10=0.8191646646 test_R@10=0.8171331637

## Bit-exact Iter-2 smoke

- **Command:** `grid_three_way.py --fusion-mode weighted_score_minmax --lambda-grid "0.14,0.86,0.0" --candidates 30 --device cuda --smoke`
- **run_id:** `20260505_211450_three_way_smoke_w_0p14_0p86_0p00`
- **test R@10:** `0.8161153520` (exact match to `EXPECTED_ITER2_TEST_R10`; `|delta|` = 0)

## Pareto figure

- `reports/figures/three_way_pareto.png` (SPEC path)
- Mirror: `experiments/three_way_fusion_v1/reports/figures/three_way_pareto.png`
- **Note:** Emitted post-grid with `MPLBACKEND=Agg` (finalize step left experiment figures dir empty in this workspace run).

## Recommendation for TASK-030 / DECISION-045

Gate **PASS:** adopt 3-way fusion hyperparameters from val-primary winner
`(lambda_sparse, lambda_dense, lambda_bm25)=(0.10, 0.70, 0.20)`, `weighted_score_minmax`, `candidates=30`,
as Iter-3 `final.yaml` candidate before production promotion task. Val tail R@10 on the winner
(0.171) is below the partial-pass-only tail threshold (0.27); monitor tail on test and calibration
before shipping.

## Orthogonality JSON (abridged)

See `experiments/three_way_fusion_v1/reports/orthogonality.json` for full payload (TASK-017 + fusion-path fields).
