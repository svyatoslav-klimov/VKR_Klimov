# Iter-2 formal closed-tasks audit — TASK-038

- **Auditor:** ROLE_04 (QA / Audit)
- **Date (UTC):** 2026-05-05
- **Scope:** Iter-2 only (DECISION-2026-05-05-044 freeze; DECISION-2026-04-28-034 §1 deferred audit). TASK-037 / TASK-040 / TASK-030 intentionally **out of scope** (Iter-3 backlog per plan §6.4).
- **Primary inputs:** `.ai/plans/2026-05-06_TASK-038_audit-report-iter2.md` §9 verbatim mandate; `reports/iteration_2_close.md`; `ai_docs/SPEC.md` anti-leak / anti-mix / FAISS / risk register; eight Iter-2 task plans TASK-010–021; `reports/leaderboard.csv` (25 Iter-2-visible rows after excluding `three_way_hybrid`).

---

## §1 — Audit context

Hand-off pre-condition from DECISION-034 §1 deferred a full ROLE_04 sign-off until Iter-3 as **TASK-038**. This report closes that carry-over for the **eight closed Iter-2 tasks** plus the **Iter-2 winner** and **TASK-018 / TASK-017 / TASK-020 artefacts** named in the TASK-038 plan.

**Method.** ROLE_04 §2 checklist was applied with machine assistance from `tools/audit_iter2.py` (leakage timestamps + row-overlap on canonical parquets; manifest `splits_version` cross-alignment; FAISS centroid L2-norm sweep for seven retrieval directories; n=20 stratified `HybridModel.predict_topk` sanity on winner + TASK-018 reproduction; classes.json SHA256 parity). Narrative cross-checks covered manifest schema presence, `metrics.json` diagnostics, R-008 active-code scope, and forbidden `experiments/**` imports.

**Infra note.** This run executed on a workstation with `data/splits/*.parquet`, `themes.faiss`, and GPU available; where a partial checkout lacks those assets, the audit script documents `PASS_ARCHIVE` or skips live FAISS norms (see JSON `infra_note_parquet`).

---

## §2 — Per-task verdict (8 closed tasks)

| Task | Verdict | Evidence (1 line) |
|---|---|---|
| TASK-010 Manifest / leaderboard | ✓ | All 25 leaderboard `run_id`s resolve to a JSON manifest with `splits` + `data.splits_version` + `taxonomy_version` + `environment.python` + packages (python check 2026-05-05). |
| TASK-011 Hybrid batched predict | ✓ | Sanity `predict_topk` used production `HybridModel.load` / batched encode (CUDA bs=64 ladder first step) with no load errors. |
| TASK-012 Prefix UX | ✓ | Out of primary metric path; leaderboard prefix_eval rows + DECISION-042 waiver documented in `iteration_2_close.md`. |
| TASK-013 Tail diagnostics | ✓ | `reports/tail_diagnostics/` present per scoreboard; verdict `synthetic-only-option` is architectural, not a maths bug. |
| TASK-017 BM25 | ✓ | Winner manifest under `experiments/bm25_v1/artifacts/manifests/`; `classes.json` SHA256 matches sparse/retrieval canon. |
| TASK-018 Encoder ablation | ✓ | Seven retrieval dirs checked: `normalize_embeddings=true`, `ntotal==len(classes)`, max L2 deviation from 1.0 below 1e-6 (actually ~2.6e-8). Reproduction run `20260505_115158_*` R@10 equals winner JSON bit-for-bit. |
| TASK-020 Calibration | **Major** | **R-016.b violation:** `experiments/calibration_v1/src/` imports `src.models.hybrid` (`fit_calibration.py`, `hybrid_top1_scores.py`). See §12 / `reports/audit_findings_iter2.csv` AUD-20260505-01. |
| TASK-021 Error analysis | ✓ | Bundle paths listed in `iteration_2_close.md`; prior architect closure without Blocker recorded; this audit did not re-parse large bundle tables line-by-line (scope = methodology gates). |

---

## §3 — R-001 (FAISS / embedding norm) verification

Executed `faiss.read_index` + `IndexFlat` centroid `reconstruct(i)` for retrieval run directories:

`20260427_165442_*`, `20260505_103930_*`, `20260505_111402_*`, `20260505_112504_*`, `20260505_113354_*`, `20260505_115333_*`, `20260505_124426_*`.

Results in `reports/leakage_checks.json` under `r001_fais_norm_checks`: all `ok: true`; `meta_normalize_embeddings: true`; `max_norm_delta` O(1e-8); `ntotal_equals_classes: true`. No `deserialize_index` usage detected in `src/` (search 2026-05-05).

---

## §4 — R-002 / R-008 isolation

- **Manifest coverage:** Every Iter-2-visible leaderboard row maps to `artifacts/manifests/<run_id>.json` **or** `experiments/<exp>/artifacts/manifests/<run_id>.json` (see `leakage_checks.json` `manifest_found` field).  
- **R-008 writes to `exp/`:** Active training / eval code under `src/` (excluding frozen `src/_legacy/**` per protocol) shows no new write paths to `./exp/` when searching for `exp/` string usage; remaining `exp/` strings are confined to `_legacy` stubs and comments. `configs/base.yaml` still references `exp/reports/...` in a template field — acceptable as config residue if unused by Iter-2 runs; **watch** for accidental reactivation.

---

## §5 — R-003 anti-mixing (hybrid winner)

From `reports/runs/20260427_171719_hybrid_weighted_score_minmax/metrics.json` `hybrid_diagnostics`:

- `pearson_sparse_dense_val` ≈ 0.2247 (< 0.7 rule-of-thumb from hybrid skill).
- `top1_change_rate_val` ≈ 0.1016 (well above 5% anti-collapse threshold).
- λ-endpoint grid: `lambda=0` recall@10 ≈ 0.7725; `lambda=1.0` recall@10 ≈ 0.6989; best λ=0.14 recall@10 matches headline 0.8161.

**Fusion materially helps:** headline R@10 − max(endpoint components) ≈ +0.0436 (> +0.005 requirement in plan A4).

---

## §6 — R-004 class consistency

`tools/audit_iter2.py` recorded matching `classes.json` raw SHA256 for:

- Winner hybrid `20260427_171719_*`
- Sparse source `20260427_143358_*`
- Retrieval source `20260427_165442_*`
- BM25 winner `20260429_023849_*` (under `experiments/bm25_v1/artifacts/sparse/...`)

See `reports/metric_sanity_checks.json` → `winner_sparse_retrieval_bm25_classes_hashes` (all hashes identical; `mismatch` list empty).

**Taxonomy.** All 25 manifests share one `taxonomy_version` prefix `sha256:85f23a78...` (see `leakage_checks.json` taxonomy histogram).

---

## §7 — R-005 anti-leakage

`tools/audit_iter2.py` reproduced SPEC §6.4 ordering:

`train.created_at.max() < val.created_at.min() < val.created_at.max() < test.created_at.min()`

with **zero** `text` overlap train/val and val/test (`row_overlap_* = 0` for canonical parquets; see first rows in `leakage_checks.json`).

`splits_version` hash `sha256:819b765d...` is uniform across audited manifests (`manifest_splits_version_match: true`). Code-level proofs that TF-IDF / BM25 statistics / FAISS centroids ingest **train-only** texts were **not line-audited in this pass** beyond existing TASK-010 / TASK-018 post-mortems; no contradictory signal found in manifests (hashes align).

---

## §8 — Reproducibility (SPEC §11)

- Winner vs TASK-018 reproduction metrics JSON (`20260505_115158_*`): **`recall_at_k["10"]` identical representations in Python** (difference 0.0 when parsed from twin JSON floats). Matches DECISION-044 evidence (Δ ~ 3.5e-7 narrative — below float print precision in this archival pair).
- `seed: 42` present in manifest + metrics for winner path.
- `metric_sanity_checks.json`: n=20 fresh `HybridModel.predict_topk` CUDA bs=64, sample recall = 0.85 vs archived 0.8161 (`abs_diff` ~ 0.034, `ok=true` under plan tolerance for finite sample variance).

---

## §9 — Manifest schema / R-011 / R-012 (TASK-010)

**PASS (baseline TASK-010 bar):**

- Mandatory `environment` stanza populated with Python version + dependency dict (manifest sweep over 25 rows — no failures).

**GAP vs TASK-038 plan wording (Minor documentation drift, not maths):**

The plan bullet listed top-level manifest keys such as `data.training_data_policy` and mirrored `head_mid_tail_* thresholds` duplicated from YAML. Older hybrid manifests encode practical thresholds inside `model.params.fusion.diversity_*` rather than mirrored duplicate keys — functionally adequate because `configs/best.yaml` + frozen `fusion_config.json` remain canonical. Recommend aligning future manifests with TASK-036 `training_data_policy` requirement without rewriting historical manifests.

Leaderboard satisfies TASK-010 22-column contract for Iter-2 rows (CSV header validated).

---

## §10 — head_mid_tail practical + strict parity (SPEC §6.3)

Winner `metrics.json` contains **`head_mid_tail`** (mode `practical`) and **`head_mid_tail_strict`** sibling objects with thresholds matching `configs/best.yaml` eval block (50/10 practical; 500/50 strict).

---

## §11 — R-013 / R-014 / R-015 audit-trail integrity

- Eight Iter-2 task plans exist under `.ai/plans/` with matching TASK_QUEUE references (grep 2026-05-05).  
- **Git-delete audit:** Local `git log --diff-filter=D --name-only -- .ai/plans/` returned exit code **128** (environment not furnishing full git metadata in this Cursor session); R-013 “no deletion” clause **not independently machine-verified here**. Recommendation: rerun on CI/dev clone with intact git DB.

---

## §12 — R-016 / change_category sanity

Frontmatter categories in task plans align with artefacts described in `reports/iteration_2_close.md` (experiment vs production_diag vs production_grid). **Formal git-diff cross-correlation per plan vs frontmatter was not executed** — low yield given frozen plans + architect scoreboard alignment.

---

## §13 — R-016.b experiment leakage grep (hard gate)

```
rg '^from src\.models' experiments/ -g '*.py'
rg '^from src\.cli' experiments/ -g '*.py'
```

yielded **`experiments/calibration_v1/src/fit_calibration.py`** and **`hybrid_top1_scores.py`** importing `HybridModel`.

**Severity Major** relative to whitelist (Rule 85). Calibration outputs remain usable offline but the import pattern violates DECISION-036 isolation spelled out for ROLE_04 acceptance **A12**. Tracked as **AUD-20260505-01**.

---

## §14 — R-006 / R-007 subprocess + encoding

Spot checks:

- Tools tree `subprocess` calls did **not** show naive `"python"` string invocation (grep 2026-05-05). Production guidance remains `sys.executable` per R-006.  
- Deliverables (`docs/audit_report_iter2.md`, JSON, CSV, tests, `tools/audit_iter2.py`) written UTF-8 without BOM (`ensure_ascii=false` dumps). Random report inspection for mojibake not escalated beyond spot UTF-8 read of leaderboard + manifests.

---

## §15 — Findings summary

| ID | Sev | Brief |
|---|---|---|
| AUD-20260505-01 | Major | R-016.b: `src.models.hybrid` import inside `experiments/calibration_v1/src/` (two files). |

Full row: `reports/audit_findings_iter2.csv`.

---

## §16 — Recommendations

1. **REWORK TASK-020 (calibration codebase isolation)** OR publish an explicit DECISION-record waiving experiment import ban for calibrated score extraction — must cite why fork is infeasible and document blast radius. Preferred engineering path: relocate minimal loader code into `experiments/calibration_v1/src/` without touching `src.models.*`.  
2. **CI hook:** Extend QA script to fail on `rg '^from src\\.(models|cli)' experiments/`.  
3. **Optional cleanliness:** Dedup ambiguous duplicate leaderboard retrieval row pair (`20260505_111402_*` vs `20260505_111627_*` identical metrics — data hygiene only).  
4. **Git history spot-check** for `.ai/plans/` deletions once repository metadata available locally.

Until **Recommendation (1)** is resolved, packaged Iter-3 delivery gates should treat TASK-038 as **PARTIAL PASS** rather than unconditional green.

---

## §17 — Sign-off

**Final verdict:** **PARTIAL PASS** — core Iter-2 metric integrity, reproducibility artefacts, leakage ordering, anti-mixing diagnostics, class hashes, FAISS norms PASS; **experiment isolation firewall fails A12 / R-016.b** pending TASK-020 fix or waived DECISION.

**Acceptance checklist (TASK-038 plan §5 A1–A17):**

- Passed with evidence:** A1,A2,A3,A4,A5,A6,A7,A8,A10 (minus git-delete verify), A13,A14,A15,A16,A17.  
- **Partial / qualified:** **A9** (manifest mirrors optional keys vs textual plan wording), **A11** (no diff-based category proof), **A12** (**FAIL until calibration import removed**).

**Artifacts**

| Path | Role |
|---|---|
| `docs/audit_report_iter2.md` | This narrative |
| `reports/leakage_checks.json` | Automated leakage + `r001` matrix |
| `reports/metric_sanity_checks.json` | n=20 sanity + classes hashes |
| `reports/audit_findings_iter2.csv` | Formal finding row(s) |
| `tools/audit_iter2.py` | Reproducer |
| `tests/test_audit_iter2.py` | Regression clamps |

ROLE_04 — TASK-038 — Iter-2 formal audit closure **2026-05-05 (UTC)**.

---

## Appendix — Commands executed (ROLE_04 self-execution)

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe tools/audit_iter2.py
.\.venv\Scripts\python.exe -m pytest tests/test_audit_iter2.py -v
```

Additional manual greps documented in §11 / §13 / §14.
