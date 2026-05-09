"""Model families: baselines, sparse, retrieval, hybrid, (P2) cross-encoder.

Modules
-------
- ``baseline_most_frequent``: Top-k most frequent topics in train; lower
  bound for Recall@k (SPEC §2 P0).
- ``sparse``: TF-IDF + LR / LinearSVC / SGD; primary fast classifier
  (SPEC §6.4, configs/sparse.yaml).
- ``retrieval``: bi-encoder + FAISS (centroids / label /
  label_plus_centroid / document_knn) with FAISS-invariant
  (SPEC §6.6, R-001/R-005).
- ``hybrid``: sparse + dense fusion (weighted_score / RRF) with
  anti-mixing diagnostics (SPEC §6.5, R-003).
- ``io``: save/load artifacts, ``meta.json``, ``classes.json``,
  manifest write helpers.
- ``three_way_hybrid``: sparse + dense + BM25 late fusion (TASK-040).
"""

from src.models.three_way_hybrid import ThreeWayHybrid

__all__ = ["ThreeWayHybrid"]
