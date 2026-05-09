"""Evaluation: metrics, slices, prefix simulation.

Modules
-------
- ``metrics``: Recall@k, Macro-F1, Weighted-F1, Accuracy@1, MRR@10,
  nDCG@10. Primary metric is Recall@10 (SPEC §6.3, R-004).
- ``slices``: head/mid/tail blocks (practical + strict, SPEC §6.3,
  DECISION-009) and time-slice reports.
- ``prefix_eval``: offline simulation for prefix lengths
  10/20/30/50/100/full with confidence threshold tau
  (SPEC §6.7, configs/prefix.yaml).
"""
