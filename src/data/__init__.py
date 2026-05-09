"""Data layer: raw -> interim -> processed -> splits, plus taxonomy build.

Modules
-------
- ``prepare``: raw CSV/parquet -> normalized ``data/interim/clean.csv``
  (lower, NFC, collapse spaces, PII mask, length filter; SPEC §6.2).
- ``split``: ``clean.csv`` -> time-based train/val/test (SPEC §6.1).
- ``build_topics``: unique normalized themes + alias merge ->
  ``data/processed/topics.csv`` and ``taxonomy_version`` (SPEC §4.2.1).
- ``splits_version``: SHA256 hashes of split parquet files used in
  every run manifest (SPEC §4.6, TASK-002a).
"""
