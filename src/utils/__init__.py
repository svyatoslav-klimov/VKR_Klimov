"""Cross-cutting utilities: seeding, logging, manifest, encoding.

Modules
-------
- ``seed``: deterministic seed setup (numpy, torch, random; SPEC §11).
- ``logging``: structured logging with run_id context.
- ``run_manifest``: ``build_manifest(...)``,
  ``file_sha256``, ``taxonomy_version`` helpers
  (SPEC §4.6, skill ``run-manifest-builder``).
- ``encoding``: UTF-8 IO helpers for Windows
  (SPEC §13 R-007, ``.cursor/rules/60-windows-encoding.mdc``).
- ``paths``: canonical artifact path resolver (refuses ``exp/**``,
  R-008).
"""
