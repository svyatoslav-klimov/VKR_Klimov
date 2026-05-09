"""Top-level package: thematic-suggestion module for citizen appeals.

See `ai_docs/SPEC.md` §3 for module map and §5 for required signatures.

Subpackages
-----------
- ``src.data``   : prepare / split / build_topics / splits_version
- ``src.models`` : sparse, retrieval, hybrid, baselines, io
- ``src.eval``   : metrics, slices, prefix_eval
- ``src.utils``  : seed, logging, run_manifest, encoding, paths
- ``src.cli``    : entry-points (see SPEC §10)

Legacy code from the previous iteration is preserved read-only under
``src._legacy/`` for reference (mapping is documented in
``ai_docs/architecture_init_report.md``). New code MUST NOT import from
``src._legacy`` or write to ``exp/`` (R-008).
"""
