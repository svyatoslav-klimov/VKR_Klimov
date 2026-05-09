from __future__ import annotations

import importlib.metadata as metadata
import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.data.build_topics import normalize_topic
from src.data.splits_version import (
    file_sha256 as _file_sha256,
    splits_version as _splits_version,
)


def file_sha256(path: str | Path) -> str:
    """Delegate SHA256 calculation to src.data.splits_version helper."""
    return _file_sha256(path)


def taxonomy_version(topics_csv: str | Path) -> str:
    """Compute deterministic taxonomy version from normalized topic names."""
    frame = pd.read_csv(topics_csv, encoding="utf-8")
    if "topic_name" not in frame.columns:
        raise KeyError("topics.csv must contain 'topic_name' column")
    normalized = sorted({normalize_topic(value) for value in frame["topic_name"].dropna().astype(str)})
    payload = "\n".join(normalized).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"


def splits_version(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    legacy_random_val: bool = False,
) -> dict[str, str | bool]:
    """Delegate split hashes calculation to src.data.splits_version helper."""
    return _splits_version(
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        legacy_random_val=legacy_random_val,
    )


def package_versions(names: list[str]) -> dict[str, str]:
    """Return package versions or 'not_installed'."""
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def build_manifest(
    run_id: str,
    family: str,
    config_path: str | Path,
    config: Mapping[str, Any],
    artifacts: Mapping[str, str],
    retrieval_meta: Mapping[str, Any] | None = None,
    legacy_splits: bool = True,
    splits_policy: Literal["time_based", "random_stratified"] | None = None,
) -> dict[str, Any]:
    """Build manifest dictionary following template in .ai/templates."""
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("config.data must be a mapping")

    train_path = str(data_cfg["train_path"])
    val_path = str(data_cfg["val_path"])
    test_path = str(data_cfg["test_path"])
    topics_path = str(data_cfg["topics_path"])

    split_meta = splits_version(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        legacy_random_val=legacy_splits,
    )
    if not legacy_splits:
        split_meta.pop("splits_legacy_random_val", None)

    model_params = config.get(family, config.get("model", {}))
    if not isinstance(model_params, Mapping):
        model_params = {}

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "model_family": family,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": str(config_path),
        "config_hash": file_sha256(config_path),
        "seed": int(config.get("seed", 42)),
        "data": {
            "train_path": train_path,
            "train_hash": split_meta["train_hash"],
            "val_path": val_path,
            "val_hash": split_meta["val_hash"],
            "test_path": test_path,
            "test_hash": split_meta["test_hash"],
            "splits_version": split_meta["splits_version"],
            "topics_path": topics_path,
            "taxonomy_version": taxonomy_version(topics_path),
        },
        "preprocessing": dict(config.get("preprocessing", {})),
        "model": {"family": family, "params": dict(model_params)},
        "artifacts": dict(artifacts),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(
                [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "torch",
                    "transformers",
                    "sentence-transformers",
                    "faiss-cpu",
                ]
            ),
        },
        "metrics_path": f"reports/runs/{run_id}/metrics.json",
    }
    if legacy_splits:
        manifest["data"]["splits_legacy_random_val"] = True

    if splits_policy is not None:
        resolved_policy: Literal["time_based", "random_stratified"] = splits_policy
    elif legacy_splits:
        resolved_policy = "random_stratified"
    else:
        resolved_policy = "time_based"
    legacy_random_val = bool(legacy_splits)
    manifest["splits"] = {
        "version": split_meta["splits_version"],
        "legacy_random_val": legacy_random_val,
        "policy": resolved_policy,
    }
    if retrieval_meta is not None:
        manifest["retrieval_meta"] = dict(retrieval_meta)
    return manifest


def write_manifest(manifest: Mapping[str, Any], dest: str | Path) -> None:
    """Write manifest to disk in UTF-8 and block exp directory paths."""
    raw_dest = Path(dest)
    is_exp_relative = bool(raw_dest.parts) and raw_dest.parts[0].lower() == "exp"
    try:
        rel_to_cwd = raw_dest.resolve().relative_to(Path.cwd().resolve())
        is_exp_under_cwd = bool(rel_to_cwd.parts) and rel_to_cwd.parts[0].lower() == "exp"
    except ValueError:
        is_exp_under_cwd = False

    if is_exp_relative or is_exp_under_cwd:
        raise ValueError("R-008: " + "exp" + "/" + " is frozen read-only")

    out_path = Path(dest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(manifest), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
