"""Materialize three-way fusion artifacts (TASK-040)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.utils.run_manifest import build_manifest, file_sha256, write_manifest

logger = logging.getLogger(__name__)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def train_three_way_cmd(config_path: str) -> str:
    """Write ``artifacts/three_way_hybrid/<run_id>/`` + manifest stub."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = _load_yaml(config_path)
    family = str(config.get("model_family", "three_way_hybrid"))
    if family != "three_way_hybrid":
        raise ValueError(f"model_family must be three_way_hybrid, got {family!r}")

    tw = dict(config.get("three_way") or config.get("three_way_hybrid") or {})
    if not tw:
        raise ValueError("config must contain three_way or three_way_hybrid section")

    sparse_dir = str(tw["sparse_artifacts_dir"]).replace("\\", "/")
    ret_dir = str(tw["retrieval_artifacts_dir"]).replace("\\", "/")
    bm25_dir = str(tw["bm25_artifacts_dir"]).replace("\\", "/")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_ts}_three_way_promoted"

    out_dir = Path("artifacts") / "three_way_hybrid" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(sparse_dir) / "classes.json", out_dir / "classes.json")
    cls_sha = file_sha256(out_dir / "classes.json")

    fusion_payload: dict[str, Any] = {
        "fusion_mode": str(tw.get("fusion_mode", "weighted_score_minmax")),
        "norm": str(tw.get("norm", "minmax")),
        "candidates": int(tw.get("candidates", 30)),
        "lambda_sparse": float(tw["lambda_sparse"]),
        "lambda_dense": float(tw["lambda_dense"]),
        "lambda_bm25": float(tw["lambda_bm25"]),
        "k_rrf": int(tw.get("k_rrf", 60)),
        "sparse_artifacts_dir": sparse_dir,
        "retrieval_artifacts_dir": ret_dir,
        "bm25_artifacts_dir": bm25_dir,
        "classes_sha256": cls_sha,
    }
    if tw.get("iter2_hybrid_lam_sparse_reference") is not None:
        fusion_payload["iter2_hybrid_lam_sparse_reference"] = float(
            tw["iter2_hybrid_lam_sparse_reference"]
        )

    fusion_path = out_dir / "fusion_config.json"
    with fusion_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(fusion_payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    ret_man_path = Path(str(tw.get("retrieval_manifest_path", "")))
    meta_path = out_dir / "meta.json"
    if ret_man_path.is_file():
        with ret_man_path.open("r", encoding="utf-8") as rf:
            rman = json.load(rf)
        rmeta = dict(rman.get("retrieval_meta", rman))
    else:
        with (Path(ret_dir) / "meta.json").open("r", encoding="utf-8") as mf:
            rmeta = dict(json.load(mf))
    rmeta["promotion_run"] = {
        "run_id": run_id,
        "fusion_config": fusion_payload,
    }
    with meta_path.open("w", encoding="utf-8", newline="\n") as mf:
        json.dump(rmeta, mf, ensure_ascii=False, indent=2)
        mf.write("\n")

    ad = str(out_dir).replace("\\", "/")
    artifacts_paths = {
        "fusion_config": f"{ad}/fusion_config.json",
        "classes": f"{ad}/classes.json",
        "meta": f"{ad}/meta.json",
    }
    cfg_posix = str(Path(config_path).as_posix())
    cfg_full = dict(config)
    if "three_way_hybrid" not in cfg_full:
        cfg_full["three_way_hybrid"] = dict(tw)

    manifest = build_manifest(
        run_id,
        family,
        cfg_posix,
        cfg_full,
        artifacts_paths,
        retrieval_meta=rmeta,
        legacy_splits=False,
        splits_policy="time_based",
    )
    manifest["metrics_path"] = f"reports/runs/{run_id}/metrics.json"
    manifest_path = Path("artifacts") / "manifests" / f"{run_id}.json"
    write_manifest(manifest, manifest_path)

    logger.info("Wrote %s", ad)
    print(run_id)
    return run_id


def main(args: argparse.Namespace) -> int:
    train_three_way_cmd(args.config)
    return 0
