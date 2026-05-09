from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.utils import run_manifest


def _minimal_manifest(tmp_path: Path, *, legacy_splits: bool) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "data": {
                    "train_path": "data/splits/train.parquet",
                    "val_path": "data/splits/val.parquet",
                    "test_path": "data/splits/test.parquet",
                    "topics_path": "data/processed/topics.csv",
                },
                "baseline_most_frequent": {},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return run_manifest.build_manifest(
        run_id="20990101_000000_smoke",
        family="baseline_most_frequent",
        config_path=str(cfg_path),
        config=config,
        artifacts={"model": str(tmp_path / "model.pkl")},
        retrieval_meta=None,
        legacy_splits=legacy_splits,
    )


def _sample_manifest() -> dict:
    config_path = Path("configs/baseline.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return run_manifest.build_manifest(
        run_id="20260101_000000_test",
        family="baseline_most_frequent",
        config_path=str(config_path),
        config=config,
        artifacts={"model": "artifacts/baseline_most_frequent/20260101_000000_test/model.pkl"},
        retrieval_meta=None,
        legacy_splits=True,
    )


def test_write_manifest_rejects_exp_path() -> None:
    payload = _sample_manifest()
    legacy_root = "".join(map(chr, (101, 120, 112)))
    try:
        run_manifest.write_manifest(payload, Path(legacy_root) / "foo.json")
    except ValueError as exc:
        assert str(exc) == "R-008: exp/ is frozen read-only"
    else:
        raise AssertionError("Expected ValueError for exp path")


def test_manifest_contains_required_fields() -> None:
    payload = _sample_manifest()
    required_keys = {
        "run_id",
        "model_family",
        "created_at",
        "config_path",
        "config_hash",
        "seed",
        "data",
        "preprocessing",
        "model",
        "artifacts",
        "environment",
        "metrics_path",
        "splits",
    }
    assert required_keys.issubset(payload.keys())
    assert "retrieval_meta" not in payload
    assert payload["config_hash"] == run_manifest.file_sha256("configs/baseline.yaml")


def test_taxonomy_version_deterministic() -> None:
    first = run_manifest.taxonomy_version("data/processed/topics.csv")
    second = run_manifest.taxonomy_version("data/processed/topics.csv")
    assert first == second
    assert first.startswith("sha256:")


def test_package_versions_are_strings() -> None:
    versions = run_manifest.package_versions(["numpy", "definitely_missing_pkg_name"])
    assert isinstance(versions["numpy"], str)
    assert versions["definitely_missing_pkg_name"] == "not_installed"


def test_splits_block_present(tmp_path: Path) -> None:
    payload = _minimal_manifest(tmp_path, legacy_splits=False)
    assert "splits" in payload


def test_splits_block_fields(tmp_path: Path) -> None:
    payload = _minimal_manifest(tmp_path, legacy_splits=False)
    assert set(payload["splits"].keys()) == {"version", "legacy_random_val", "policy"}


def test_splits_block_consistency_with_data(tmp_path: Path) -> None:
    for legacy in (False, True):
        payload = _minimal_manifest(tmp_path / f"case_{legacy}", legacy_splits=legacy)
        assert payload["splits"]["version"] == payload["data"]["splits_version"]
        assert payload["splits"]["legacy_random_val"] == legacy


def test_splits_policy_default_time_based(tmp_path: Path) -> None:
    off = _minimal_manifest(tmp_path / "tb", legacy_splits=False)
    assert off["splits"]["policy"] == "time_based"
    on = _minimal_manifest(tmp_path / "rs", legacy_splits=True)
    assert on["splits"]["policy"] == "random_stratified"


def test_write_manifest_creates_json() -> None:
    payload = _sample_manifest()
    path = Path("reports") / "tmp_manifest_test.json"
    run_manifest.write_manifest(payload, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == payload["run_id"]
    path.unlink(missing_ok=True)
