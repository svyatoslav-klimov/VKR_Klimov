from __future__ import annotations

from pathlib import Path

import pytest

from src.data.splits_version import splits_version


TRAIN_PATH = "data/splits/train.parquet"
VAL_PATH = "data/splits/val.parquet"
TEST_PATH = "data/splits/test.parquet"


def _exp_file_snapshot() -> set[str]:
    exp_dir = Path("exp")
    if not exp_dir.exists():
        return set()
    return {
        str(path.relative_to(exp_dir)).replace("\\", "/")
        for path in exp_dir.rglob("*")
        if path.is_file()
    }


def test_splits_version_is_deterministic_and_has_required_fields() -> None:
    before_exp = _exp_file_snapshot()
    first = splits_version(TRAIN_PATH, VAL_PATH, TEST_PATH, legacy_random_val=True)
    second = splits_version(TRAIN_PATH, VAL_PATH, TEST_PATH, legacy_random_val=True)
    after_exp = _exp_file_snapshot()

    assert first == second
    assert after_exp == before_exp
    assert first["splits_legacy_random_val"] is True

    for key in ("train_hash", "val_hash", "test_hash", "splits_version"):
        value = first[key]
        assert isinstance(value, str)
        assert value.startswith("sha256:")
        assert len(value) == 71


def test_splits_version_raises_on_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        splits_version("data/splits/does_not_exist.parquet", VAL_PATH, TEST_PATH)
