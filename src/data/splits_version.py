from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path


def file_sha256(path: str | PathLike[str]) -> str:
    """Return sha256 checksum with prefix for a binary file."""
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def splits_version(
    train_path: str,
    val_path: str,
    test_path: str,
    legacy_random_val: bool = True,
) -> dict[str, str | bool]:
    """Build split hashes payload for legacy split manifest sections."""
    train_hash = file_sha256(train_path)
    val_hash = file_sha256(val_path)
    test_hash = file_sha256(test_path)

    aggregate = hashlib.sha256(
        f"{train_hash}\n{val_hash}\n{test_hash}".encode("utf-8")
    ).hexdigest()

    return {
        "train_path": train_path,
        "train_hash": train_hash,
        "val_path": val_path,
        "val_hash": val_hash,
        "test_path": test_path,
        "test_hash": test_hash,
        "splits_version": f"sha256:{aggregate}",
        "splits_legacy_random_val": legacy_random_val,
    }


def build_splits_version(paths: list[str] | tuple[str, ...]) -> dict[str, str | bool]:
    """Compatibility wrapper for CLI calls with comma-separated paths."""
    if len(paths) != 3:
        raise ValueError("Expected exactly 3 paths: train,val,test")
    return splits_version(paths[0], paths[1], paths[2], legacy_random_val=True)


def write_splits_version(report_path: str | Path, payload: dict[str, str | bool]) -> Path:
    """Write splits version payload as UTF-8 JSON with deterministic formatting."""
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return out_path
