from __future__ import annotations

import argparse
import json

from src.data.splits_version import splits_version, write_splits_version


def main(args: argparse.Namespace) -> int:
    """CLI command for deterministic splits_version_legacy report."""
    paths = [part.strip() for part in str(args.paths).split(",") if part.strip()]
    if len(paths) != 3:
        raise ValueError(
            "Expected exactly 3 comma-separated paths in --paths: train,val,test"
        )

    payload = splits_version(
        train_path=paths[0],
        val_path=paths[1],
        test_path=paths[2],
        legacy_random_val=True,
    )
    write_splits_version("reports/split/splits_version_legacy.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
