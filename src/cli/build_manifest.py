from __future__ import annotations

import argparse

from src.utils.run_manifest import build_manifest


def main(args: argparse.Namespace) -> int:
    """CLI stub for build-manifest."""
    build_manifest(config_path=args.config)
    return 0
