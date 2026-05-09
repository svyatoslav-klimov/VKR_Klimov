from __future__ import annotations

import argparse

from src.data.build_topics import build_topics


def main(args: argparse.Namespace) -> int:
    """CLI command for deterministic topics build."""
    build_topics(config_path=args.config)
    return 0
