from __future__ import annotations

import argparse

from src.data.prepare import prepare_data


def main(args: argparse.Namespace) -> int:
    """CLI: time-based splits per ``configs/data.yaml`` (TASK-002b)."""
    prepare_data(config_path=args.config)
    return 0
