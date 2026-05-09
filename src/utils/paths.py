from __future__ import annotations

from pathlib import Path


def ensure_writable_path(path: str | Path) -> Path:
    """Reject writes into frozen legacy exp/ paths (R-008)."""
    raw = Path(path)
    parts = raw.parts
    if parts and parts[0].lower() == "exp":
        raise RuntimeError(f"R-008: writing to legacy frozen exp directory is forbidden: {path}")
    return raw
