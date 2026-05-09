from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any


def ensure_utf8_stdio() -> None:
    """Wrap stdout/stderr with UTF-8 text wrappers when buffers exist."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def read_json_utf8(path: str | Path) -> dict[str, Any]:
    """Read UTF-8 JSON."""
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return data


def write_json_utf8(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write UTF-8 JSON with deterministic formatting."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return out
