"""Unit tests for tools/benchmark_latency.py (TASK-034, no heavy model load)."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_bm():
    path = REPO / "tools" / "benchmark_latency.py"
    spec = importlib.util.spec_from_file_location("benchmark_latency_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bm():
    return _load_bm()


def test_hardware_id_stable(bm):
    parts = {
        "cpu_model": "TestCPU",
        "ram_gb": "ram_gb=16",
        "gpu_name": "Test GPU",
        "vram_gb": "vram_gb=8",
        "cuda": "cuda=12.0",
        "torch": "torch=2.0.0",
        "py": "py=3.11.0",
    }
    fp = bm.hardware_fingerprint_string(parts)
    expected = hashlib.sha256(fp.encode("utf-8")).hexdigest()[:16]
    assert bm.hardware_id_from_string(fp) == expected


def test_percentiles_match_numpy(bm):
    laps = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=np.float64)
    p50, p95, p99 = bm.compute_percentiles_ms(laps)
    assert p50 == float(np.percentile(laps, 50))
    assert p95 == float(np.percentile(laps, 95))
    assert p99 == float(np.percentile(laps, 99))


def test_csv_schema_round_trip(bm, tmp_path: Path):
    row = {
        "family": "sparse",
        "run_id": "rid",
        "device": "cpu",
        "batch_size": 32,
        "n_warmup": 10,
        "n_bench_actual": 1000,
        "n_bench_target": 1000,
        "p50_ms": 1.5,
        "p95_ms": 2.5,
        "p99_ms": 3.0,
        "throughput_texts_per_sec": 100.0,
        "model_load_seconds_cold": 0.1,
        "hardware_id": "a" * 16,
        "recall_at_10_test": 0.7,
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "notes": "",
    }
    p = tmp_path / "t.csv"
    bm.write_csv_append(p, row, bm.LATENCY_CSV_COLUMNS, write_header=True)
    df = pd.read_csv(p, encoding="utf-8")
    assert list(df.columns) == bm.LATENCY_CSV_COLUMNS
    assert df.iloc[0]["family"] == "sparse"
    assert int(df.iloc[0]["n_bench_actual"]) == 1000


def test_compute_percentiles(bm):
    laps = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    p50, p95, p99 = bm.compute_percentiles_ms(laps)
    assert p50 == 30.0
    assert abs(p95 - 48.0) < 1e-6
    assert abs(p99 - 49.6) < 0.01


def test_n_bench_adaptive_calc(bm):
    n, note = bm.adaptive_n_bench_from_probe(
        device="cuda",
        batch_size=64,
        n_bench_target=1000,
        p50_probe_ms=5000.0,
        max_cell_min=30.0,
    )
    assert n == 1000
    assert note == ""

    n2, note2 = bm.adaptive_n_bench_from_probe(
        device="cpu",
        batch_size=128,
        n_bench_target=1000,
        p50_probe_ms=500_000.0,
        max_cell_min=30.0,
    )
    assert n2 < 1000
    assert n2 >= 200
    assert "reduced" in note2 or note2 == "reduced_n_bench_due_to_cpu_budget"


def test_md_generation_minimal(bm, tmp_path: Path):
    lat = tmp_path / "latency_full.csv"
    cold = tmp_path / "cold_vs_warm.csv"
    rows = []
    for i, fam in enumerate(bm.CANONICAL_RUNS):
        rows.append(
            {
                "family": fam,
                "run_id": bm.CANONICAL_RUNS[fam][0],
                "device": "cuda",
                "batch_size": 32,
                "n_warmup": 10,
                "n_bench_actual": 1000,
                "n_bench_target": 1000,
                "p50_ms": 10.0 + i,
                "p95_ms": 20.0,
                "p99_ms": 30.0,
                "throughput_texts_per_sec": 50.0,
                "model_load_seconds_cold": 1.0,
                "hardware_id": "abc",
                "recall_at_10_test": 0.8 - i * 0.01,
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "notes": "",
            }
        )
    pd.DataFrame(rows)[bm.LATENCY_CSV_COLUMNS].to_csv(lat, index=False, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "family": "sparse",
                "run_id": "r",
                "device": "cpu",
                "model_load_seconds_cold": 1,
                "model_load_seconds_warm": 0.5,
                "first_inference_ms": 10,
                "post_warmup_p50_ms": 5,
                "hardware_id": "x",
                "timestamp_utc": "2026-01-01T00:00:00Z",
            }
        ]
    )[bm.COLD_WARM_COLUMNS].to_csv(cold, index=False, encoding="utf-8")
    out_md = tmp_path / "latency_full.md"
    bm.generate_latency_markdown(
        REPO,
        lat,
        cold,
        out_md,
        "../figures/latency_pareto.png",
        wall_s=120.0,
        hardware_expanded="cpu=test",
        hardware_id="deadbeef",
        gpu_fallback_ladder=[],
    )
    txt = out_md.read_text(encoding="utf-8")
    assert "hardware" in txt.lower() and "fingerprint" in txt.lower()
    assert "Pareto" in txt
    for fam in bm.CANONICAL_RUNS:
        assert fam in txt


def test_pareto_figure_smoke(bm, tmp_path: Path):
    lat = tmp_path / "latency_full.csv"
    rows = []
    for fam in bm.CANONICAL_RUNS:
        rows.append(
            {
                "family": fam,
                "run_id": bm.CANONICAL_RUNS[fam][0],
                "device": "cuda",
                "batch_size": 32,
                "n_warmup": 10,
                "n_bench_actual": 1000,
                "n_bench_target": 1000,
                "p50_ms": 12.0,
                "p95_ms": 20.0,
                "p99_ms": 25.0,
                "throughput_texts_per_sec": 40.0,
                "model_load_seconds_cold": 1.0,
                "hardware_id": "abc",
                "recall_at_10_test": 0.75,
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "notes": "",
            }
        )
    pd.DataFrame(rows)[bm.LATENCY_CSV_COLUMNS].to_csv(lat, index=False, encoding="utf-8")
    png = tmp_path / "p.png"
    bm.write_pareto_figure(lat, png)
    assert png.is_file()
    assert png.stat().st_size > 0
