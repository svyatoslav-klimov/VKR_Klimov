"""Latency and throughput benchmark grid for production inference (TASK-034)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

CANONICAL_RUNS: dict[str, tuple[str, str]] = {
    "sparse": (
        "20260427_143358_sparse_linear_svc",
        "artifacts/sparse_tfidf/20260427_143358_sparse_linear_svc",
    ),
    "retrieval": (
        "20260427_165442_retrieval_e5",
        "artifacts/retrieval_e5/20260427_165442_retrieval_e5",
    ),
    "hybrid": (
        "20260427_171719_hybrid_weighted_score_minmax",
        "artifacts/hybrid/20260427_171719_hybrid_weighted_score_minmax",
    ),
    "three_way_hybrid": (
        "20260507_135733_three_way_promoted",
        "artifacts/three_way_hybrid/20260507_135733_three_way_promoted",
    ),
}

LATENCY_CSV_COLUMNS: list[str] = [
    "family",
    "run_id",
    "device",
    "batch_size",
    "n_warmup",
    "n_bench_actual",
    "n_bench_target",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "throughput_texts_per_sec",
    "model_load_seconds_cold",
    "hardware_id",
    "recall_at_10_test",
    "timestamp_utc",
    "notes",
]

COLD_WARM_COLUMNS: list[str] = [
    "family",
    "run_id",
    "device",
    "model_load_seconds_cold",
    "model_load_seconds_warm",
    "first_inference_ms",
    "post_warmup_p50_ms",
    "hardware_id",
    "timestamp_utc",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hardware_components_dict() -> dict[str, str]:
    """Resolved hardware fingerprint components (before hashing)."""
    import platform

    import torch

    cpu_model = platform.processor() or platform.machine() or "unknown_cpu"

    ram_gb = 0
    ram_marker = ""
    try:
        import psutil

        ram_gb = int(round(psutil.virtual_memory().total / (1024**3)))
    except Exception:
        ram_gb = int(os.cpu_count() or 4) * 2
        ram_marker = "|psutil_missing_estimated"

    gpu_name = "none"
    vram_gb = 0
    cuda_rt = "none"
    if torch.cuda.is_available():
        gpu_name = str(torch.cuda.get_device_name(0))
        cuda_rt = str(torch.version.cuda or "unknown")
        try:
            vram_gb = int(
                round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            )
        except Exception:
            vram_gb = 0

    return {
        "cpu_model": cpu_model,
        "ram_gb": f"ram_gb={ram_gb}{ram_marker}",
        "gpu_name": gpu_name,
        "vram_gb": f"vram_gb={vram_gb}",
        "cuda": f"cuda={cuda_rt}",
        "torch": f"torch={torch.__version__}",
        "py": f"py={sys.version.split()[0]}",
    }


def hardware_fingerprint_string(components: dict[str, str] | None = None) -> str:
    parts = components or hardware_components_dict()
    return "|".join(
        [
            parts["cpu_model"],
            parts["ram_gb"],
            parts["gpu_name"],
            parts["vram_gb"],
            parts["cuda"],
            parts["torch"],
            parts["py"],
        ]
    )


def hardware_id_from_string(fp: str) -> str:
    return hashlib.sha256(fp.encode("utf-8")).hexdigest()[:16]


def compute_hardware_id() -> str:
    return hardware_id_from_string(hardware_fingerprint_string())


def compute_percentiles_ms(laps_ms: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(laps_ms, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def adaptive_n_bench_from_probe(
    *,
    device: str,
    batch_size: int,
    n_bench_target: int,
    p50_probe_ms: float,
    max_cell_min: float,
) -> tuple[int, str]:
    """DECISION-054: CUDA always target; CPU small batch always target; large CPU may reduce."""
    max_cell_ms = max_cell_min * 60_000.0
    p50_safe = max(float(p50_probe_ms), 1e-6)
    need_ms = n_bench_target * p50_safe

    if device == "cuda":
        return n_bench_target, ""

    if device == "cpu" and batch_size <= 32:
        if need_ms <= max_cell_ms:
            return n_bench_target, ""
        return n_bench_target, "may_exceed_max_cell_min"

    # CPU batch 64 or 128
    if need_ms <= max_cell_ms:
        return n_bench_target, ""
    n_cap = max(200, int(max_cell_ms / p50_safe))
    n_cap = min(n_bench_target, n_cap)
    note = "reduced_n_bench_due_to_cpu_budget" if n_cap < n_bench_target else ""
    return max(n_cap, 200), note


def local_hf_snapshot_or_id(model_id: str) -> str:
    """Prefer cached snapshot dir so SentenceTokenizer avoids Hub API (HF 500 / offline quirks)."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = hf_home / "hub"
    slug = "models--" + model_id.replace("/", "--")
    snaps = hub / slug / "snapshots"
    if not snaps.is_dir():
        return model_id
    candidates = sorted(
        [p for p in snaps.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return model_id
    return str(candidates[0])


def read_recall_at_10(repo: Path, run_id: str) -> float:
    p = repo / "reports" / "runs" / run_id / "metrics.json"
    with p.open("r", encoding="utf-8") as f:
        m = json.load(f)
    return float(m["recall_at_k"]["10"])


def load_text_pool(repo: Path, pool_path: Path, n_rows: int, seed: int) -> list[str]:
    df = pd.read_parquet(pool_path)
    texts = df["text"].astype(str).tolist()
    rng = random.Random(seed)
    if len(texts) > n_rows:
        texts = [texts[i] for i in rng.sample(range(len(texts)), n_rows)]
    return texts


def sample_batch(
    texts: list[str], batch_size: int, rng: np.random.Generator
) -> list[str]:
    idx = rng.integers(0, len(texts), size=batch_size)
    return [texts[int(i)] for i in idx]


# --- Family loaders: return (handle, predict_fn); predict_fn times one batch ---


def _load_sparse(
    artifacts_dir: Path, _device: str, _batch_size: int
) -> tuple[Any, Callable[..., Any]]:
    from src.models import sparse as sparse_mod

    model = sparse_mod.load(artifacts_dir)

    def predict_fn(texts: list[str], k: int = 10) -> Any:
        return sparse_mod.predict_topk(model, texts, k=k)

    return model, predict_fn


def _load_retrieval_impl(
    artifacts_dir: Path, device: str, encode_batch_size: int
) -> tuple[Any, Callable[..., Any]]:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    from src.models.retrieval import encode_passages, search_with_query_embeddings

    root = Path(artifacts_dir).resolve()
    with (root / "meta.json").open("r", encoding="utf-8") as f:
        meta: dict[str, Any] = json.load(f)
    model_name = str(meta["model_name"])
    norm = bool(meta["normalize_embeddings"])
    p_q = str(meta.get("e5_prefix_query", "query: "))
    classes_relp = str(meta.get("classes_path", "classes.json"))
    class_path = (
        root / classes_relp if not Path(classes_relp).is_absolute() else Path(classes_relp)
    )
    with class_path.open("r", encoding="utf-8") as f:
        classes: list[str] = json.load(f)
    index = faiss.read_index(str((root / "themes.faiss").resolve()))
    dev = device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    enc = SentenceTransformer(model_name, device=dev)
    msl = 512
    try:
        for m in enc:
            m.max_seq_length = msl
    except Exception:
        pass
    bs_enc = int(encode_batch_size)

    @dataclass
    class _Bundle:
        encoder: Any
        index: Any
        classes: list[str]
        prefix_q: str
        norm: bool
        encode_bs: int

    bundle = _Bundle(
        encoder=enc,
        index=index,
        classes=classes,
        prefix_q=p_q,
        norm=norm,
        encode_bs=bs_enc,
    )
    handle = bundle

    def predict_fn(texts: list[str], k: int = 10) -> Any:
        bsz = min(bundle.encode_bs, max(1, len(texts)))
        emb = encode_passages(
            bundle.encoder,
            texts,
            bundle.prefix_q,
            bundle.norm,
            bsz,
        )
        search_with_query_embeddings(
            bundle.index, bundle.classes, emb, k
        )

    return handle, predict_fn


def _load_hybrid(
    artifacts_dir: Path, device: str, batch_size: int
) -> tuple[Any, Callable[..., Any]]:
    from src.models.hybrid import HybridModel

    dev = device
    import torch

    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    model = HybridModel.load(
        artifacts_dir, device=dev, batch_size=int(batch_size), sparse_batch_size=256
    )

    def predict_fn(texts: list[str], k: int = 10) -> Any:
        model.predict_topk(texts, k=k)

    return model, predict_fn


def _load_three_way(
    artifacts_dir: Path, device: str, batch_size: int
) -> tuple[Any, Callable[..., Any]]:
    import torch
    from src.models.three_way_hybrid import ThreeWayHybrid

    dev = device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    model = ThreeWayHybrid.load(
        artifacts_dir, device=dev, batch_size=int(batch_size), sparse_batch_size=256
    )

    def predict_fn(texts: list[str], k: int = 10) -> Any:
        model.predict_topk(texts, k=k)

    return model, predict_fn


FAMILY_LOADERS: dict[
    str,
    Callable[[Path, str, int], tuple[Any, Callable[..., Any]]],
] = {
    "sparse": _load_sparse,
    "retrieval": _load_retrieval_impl,
    "hybrid": _load_hybrid,
    "three_way_hybrid": _load_three_way,
}


def load_predict_pair(
    family: str, artifacts_dir: Path, device: str, batch_size: int
) -> tuple[Any, Callable[..., Any]]:
    return FAMILY_LOADERS[family](artifacts_dir, device, batch_size)


def try_load_with_oom_ladder(
    family: str,
    art: Path,
    device: str,
    requested_bs: int,
    gpu_fallback_ladder: list[str],
) -> tuple[Any, Callable[..., Any], int, str]:
    """Return handle, predict_fn, effective batch_size, note."""
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        h, p = load_predict_pair(family, art, device, requested_bs)
        return h, p, requested_bs, ""

    bs = int(requested_bs)
    while bs >= 8:
        try:
            if bs != requested_bs:
                gpu_fallback_ladder.append(
                    f"{family}:{device}:requested_bs={requested_bs}->effective_bs={bs}"
                )
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            h, p = load_predict_pair(family, art, "cuda", bs)
            return h, p, bs, "" if bs == requested_bs else f"oom_fallback_to_bs{bs}"
        except torch.cuda.OutOfMemoryError:
            gpu_fallback_ladder.append(f"{family}:cuda:bs={bs}:OOM")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            bs //= 2
    if 1 <= int(requested_bs) < 8:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            h, p = load_predict_pair(family, art, "cuda", int(requested_bs))
            return h, p, int(requested_bs), ""
        except torch.cuda.OutOfMemoryError:
            gpu_fallback_ladder.append(
                f"{family}:cuda:bs={int(requested_bs)}:OOM"
            )
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return None, lambda *a, **k: None, 0, "oom_skipped"


def run_warmup_only(
    predict_fn: Callable[..., Any],
    texts: list[str],
    batch_size: int,
    n_warmup: int,
    rng: np.random.Generator,
    k: int,
) -> None:
    for _ in range(n_warmup):
        batch = sample_batch(texts, batch_size, rng)
        predict_fn(batch, k=k)


def run_remaining_bench(
    predict_fn: Callable[..., Any],
    texts: list[str],
    batch_size: int,
    n_remaining: int,
    rng: np.random.Generator,
    k: int,
) -> list[float]:
    laps: list[float] = []
    for _ in range(n_remaining):
        batch = sample_batch(texts, batch_size, rng)
        t0 = time.perf_counter()
        predict_fn(batch, k=k)
        laps.append((time.perf_counter() - t0) * 1000.0)
    return laps


def bench_one_cell(
    *,
    family: str,
    run_id: str,
    artifacts_dir: Path,
    device_req: str,
    batch_size: int,
    n_bench_target: int,
    n_warmup_default: int,
    max_cell_min: float,
    texts: list[str],
    seed: int,
    model_load_cold: float,
    recall: float,
    hardware_id: str,
    gpu_fallback_ladder: list[str],
    k_pred: int = 10,
    cell_t0: float | None = None,
    max_cell_wall_t: float | None = None,
) -> dict[str, Any]:
    """Run single benchmark cell; returns CSV row dict."""
    import torch

    if device_req == "cuda" and not torch.cuda.is_available():
        return {
            "family": family,
            "run_id": run_id,
            "device": "cuda",
            "batch_size": batch_size,
            "n_warmup": 0,
            "n_bench_actual": 0,
            "n_bench_target": n_bench_target,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "throughput_texts_per_sec": 0.0,
            "model_load_seconds_cold": model_load_cold,
            "hardware_id": hardware_id,
            "recall_at_10_test": recall,
            "timestamp_utc": _utc_now(),
            "notes": "cuda_unavailable",
        }

    dev_eff = device_req
    if device_req == "cuda":
        handle, predict_fn, bs_eff, oom_note = try_load_with_oom_ladder(
            family, artifacts_dir, "cuda", batch_size, gpu_fallback_ladder
        )
        if oom_note == "oom_skipped" or handle is None:
            return {
                "family": family,
                "run_id": run_id,
                "device": "cuda",
                "batch_size": batch_size,
                "n_warmup": 0,
                "n_bench_actual": 0,
                "n_bench_target": n_bench_target,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "throughput_texts_per_sec": 0.0,
                "model_load_seconds_cold": model_load_cold,
                "hardware_id": hardware_id,
                "recall_at_10_test": recall,
                "timestamp_utc": _utc_now(),
                "notes": "oom_skipped",
            }
        notes_extra = oom_note
    else:
        handle, predict_fn = load_predict_pair(
            family, artifacts_dir, "cpu", batch_size
        )
        bs_eff = batch_size
        notes_extra = ""

    warmup_n = n_warmup_default if batch_size <= 32 else 20
    rng = np.random.default_rng(seed)

    run_warmup_only(
        predict_fn, texts, bs_eff, warmup_n, rng, k_pred
    )

    probe_laps: list[float] = []
    for _ in range(50):
        batch = sample_batch(texts, bs_eff, rng)
        t0 = time.perf_counter()
        predict_fn(batch, k=k_pred)
        probe_laps.append((time.perf_counter() - t0) * 1000.0)

    p50_probe = float(
        np.percentile(np.asarray(probe_laps, dtype=np.float64), 50)
    )

    n_actual, adapt_note = adaptive_n_bench_from_probe(
        device=device_req,
        batch_size=batch_size,
        n_bench_target=n_bench_target,
        p50_probe_ms=p50_probe,
        max_cell_min=max_cell_min,
    )
    notes_parts = [adapt_note, notes_extra]
    notes = ";".join([x for x in notes_parts if x])

    if n_actual < len(probe_laps):
        probe_laps = probe_laps[:n_actual]

    more_laps: list[float] = []
    if n_actual > len(probe_laps):
        more_laps = run_remaining_bench(
            predict_fn,
            texts,
            bs_eff,
            n_actual - len(probe_laps),
            rng,
            k_pred,
        )

    all_laps = probe_laps + more_laps
    total_ms = float(np.sum(all_laps))
    total_texts = float(len(all_laps) * bs_eff)
    throughput = (total_texts / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    p50, p95, p99 = compute_percentiles_ms(np.asarray(all_laps, dtype=np.float64))

    if (
        max_cell_wall_t is not None
        and cell_t0 is not None
        and (time.perf_counter() - cell_t0) > max_cell_wall_t
    ):
        notes = (notes + ";" if notes else "") + "cell_time_over_budget_truncated"

    del handle, predict_fn
    if device_req == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "family": family,
        "run_id": run_id,
        "device": device_req,
        "batch_size": int(bs_eff if device_req == "cuda" else batch_size),
        "n_warmup": warmup_n,
        "n_bench_actual": len(all_laps),
        "n_bench_target": n_bench_target,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput_texts_per_sec": throughput,
        "model_load_seconds_cold": model_load_cold,
        "hardware_id": hardware_id,
        "recall_at_10_test": recall,
        "timestamp_utc": _utc_now(),
        "notes": notes,
    }


def cold_warm_row_for_family(
    family: str,
    artifacts_dir: Path,
    run_id: str,
    device_req: str,
    texts: list[str],
    hw_id: str,
    gpu_fallback_ladder: list[str],
    k: int,
    seed: int,
) -> dict[str, Any]:
    """One cold/warm measurement row."""
    import torch

    if device_req == "cuda" and not torch.cuda.is_available():
        return {
            "family": family,
            "run_id": run_id,
            "device": "cuda",
            "model_load_seconds_cold": 0.0,
            "model_load_seconds_warm": 0.0,
            "first_inference_ms": 0.0,
            "post_warmup_p50_ms": 0.0,
            "hardware_id": hw_id,
            "timestamp_utc": _utc_now(),
        }

    dev = device_req
    if dev == "cuda":
        t0 = time.perf_counter()
        h, fn, bs_e, _note = try_load_with_oom_ladder(
            family, artifacts_dir, "cuda", 32, gpu_fallback_ladder
        )
        cold = time.perf_counter() - t0
        if h is None:
            return {
                "family": family,
                "run_id": run_id,
                "device": "cuda",
                "model_load_seconds_cold": 0.0,
                "model_load_seconds_warm": 0.0,
                "first_inference_ms": 0.0,
                "post_warmup_p50_ms": 0.0,
                "hardware_id": hw_id,
                "timestamp_utc": _utc_now(),
            }
        batch1 = sample_batch(texts, 1, np.random.default_rng(seed))
        t1 = time.perf_counter()
        fn(batch1, k=k)
        first_ms = (time.perf_counter() - t1) * 1000.0
        t2 = time.perf_counter()
        h2, fn2, _, _ = try_load_with_oom_ladder(
            family, artifacts_dir, "cuda", 32, gpu_fallback_ladder
        )
        warm = time.perf_counter() - t2
        rng = np.random.default_rng(seed + 1)
        for _ in range(20):
            fn2(sample_batch(texts, min(32, bs_e), rng), k=k)
        laps = []
        for _ in range(50):
            tb = time.perf_counter()
            fn2(sample_batch(texts, min(32, bs_e), rng), k=k)
            laps.append((time.perf_counter() - tb) * 1000.0)
        post50 = float(np.percentile(np.asarray(laps, dtype=np.float64), 50))
        del h, h2
        torch.cuda.empty_cache()
        return {
            "family": family,
            "run_id": run_id,
            "device": "cuda",
            "model_load_seconds_cold": cold,
            "model_load_seconds_warm": warm,
            "first_inference_ms": first_ms,
            "post_warmup_p50_ms": post50,
            "hardware_id": hw_id,
            "timestamp_utc": _utc_now(),
        }

    t0 = time.perf_counter()
    h, fn = load_predict_pair(family, artifacts_dir, "cpu", 32)
    cold = time.perf_counter() - t0
    batch1 = sample_batch(texts, 1, np.random.default_rng(seed))
    t1 = time.perf_counter()
    fn(batch1, k=k)
    first_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    h2, fn2 = load_predict_pair(family, artifacts_dir, "cpu", 32)
    warm = time.perf_counter() - t2
    rng = np.random.default_rng(seed + 1)
    b32 = min(32, max(1, len(texts)))
    for _ in range(20):
        fn2(sample_batch(texts, b32, rng), k=k)
    laps = []
    for _ in range(50):
        tb = time.perf_counter()
        fn2(sample_batch(texts, b32, rng), k=k)
        laps.append((time.perf_counter() - tb) * 1000.0)
    post50 = float(np.percentile(np.asarray(laps, dtype=np.float64), 50))
    return {
        "family": family,
        "run_id": run_id,
        "device": "cpu",
        "model_load_seconds_cold": cold,
        "model_load_seconds_warm": warm,
        "first_inference_ms": first_ms,
        "post_warmup_p50_ms": post50,
        "hardware_id": hw_id,
        "timestamp_utc": _utc_now(),
    }


def write_csv_append(
    path: Path, row: dict[str, Any], columns: list[str], write_header: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in columns})


def load_existing_latency_keys(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, encoding="utf-8")
    keys: set[tuple[str, str, str, int]] = set()
    for _, r in df.iterrows():
        keys.add(
            (
                str(r["family"]),
                str(r["run_id"]),
                str(r["device"]),
                int(r["batch_size"]),
            )
        )
    return keys


def generate_latency_markdown(
    repo: Path,
    latency_csv: Path,
    cold_csv: Path,
    out_md: Path,
    figure_rel: str,
    *,
    wall_s: float,
    hardware_expanded: str,
    hardware_id: str,
    gpu_fallback_ladder: list[str],
) -> None:
    df = pd.read_csv(latency_csv, encoding="utf-8")
    strict_cuda = df[
        (df["device"] == "cuda")
        & (df["n_bench_actual"] >= 1000)
        & (df["notes"].fillna("") != "cuda_unavailable")
        & (df["notes"].fillna("") != "oom_skipped")
    ]
    strict_cpu32 = df[
        (df["device"] == "cpu")
        & (df["batch_size"] <= 32)
        & (df["n_bench_actual"] >= 1000)
    ]
    relaxed = df[
        (df["device"] == "cpu")
        & (df["batch_size"].isin([64, 128]))
        & (df["n_bench_actual"] < 1000)
    ]
    n_non_compliant = len(
        df[
            (df["n_bench_actual"] < df["n_bench_target"])
            & (df["n_bench_actual"] > 0)
        ]
    )

    sub = df[(df["batch_size"] == 32) & (df["device"] == "cuda")].copy()
    pareto_lines = []
    for fam in CANONICAL_RUNS:
        r = sub[sub["family"] == fam]
        if len(r) == 0:
            continue
        best = r.loc[r["p50_ms"].idxmin()]
        pareto_lines.append(
            f"| {fam} | {best['p50_ms']:.2f} | {best['p95_ms']:.2f} | "
            f"{best['throughput_texts_per_sec']:.2f} | {best['recall_at_10_test']:.5f} |"
        )

    try:
        rel_lat = latency_csv.relative_to(repo)
    except ValueError:
        rel_lat = latency_csv
    try:
        rel_cold = cold_csv.relative_to(repo)
    except ValueError:
        rel_cold = cold_csv

    lines = [
        "# Latency benchmark summary (TASK-034)",
        "",
        "## Hardware fingerprint",
        "",
        f"- **Expanded:** `{hardware_expanded}`",
        f"- **hardware_id:** `{hardware_id}`",
        "",
        f"- **Source CSV:** `{rel_lat}`",
        f"- **Cold vs warm:** `{rel_cold}`",
        "",
        f"- **Total wall time (grid):** {wall_s / 60:.2f} min",
        f"- **Rows in latency_full.csv:** {len(df)}",
        f"- **Strict CUDA cells (n>=1000):** {len(strict_cuda)}",
        f"- **Strict CPU bs<=32 (n>=1000):** {len(strict_cpu32)}",
        f"- **Relaxed CPU large-batch rows (n<target):** {len(relaxed)}",
        f"- **Cells with n_bench_actual < n_bench_target:** {n_non_compliant}",
        "",
        "## n_bench compliance (DECISION-054)",
        "",
        "CUDA: target 1000+ per cell where CUDA ran. "
        "CPU batch<=32: 1000+ where feasible within max-cell-min. "
        "CPU batch 64/128: minimum 200 with optional reduction and notes.",
        "",
        "## Pareto (batch=32, CUDA): quality vs latency (p50)",
        "",
        "| family | p50_ms | p95_ms | throughput | R@10 test |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(pareto_lines)
    lines.extend(
        [
            "",
            f"![Pareto]({figure_rel})",
            "",
            "## GPU OOM fallback ladder",
            "",
        ]
    )
    if gpu_fallback_ladder:
        for g in gpu_fallback_ladder:
            lines.append(f"- `{g}`")
    else:
        lines.append("- (none)")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def write_pareto_figure(latency_csv: Path, out_png: Path, dpi: int = 120) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(latency_csv, encoding="utf-8")
    sub = df[(df["batch_size"] == 32) & (df["device"] == "cuda")]
    sub = sub[sub["notes"].fillna("") != "cuda_unavailable"]
    sub = sub[sub["notes"].fillna("") != "oom_skipped"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for fam, grp in sub.groupby("family"):
        r = grp.loc[grp["p50_ms"].idxmin()]
        ax.scatter(
            r["p50_ms"],
            r["recall_at_10_test"],
            s=80,
            label=str(fam),
        )
        ax.annotate(str(fam), (r["p50_ms"], r["recall_at_10_test"]), fontsize=8)
    ax.set_xlabel("p50 latency (ms)")
    ax.set_ylabel("R@10 (test)")
    ax.set_title("R@10 vs p50 (batch=32, CUDA)")
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def iter_cell_order(
    families: list[str],
    devices: list[str],
    batch_sizes: list[int],
) -> list[tuple[str, str, int]]:
    """CUDA cells first (rule 95), then CPU."""
    order: list[tuple[str, str, int]] = []
    for dev in ("cuda", "cpu"):
        if dev not in devices:
            continue
        for fam in families:
            for bs in batch_sizes:
                order.append((fam, dev, bs))
    return order


def _apply_sentence_transformer_local_hf() -> None:
    """Use cached HF snapshot dirs when available (avoids Hub API during benchmark)."""
    import sentence_transformers as st

    if getattr(st.SentenceTransformer, "_orllm_local_hf", False):
        return
    _orig = st.SentenceTransformer

    def _wrapper(model_name_or_path, *args, **kwargs):
        m = str(model_name_or_path)
        p = Path(m)
        if p.is_dir() and (p / "config.json").is_file():
            resolved = m
        else:
            resolved = local_hf_snapshot_or_id(m)
        inst = _orig(resolved, *args, **kwargs)
        return inst

    setattr(_wrapper, "_orllm_local_hf", True)
    st.SentenceTransformer = _wrapper


def main() -> int:
    _apply_sentence_transformer_local_hf()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Latency & throughput benchmark grid (TASK-034).",
    )
    ap.add_argument(
        "--families",
        nargs="+",
        default=list(CANONICAL_RUNS.keys()),
        help="Model families to benchmark",
    )
    ap.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "cuda"],
        help="Devices (CUDA cells run first internally)",
    )
    ap.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 8, 32, 64, 128],
        help="Batch sizes",
    )
    ap.add_argument("--n-bench", type=int, default=1000)
    ap.add_argument("--n-warmup", type=int, default=100)
    ap.add_argument("--max-cell-min", type=float, default=30.0)
    ap.add_argument("--total-budget-h", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "benchmarks",
    )
    ap.add_argument(
        "--figures-out",
        type=Path,
        default=ROOT / "reports" / "figures",
    )
    ap.add_argument(
        "--topics",
        type=Path,
        default=ROOT / "data" / "processed" / "topics.csv",
    )
    ap.add_argument(
        "--texts-pool",
        type=Path,
        default=ROOT / "data" / "splits" / "train.parquet",
    )
    ap.add_argument("--texts-pool-rows", type=int, default=5000)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    latency_csv = out_dir / "latency_full.csv"
    cold_csv = out_dir / "cold_vs_warm.csv"
    md_path = out_dir / "latency_full.md"
    fig_path = Path(args.figures_out).resolve() / "latency_pareto.png"

    hw_exp = hardware_fingerprint_string()
    hw_id = hardware_id_from_string(hw_exp)

    texts = load_text_pool(
        ROOT, Path(args.texts_pool), int(args.texts_pool_rows), int(args.seed)
    )
    gpu_fallback_ladder: list[str] = []

    recalls = {
        fam: read_recall_at_10(ROOT, CANONICAL_RUNS[fam][0])
        for fam in args.families
        if fam in CANONICAL_RUNS
    }

    grid_t0 = time.perf_counter()
    budget_s = float(args.total_budget_h) * 3600.0

    existing: set[tuple[str, str, str, int]] = set()
    if args.skip_existing and latency_csv.is_file():
        existing = load_existing_latency_keys(latency_csv)

    cold_written = cold_csv.is_file()
    cold_keys: set[tuple[str, str]] = set()
    if cold_csv.is_file():
        cdf = pd.read_csv(cold_csv, encoding="utf-8")
        for _, r in cdf.iterrows():
            cold_keys.add((str(r["family"]), str(r["device"])))

    # Phase: cold vs warm (append if missing)
    for fam in args.families:
        if fam not in CANONICAL_RUNS:
            logger.warning("Unknown family %s — skip", fam)
            continue
        run_id, rel_art = CANONICAL_RUNS[fam]
        art = (ROOT / rel_art).resolve()
        if not art.is_dir():
            logger.error("Missing artifacts: %s", art)
            continue
        for dev in args.devices:
            if (fam, dev) in cold_keys:
                continue
            if time.perf_counter() - grid_t0 > budget_s:
                logger.error("Total budget exceeded before cold/warm — abort")
                return 2
            row_cw = cold_warm_row_for_family(
                fam,
                art,
                run_id,
                dev,
                texts,
                hw_id,
                gpu_fallback_ladder,
                k=10,
                seed=args.seed + hash((fam, dev)) % 10_000,
            )
            write_csv_append(
                cold_csv,
                row_cw,
                COLD_WARM_COLUMNS,
                write_header=not cold_written,
            )
            cold_written = True

    # Cold load map for CSV column (use cold_vs_warm if present)
    cold_load_map: dict[tuple[str, str], float] = {}
    if cold_csv.is_file():
        cdf = pd.read_csv(cold_csv, encoding="utf-8")
        for _, r in cdf.iterrows():
            cold_load_map[(str(r["family"]), str(r["device"]))] = float(
                r["model_load_seconds_cold"]
            )

    cells = iter_cell_order(
        list(args.families),
        list(args.devices),
        list(args.batch_sizes),
    )

    file_has_header = latency_csv.is_file()

    for fam, dev, bs in cells:
        if time.perf_counter() - grid_t0 > budget_s:
            logger.error("Total wall budget exceeded — stop grid")
            break
        if fam not in CANONICAL_RUNS:
            continue
        run_id, rel_art = CANONICAL_RUNS[fam]
        key = (fam, run_id, dev, bs)
        if args.skip_existing and key in existing:
            logger.info("skip existing %s", key)
            continue
        art = (ROOT / rel_art).resolve()
        if not art.is_dir():
            logger.error("skip missing dir %s", art)
            continue

        ml_cold = cold_load_map.get((fam, dev), 0.0)
        cell_t0 = time.perf_counter()
        row = bench_one_cell(
            family=fam,
            run_id=run_id,
            artifacts_dir=art,
            device_req=dev,
            batch_size=bs,
            n_bench_target=int(args.n_bench),
            n_warmup_default=int(args.n_warmup),
            max_cell_min=float(args.max_cell_min),
            texts=texts,
            seed=args.seed + bs + hash(fam) % 10000,
            model_load_cold=ml_cold,
            recall=recalls.get(fam, float("nan")),
            hardware_id=hw_id,
            gpu_fallback_ladder=gpu_fallback_ladder,
            cell_t0=cell_t0,
            max_cell_wall_t=float(args.max_cell_min) * 60.0,
        )
        write_csv_append(
            latency_csv,
            row,
            LATENCY_CSV_COLUMNS,
            write_header=not file_has_header,
        )
        file_has_header = True
        logger.info(
            "cell done %s %s bs=%s n=%s p50=%.2fms",
            fam,
            dev,
            row["batch_size"],
            row["n_bench_actual"],
            row["p50_ms"],
        )
        if (
            time.perf_counter() - cell_t0 > float(args.max_cell_min) * 60.0
            and row["n_bench_actual"] < row["n_bench_target"]
        ):
            logger.warning("cell exceeded max-cell-min; results may be partial")

    wall_s = time.perf_counter() - grid_t0
    generate_latency_markdown(
        ROOT,
        latency_csv,
        cold_csv,
        md_path,
        figure_rel="../figures/latency_pareto.png",
        wall_s=wall_s,
        hardware_expanded=hw_exp,
        hardware_id=hw_id,
        gpu_fallback_ladder=gpu_fallback_ladder,
    )
    try:
        write_pareto_figure(latency_csv, fig_path)
    except Exception as exc:
        logger.warning("pareto figure skipped: %s", exc)

    logger.info("wrote %s", latency_csv)
    logger.info("wrote %s", md_path)
    logger.info("wall_s=%.1f", wall_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
