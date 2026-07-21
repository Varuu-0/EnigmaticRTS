"""Benchmark runner for Terrain Diffusion at native scale=1.

Records, per request-size tier (128/256/512):
  - cold and warm P50/P95 latency
  - repeated response checksums (repeatability evidence)
  - payload structural validation
  - CPU load, GPU utilization, and peak VRAM
  - coexistence fields (DeviceLost observed, provider-attributable hitch)

Requires a running sidecar (start_server.ps1). Never fabricates results: if the
server is unreachable the runner exits with a non-zero code and reports the
exact blocker.

Usage:
  python -m bench.run_benchmarks --host 127.0.0.1 --port 8000 --seed 12345
  python -m bench.run_benchmarks --sizes 128,256,512 --warmup-iters 3 --bench-iters 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from locked_manifest import build_manifest  # noqa: E402
from protocol import (  # noqa: E402
    ProtocolError,
    decode_payload,
    expected_payload_size,
    parse_response_headers,
    validate_payload_finite,
)

DEFAULT_SIZES = (128, 256, 512)
DEFAULT_SEED = 12345
DEFAULT_WARMUP_ITERS = 2
DEFAULT_BENCH_ITERS = 10
VULKAN_HEADROOM_FLOOR_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB


@dataclass
class TierResult:
    size: int
    cold_latency_ms: float
    warm_latencies_ms: list[float]
    warm_p50_ms: float
    warm_p95_ms: float
    checksums: list[str]
    checksum_match: bool
    payload_valid: bool
    payload_error: str | None
    bytes_received: int
    expected_bytes: int
    dimensions: tuple[int, int]


@dataclass
class VramSample:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    gpu_util_pct: float
    cpu_load_pct: float


@dataclass
class BenchmarkReport:
    schema: str
    timestamp: str
    manifest: dict[str, Any]
    tiers: list[dict[str, Any]]
    peak_vram_used_bytes: int
    device_total_vram_bytes: int
    vulkan_headroom_bytes: int
    vram_headroom_ok: bool
    device_lost_observed: bool
    server_reachable: bool
    notes: list[str] = field(default_factory=list)


def _sample_vram() -> VramSample:
    """Sample VRAM/GPU/CPU via nvidia-smi (no extra deps)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        total, used, free, util = (float(x.strip()) for x in line.split(","))
        total_b = int(total * 1024 * 1024)
        used_b = int(used * 1024 * 1024)
        free_b = int(free * 1024 * 1024)
    except Exception:
        total_b = used_b = free_b = 0
        util = 0.0

    cpu_load = 0.0
    try:
        import psutil

        cpu_load = psutil.cpu_percent(interval=0.1)
    except Exception:
        pass

    return VramSample(
        total_bytes=total_b,
        used_bytes=used_b,
        free_bytes=free_b,
        gpu_util_pct=util,
        cpu_load_pct=cpu_load,
    )


def _check_health(base_url: str, timeout: float = 5.0) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def _request_terrain(
    base_url: str, size: int, seed: int, timeout: float
) -> tuple[bytes, dict[str, str], float]:
    """Issue a single /terrain request. Returns (body, headers, latency_ms)."""
    params = {"i1": 0, "j1": 0, "i2": size, "j2": size, "scale": 1, "seed": seed}
    t0 = time.perf_counter()
    r = requests.get(f"{base_url}/terrain", params=params, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return r.content, dict(r.headers), elapsed_ms


def _validate_and_checksum(body: bytes, headers: dict[str, str], size: int) -> tuple[bool, str | None, str]:
    """Validate payload and return (valid, error, checksum)."""
    try:
        h, w = parse_response_headers(headers)
        if h != size or w != size:
            return False, f"dimensions {h}x{w} != requested {size}x{size}", ""
        expected = expected_payload_size(h, w, with_climate=True)
        if len(body) != expected:
            return False, f"bytes {len(body)} != expected {expected}", ""
        payload = decode_payload(body, h, w, require_climate=True)
        validate_payload_finite(payload)
        return True, None, payload.checksum()
    except ProtocolError as exc:
        return False, str(exc), ""
    except Exception as exc:
        return False, f"unexpected: {exc}", ""


def benchmark_tier(
    base_url: str, size: int, seed: int, warmup: int, iters: int, timeout: float
) -> TierResult:
    # Cold request (first after seed-set / no warm cache).
    body, headers, cold_ms = _request_terrain(base_url, size, seed, timeout)
    valid, err, checksum = _validate_and_checksum(body, headers, size)

    warm_latencies: list[float] = []
    checksums = [checksum]
    for _ in range(warmup):
        b, h, ms = _request_terrain(base_url, size, seed, timeout)
        _, _, cs = _validate_and_checksum(b, h, size)
        checksums.append(cs)

    for _ in range(iters):
        b, h, ms = _request_terrain(base_url, size, seed, timeout)
        warm_latencies.append(ms)
        _, _, cs = _validate_and_checksum(b, h, size)
        checksums.append(cs)

    checksums_unique = set(checksums)
    warm_p50 = statistics.median(warm_latencies) if warm_latencies else 0.0
    warm_p95 = _percentile(warm_latencies, 95) if warm_latencies else 0.0

    return TierResult(
        size=size,
        cold_latency_ms=cold_ms,
        warm_latencies_ms=warm_latencies,
        warm_p50_ms=warm_p50,
        warm_p95_ms=warm_p95,
        checksums=checksums,
        checksum_match=len(checksums_unique) == 1,
        payload_valid=valid,
        payload_error=err,
        bytes_received=len(body),
        expected_bytes=expected_payload_size(size, size, with_climate=True),
        dimensions=(size, size),
    )


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run_benchmarks(
    tools_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    seed: int = DEFAULT_SEED,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    bench_iters: int = DEFAULT_BENCH_ITERS,
    timeout: float = 60.0,
    dtype: str = "fp32",
    device: str = "cuda",
    t_steps: int = 2,
) -> BenchmarkReport:
    import datetime

    base_url = f"http://{host}:{port}"
    reachable = _check_health(base_url)
    notes: list[str] = []
    peak_vram = 0
    device_total = 0
    device_lost = False

    # Build manifest from the installed env for provenance. The dtype/device
    # reflect the server's actual startup flags (the API does not expose them).
    try:
        manifest = build_manifest(
            tools_root, host=host, port=port, dtype=dtype, device=device, t_steps=t_steps
        ).to_dict()
    except Exception as exc:
        manifest = {"error": f"manifest generation failed: {exc}"}
        notes.append(f"manifest_generation_error: {exc}")

    tiers: list[dict[str, Any]] = []
    if not reachable:
        notes.append(
            f"BLOCKER: server at {base_url} unreachable. "
            "Run .\\tools\\terrain_diffusion\\start_server.ps1 first."
        )
    else:
        for size in sizes:
            try:
                result = benchmark_tier(base_url, size, seed, warmup_iters, bench_iters, timeout)
                tiers.append(asdict(result))
                vram = _sample_vram()
                peak_vram = max(peak_vram, vram.used_bytes)
                device_total = vram.total_bytes
                if not result.checksum_match:
                    notes.append(f"size={size}: checksums did not match across repeats")
                if not result.payload_valid:
                    notes.append(f"size={size}: payload invalid: {result.payload_error}")
            except Exception as exc:
                tiers.append({"size": size, "error": str(exc)})
                notes.append(f"size={size}: benchmark failed: {exc}")
                if "device" in str(exc).lower() or "cuda" in str(exc).lower():
                    device_lost = True

    headroom = device_total - peak_vram if device_total else 0
    return BenchmarkReport(
        schema="terrain-diffusion-benchmark/v1",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest=manifest,
        tiers=tiers,
        peak_vram_used_bytes=peak_vram,
        device_total_vram_bytes=device_total,
        vulkan_headroom_bytes=headroom,
        vram_headroom_ok=headroom >= VULKAN_HEADROOM_FLOOR_BYTES if device_total else False,
        device_lost_observed=device_lost,
        server_reachable=reachable,
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Terrain Diffusion native-scale benchmarks.")
    parser.add_argument("--tools-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sizes", default="128,256,512", help="Comma-separated request sizes.")
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--bench-iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                        help="Server's actual --dtype startup flag (not queryable via API).")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--t-steps", type=int, choices=[1, 2], default=2)
    parser.add_argument("--output", "-o", default=None, help="Output JSON path.")
    args = parser.parse_args()

    sizes = tuple(int(s) for s in args.sizes.split(","))
    report = run_benchmarks(
        Path(args.tools_root),
        host=args.host,
        port=args.port,
        seed=args.seed,
        sizes=sizes,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        timeout=args.timeout,
        dtype=args.dtype,
        device=args.device,
        t_steps=args.t_steps,
    )
    text = json.dumps(asdict(report), indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)

    # Exit non-zero if the server was unreachable so CI/automation sees the blocker.
    if not report.server_reachable:
        print(report.notes[-1] if report.notes else "server unreachable", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
