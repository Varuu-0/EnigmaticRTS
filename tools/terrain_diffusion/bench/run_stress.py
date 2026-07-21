"""Bounded coexistence/stress runner for Terrain Diffusion + the Bevy game.

This runner provides an **honest** coexistence path: it can optionally launch
the feature-enabled game in stress mode (``--terrain-diffusion-stress``) while
issuing sustained sidecar requests. The game's own main-thread provider timing
report is the sole source for the ``<= 1 ms P95`` hitch gate — sidecar
inter-request jitter is recorded for diagnostics but is **not** the gate.

Exit gates (all must pass for a green exit):
  - at least 1 GiB Vulkan VRAM headroom remains
  - no DeviceLost (detected from game process output or sidecar errors)
  - provider-attributable main-thread hitch <= 1 ms P95 (from the game report)

If the game stress mode is not requested, the hitch gate is left ``unproven``
and the runner exits non-zero with the exact blocker documented.

Usage:
  # Sidecar-only stress (hitch gate stays unproven):
  python -m bench.run_stress --duration-minutes 30 --host 127.0.0.1 --port 8000

  # Full coexistence with game (hitch gate from game report):
  python -m bench.run_stress --duration-minutes 5 --launch-game \\
      --game-binary target\\debug\\er_game.exe \\
      --game-args "--earth-scale --terrain-diffusion --terrain-diffusion-stress 270"
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
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

VULKAN_HEADROOM_FLOOR_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
HITCH_P95_CEILING_MS = 1.0
DEFAULT_DURATION_MINUTES = 30
DEFAULT_REQUEST_INTERVAL_S = 1.0
DEFAULT_SIZE = 256
DEFAULT_SEED = 12345
# Poll VRAM frequently so transient peaks are not missed.
DEFAULT_VRAM_POLL_INTERVAL_S = 0.5

_DEVICE_LOST_PATTERNS = re.compile(
    r"DeviceLost|device lost|Validation.*device-lost|VK_ERROR_DEVICE_LOST",
    re.IGNORECASE,
)


@dataclass
class StressSample:
    elapsed_s: float
    latency_ms: float
    vram_used_bytes: int
    vram_free_bytes: int
    gpu_util_pct: float
    cpu_load_pct: float
    checksum: str
    payload_valid: bool
    error: str | None


@dataclass
class GameReport:
    schema: str | None = None
    duration_seconds: int | None = None
    frames_recorded: int | None = None
    provider_p50_ms: float | None = None
    provider_p95_ms: float | None = None
    provider_p99_ms: float | None = None
    provider_max_ms: float | None = None
    provider_mean_ms: float | None = None
    exit_gate_hitch_ok: bool | None = None
    exit_gate_passed: bool | None = None
    raw: dict[str, Any] | None = None


@dataclass
class StressReport:
    schema: str
    started_at: str
    ended_at: str
    duration_minutes: float
    manifest: dict[str, Any]
    total_requests: int
    successful_requests: int
    failed_requests: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    hitch_p95_ms: float
    hitch_source: str
    sidecar_jitter_p95_ms: float
    peak_vram_used_bytes: int
    min_vulkan_headroom_bytes: int
    device_total_vram_bytes: int
    game_exit_code: int | None
    game_report: dict[str, Any] | None
    server_reachable: bool
    exit_gate_protocol_ok: bool
    exit_gate_headroom_ok: bool
    exit_gate_no_device_lost: bool
    exit_gate_hitch_ok: bool
    exit_gate_passed: bool
    checksums_unique: int
    checksum_match: bool
    samples: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _sample_vram() -> tuple[int, int, int, float, float]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        total, used, free, util = (float(x.strip()) for x in line.split(","))
        total_b = int(total * 1024 * 1024)
        used_b = int(used * 1024 * 1024)
        free_b = int(free * 1024 * 1024)
    except Exception:
        total_b = used_b = free_b = 0
        util = 0.0
    cpu = 0.0
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
    except Exception:
        pass
    return total_b, used_b, free_b, util, cpu


def _request(base_url: str, size: int, seed: int, timeout: float) -> tuple[bytes, dict[str, str], float]:
    params = {"i1": 0, "j1": 0, "i2": size, "j2": size, "scale": 1, "seed": seed}
    t0 = time.perf_counter()
    r = requests.get(f"{base_url}/terrain", params=params, timeout=timeout)
    elapsed = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return r.content, dict(r.headers), elapsed


def _request(base_url: str, size: int, seed: int, timeout: float) -> tuple[bytes, dict[str, str], float]:
    params = {"i1": 0, "j1": 0, "i2": size, "j2": size, "scale": 1, "seed": seed}
    t0 = time.perf_counter()
    r = requests.get(f"{base_url}/terrain", params=params, timeout=timeout)
    elapsed = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return r.content, dict(r.headers), elapsed


def _validate(body: bytes, headers: dict[str, str], size: int) -> tuple[bool, str | None, str]:
    try:
        h, w = parse_response_headers(headers)
        if h != size or w != size:
            return False, f"dim mismatch {h}x{w}", ""
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


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _check_health(base_url: str, timeout: float = 5.0) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def _scan_for_device_lost(text: str) -> bool:
    return bool(_DEVICE_LOST_PATTERNS.search(text))


class _GameMonitor:
    """Launches the game in stress mode and monitors its output."""

    def __init__(self, binary: str, args: str, output_path: Path | None):
        self.binary = binary
        self.args = args
        self.output_path = output_path
        self.process: subprocess.Popen | None = None
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        cmd = [self.binary] + self.args.split()
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.stdout_lines.append(line)

    def scan_output_for_device_lost(self) -> bool:
        return _scan_for_device_lost("".join(self.stdout_lines))

    def wait(self, timeout: float | None = None) -> int | None:
        if self.process is None:
            return None
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            return None

    @property
    def output_text(self) -> str:
        return "".join(self.stdout_lines)

    def read_game_report(self) -> GameReport | None:
        if self.output_path and self.output_path.exists():
            try:
                raw = json.loads(self.output_path.read_text())
                return GameReport(
                    schema=raw.get("schema"),
                    duration_seconds=raw.get("duration_seconds"),
                    frames_recorded=raw.get("frames_recorded"),
                    provider_p50_ms=raw.get("provider_p50_ms"),
                    provider_p95_ms=raw.get("provider_p95_ms"),
                    provider_p99_ms=raw.get("provider_p99_ms"),
                    provider_max_ms=raw.get("provider_max_ms"),
                    provider_mean_ms=raw.get("provider_mean_ms"),
                    exit_gate_hitch_ok=raw.get("exit_gate_hitch_ok"),
                    exit_gate_passed=raw.get("exit_gate_passed"),
                    raw=raw,
                )
            except Exception:
                return None
        return None


def run_stress(
    tools_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_SIZE,
    duration_minutes: float = DEFAULT_DURATION_MINUTES,
    request_interval_s: float = DEFAULT_REQUEST_INTERVAL_S,
    timeout: float = 60.0,
    launch_game: bool = False,
    game_binary: str | None = None,
    game_args: str | None = None,
    game_report_path: Path | None = None,
    dtype: str = "fp32",
) -> StressReport:
    base_url = f"http://{host}:{port}"
    started = datetime.datetime.now(datetime.timezone.utc)
    duration_s = duration_minutes * 60.0

    try:
        manifest = build_manifest(
            tools_root, host=host, port=port, dtype=dtype, t_steps=2
        ).to_dict()
    except Exception as exc:
        manifest = {"error": str(exc)}

    samples: list[StressSample] = []
    errors: list[str] = []
    notes: list[str] = []
    checksums: set[str] = set()
    peak_vram = 0
    min_headroom = float("inf")
    device_total = 0
    device_lost = False
    total = 0
    successful = 0
    failed = 0
    game_monitor: _GameMonitor | None = None
    game_report: GameReport | None = None
    game_exit_code: int | None = None

    reachable = _check_health(base_url)
    if not reachable:
        errors.append(
            f"BLOCKER: server at {base_url} unreachable. "
            "Run .\\tools\\terrain_diffusion\\start_server.ps1 first."
        )

    # Never launch a long game run when the required provider is unavailable.
    if launch_game and reachable:
        if not game_binary or not Path(game_binary).exists():
            errors.append(
                f"BLOCKER: game binary not found at {game_binary!r}. "
                "Build with: cargo build -p er_game --features terrain_diffusion"
            )
        else:
            game_monitor = _GameMonitor(game_binary, game_args or "", game_report_path)
            try:
                game_monitor.start()
                notes.append(f"Launched game stress mode: {game_binary} {game_args}")
            except Exception as exc:
                errors.append(f"Failed to launch game: {exc}")
                game_monitor = None

    t_start = time.perf_counter()
    last_vram_poll = 0.0
    # Last instantaneous VRAM/util sample; carried forward so every request
    # sample carries a real measurement even when the poll cadence is slower
    # than the request cadence.
    inst_total = 0
    inst_used = 0
    inst_free = 0
    inst_util = 0.0
    inst_cpu = 0.0

    while reachable and (time.perf_counter() - t_start) < duration_s:
        now = time.perf_counter()
        # Poll VRAM on every iteration when enough time has elapsed, so we
        # don't miss transient peaks between requests.
        if now - last_vram_poll >= DEFAULT_VRAM_POLL_INTERVAL_S:
            inst_total, inst_used, inst_free, inst_util, inst_cpu = _sample_vram()
            peak_vram = max(peak_vram, inst_used)
            device_total = max(device_total, inst_total)
            if inst_total:
                min_headroom = min(min_headroom, inst_total - inst_used)
            last_vram_poll = now

        total += 1
        lat = 0.0
        cs = ""
        valid = False
        err = None
        try:
            body, headers, lat = _request(base_url, size, seed, timeout)
            valid, err, cs = _validate(body, headers, size)
            if cs:
                checksums.add(cs)
            if valid:
                successful += 1
            else:
                failed += 1
                errors.append(f"req {total}: invalid payload: {err}")
        except Exception as exc:
            failed += 1
            err = str(exc)
            errors.append(f"req {total}: {err}")
            if "device" in err.lower() or "cuda" in err.lower() or "lost" in err.lower():
                device_lost = True

        samples.append(StressSample(
            elapsed_s=time.perf_counter() - t_start,
            latency_ms=lat,
            vram_used_bytes=inst_used,
            vram_free_bytes=inst_free,
            gpu_util_pct=inst_util,
            cpu_load_pct=inst_cpu,
            checksum=cs,
            payload_valid=valid,
            error=err,
        ))

        if request_interval_s > 0:
            time.sleep(request_interval_s)

    # Final VRAM sample.
    inst_total, inst_used, inst_free, inst_util, inst_cpu = _sample_vram()
    peak_vram = max(peak_vram, inst_used)
    device_total = max(device_total, inst_total)
    if inst_total:
        min_headroom = min(min_headroom, inst_total - inst_used)

    # Wait for the game to finish (it exits after its own duration).
    if game_monitor is not None:
        remaining = max(0.0, duration_s - (time.perf_counter() - t_start) + 30.0)
        game_exit_code = game_monitor.wait(timeout=remaining)
        if game_monitor.scan_output_for_device_lost():
            device_lost = True
            notes.append("DeviceLost detected in game process output.")
        game_report = game_monitor.read_game_report()
        if game_report is None:
            notes.append(
                "Game report not found — the game may have crashed before writing it. "
                "Hitch gate cannot be proven."
            )

    ended = datetime.datetime.now(datetime.timezone.utc)
    min_headroom_val = int(min_headroom) if min_headroom != float("inf") else 0
    headroom_ok = min_headroom_val >= VULKAN_HEADROOM_FLOOR_BYTES if device_total else False

    latencies = [s.latency_ms for s in samples if s.latency_ms > 0]
    sidecar_jitter_p95 = _percentile(
        [abs(latencies[i] - latencies[i - 1]) for i in range(1, len(latencies))]
        if len(latencies) > 1 else [],
        95,
    )

    # The hitch gate is ONLY satisfied from the game's main-thread report.
    # Sidecar inter-request jitter is diagnostic, not the gate.
    hitch_ok = False
    hitch_source = "sidecar_jitter_diagnostic_only"
    # ``hitch_p95_ms`` reports the value the gate is evaluated against: the
    # game's main-thread provider P95 when a game report exists, otherwise 0.0
    # (the gate is unproven and fails closed).
    hitch_p95_ms = 0.0
    if game_report is not None and game_report.provider_p95_ms is not None:
        hitch_source = "game_main_thread"
        hitch_p95_ms = float(game_report.provider_p95_ms)
        hitch_ok = game_report.provider_p95_ms <= HITCH_P95_CEILING_MS
        if game_report.exit_gate_hitch_ok is not None:
            hitch_ok = bool(game_report.exit_gate_hitch_ok)
    elif launch_game and game_report is None:
        hitch_source = "unproven_game_report_missing"

    protocol_ok = (
        reachable
        and total >= 2
        and successful == total
        and failed == 0
        and len(checksums) == 1
    )
    game_ok = not launch_game or game_exit_code == 0

    report = StressReport(
        schema="terrain-diffusion-stress/v2",
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
        duration_minutes=duration_minutes,
        manifest=manifest,
        total_requests=total,
        successful_requests=successful,
        failed_requests=failed,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        hitch_p95_ms=hitch_p95_ms,
        hitch_source=hitch_source,
        sidecar_jitter_p95_ms=sidecar_jitter_p95,
        peak_vram_used_bytes=peak_vram,
        min_vulkan_headroom_bytes=min_headroom_val,
        device_total_vram_bytes=device_total,
        game_exit_code=game_exit_code,
        game_report=asdict(game_report) if game_report else None,
        server_reachable=reachable,
        exit_gate_protocol_ok=protocol_ok,
        exit_gate_headroom_ok=headroom_ok,
        exit_gate_no_device_lost=not device_lost and game_ok,
        exit_gate_hitch_ok=hitch_ok,
        exit_gate_passed=protocol_ok and headroom_ok and not device_lost and game_ok and hitch_ok,
        checksums_unique=len(checksums),
        checksum_match=len(checksums) == 1,
        samples=[asdict(s) for s in samples[:1000]],
        errors=errors[:200],
        notes=notes + [
            "hitch_p95_ms is the game main-thread provider P95 (the gate value) when a game report is present, otherwise 0.0 (gate unproven).",
            "sidecar_jitter_p95_ms is sidecar inter-request jitter (diagnostic only, never the gate).",
            "The hitch gate is satisfied ONLY from game_report.provider_p95_ms (main-thread).",
            f"headroom floor = {VULKAN_HEADROOM_FLOOR_BYTES // (1024*1024)} MiB",
            f"hitch ceiling = {HITCH_P95_CEILING_MS} ms P95",
        ],
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded Terrain Diffusion coexistence stress test.")
    parser.add_argument("--tools-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--duration-minutes", type=float, default=DEFAULT_DURATION_MINUTES)
    parser.add_argument("--request-interval-s", type=float, default=DEFAULT_REQUEST_INTERVAL_S)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--launch-game", action="store_true",
                        help="Launch the feature-enabled game in stress mode alongside the sidecar.")
    parser.add_argument("--game-binary", default=None,
                        help="Path to the er_game binary (must be built with --features terrain_diffusion).")
    parser.add_argument("--game-args", default=None,
                        help="Args for the game (e.g. '--earth-scale --terrain-diffusion --terrain-diffusion-stress 270').")
    parser.add_argument("--game-report-path", default=None,
                        help="Path where the game will write its stress JSON report.")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    game_report_path = Path(args.game_report_path) if args.game_report_path else None
    report = run_stress(
        Path(args.tools_root),
        host=args.host, port=args.port, seed=args.seed, size=args.size,
        duration_minutes=args.duration_minutes, request_interval_s=args.request_interval_s,
        timeout=args.timeout, dtype=args.dtype,
        launch_game=args.launch_game, game_binary=args.game_binary,
        game_args=args.game_args, game_report_path=game_report_path,
    )
    text = json.dumps(asdict(report), indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)

    if not report.exit_gate_passed:
        gates = []
        if not report.exit_gate_protocol_ok:
            gates.append("provider_unreachable_or_protocol_failure")
        if not report.exit_gate_headroom_ok:
            gates.append(f"headroom<{VULKAN_HEADROOM_FLOOR_BYTES//(1024*1024)}MiB")
        if not report.exit_gate_no_device_lost:
            gates.append("DeviceLost")
        if not report.exit_gate_hitch_ok:
            gates.append(f"hitch>{HITCH_P95_CEILING_MS}ms({report.hitch_source})")
        print(f"EXIT GATE FAILED: {', '.join(gates)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
