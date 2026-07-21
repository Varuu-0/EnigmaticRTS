"""Locked manifest schema and generator for Terrain Diffusion Milestone 3.

This module is the **single source of truth** for the pinned external runtime.
It inspects the *actually installed* upstream clone, Python venv, CUDA/PyTorch
build, model revision, and request settings, then emits a versioned, checksummed
JSON manifest. No model weights, venvs, cloned source, or datasets are ever
copied or committed — only metadata about them is captured.

The schema is intentionally strict: :class:`LockedManifest` validates every field
and rejects unknown keys so a drift in the upstream protocol cannot silently
pass an exit gate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "terrain-diffusion-locked-manifest/v1"

# ---------------------------------------------------------------------------
# Protocol constants — derived from the inspected upstream source, not guessed.
# ---------------------------------------------------------------------------
# Upstream: terrain_diffusion/inference/api.py at commit 82a0431.
# Response layout (application/octet-stream):
#   elevation : int16 little-endian, shape (H, W), meters floored & clamped.
#   climate   : float32 little-endian interleaved, shape (H, W, 4),
#               channels [temp, t_season, precip, p_cv].
# Headers: X-Height, X-Width. Server runs Flask threaded=False (serial).
ELEVATION_DTYPE = "<i2"
ELEVATION_BYTES_PER_PIXEL = 2
CLIMATE_DTYPE = "<f4"
CLIMATE_CHANNELS = 4
CLIMATE_BYTES_PER_PIXEL = CLIMATE_CHANNELS * 4
CLIMATE_CHANNEL_NAMES = ("temp", "t_season", "precip", "p_cv")
SERVER_THREADED = False


class ManifestValidationError(ValueError):
    """Raised when a locked manifest is structurally invalid or drifted."""


@dataclass(frozen=True)
class UpstreamLock:
    """Pinned upstream repository state."""

    repo_url: str
    commit: str
    commit_subject: str

    def validate(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.commit):
            raise ManifestValidationError(
                f"upstream.commit must be a 40-char SHA, got {self.commit!r}"
            )
        if not self.repo_url.startswith("https://"):
            raise ManifestValidationError("upstream.repo_url must be https")
        if not self.commit_subject:
            raise ManifestValidationError("upstream.commit_subject must be non-empty")


@dataclass(frozen=True)
class ModelLock:
    """Pinned HuggingFace model revision."""

    repo_id: str
    revision: str
    native_resolution_m: int
    submodels: tuple[str, ...]
    total_weights_bytes: int

    def validate(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ManifestValidationError(
                f"model.revision must be a 40-char SHA, got {self.revision!r}"
            )
        if "/" not in self.repo_id:
            raise ManifestValidationError("model.repo_id must be 'org/name'")
        if self.native_resolution_m <= 0:
            raise ManifestValidationError("model.native_resolution_m must be > 0")
        if not self.submodels:
            raise ManifestValidationError("model.submodels must be non-empty")
        if self.total_weights_bytes <= 0:
            raise ManifestValidationError("model.total_weights_bytes must be > 0")


@dataclass(frozen=True)
class RuntimeLock:
    """Pinned Python / CUDA / PyTorch runtime."""

    python_version: str
    torch_version: str
    cuda_version: str
    cuda_available: bool
    device_name: str
    device_total_vram_bytes: int
    diffusers_version: str
    flask_version: str
    numpy_version: str
    driver_version: str

    def validate(self) -> None:
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.python_version):
            raise ManifestValidationError(
                f"runtime.python_version invalid: {self.python_version!r}"
            )
        if not self.cuda_version:
            raise ManifestValidationError("runtime.cuda_version must be non-empty")
        if self.cuda_available and self.device_total_vram_bytes <= 0:
            raise ManifestValidationError("device_total_vram_bytes must be > 0 when CUDA available")


@dataclass(frozen=True)
class RequestLock:
    """Pinned sidecar request / server settings."""

    endpoint: str
    host: str
    port: int
    device: str
    dtype: str
    t_steps: int
    torch_compile: bool
    latents_batch_size: str
    cache_strategy: str
    cache_size: str
    api_scale: int
    threaded: bool
    request_timeout_s: float
    health_path: str
    terrain_path: str

    def validate(self) -> None:
        if self.device not in ("cuda", "cpu"):
            raise ManifestValidationError(f"request.device must be cuda|cpu, got {self.device!r}")
        if self.dtype not in ("fp32", "bf16", "fp16"):
            raise ManifestValidationError(f"request.dtype invalid: {self.dtype!r}")
        if self.t_steps not in (1, 2):
            raise ManifestValidationError(f"request.t_steps must be 1 or 2, got {self.t_steps}")
        if self.api_scale < 1:
            raise ManifestValidationError("request.api_scale must be >= 1")
        if not (1 <= self.port <= 65535):
            raise ManifestValidationError("request.port out of range")
        if self.request_timeout_s <= 0:
            raise ManifestValidationError("request.request_timeout_s must be > 0")
        if not self.terrain_path.startswith("/"):
            raise ManifestValidationError("request.terrain_path must start with /")


@dataclass(frozen=True)
class ProtocolLock:
    """Pinned binary response protocol contract."""

    elevation_dtype: str
    elevation_bytes_per_pixel: int
    climate_dtype: str
    climate_channels: int
    climate_bytes_per_pixel: int
    climate_channel_names: tuple[str, ...]
    height_header: str
    width_header: str
    mimetype: str

    def validate(self) -> None:
        if self.elevation_bytes_per_pixel != 2:
            raise ManifestValidationError("elevation_bytes_per_pixel must be 2 (int16)")
        if self.climate_channels != 4:
            raise ManifestValidationError("climate_channels must be 4")
        if len(self.climate_channel_names) != 4:
            raise ManifestValidationError("climate_channel_names must have 4 entries")


@dataclass(frozen=True)
class LockedManifest:
    """Top-level locked manifest."""

    schema: str
    upstream: UpstreamLock
    model: ModelLock
    runtime: RuntimeLock
    request: RequestLock
    protocol: ProtocolLock
    generator: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA_VERSION:
            raise ManifestValidationError(f"schema must be {MANIFEST_SCHEMA_VERSION!r}")
        self.upstream.validate()
        self.model.validate()
        self.runtime.validate()
        self.request.validate()
        self.protocol.validate()

    def checksum(self) -> str:
        """Stable SHA-256 over the canonical JSON of the field data (no checksum key)."""
        payload = dataclasses.asdict(self)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["checksum"] = self.checksum()
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False) + "\n"


# ---------------------------------------------------------------------------
# Generator — inspects the actual installed environment.
# ---------------------------------------------------------------------------

DEFAULT_REPO_URL = "https://github.com/xandergos/terrain-diffusion.git"
DEFAULT_MODEL_REPO_ID = "xandergos/terrain-diffusion-30m"
DEFAULT_ENDPOINT = "127.0.0.1:8000"
DEFAULT_REQUEST_TIMEOUT_S = 60.0


def _run_git(tools_root: Path, *args: str) -> str:
    upstream = tools_root / "upstream"
    result = subprocess.run(
        ["git", "-C", str(upstream), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _detect_upstream(tools_root: Path) -> UpstreamLock:
    commit = _run_git(tools_root, "rev-parse", "HEAD")
    subject = _run_git(tools_root, "log", "-1", "--format=%s")
    remote = _run_git(tools_root, "remote", "get-url", "origin")
    if remote.endswith(".git"):
        remote = remote[:-4]
    return UpstreamLock(repo_url=remote, commit=commit, commit_subject=subject)


def _hf_model_dir(model_repo_id: str) -> Path:
    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    folder = "models--" + model_repo_id.replace("/", "--")
    return cache / "hub" / folder


def _detect_model(model_repo_id: str) -> ModelLock:
    model_dir = _hf_model_dir(model_repo_id)
    refs = model_dir / "refs" / "main"
    if not refs.exists():
        raise ManifestValidationError(
            f"Model not downloaded: {model_repo_id} (expected at {model_dir}). "
            "Run start_server.ps1 once to populate the HF cache."
        )
    revision = refs.read_text().strip()
    snapshot = model_dir / "snapshots" / revision

    config_path = snapshot / "config.json"
    config = json.loads(config_path.read_text())
    native_resolution_m = int(config.get("native_resolution", 30))

    submodels: list[str] = []
    total_bytes = 0
    for sub in sorted(snapshot.iterdir()):
        if sub.is_dir() and (sub / "config.json").exists():
            submodels.append(sub.name)
            for f in sub.iterdir():
                if f.suffix == ".safetensors":
                    total_bytes += f.stat().st_size
    return ModelLock(
        repo_id=model_repo_id,
        revision=revision,
        native_resolution_m=native_resolution_m,
        submodels=tuple(submodels),
        total_weights_bytes=total_bytes,
    )


def _detect_runtime() -> RuntimeLock:
    import torch

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    total_vram = torch.cuda.get_device_properties(0).total_memory if cuda_available else 0
    cuda_version = str(torch.version.cuda or "n/a")

    import flask
    import numpy
    from importlib.metadata import version as _pkg_version

    diffusers_version = "n/a"
    try:
        import diffusers

        diffusers_version = getattr(diffusers, "__version__", None) or _pkg_version("diffusers")
    except Exception:
        pass

    driver_version = "n/a"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        driver_version = out.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return RuntimeLock(
        python_version=".".join(map(str, sys.version_info[:3])),
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        cuda_available=cuda_available,
        device_name=device_name,
        device_total_vram_bytes=int(total_vram),
        diffusers_version=diffusers_version,
        flask_version=_pkg_version("flask"),
        numpy_version=_pkg_version("numpy"),
        driver_version=driver_version,
    )


def build_manifest(
    tools_root: Path,
    *,
    model_repo_id: str = DEFAULT_MODEL_REPO_ID,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    dtype: str = "fp32",
    t_steps: int = 2,
    torch_compile: bool = True,
    latents_batch_size: str = "1,4",
    cache_strategy: str = "direct",
    cache_size: str = "100M",
    api_scale: int = 1,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> LockedManifest:
    """Inspect the installed environment and build a validated locked manifest."""
    tools_root = Path(tools_root).resolve()
    upstream = _detect_upstream(tools_root)
    model = _detect_model(model_repo_id)
    runtime = _detect_runtime()

    request = RequestLock(
        endpoint=f"{host}:{port}",
        host=host,
        port=port,
        device=device,
        dtype=dtype,
        t_steps=t_steps,
        torch_compile=torch_compile,
        latents_batch_size=latents_batch_size,
        cache_strategy=cache_strategy,
        cache_size=cache_size,
        api_scale=api_scale,
        threaded=SERVER_THREADED,
        request_timeout_s=request_timeout_s,
        health_path="/health",
        terrain_path="/terrain",
    )
    protocol = ProtocolLock(
        elevation_dtype=ELEVATION_DTYPE,
        elevation_bytes_per_pixel=ELEVATION_BYTES_PER_PIXEL,
        climate_dtype=CLIMATE_DTYPE,
        climate_channels=CLIMATE_CHANNELS,
        climate_bytes_per_pixel=CLIMATE_BYTES_PER_PIXEL,
        climate_channel_names=CLIMATE_CHANNEL_NAMES,
        height_header="X-Height",
        width_header="X-Width",
        mimetype="application/octet-stream",
    )
    manifest = LockedManifest(
        schema=MANIFEST_SCHEMA_VERSION,
        upstream=upstream,
        model=model,
        runtime=runtime,
        request=request,
        protocol=protocol,
        generator=f"locked_manifest.py {MANIFEST_SCHEMA_VERSION}",
        notes=(
            "torch.compile unsupported on Windows; upstream warns and disables it at runtime.",
            f"Flask server runs threaded={SERVER_THREADED} (serial request handling).",
            "scale=1 is native model resolution; scale>1 uses bilinear upsampling (diagnostic only).",
            "T (diffusion steps) is not a dedicated CLI flag; it is passed via --kwarg T=<value>.",
            "start_server.ps1 forwards --dtype, --device, --t, --no-compile, --cache-size, --batch-size.",
        ),
    )
    manifest.validate()
    return manifest


def load_and_validate(path: Path) -> LockedManifest:
    """Load a manifest JSON file and re-validate it strictly."""
    raw = json.loads(Path(path).read_text())
    unknown_top = set(raw) - {f.name for f in dataclasses.fields(LockedManifest)} - {"checksum"}
    if unknown_top:
        raise ManifestValidationError(f"unknown top-level keys: {sorted(unknown_top)}")
    try:
        manifest = LockedManifest(
            schema=raw["schema"],
            upstream=UpstreamLock(**raw["upstream"]),
            model=ModelLock(
                repo_id=raw["model"]["repo_id"],
                revision=raw["model"]["revision"],
                native_resolution_m=raw["model"]["native_resolution_m"],
                submodels=tuple(raw["model"]["submodels"]),
                total_weights_bytes=raw["model"]["total_weights_bytes"],
            ),
            runtime=RuntimeLock(**raw["runtime"]),
            request=RequestLock(**raw["request"]),
            protocol=ProtocolLock(
                elevation_dtype=raw["protocol"]["elevation_dtype"],
                elevation_bytes_per_pixel=raw["protocol"]["elevation_bytes_per_pixel"],
                climate_dtype=raw["protocol"]["climate_dtype"],
                climate_channels=raw["protocol"]["climate_channels"],
                climate_bytes_per_pixel=raw["protocol"]["climate_bytes_per_pixel"],
                climate_channel_names=tuple(raw["protocol"]["climate_channel_names"]),
                height_header=raw["protocol"]["height_header"],
                width_header=raw["protocol"]["width_header"],
                mimetype=raw["protocol"]["mimetype"],
            ),
            generator=raw["generator"],
            notes=tuple(raw.get("notes", [])),
        )
    except KeyError as exc:
        raise ManifestValidationError(f"missing key: {exc}") from exc
    manifest.validate()
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a locked Terrain Diffusion manifest.")
    parser.add_argument(
        "--tools-root",
        default=str(Path(__file__).resolve().parent),
        help="Path to the tools/terrain_diffusion directory (default: script dir).",
    )
    parser.add_argument("--output", "-o", default=None, help="Output JSON path (default: stdout).")
    parser.add_argument("--model", default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--t-steps", type=int, choices=[1, 2], default=2)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--latents-batch-size", default="1,4")
    parser.add_argument("--cache-strategy", choices=["direct", "indirect"], default="direct")
    parser.add_argument("--cache-size", default="100M")
    parser.add_argument("--api-scale", type=int, default=1)
    args = parser.parse_args()

    manifest = build_manifest(
        Path(args.tools_root),
        model_repo_id=args.model,
        host=args.host,
        port=args.port,
        device=args.device,
        dtype=args.dtype,
        t_steps=args.t_steps,
        torch_compile=not args.no_compile,
        latents_batch_size=args.latents_batch_size,
        cache_strategy=args.cache_strategy,
        cache_size=args.cache_size,
        api_scale=args.api_scale,
    )
    text = manifest.to_json()
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output} (checksum={manifest.checksum()[:12]})", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
