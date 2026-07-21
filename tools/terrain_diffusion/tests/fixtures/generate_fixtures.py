"""Generate deterministic binary protocol fixtures (no model download required).

Creates a small set of valid and malformed `/terrain` response payloads so the
strict decoder and the Rust-side adapter can be tested offline.

Run:  python -m tests.fixtures.generate_fixtures --out tests/fixtures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Import the protocol constants from the parent package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from protocol import (  # noqa: E402
    CLIMATE_CHANNELS,
    CLIMATE_DTYPE,
    ELEVATION_DTYPE,
    decode_payload,
    expected_payload_size,
)

FIXTURE_SEED = 0x5EEDF100
DEFAULT_H = 4
DEFAULT_W = 6


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_valid_payload(h: int = DEFAULT_H, w: int = DEFAULT_W, *, seed: int = FIXTURE_SEED) -> bytes:
    """A well-formed response: int16 elevation + float32 climate (H*W*4)."""
    rng = _rng(seed)
    elev = rng.integers(-500, 4000, size=(h, w), dtype=np.int16)
    climate = rng.standard_normal((h, w, CLIMATE_CHANNELS)).astype(np.float32)
    climate[:, :, 0] = np.clip(climate[:, :, 0] * 8 + 15, -60, 40)  # temp C
    climate[:, :, 2] = np.clip(climate[:, :, 2] * 400 + 1000, 0, 5000)  # precip mm
    return elev.astype(ELEVATION_DTYPE).tobytes() + climate.astype(CLIMATE_DTYPE).tobytes()


def make_valid_elevation_only(h: int = DEFAULT_H, w: int = DEFAULT_W, *, seed: int = FIXTURE_SEED) -> bytes:
    """A well-formed elevation-only response (no climate bytes)."""
    rng = _rng(seed + 1)
    elev = rng.integers(-100, 2000, size=(h, w), dtype=np.int16)
    return elev.astype(ELEVATION_DTYPE).tobytes()


def make_truncated_payload(h: int = DEFAULT_H, w: int = DEFAULT_W, *, seed: int = FIXTURE_SEED) -> bytes:
    """Body shorter than elevation-only — must be rejected."""
    return make_valid_payload(h, w, seed=seed)[:5]


def make_oversized_payload(h: int = DEFAULT_H, w: int = DEFAULT_W, *, seed: int = FIXTURE_SEED) -> bytes:
    """Body longer than full payload — must be rejected."""
    return make_valid_payload(h, w, seed=seed) + b"\x00" * 10


def make_elevation_clamped_payload(h: int = DEFAULT_H, w: int = DEFAULT_W) -> bytes:
    """Elevation with extreme but valid int16 values (boundary test)."""
    elev = np.full((h, w), -32768, dtype=np.int16)
    elev[0, 0] = 32767
    climate = np.zeros((h, w, CLIMATE_CHANNELS), dtype=np.float32)
    return elev.tobytes() + climate.tobytes()


def make_nan_climate_payload(h: int = DEFAULT_H, w: int = DEFAULT_W, *, seed: int = FIXTURE_SEED) -> bytes:
    """Valid byte layout but climate contains NaN — structural decode passes, finite check fails."""
    payload = bytearray(make_valid_payload(h, w, seed=seed))
    # Corrupt the first climate float32 to NaN (bytes 0x7FC00000 little-endian).
    elev_bytes = h * w * ELEVATION_DTYPE.itemsize
    payload[elev_bytes:elev_bytes + 4] = b"\x00\x00\xc0\x7f"
    return bytes(payload)


def make_wrong_dtype_interleaved_payload(h: int = DEFAULT_H, w: int = DEFAULT_W, *, seed: int = FIXTURE_SEED) -> bytes:
    """Climate bytes are int16-sized (half length) — must be rejected by length check."""
    rng = _rng(seed + 2)
    elev = rng.integers(0, 1000, size=(h, w), dtype=np.int16)
    fake_climate = rng.integers(0, 100, size=(h, w, CLIMATE_CHANNELS), dtype=np.int16)
    return elev.tobytes() + fake_climate.tobytes()


FIXTURES = {
    "valid_full.bin": make_valid_payload,
    "valid_elev_only.bin": make_valid_elevation_only,
    "valid_clamped.bin": make_elevation_clamped_payload,
    "malformed_truncated.bin": make_truncated_payload,
    "malformed_oversized.bin": make_oversized_payload,
    "malformed_nan_climate.bin": make_nan_climate_payload,
    "malformed_wrong_dtype.bin": make_wrong_dtype_interleaved_payload,
}


def generate_all(out_dir: Path) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, fn in FIXTURES.items():
        data = fn(DEFAULT_H, DEFAULT_W) if name == "valid_clamped.bin" else fn(DEFAULT_H, DEFAULT_W)
        path = out_dir / name
        path.write_bytes(data)
        manifest[name] = {
            "bytes": len(data),
            "expected_size_full": expected_payload_size(DEFAULT_H, DEFAULT_W, with_climate=True),
            "expected_size_elev_only": expected_payload_size(DEFAULT_H, DEFAULT_W, with_climate=False),
            "dimensions": [DEFAULT_H, DEFAULT_W],
        }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic protocol fixtures.")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent), help="Output directory.")
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--w", type=int, default=DEFAULT_W)
    args = parser.parse_args()

    out = Path(args.out)
    info = generate_all(out)
    for name, meta in sorted(info.items()):
        print(f"{name}: {meta['bytes']} bytes")
    print(f"\nGenerated {len(info)} fixtures in {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
