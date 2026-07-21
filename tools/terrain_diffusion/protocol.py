"""Binary protocol decoder for the Terrain Diffusion `/terrain` response.

Implements the exact wire format inspected in upstream
`terrain_diffusion/inference/api.py` (commit 82a0431):

  elevation : int16 little-endian (H * W * 2 bytes), meters floored & clamped.
  climate   : float32 little-endian interleaved (H * W * 4 * 4 bytes),
              channels [temp, t_season, precip, p_cv].

Headers `X-Height` and `X-Width` carry the output pixel dimensions. The server
runs with `threaded=False` so responses are serial.

This module is dependency-light (numpy only) so it can be imported by tests that
must not require a model download or a running server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ELEVATION_DTYPE = np.dtype("<i2")
CLIMATE_DTYPE = np.dtype("<f4")
CLIMATE_CHANNELS = 4
CLIMATE_CHANNEL_NAMES = ("temp", "t_season", "precip", "p_cv")
HEIGHT_HEADER = "X-Height"
WIDTH_HEADER = "X-Width"


class ProtocolError(ValueError):
    """Raised when a binary payload does not conform to the locked protocol."""


@dataclass(frozen=True)
class TerrainPayload:
    """Decoded terrain response."""

    height: int
    width: int
    elevation: np.ndarray  # (H, W) int16
    climate: np.ndarray | None  # (H, W, 4) float32, interleaved

    def checksum(self) -> str:
        """SHA-256 over the raw response bytes (elevation || climate).

        Reproducible across runs for identical model+seed+box+dtype.
        """
        import hashlib

        h = hashlib.sha256()
        h.update(self.elevation.tobytes())
        if self.climate is not None:
            h.update(self.climate.tobytes())
        return h.hexdigest()


def expected_payload_size(h: int, w: int, *, with_climate: bool = True) -> int:
    """Byte length of a valid response for the given dimensions."""
    elev = h * w * ELEVATION_DTYPE.itemsize
    if not with_climate:
        return elev
    return elev + h * w * CLIMATE_CHANNELS * CLIMATE_DTYPE.itemsize


def parse_response_headers(headers: dict[str, str] | Sequence[tuple[str, str]]) -> tuple[int, int]:
    """Extract (height, width) from response headers, case-insensitively."""
    norm: dict[str, str] = {}
    for k, v in (headers.items() if isinstance(headers, dict) else headers):
        norm[k.lower()] = v
    if HEIGHT_HEADER.lower() not in norm:
        raise ProtocolError(f"missing {HEIGHT_HEADER} header")
    if WIDTH_HEADER.lower() not in norm:
        raise ProtocolError(f"missing {WIDTH_HEADER} header")
    try:
        h = int(norm[HEIGHT_HEADER.lower()])
        w = int(norm[WIDTH_HEADER.lower()])
    except ValueError as exc:
        raise ProtocolError(f"non-integer X-Height/X-Width: {exc}") from exc
    if h <= 0 or w <= 0:
        raise ProtocolError(f"non-positive dimensions: h={h} w={w}")
    return h, w


def decode_payload(
    body: bytes,
    height: int,
    width: int,
    *,
    require_climate: bool = True,
) -> TerrainPayload:
    """Strictly decode a raw `/terrain` response body.

    Raises :class:`ProtocolError` on any structural mismatch so malformed
    fixtures and live drift are rejected rather than silently mishandled.
    """
    if height <= 0 or width <= 0:
        raise ProtocolError(f"non-positive dimensions: h={height} w={width}")

    elev_bytes = height * width * ELEVATION_DTYPE.itemsize
    climate_bytes = height * width * CLIMATE_CHANNELS * CLIMATE_DTYPE.itemsize
    total_with_climate = elev_bytes + climate_bytes

    if require_climate:
        if len(body) != total_with_climate:
            raise ProtocolError(
                f"body length {len(body)} != expected {total_with_climate} "
                f"(elev={elev_bytes} + climate={climate_bytes}) for {height}x{width}"
            )
    else:
        if len(body) == elev_bytes:
            climate_bytes = 0
        elif len(body) == total_with_climate:
            pass
        else:
            raise ProtocolError(
                f"body length {len(body)} matches neither elevation-only ({elev_bytes}) "
                f"nor full ({total_with_climate}) for {height}x{width}"
            )

    elev = np.frombuffer(body[:elev_bytes], dtype=ELEVATION_DTYPE).reshape(height, width)

    climate: np.ndarray | None = None
    if climate_bytes:
        climate = (
            np.frombuffer(body[elev_bytes:elev_bytes + climate_bytes], dtype=CLIMATE_DTYPE)
            .reshape(height, width, CLIMATE_CHANNELS)
        )

    return TerrainPayload(height=height, width=width, elevation=elev, climate=climate)


def validate_payload_finite(payload: TerrainPayload) -> None:
    """Assert elevation is in int16 range and climate is finite."""
    if payload.elevation.min() < np.iinfo(np.int16).min:
        raise ProtocolError("elevation below int16 min")
    if payload.elevation.max() > np.iinfo(np.int16).max:
        raise ProtocolError("elevation above int16 max")
    if payload.climate is not None:
        if not np.all(np.isfinite(payload.climate)):
            raise ProtocolError("climate contains non-finite values")
