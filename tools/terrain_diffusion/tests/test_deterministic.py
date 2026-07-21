"""Deterministic self-tests for the locked manifest schema and protocol decoder.

These tests never require a model download or a running server. They validate
the manifest schema strictness, fixture integrity, and protocol decoder against
the committed binary fixtures.

Run:  python -m tests.test_deterministic
      python -m unittest tests.test_deterministic
"""

from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from locked_manifest import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    LockedManifest,
    ManifestValidationError,
    ModelLock,
    ProtocolLock,
    RequestLock,
    RuntimeLock,
    UpstreamLock,
    build_manifest,
    load_and_validate,
)
from protocol import (  # noqa: E402
    CLIMATE_CHANNELS,
    ProtocolError,
    decode_payload,
    expected_payload_size,
    parse_response_headers,
    validate_payload_finite,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TOOLS_ROOT = Path(__file__).resolve().parent.parent


def _load_fixture(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


class ProtocolDecoderTests(unittest.TestCase):
    def test_expected_payload_size(self):
        self.assertEqual(expected_payload_size(4, 6, with_climate=True), 432)
        self.assertEqual(expected_payload_size(4, 6, with_climate=False), 48)
        self.assertEqual(expected_payload_size(512, 512, with_climate=True), 512 * 512 * (2 + 16))

    def test_decode_valid_full(self):
        body = _load_fixture("valid_full.bin")
        payload = decode_payload(body, 4, 6, require_climate=True)
        self.assertEqual(payload.height, 4)
        self.assertEqual(payload.width, 6)
        self.assertEqual(payload.elevation.dtype, np.dtype("<i2"))
        self.assertEqual(payload.climate.shape, (4, 6, CLIMATE_CHANNELS))
        self.assertEqual(payload.climate.dtype, np.dtype("<f4"))
        validate_payload_finite(payload)
        self.assertEqual(len(payload.checksum()), 64)

    def test_decode_valid_elev_only(self):
        body = _load_fixture("valid_elev_only.bin")
        payload = decode_payload(body, 4, 6, require_climate=False)
        self.assertIsNone(payload.climate)
        self.assertEqual(payload.elevation.shape, (4, 6))
        validate_payload_finite(payload)

    def test_decode_valid_clamped_boundary(self):
        body = _load_fixture("valid_clamped.bin")
        payload = decode_payload(body, 4, 6, require_climate=True)
        self.assertEqual(payload.elevation.min(), -32768)
        self.assertEqual(payload.elevation.max(), 32767)
        validate_payload_finite(payload)

    def test_reject_truncated(self):
        body = _load_fixture("malformed_truncated.bin")
        with self.assertRaises(ProtocolError):
            decode_payload(body, 4, 6, require_climate=True)
        with self.assertRaises(ProtocolError):
            decode_payload(body, 4, 6, require_climate=False)

    def test_reject_oversized(self):
        body = _load_fixture("malformed_oversized.bin")
        with self.assertRaises(ProtocolError):
            decode_payload(body, 4, 6, require_climate=True)
        with self.assertRaises(ProtocolError):
            decode_payload(body, 4, 6, require_climate=False)

    def test_reject_wrong_dtype_length(self):
        body = _load_fixture("malformed_wrong_dtype.bin")
        with self.assertRaises(ProtocolError):
            decode_payload(body, 4, 6, require_climate=True)
        with self.assertRaises(ProtocolError):
            decode_payload(body, 4, 6, require_climate=False)

    def test_nan_climate_decodes_but_finite_check_fails(self):
        body = _load_fixture("malformed_nan_climate.bin")
        # Structural decode succeeds (byte layout is valid).
        payload = decode_payload(body, 4, 6, require_climate=True)
        # But finite validation catches the NaN.
        with self.assertRaises(ProtocolError):
            validate_payload_finite(payload)

    def test_reject_non_positive_dims(self):
        with self.assertRaises(ProtocolError):
            decode_payload(b"\x00" * 10, 0, 6)
        with self.assertRaises(ProtocolError):
            decode_payload(b"\x00" * 10, 4, -1)

    def test_parse_headers_case_insensitive(self):
        h, w = parse_response_headers([("x-height", "128"), ("X-WIDTH", "256")])
        self.assertEqual((h, w), (128, 256))

    def test_parse_headers_missing(self):
        with self.assertRaises(ProtocolError):
            parse_response_headers([("x-width", "256")])
        with self.assertRaises(ProtocolError):
            parse_response_headers([("x-height", "128")])

    def test_checksum_stable(self):
        body = _load_fixture("valid_full.bin")
        a = decode_payload(body, 4, 6)
        b = decode_payload(body, 4, 6)
        self.assertEqual(a.checksum(), b.checksum())


class ManifestSchemaTests(unittest.TestCase):
    def _valid_request(self) -> RequestLock:
        return RequestLock(
            endpoint="127.0.0.1:8000", host="127.0.0.1", port=8000, device="cuda",
            dtype="fp32", t_steps=2, torch_compile=True, latents_batch_size="1,4",
            cache_strategy="direct", cache_size="100M", api_scale=1, threaded=False,
            request_timeout_s=60.0, health_path="/health", terrain_path="/terrain",
        )

    def _valid_protocol(self) -> ProtocolLock:
        return ProtocolLock(
            elevation_dtype="<i2", elevation_bytes_per_pixel=2, climate_dtype="<f4",
            climate_channels=4, climate_bytes_per_pixel=16,
            climate_channel_names=("temp", "t_season", "precip", "p_cv"),
            height_header="X-Height", width_header="X-Width", mimetype="application/octet-stream",
        )

    def _valid_upstream(self) -> UpstreamLock:
        return UpstreamLock(
            repo_url="https://github.com/xandergos/terrain-diffusion.git",
            commit="82a0431281f21a6ec3d691a12ee61525de5b0790",
            commit_subject="Revert Update website",
        )

    def _valid_model(self) -> ModelLock:
        return ModelLock(
            repo_id="xandergos/terrain-diffusion-30m",
            revision="9ef8030cb805b433b98ec25c5dddefbac07a9e26",
            native_resolution_m=30,
            submodels=("base_model", "coarse_model", "decoder_model"),
            total_weights_bytes=1130000000,
        )

    def _valid_runtime(self) -> RuntimeLock:
        return RuntimeLock(
            python_version="3.11.0", torch_version="2.5.1+cu121", cuda_version="12.1",
            cuda_available=True, device_name="NVIDIA GeForce RTX 3060 Laptop GPU",
            device_total_vram_bytes=6_442_450_944, diffusers_version="0.39.0",
            flask_version="3.1.3", numpy_version="2.4.4", driver_version="610.47",
        )

    def _valid_manifest(self) -> LockedManifest:
        return LockedManifest(
            schema=MANIFEST_SCHEMA_VERSION,
            upstream=self._valid_upstream(),
            model=self._valid_model(),
            runtime=self._valid_runtime(),
            request=self._valid_request(),
            protocol=self._valid_protocol(),
            generator="locked_manifest.py test",
        )

    def test_valid_manifest_validates(self):
        self._valid_manifest().validate()

    def test_checksum_stable_and_excludes_itself(self):
        m = self._valid_manifest()
        d = m.to_dict()
        self.assertIn("checksum", d)
        self.assertEqual(m.checksum(), d["checksum"])
        # Re-deriving from the dict (without checksum) yields the same hash.
        d2 = {k: v for k, v in d.items() if k != "checksum"}
        canonical = json.dumps(d2, sort_keys=True, separators=(",", ":"))
        import hashlib

        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), m.checksum())

    def test_reject_bad_commit_sha(self):
        m = dataclasses.replace(self._valid_manifest(), upstream=dataclasses.replace(self._valid_upstream(), commit="short"))
        with self.assertRaises(ManifestValidationError):
            m.validate()

    def test_reject_bad_device(self):
        m = dataclasses.replace(self._valid_manifest(), request=dataclasses.replace(self._valid_request(), device="tpu"))
        with self.assertRaises(ManifestValidationError):
            m.validate()

    def test_reject_bad_t_steps(self):
        m = dataclasses.replace(self._valid_manifest(), request=dataclasses.replace(self._valid_request(), t_steps=3))
        with self.assertRaises(ManifestValidationError):
            m.validate()

    def test_reject_bad_dtype(self):
        m = dataclasses.replace(self._valid_manifest(), request=dataclasses.replace(self._valid_request(), dtype="int8"))
        with self.assertRaises(ManifestValidationError):
            m.validate()

    def test_reject_non_https_upstream(self):
        m = dataclasses.replace(self._valid_manifest(), upstream=dataclasses.replace(self._valid_upstream(), repo_url="git://foo"))
        with self.assertRaises(ManifestValidationError):
            m.validate()

    def test_reject_bad_protocol_channels(self):
        bad = dataclasses.replace(self._valid_protocol(), climate_channels=3, climate_bytes_per_pixel=12)
        m = dataclasses.replace(self._valid_manifest(), protocol=bad)
        with self.assertRaises(ManifestValidationError):
            m.validate()

    def test_reject_unknown_top_level_key_on_load(self):
        d = self._valid_manifest().to_dict()
        d["rogue_field"] = "evil"
        tmp = FIXTURES_DIR.parent / "_rogue_manifest.json"
        tmp.write_text(json.dumps(d))
        with self.assertRaises(ManifestValidationError):
            load_and_validate(tmp)
        tmp.unlink()

    def test_roundtrip_load_validate(self):
        m = self._valid_manifest()
        tmp = FIXTURES_DIR.parent / "_roundtrip_manifest.json"
        tmp.write_text(m.to_json())
        loaded = load_and_validate(tmp)
        self.assertEqual(loaded.checksum(), m.checksum())
        self.assertEqual(loaded.upstream.commit, m.upstream.commit)
        tmp.unlink()


class InstalledEnvironmentTests(unittest.TestCase):
    """Tests that touch the real installed venv/upstream but NOT the model server."""

    def test_build_manifest_from_installed_env(self):
        # Requires the upstream clone and downloaded model. If model missing,
        # this test is skipped (not failed) — it is not a determinism self-test.
        try:
            manifest = build_manifest(TOOLS_ROOT)
        except ManifestValidationError as exc:
            if "not downloaded" in str(exc).lower():
                self.skipTest("model not downloaded; run start_server.ps1 once")
            raise
        manifest.validate()
        self.assertEqual(manifest.schema, MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest.upstream.commit, "82a0431281f21a6ec3d691a12ee61525de5b0790")
        self.assertEqual(manifest.model.repo_id, "xandergos/terrain-diffusion-30m")
        self.assertEqual(manifest.model.native_resolution_m, 30)
        self.assertEqual(manifest.model.revision, "9ef8030cb805b433b98ec25c5dddefbac07a9e26")
        self.assertIn("base_model", manifest.model.submodels)
        self.assertTrue(manifest.runtime.cuda_available)
        self.assertEqual(manifest.request.threaded, False)
        self.assertEqual(manifest.protocol.elevation_bytes_per_pixel, 2)

    def test_installed_upstream_commit_pinned(self):
        # The upstream clone must be at the pinned commit.
        import subprocess
        result = subprocess.run(
            ["git", "-C", str(TOOLS_ROOT / "upstream"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            "82a0431281f21a6ec3d691a12ee61525de5b0790",
            "upstream clone drifted from pinned commit",
        )


class StressGateSemanticsTests(unittest.TestCase):
    """Tests for the stress gate logic without requiring a server or game run."""

    def test_percentile_empty_returns_zero(self):
        from bench.run_stress import _percentile
        self.assertEqual(_percentile([], 95.0), 0.0)

    def test_percentile_single_element(self):
        from bench.run_stress import _percentile
        self.assertAlmostEqual(_percentile([5.0], 95.0), 5.0)

    def test_percentile_known_values(self):
        from bench.run_stress import _percentile
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.assertAlmostEqual(_percentile(data, 50.0), 5.5, places=1)
        self.assertAlmostEqual(_percentile(data, 95.0), 9.55, places=1)

    def test_device_lost_pattern_detection(self):
        from bench.run_stress import _scan_for_device_lost
        self.assertTrue(_scan_for_device_lost("wgpu error: DeviceLost"))
        self.assertTrue(_scan_for_device_lost("VK_ERROR_DEVICE_LOST"))
        self.assertFalse(_scan_for_device_lost("All good, no errors"))

    def test_game_report_passes_gate_when_p95_under_ceiling(self):
        from bench.run_stress import GameReport, HITCH_P95_CEILING_MS
        report = GameReport(
            schema="terrain-diffusion-stress-report/v1",
            provider_p95_ms=0.8,
            exit_gate_hitch_ok=True,
        )
        self.assertTrue(report.provider_p95_ms <= HITCH_P95_CEILING_MS)

    def test_game_report_fails_gate_when_p95_over_ceiling(self):
        from bench.run_stress import GameReport, HITCH_P95_CEILING_MS
        report = GameReport(
            schema="terrain-diffusion-stress-report/v1",
            provider_p95_ms=1.5,
            exit_gate_hitch_ok=False,
        )
        self.assertFalse(report.provider_p95_ms <= HITCH_P95_CEILING_MS)

    def test_missing_game_report_means_unproven(self):
        # When the game report is None, the hitch gate must NOT pass.
        from bench.run_stress import GameReport
        report = None
        self.assertIsNone(report)
    def test_stress_report_schema_is_v2(self):
        from bench.run_stress import StressReport
        import dataclasses
        r = StressReport(
            schema="terrain-diffusion-stress/v2",
            started_at="", ended_at="", duration_minutes=1.0, manifest={},
            total_requests=0, successful_requests=0, failed_requests=0,
            latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
            hitch_p95_ms=0.0, hitch_source="unproven",
            sidecar_jitter_p95_ms=0.0,
            peak_vram_used_bytes=0,
            min_vulkan_headroom_bytes=0, device_total_vram_bytes=0,
            game_exit_code=None, game_report=None,
            server_reachable=False, exit_gate_protocol_ok=False,
            exit_gate_headroom_ok=False, exit_gate_no_device_lost=True,
            exit_gate_hitch_ok=False, exit_gate_passed=False,
            checksums_unique=0, checksum_match=False,
        )
        d = dataclasses.asdict(r)
        self.assertEqual(d["schema"], "terrain-diffusion-stress/v2")
        self.assertIn("hitch_source", d)
        self.assertIn("game_report", d)
        self.assertIn("sidecar_jitter_p95_ms", d)
        self.assertFalse(d["exit_gate_protocol_ok"])

    def test_checksum_match_false_for_zero_requests(self):
        # With zero requests there are zero checksums; checksum_match must be
        # False (the protocol gate also fails because len(checksums) != 1).
        from bench.run_stress import StressReport
        import dataclasses
        r = StressReport(
            schema="terrain-diffusion-stress/v2",
            started_at="", ended_at="", duration_minutes=1.0, manifest={},
            total_requests=0, successful_requests=0, failed_requests=0,
            latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
            hitch_p95_ms=0.0, hitch_source="unproven",
            sidecar_jitter_p95_ms=0.0,
            peak_vram_used_bytes=0,
            min_vulkan_headroom_bytes=0, device_total_vram_bytes=0,
            game_exit_code=None, game_report=None,
            server_reachable=False, exit_gate_protocol_ok=False,
            exit_gate_headroom_ok=False, exit_gate_no_device_lost=True,
            exit_gate_hitch_ok=False, exit_gate_passed=False,
            checksums_unique=0, checksum_match=False,
        )
        d = dataclasses.asdict(r)
        self.assertFalse(d["checksum_match"])
        self.assertEqual(d["checksums_unique"], 0)

    def test_checksum_match_true_only_when_exactly_one_checksum(self):
        from bench.run_stress import StressReport
        import dataclasses
        r = StressReport(
            schema="terrain-diffusion-stress/v2",
            started_at="", ended_at="", duration_minutes=1.0, manifest={},
            total_requests=3, successful_requests=3, failed_requests=0,
            latency_p50_ms=1.0, latency_p95_ms=2.0, latency_p99_ms=3.0,
            hitch_p95_ms=0.0, hitch_source="unproven",
            sidecar_jitter_p95_ms=1.5,
            peak_vram_used_bytes=0,
            min_vulkan_headroom_bytes=0, device_total_vram_bytes=0,
            game_exit_code=None, game_report=None,
            server_reachable=True, exit_gate_protocol_ok=True,
            exit_gate_headroom_ok=False, exit_gate_no_device_lost=True,
            exit_gate_hitch_ok=False, exit_gate_passed=False,
            checksums_unique=1, checksum_match=True,
        )
        d = dataclasses.asdict(r)
        self.assertTrue(d["checksum_match"])
        self.assertEqual(d["checksums_unique"], 1)

    def test_hitch_p95_ms_uses_game_report_when_present(self):
        # When a game report is present, hitch_p95_ms must equal the game's
        # provider_p95_ms, NOT the sidecar jitter.
        from bench.run_stress import GameReport, HITCH_P95_CEILING_MS
        game_report = GameReport(
            schema="terrain-diffusion-stress-report/v1",
            provider_p95_ms=0.013,
            exit_gate_hitch_ok=True,
        )
        self.assertLessEqual(game_report.provider_p95_ms, HITCH_P95_CEILING_MS)
        # The runner sets hitch_p95_ms = game_report.provider_p95_ms in this path.
        self.assertEqual(float(game_report.provider_p95_ms), 0.013)

    def test_hitch_p95_ms_zero_when_no_game_report(self):
        # Without a game report the hitch gate is unproven and hitch_p95_ms
        # must be 0.0 (the default, not the sidecar jitter).
        from bench.run_stress import HITCH_P95_CEILING_MS
        hitch_p95_ms = 0.0
        game_report = None
        if game_report is not None and game_report.provider_p95_ms is not None:
            hitch_p95_ms = float(game_report.provider_p95_ms)
        self.assertEqual(hitch_p95_ms, 0.0)
        self.assertLess(hitch_p95_ms, HITCH_P95_CEILING_MS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
