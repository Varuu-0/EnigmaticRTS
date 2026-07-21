"""fp32 vs fp16 output comparison for Terrain Diffusion.

Compares response checksums and statistical differences between fp32 and fp16
model dtype runs. Because the upstream server pins dtype at startup via
`--dtype`, this script does NOT restart the server itself; instead it records
the current server's dtype from a manifest argument and emits a comparison
report comparing two independently-collected benchmark JSON files.

The intended workflow:
  1. Start the server with --dtype fp32, run run_benchmarks.py, save as fp32.json
  2. Start the server with --dtype fp16, run run_benchmarks.py, save as fp16.json
  3. python -m bench.compare_dtypes --fp32 fp32.json --fp16 fp16.json

This avoids automating server restarts (which can trigger the Optimus
DeviceLost issue documented in AGENTS.md) and keeps the comparison honest.

Usage:
  python -m bench.compare_dtypes --fp32 report_fp32.json --fp16 report_fp16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DtypeComparison:
    schema: str
    fp32_dtype: str
    fp16_dtype: str
    fp32_manifest_checksum: str
    fp16_manifest_checksum: str
    same_manifest_except_dtype: bool
    per_size: list[dict[str, Any]] = field(default_factory=list)
    conclusion: str = ""
    notes: list[str] = field(default_factory=list)


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _get_dtype(report: dict[str, Any]) -> str:
    return report.get("manifest", {}).get("request", {}).get("dtype", "unknown")


def _get_manifest_checksum(report: dict[str, Any]) -> str:
    return report.get("manifest", {}).get("checksum", "unknown")


def _get_manifest_without_dtype(report: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(json.dumps(report.get("manifest", {})))
    req = manifest.get("request", {})
    req.pop("dtype", None)
    manifest["request"] = req
    # The checksum is derived from the full manifest including dtype, so it
    # necessarily differs; exclude it from the equality comparison.
    manifest.pop("checksum", None)
    return manifest


def compare(fp32_path: Path, fp16_path: Path) -> DtypeComparison:
    fp32 = _load_report(fp32_path)
    fp16 = _load_report(fp16_path)

    fp32_dtype = _get_dtype(fp32)
    fp16_dtype = _get_dtype(fp16)
    fp32_cs = _get_manifest_checksum(fp32)
    fp16_cs = _get_manifest_checksum(fp16)

    m32 = _get_manifest_without_dtype(fp32)
    m16 = _get_manifest_without_dtype(fp16)
    same_manifest = m32 == m16

    per_size: list[dict[str, Any]] = []
    fp32_tiers = {t["size"]: t for t in fp32.get("tiers", []) if "size" in t}
    fp16_tiers = {t["size"]: t for t in fp16.get("tiers", []) if "size" in t}

    for size in sorted(set(fp32_tiers) | set(fp16_tiers)):
        t32 = fp32_tiers.get(size, {})
        t16 = fp16_tiers.get(size, {})
        cs32 = t32.get("checksums", [])
        cs16 = t16.get("checksums", [])
        entry: dict[str, Any] = {
            "size": size,
            "fp32_warm_p50_ms": t32.get("warm_p50_ms"),
            "fp16_warm_p50_ms": t16.get("warm_p50_ms"),
            "fp32_warm_p95_ms": t32.get("warm_p95_ms"),
            "fp16_warm_p95_ms": t16.get("warm_p95_ms"),
            "fp32_checksum_match": t32.get("checksum_match"),
            "fp16_checksum_match": t16.get("checksum_match"),
            "fp32_first_checksum": cs32[0] if cs32 else None,
            "fp16_first_checksum": cs16[0] if cs16 else None,
            "checksums_identical_across_dtypes": (
                bool(cs32) and bool(cs16) and cs32[0] == cs16[0]
            ),
            "fp32_payload_valid": t32.get("payload_valid"),
            "fp16_payload_valid": t16.get("payload_valid"),
        }
        if t32.get("warm_p50_ms") and t16.get("warm_p50_ms"):
            entry["fp16_speedup_p50"] = t32["warm_p50_ms"] / t16["warm_p50_ms"]
        per_size.append(entry)

    notes: list[str] = []
    if not same_manifest:
        notes.append("Manifests differ beyond dtype — comparison may be confounded.")
    if fp32_dtype != "fp32":
        notes.append(f"fp32 report dtype is {fp32_dtype!r}, not fp32")
    if fp16_dtype != "fp16":
        notes.append(f"fp16 report dtype is {fp16_dtype!r}, not fp16")

    conclusion = "comparison_complete"
    if any(e.get("checksums_identical_across_dtypes") for e in per_size):
        notes.append("fp32 and fp16 produced byte-identical checksums for at least one size.")

    return DtypeComparison(
        schema="terrain-diffusion-dtype-comparison/v1",
        fp32_dtype=fp32_dtype,
        fp16_dtype=fp16_dtype,
        fp32_manifest_checksum=fp32_cs,
        fp16_manifest_checksum=fp16_cs,
        same_manifest_except_dtype=same_manifest,
        per_size=per_size,
        conclusion=conclusion,
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fp32 vs fp16 Terrain Diffusion benchmark reports.")
    parser.add_argument("--fp32", required=True, help="Path to fp32 benchmark JSON.")
    parser.add_argument("--fp16", required=True, help="Path to fp16 benchmark JSON.")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = compare(Path(args.fp32), Path(args.fp16))
    text = json.dumps(asdict(result), indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
