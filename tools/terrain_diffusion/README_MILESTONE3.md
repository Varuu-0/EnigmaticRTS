# Terrain Diffusion Milestone 3 Tooling

Reproducibility and performance-gate tooling for the Terrain Diffusion sidecar.
These tools pin the external runtime, validate the binary protocol, benchmark
native-scale performance, and run bounded coexistence/stress tests.

All code lives under `tools/terrain_diffusion/` and runs inside the existing
`.venv` (created by `setup_venv.ps1`). No model weights, venvs, cloned
upstream source, or datasets are committed — only metadata, source, and
deterministic fixtures.

## Files

| File | Purpose |
|------|---------|
| `locked_manifest.py` | Locked manifest schema + generator. Inspects the installed upstream commit, model SHA, Python, CUDA/PyTorch, dtype/device, T, and request settings. |
| `protocol.py` | Strict binary protocol decoder for `/terrain` responses (int16 elevation + float32 climate). |
| `tests/fixtures/generate_fixtures.py` | Generates deterministic valid + malformed binary fixtures (no model needed). |
| `tests/fixtures/*.bin` | Committed binary fixtures (7 files). |
| `tests/test_deterministic.py` | 36 deterministic self-tests (schema, protocol, fixtures, gate semantics). No model download required. |
| `bench/run_benchmarks.py` | Native scale=1 benchmark at 128/256/512: cold/warm P50/P95, checksums, payload validation, VRAM/CPU/GPU. |
| `bench/run_stress.py` | Bounded coexistence/stress runner: VRAM headroom, DeviceLost, hitch P95. |
| `bench/compare_dtypes.py` | fp32 vs fp16 comparison from two benchmark JSON reports. |

## Reproducing

### 1. Deterministic self-tests (no server, no model download)

```powershell
$td = "tools\terrain_diffusion"
$py = "$td\.venv\Scripts\python.exe"
Push-Location $td
& $py -m unittest tests.test_deterministic -v
Pop-Location
```

### 2. Generate the locked manifest

```powershell
& $py tools\terrain_diffusion\locked_manifest.py --tools-root tools\terrain_diffusion -o tools\terrain_diffusion\locked_manifest.json
```

### 3. Regenerate fixtures (if needed)

```powershell
& $py tools\terrain_diffusion\tests\fixtures\generate_fixtures.py --out tools\terrain_diffusion\tests\fixtures
```

### 4. Run benchmarks (requires running server)

```powershell
# Terminal 1: start the server (fp32)
.\tools\terrain_diffusion\start_server.ps1

# Terminal 2: run the benchmark
$py = "tools\terrain_diffusion\.venv\Scripts\python.exe"
Push-Location tools\terrain_diffusion
& $py -m bench.run_benchmarks --host 127.0.0.1 --port 8000 --seed 12345 --dtype fp32 -o bench\benchmark_fp32.json
Pop-Location
```

### 5. fp32/fp16 comparison

Start the server with `--dtype fp16` (edit `start_server.ps1` or run directly),
re-run the benchmark with `--dtype fp16`, then:

```powershell
Push-Location tools\terrain_diffusion
& $py -m bench.compare_dtypes --fp32 bench\benchmark_fp32.json --fp16 bench\benchmark_fp16.json
Pop-Location
```

### 6. Full coexistence stress test (30 minutes)

```powershell
& $py -m tools.terrain_diffusion.bench.run_stress `
  --duration-minutes 30 --host 127.0.0.1 --port 8000 `
  --seed 12345 --size 256 --request-interval-s 1 --dtype fp32 `
  --launch-game --game-binary target\debug\er_game.exe `
  --game-args "--earth-scale --terrain-diffusion --terrain-diffusion-stress 1800 --terrain-diffusion-stress-output tools\terrain_diffusion\bench\game_stress_30min.json" `
  --game-report-path tools\terrain_diffusion\bench\game_stress_30min.json `
  --output tools\terrain_diffusion\bench\stress_30min.json
```

## Pinned runtime (as inspected)

- **Upstream commit:** `82a0431281f21a6ec3d691a12ee61525de5b0790`
- **Model revision:** `9ef8030cb805b433b98ec25c5dddefbac07a9e26`
- **Python:** 3.11.0
- **PyTorch:** 2.5.1+cu121 (CUDA 12.1)
- **Native resolution:** 30 m/pixel (from model `config.json`)
- **Server:** Flask, `threaded=False` (serial request handling)
- **Protocol:** int16-le elevation + float32-le interleaved climate (4 channels)

## Exit gate status

| Gate | Evidence |
|------|----------|
| Repeated locked requests produce matching checksums | PASS: 1,512/1,512 valid native-scale requests, one checksum |
| >=1 GiB Vulkan VRAM headroom | PASS: 2.33 GiB minimum headroom of 6 GiB |
| No DeviceLost | PASS: clean game exit, no DeviceLost detected |
| Hitch <=1 ms P95 (game main-thread) | PASS: 0.013 ms P95 over 60,000 retained frame samples |

### Accepted coexistence results (30 minutes, RTX 3060)

The accepted run lasted from `2026-07-21T00:35:48Z` through
`2026-07-21T01:05:53Z`. Sidecar request latency was P50 `27.21 ms`, P95
`55.93 ms`, and P99 `78.37 ms`. Provider main-thread work was P50 `0.006 ms`,
P95 `0.013 ms`, P99 `0.020 ms`, and max `0.233 ms`. The runner fails closed if
the sidecar is unreachable, payloads or checksums drift, the game exits
unsuccessfully, VRAM headroom falls below the floor, DeviceLost appears, or the
game timing report is absent.

Short smoke reproduction:
```powershell
# Terminal 1: start the sidecar
.\tools\terrain_diffusion\start_server.ps1 -Dtype fp32
# Terminal 2: run the coexistence stress
$td = "tools\terrain_diffusion"
$py = "$td\.venv\Scripts\python.exe"
Push-Location $td
& $py -m bench.run_stress --duration-minutes 1.5 --launch-game `
    --game-binary "target\debug\er_game.exe" `
    --game-args "--earth-scale --terrain-diffusion --terrain-diffusion-stress 60 --terrain-diffusion-stress-output bench\game_stress_report.json" `
    --game-report-path "bench\game_stress_report.json" `
    -o bench\stress_coexistence_smoke.json
Pop-Location
```

### Game-side stress mode

The game binary (built with `--features terrain_diffusion`) supports:
- `--terrain-diffusion-stress <seconds>`: run for N seconds, then exit
- `--terrain-diffusion-stress-output <path>`: write JSON report
- `--terrain-diffusion-stress-hitch-ceiling <ms>`: P95 gate ceiling (default 1.0)

The game's `ProviderTimingHistory` times only the Bevy `Update` systems owned
by the Terrain Diffusion plugin (`queue_camera_tiles`, `poll_tile_request`,
`start_tile_request`, `publish_diagnostic`). The async HTTP/inference work
runs on the compute thread pool and is explicitly excluded.

## Notes

- `start_server.ps1` now forwards `--dtype`, `--device`, T via `--kwarg T=`,
  `--no-compile`, `--cache-size`, `--batch-size` to the upstream CLI.
- The report has two distinct hitch metrics that must not be conflated:
  - `hitch_p95_ms` is the **gate value**: the game's main-thread provider P95
    (`game_report.provider_p95_ms`) when a game report is present, or `0.0`
    when the gate is unproven (no game report). This is the value the
    `<= 1 ms` exit gate is evaluated against.
  - `sidecar_jitter_p95_ms` is **diagnostic only**: the P95 of sidecar
    inter-request latency deltas. It reflects HTTP/network/model jitter and
    is never used as the gate.
- The `hitch_source` field identifies whether the gate was evaluated from
  `game_main_thread` or left `unproven` (no game report).
- Per-sample `vram_used_bytes` / `vram_free_bytes` are instantaneous
  point-in-time measurements (carried forward between polls). The top-level
  `peak_vram_used_bytes` / `min_vulkan_headroom_bytes` are running aggregates
  (max used / min headroom) over the whole run.
- Per-sample `gpu_util_pct` / `cpu_load_pct` are the actual sampled values
  from `nvidia-smi` / `psutil` at the most recent poll.
- `checksum_match` is true only when exactly one unique checksum was
  observed (`checksums_unique == 1`); zero requests yields `checksum_match=false`.
- Generated JSON outputs and `locked_manifest.json` are gitignored.
