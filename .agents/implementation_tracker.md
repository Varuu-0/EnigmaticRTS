# EnigmaticRTS Planet Improvement Tracker

Updated: 2026-10-08 17:14 EDT

This tracker is the source-of-truth checklist for the planet visual, runtime, and compile-time improvement work. It records Git checkpoints, task-owned backups, evidence status, and the next implementation gate. It does not claim a change is complete until the listed verification is recorded.

## Milestones

| ID | Milestone | Status | Git checkpoint | Evidence / notes |
|---|---|---|---|---|
| M0 | Existing repository baseline and prior milestone history | complete | `2019d19` (`feat(environment): add versioned climate and soil atlases`) | Workspace history inspected; no task commit existed for the continuation work. |
| M1 | Task bookkeeping and change inventory | complete | `e6842a2` (`chore(process): add planet improvement tracker`) | Tracker committed alone; implementation and unrelated changes remain unstaged. Plan now requires narrow milestone commits and records the screenshot/benchmark oversight. |
| M2 | Mesh generation allocation reduction | measured, committed | `22ca5d4` (`perf(mesh): reduce terrain scratch allocations`) | Fixed 17x17 scratch arrays in `crates/er_terrain/src/mesh_gen.rs`. `cargo bench -p er_terrain --bench chunk_gen_bench -- --noplot` measured depth-6 generation at 786.86 us with Criterion reporting a statistically significant 9.16% improvement; depth 0 showed no significant change and depth 12 has no matched prior comparison. This is not a 2x result. Tests and deterministic mesh checksums remain passing. |
| M3 | Benchmark truthfulness and low-altitude coverage | implemented, runtime convergence blocked | `289a765` (`test(bench): track terrain queue drain`) | Added 10 km/1 km/100 m/25 m scenarios, p99 reporting, convergence checks, failure marker, and per-scenario pending peak/minimum telemetry. Focused tests pass. Runtime evidence still measures before convergence. |
| M4 | Screenshot evidence separation and convergence | partial | none | Terrain-only fixed coverage exists; full-scene capture path still needs explicit command/scenario coverage. `origin_shift_after` timeout remains a required failure to diagnose, not hide. |
| M5 | Shader visual refinement | provisional/unverified | none | Terrain/ocean shader edits are present but have no matched live capture, active-path proof, or GPU timing evidence. Treat as provisional. |
| M6 | Streaming/origin-shift convergence | queue diagnostics committed, diagnosis pending | `d1b1c3d` (`perf(telemetry): split terrain mesh backlog`) | Terrain debug state and benchmark reports now separate async mesh tasks from queued descriptors. Focused benchmark tests pass. Next runtime run will identify whether close-range backlog is worker saturation or requeue admission pressure. |
| M7 | Compile iteration measurement | implemented, cache-backed iteration verified | `d5eb2c4` (`chore(process): record scope and build gates`) | Repository-local `sccache` wrapper is enabled. Cached workspace build completed in 0.83 s; prior evidence also records no-op 0.73 s, game-only 6.76 s, shader-only 6.90 s, shared-world 7.17 s, and feature build 15.31 s. These are cached iteration measurements, not clean-build timings. |
| M8 | Integrated acceptance and user visual review | not started | none | Requires workspace tests, feature build, mesh checksums, corrected benchmark, full-scene captures, soak, and matched before/after pixel review. |

## Current Worktree Inventory

### Task-owned or provisional changes

- `crates/er_terrain/src/mesh_gen.rs`: fixed-size stack scratch arrays; backed up at `backups/terrain-perf-20261008/mesh_gen.rs` and `.agents/continuation_backups/20261008_171158/` for relevant harness files.
- `crates/er_terrain/assets/shaders/terrain_fragment.wgsl`: provisional macro/strata/snow/atmospheric additions; visual and active-path validation pending.
- `crates/er_terrain/assets/shaders/ocean_fragment.wgsl`: provisional ocean depth/wave/Fresnel additions; active-path validation pending.
- `.cargo/config.toml`: repository-local `sccache` compiler wrapper plus existing `rust-lld.exe` linker; backed up at `.agents/continuation_backups/20261008_180504/config.toml`.
- `.agents/`: evidence, backups, and this tracker. Do not delete prior evidence.
- `.perf-backup/` and `backups/`: preserved rollback material.

### Explicitly unrelated

- `.kilo/Rules.md` has pre-existing changes of unknown provenance. Do not rewrite, stage, or commit it as part of this task.

## Validation Ledger

- `cargo test --workspace`: passed after the benchmark convergence changes, including CPU/WGSL parity, cube seams, climate, landform, streaming integration, and shader parity tests.
- `cargo bench -p er_terrain --bench mesh_pipeline_bench`: passed; deterministic checksums matched. Timing samples are mixed and do not establish a general speedup.
- `cargo build -p er_game --features terrain_diffusion`: passed in approximately 6m10s in the recorded environment.
- Screenshot harness: known non-convergence; `origin_shift_after` timed out after 600 frames with 252 pending meshes. Exit code 0 was insufficient evidence; the telemetry timeout must remain a failure for acceptance runs.
- Benchmark harness: convergence failures now produce a machine-readable `BENCHMARK_FAILED` marker; low-altitude scenarios and p99 reporting are implemented. Runtime evidence at `.agents/evidence/bench_20261008_1722.txt` measured all scenarios before convergence, with roughly 250 pending meshes even at far orbit. The process returned code 0 despite `AppExit::error`, so callers must inspect the marker/report until shell exit propagation is separately fixed.
- Long-warmup evidence: `.agents/evidence/bench_20261008_1729_warmup.txt` used 600 warmup / 120 measure frames and still failed convergence. Far orbit reached pending min 0 and final 95, but close scenarios held pending minima around 242–249. Active chunks grew from 774 at far to 11,928 at 25 m.
- Queue-category evidence: `.agents/evidence/bench_20261008_1734_queue_categories.txt` showed close-range final backlogs dominated by queued descriptors, not async workers: approximately 237–244 queued versus 8–13 async tasks. This identifies admission/requeue pressure as the leading bottleneck.
- Close-range performance: the queue-category run averaged 11.15 ms at 1 km, 18.21 ms at 100 m, and 23.69 ms at 25 m. The run still emitted `BENCHMARK_FAILED`.
- Streaming experiment: a reversible breadth-cap increase from 16 to 64 was attempted after backup `.agents/continuation_backups/20261008_173554/systems.rs`, but its dependency rebuild stalled before tests completed. The cap was restored; no runtime improvement is claimed and no cap change remains in the worktree.
- Workspace regression gate: passed before the cap experiment; the cap experiment itself was not accepted.
- Baseline runtime report: `.agents/evidence/rts_baseline_20261008_1814.txt` used the fixed default seed, 600 warmup frames, and 120 measured frames across far through 25 m scenarios on the RTX 3060 Laptop GPU/Vulkan. It emitted `BENCHMARK_FAILED` because every measured scenario still had pending mesh work. Key frame-time averages were 10.00 ms at 1 km, 14.67 ms at 100 m, and 19.87 ms at 25 m; p99 values were 12.86, 18.20, and 23.52 ms respectively. Close-range pending totals were 142, 144, and 133, with 125–136 queued descriptors and 8–10 async tasks.
- Admission-bound observation: compared with `.agents/evidence/bench_20261008_1734_queue_categories.txt` (approximately 237–244 queued close-range descriptors), the bounded breadth admission change reduced the observed close-range queued backlog by roughly 47–49%. This is a queue-pressure improvement, not yet a 2x frame-time or convergence improvement.
- Breadth dispatch experiment: `.agents/evidence/rts_experiment_breadth32_rejection_20261008_1820.txt` records a rejected 32-slot experiment. At 25 m, average frame time regressed 19.87 -> 24.00 ms (+20.8%) and active chunks grew 11,793 -> 14,931; at 100 m, frame time regressed 14.67 -> 21.06 ms (+43.6%). The 16-slot scheduler was restored from `.agents/continuation_backups/20261008_181755/systems.rs`.
- Strict admission milestone: `.agents/evidence/rts_experiment_admission64_summary_20261008_1830.txt` records the accepted non-focus ceiling of 64 outstanding descriptors. At 25 m it reduced average frame time 19.87 -> 18.59 ms (6.4%), p99 23.52 -> 23.26 ms (1.1%), active chunks 11,793 -> 11,082 (6.0%), and pending meshes 133 -> 54 (59.4%). At 100 m average frame time fell 14.67 -> 13.99 ms (4.6%). `cargo test -p er_terrain --lib` passed 91/91. The benchmark still emitted `BENCHMARK_FAILED`, so this is not a 2x or converged-runtime claim.
- Compile timing provenance: prior `cargo build -p er_game --quiet` no-op wrapper measured 20.52s; repository-local sccache now records no-op 0.73s, game-only touch 6.76s, shader-only touch 6.90s, shared-world touch 7.17s, and `terrain_diffusion` feature build 15.31s. Full evidence is `.agents/evidence/compile_20261008_1808.txt`; these are cached iteration measurements, not clean-build timings.
- Long-warmup follow-up: a 2,000-frame / 60-frame measurement run with admission64 was started but canceled while still in the altitude_100m scenario after the 1 km case remained non-converged with 62 pending meshes. The harness writes reports only on completion, so no evidence file was produced and the run is invalid for comparison.
- Duplicate-request coalescing experiment: exact queued-descriptor coalescing was tested and rejected after matched runtime evidence showed a frame-time regression; see `.agents/evidence/rts_experiment_coalesce_20261008_1840.txt`. The source was restored from `.agents/continuation_backups/20261008_183857/systems.rs`.

## Backup and Rollback Ledger

- Mesh scratch rollback: `Copy-Item backups/terrain-perf-20261008/mesh_gen.rs crates/er_terrain/src/mesh_gen.rs -Force` after inspecting later edits.
- Harness backup: `.agents/continuation_backups/20261008_171158/bench.rs` and `screenshot_test.rs`.
- Before any additional source edit, create a new timestamped backup and record it here.
- Never use `git reset --hard` or overwrite unrelated worktree changes.
- Compiler-cache rollback: `Copy-Item .agents/continuation_backups/20261008_180504/config.toml .cargo/config.toml -Force`.

## Next Steps

1. Inspect request producers and add category telemetry for split spawns, stitch rebuilds, learned-field refreshes, and blend follow-ups.
2. Identify repeated equivalent requests or stale requeues before changing any admission limit.
3. Add the smallest producer-side coalescing or stale-request suppression change, with a fresh backup and focused tests.
4. Re-run the fixed-seed baseline and require convergence or document the remaining bounded backlog.
5. Compare frame-time p95/p99, active chunks, queued/async backlog, RAM, VRAM, and draw work against `.agents/evidence/rts_baseline_20261008_1814.txt`.
6. Validate shader active paths and matched pixels; retain or revise only evidence-supported edits.
7. Complete compile evidence for game-only, shared-crate, shader-only, isolated clean, and feature builds with cache/profile provenance.
8. Create a milestone commit per accepted implementation slice and finish with integrated gates; do not claim 2x improvement from compilation or a non-converged run alone.

## Commit Policy

Milestone commits are narrow and auditable. Each accepted implementation commit must:

- stage only files owned by that milestone;
- leave `.kilo/Rules.md` and unrelated files unstaged;
- include verification results in this tracker before or immediately after the commit;
- record rollback instructions;
- avoid claiming unmeasured visual or FPS improvements.

- Source backup before admission-bound edit: `.agents/continuation_backups/20261008_175334/systems.rs`.
- Admission-bound rollback: `Copy-Item .agents/continuation_backups/20261008_175334/systems.rs crates/er_terrain/src/systems.rs -Force`.

- Source backup before breadth-dispatch experiment: `.agents/continuation_backups/20261008_181755/systems.rs`.

- Source backup before duplicate-request coalescing: `.agents/continuation_backups/20261008_183857/systems.rs`.
- Rejected duplicate-request coalescing experiment: `.agents/evidence/rts_experiment_coalesce_20261008_1840.txt` completed the identical 600/120 workload but remained `BENCHMARK_FAILED`. It reduced 25 m pending meshes `133 -> 54` and queued descriptors `125 -> 46`, but regressed average frame time `19.87 -> 22.45 ms` and increased active chunks `11,793 -> 12,798`. Source was restored from `.agents/continuation_backups/20261008_183857/systems.rs`; no coalescing code remains accepted.
- Mesh generation benchmark evidence: `.agents/evidence/mesh_baseline_20261008_1847.txt` records `cargo bench -p er_terrain --bench chunk_gen_bench -- --noplot`. Generation medians were depth 0 `957.82 us`, depth 6 `786.86 us`, depth 12 `781.12 us`; Criterion reported depth 6 `-9.1564%` versus its saved prior sample (`p=0.00`), while depth 0 reported no significant change. Screen-error medians were `52.139 ns`, `54.607 ns`, and `52.898 ns` for depths 0, 6, and 12. `cargo test -p er_terrain --lib` passed 91/91 afterward.
- Workspace regression after mesh milestone: `cargo test --workspace` passed with exit code 0. Terrain unit tests remained 91/91; streaming, parity, seam, climate, landform, and game tests all passed.
- Post-mesh runtime evidence: `.agents/evidence/rts_postmesh_20261008_1849.txt` used the identical 600/120 workload after the committed scratch-array change. It still emitted `BENCHMARK_FAILED`; 25 m averaged `21.36 ms` with 58 pending meshes, versus baseline `19.87 ms` and 133 pending meshes. Active chunks were `11,109` versus `11,793`. This confirms queue reduction but does not establish an end-to-end frame-time improvement or convergence.
- Cached workspace build checkpoint: `cargo build --workspace --quiet` completed successfully in `0.83 s` with the repository-local compiler cache enabled.
- RTS scope blocker: the workspace manifest currently contains only `er_core`, `er_world`, `er_terrain`, `er_render`, and `er_game`. No RTS simulation, navigation, networking, or save crate exists, so a truthful live-RTS baseline and 2x RTS comparison cannot be produced without first adding those systems or explicitly approving a synthetic RTS workload.
- Milestone metadata repair: M2 now points to committed mesh change `22ca5d4`; M7 now points to build/scope ledger commit `d5eb2c4`.
- Scope decision (2026-10-08): continue optimizing the existing planet renderer/terrain runtime only; do not add RTS simulation systems yet. The next acceptance gate is converged terrain streaming plus matched frame-time/draw-work evidence, while the RTS-specific 2x claim remains out of scope until those systems exist.
