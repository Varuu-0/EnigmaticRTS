# EnigmaticRTS Planet Improvement Tracker

Updated: 2026-10-08 17:14 EDT

This tracker is the source-of-truth checklist for the planet visual, runtime, and compile-time improvement work. It records Git checkpoints, task-owned backups, evidence status, and the next implementation gate. It does not claim a change is complete until the listed verification is recorded.

## Milestones

| ID | Milestone | Status | Git checkpoint | Evidence / notes |
|---|---|---|---|---|
| M0 | Existing repository baseline and prior milestone history | complete | `2019d19` (`feat(environment): add versioned climate and soil atlases`) | Workspace history inspected; no task commit existed for the continuation work. |
| M1 | Task bookkeeping and change inventory | complete | `e6842a2` (`chore(process): add planet improvement tracker`) | Tracker committed alone; implementation and unrelated changes remain unstaged. Plan now requires narrow milestone commits and records the screenshot/benchmark oversight. |
| M2 | Mesh generation allocation reduction | verified, not yet committed | pending validation checkpoint | Fixed 17x17 scratch arrays in `crates/er_terrain/src/mesh_gen.rs`. Tests, mesh benchmark checksums, formatting, and Clippy previously passed. This is an allocation reduction, not a proven FPS gain. |
| M3 | Benchmark truthfulness and low-altitude coverage | implemented, runtime convergence blocked | `289a765` (`test(bench): track terrain queue drain`) | Added 10 km/1 km/100 m/25 m scenarios, p99 reporting, convergence checks, failure marker, and per-scenario pending peak/minimum telemetry. Focused tests pass. Runtime evidence still measures before convergence. |
| M4 | Screenshot evidence separation and convergence | partial | none | Terrain-only fixed coverage exists; full-scene capture path still needs explicit command/scenario coverage. `origin_shift_after` timeout remains a required failure to diagnose, not hide. |
| M5 | Shader visual refinement | provisional/unverified | none | Terrain/ocean shader edits are present but have no matched live capture, active-path proof, or GPU timing evidence. Treat as provisional. |
| M6 | Streaming/origin-shift convergence | queue diagnostics committed, diagnosis pending | `d1b1c3d` (`perf(telemetry): split terrain mesh backlog`) | Terrain debug state and benchmark reports now separate async mesh tasks from queued descriptors. Focused benchmark tests pass. Next runtime run will identify whether close-range backlog is worker saturation or requeue admission pressure. |
| M7 | Compile iteration measurement | implemented, cache-backed iteration verified | pending milestone commit | Repository-local `sccache` wrapper is enabled in `.cargo/config.toml`; a no-op `cargo build -p er_game --quiet` completed in 0.69 s with 70.96% Rust cache-hit rate and no cache errors. Prior uncached-wrapper evidence was 20.52 s. Clean-build and shared-crate timing evidence remain pending. |
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
- Shader validation: not yet performed with matched runtime captures or GPU timings.
- Compile timing provenance: prior `cargo build -p er_game --quiet` no-op wrapper measured 20.52s; repository-local sccache now records no-op 0.73s, game-only touch 6.76s, shader-only touch 6.90s, shared-world touch 7.17s, and `terrain_diffusion` feature build 15.31s. Full evidence is `.agents/evidence/compile_20261008_1808.txt`; these are cached iteration measurements, not clean-build timings.

## Backup and Rollback Ledger

- Mesh scratch rollback: `Copy-Item backups/terrain-perf-20261008/mesh_gen.rs crates/er_terrain/src/mesh_gen.rs -Force` after inspecting later edits.
- Harness backup: `.agents/continuation_backups/20261008_171158/bench.rs` and `screenshot_test.rs`.
- Before any additional source edit, create a new timestamped backup and record it here.
- Never use `git reset --hard` or overwrite unrelated worktree changes.
- Compiler-cache rollback: `Copy-Item .agents/continuation_backups/20261008_180504/config.toml .cargo/config.toml -Force`.

## Next Steps

1. Inspect request producers and refresh queues to identify repeated descriptor requeueing or stale work.
2. Add the smallest bounded coalescing or stale-request suppression change, with a fresh backup and focused tests.
3. Run focused `cargo test -p er_game`, then workspace tests sequentially after any streaming change.
4. Run the corrected low-altitude benchmark and compare queued/async backlog, p95/p99 frame time, active chunks, RAM, and VRAM against `.agents/evidence/bench_20261008_1734_queue_categories.txt`.
5. Validate shader active paths and matched pixels; retain or revise only evidence-supported edits.
6. Complete compile evidence for game-only, shared-crate, shader-only, isolated clean, and feature builds with cache/profile provenance.
7. Create a milestone commit per accepted implementation slice, staging only task-owned files and updating this tracker with the commit hash.
8. Finish with integrated gates and a walkthrough report; do not claim visual/FPS completion from compilation alone.

## Commit Policy

Milestone commits are narrow and auditable. Each accepted implementation commit must:

- stage only files owned by that milestone;
- leave `.kilo/Rules.md` and unrelated files unstaged;
- include verification results in this tracker before or immediately after the commit;
- record rollback instructions;
- avoid claiming unmeasured visual or FPS improvements.

- Source backup before admission-bound edit: `.agents/continuation_backups/20261008_175334/systems.rs`.
- Admission-bound rollback: `Copy-Item .agents/continuation_backups/20261008_175334/systems.rs crates/er_terrain/src/systems.rs -Force`.
