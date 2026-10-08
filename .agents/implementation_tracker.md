# EnigmaticRTS Planet Improvement Tracker

Updated: 2026-10-08 17:14 EDT

This tracker is the source-of-truth checklist for the planet visual, runtime, and compile-time improvement work. It records Git checkpoints, task-owned backups, evidence status, and the next implementation gate. It does not claim a change is complete until the listed verification is recorded.

## Milestones

| ID | Milestone | Status | Git checkpoint | Evidence / notes |
|---|---|---|---|---|
| M0 | Existing repository baseline and prior milestone history | complete | `2019d19` (`feat(environment): add versioned climate and soil atlases`) | Workspace history inspected; no task commit existed for the continuation work. |
| M1 | Task bookkeeping and change inventory | complete | `e6842a2` (`chore(process): add planet improvement tracker`) | Tracker committed alone; implementation and unrelated changes remain unstaged. Plan now requires narrow milestone commits and records the screenshot/benchmark oversight. |
| M2 | Mesh generation allocation reduction | verified, not yet committed | pending validation checkpoint | Fixed 17x17 scratch arrays in `crates/er_terrain/src/mesh_gen.rs`. Tests, mesh benchmark checksums, formatting, and Clippy previously passed. This is an allocation reduction, not a proven FPS gain. |
| M3 | Benchmark truthfulness and low-altitude coverage | implemented, runtime run pending | `2000e62` (`test(bench): cover low altitude and tail latency`) | Added 10 km/1 km/100 m/25 m Earth-scale scenarios, p99 reporting, convergence failure handling, and regression tests. Focused benchmark tests pass; runtime benchmark still needs to be executed. |
| M4 | Screenshot evidence separation and convergence | partial | none | Terrain-only fixed coverage exists; full-scene capture path still needs explicit command/scenario coverage. `origin_shift_after` timeout remains a required failure to diagnose, not hide. |
| M5 | Shader visual refinement | provisional/unverified | none | Terrain/ocean shader edits are present but have no matched live capture, active-path proof, or GPU timing evidence. Treat as provisional. |
| M6 | Streaming/origin-shift convergence | blocked on M3/M4 telemetry | none | Identify queue saturation, LOD churn, stale work, origin rebasing, or mesh latency before changing runtime behavior. |
| M7 | Compile iteration measurement | queued | none | Measure no-op, game-only, shared-crate, shader-only, isolated clean, and feature builds with cache/profile provenance. Preserve default debug info. |
| M8 | Integrated acceptance and user visual review | not started | none | Requires workspace tests, feature build, mesh checksums, corrected benchmark, full-scene captures, soak, and matched before/after pixel review. |

## Current Worktree Inventory

### Task-owned or provisional changes

- `crates/er_terrain/src/mesh_gen.rs`: fixed-size stack scratch arrays; backed up at `backups/terrain-perf-20261008/mesh_gen.rs` and `.agents/continuation_backups/20261008_171158/` for relevant harness files.
- `crates/er_terrain/assets/shaders/terrain_fragment.wgsl`: provisional macro/strata/snow/atmospheric additions; visual and active-path validation pending.
- `crates/er_terrain/assets/shaders/ocean_fragment.wgsl`: provisional ocean depth/wave/Fresnel additions; active-path validation pending.
- `.agents/`: evidence, backups, and this tracker. Do not delete prior evidence.
- `.perf-backup/` and `backups/`: preserved rollback material.

### Explicitly unrelated

- `.kilo/Rules.md` has pre-existing changes of unknown provenance. Do not rewrite, stage, or commit it as part of this task.

## Validation Ledger

- `cargo test --workspace`: passed after restoring the default Cargo debug profile.
- `cargo bench -p er_terrain --bench mesh_pipeline_bench`: passed; deterministic checksums matched. Timing samples are mixed and do not establish a general speedup.
- `cargo build -p er_game --features terrain_diffusion`: passed in approximately 6m10s in the recorded environment.
- Screenshot harness: known non-convergence; `origin_shift_after` timed out after 600 frames with 252 pending meshes. Exit code 0 was insufficient evidence; the telemetry timeout must remain a failure for acceptance runs.
- Benchmark harness: currently lacks a pending-work convergence gate and does not yet cover true low-altitude Earth-scale views.
- Shader validation: not yet performed with matched runtime captures or GPU timings.
- Compile timing provenance: prior clean/no-op/touch measurements exist, but the clean timing provenance is uncertain and must not be used as a claim.

## Backup and Rollback Ledger

- Mesh scratch rollback: `Copy-Item backups/terrain-perf-20261008/mesh_gen.rs crates/er_terrain/src/mesh_gen.rs -Force` after inspecting later edits.
- Harness backup: `.agents/continuation_backups/20261008_171158/bench.rs` and `screenshot_test.rs`.
- Before any additional source edit, create a new timestamped backup and record it here.
- Never use `git reset --hard` or overwrite unrelated worktree changes.

## Next Steps

1. Add benchmark convergence state and explicit low-altitude scenarios; add tests for parser/report behavior.
2. Run focused `cargo test -p er_game`, then workspace tests sequentially.
3. Run the corrected benchmark and record raw output under a new `.agents` evidence directory.
4. Diagnose `origin_shift_after` using pending/queued mesh telemetry before changing timeouts or LOD policy.
5. Validate shader active paths and matched pixels; retain or revise only evidence-supported edits.
6. Measure compile iteration with cache/profile provenance.
7. Create a milestone commit per accepted implementation slice, staging only task-owned files and updating this tracker with the commit hash.
8. Finish with integrated gates and a walkthrough report; do not claim visual/FPS completion from compilation alone.

## Commit Policy

Milestone commits are narrow and auditable. Each accepted implementation commit must:

- stage only files owned by that milestone;
- leave `.kilo/Rules.md` and unrelated files unstaged;
- include verification results in this tracker before or immediately after the commit;
- record rollback instructions;
- avoid claiming unmeasured visual or FPS improvements.
