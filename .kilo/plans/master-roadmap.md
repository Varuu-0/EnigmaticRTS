# EnigmaticRTS Master Execution Roadmap

## Purpose

This is the authoritative implementation order after reconciling:

- `1782755385739-planet-solar-sim-plan.md`
- `1782703038625-enigmatic-rts-world-plan.md`
- `earth-scale-natural-terrain-plan.md`
- `terrain-diffusion-integration-plan.md`

It supersedes stale defaults in older plans. It is intentionally milestone-gated: do not begin a later milestone until the stated evidence exists.

## Current Baseline

Completed and committed as of `78c9836`:

- Rust/Bevy workspace with `er_core`, `er_world`, `er_terrain`, `er_render`, and `er_game`.
- Deterministic f64 spherified-cube math, CPU/WGSL parity tests, biomes, water, chunked LOD, skirts, edge stitching, culling, frame diagnostics, and dynamic chunk admission.
- Earth preset: `6_371_000 m` radius, metric terrain sampling, LOD 17, terrain-relative camera floor, and radius-scaled projection distance.
- Feature-gated Terrain Diffusion sidecar with native `scale=1`, validated response decoding, bounded loopback requests, fallback terrain, local 3x3 cache, and setup/launcher scripts.
- Hybrid shoreline correction: procedural global macro field decides land/ocean; learned tiles refine local relief only. The procedural-only ocean sphere is disabled in hybrid mode because it cannot classify CPU-resident learned heights.
- Sun, starfield, clouds, settings menu, basic time progression, and diagnostic screenshots.

Not complete:

- No floating origin/camera-relative terrain mesh; sub-100 m Earth exploration is not yet robust.
- No finite-sphere learned chart contract, disk tile cache, coherent conditioning map, production queue, seam gate, or authoritative tile persistence.
- No multi-body solar system, Keplerian orbits, geometry-derived day/night, scalable system camera, CSM shadows, or body renderer.
- No voxel terrain, world placements, resources, sim/nav/save/network crates, RTS units, economy, discovery, threats, launch sequence, or co-op.

## Non-Negotiable Engineering Rules

1. Keep all authoritative planet and simulation math in f64. Convert only camera-relative positions to f32 at the render boundary.
2. Preserve procedural fallback on every learned-data miss, failure, timeout, cache rejection, and sidecar shutdown.
3. Terrain mesh workers may read immutable snapshots only. They never execute HTTP, Python, disk I/O, blocking locks, or model inference.
4. Terrain Diffusion `scale=1` is native 30 m. Any API upsampling is diagnostic only and must remain labeled non-native.
5. Sea datum, water visibility, normals, material masks, and collision must derive from a single composed field revision. Do not reintroduce separate ocean classification.
6. Terrain refinement has no active-chunk ceiling. Pace work with per-frame split budgets and bounded mesh-job backpressure without evicting coverage.
7. Do not use Vulkan `Immediate` present mode on this Optimus system. Keep `AutoNoVsync`/Mailbox and startup-only window-mode changes.
8. Do not make learned runtime output gameplay-authoritative until it has a persisted canonical cache/bake policy.
9. Do not begin T5+ space/4X content until the T1-T4 launch MVP is accepted.

## Milestone 0: Stabilize The Existing Baseline

### 0.1 Establish reproducible evidence

1. Add a machine-readable baseline manifest: Rust version, Bevy version, GPU adapter, present mode, terrain preset, LOD config, and Terrain Diffusion model/runtime metadata.
2. Add fixed-seed screenshot scenarios for globe, orbit, surface, coastline, mountain, cube edge, and cube corner.
3. Record CPU frame P50/P95/P99, GPU frame time, active/visible/pending chunks, mesh backlog, memory, and tile telemetry for each scenario.
4. Store only selected golden screenshots; keep ad-hoc `screenshots/` ignored.
5. Define explicit target hardware profiles: this RTX 3060 Optimus laptop first, then a desktop reference profile later.

### 0.2 Harden regression automation

1. Make screenshot mode report whether a capture settled or timed out; a timeout must fail an acceptance screenshot job unless explicitly marked exploratory.
2. Add structural tests for projection near/far calculation at miniature, Earth orbit, and Earth close altitude.
3. Add tests for hybrid shoreline classification: learned local relief must not change the procedural global land/ocean mask.
4. Add stress tests for LOD split/merge, root coverage, fast pan, teleport, VSync mode, and bounded mesh backpressure.

### Exit Gate

- `cargo test --workspace` passes.
- Fixed-seed screenshots are reproducible enough for review.
- No terrain hole, shader error, or projection clipping in globe/orbit/surface scenarios.

**Status: Complete (2026-07-20).** `screenshots/milestone0-1-final/` contains the 14-view Earth-scale acceptance matrix, per-scenario telemetry, and baseline manifest for the RTX 3060 Optimus profile. Every scenario settled without timeout or pending mesh work. Telemetry records CPU P50/P95/P99, GPU timing, chunk/mesh state, memory, origin, and source coverage. Projection, shoreline, LOD, root-coverage, screenshot failure, and manifest tests pass in the 261-test workspace suite.

## Milestone 1: Earth-Scale Precision And Surface Camera

### 1.1 Implement floating origin

1. Define `RenderOrigin` as an f64 world-space origin with generation/revision.
2. Split absolute simulation/terrain coordinates from camera-relative render transforms.
3. Shift terrain chunk transforms, sun/body transforms, and camera-relative shader uniforms whenever the camera crosses a configurable meter cell.
4. Ensure async mesh products carry absolute source coordinates and are converted only when attached to render entities.
5. Add generation checks so stale meshes produced before an origin shift cannot be attached incorrectly.

### 1.2 Replace center-orbit-only surface behavior

1. Add a surface target represented by stable direction/chart-local meter coordinates.
2. Support surface pan along the tangent plane, then reproject to the sphere and terrain height.
3. Clamp altitude to composed terrain height plus clearance and slope-safe clearance.
4. Smoothly transition among surface, planet-orbit, and later system-view control regimes without f32 precision jumps.
5. Tie normal sampling spacing to active chunk vertex spacing and field halo requirements.

### 1.3 Validate Earth detail

1. Telemetry must show altitude, nearest chunk width, nearest vertex spacing, normal-difference spacing, source coverage, and render origin.
2. Capture at 10 m, 100 m, 1 km, 10 km, and orbital altitudes.
3. Test camera traversal across cube edges and origin-shift cells.

### Exit Gate

- At 10 m above terrain, stable rendering has <=5 m vertex spacing at the configured close target.
- No camera penetration, jitter, terrain disappearance, or origin-shift pop.
- Far roots continuously cover the globe.

**Status: Complete (2026-07-20).** The floating origin uses f64 absolute coordinates, generation-tagged async mesh anchors, and chunk-local f32 render transforms. Surface panning, slope-safe clearance, origin/cube-edge traversal, transition continuity, and stale-mesh rebasing have focused regressions. In `screenshots/milestone0-1-final/`, the 10 m scenario settled at LOD 17 with `4.77197 m` vertex spacing and no camera penetration; the origin-boundary pair advanced generation 14 to 15 with zero pending or cross-generation mesh attachments.

## Milestone 2: Natural Procedural Fallback

### 2.1 Formalize meter-space field layers

1. Separate `GlobalMacroField`, `ProceduralResidualField`, and composed `TerrainField` APIs.
2. Replace implicit frequency constants with named meter wavelengths: continental, mountain belt, foothill, ridge, and micro detail.
3. Maintain spherical continuity using metric 3D sampling, never face-local discontinuous noise.
4. Keep low-frequency macro elevation separately available for coastlines, broad biome logic, and stable water classification.

### 2.2 Add credible fallback landforms

1. Add seeded tectonic/mountain-belt masks.
2. Add ridges, valleys/canyons, slope-limited talus, erosion intensity, and drainage approximations.
3. Add deterministic brush-like mountain, plateau, crater, canyon, and ridge displacement only after measuring their effect on LOD/seams.
4. Spatially index brushes on CPU; use a capped, parity-tested buffer path if GPU evaluation is needed.
5. Include brush contribution in low-frequency displacement so edge stitching and terrain normals remain correct.

### 2.3 Improve terrain materials

1. Replace normalized-direction grain as primary material detail with meter-scaled triplanar albedo/normal masks.
2. Blend materials from slope, curvature, aspect, drainage, altitude, climate, snowline, and rock exposure.
3. Keep water/coast/wetness tied to composed datum, but use global macro ownership for broad ocean/land decisions.
4. Add coherent atmospheric and directional-light response only after geometry is accepted.

### Exit Gate

- Procedural-only Earth mode has readable ranges, valleys, cliffs, coastlines, and material variation from orbit to 2-10 m altitude.
- Fixed-seed height profiles and slope histograms stay within approved bounds.
- CPU/WGSL parity and LOD seam tests remain green.

**Status: Complete (2026-07-16; revalidated 2026-07-20).** `screenshots/milestone2-final-v16/` contains the reviewed 14-view fixed-seed acceptance matrix. All scenarios settled with zero pending terrain work, zero cross-generation mesh attachments, and 100% procedural source coverage. `cargo test --workspace` passes all 261 tests, including fixed-seed profiles, CPU/WGSL brush parity, and LOD/water seam coverage. Close-view debug performance remains a later optimization target rather than a Milestone 2 gate blocker.

## Milestone 3: Terrain Diffusion Reproducibility And Performance Gate

### 3.1 Lock the external runtime

1. Pin upstream repository commit, model revision/SHA, Python version, CUDA/PyTorch build, dtype, device, request settings, and launcher arguments.
2. Generate and commit a small manifest schema; do not commit model weights, venvs, cloned upstream source, or datasets.
3. Add fixture responses for valid and malformed protocol payloads.
4. Add a benchmark script for 128, 256, and 512 requests at native scale.

### 3.2 Measure coexistence with Vulkan

1. Measure cold/warm P50/P95 latency, response checksum, peak VRAM, CPU load, game frame P95, and `DeviceLost` behavior.
2. Compare fp32/fp16 only after confirming output behavior and repeatability.
3. Run a 30-minute sidecar-plus-game stress scenario on the RTX 3060.
4. Set queue concurrency, prefetch lead, and cache budgets from measurements, not assumptions.

### Exit Gate

- Repeated locked requests produce matching checksums.
- At least 1 GiB Vulkan VRAM headroom remains.
- No DeviceLost and no provider-attributable main-thread hitch above 1 ms P95.

**Status: Complete (2026-07-20).** Project-owned tooling under `tools/terrain_diffusion/` locks and validates upstream commit `82a0431281f21a6ec3d691a12ee61525de5b0790`, model revision `9ef8030cb805b433b98ec25c5dddefbac07a9e26`, Python/PyTorch/CUDA/runtime settings, protocol fixtures, native 128/256/512 benchmarks, dtype comparison, and fail-closed coexistence gates. The final 30-minute RTX 3060 run completed 1,512/1,512 native-scale requests with one stable checksum, `2.33 GiB` minimum VRAM headroom, clean game exit, no `DeviceLost`, and game-measured provider main-thread P95 of `0.013 ms` (`60,000` retained frame samples). Generated reports remain ignored; reproduction commands and measured summaries are in `tools/terrain_diffusion/README_MILESTONE3.md`.

## Milestone 4: Sphere-Native Learned Charts And Cache Format

### 4.1 Replace the diagnostic atlas

1. Keep the current 3x3 cube-face tile window marked diagnostic until this milestone exits.
2. Create `SurfaceChart`, `SurfacePatchId`, `SurfaceRegion`, and versioned chart metadata with tangent basis, bounds, meter scale, halo, and ownership rule.
3. Build persistent 2-10 km global macro elevation/climate that owns continental layout and the global sea datum.
4. Evaluate cube-face ghost-margin mapping and overlapping tangent charts with the same analytic fields.
5. Choose one mapping only after measuring edge and corner value/derivative continuity.

### 4.2 Define versioned cache records

1. Tile key must include world seed, chart projection revision, model revision, conditioning revision, residual revision, datum, dimensions, pixel scale, halo, and request bounds.
2. Store elevation meters, four climate channels, checksum, manifest identity, and creation metadata.
3. Serialize atomically: temp file, fsync where applicable, checksum verify, then rename.
4. Implement bounded RAM LRU plus bounded disk cache, corruption rejection, migration versioning, eviction, and regeneration.

### Exit Gate

- Synthetic fields differ by <1 cm and normals by <0.1 degrees at every face edge/corner and chart overlap.
- Cache eviction/reload produces the same locked tile checksum.
- No runtime code still depends on a four-tile-per-face atlas assumption.

## Milestone 5: Production Learned Streaming

### 5.1 Build the provider pipeline

1. Use one bounded priority queue with request coalescing, timeout, cancellation, exponential backoff, and service health state.
2. Priority order: visible surface, camera-forward corridor, normal halo, prefetch ring, far/root coverage, warmup.
3. Decode and validate off-thread; cache only complete, finite, checksummed payloads.
4. Expose queue depth, resident/pending/failed tile counts, cache hit rate, fallback percentage, latency P50/P95, and rebuild counts.

### 5.2 Integrate without pops

1. A chunk uses learned data only when every elevation and normal halo dependency is resident.
2. Rebuild only intersecting chunks when a tile revision changes.
3. Blend height/material provenance over a defined transition interval; never blend world coordinates.
4. Generate normals, water, and material data from the same composited snapshot.
5. Feed learned climate into visual biome shading only after documented unit conversion; gameplay climate remains canonical procedural until the bake policy exists.

### Exit Gate

- Teleport shows immediate procedural terrain then smooth learned refinement.
- No black chunks, holes, coastline crawl, height step, normal seam, or main-thread hitch.
- Warm tiles stay ahead of normal camera movement on measured hardware.

## Milestone 6: Terrain Presentation And Rendering Quality

1. Add PBR-like material response, physically scaled detail normals, coast foam, wetness, snow, volcanic and sedimentary masks.
2. Add near-field cascaded shadows or an equivalent measured directional-shadow solution; keep far terrain unshadowed when appropriate.
3. Restore atmosphere only if it can be integrated without the detached-shell artifact; otherwise use terrain/sky horizon scattering.
4. Build screenshot/perceptual tests for mountain interiors, valleys, coasts, deserts, snow, volcanics, all face edges/corners, LOD transitions, and learned/procedural transitions.
5. Conduct manual art review at globe, orbit, tactical, and close exploration distances.

### Exit Gate

- Terrain reads as landform geometry and material, not vertex-color chunks.
- All source transitions are visually continuous in the review matrix.

## Milestone 7: Seeded Solar System And Physical Day/Night

### 7.1 System generation

1. Add `SystemParams`, `CelestialBody`, `OrbitElements`, `SystemClock`, and deterministic IDs derived from `SystemSeed`.
2. Generate one star, home planet, several planets, and constrained moons from seeded distributions.
3. Implement pure f64 Kepler position/rotation functions; keep star at origin initially.
4. Add deterministic system-view debug spheres, labels, and orbit rings.

### 7.2 Rendering and lighting

1. Implement a `BodyRenderer` for planet, moon, atmosphere, cloud, and material instances.
2. Derive `sun_dir` from home-planet and star positions each frame.
3. Apply axial tilt and planet rotation to terrain lighting, atmosphere, clouds, and star visibility.
4. Add day/night test: independent lit-cell calculation, moving terminator, and configured day duration.

### Exit Gate

- Same `SystemSeed` and sampled clock values produce bit-identical orbital results.
- Day/night follows geometry rather than a decorative time animation.

## Milestone 8: Surface-To-System Camera

1. Extend camera zones: surface, planet orbit, and system view.
2. Maintain surface-normal up orientation and stable tangent-plane panning.
3. Shift render origin around the camera/home body at surface/planet range and around system barycenter/focus at system range.
4. Add body selection/focus fly-to, adaptive clips, orbit ring toggle, and body labels.
5. Test round trips surface -> planet -> system -> surface with no jitter or origin pop.

### Exit Gate

- Every body can be focused; home terrain remains stable after returning from system view.

## Milestone 9: Planetary World Data And Voxel Foundation

1. Add deterministic region generation for resources, hazards, POIs, faction nests, and weather inputs, lazily cached by seed and region ID.
2. Add authoritative terrain query APIs: height, normal, biome, water, slope, traversability, LOS, and source revision.
3. Implement sparse tangent-plane voxel patches, edit logs, heightfield openings, remeshing, and terrain-query routing.
4. Validate voxel-to-heightfield seams before creating digging enemies or player construction.

### Exit Gate

- A cave can be opened from the surface, edited, rendered, queried, saved, and revisited without an LOD seam.

## Milestone 10: T1 RTS Simulation Core

1. Create `er_sim`, `er_nav`, and `er_save` crates only when their interfaces are ready.
2. Define host-authoritative `FixedUpdate` sets: input, orders, navigation, movement, collision, combat, economy, director, discovery, persistence.
3. Implement core, mass/energy economy, extractors, generators, constructor/build queues, T1 factory, and first land scout/unit.
4. Add spatial hash, terrain collision, spherical flow-field navigation, local avoidance, formations, and deterministic placement/query tests.
5. Profile 1,000 units before adding content breadth.

### Exit Gate

- A player can establish a functioning T1 base and command 1,000 ground units within the tick budget.

## Milestone 11: Discovery, Threats, And T2-T4 Progression

1. Add fog-of-war plus Unknown -> Scanned -> Studied knowledge states.
2. Add scanner energy costs, salvage drops, data-driven tech DAG, resources, POIs, and research unlocks.
3. Add the first two faction archetypes, nests, wave budget, and footprint/storyteller escalation director.
4. Add T2 air/AA/core backup, T3 artillery/shields/detection/voxel engineering, then T4 launch research and launch sequence.
5. Add scripted end-to-end tests: crash -> build -> scan -> fight -> salvage -> research -> launch.

### Exit Gate

- T1-T4 single-player campaign loop reaches the launch milestone with meaningful escalation and recoverable performance.

## Milestone 12: Persistence, Scale, And Co-op

1. Version campaign saves: world seed, macro map, canonical learned tile metadata/checksums, voxel edits, depletion, fog, entities, research, economy, director, and clock.
2. Implement commitment autosave and loss/new-seed flow.
3. Add bake-first learned-world generation; saved terrain reloads with the sidecar off.
4. Profile 1k/10k/50k units. Add GPU movement/collision/targeting only when CPU profiling justifies it.
5. Add host-authoritative Lightyear co-op after save and authority boundaries are stable; interest management derives from fog.

### Exit Gate

- Save/load reproduces campaign terrain and state without the sidecar.
- 50k-unit target is met on the defined hardware tier.
- Two-player drop-in preserves host authority and bounded replication.

## Final Validation Matrix

Every release candidate requires:

- Workspace unit/integration/parity/seam tests.
- Fixed-seed terrain and system determinism tests.
- Screenshot review across all terrain, water, LOD, chart, and lighting boundaries.
- Cold/warm sidecar metrics and 30-minute Vulkan/CUDA stress evidence when learned terrain is enabled.
- Camera/origin-shift traversal checks.
- Save/load and sidecar-off replay checks once campaign persistence exists.
- End-to-end T1-T4 scenario once gameplay work begins.

## Explicit Deferrals

- Model fine-tuning, cube-map model adaptation, ONNX/TensorRT/native inference: only after Milestones 3-6 identify a concrete blocker.
- T5-T10 space/4X gameplay: only after the T4 launch MVP.
- Competitive networking, audio, broad UI/accessibility/localization, naval gameplay, and authored narrative campaigns: separate later plans.
