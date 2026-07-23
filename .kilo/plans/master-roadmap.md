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

**Status: Complete (2026-07-22).** The live hybrid field now uses sphere-native, exact-grid charts (`652` per Earth face; `4` per miniature face), a versioned v2 cache record with SHA-256 payload and BLAKE3 integrity checks, four retained climate channels, bounded RAM/disk caches, atomic persistence, corruption/version rejection, and background promotion. Maintained seam/stress gates measure maximum edge height difference `4.734e-5 m`, edge normal difference `4.596e-3 deg`, and zero eight-corner difference. The full Earth provider atlas maps all `2,550,624` tiles without wrong-face, overflow, or key mismatches, and eviction/reload checksum parity passes for RAM and disk.

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

**Status: Complete (2026-07-22).** The production path uses the six-class bounded/coalescing queue, exact provider/chart coordinates, off-thread decode/validation/persistence, serial-sidecar-aware dispatch, renewable active-demand leases, retries/backoff/health, whole-chunk elevation/normal halo snapshots, targeted chart-footprint rebuilds, per-chart provenance blending, and visual-only learned climate. The corrected RTX 3060 `m5-surface-test-report/v2` gate proved both boundary tiles initially absent from RAM and disk, reached Earth LOD 17 at `4.77197 m` spacing, loaded two exact halo dependencies, completed a monotonic blend and targeted rebuilds, then moved `299.815 m` across the provider boundary with `963/963` camera-local halo-resident frames, `100%` minimum local learned coverage, `0%` maximum local fallback, zero provider failures, zero cross-generation attaches, and provider main-thread P95 `0.254 ms`. The final report is `C:\Users\varun\AppData\Local\Temp\kilo\m5_surface_test_v2_lease_report.json`; all acceptance predicates passed without `DeviceLost`.

## Milestone 6: Biomes, Natural Systems, And Terrain Presentation

### Intent

Milestone 6 replaces the current 14-way hard biome ladder with a deterministic, physically interpretable environmental system. Climate biome, landform, geology, soil, hydrology, cryosphere, and disturbance are independent layers that compose into habitats; `Mountain` and `Plains` are landforms, not climates, so the system can represent montane rainforest, alpine tundra, desert mountains, prairie plains, forested floodplains, and other natural combinations without destructive override rules.

The target is ecological plausibility rather than a full Earth-system simulation. Every field must be explainable, inspectable, seam-safe on the sphere, deterministic from the planet seed, bounded in cost, and useful to both rendering and future RTS queries. Continuous environmental fields and biome memberships are authoritative; discrete names such as `HotDesert` or `TemperateRainforest` are summaries, not the source of truth.

### Role In The Larger Game

Milestone 6 is an initial world-generation and bake milestone, not a runtime environmental simulation. It reconstructs the long-term results of climate, erosion, drainage, soil formation, disturbance history, and ecological competition once, then exposes a mostly immutable `WorldEnvironmentSnapshot`. The apparent interactions explain why terrain and habitats were generated where they are; they do not continuously execute while the player is building an army.

1. Planet seed, planet parameters, and `EnvironmentModelVersion` produce versioned climate, hydrology, substrate, soil, biome, structure, material, and feature atlases during deterministic generation or an explicit offline bake.
2. Normal gameplay performs bounded read-only sampling of the baked fields. Camera movement and terrain LOD may stream representations, but do not regenerate regional ecology or change canonical classifications.
3. Milestone 7 changes illumination and day/night presentation only. It does not move biome boundaries, recalculate climate, or advance seasons.
4. Milestone 9 exposes the M6 snapshot through authoritative height, normal, biome, water, slope, traversability, line-of-sight, placement, hazard, and region queries. Resource and POI generation may use M6 suitability fields but remains a separate deterministic layer.
5. Milestone 10 navigation and simulation consume stable costs, cover, barriers, and region metadata from M6. They must not rerun climate, watershed, soil, or succession models per tick.
6. Milestone 12 saves the seed, model version, canonical atlas/cache identity, and only true gameplay deltas. A saved campaign must not require replaying environmental history or contacting the learned sidecar.
7. Future terraforming, fire, flooding, harvesting, or scripted world changes, if approved, are sparse gameplay overlays on the immutable generated baseline. They do not silently mutate or invalidate the original M6 environment snapshot.

### Non-Negotiable Scope Rules

1. Keep canonical climate, biome, hydrology, soil, feature, and traversability data procedural and seed-derived. Learned climate remains a blended visual input until a persisted canonical bake policy exists.
2. Preserve procedural macro shoreline ownership and derive coast, wetland, river-mouth, and water masks from the same composed field revision as terrain geometry.
3. Sample all global and regional fields in metric sphere space or a finite sphere-native atlas. Face-local noise, chunk-local random placement, and LOD-dependent feature identity are forbidden.
4. Separate potential natural vegetation from landform and transient disturbance. Do not collapse mountains, beaches, snow, volcanism, wetlands, or rivers into one mutually exclusive biome enum.
5. Use physical units at field boundaries: degrees Celsius, millimetres per year/month, metres, slope, catchment area, and normalized material masks with documented conversions.
6. Keep mesh workers snapshot-only and non-blocking. Expensive global solves are versioned background or startup atlas builds; visible chunk sampling is immutable, bounded, and cacheable.
7. Do not add dynamic weather, fluid simulation, vegetation growth, fire spread, animal ecology, resource placement, or terraforming in this milestone. Store static normals, suitability, long-term disturbance-history context, potential habitat, and generated-state masks for later systems.
8. Every subsection follows research -> plan -> implement -> verify and must pass its local gate before the next subsection begins.

### 6.1 Lock The Scientific Model And Baseline

1. Record the current biome pipeline and its known gaps: normalized latitude/noise climate, hard Whittaker-like thresholds, climate/landform conflation, shader-side reclassification, unused registry data, visual learned climate dropped before rendering, and heuristic rather than watershed-derived drainage.
2. Adopt a documented hybrid model:
   - Köppen-Geiger classes are climate diagnostics and regression labels.
   - Holdridge biotemperature and potential-evapotranspiration ratio provide continuous vegetation envelopes and altitudinal belts.
   - A Thornthwaite/Feddema-style moisture index or a documented equivalent provides continuous water stress.
   - Local landform, soil, hydrology, disturbance, and substrate determine the realized habitat within the climate envelope.
3. Define one versioned `EnvironmentModelVersion` covering algorithms, field resolution, constants, biome profiles, and cache compatibility. A model change must invalidate incompatible atlases and golden statistics explicitly.
4. Define an Earth-like reference preset and at least two non-Earth stress presets. Lock expected climate ranges, land/ocean ratio, broad biome distribution bounds, hydrology density, and performance baselines without requiring Earth-identical geography.
5. Add debug views for every driver before art tuning: annual and seasonal temperature, precipitation, PET/aridity, prevailing wind, continentality, elevation, slope, aspect, relief, flow accumulation, wetness, substrate, soil properties, disturbance, top biome memberships, ecotone strength, and final material masks.

#### 6.1 Gate

- Every environmental output has units, range, spatial scale, provenance, update frequency, and canonical/visual authority documented.
- Fixed-seed baseline statistics and screenshots exist before classification or shader replacement begins.

### 6.2 Build Physical Climate Normals

1. Add a `ClimateNormals` sample with annual mean temperature, warmest/coldest-month temperature, growing-season length and heat sum, annual precipitation, driest/wettest-month precipitation, precipitation seasonality, snow fraction, PET, climatic water balance, and aridity/moisture index.
2. Compute deterministic monthly normals or an equivalent bounded harmonic representation from latitude, axial tilt, elevation, environmental lapse rate, land/ocean mask, distance to coast, and seeded regional variation. This is a static climatology and must not depend on Milestone 7's runtime orbital clock.
3. Build a coarse, versioned, sphere-native climate atlas for processes that require neighborhood information. Include oceanic temperature moderation, continental interior seasonality, latitude-band circulation, ITCZ/subtropical-dry/polar tendencies, and deterministic prevailing wind vectors.
4. Advect available moisture inland over the atlas. Apply wind-oriented orographic uplift, precipitation loss, and lee-side drying so the same mountain chain produces wet windward habitats and a coherent rain shadow rather than symmetric noise.
5. Downscale atlas climate to terrain samples using elevation lapse, slope aspect/insolation, topographic exposure, cold-air drainage, and local wetness. Keep local perturbations lower amplitude than macro climate so noise cannot invert global climate structure.
6. Retain learned `VisualClimate` only as a provenance-weighted material variation. It may adjust visual phenology, wetness, snow, and palette within bounded limits, but it must not change canonical biome IDs, hydrology, traversability, or natural-feature placement.
7. Validate against scientific invariants: opposite hemispheres have opposite seasonal phases; elevation generally cools at the configured lapse rate; coastal temperature range is lower than continental interiors; windward precipitation exceeds comparable lee-side precipitation; and warm dry climates have higher evaporative deficit than equally dry cold climates.

#### 6.2 Gate

- Climate is continuous across cube faces/corners and stable across cache eviction, thread count, and LOD.
- Climate diagnostics produce plausible zonal, continental, monsoon-like, maritime, rain-shadow, and elevational patterns on the fixed presets.
- Atlas generation and visible sampling meet explicit time, memory, and cache-hit budgets.

### 6.3 Separate Landform, Geology, And Soil

1. Add a `LandformSample` derived from the composed terrain field: elevation above sea level, slope, aspect, profile/plan curvature, local relief, ruggedness, exposure, valley/ridge position, plateau likelihood, and coastal proximity.
2. Classify landform independently at multiple scales: abyssal plain, continental shelf/slope, coastal plain, beach/dune, floodplain, plain, rolling hill, plateau/mesa, valley, canyon, mountain slope, summit, cliff, karst-like terrain, volcanic edifice, and crater/caldera. Preserve continuous descriptors beside any label.
3. Add broad deterministic geologic/substrate provinces aligned with tectonic and volcanic structure. Expose parent-material properties such as grain size, permeability, erodibility, carbonate/silicate tendency, mineral color, and volcanic age without pretending to simulate full plate tectonics or geochemistry.
4. Add a functional `SoilSample` based on the CLORPT factors available here: climate, potential vegetation/organics, relief, parent material, and deterministic surface age/stability. Track soil depth, texture, drainage, water capacity, fertility, organic content, permafrost, salinity, and bare-substrate fraction.
5. Make soil-landform feedbacks explicit: steep exposed slopes lose soil; stable grasslands build deep organic soils; warm wet old surfaces become deeply weathered; arid closed basins accumulate salts; volcanic ash can be fertile after initial barren succession; saturated cold depressions accumulate peat/permafrost.
6. Derive rock, scree/talus, sediment, sand, mud, peat, salt, and soil exposure masks from these fields. Remove conflicting shader-only guesses or make them consume the canonical fields.

#### 6.3 Gate

- A climatic biome persists across different landforms while its realized habitat and materials change appropriately.
- Soil and substrate fields satisfy boundedness and causality tests and remain continuous at face, chunk, and LOD boundaries.

### 6.4 Build Static Hydrology And Cryosphere

1. Replace the noise-only drainage proxy for biome decisions with a deterministic macro hydrology atlas built from the same sea datum and macro elevation revision as terrain.
2. Perform documented depression handling, sphere-safe multiple-flow routing, catchment/flow accumulation, basin and outlet assignment, and stream ordering on a fixed atlas topology. Tie-breaks must be deterministic and independent of task scheduling.
3. Derive perennial/intermittent channel likelihood, discharge class, river width corridor, riparian influence, floodplain potential, lake/depression, delta/estuary, alluvial fan, groundwater convergence, and topographic wetness. Preserve small-scale procedural drainage only as sub-grid visual detail.
4. Derive wetlands from hydroperiod, water-table proxy, salinity, climate, and vegetation structure. Distinguish marsh, swamp, bog, fen, wet meadow, salt marsh, mangrove, and seasonally flooded grassland where their prerequisites exist.
5. Derive coast type from wave exposure proxy, substrate, slope, sediment supply, climate, and river mouths: sandy/gravel beach, dune, rocky shore, cliff, tidal flat, salt marsh, mangrove, delta, estuary, and fjord-like coast.
6. Add static cryosphere fields: seasonal snow potential, permanent snow, glacier accumulation/ablation balance, glacier-flow corridor, permafrost, patterned-ground potential, and sea/shore ice susceptibility. Snowline and treeline follow climate and moisture, not a global elevation cutoff.
7. Keep water flow and ice evolution static in M6. Rivers, floods, glaciers, and seasonal snow are deterministic potential/state masks; time-dependent simulation is deferred.

#### 6.4 Gate

- Flow accumulation is non-decreasing downstream except at documented splits; river paths terminate at an ocean, lake, or valid closed basin and do not form accidental loops.
- Wetlands occur only under valid hydrologic conditions, riparian influence decays away from channels, and rain shadows alter basin ecology coherently.
- Hydrology is seam-safe, model-versioned, reproducible, and bounded in build/runtime cost.

### 6.5 Replace Hard Biomes With Continuous Ecosystem Membership

1. Introduce orthogonal data contracts such as `ClimateSample`, `LandformSample`, `SoilSample`, `HydrologySample`, `DisturbanceSample`, and `BiomeMixture`. Preserve a dominant summary for compatibility, but never discard the underlying fields.
2. Represent at least the top four ecosystem memberships plus normalized weights, dominance confidence, and ecotone strength. Membership functions use climate and edaphic envelopes with smooth, data-driven response curves rather than a stencil of hard classifications.
3. Build a hierarchical, data-driven catalog with stable IDs and functional traits. At minimum cover:
   - Tropical: lowland rainforest, seasonal/moist forest, dry forest, savanna, thorn scrub, montane/cloud forest.
   - Subtropical and arid: hot desert, cold desert, semi-desert, xeric shrubland, steppe, Mediterranean woodland/shrubland, dune and salt-flat communities.
   - Temperate: prairie/grassland, meadow, deciduous forest, mixed forest, temperate rainforest, conifer forest, shrubland/heath.
   - Boreal and polar: taiga, forest-tundra, shrub/graminoid tundra, polar desert, alpine meadow, fellfield, nival rock, glacier/ice cap.
   - Azonal: riparian woodland, floodplain forest/grassland, marsh, swamp, bog, fen, mangrove, salt marsh, beach/dune, cliff, scree, volcanic pioneer, lava/barren substrate.
   - Aquatic/coastal context: river, lake/littoral, estuary/delta, shallow shelf, deep ocean, and abyssal substrate summaries without replacing the existing terrain-owned water surface.
4. Give each profile climate limits/optima, growing-season requirements, water balance tolerance, soil/substrate affinity, salinity tolerance, flood/fire/frost tolerance, canopy/ground-cover structure, rooting and roughness traits, albedo/roughness/material ranges, and future traversability/hazard metadata.
5. Compute realized structure continuously: tree, shrub, grass/herb, moss/lichen, litter, bare soil, exposed rock, sand, wet ground, snow, and ice cover fractions. A forest-to-grassland transition changes canopy and ground cover gradually instead of changing one color enum.
6. Use Köppen, Holdridge, and biome names as diagnostics over the continuous fields. Add impossible-combination assertions rather than forcing every valid environment into a single textbook category.
7. Replace `ToxicBog` as an Earth-like default with scientifically grounded wetland types. Keep exotic/toxic profiles available only through an explicit non-Earth preset or future hazard layer.

#### 6.5 Gate

- Every catalog ecosystem occurs only when its prerequisite climate, substrate, and hydrology envelope is satisfied.
- Mixture weights are finite, non-negative, normalized, deterministic, and spatially autocorrelated without becoming continent-wide monotony.
- Fixed-seed latitude/elevation transects produce credible tropical-to-polar and lowland-to-nival sequences.

### 6.6 Model Ecotones, Static History, And Cross-Biome Interactions

1. Make ecotones first-class. Transition width depends on the gradient and process: broad climate ecoclines, narrower treelines/snowlines, channel-shaped riparian corridors, tide/elevation-controlled coasts, and disturbance-maintained forest/savanna mosaics.
2. Build a data-driven interaction table whose rules modify continuous memberships and structure, not world coordinates:
   - Orography: windward forest/cloud forest -> treeline/alpine -> dry lee steppe/desert.
   - Rivers: headwater wet meadow -> riparian woodland -> floodplain/wetland -> delta/estuary/mangrove or salt marsh.
   - Arid basins: mountain runoff -> alluvial fan -> oasis/riparian corridor -> playa/salt flat.
   - Fire: dry grass promotes frequent fire; fire suppresses woody closure; wet forest resists fire; post-fire pioneer and regrowth masks create mosaics.
   - Coasts: salinity, inundation, wave exposure, temperature, and sediment choose mangrove, marsh, dune, beach, tidal flat, or cliff.
   - Elevation: lower montane -> montane -> subalpine -> treeline/krummholz -> alpine meadow/fellfield -> nival/glacial, shifted by aspect and maritime/continental climate.
   - Volcanism: fresh lava/ash -> barren pioneer -> grass/shrub -> climate-appropriate mature cover; geothermal wet areas may form local anomalies.
   - Soil/hydrology: wetness enables peat/wetland; poor drainage excludes drought vegetation; shallow rocky soil suppresses closed forest even under suitable macro climate.
3. Add deterministic long-term disturbance potential and generated history/state fields for fire, flood, wind exposure, avalanche, volcanic succession, and glacial disturbance. These are generation inputs that explain the baked vegetation mosaic; they do not evolve during normal gameplay.
4. Generate deterministic biome-region and discrete-feature IDs from seed, feature kind, and canonical atlas cell/anchor, never floating-point camera position. IDs remain stable across LOD, chunk ordering, thread count, and cache eviction.
5. Expose reason codes/contributions in debug tooling so a point can report, for example, `temperate forest limited by shallow soil`, `savanna maintained by dry-season fire`, or `riparian woodland caused by stream order 3`.

#### 6.6 Gate

- Every required interaction has a deterministic unit/integration test and a fixed-seed visual scenario.
- Ecotones remain continuous through learned/procedural visual blending, cube faces/corners, hydrology atlas boundaries, and terrain LOD transitions.

### 6.7 Render The Environmental System

1. Make the renderer consume canonical environment samples instead of independently reclassifying temperature/moisture with different thresholds. Use a bounded representation of dominant memberships, structural cover, landform, substrate, wetness, snow/ice, and feature masks; benchmark vertex attributes versus chunk-local lookup textures/buffers before locking the format.
2. Replace hardcoded duplicate palettes with versioned biome/material profiles shared by CPU diagnostics and GPU upload. Validate asset completeness, stable IDs, physical ranges, and fallback behavior at startup/tests.
3. Add PBR-like material response with physically scaled detail normals and distance-aware filtering. Blend albedo, roughness, specular response, normal strength, macro variation, and cover by material fractions rather than tinting vertex colors.
4. Render natural surface evidence at appropriate scales: grass/herb and litter breakup, canopy-distance darkening, bark/forest-floor tone, exposed bedrock and cliff strata, talus, sediment, dunes/ripples, cracked playa, peat/mud, wetness, river/floodplain deposits, coast foam, snow, firn, glacier ice, lava/ash, and weathering.
5. Keep geometry/detail scale honest. Fade sub-pixel features by footprint and LOD; never allow triplanar noise, normal maps, or vegetation masks to swim, alias, reveal cube faces, or make ocean-classified terrain rough.
6. Add near-field cascaded shadows or an equivalent measured directional-shadow solution. Bias and cascade policy must work from globe to close surface; keep far terrain unshadowed when the visual benefit does not justify cost.
7. Restore atmosphere only if it can be integrated without detached-shell artifacts. Otherwise use terrain/sky horizon scattering that respects sun direction, altitude, and aerial perspective.
8. Preserve smooth LOD and learned/procedural transitions. Environment/material changes must use the same transition revision and may not reintroduce coastline crawl, normal seams, or one-frame biome pops.

#### 6.7 Gate

- Terrain reads as geometry, climate, substrate, soil, water, and vegetation structure rather than colored chunks.
- At close range, visually distinct habitats have distinct material structure; at tactical/orbit range, they merge into coherent regional patterns without high-frequency noise.

### 6.8 Verification, Performance, And Art Review

1. Add CPU and GPU parity tests for every field evaluated on both sides. Prefer one canonical CPU computation plus packed render data where duplicated WGSL logic would create drift.
2. Add deterministic tests across repeated runs, thread counts, cache eviction, atlas serialization, chunk ordering, and supported hardware. Add cache corruption/version rejection tests for every persisted atlas.
3. Add cube-face/corner, same-LOD, mixed-LOD, and source-transition seam tests for climate, biome weights, hydrology, material masks, normals, and discrete feature IDs.
4. Add property tests for climate monotonicities, water balance, flow topology, biome prerequisites, soil causality, ecotone normalization, treeline/snowline order, wetland prerequisites, and downstream continuity.
5. Add statistical fixed-seed gates for land/ocean fraction, climate distributions, biome area/richness, spatial autocorrelation, patch-size distribution, edge density, river density/order, wetland fraction, treeline elevation, and bare/vegetated cover. Use broad scientifically defensible ranges, not an Earth map checksum.
6. Build deterministic screenshot/perceptual scenarios for tropical lowland and montane forests, savanna/forest mosaic, prairie/steppe, hot and cold deserts, rain-shadow transect, temperate and boreal forests, tundra/taiga and treeline, alpine/nival/glacier sequence, river/riparian/floodplain/delta, marsh/bog/mangrove, sandy and cliff coasts, volcanic succession, all cube edges/corners, LOD transitions, and learned/procedural visual transitions.
7. Conduct manual art review at globe, orbit, tactical, and close exploration distances under multiple sun angles. Review both representative interiors and every major interaction/ecotone, not only isolated biome centers.
8. Benchmark climate-atlas build, hydrology-atlas build, cache memory, chunk-generation P50/P95/P99, main-thread provider/streaming work, CPU/GPU frame time, draw count, vertex bandwidth, and shader cost against the Milestone 5 plus LOD-transition baseline.
9. Define budgets before implementation. No unbounded per-chunk feature lists, per-frame global solves, mesh-worker I/O, steady-state material allocation, or hidden feature count growth is accepted. If a detailed field exceeds budget, preserve the canonical coarse field and degrade only render detail by distance.
10. Add a deterministic `--m6-test` state machine and machine-readable report covering local gates, representative screenshots, seam/stability evidence, and performance. Hardware-only claims require a real measured run.

### 6.9 Explicit Deferrals

1. Defer dynamic seasons, weather fronts, storms, snow accumulation/melt, floods, fire spread, succession over simulation time, vegetation growth, and climate change.
2. Defer individual species, plant competition, fauna, food webs, disease, migration, and population ecology. M6 models ecosystem structure and functional traits, not organisms.
3. Defer 3D tree/grass populations, river fluid/mesh simulation, destructible vegetation, harvestable resources, gameplay hazards, pathfinding, and terrain modification unless separately promoted by an approved roadmap revision.
4. Defer learned climate becoming gameplay-authoritative until a canonical bake/version/migration policy is complete.
5. Defer all continuous environmental recomputation during gameplay. Runtime systems read the immutable M6 baseline plus explicit sparse gameplay overlays.

### Research Basis

- Köppen-Geiger climate classification and modern high-resolution maps: Peel et al. (2007), <https://hess.copernicus.org/articles/11/1633/2007/>; Beck et al. (2023), <https://www.nature.com/articles/s41597-023-02549-6>.
- Holdridge life zones, biotemperature, precipitation, and PET ratio: Holdridge (1967), <https://app.ingemmet.gob.pe/biblioteca/pdf/Amb-56.pdf>.
- Comparison and limitations of Köppen, Holdridge, Thornthwaite, and Whittaker systems: Navarro and Tapiador (2024), <https://doi.org/10.1088/2752-5295/ad6632>; Navarro et al. (2025), <https://www.nature.com/articles/s41597-025-04387-0>.
- Ecotones as continuous multi-zone climate envelopes: Navarro et al. (2024), <https://www.nature.com/articles/s41612-024-00581-w>.
- Topographic wetness and sensitivity to flow-routing method: Sørensen et al. (2006), <https://hess.copernicus.org/articles/10/101/2006/>; Kopecký et al. (2021), <https://doi.org/10.1016/j.scitotenv.2020.143785>.
- Global climatic treeline controls: Körner and Paulsen (2004), <https://doi.org/10.1111/j.1365-2699.2003.01043.x>; Paulsen and Körner (2014), <https://doi.org/10.1007/s00035-014-0124-0>.
- Global ecosystem hierarchy and transitional freshwater/coastal systems: IUCN Global Ecosystem Typology 2.0, <https://portals.iucn.org/library/node/49250>.
- Soil-forming factors: Jenny (1941), *Factors of Soil Formation*, <https://soilandhealth.org/wp-content/uploads/01aglibrary/010159.Jenny.pdf>.

### Exit Gate

- The same seed and model version reproduce bit-identical canonical environment summaries, hydrology topology, biome memberships, and feature IDs after cache rebuild.
- The generated `WorldEnvironmentSnapshot` is immutable during normal gameplay, supports bounded read-only world queries, and has a versioned persistence contract for later save files.
- Climate, water balance, landform, substrate, soil, hydrology, cryosphere, disturbance, and ecosystem membership are independently inspectable and compose without mutually exclusive override artifacts.
- Plains, mountains, deserts, forests, grasslands, wetlands, coasts, rivers, volcanic terrain, tundra, and alpine systems show their required internal variation and cross-biome interactions in the review matrix.
- All climate, biome, hydrology, feature, material, source, face, chunk, and LOD transitions are visually continuous and pass automated seam/property gates.
- The deterministic `--m6-test` report passes on measured target hardware with bounded atlas generation, memory, chunk work, frame time, and transient rendering cost.
- Manual review accepts globe, orbit, tactical, and close views as coherent natural landscapes rather than vertex-color chunks or unrelated noise masks.

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
