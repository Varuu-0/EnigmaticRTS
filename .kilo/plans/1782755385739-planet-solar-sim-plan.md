# Planet & Solar-System Simulator — Build Plan (Implementation-Grade)

**Pass scope: CREATE + RUN + SEE THE PLANETS — nothing else.**

A standalone, deterministic, rendered **planet + mini-solar-system simulator**, Planetary-Annihilation-style. A large static home planet (terrain/biomes/water) under a **sun-driven day/night cycle** set in a live mini solar system (sun + planets + moons orbiting). The home planet is the only full-detail world; other bodies are simpler sun-lit spheres.

This plan derives from and stays consistent with `1782703038625-enigmatic-rts-world-plan.md` (the "world plan"). It implements the world plan's build-order items 1–4 (terrain/procgen) **plus** a mini solar system, PA-style day/night, and a scalable camera — exactly enough to **generate, run, and see** the planets. **It does NOT implement any RTS gameplay, UI menus, query APIs, or full test ceremony.** Those are fenced off in §99 "Later."

**Scope of this document:** an ordered, milestone-gated, heavily-granular task list an implementation agent can execute sequentially without drifting. All material design decisions are resolved in §0.

---

## 0. Resolved Decisions (locked — do not re-litigate)

| Area | Decision |
|---|---|
| This pass | **Create + run + see the planets only.** Generate the home planet + mini solar system, run them (deterministic clock/orbits/rotation), and render/explorable via a camera. No menus, no RTS, no query APIs. |
| Home planet | Spherified-cube heightfield **quadtree chunked LOD**; radius **~12000 u (tunable)**; max quadtree depth **~12**; **aggressive** LOD culling (high screen-error threshold, 0.8× merge hysteresis, distance cull, active-chunk cap, ≤4 splits/merges per frame). Static (no runtime terrain editing). |
| Elevation | Deterministic **pure function** of `(seed, dir)`, f64; CPU `fastnoise-lite` + **WGSL port kept in parity** (CI test, tolerance 1e-4). |
| Biomes | Latitude-aware **Whittaker** classification (temperature from latitude+altitude+lapse+noise, moisture from noise+rain-shadow); ~8–10 biome set reused from world plan §5.3. **Crisp GPU fragment-shader** biome classification (Whittaker mirrored in WGSL, parity-tested). Weather/hazard fields are **data only, not simulated.** |
| Caching (best-for-perf) | **CPU memoization cache** over all pure world functions (elevation/biome/water/normal) keyed by quantized direction/nav-cell. **GPU split-octave**: slow low-frequency octaves precomputed per-vertex on CPU (from cache) → vertex attributes; fragment shader only adds cheap high-frequency octaves + biome table. |
| Water | Sea-level fill + lakes in local minima + depth bands; rendered as a **separate sea-level sphere mesh** (depth-band color + sun specular glint) shown through below-sea terrain. |
| Math | **f64 world / f32 render**; spherified-cube projection; **origin-shifting** that works across surface ↔ system scales. |
| Solar system | Seeded `SystemSeed` (ChaCha8) → star params + ~4 planets (tunable 2–8) + moons; **parametric Keplerian** orbits (elliptical; moons orbit planets); body positions are a **pure function of `(SystemSeed, clock)`**; deterministic. |
| Other bodies | Real sun-lit rotating spheres with procedural surface tint + per-body atmosphere (thin/none for rocks). NOT full LOD terrain. |
| Day/night (PA-style) | Home planet axial rotation + axial tilt + orbit → sun direction `= normalize(planetPos − sunPos)`; lit hemisphere + terminator sweep; night side low ambient + starlight. ~3-min default day (tunable). |
| Camera | **Single scalable camera**: surface ↔ planet ↔ system, smooth blended transitions; click/focus-to-body fly-to; origin-shifting at all scales. |
| Lighting/render | Directional sun + hemisphere ambient fill + **cascaded shadow maps (near-field home terrain only)**; far terrain = sun-direction shading. **Rayleigh shell-sphere atmosphere** (analytical, not full raymarch). Sun = bright emissive sphere + glow. Procedural deterministic **starfield** (no asset). No night-side city lights. |
| Entry (minimal, no menu) | Auto-generate from a **seed** (CLI arg / env var / random default) on launch → straight into the planet view. A **Pause** key to stop the clock; a couple bare speed keys optional. Full time-warp UI + TPS readout = Later. |
| Workspace | Create only the 5 planet-side crates: `core`, `world`, `terrain`, `render`, `game`. The 5 RTS crates (`sim/nav/gpu_sim/net/save`) are **deferred** (added by the RTS plan). |
| Validation (this pass) | **Minimal only:** CPU↔WGSL elevation+biome parity, procgen+system determinism, LOD seam, day/night correctness, and "runs at 60 FPS." Full criterion bench suite + audits = Later. |
| Persistence | **None.** Regenerate from seed each session. |

---

## 1. Do-Not-Drift Guardrails (read before every phase)

The implementing agent MUST NOT, in this pass:
- Build any RTS gameplay (units/combat/economy/AI/director/discovery/net/save) or the `sim/nav/gpu_sim/net/save` crates.
- Build weather, animated environmental hazards, or seasonal climate **simulation** (biome weather/hazard fields are data only).
- Build **voxel patches** (caves/subsurface/diggable) — the heightfield↔voxel seam is RTS-later.
- Place resource deposits, authored POIs, or faction nests.
- Start the T4 launch sequence or space/orbital layer.
- Build a **main menu UI**, time-warp UI, TPS readout, or **query APIs** (these are §99 Later).
- Edit terrain at runtime (the planet is static).
- Re-litigate locked decisions in §0.
- Skip the parity/determinism/seam checks — they are acceptance criteria.

If a task seems to require any of the above, stop and re-read §0; the requirement is out of scope for this pass.

---

## 2. Workspace & Crate Map (this pass — planet-side only)

```
enigmatic-rts/              (workspace root; E:\Projects\EnigmaticRTS)
  Cargo.toml                # workspace + [workspace.dependencies]
  crates/
    core/      # IMPLEMENT: shared types, math, seed/RNG (ChaCha8), config
    world/     # IMPLEMENT: elevation, biomes, water, planet params, solar-system gen, Kepler orbits, query cache
    terrain/   # IMPLEMENT: heightfield quadtree LOD, chunk mesh gen, skirts/edge-stitch, culling, aggressive LOD controller
    render/    # IMPLEMENT: materials, biome/triplanar shaders, atmosphere, ocean, starfield, sun, shadows, body renderer
    game/      # IMPLEMENT: plugin composition, states, scalable camera, minimal input (seed + pause), run loop
  assets/shaders/  # *.wgsl (elevation, biome, terrain_displace, atmosphere, ocean, starfield, sun, shadow)
  benches/         # (Later — minimal smoke bench only this pass)
  tests/           # integration: parity, procgen determinism, seam, system determinism, daynight
```

Pinned deps (verify latest at bootstrap; match world plan §2.1): `bevy 0.18` (3d + bevy_dev_tools), `fastnoise-lite 1.1` (f64 feature), `bevy_tracy 0.18` + `puffin` (feature-gated), `glam 0.29`, `rand_chacha` (ChaCha8), `leafwing-input-manager 0.16` (or `bevy_enhanced-input`). Dev profile `opt-level = 1`; release `LTO="thin"`, `codegen-units=1`.

Top-level Bevy states (this pass): `Generating` → `InGame`. (A `MainMenu` is Later; we auto-generate on launch.)

---

## 3. Build Phases (execute strictly in order; each phase's AC must pass before the next)

### Phase 0 — Bootstrap & workspace skeleton (planet-side only)
**Goal:** A compiling Bevy workspace with the 5 planet crates, profiling, deterministic RNG/seed core, and tunable config.

- [ ] 0.1 Verify/record final dependency versions; add a `# Versions` comment block to the root `Cargo.toml`.
- [ ] 0.2 Create root `Cargo.toml`: `[workspace] resolver="2"`, `members = ["crates/core","crates/world","crates/terrain","crates/render","crates/game"]`; `[workspace.package]`; `[workspace.dependencies]` for `bevy`, `glam`, `rand_chacha`, `fastnoise-lite` so crates inherit.
- [ ] 0.3 Create the 5 crate folders, each with `Cargo.toml` (workspace deps + intra-workspace deps: `world`→`core`; `terrain`→`core`+`world`; `render`→`core`+`world`; `game`→all) and `src/lib.rs`.
- [ ] 0.4 In each `lib.rs`: `pub fn version() -> &'static str { "0" }` + a crate-doc comment stating purpose and "planet-creation pass; RTS crates deferred."
- [ ] 0.5 Add `.gitignore` (`/target`, `*.lock` optional).
- [ ] 0.6 `game/src/main.rs`: minimal Bevy app — `DefaultPlugins`, window titled "Planet Solar Sim", a 2D FPS text overlay (from `Diagnostics` or a custom frame counter), a clear-color background.
- [ ] 0.7 Add `bevy_tracy`+`puffin` behind a `profiling` feature in `game`; start the tracy proxy when enabled; (manual) verify it connects.
- [ ] 0.8 In `core`: add `rand_chacha` + `glam`. Implement `core::rng`: a `ChaCha8Rng` newtype wrapper with `from_seed(u64)` and a deterministic `child(seed: u64)`/sub-stream helper.
- [ ] 0.9 Implement `core::seed`: newtypes `Seed(u64)`, `PlanetSeed(u64)`, `SystemSeed(u64)`; `SystemSeed::planet_seed(system, planet_index) -> PlanetSeed` (deterministic via ChaCha8). Stub `SystemSeed::star_params()`/`body_count()` → `unimplemented!()` (filled Phase 8).
- [ ] 0.10 Implement `core::config`: `pub const` constants (PLANET_RADIUS_DEFAULT=12000.0, MAX_QUADTREE_DEPTH=12, CHUNK_VERT_RES=17, CHUNK_QUADS_PER_EDGE=16, LOD_SPLIT_BUDGET_PER_FRAME=4, SCREEN_ERROR_THRESHOLD=3.0, MERGE_HYSTERESIS=0.8, MAX_RENDER_DISTANCE=PLANET_RADIUS*8, ACTIVE_CHUNK_CAP=512, FIXED_TPS=30, DEFAULT_DAY_LENGTH_SEC=180.0) + a `Tunables` resource for runtime overrides.
- [ ] 0.11 Set `[profile.dev] opt-level=1`, `[profile.release] lto="thin" codegen-units=1`.
- [ ] 0.12 Verify: `cargo build --workspace` succeeds; `cargo run -p game` opens the window with the FPS overlay; tracy connects (feature on); `cargo test --workspace` passes (no tests yet).

**AC:** Workspace compiles; game window opens with FPS overlay; tracy connects; RNG/seed/config compile; no RTS crates exist.

---

### Phase 1 — Coordinate & math foundation
**Goal:** f64 world math, spherified-cube projection, origin-shifting, and the spherical cell key — no rendering yet.

- [ ] 1.1 `core::math`: define `WorldPos(pub DVec3)` (f64 planet/system space) and `RenderPos(pub Vec3)` (f32 camera-relative); conversions + tests.
- [ ] 1.2 Implement the 6-face tables `face_corner[f]`, `face_u[f]`, `face_v[f]` (±X,±Y,±Z) as `const` arrays.
- [ ] 1.3 `uv_to_dir(face, u, v) -> DVec3`: cube point `= face_corner + u*face_u + v*face_v`; `normalize` → unit sphere dir.
- [ ] 1.4 `dir_to_uv(dir) -> (face, u, v)`: pick the dominant axis → face; compute UV on that face; clamp/handle edge precision.
- [ ] 1.5 Unit test: `uv_to_dir ∘ dir_to_uv` round-trips within 1e-9 across 10k random dirs (all faces).
- [ ] 1.6 Tangent frame: `surface_normal(dir) = dir`; `tangent/dir = d(dir)/du`, `bitangent = d(dir)/dv` via finite difference (ε≈1e-5); orthonormalize. Unit test orthonormal within 1e-6.
- [ ] 1.7 `dir_to_surface(dir, radius, elevation) -> WorldPos`: `planet_center + dir*(radius + elevation)` (elevation stubbed 0 here; wired Phase 2).
- [ ] 1.8 `OriginOffset` resource (`DVec3`) + `world_to_render(pos, origin) -> Vec3` (`(pos - origin) as f32`) + `recenter(origin, camera_world_pos)` that snaps origin to the camera cell.
- [ ] 1.9 `CellKey { face: u8, i: u32, j: u32, lod: u8 }` + `dir_to_cell`/`cell_to_dir`/`cell_center` at a given lod; `cell_size(lod, radius)`.
- [ ] 1.10 Neighbor adjacency across cube-face boundaries (small lookup table); `cell_neighbors(key) -> [CellKey; up to 4]`. Unit test wraps correctly across all 12 face edges.
- [ ] 1.11 Expose `core::math`/`core::seed`/`core::config` to the other crates via workspace deps.

**AC:** All math unit tests pass; round-trip within 1e-9; tangent frame orthonormal; cell adjacency correct across faces; no rendering yet.

---

### Phase 2 — Elevation function + noise (CPU + WGSL parity)
**Goal:** The single deterministic elevation pure function on CPU, plus a WGSL port proven in parity.

- [ ] 2.1 Add `fastnoise-lite` (f64 feature) to `world`.
- [ ] 2.2 `world::elevation::params`: derive from `PlanetSeed` via ChaCha8 — amplitudes A..E (continental, mountains, hills, detail, biome_mod), frequencies, `sea_level`; stored as a `ElevationParams` struct usable as CPU data and as a WGSL uniform.
- [ ] 2.3 `world::elevation::noise`: wrappers over fastnoise-lite (OpenSimplex2 fBm, ridged-multifractal, value noise, domain-warp) seeded from params.
- [ ] 2.4 `elevation(dir: DVec3, params) -> f64`: `continental*A + (mountains masked to continental>0)*B + hills*C + detail*D + biome_mod*E`, domain-warped; pure function of (params, dir).
- [ ] 2.5 Make the noise config fully deterministic (seeded octaves/freqs/amps); no hidden state; same inputs → same output bit-for-bit on CPU.
- [ ] 2.6 Port noise + elevation to WGSL in `assets/shaders/elevation.wgsl` using the FastNoiseLite GLSL port as the shared basis; identical octaves/params; expose a `uniform ElevationParams` mirroring the CPU struct.
- [ ] 2.7 Add a WGSL compute entry `elevation_eval(dirs: storage buffer) -> (elevs: storage buffer)` for the parity test (and later chunk-gen).
- [ ] 2.8 Integration test `tests/parity.rs`: for a fixed `PlanetSeed` + 10k random `dir`s, compare CPU `elevation` vs WGSL `elevation_eval` readback; assert max abs diff ≤ **1e-4**.
- [ ] 2.9 Wire `dir_to_surface` (Phase 1.7) to call `elevation`.
- [ ] 2.10 Criterion smoke bench: CPU elevation evals/sec + a 17×17 chunk elevation cost (baseline only).

**AC:** Parity ≤1e-4 for 10k samples; deterministic per seed; bench baseline recorded; no rendering yet.

---

### Phase 3 — Sphere + heightfield quadtree chunked LOD
**Goal:** A streaming, seam-free, aggressively-culled spherical heightfield you can fly around, GPU-displaced from a flat sphere mesh.

- [ ] 3.1 `terrain::chunk`: `Chunk { key: CellKey, depth: u8, vert_res: u8=17 }` + a `ChunkId`/handle; `ChunkMesh { positions, indices, attrs }`.
- [ ] 3.2 `terrain::quadtree`: per-face `Quadtree` (root = whole face; `split`/`merge` quarter it); an `ActiveChunks` cache keyed by `CellKey` with insert/remove + parent/child lookups.
- [ ] 3.3 Screen-space error metric: `chunk_error(chunk, camera) -> f32` from camera distance + chunk world size (radius + elevation-range estimate). Tune via `SCREEN_ERROR_THRESHOLD`.
- [ ] 3.4 Split/merge rules: split if `error > threshold && depth < MAX_QUADTREE_DEPTH`; merge if all children `error < threshold * MERGE_HYSTERESIS (0.8)`.
- [ ] 3.5 **Aggressive LOD extras:** hard `ACTIVE_CHUNK_CAP` (evict farthest/lowest-error first when exceeded), **distance cull** (drop beyond `MAX_RENDER_DISTANCE`), tunable high threshold; record defaults in config.
- [ ] 3.6 Frame-budget controller: queue split/merge ops; apply ≤ `LOD_SPLIT_BUDGET_PER_FRAME` (4)/frame; carry over the rest; report pending count.
- [ ] 3.7 Chunk mesh gen on CPU via `AsyncComputeTaskPool`: build 17×17 vertices on the **ideal** sphere (flat, undisplaced) + indices + per-vertex `dir` attribute; parallelize; pool/recycle `Mesh`es.
- [ ] 3.8 Pass low-frequency elevation octaves per-vertex from the cache (Phase 4) as attributes here, once Phase 4 lands; for now pass `dir` + a placeholder.
- [ ] 3.9 **Seam fixing — skirts:** drop a vertical skirt of vertices below each chunk edge (always-on baseline).
- [ ] 3.10 **Seam fixing — edge-stitch:** upload chunk corner heights + neighbor LOD depth; in the vertex shader collapse edge vertices toward neighbors via `LOD_delta = neighborDepth - chunkDepth` (mask N least-significant vertex-index bits, N = delta). Higher-detail chunk resolves the mismatch.
- [ ] 3.11 `assets/shaders/terrain_displace.wgsl` vertex shader: displace radially by `elevation(dir, params)` (WGSL); compute world pos + normal (derivatives/analytic); pass `elevation/latitude/moisture_low` to fragment.
- [ ] 3.12 Culling: hierarchical AABB frustum cull per node + **horizon cull** (`dot(chunkCenterDir, cameraDir) < -cos(asp + halfChunkAngle)`, `asp = asin(radius/distance)`).
- [ ] 3.13 LOD debug overlay (bevy_dev_tools, hotkey-toggled): chunk bounds + depth + split/merge events + active-chunk count + pending count.
- [ ] 3.14 Integration test `tests/seam.rs`: headless render adjacent LOD chunks at differing depths; assert no depth-buffer discontinuities (cracks) on shared edges; assert skirts + edge-stitch both pass.
- [ ] 3.15 Smoke bench: chunk-gen time/resolution; active-chunk count vs FPS at surface and orbit with aggressive LOD.

**AC:** Fly a debug camera orbit↔surface; adaptive LOD works; no seams/jitter; horizon+frustum cull far chunks; frame-budget prevents stutter; seam test passes; aggressive LOD bounds chunk count.

---

### Phase 4 — Biome classification + parity + caching layer
**Goal:** Deterministic Whittaker biomes on CPU and WGSL (parity), the memoization cache, and the GPU split-octave optimization.

- [ ] 4.1 `world::biome`: enum (MVP ~8–10: Ocean depth bands, Beach, Grassland, Forest, Jungle, Desert, Tundra, Snow, Mountains/Alpine, ToxicBog, Volcanic) — reuse world plan §5.3.
- [ ] 4.2 `temperature(dir, elevation, seed) -> f64`: latitude (axial tilt + latitude) + altitude lapse + noise. Pure.
- [ ] 4.3 `moisture(dir, elevation, seed) -> f64`: noise + rain-shadow from mountains. Pure.
- [ ] 4.4 `biome(dir, elevation, seed) -> Biome`: Whittaker table (temp×moisture); high-alt → Alpine/Rock; below sea → Ocean depth band. Pure.
- [ ] 4.5 Per-biome data struct: `palette`, `traversability_cost`, `water_behavior`, **data-only** `weather_profile`/`hazard_set` (present, unused). Store in a `BiomeRegistry` loaded from a RON/JSON data file.
- [ ] 4.6 Port to `assets/shaders/biome.wgsl`: mirror temperature/moisture + Whittaker table; reuse WGSL noise from Phase 2.
- [ ] 4.7 Extend `tests/parity.rs`: for fixed seed + 10k dirs, CPU vs WGSL `biome` — assert exact match (or ≤0.1% border mismatch with documented tolerance).
- [ ] 4.8 `world::cache::WorldCache`: keyed by `CellKey`, storing `{ elevation, normal, biome, water_depth, traversability }`; LRU/size-bounded; thread-safe (lock or per-task locals merged); invalidate by seed only.
- [ ] 4.9 Route all CPU elevation/biome/water/normal calls through the cache; smoke bench cache hit-rate + speedup vs uncached.
- [ ] 4.10 **GPU split-octave**: in chunk mesh gen, precompute **low-frequency** octaves (continental + mountains) per vertex on CPU via the cache → vertex attributes `low_freq_elev`, `moisture_low`; vertex shader adds mid/high-freq from WGSL; fragment uses these + Whittaker table. Update parity test to cover the split path.

**AC:** Biome parity passes; cache hit-rate >95% on repeated queries with measured speedup; split-octave keeps parity within tolerance; biome data file loads.

---

### Phase 5 — Water (sea-level fill, lakes, depth bands) + ocean render
**Goal:** Correct water identification + a cheap planet-scale ocean render.

- [ ] 5.1 `world::water::water_depth(dir, elevation, sea_level) -> f64` (negative = submerged depth; positive = land above sea). Pure, cached.
- [ ] 5.2 Lake detection in local minima above sea (basin filled to local spill elev); `is_water(dir) -> bool`, `water_surface_elev(dir) -> f64`. Pure, cached.
- [ ] 5.3 Depth bands (shallow/mid/deep/abyss) for color; data in `BiomeRegistry`/ocean data.
- [ ] 5.4 Ocean render: a separate **sea-level sphere mesh** (lower LOD; static sphere or small quadtree) at `radius + sea_level`; depth-band color + sun specular (Phase 7).
- [ ] 5.5 Clip/hide terrain below sea where the ocean should show (vertex discard or ocean-sphere occlusion); verify no z-fighting (small offset).
- [ ] 5.6 Lakes: small planar water patches at local water elevations (simple).
- [ ] 5.7 `assets/shaders/ocean.wgsl`: depth-band color (from cache-backed attribute or depth texture), sun specular (Phase 7), subtle time-based normal ripple (NOT weather-driven).
- [ ] 5.8 Test: `is_water` consistent with `elevation < sea_level` (+ lake rules); ocean sphere shows where expected; no z-fighting in a headless render snapshot.

**AC:** Oceans fill below sea with depth-band color + sun glint; lakes fill minima; no z-fighting; water identification deterministic + cached.

---

### Phase 6 — Terrain & world rendering (biome shader, atmosphere, starfield, sun, body renderer)
**Goal:** The planet *looks* PA: biome-colored low-poly terrain, atmosphere limb/terminator, starfield, sun disc, and the other bodies as simpler spheres.

- [ ] 6.1 `assets/shaders/terrain_surface.wgsl` fragment: per-pixel Whittaker biome (Phase 4) from passed elevation/latitude/moisture + high-freq detail; sample biome palette → base color; flat/normal-shaded low-poly look; triplanar procedural detail (no UV seams).
- [ ] 6.2 Material: a custom `Material` impl (or `StandardMaterial`-derived) exposing biome/terrain uniforms; integrate with Bevy's `Material` pipeline.
- [ ] 6.3 Beach/waterline blend (sand → shallow water) via biome classification.
- [ ] 6.4 `assets/shaders/atmosphere.wgsl`: shell-sphere mesh (slightly larger than the planet) with **analytical Rayleigh** approximation (not raymarch): sky color from sun dir + view dir, blue limb from space, terminator sunset tones, altitude fade. Driven by sun direction (Phase 9) + atmosphere params from `PlanetSeed`.
- [ ] 6.5 Per-body atmosphere: flag/thickness (thin/none for rocks); reuse the shell shader with scaled params.
- [ ] 6.6 `assets/shaders/starfield.wgsl`: procedural deterministic star points on a far skybox sphere from `SystemSeed` (stable per seed); faint at day, visible at night (tie to sun elevation in Phase 9). No asset.
- [ ] 6.7 Sun render: bright emissive sphere at the sun position (Phase 8/9) + simple additive glow; sun color from star params (temperature → color).
- [ ] 6.8 `render::body::BodyRenderer`: renders sun/home/other uniformly with a `DetailTier { Sun, HomePlanet, SimpleSphere }` enum; other planets/moons = a single low-poly sphere mesh + the body's tint/atmosphere shader (NOT terrain LOD).
- [ ] 6.9 Headless visual snapshot tests: planet renders with biome colors; atmosphere limb visible from orbit; starfield visible on night side; sun disc visible; other bodies render as tinted spheres.

**AC:** Surface = biome-colored low-poly terrain with crisp borders; from orbit a blue atmosphere limb; night side shows stars; sun is a bright disc; other bodies are simpler tinted spheres.

---

### Phase 7 — Lighting & shadows
**Goal:** Sun-driven PBR-ish lighting with near-field shadows.

- [ ] 7.1 `render::light::SunLight` resource: direction `= normalize(homePlanetPos - sunPos)` computed each frame in f64 → f32 (Phase 9 wires the positions; here build the resource + render integration).
- [ ] 7.2 Hemisphere ambient fill (sky color from atmosphere, ground color from biome) for the home planet; low night-side ambient + starlight term.
- [ ] 7.3 **Cascaded shadow maps**, near-field only: a small cascade covering a few hundred units around the camera; render terrain depth from the sun's POV; sample in the terrain fragment; far terrain = sun-direction shading only (distance-gated).
- [ ] 7.4 Shadow acne/peter-panning mitigation (bias + normal offset); tunable.
- [ ] 7.5 Other bodies lit by the same sun direction; no cast shadows.
- [ ] 7.6 Sun specular glint on the ocean (Phase 5.7) tied to sun direction.
- [ ] 7.7 Visual snapshot: terminator sweeps terrain as the planet rotates; near-field shadows cast; night side dark + starlit.

**AC:** Day/night lighting reads correctly; near-field shadows cast; far terrain unshadowed but shaded; perf within budget with shadows on.

---

### Phase 8 — Solar system generation + orbits + system clock
**Goal:** A deterministic mini solar system: star + planets + moons, parametric Keplerian orbits, a system clock.

- [ ] 8.1 `world::system::SystemParams` from `SystemSeed` (ChaCha8): star params (temperature→color, size, luminosity); `body_count` (planets, ~4 default, tunable 2–8); per-planet params (orbital radius, eccentricity, inclination, period, planet radius, axial tilt, rotation period, atmosphere flag+thickness, surface tint, moon count + per-moon params: radius, orbit radius, period).
- [ ] 8.2 Flag exactly **one** planet as the **home** (full-detail) via a deterministic flag in body params; derive its `PlanetSeed` via `SystemSeed::planet_seed` (complete the Phase 0 stubs).
- [ ] 8.3 `body_position(body, clock) -> WorldPos`: parametric Kepler — mean anomaly → eccentric anomaly (Newton/series) → true anomaly → ellipse position; apply inclination; moons orbit their planet (`planet_pos + moon_orbit`).
- [ ] 8.4 `body_rotation(body, clock) -> Quat`: axial tilt + spin about the tilted axis at the rotation period.
- [ ] 8.5 `SystemClock` resource: sim-time seconds; advanced each fixed tick by `dt * time_scale` (time-warp scale; minimal this pass — Pause + 1×). Pure positions over time = f(SystemSeed, clock).
- [ ] 8.6 Determinism test `tests/system_determinism.rs`: fixed `SystemSeed`, sample clocks → **bit-identical** body positions/rotations across runs/threads.
- [ ] 8.7 Store generated `SystemParams` once; bodies query it; no mutation except the clock.
- [ ] 8.8 Smoke bench: whole-system orbit eval/tick (a few dozen bodies).
- [ ] 8.9 Debug "system view" (positions only, simple spheres) to visually verify orbits — temporary, replaced by the real scalable camera in Phase 10.

**AC:** Deterministic mini solar system from a seed; Keplerian orbits + moons visually correct in debug view; determinism test bit-identical; home planet flagged + `PlanetSeed` derived.

---

### Phase 9 — Day/night integration (PA-style)
**Goal:** Home planet rotation+tilt+orbit drive real sun-direction lighting; the terminator sweeps.

- [ ] 9.1 Compute `home_planet_world_pos(clock) = body_position(home, clock)` (Phase 8); `sun_pos = star_position` (star at system origin for simplicity; note in code).
- [ ] 9.2 `sun_dir = normalize(home_planet_world_pos - sun_pos)` in f64 each frame → f32 (feeds Phase 7 light).
- [ ] 9.3 Surface sun angle: at a point with normal `dir`, `illum = max(0, dot(dir, sun_dir_in_planet_frame))` accounting for the planet's current rotation; wire shaders to the rotated frame.
- [ ] 9.4 Implement the home planet's rotation in the render transform (PA approach: camera is surface-fixed, planet rotates under it; document the choice).
- [ ] 9.5 Tie atmosphere + starfield + night-side ambient (Phases 6/7) to sun angle: terminator sunset tones, stars fade in on the night side, night-side low ambient.
- [ ] 9.6 Day/night correctness test `tests/daynight.rs`: fixed seed + clock → the "lit" cell set (`dot(normal, sun_dir) > 0`) matches an independent CPU computation; terminator moves with the clock; sun direction consistent with the orbit position.
- [ ] 9.7 Verify a full day = `DEFAULT_DAY_LENGTH_SEC` (~180s) at 1× (tunable via rotation period).
- [ ] 9.8 Visual: terminator sweeps across terrain; night side dark + starlit; sunrise/sunset atmosphere colors.

**AC:** Day/night physically derived from sun↔planet geometry; terminator sweeps; night side dark + starlit; day-length matches the rotation period; correctness test passes.

---

### Phase 10 — Scalable camera (surface ↔ planet ↔ system)
**Goal:** One camera that seamlessly zooms surface ↔ planet ↔ system, focus-to-body, origin-shifting across scales.

- [ ] 10.1 `game::camera`: single scalable controller with a continuous zoom param mapping to three zones — **surface** (near terrain, surface-up, walk/orbit/dolly), **planet** (orbit the home planet as a sphere), **system** (all bodies + orbit rings) — with smooth blended transitions (lerp across thresholds).
- [ ] 10.2 **Surface-up orientation**: up = surface normal at the camera target, smoothly blended across the sphere (no flips); recompute as the target moves.
- [ ] 10.3 **Origin-shifting across scales**: surface/planet view → origin follows camera/home planet; system view → origin at the barycenter (or focused body); positions are f64, rendered `world - origin → f32`. Transition without precision pops (double-buffer origin, lerp on zone change).
- [ ] 10.4 Controls: zoom (wheel/keys), pan along surface (surface), orbit (drag), edge-scroll (surface). Click/focus a body in system view → camera flies to it (interpolate target + origin). Other bodies viewable up close but not landable.
- [ ] 10.5 System-view **orbit rings** (drawn, toggleable) + seeded body names/labels (toggleable).
- [ ] 10.6 Camera near/far planes adaptive per zone (tight near-plane at surface; huge far-plane at system).
- [ ] 10.7 Test: zoom from a unit above the surface out to the whole system and back without jitter/pops; focus-to-body reaches the sun, planets, moons.

**AC:** Seamless zoom surface → planet → system and back; no jitter/pops; surface-up stable around the sphere; focus-to-body works for every body; origin-shifting transitions cleanly.

---

### Phase 11 — Wire-up + minimal entry + run-it check
**Goal:** The planets actually generate, run, and render on launch with a seed — no menu.

- [ ] 11.1 Seed entry: read seed from CLI arg (`--seed <u64>`), else env var, else a random default; parse to `SystemSeed`.
- [ ] 11.2 `Generating` state: build `SystemParams` + home `PlanetSeed` from the seed (Phase 8); prime the `WorldCache` for the home planet's initial visible chunks; brief loading screen (budgeted, not instant); → `InGame`.
- [ ] 11.3 `InGame`: spawn the system (sun + planets + moons via `BodyRenderer`), the home planet terrain LOD, ocean, atmosphere, starfield; start the `SystemClock`; wire sun direction + lighting (Phases 7/9); attach the scalable camera (Phase 10) to the home planet.
- [ ] 11.4 Minimal controls: **Pause** key (toggles `time_scale` 0↔1); optional 2×/5× speed keys (no full time-warp UI — that's Later). Escape to quit.
- [ ] 11.5 FPS overlay already from Phase 0; add active-chunk count + current sim clock to the overlay (text only — not a TPS-rich UI).
- [ ] 11.6 Run-it check: launch with a fixed seed → a planet + solar system renders, the planet rotates, the day/night terminator sweeps, the camera zooms surface↔system; target **60 FPS** on a mid-high GPU (profile with tracy; record frame times).
- [ ] 11.7 Regenerate-by-relaunch: different seed (arg) → different planet/system; same seed → identical (ties to determinism tests).

**AC:** On launch the planets generate + run + render at ~60 FPS; day/night sweeps; camera works; Pause toggles the clock; determinism holds across relaunches.

---

### Phase 12 — Minimal validation (this pass only)
**Goal:** The small set of tests that prove the planets are correct + deterministic. (Full bench suite + audits = Later.)

- [ ] 12.1 `tests/parity.rs`: CPU↔WGSL elevation ≤1e-4 (10k) + biome parity (exact/≤0.1% border). CI.
- [ ] 12.2 `tests/procgen_determinism.rs`: same `SystemSeed`/`PlanetSeed` → byte-identical `SystemParams`, biome map, water map. CI.
- [ ] 12.3 `tests/seam.rs`: no cracks between adjacent LOD chunks + at depth boundaries; voxel-free (voxel patches out of scope — assert not built). CI.
- [ ] 12.4 `tests/system_determinism.rs`: body positions/rotations bit-identical across runs/threads at sampled clocks. CI.
- [ ] 12.5 `tests/daynight.rs`: lit-cell set matches independent CPU computation; terminator moves with clock; sun direction consistent with orbit. CI.
- [ ] 12.6 `cargo test --workspace` is green; `cargo run -p game -- --seed 12345` runs at ~60 FPS.
- [ ] 12.7 Note the test/run commands in a top-of-file comment in the root `Cargo.toml`.

**AC:** All CI tests green; parity within tolerance; determinism bit-identical; the sim runs at ~60 FPS.

---

## 4. Validation Plan (this pass — minimal)
- **Parity:** CPU vs WGSL elevation ≤1e-4; biome exact/border-tolerant. CI.
- **Determinism:** `SystemSeed`/`PlanetSeed` → byte-identical system + planet across runs. CI.
- **Seam/LOD:** no cracks; aggressive LOD bounds chunk count. Headless render + structural tests.
- **Day/night:** lit-cell set + terminator + sun-direction consistency. CI.
- **Run-it:** launches from a seed, renders at ~60 FPS, day/night sweeps.

---

## 5. Risks & Mitigations
| # | Risk | Mitigation |
|---|---|---|
| W1 | CPU↔WGSL noise/biome parity drift | Shared FastNoiseLite GLSL basis; CI parity test (Phase 2.8, 4.7); split-octave path re-tested. |
| W2 | LOD seam cracks at depth boundaries | Skirts always-on + edge-stitch; headless seam test (Phase 3.14). |
| W3 | f32 precision at 12000 u / system scale | f64 world + origin-shifting across surface↔system (Phase 1.8, 10.3). |
| W4 | Aggressive LOD hurts visual quality | Tunable thresholds + debug overlay (Phase 3.13). |
| W5 | Scope creep into RTS/menu/query APIs/weather/voxel | §1 guardrails; §99 Later list; no RTS crates created. |
| W6 | Solar-system determinism across threads | Pure f(seed,clock) + bit-identical test (Phase 8.6). |
| W7 | Cache memory growth | LRU/size-bounded `WorldCache`; invalidate by seed only (Phase 4.8). |
| W8 | Shadow-map cost on large terrain | Near-field cascade only; far terrain unshadowed (Phase 7.3). |

---

## 99. LATER (explicitly OUT of this pass — do not build now)
Fenced-off work for subsequent passes (kept here so the agent does not drift into them):
- **Main menu UI** (seed field + Randomize + Generate), **regenerate hotkey**, full **time-warp UI** (Pause/1×/2×/5×/10×/100×) + **current-TPS readout**.
- **Home-planet query APIs** for the future RTS (height/normal/biome/water/traversability/LOS) — the RTS integration seam.
- **Full criterion perf bench suite** (cache hit-rate, chunk gen, LOD controller, orbit eval) + regression gating.
- **Aggressive-LOD audit** + **do-not-drift audit gate** milestone.
- **RTS crates** `sim/nav/gpu_sim/net/save` (units, movement, combat, economy, AI, director, discovery, networking, save/load).
- **Voxel patches** (caves/subsurface/diggable) + the heightfield↔voxel seam (world-plan R1).
- **Weather / hazard / season simulation** (biome fields are data only here).
- **Resource deposits / authored POIs / faction nests.**
- **T4 launch sequence + space/orbital layer.**
- **Persistence** (regenerate from seed each session this pass).
- **Full per-chunk GPU biome/elevation texture cache** (deferred optimization).
- **Audio, full UI/UX, accessibility, localization.**

---

## 7. Implementation Note

This is a **plan only**. Execution requires switching to an implementation-capable agent. Begin at Phase 0 and proceed **strictly in order**; each phase's acceptance criteria must pass before the next. Obey the §1 do-not-drift guardrails. This pass = **create + run + see the planets, nothing else** — anything in §99 is explicitly deferred. Keep elevation/biome/orbits **pure functions** of `(seed[, dir][, clock])` and route all CPU queries through the memoization cache. The sim has **no menus, no networking, no save** this pass.
