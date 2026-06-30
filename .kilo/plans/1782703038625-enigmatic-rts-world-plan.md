# EnigmaticRTS — World Plan (Implementation-Grade)

**Vision:** Crashlanded survivors on an unknown, hostile, procedurally generated spherical planet. Unknown-threat survival-discovery: the planet *is* the enigma. Survive, discover, research; at **Tier 4** upgrade your crashed core into a launch vehicle and escape — then escalate orbital → planetary → multi-planetary → solar-system scale (procedurally endless).

**Status:** Greenfield, empty repo at `E:\Projects\EnigmaticRTS`. This plan specifies the **planet phase (Tiers 1–4) as the MVP** in implementation-ready detail; the space/4X layers (T5–10) are sketched for coherence and built later. All material design decisions are resolved; folded items are research-backed defaults an implementer may override.

**Scope of this document:** technical architecture, world/procgen spec, simulation, economy, threats, discovery, networking, persistence, rendering, camera, 10-tier progression map, milestone-gated build order, validation, risks. Not covered: audio, full UI/UX, accessibility, lore content passes.

---

## 1. Resolved Design Decisions (with rationale)

| Area | Decision | Why |
|---|---|---|
| Engine | **Bevy 0.18 (Rust)** | ECS-native, data-oriented; proven to hit 1–2M units via GPU compute (prototype); custom-engine feel without rebuilding renderer/editor. |
| Networking | **`lightyear` 0.26** host-authoritative state-sync + interest management; co-op **later** | Drop-in = send full state on connect; no determinism requirement (critical with procgen/floats/voxels); interest management essential for 50k+ units. |
| Terrain base | Extend **`planetmap`** (Ralith, Rust) spherified-cube streaming LOD | Avoids rebuilding the hardest system; provides 6-cubemap quadtree, GPU buffers, `parry` collision, SIMD noise. |
| Perf tier | Strategic↔Massive (50k–100k+ units). **Massive = the enemy swarm, not the player colony.** | Reconciles survival tension with RTS scale; you're outnumbered by what you don't understand. |
| Terrain model | **Hybrid:** heightfield macro (spherified cube + 6 quadtree chunked-LOD) + **medium-def voxel patches** | Heightfield is fast & pure-function (deterministic); voxels enable caves/burrows/diggable bases/destruction only where needed. |
| Planet scale | **Gameplay-scale traversable sphere** | Circumnavigable over a long session, dense enough to fill with procgen; "leave" = reach orbit via tech. |
| World comp | **Biome/region-driven** (latitude-aware climate bands) | Biomes act as readable signals for the discovery/scouting loop. |
| Theme | Unknown-threat survival discovery; planet = enigma; **escalation director** drives all threats | One engine, four outputs (env + fauna + factions + "planet wakes up") — scope-conscious. |
| Threats | **2–4 procedural-archetype factions** (dig/fly/sneak/tank/swarm) + **authored factions** | Archetypes compose into per-planet factions; each pressures a different defense (subsurface/AA/detection/focus-fire/AoE). |
| Discovery | **Layered:** spatial fog + knowledge fog + **salvage-on-kill** + study POIs | Scan → know stats; kill → faction-specific tech. Can't fully tech up vs a faction without fighting it. |
| Player premise | Crashlanded survivors (Rimworld *premise only*); **PA/TA-style RTS** | No pawn sim; streaming mass+energy; tiered factories; mass-unit formations. |
| Commander | **Crashed core** = construction seed + **T4 escape ship** | Fuses TA-commander + survival premise + escape vehicle; core death pre-redundancy = lose-condition. |
| Economy | **Two-currency:** mass+energy (build) / salvage+study (progress) | Build with world-resources; progress via combat/discovery; extraction feeds escalation. |
| Tech tree | **Tiered backbone, 10 tiers = escalating scale**; salvage-gated specializations; **T4 = escape capstone** | Tiers map to scale jumps (planet→orbit→…→solar system). |
| Failure | **Commitment-style autosave**, fresh planet on loss | Survival tension; no savescum; self-contained campaigns. |
| Art | **Stylized low-poly + procedural/shader textures** + medium-def voxel detail zones | Solo-dev-feasible asset pipeline; instancing-friendly; cohesive heightfield+voxel look. |
| Phasing | **Planet phase (T1–4) MVP first**; space/4X (T5–10) later, procedurally endless | Keeps solo-dev scope survivable while preserving the full vision. |

---

## 2. Technical Foundation

### 2.1 Pinned dependency targets (verify latest at bootstrap)
```toml
[dependencies]
bevy = { version = "0.18", features = ["3d", "bevy_dev_tools"] }   # cargo "collections"; 3d subset
# Networking (co-op phase)
lightyear = { version = "0.26", features = ["serialization", "interpolation", "prediction"] }
# Terrain LOD foundation (extend, possibly fork)
# planetmap = { git = "https://github.com/Ralith/planetmap" }       # pin commit; last push ~Feb 2026
# Procedural noise
fastnoise-lite = "1.1"                                                # OpenSimplex2/Perlin/Cellular/domain-warp, f64 feature
# Compute shader ergonomics (optional wrapper)
bevy_app_compute = "0.18"                                             # or roll raw PipelineCache compute
# Input
leafwing-input-manager = "0.16"                                      # or bevy_enhanced-input
# Save
bevy_save = "0.18"        # or custom serde snapshot
# Profiling
bevy_tracy = "0.18"       # + puffin; frame timelines, system times
# Math
glam = "0.29"             # f32; use f64 wrappers for planet-scale positions
```
Dev profiles: `[profile.dev]` opt-level = 1 on local code (fast recompile) **plus** `[profile.dev.package."*"] opt-level = 3` (compile Bevy + all deps at -O3 in debug → near-release runtime without losing fast local rebuilds — the standard Bevy dev-profile trick; this is what keeps a dev ECS usable); `[profile.stress-test]` inherits release; `[profile.release]` LTO="thin", codegen-units=1 for the sim.

### 2.2 Cargo workspace layout
```
enigmatic-rts/
  Cargo.toml              # workspace
  crates/
    core/                 # shared types, math, seed/RNG (ChaCha8), ids, config
    world/                # procgen: biome/elevation/resources/hazards/POI/factions; deterministic queries
    terrain/              # heightfield LOD + voxel patches + GPU mesh + culling; queries
    sim/                  # ECS components/systems: movement, combat, economy, pathfinding, AI, director
    nav/                  # spherical flow-field pathfinding + spatial hash
    gpu_sim/              # wgpu compute unit-sim hot path + ECS bridge + readback
    render/               # materials, instancing, atmosphere, fog-of-war render, voxel render
    net/                  # lightyear integration (stubbed in MVP; co-op phase later)
    save/                 # serde snapshot + commitment autosave + versioning
    game/                 # plugin composition, states, camera, input, UI glue
  assets/shaders/         # *.wgsl (terrain displacement, biome color, atmosphere, compute unit-sim)
  benches/                # criterion: ECS query, pathfinding, noise, 1k/10k/50k unit tick
  tests/                  # integration: procgen determinism, save round-trip, terrain seam
```

### 2.3 Bevy schedules & system sets
Use `bevy_state` for top-level modes: `MainMenu`, `Worldgen`, `InGame` (sub-states: `SurfacePlay`, `LaunchSequence`, `Orbital`(later)), `Paused`, `GameOver`.

Define a `SimSet` enum (Bevy 0.18 `#[derive(SystemSet)]`) and order via `.chain()` on `FixedUpdate`:
```
InputCapture → Orders → PathfindingRequest → FlowFieldBuild → Movement → Collision → Combat → Economy → Director → Discovery → Persistence(autosave throttle)
```
- **FixedUpdate** runs the authoritative sim at fixed tick (default **30 Hz**; tunable 20–60). Render interpolates between ticks.
- **Render is non-authoritative** and runs on `Update`; it reads sim state (interpolated) and never mutates game truth.
- Use `remove_systems_in_set` to strip netcode systems in pure-SP builds.

---

## 3. Coordinate Systems & Math

### 3.1 Precision: f64 for world, f32 for render
Planet-scale distances break f32. Use **f64** for all planet/position/elevation math and seed-derived coordinates. Convert to **f32** only at the render boundary (camera-relative, origin-shifted).

### 3.2 Spherified-cube projection (the planet surface)
Six cube faces (+X,−X,+Y,−Y,+Z,−Z). For a face-local UV ∈ [0,1]² on face *f*:
```
cube = face_corner[f] + u*face_u[f] + v*face_v[f]      // point on unit cube face
dir  = normalize(cube)                                   // project to unit sphere (spherified cube)
p    = planet_center + dir * (radius + elevation(dir, seed))
```
- Each face = root of a **quadtree**; chunk = quadtree leaf with constant vertex resolution (e.g. **17×17**, 16 quads/edge) but varying world area → LOD.
- Surface tangent frame at any point: `normal = dir`; `tangent`/`bitangent` from d(dir)/du, d(dir)/dv.

### 3.3 Origin shifting (floating-origin)
Keep the active camera near the origin. Maintain a global `f64` planet offset; render positions = `world_pos_f64 - origin_f64` → `f32`. Recompute origin when camera crosses a cell. Eliminates jitter at planetary distance. (planetmap supports this; extend for our sim.)

### 3.4 Surface-local grid (for pathfinding/fog/voxels)
Two grid systems over the sphere:
- **Navigation/fog grid:** aligned with the terrain quadtree chunks (each chunk owns a fixed NxN logical grid of cells); neighbors traverse chunk edges (and cube-face boundaries) via a small adjacency table. Cell size ≈ 2–4 m at fine LOD near the player.
- **Voxel patch grid:** a local tangent-plane block grid (e.g. 16³ or 32³ blocks of voxels) anchored to a surface region; coordinates local to the patch (see §5).

---

## 4. Terrain System

### 4.1 Heightfield macro (spherified cube + quadtree chunked LOD)
**Per chunk:** a 17×17 vertex grid on the *ideal* sphere; GPU vertex shader displaces radially by the elevation function (so CPU uploads flat sphere mesh; GPU adds mountains). Same noise function re-evaluated in fragment shader for crisp biome boundaries on coarse LOD.

**Quadtree LOD selection** (per face, per frame):
- Compute screen-space error for each chunk from camera distance + chunk world size; if `error > threshold` and `depth < maxDepth`, **split** into 4 children; if all children's error < threshold*0.8 (hysteresis), **merge**.
- **Frame budget:** cap chunk splits/merges per frame (e.g. ≤ 4) to prevent stutter; queue the rest.

**Seam fixing (no cracks between LODs):**
- **Skirts:** drop a vertical skirt of vertices below each chunk edge (cheap, robust).
- **Edge-stitching (preferred for flat-shaded low-poly):** upload chunk corner heights; in the vertex shader, collapse extra edge vertices toward neighbors based on the **LOD delta** = `neighborDepth - chunkDepth`; mask the N least-significant bits of the integer vertex index where N = delta (Ralith/planetmap technique). N≥0; the higher-detail chunk handles the mismatch.

**Culling:** hierarchical AABB frustum cull per quadtree node + **horizon cull** (skip chunks behind the planet's curvature: if `dot(chunkCenterDir, cameraDir) < -cos(asp + halfChunkAngle)` where asp = arcsin(radius/distance), skip).

**Mesh generation:** CPU (Bevy `Task`/`AsyncComputeTaskPool`) generates chunk vertex/index buffers in parallel; pool/recycle meshes. planetmap provides the cache manager + GPU-resident buffers; extend for our biomes/shaders.

### 4.2 Elevation function (deterministic, pure, f64)
Single source of truth, evaluated on CPU (game logic) **and** in WGSL (render). Authored once; kept in sync.

```
elevation(dir, seed): f64 =
   continental  * A   // OpenSimplex2, low freq, fBm 4 oct -> landmasses/oceans
 + mountains    * B   // ridged-multifractal, medium freq, masked to continental>0
 + hills        * C   // fBm, high freq
 + detail       * D   // value noise, very high freq
 + biome_mod    * E   // per-biome vertical scale (e.g. dunes, plateaus)
domain-warped (OpenSimplex2 domain warp) for natural coastlines.
sea_level: f64; below = ocean.
```
- Implemented via `fastnoise-lite` (f64 feature) on CPU; **mirrored by hand in WGSL** for the vertex/fragment shaders (FastNoiseLite has a GLSL port — reuse it to guarantee parity). **Test parity** (§validation): same seed/dir must produce identical heights within 1e-4.
- Because it's a pure function of `(seed, dir)`, terrain needs **no replication** over the network — only the seed + voxel deltas.

### 4.3 Biome classification (Whittaker, latitude-aware)
From elevation + latitude → **temperature** (latitude + altitude lapse + noise) and **moisture** (noise + rain-shadow from mountains). Classify via a Whittaker-style biome table:
```
cold/dry -> Tundra;        cold/wet -> Boreal/SnowForest
temp/dry -> Steppe/Desert;  temp/wet -> TemperateForest
hot/dry  -> Desert/Scorched; hot/wet -> Jungle/ToxicBog
high alt -> Alpine/Rock;    below sea -> Ocean (depth bands: shallow/deep/abyss)
```
Each biome declares: traversability cost, resource suitability, hazard set, faction-nest preference, palette/texture set, weather profile. Biome boundaries feed the discovery system (biome = a readable signal).

### 4.4 Voxel patches (medium-def, localized, mutable)
**Purpose:** caves, subsurface resource veins, diggable bunkers/bases, destructible fortifications, burrowing-faction nests.

**Data structure:** per patch, a **chunked voxel volume** anchored to a surface region. Patch = AABB in surface-local tangent space (origin at a surface point, +up = surface normal). Voxel resolution ~0.5–1.0 m ("medium-def": chunky but not Minecraft-blocky). Internally: array-of-blocks `Vec<Block>` (Block = u16 material id + u8 density) chunked 16³ for editing locality. Use a **sparse** map: only patches the player has opened exist in full; unopened subterranean content is procedural-on-demand (generate cave when first dug into).

**Seam integration with heightfield (HARD — §failure modes):**
- A patch "opens" the heightfield: where a patch's top voxels are below the heightfield surface, the heightfield renders a **hole/blended transition** (the patch's voxel surface meets the heightfield). Implementation: mark heightfield chunks overlapping a patch as "patched"; the vertex shader samples the patch height (via a texture/buffer) instead of the base elevation where patched.
- Destruction propagation: editing a voxel marks the patch dirty; re-mesh the affected voxel chunks; mark overlapping heightfield chunks dirty for re-stitch.
- Game-logic terrain query routes: `if point in patch AABB: sample voxel; else: elevation(dir, seed)`.

**State & networking:** voxel edits are **mutable replicated state** (not a pure function). Store as a log of edits (compact: cell-index + new block) → delta-compressed over lightyear for co-op. Save = edit log + opened-patch metadata.

---

## 5. World Procedural Generation

### 5.1 Pipeline (deterministic from a `PlanetSeed`)
```
seed -> RNG(ChaCha8) -> planet params (radius, axial tilt, sea level, atmosphere, biome palette weights, faction roster, hostility baseline)
     -> per-face quadtree of elevation(seed, dir)           # pure function, sampled on demand
     -> climate bands (latitude temp + moisture noise)      # pure
     -> biome classification per region                      # pure
     -> water bodies (sea level fill + lakes in local minima)
     -> surface resource deposits (Poisson-disk, biome-suitability-weighted)
     -> subsurface veins (voxel-patch metadata; materialized on dig)
     -> environmental hazard zones (per biome: storms, toxic, radiation, geothermal)
     -> weather/day-night params (axial tilt -> seasons; biome weather profiles)
     -> faction nests (place per-archetype in preferred biomes; density ~ hostility)
     -> authored POIs (scarce; hand-authored templates instantiated at curated sites)
```
Everything above the voxel edits is **pure / regenerable** — only the seed + voxel edit-log + depletion state + faction state need saving.

### 5.2 Generation execution
- Elevation/biome: sampled lazily as chunks load (never precompute the whole sphere).
- Placements (resources/POIs/nests): computed per-quadtree-region on first visibility (cached), using seeded Poisson-disk + suitability masks. Cached so re-visit is stable.
- Threaded via Bevy task pool; budget-limited per frame.

### 5.3 Biome set (MVP, ~8–10)
Ocean(depth bands), Beach, Grassland, Forest, Jungle, Desert, Tundra, Snow, Mountains/Alpine, ToxicBog, Volcanic. Each with palette, weather, hazards, resource types, faction affinities.

---

## 6. Simulation Architecture

### 6.1 ECS component catalog (illustrative)
```
// Identity & spatial
struct Position(DVec3);          // f64 world
struct Velocity(DVec3);
struct Heading(Quat);
struct FactionId(Entity);        // which faction; player faction is a special id
struct Team(u8);                 // player / enemy-faction-N / neutral
struct SpatialCell(IVec3);       // spatial-hash key (auto-updated)

// RTS/unit
struct UnitTag;
struct Health{ cur:f32, max:f32, armor:f32, shield:f32 }
struct MoveSpeed(f32);
struct Locomotion{ kind: Land|Air|Burrow }
struct Orders(VecDeque<Order>);   // queue: Move/Attack/Build/Gather/Scan/Stop
struct Formation(Entity);         // formation controller entity
struct Production{ queue, progress }
struct Vision{ radius:f32, los:bool }
struct ThreatKnowledge{ state: UnknownContact|Scanned|Studied, stats_known:BitSet }

// Economy
struct ResourceNode{ kind: Mass|Energy, amount:f64, rate:f64 }
struct Extractor{ target_node: Entity }
struct Storage{ mass:f64, energy:f64 }
struct Constructor{ assist_target: Option<Entity>, build_rate:f32 }

// Combat
struct Weapon{ kind, range, damage, rof, projectile|hitscan, aoe, target_domain: Ground|Air|Both }
struct Target(Option<Entity>);

// Discovery
struct Scanner{ range:f32, cost_per_tick:f32 }
struct SalvageDrop{ faction: FactionArchetype, tech_points:f32, chance:f32 }

// Director/threat
struct Aggro{ level:f32 };        // per faction / global
struct Spawner{ archetype, budget, cooldown }
```
Resources: `GameClock`, `Economy{ mass:f64, energy:f64, rates }`, `DirectorState{ threat_curve, next_event_tick, footprint_score }`, `FogGrid`, `FlowFieldCache`, `SpatialHash`, `PlanetSeed`, `OriginOffset`.

### 6.2 Fixed-timestep authoritative sim
- `FixedUpdate` @ 30 Hz advances truth. Movement/combat/economy run here. **Tick rate is the biggest sim-cost lever and is tunable:** Planetary Annihilation ran its authoritative sim at only **10 Hz** (render 60+) to keep CPU cheap; we default to **30 Hz** for responsiveness and can drop toward 10 Hz when CPU-bound. The sim tick is decoupled from render FPS — render only interpolates between ticks, so lowering the tick doesn't lower visual smoothness.
- **Non-bit-deterministic by design** (state-sync): floats, multithreading, unordered iteration all OK because the host is the single source of truth and ships snapshots. We do **not** need SupCom-style state hashing.
- Player commands are timestamped intents queued into `Orders`; the host applies them at the next tick.

### 6.3 Spatial hash broadphase
Uniform 3D grid keyed by `floor(pos / cell_size)`; cell_size ≈ 2× largest unit radius (e.g. 8–16 m). Each tick: rebuild cell→unit lists (parallel), query neighbors for collision/targeting. For 100k+ push the hash onto GPU (§6.6).

### 6.4 Pathfinding: spherical flow-field (per goal)
Per Elijah Emerson (SupCom 2) / AoE4: **cost field → integration field → flow field**. The field is built **per goal**, not per unit → scales with goals, not unit count.

- **Cost field:** per nav cell (aligned to terrain chunks): base traversability (slope from elevation derivative, biome cost, blocked by water/voxel-walls). 1 byte.
- **Integration field:** wavefront (Dijkstra/Eikonal) expanding from the goal cell, `integrated[c] = min(neighbor.integrated) + cost[c]`. 16–24 bit.
- **Flow field:** each cell = 8-way direction toward lowest-integrated neighbor + LOS flag (Bresenham clear-to-goal → steer directly). 1 byte.
- Units sample the flow field at their cell; **blend** across cells for smoothness; add **boid local avoidance** (separation/alignment) for collision.
- **Hierarchical:** coarse sector field (cheap, long-range) → fine per-sector fields near the unit/goal. Recompute only dirty sectors (chunk built/destroyed).
- **Air units:** straight-line steering + obstacle avoidance (no flow field; ignore ground cost, respect altitude ceiling & AA threat map).
- **Burrow faction:** uses voxel-patch connectivity (separate nav graph inside patches) + surface flow field at exits.
- Reference impl: filipkunc's MIT `flowfield` Rust crate (~1.2k LOC) — adapt to spherical chunk grid. Target: <5 ms rebuild for a 256² sector on CPU (benchmarked ~3 ms for 128×80).

### 6.5 GPU compute hot path (when pushing 100k+)
Default (≤ tens of thousands): CPU ECS systems (parallel via Bevy `Task`). When the unit count crosses a threshold (e.g. >30k active), move the **hot path** (movement, collision, spatial hash, targeting) to wgpu compute:
- Unit state lives in GPU **storage buffers** (SoA arrays: positions, velocities, headings, faction, health).
- **Spatial hashing on GPU:** bin units into cells, **bitonic sort** by cell key, compute cell offsets (parallel prefix sum) → O(n log n) but GPU-parallel (proven in the Bevy 1–2M battle sim).
- Compute shaders: `movement.wgsl`, `collision.wgsl`, `targeting.wgsl` dispatched each tick.
- **Render directly from GPU buffers** (instanced indirect draw) to avoid GPU→CPU round-trips for rendering.
- **Selective GPU→CPU readback** via Bevy `Readback` + `ReadbackComplete` only for: units entering a player's vision/interest, deaths (for salvage events), and any state the CPU/AI needs. Keep readback bandwidth-bounded.
- ECS↔GPU bridge: a `Prepare`-stage system writes changed CPU orders into the GPU buffers; a post-compute system reads back the event queue. `bevy_app_compute` simplifies multi-pass + staging buffers; or raw `PipelineCache::queue_compute_pipeline` + `RenderContext` dispatch.
- **Strategy:** CPU ECS is the default and always-correct path; GPU compute is an **optimization tier**, not a day-one dependency — ship CPU-only first, add GPU when profiling demands.

### 6.6 Formations & steering
Formation controller entity holds a shape (line/column/wedge) + unit slot assignments; slots map to flow-field destinations. Steering = flow-field direction (global) + boids (local: separation > alignment > cohesion) + arrival. Mass units share one flow field per order → cheap.

### 6.7 Unit compute budget & ECS layout rules (PA lessons)
Per-unit CPU cost is dominated by **locomotion tier**, not raw count — factor this into archetype cost balance (§9.2):
- **Ground (expensive):** continuous 3D terrain collision (our heightfield/CSG) + unit↔unit micro-collision. Bunched ground units (PA "Dox" swarms) are the primary cause of server **time dilation** (sim dips below real-time). Budget/cap ground-unit density and collision work; coarse-step or throttle when overloaded.
- **Air / orbital (cheap):** free-space steering, no terrain collision → a fraction of ground CPU. A "swarm" of cheap air units is far cheaper than a swarm of ground bots.
- **Burrow:** voxel-patch nav (§6.4) + surface flow-field at exits; cost sits between ground and air.

ECS layout for 10k–100k:
- **Flat, lightweight, contiguous components.** Avoid per-unit `Parent`/`Children` hierarchies — represent sub-parts (e.g. turret yaw) as a float component on the unit entity, not a child entity; hierarchy traversal is expensive at scale.
- **One `Handle<Mesh>` + one `Handle<Material>` per unit type** so Bevy auto-batches/instances draw calls; put per-instance data (tint, animation frame, team color) as a small per-instance index (u32) read in the vertex shader. **Bake animations to VATs (vertex-animation textures), not CPU skeletal skinning** (already in §12.1).
- World positions stay **f64 `DVec3`** on the sphere (do **not** collapse to 2D like a flat-map RTS despite the perf temptation); the **flow-field/nav grid is the 2D-per-chunk layer** (§6.4), and local avoidance can run in the surface tangent plane.

---

## 7. Economy

### 7.1 Streaming mass + energy (TA/SupCom-style)
- **Mass** from extractors on surface deposits + subsurface veins (voxel mining) + a trickle from salvage reprocessing. **Energy** from generators: solar (day-cycle dependent), geothermal (volcanic biomes), biofuel (biomass), thermal. Storage is a **rate buffer**, not a stockpile-and-spend cap: spending is clamped by `min(storage, production_rate - consumption_rate)`; negative net → construction stalls (the TA "stall").
- Each constructor/building has a build cost in mass+energy rates; assisting stacks rates.
- Numbers (tunable): T1 extractor ~+5 mass/s; T1 solar ~+20 energy/s; constructor build ~10 mass/s draw.

### 7.2 Progression currency (salvage + study → research)
- **Salvage:** on kill, per-archetype drop table (e.g. flying → 0–3 "Aero Cores"; swarm → 0–10 "Biomatter"; tanky → 1 guaranteed "Heavy Plate"). Drops are faction-specific **tech materials**.
- **Study:** scanning an authored POI / dissecting a live specimen yields **Data** (research points).
- Tech-tree nodes cost (tech-materials-of-archetype-X + Data). This binds "fight faction X" → "unlock X's tech branch."

### 7.3 Tech tree (structure)
Tiered backbone T1→T10 (scale-mapped, §11). Within each planet-tier (T1–4), specialization branches gated by salvage:
- T1: base land factory, scout, extractor, solar, wall.
- T2: land T2 (heavy tanks), **air factory** (unlocks after encountering/researching the flying faction's aero-cores), AA, geothermal.
- T3: arty, shields, **stealth/detection** (gated by sneaking-faction salvage), bunkers/voxel engineering (gated by digging-faction salvage), swarm-counter AoE.
- **T4 capstone:** orbital-launch research (Data-heavy) + construct launch pad + upgraded core → launch.
- Data structure: a DAG of `TechNode { id, tier, cost: Materials+Data, unlocks: [UnitId|BuildingId|Ability], requires: [NodeId], archetype_gate: Option<FactionArchetype> }`.

---

## 8. Commander / Core

- The **crashed core** is the player's seed unit: builds the first extractor & factory (TA commander pattern), projects a small build radius, has significant HP + a self-defense weapon.
- **Redundancy:** once the player builds a "Core Backup" structure (mid T2), core death is no longer an instant loss (colony continues from backup; backup can reconstruct a core at reduced capacity). Before redundancy → core death = campaign loss.
- **T4 launch:** the core (or a dedicated launch vehicle constructed from core-blueprints) is upgraded with orbital-launch tech; the launch sequence is a dramatic set-piece (the planet's escalation director spikes hardest as you attempt to leave). On success → transition to orbital phase (stub in MVP).

---

## 9. Threats & Escalation Director

### 9.1 The director (one engine, four outputs)
A storyteller-style AI (Rimworld) + footprint-reactive (Factorio pollution) hybrid:
- **Footprint score** (per tick): sum of (mass extraction rate, energy generation, territory held, noise/emissions, units fielded). Drives **reactive aggression**.
- **Threat curve** (storyteller): a desired-threat-over-time function scaled to colony wealth + tier; schedules **proactive events** (raids, surges, environmental crises) to hit the curve — tension even if the player turtles.
- **Outputs:** (a) environmental hazard intensity (storm frequency/severity, toxic spread, geological events), (b) fauna swarm triggers/sizes, (c) faction mobilization (which factions, how aggressive, composition), (d) the "planet wakes up" curve = the meta-shape (escalation steepens with tier; near-T4-launch the planet fights hardest).
- Difficulty parameters are data-driven (Easy/Story/Intense presets) and affect the threat curve + footprint sensitivity.

### 9.2 Faction archetypes (behavior templates + per-planet params)
Each archetype = a behavior template with parametric variation (size, palette, damage type, tech-drop) per planet:
- **Swarm:** many weak melee/ranged; overwhelm via numbers; nests in grasslands/jungle; cheap flow-field pathing. Counter: AoE, chokepoints, walls.
- **Tanky-elite:** few high-HP high-damage units; slow; frontal assault. Counter: focus fire, mobility, artillery, shields.
- **Flying:** air units bypass ground terrain/walls; nest in mountains/cliffs. Counter: AA, airspace control, interceptors.
- **Sneaking:** stealth/cloak; ambush; stay "unknown contact" longer (defeat scanning until close); burrow-ambush. Counter: detection/scanner nets, sensors, perimeter lighting.
- **Digging:** burrow via voxel patches; attack from below; undermine bases. Counter: subterranean sensors, foundation reinforcement, counter-mining.
Composing 2–4 archetypes per planet (seeded) yields a distinct faction identity each campaign. **Authored factions** = rare, hand-tuned encounters at curated POIs (narrative/special mechanics), distinct from the procedural archetypes.

### 9.3 Spawning & AI
- Faction nests are placed in preferred biomes (§5); they spawn waves on a budget/cooldown scaled by director output.
- AI per archetype is a small behavior tree / utility AI: acquire target (respect vision/threat-knowledge), choose attack (rush/flank/ambush/burrow), retreat thresholds. Reuse the flow-field + spatial hash. Keep AI state lean (one goal + target + state enum) for 100k scalability.

---

## 10. Discovery System

### 10.1 Spatial fog-of-war (vision)
- Each unit/building has `Vision { radius, los }`. LOS on the heightfield: sample elevation along the ray; occluded if terrain blocks (cheap raymarch over nav cells). Air units see further / ignore terrain occlusion.
- **Fog grid** (chunked, aligned to nav grid): cells are `Unknown | Explored(dim) | Visible`. Visible = currently in some ally's vision; Explored = seen before (rendered dim/static); Unknown = black.
- For co-op: this grid **is** lightyear interest management — the host only replicates entities in a client's visible/explored-relevant set.

### 10.2 Knowledge fog (the "unknown contact" layer)
Enemy units are not fully known on first sight:
```
UnknownContact -> (scanned by scanner unit/building within range, costs energy+time)
                -> Scanned (stats known: HP/armor/damage/move/special; behavior pattern estimated)
                -> (killed at least N of this faction OR studied a POI about it)
                -> Studied (full: weaknesses, tech-tree branch unlocked, codex entry)
```
- Scanners are a unit/building line (T1 sensor, T2 scanner, T3 orbital-ish) consuming energy; scanning reveals an area and tags contacts.
- The **sneaking faction** resists scanning (cloaks; needs close/active detection) — extends the "unknown" tension.

### 10.3 Salvage drop tables (per archetype)
```
Swarm     -> Biomatter (0–10, chance 0.6), low Data
Tanky     -> HeavyPlate (1, guaranteed), Medium Data
Flying    -> AeroCore (0–3, chance 0.5)
Sneaking  -> StealthNode (0–2, chance 0.4)
Digging   -> TerraSpore (0–2, chance 0.5) -> enables bunker/voxel-engineering tech
Authored  -> unique drops (Data-rich, narrative)
```
Drop tables are data (RON/JSON) for tuning without code changes.

---

## 11. Combat
- **Damage model:** `effective = damage * (1 - armor/(armor+K)) - shield`; shields regen out-of-combat; armor-piercing weapon flag.
- **Weapon domains:** Ground / Air / Both — ties to AA and the flying faction. AoE weapons (arty, swarm counters) with falloff. Hitscan vs projectile (projectiles for visible drama + AA intercept).
- **Targeting:** nearest-threat / highest-threat / lowest-HP heuristics, filtered by weapon domain & range; runs in the combat system using the spatial hash (CPU) or `targeting.wgsl` (GPU).
- **Formations in combat:** hold shape until engagement, then fluid (boids) with role-based spacing.

---

## 12. Rendering

### 12.1 Stylized low-poly + procedural textures
- Low-poly meshes (few hundred–low thousands of tris per unit) with **flat/normal-shaded** stylized look; biome coloring via procedural shader (noise-driven palette + triplanar texturing to avoid UV seams on the sphere). Procedural textures keep the asset pipeline solo-feasible (shader-authored, not hand-painted).
- **Entities Graphics** (Bevy) or custom `RenderContext` indirect instancing for mass units; per-instance data (transform, tint, animation frame) in storage buffers. Bake animations to textures (GPU vertex texture animation) for huge crowds.

### 12.2 Voxel patch rendering
Chunked greedy-meshing of voxel patches (merge same-material faces) into low-poly meshes consistent with the macro style. Re-mesh only dirty chunks on edit.

### 12.3 Atmosphere & lighting
- **Rayleigh-scattering atmosphere** shader on a transparent shell sphere (blue sky from surface, thin glow from orbit, terminator sunset tones) — port from the cuberact reference approach.
- Single directional "sun" (axial tilt → day/night + seasons); shadow maps for near field; fog-of-war overlay render (unexplored = black, explored = dim).

### 12.4 LOD for units
Distance-based LOD: full mesh near, impostor/billboard far, culled when in Unknown fog or offscreen.

---

## 13. Camera & Controls

- **Spherical RTS camera:** pivot around the planet; "up" = surface normal (smoothly blended across the sphere to avoid flips); zoom from surface view → high orbit. Origin-shifted (§3.3).
- Controls: pan along surface (move camera target along terrain), rotate (orbit around target), zoom (dolly). Edge-scroll + minimap-click-to-jump.
- **Selection/control groups** (TA-style): drag-select, box select, control-group assignments, attack-move, patrol, assist, build-queue UI.
- **Minimap:** cylindrical/spherical projection of the explored fog grid; click-to-focus.

---

## 14. Networking (co-op phase — stubbed in MVP)

- **lightyear** server-authoritative; host runs the sim, clients send commands & receive snapshots.
- **Replication config:** marker components `Once` (faction id, type); mutable components `Full` (Position, Health, Orders); high-frequency transforms via snapshot interpolation (10 Hz server updates, client interpolation). **Bandwidth scales with *active* units only:** ship state as curves/keyframes — the host emits a delta only when a unit's state curve changes, so an idle unit sends ~0 bytes (a 100k swarm that's mostly idle costs little). This is exactly what lightyear delta-compression + interest management (next bullet) provide; it is why state-sync bandwidth tracks active units, not total.
- **Interest management = spatial fog:** host replicates only entities in a client's visible/explored-relevant set (reuse the fog grid). Critical for 50k+ (don't send the whole swarm).
- **Prediction/rollback** for the controlling client's own units (lightyear built-in) so commands feel responsive.
- **Drop-in:** on connect, host sends a full relevant-state snapshot (seed + fog grid + visible entities + research + voxel edit-log prefix).
- Build the sim **host-authoritative from day one** (decisions applied at host tick) even while SP-only, so co-op is additive, not a rewrite. Strip netcode systems with `remove_systems_in_set` in SP builds.

---

## 15. Persistence

### 15.1 Save schema
Serialize: `PlanetSeed`, voxel-patch **edit log** (compact), resource-node **depletion** state, faction **state** (aggro, nests HP, spawned counts), `Economy` + research tree state, `FogGrid` (explored cells), active entities (player units/buildings + notable hostiles), `DirectorState`, `GameClock`, settings. Terrain/biomes are **regenerated** from the seed (not stored).

### 15.2 Commitment-style autosave
- Single save slot per campaign; **autosave every N seconds** (e.g. 60s) + on milestone events; **no manual save**.
- On loss (core death pre-redundancy / overwhelm / planet-inhabitable): autosave is not overwritten after the fatal tick → campaign ends → player starts a **fresh procgen planet** (new seed).
- Serialization via `serde` + `bevy_save` or a custom binary snapshot; **versioned** (`SAVE_VERSION`) with a migration stub for forward-compat.
- Large-world mitigation: stream/chunk the save; don't block the sim (write on a task).

---

## 16. The 10-Tier Progression Map (scale escalation)

| Tier | Scale | Scope (MVP = T1–4) |
|---|---|---|
| **T1** | Surface | Crash; core builds first extractor/factory; scouts; first unknown contacts; survive first waves. |
| **T2** | Surface | Land T2 + air factory (post-flying-faction); AA; geothermal; sensor/scanner; core-backup (redundancy). |
| **T3** | Surface | Arty/shields; stealth/detection (post-sneaking); voxel engineering/bunkers (post-digging); AoE swarm-counters. |
| **T4** | Surface→Orbit | **Escape capstone:** orbital-launch research + launch pad + upgraded core → launch sequence (director spikes). **MVP done.** |
| T5 | Orbital | (Later) orbit around starting planet; satellites; orbital defense; drop pods back to surface. |
| T6 | Planetary | Full planetary control; orbital construction; planetary-scale logistics. |
| T7 | Multi-planetary | Reach/exploit other procgen planets; each a fresh planet-phase. |
| T8–10 | Solar-system | Interplanetary empire; full 4X; the enigma expands to a cosmic/galactic mystery. Procedurally endless. |

**Scope mitigation for 10 tiers:** parametric tier scaling (power multipliers + a few new unlocks/tier), procedural unit/building variation, and hard phasing (build T1–4 to the T4 launch milestone before any space work).

---

## 17. Build Order (MVP = T1–4) — milestone-gated with acceptance criteria

1. **Bootstrap** — Cargo workspace + crates; Bevy 0.18 runs a window; tracy profiling; input stub. **AC:** empty Bevy window, tracy connected.
2. **Sphere + heightfield LOD** — spherified cube, quadtree chunked LOD, GPU displacement shader, skirts/edge-stitch, origin-shifting, frustum+horizon cull. **AC:** fly orbit↔surface with adaptive LOD, no seams/jitter, CPU noise == WGSL noise within 1e-4.
3. **Voxel patches** — chunked voxel volume + seam blend + edit/re-mesh + game-logic query routing. **AC:** dig a cave from surface; renders + queries correctly across LOD boundaries; destruction propagates.
4. **World procgen** — climate bands, biomes, water, resources, veins, hazards, weather/day-night, faction nests, authored POIs; lazy + cached. **AC:** generate + traverse a coherent planet; biome signals match placed content; seed reproducible.
5. **Sim core** — ECS components, FixedUpdate @30Hz, mass+energy streaming, extractors/generators/constructors, crashed core. **AC:** build a base; economy flows; stalls correctly on negative net.
6. **RTS units + pathfinding** — T1 factory, land+air units, formations, spherical flow-field (cost/integration/flow/LOS), spatial hash, boid avoidance. **AC:** command units; 1k units smooth (<16ms tick on target HW).
7. **Discovery** — spatial + knowledge fog, scanners, salvage-on-kill, study POIs, research tree (T1→T2→T3 salvage-gated). **AC:** scan a contact → stats; kill → salvage → unlock a tech node; sneaking faction resists scanning.
8. **Threats + director** — 1–2 archetypes (swarm + one other) first; authored-POI encounters; director (footprint + storyteller). **AC:** survive escalating waves; director pacing feels tense not spammy across tiers.
9. **T4 escape** — orbital-launch research + launch pad + core upgrade + launch sequence + space-layer transition stub. **AC:** complete a launch → MVP gate passed.
10. **Persistence + failure** — commitment autosave (serde), lose-conditions, fresh-planet restart. **AC:** save/load round-trip preserves world; lose → restart on new seed.
11. **Perf scaling** — profile 1k/10k/50k; GPU-compute hot path for 100k+; instanced mass rendering. **AC:** 50k units @ target FPS; 100k stress path runs.
12. **Co-op (later)** — lightyear integration; interest mgmt = fog; drop-in. **AC:** 2-player drop-in co-op with interpolated remote units.

---

## 18. Validation Plan

- **Terrain parity test:** CPU `fastnoise-lite` vs WGSL port, identical seed/dir → heights within 1e-4 (unit test, run in CI).
- **Seam test:** render adjacent LOD chunks + a voxel patch opening; assert no cracks/gaps (visual + depth-buffer diff in a headless render test).
- **Procgen determinism:** same seed → byte-identical biome/resource/POI placement map (integration test).
- **Pathfinding:** flow-field correctness (known maze → correct integrated costs + LOS) + benchmark (<5 ms/sector at 256², criterion bench).
- **Perf benches (criterion):** ECS tick at 1k/10k/50k; spatial hash rebuild; noise sampling; flow-field rebuild. Targets: 50k-unit tick <16 ms on a high-end CPU (single-threaded budget; multi-threaded sim aims lower).
- **Economy/loop:** end-to-end "crash → build → research → salvage → unlock → T4 launch" scripted scenario passes.
- **Save:** round-trip preserves entities + voxel edits + research + director + fog; loss path triggers fresh seed.
- **Discovery:** state machine transitions Unknown→Scanned→Studied; salvage drop tables conform to data.
- **Co-op:** (phase 12) drop-in reproducibly syncs; interest management bounds bandwidth under a 50k-swarm load test.
- **Milestone gate:** T4 launch completes → MVP accepted.

---

## 19. Risks & Mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Heightfield↔voxel seam + destruction propagation | High | Single elevation source of truth; skirts + edge-stitch; test destruction at LOD boundaries early (build-order 3). |
| R2 | GPU-compute unit sim (pathfinding/AI not just movement) is hard | High | Ship CPU ECS first; GPU compute is an optimization tier, not day-one. Keep AI state minimal. |
| R3 | 10-tier content load | High | Parametric tier scaling; procedural unit/building variation; hard phasing (T1–4 first). |
| R4 | Replication bandwidth at 50k+ | Med | lightyear interest management (=fog) + delta compression + 10 Hz + interpolation; selective readback. |
| R5 | `lightyear`/`planetmap` single-maintainer churn | Med | Pin versions; vendor/fork on demand; isolate behind our own traits. |
| R6 | CPU/GPU noise parity drift | Med | Shared FastNoiseLite GLSL port; CI parity test (§18). |
| R7 | Survival tension vs RTS scale | Med | Crashed core + director stakes + "you're outnumbered" framing carry tension; player colony stays meaningful. |
| R8 | f32 precision at planet scale | Med | f64 world math + origin shifting; convert to f32 only at render. |
| R9 | Solo-dev scope creep into space layer | High | Hard gate: no T5+ work until T4 launch milestone validated. |

---

## 20. Open Questions / Out of Scope

- **Space/4X layer (T5–10) detail** — unspecified beyond the scale sketch; design when T4 MVP is solid.
- **Authored-faction narrative & POI content** — count/depth deferred to a content pass.
- **Naval domain** — optional, deferred (water exists; add if time).
- **Meta-progression between campaigns** — currently none/minimal; revisit for roguelite feel.
- **Audio, full UI/UX, accessibility, localization** — separate passes.
- **Competitive MP balance / anti-cheat** — out of scope (co-op only; host authoritative).

---

## 21. References (research basis)
- Bevy 0.18 release notes + ECS schedule/SystemSet docs (docs.rs/bevy_ecs/0.18) + RenderApp/RenderGraph architecture (DeepWiki).
- Bevy compute: `examples/shader/gpu_readback.rs` (`Readback`/`ReadbackComplete`, `PipelineCache::queue_compute_pipeline`); `bevy_app_compute`; `bevy_gpu_test`.
- `lightyear` 0.26 (docs.rs, book) — server-authoritative, interest management, prediction/rollback, snapshot interpolation.
- `Ralith/planetmap` — streaming planetary terrain, 6 cubemap quadtrees, parry collision, SIMD noise.
- Flow-field pathfinding: Emerson, *Crowd Pathfinding and Steering Using Flow Field Tiles* (Game AI Pro); redblobgames; filipkunc `flowfield` Rust crate (~1.2k LOC, MIT).
- `fastnoise-lite` (Rust) — OpenSimplex2/Perlin/Cellular/domain-warp, f64 feature.
- Spherified-cube + chunked LOD: acko.net "Making Worlds"; cuberact planet-chunked-lod; tigerabrodi procgen pipeline.
- RTS netcode: Glenn Fiedler (Gaffer On Games) lockstep vs state-sync; mas-bandwidth network-model selection; yal.cc deterministic prep.
- Planetary Annihilation engine architecture (Uber Entertainment): 10 Hz authoritative client-server sim vs 60+ FPS render, flow-field pathfinding, curve/keyframe-based networking (bandwidth scales with active units — idle = 0 bytes), and CSG terrain + per-unit micro-collisions (ground units expensive; air/orbital cheap) as the cause of server time dilation — analyses at forrestthewoods.com and 0fps.net.

---

## 22. Implementation Note

This is a **plan only**. Execution requires switching to an implementation-capable agent. Begin at build-order item 1 (bootstrap) and proceed milestone-gated per the acceptance criteria; do **not** start the space/4X layers (T5–10) until the T4 launch milestone (build-order item 9) is validated. Keep the sim host-authoritative from day one (even SP-only) so co-op remains additive. Ship CPU ECS first; add the GPU-compute hot path only when profiling demands it.
