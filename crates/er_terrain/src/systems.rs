use bevy::camera::visibility::NoFrustumCulling;
use bevy::ecs::schedule::ApplyDeferred;
use bevy::ecs::schedule::SystemSet;
use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;
use bevy::shader::Shader;
use bevy::tasks::{futures::check_ready, AsyncComputeTaskPool, Task};
use er_core::config::{
    PlanetPreset, ACTIVE_CHUNK_CAP, CHUNK_QUADS_PER_EDGE, LOD_SPLIT_BUDGET_PER_FRAME,
    MAX_INFLIGHT_TERRAIN_MESHES, MERGE_HYSTERESIS, PLANET_RADIUS_DEFAULT, SCREEN_ERROR_THRESHOLD,
};
use er_core::math::{cell_size, cell_to_dir, CellKey, NeighborSide};
use er_core::seed::PlanetSeed;
use er_world::cache::WorldCache;
use er_world::elevation::{elevation_params, ElevationParams};
use er_world::params::{planet_params as make_planet_params, PlanetParams};
use er_world::terrain_field::{
    HybridTerrainField, MacroTerrainField, ProceduralTerrainField, TerrainField,
    TerrainSampleSource, TerrainSourceMode,
};
use glam::DVec3;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use crate::chunk::{ChunkComponent, HoldForMerge, HoldHidden};
use crate::culling::{
    frustum_cull_sphere, is_below_horizon, is_beyond_render_distance, is_minimum_coverage_chunk,
};
use crate::debug::TerrainDebugInfo;
use crate::lod::{chunk_camera_distance, screen_error, should_merge_parent, should_split};
use crate::material::{TerrainMaterial, TerrainMaterialUniform, FRAGMENT_SHADER, VERTEX_SHADER};
use crate::mesh_gen::{generate_chunk_mesh_stitched, normal_sample_spacing_m, StitchNeighbors};
use crate::ocean::{setup_ocean, update_ocean_time, OceanMaterial};
use crate::quadtree::{
    children_of, coarser_neighbor_across_edge, parent_of, root_chunks, ActiveChunks, RetainedMerge,
    RetainedMerges, RetainedSplit, RetainedSplits,
};

const NEIGHBOR_SIDES: [NeighborSide; 4] = [
    NeighborSide::NegU,
    NeighborSide::PosU,
    NeighborSide::NegV,
    NeighborSide::PosV,
];

#[derive(Resource, Clone, Copy)]
pub struct SunDirection(pub Vec3);

impl Default for SunDirection {
    fn default() -> Self {
        Self(Vec3::new(0.5, 0.8, 0.3).normalize())
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct CameraWorldPosition(pub DVec3);

#[derive(Resource, Clone, Debug)]
pub struct RenderOrigin {
    pub world: DVec3,
    pub generation: u64,
    pub cell_size_m: f64,
}

impl RenderOrigin {
    pub fn with_cell_size(cell_size_m: f64) -> Self {
        assert!(cell_size_m.is_finite() && cell_size_m > 0.0);
        Self {
            cell_size_m,
            ..Self::default()
        }
    }

    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(
            self.world.x as f32,
            self.world.y as f32,
            self.world.z as f32,
        )
    }
}

impl Default for RenderOrigin {
    fn default() -> Self {
        Self {
            world: DVec3::ZERO,
            generation: 0,
            cell_size_m: 1000.0,
        }
    }
}

pub struct PendingMeshPayload {
    pub mesh: Mesh,
    pub key: CellKey,
    pub source_anchor: DVec3,
    pub origin_generation: u64,
}

#[derive(SystemSet, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TerrainUpdate;

#[derive(Resource)]
pub struct TerrainState {
    pub planet_radius: f64,
    pub elevation_scale: f32,
    pub params: ElevationParams,
    pub planet_params: PlanetParams,
    pub source_mode: TerrainSourceMode,
    pub field: Arc<dyn TerrainField>,
    pub base_uniform: TerrainMaterialUniform,
    pub max_quadtree_depth: u8,
    pub screen_error_threshold: f32,
    pub merge_hysteresis: f32,
    pub max_render_distance: f64,
    pub active_chunk_cap: usize,
    pub lod_split_budget_per_frame: usize,
    pub max_inflight_terrain_meshes: usize,
}

impl TerrainState {
    pub fn new(planet_radius: f64, elevation_scale: f32, seed: PlanetSeed) -> Self {
        Self::with_optional_macro_field(
            planet_radius,
            elevation_scale,
            seed,
            PlanetPreset::default(),
            None,
        )
    }

    pub fn for_preset(preset: PlanetPreset, elevation_scale: f32, seed: PlanetSeed) -> Self {
        Self::with_optional_macro_field(preset.radius_m(), elevation_scale, seed, preset, None)
    }

    fn with_optional_macro_field(
        planet_radius: f64,
        elevation_scale: f32,
        seed: PlanetSeed,
        preset: PlanetPreset,
        macro_field: Option<Arc<dyn MacroTerrainField>>,
    ) -> Self {
        let params = elevation_params(seed);
        let planet_params = make_planet_params(seed);
        let (cache_capacity, cache_lod) = preset_cache_config(preset);
        let cache = Arc::new(WorldCache::with_lod(cache_capacity, cache_lod));
        let fallback: Arc<dyn TerrainField> = match preset {
            PlanetPreset::MiniatureDebug => Arc::new(ProceduralTerrainField::with_cache(
                params,
                planet_params,
                Arc::clone(&cache),
            )),
            PlanetPreset::EarthScale => Arc::new(ProceduralTerrainField::with_cache_metric(
                params,
                planet_params,
                Arc::clone(&cache),
                planet_radius,
            )),
        };
        let (source_mode, field): (TerrainSourceMode, Arc<dyn TerrainField>) =
            if let Some(macro_field) = macro_field {
                (
                    TerrainSourceMode::HybridLearned,
                    Arc::new(HybridTerrainField::new(fallback, macro_field)),
                )
            } else {
                (TerrainSourceMode::Procedural, fallback)
            };
        let base_uniform = TerrainMaterialUniform::from_params(
            &params,
            planet_radius as f32,
            elevation_scale,
            &planet_params,
        );
        Self {
            planet_radius,
            elevation_scale,
            params,
            planet_params,
            source_mode,
            field,
            base_uniform,
            max_quadtree_depth: preset.max_quadtree_depth(),
            screen_error_threshold: SCREEN_ERROR_THRESHOLD,
            merge_hysteresis: MERGE_HYSTERESIS,
            max_render_distance: preset.max_render_distance_m(),
            active_chunk_cap: ACTIVE_CHUNK_CAP,
            lod_split_budget_per_frame: LOD_SPLIT_BUDGET_PER_FRAME,
            max_inflight_terrain_meshes: MAX_INFLIGHT_TERRAIN_MESHES,
        }
    }
}

fn preset_cache_config(preset: PlanetPreset) -> (usize, u8) {
    match preset {
        PlanetPreset::MiniatureDebug => (262144, 16),
        PlanetPreset::EarthScale => (1048576, 22),
    }
}

#[derive(Resource, Default)]
pub struct PendingChunkMeshes(pub HashMap<Entity, Task<PendingMeshPayload>>);

struct ChunkMeshRequest {
    entity: Entity,
    key: CellKey,
    radius: f64,
    elevation_scale: f32,
    field: Arc<dyn TerrainField>,
    origin_generation: u64,
    source_anchor: DVec3,
    stitch_neighbors: StitchNeighbors,
}

/// Cheap mesh descriptors waiting for a bounded slot on the async compute pool.
/// Keeping them outside the pool allows camera movement to reprioritize work.
#[derive(Resource, Default)]
pub struct QueuedChunkMeshes(HashMap<Entity, ChunkMeshRequest>);

impl QueuedChunkMeshes {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    fn contains(&self, entity: Entity) -> bool {
        self.0.contains_key(&entity)
    }
}

/// Chunks awaiting a rebuild after resident learned macro data changes. Work is
/// throttled so a sidecar tile arrival cannot monopolize mesh worker capacity.
#[derive(Resource, Default)]
struct TerrainFieldRefresh {
    revision: u64,
    queued: VecDeque<(Entity, CellKey)>,
}

const LEARNED_FIELD_REBUILD_BUDGET: usize = 8;

/// All terrain chunks share one bind group. Per-chunk geometry is baked into
/// each mesh, which lets Bevy batch their render-phase work.
#[derive(Resource, Clone)]
pub struct SharedTerrainMaterial(pub Handle<TerrainMaterial>);

/// A root mesh retained after it has been replaced by detailed children. It is
/// hidden during normal play and becomes the guaranteed terrain coverage floor
/// once detailed chunks are beyond the render distance.
#[derive(Component)]
struct FarCoverageRoot;

pub struct TerrainPlugin {
    pub planet_radius: f64,
    pub elevation_scale: f32,
    pub seed: PlanetSeed,
    pub preset: PlanetPreset,
    render_origin_cell_size_m: f64,
    macro_field: Option<Arc<dyn MacroTerrainField>>,
}

impl Default for TerrainPlugin {
    fn default() -> Self {
        Self {
            planet_radius: PLANET_RADIUS_DEFAULT,
            elevation_scale: 1000.0,
            seed: PlanetSeed(0xC0FFEE),
            preset: PlanetPreset::default(),
            render_origin_cell_size_m: 1000.0,
            macro_field: None,
        }
    }
}

impl TerrainPlugin {
    /// Enables the learned macro path while preserving procedural residuals and
    /// fallback behavior for every tile that is not yet resident.
    pub fn with_hybrid_macro_field(mut self, macro_field: Arc<dyn MacroTerrainField>) -> Self {
        self.macro_field = Some(macro_field);
        self
    }

    pub fn with_render_origin_cell_size(mut self, cell_size_m: f64) -> Self {
        assert!(cell_size_m.is_finite() && cell_size_m > 0.0);
        self.render_origin_cell_size_m = cell_size_m;
        self
    }

    /// Use `PlanetPreset` parameters (radius, LOD depth, render distance) while
    /// keeping the caller's `elevation_scale` and `seed`.
    pub fn from_preset(preset: PlanetPreset, elevation_scale: f32, seed: PlanetSeed) -> Self {
        Self {
            planet_radius: preset.radius_m(),
            elevation_scale,
            seed,
            preset,
            render_origin_cell_size_m: 1000.0,
            macro_field: None,
        }
    }
}

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(TerrainState::with_optional_macro_field(
            self.planet_radius,
            self.elevation_scale,
            self.seed,
            self.preset,
            self.macro_field.clone(),
        ))
        .insert_resource(ActiveChunks::default())
        .insert_resource(RetainedSplits::default())
        .insert_resource(RetainedMerges::default())
        .insert_resource(TerrainDebugInfo::default())
        .insert_resource(PendingChunkMeshes::default())
        .insert_resource(QueuedChunkMeshes::default())
        .insert_resource(TerrainFieldRefresh::default())
        .insert_resource(crate::profiler::FrameProfiler::default())
        .insert_resource(SunDirection::default())
        .insert_resource(CameraWorldPosition::default())
        .insert_resource(RenderOrigin::with_cell_size(self.render_origin_cell_size_m))
        .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
        .add_plugins(MaterialPlugin::<OceanMaterial>::default())
        .add_systems(Startup, (setup_terrain, setup_ocean))
        .add_systems(PreUpdate, profiler_clear)
        .add_systems(
            Update,
            (
                apply_render_origin_to_chunks,
                update_lod,
                queue_resident_field_refreshes,
                process_lod_queue,
                ApplyDeferred,
                dispatch_chunk_meshes,
                apply_pending_chunk_meshes,
                ApplyDeferred,
                finalize_retirements,
                ApplyDeferred,
                finalize_retained_merges,
                ApplyDeferred,
                queue_stitch_rebuilds,
                cull_chunks,
                update_debug_info,
                update_ocean_time,
            )
                .chain()
                .in_set(TerrainUpdate),
        );
    }
}

fn setup_terrain(
    mut commands: Commands,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut terrain_state: ResMut<TerrainState>,
    mut active_chunks: ResMut<ActiveChunks>,
    mut queued_meshes: ResMut<QueuedChunkMeshes>,
    origin: Res<RenderOrigin>,
) {
    let vertex_source = format!(
        "{}\n{}\n{}\n{}",
        include_str!("../../er_world/assets/shaders/elevation.wgsl"),
        include_str!("../assets/shaders/spherify.wgsl"),
        include_str!("../assets/shaders/terrain_uniform.wgsl"),
        include_str!("../assets/shaders/terrain_vertex.wgsl")
    );
    let vertex_handle = shaders.add(Shader::from_wgsl(vertex_source, "terrain_vertex"));
    let _ = VERTEX_SHADER.set(vertex_handle);

    let fragment_source = format!(
        "{}\n{}",
        include_str!("../assets/shaders/terrain_uniform.wgsl"),
        include_str!("../assets/shaders/terrain_fragment.wgsl")
    );
    let fragment_handle = shaders.add(Shader::from_wgsl(fragment_source, "terrain_fragment"));
    let _ = FRAGMENT_SHADER.set(fragment_handle);

    let uniform = TerrainMaterialUniform::from_params(
        &terrain_state.params,
        terrain_state.planet_radius as f32,
        terrain_state.elevation_scale,
        &terrain_state.planet_params,
    );
    terrain_state.base_uniform = uniform;
    let material = materials.add(TerrainMaterial { uniform });
    commands.insert_resource(SharedTerrainMaterial(material.clone()));

    for key in root_chunks() {
        let entity = spawn_chunk_entity(
            &mut commands,
            &mut queued_meshes,
            &material,
            key,
            terrain_state.planet_radius,
            terrain_state.elevation_scale,
            Arc::clone(&terrain_state.field),
            &origin,
            [None; 4],
        );
        active_chunks.insert(key, entity);
    }
}

fn profiler_clear(mut profiler: ResMut<crate::profiler::FrameProfiler>) {
    profiler.clear();
}

fn apply_render_origin_to_chunks(
    origin: Res<RenderOrigin>,
    terrain_state: Res<TerrainState>,
    mut chunks: Query<(&ChunkComponent, &mut Transform)>,
) {
    if !origin.is_changed() {
        return;
    }

    for (chunk, mut transform) in &mut chunks {
        transform.translation =
            chunk_render_translation(chunk.key, terrain_state.planet_radius, origin.world);
    }
}

fn chunk_render_translation(key: CellKey, radius: f64, origin: DVec3) -> Vec3 {
    (cell_to_dir(key) * radius - origin).as_vec3()
}

fn queue_resident_field_refreshes(
    terrain_state: Res<TerrainState>,
    mut refresh: ResMut<TerrainFieldRefresh>,
    chunk_query: Query<(Entity, &ChunkComponent)>,
    active_chunks: Res<ActiveChunks>,
    mut queued_meshes: ResMut<QueuedChunkMeshes>,
    pending: Res<PendingChunkMeshes>,
    origin: Res<RenderOrigin>,
) {
    let revision = terrain_state.field.revision();
    if revision != refresh.revision {
        refresh.revision = revision;
        refresh.queued.clear();
        refresh.queued.extend(
            chunk_query
                .iter()
                .map(|(entity, chunk)| (entity, chunk.key)),
        );
    }

    let attempts = refresh.queued.len().min(LEARNED_FIELD_REBUILD_BUDGET);
    for _ in 0..attempts {
        let Some((entity, key)) = refresh.queued.pop_front() else {
            break;
        };
        if !chunk_query
            .get(entity)
            .is_ok_and(|(_, chunk)| pending_mesh_matches_chunk(chunk.key, key))
        {
            continue;
        }
        if pending.0.contains_key(&entity) || queued_meshes.contains(entity) {
            // Let the in-flight mesh finish, then rebuild it from the newest
            // resident field data on a later frame.
            refresh.queued.push_back((entity, key));
            continue;
        }
        queue_chunk_mesh(
            &mut queued_meshes,
            entity,
            key,
            terrain_state.planet_radius,
            terrain_state.elevation_scale,
            Arc::clone(&terrain_state.field),
            origin.generation,
            stitch_neighbors(key, &active_chunks),
        );
    }
}

fn breadth_chunk_cap(active_chunk_cap: usize, max_depth: u8) -> usize {
    active_chunk_cap.saturating_sub(usize::from(max_depth) * 3)
}

fn update_lod(
    camera_query: Query<(&GlobalTransform, &Projection), With<Camera3d>>,
    camera_world: Res<CameraWorldPosition>,
    mut active_chunks: ResMut<ActiveChunks>,
    terrain_state: Res<TerrainState>,
    origin: Res<RenderOrigin>,
    retained_splits: Res<RetainedSplits>,
    retained_merges: Res<RetainedMerges>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let camera_pos = camera_world.0;

    let keys: Vec<CellKey> = active_chunks.chunks.keys().copied().collect();
    active_chunks.clear_pending();
    let camera_dir = camera_pos.normalize_or(DVec3::Y);
    let breadth_cap = breadth_chunk_cap(
        terrain_state.active_chunk_cap,
        terrain_state.max_quadtree_depth,
    );
    let split_capacity_blocked = active_chunks.len() + 3 > breadth_cap;
    let split_frustum = camera_query
        .single()
        .ok()
        .and_then(|(transform, projection)| {
            let Projection::Perspective(perspective) = projection else {
                return None;
            };
            Some((
                transform.translation(),
                *transform.forward(),
                *transform.right(),
                *transform.up(),
                (perspective.fov / 2.0).cos(),
                perspective.aspect_ratio,
            ))
        });

    for &key in &keys {
        if is_below_horizon(key, camera_pos, terrain_state.planet_radius) {
            continue;
        }
        if let Some((cam_pos, forward, right, up, fov_cos, aspect)) = split_frustum {
            let chunk_center = cell_to_dir(key) * terrain_state.planet_radius;
            let sphere_center = (chunk_center - origin.world).as_vec3();
            let sphere_radius = cell_size(key.lod, terrain_state.planet_radius) as f32
                + terrain_state.elevation_scale * 3.0;
            if frustum_cull_sphere(
                sphere_center,
                sphere_radius,
                cam_pos,
                forward,
                right,
                up,
                fov_cos,
                aspect,
            ) {
                continue;
            }
        }

        // If this chunk is waiting for its parent mesh to finish during a
        // retained merge, don't re-evaluate its LOD — the merge decision is
        // already in flight.
        if let Some(parent) = parent_of(key) {
            if retained_merges.map.contains_key(&parent) {
                continue;
            }
        }

        if should_split(
            key,
            camera_pos,
            terrain_state.planet_radius,
            terrain_state.max_quadtree_depth,
            terrain_state.screen_error_threshold,
        ) {
            // Once the cap is full, breadth splits cannot be admitted. Keep
            // evaluating only the camera-containing path so reclaimed capacity
            // is spent on close detail instead of sorting thousands of blocked
            // candidates every frame.
            if !split_capacity_blocked || cell_contains_direction(key, camera_dir) {
                active_chunks.pending_splits.push(key);
            }
        }
    }

    let mut parents_to_check: HashSet<CellKey> = HashSet::new();
    for &key in &keys {
        if let Some(parent) = parent_of(key) {
            parents_to_check.insert(parent);
        }
    }
    for parent_key in parents_to_check {
        // A retained root mesh is the far-distance coverage floor. Keep the
        // active tree at least one level below it so merging cannot recreate a
        // second root entity on top of that fallback.
        if is_minimum_coverage_chunk(parent_key) {
            continue;
        }
        // Skip parents already splitting (retained split) or already waiting on
        // a retained merge parent mesh.
        if retained_splits.map.contains_key(&parent_key)
            || retained_merges.map.contains_key(&parent_key)
        {
            continue;
        }

        let children = children_of(parent_key);
        if children.iter().all(|c| active_chunks.contains(c)) {
            if should_merge_parent(
                parent_key,
                camera_pos,
                terrain_state.planet_radius,
                terrain_state.screen_error_threshold,
                terrain_state.merge_hysteresis,
            ) {
                active_chunks.pending_merges.push(parent_key);
            }
        }
    }

    sort_pending_splits_by_screen_error(
        &mut active_chunks.pending_splits,
        camera_pos,
        terrain_state.planet_radius,
    );

    profiler.record("update_lod", t0.elapsed());
}

#[allow(clippy::too_many_arguments)]
fn process_lod_queue(
    mut commands: Commands,
    terrain_material: Res<SharedTerrainMaterial>,
    mut active_chunks: ResMut<ActiveChunks>,
    mut retained: ResMut<RetainedSplits>,
    mut retained_merges: ResMut<RetainedMerges>,
    terrain_state: Res<TerrainState>,
    camera_world: Res<CameraWorldPosition>,
    origin: Res<RenderOrigin>,
    mesh_query: Query<(), With<Mesh3d>>,
    mut debug: ResMut<TerrainDebugInfo>,
    mut queued_meshes: ResMut<QueuedChunkMeshes>,
    pending_meshes: Res<PendingChunkMeshes>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let camera_pos = camera_world.0;
    let budget = terrain_state.lod_split_budget_per_frame;
    let camera_dir = camera_pos.normalize_or(DVec3::Y);
    let breadth_cap = breadth_chunk_cap(
        terrain_state.active_chunk_cap,
        terrain_state.max_quadtree_depth,
    );

    let mut splits_done = 0usize;
    let pending_splits: Vec<CellKey> = std::mem::take(&mut active_chunks.pending_splits);
    let split_budget = budget;
    for requested_key in pending_splits {
        if splits_done >= split_budget {
            break;
        }
        let key = balancing_split_for(requested_key, &active_chunks).unwrap_or(requested_key);
        let required_for_focus = cell_contains_direction(requested_key, camera_dir);
        let outstanding_meshes = pending_meshes.0.len() + queued_meshes.len();
        if !can_admit_split_meshes(
            required_for_focus,
            outstanding_meshes,
            terrain_state.max_inflight_terrain_meshes,
        ) {
            continue;
        }
        // Keep one complete camera-containing refinement path inside the hard
        // cap. A balancing dependency of that path uses the same reservation;
        // otherwise a deep focused leaf could border a much coarser spherical
        // chord and produce a visible terrain curtain.
        let split_cap = if required_for_focus {
            terrain_state.active_chunk_cap
        } else {
            breadth_cap
        };
        if active_chunks.len() + 3 > split_cap {
            continue;
        }
        if !active_chunks.contains(&key) {
            continue;
        }

        // Do not split a child until its parent's retained handoff has
        // finalized. Otherwise the child can be despawned by its own split
        // while the outer retention still waits for that exact entity,
        // leaving the coarse fallback visible forever over detailed terrain.
        if is_child_of_retained_split(key, &retained) {
            continue;
        }

        let parent_entity = *active_chunks.chunks.get(&key).unwrap();

        // Gate: only split a chunk that already has a mesh. A freshly-spawned
        // (still-meshless) child is never split, so the cascade is serialized
        // one LOD level per mesh round-trip — the retained parent stays as the
        // visible fallback until the child mesh is ready.
        if mesh_query.get(parent_entity).is_err() {
            continue;
        }

        // Retain the parent as a visible fallback instead of despawning it.
        // It is dropped from the LOD set (so it is not re-split/merged) but
        // stays rendered until all four children have meshes.
        active_chunks.remove(&key);

        // Ensure the retained parent is visible. If this chunk was itself a
        // hidden child of an outer retained split, it must now render as the
        // fallback for its own children.
        if let Ok(mut e) = commands.get_entity(parent_entity) {
            e.remove::<HoldHidden>().insert(Visibility::Visible);
        }

        let children = children_of(key);
        let child_entities: [Entity; 4] = core::array::from_fn(|idx| {
            let child = children[idx];
            let child_stitch_neighbors = stitch_neighbors(child, &active_chunks);
            let entity = spawn_chunk_entity(
                &mut commands,
                &mut queued_meshes,
                &terrain_material.0,
                child,
                terrain_state.planet_radius,
                terrain_state.elevation_scale,
                Arc::clone(&terrain_state.field),
                &origin,
                child_stitch_neighbors,
            );
            // Hold hidden until all four siblings are meshed (atomic reveal).
            if let Ok(mut e) = commands.get_entity(entity) {
                e.insert(HoldHidden);
            }
            active_chunks.insert(child, entity);
            entity
        });
        retained.map.insert(
            key,
            RetainedSplit {
                parent_entity,
                children: child_entities,
            },
        );
        splits_done += 1;
    }
    debug.pending_splits = splits_done;

    let over_cap = active_chunks
        .len()
        .saturating_sub(terrain_state.active_chunk_cap);
    let merge_budget = if over_cap > 0 {
        (budget * 4).max(over_cap * 4)
    } else {
        budget
    };
    let mut merges_done = 0usize;
    let mut merged_keys = HashSet::new();
    let mut pending_merges: Vec<CellKey> = std::mem::take(&mut active_chunks.pending_merges);
    if over_cap > 0 {
        pending_merges.sort_by(|a, b| {
            let da = chunk_camera_distance(*a, camera_pos, terrain_state.planet_radius);
            let db = chunk_camera_distance(*b, camera_pos, terrain_state.planet_radius);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    for parent_key in pending_merges {
        if merges_done >= merge_budget {
            break;
        }
        if has_conflicting_merge(parent_key, &merged_keys) {
            continue;
        }
        // Don't merge a parent whose children are still meshing in a retained
        // split. Despawning those children before their queued HoldHidden /
        // Mesh3d commands are applied causes an entity-generation mismatch panic.
        if retained.map.contains_key(&parent_key) {
            continue;
        }
        let children = children_of(parent_key);
        if !children.iter().all(|c| active_chunks.contains(c)) {
            continue;
        }

        // If this parent is currently retained from an in-flight split (the
        // camera reversed before the children finished meshing), reuse its
        // already-meshed entity instead of regenerating — instant merge, no
        // gap and no orphaned fallback.
        if let Some(entry) = retained.map.remove(&parent_key) {
            for child in &children {
                if let Some(entity) = active_chunks.remove(child) {
                    despawn_chunk(&mut commands, entity);
                }
            }
            active_chunks.insert(parent_key, entry.parent_entity);
            merged_keys.insert(parent_key);
            merges_done += 1;
            continue;
        }

        // Retained merge: keep the children rendered as the visible fallback
        // while the new parent's mesh generates asynchronously. The children
        // are removed from `ActiveChunks` (so they are not re-evaluated) but
        // stay visible until `finalize_retained_merges` reveals the parent and
        // despawns them atomically. This removes the 1-2 frame black gap of the
        // old non-retained merge path.
        let mut child_entities: [Entity; 4] = [Entity::PLACEHOLDER; 4];
        for (idx, child) in children.iter().enumerate() {
            if let Some(entity) = active_chunks.remove(child) {
                // Tag the child so cull_chunks skips it and leaves it at its
                // current visibility. It will be despawned the frame the
                // parent mesh is ready.
                if let Ok(mut e) = commands.get_entity(entity) {
                    e.insert(HoldForMerge);
                }
                child_entities[idx] = entity;
            }
        }

        let parent_entity = spawn_chunk_entity(
            &mut commands,
            &mut queued_meshes,
            &terrain_material.0,
            parent_key,
            terrain_state.planet_radius,
            terrain_state.elevation_scale,
            Arc::clone(&terrain_state.field),
            &origin,
            stitch_neighbors(parent_key, &active_chunks),
        );
        // Parent starts Hidden (spawn_chunk_entity default) and without a mesh.
        // finalize_retained_merges will reveal it once its mesh arrives.
        retained_merges.map.insert(
            parent_key,
            RetainedMerge {
                parent_key,
                parent_entity,
                children: child_entities,
            },
        );
        active_chunks.insert(parent_key, parent_entity);
        merged_keys.insert(parent_key);
        merges_done += 1;
    }
    debug.pending_merges = merges_done;

    profiler.record("process_lod_queue", t0.elapsed());
}

fn cull_chunks(
    mut camera_query: Query<(&GlobalTransform, &mut Projection), With<Camera3d>>,
    mut chunk_query: Query<
        (&ChunkComponent, &mut Visibility, Option<&FarCoverageRoot>),
        (With<Mesh3d>, Without<HoldHidden>, Without<HoldForMerge>),
    >,
    terrain_state: Res<TerrainState>,
    camera_world: Res<CameraWorldPosition>,
    origin: Res<RenderOrigin>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let Ok((camera_transform, mut projection)) = camera_query.single_mut() else {
        profiler.record("cull_chunks", t0.elapsed());
        return;
    };
    let camera_pos = camera_world.0;

    let planet_radius = terrain_state.planet_radius as f32;
    let cam_dist = camera_pos.length() as f32;
    let (near, far) = camera_clip_planes(planet_radius, cam_dist);
    if let Projection::Perspective(p) = &mut *projection {
        p.near = near;
        p.far = far;
    }

    let frustum = match &*projection {
        Projection::Perspective(p) => {
            let fov_cos = (p.fov / 2.0).cos();
            let aspect = p.aspect_ratio;
            Some((
                camera_transform.translation(),
                *camera_transform.forward(),
                *camera_transform.right(),
                *camera_transform.up(),
                fov_cos,
                aspect,
            ))
        }
        _ => None,
    };

    let max_render_dist_sq = (terrain_state.max_render_distance * 1.15).powi(2);

    for (chunk, mut visibility, far_coverage_root) in &mut chunk_query {
        let key = chunk.key;

        // Retained roots prevent the planet from becoming transparent while the
        // detailed LOD tree asynchronously merges back after a large zoom-out.
        if far_coverage_root.is_some()
            && hide_far_coverage_root(cam_dist, terrain_state.max_render_distance as f32)
        {
            visibility.set_if_neq(Visibility::Hidden);
            continue;
        }
        let chunk_dir = cell_to_dir(key);
        let chunk_center = chunk_dir * terrain_state.planet_radius;

        let dist_sq = (chunk_center - camera_pos).length_squared();
        // At maximum zoom-out, retain the six root faces as a coarse coverage
        // floor. Finer chunks still obey the normal distance limit, and roots
        // continue through horizon/frustum culling below.
        if is_beyond_render_distance(key, dist_sq, max_render_dist_sq) {
            visibility.set_if_neq(Visibility::Hidden);
            continue;
        }

        if is_below_horizon(key, camera_pos, terrain_state.planet_radius) {
            visibility.set_if_neq(Visibility::Hidden);
            continue;
        }

        if let Some((cam_pos, forward, right, up, fov_cos, aspect)) = frustum {
            let sphere_center = (chunk_center - origin.world).as_vec3();
            let sphere_radius = cell_size(key.lod, terrain_state.planet_radius) as f32
                + terrain_state.elevation_scale * 3.0;
            if frustum_cull_sphere(
                sphere_center,
                sphere_radius,
                cam_pos,
                forward,
                right,
                up,
                fov_cos,
                aspect,
            ) {
                visibility.set_if_neq(Visibility::Hidden);
                continue;
            }
        }

        visibility.set_if_neq(Visibility::Visible);
    }
    profiler.record("cull_chunks", t0.elapsed());
}

fn camera_clip_planes(planet_radius: f32, camera_distance: f32) -> (f32, f32) {
    let (near, baseline_far): (f32, f32) = if camera_distance < planet_radius * 1.1 {
        (0.1, 500000.0)
    } else if camera_distance < planet_radius * 5.0 {
        (1.0, 500000.0)
    } else {
        (10.0, 5000000.0)
    };

    // The original fixed limits were calibrated for the 36 km diagnostic
    // planet. Earth-scale close views need to reach the limb, which can be
    // farther than 500 km even though the camera is only tens of kilometres
    // above the surface.
    let planet_far = (camera_distance + planet_radius) * 1.1;
    (near, baseline_far.max(planet_far))
}

fn hide_far_coverage_root(camera_distance: f32, max_render_distance: f32) -> bool {
    camera_distance <= max_render_distance
}

fn update_debug_info(
    active_chunks: Res<ActiveChunks>,
    retained_splits: Res<RetainedSplits>,
    pending: Res<PendingChunkMeshes>,
    queued_meshes: Res<QueuedChunkMeshes>,
    chunk_query: Query<&Visibility, With<ChunkComponent>>,
    mesh_query: Query<(), (With<ChunkComponent>, With<Mesh3d>)>,
    camera_world: Res<CameraWorldPosition>,
    terrain_state: Res<TerrainState>,
    origin: Res<RenderOrigin>,
    mut debug: ResMut<TerrainDebugInfo>,
    profiler: Res<crate::profiler::FrameProfiler>,
) {
    debug.active_chunks = active_chunks.len();
    debug.max_depth = active_chunks
        .chunks
        .keys()
        .map(|k| k.lod)
        .max()
        .unwrap_or(0);
    debug.pending_meshes = pending.0.len() + queued_meshes.len();
    debug.visible_chunks = chunk_query
        .iter()
        .filter(|visibility| matches!(visibility, Visibility::Visible))
        .count();
    debug.estimated_mesh_bytes =
        mesh_query.iter().count() * crate::debug::ESTIMATED_BYTES_PER_CHUNK_MESH;
    debug.frame_time_ms = profiler.total().as_secs_f32() * 1000.0;

    let cam_pos = camera_world.0;
    let radius = terrain_state.planet_radius;

    let cam_len = cam_pos.length();
    let cam_dir = if cam_len > 1e-6 {
        cam_pos / cam_len
    } else {
        DVec3::Y
    };
    let sample = terrain_state.field.sample(cam_dir);
    let terrain_elev = sample.elevation * terrain_state.elevation_scale as f64;
    debug.camera_altitude_m = cam_len - radius - terrain_elev;

    debug.render_origin_world = origin.world;
    debug.render_origin_generation = origin.generation;
    debug.source_mode = terrain_state.source_mode;

    let container = containing_ancestor_key(
        cam_dir,
        terrain_state.max_quadtree_depth,
        &active_chunks,
        &retained_splits,
    );
    let cs = cell_size(container.lod, radius);
    let vs = cs / CHUNK_QUADS_PER_EDGE as f64;
    debug.nearest_chunk_lod = container.lod;
    debug.nearest_chunk_width_m = cs;
    debug.vertex_spacing_m = vs;
    let normal_spacing = normal_sample_spacing_m(container.lod, radius);
    debug.normal_diff_spacing_m = normal_spacing;
    debug.normal_difference_span_m = 2.0 * normal_spacing;
    debug.normal_diff_epsilon_radians = (normal_spacing / radius).clamp(1e-8, 0.25);

    let coverage_angle = (vs * 4.0 / radius).clamp(1e-8, 0.25);
    let (proc, learned) = estimate_source_coverage(&*terrain_state.field, cam_dir, coverage_angle);
    let total = proc + learned;
    debug.procedural_source_coverage_percent = if total > 0 {
        proc as f32 / total as f32 * 100.0
    } else {
        0.0
    };
    debug.learned_source_coverage_percent = if total > 0 {
        learned as f32 / total as f32 * 100.0
    } else {
        0.0
    };
}

fn estimate_source_coverage(
    field: &dyn TerrainField,
    center_dir: DVec3,
    ring_angle_rad: f64,
) -> (usize, usize) {
    const RING_SAMPLES: usize = 16;

    let t = if center_dir.y.abs() < 0.99 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let tangent_u = center_dir.cross(t).normalize();
    let tangent_v = center_dir.cross(tangent_u).normalize();

    let mut procedural = 0usize;
    let mut learned = 0usize;

    for i in 0..RING_SAMPLES {
        let theta = (i as f64 / RING_SAMPLES as f64) * std::f64::consts::TAU;
        let offset = tangent_u * theta.cos() + tangent_v * theta.sin();
        let dir = (center_dir * ring_angle_rad.cos() + offset * ring_angle_rad.sin()).normalize();
        let sample = field.sample(dir);
        match sample.source {
            TerrainSampleSource::Procedural => procedural += 1,
            TerrainSampleSource::LearnedMacro => learned += 1,
        }
    }

    (procedural, learned)
}

#[allow(clippy::too_many_arguments)]
fn spawn_chunk_entity(
    commands: &mut Commands,
    queued_meshes: &mut QueuedChunkMeshes,
    material: &Handle<TerrainMaterial>,
    key: CellKey,
    radius: f64,
    elevation_scale: f32,
    field: Arc<dyn TerrainField>,
    origin: &RenderOrigin,
    stitch_neighbors: StitchNeighbors,
) -> Entity {
    let mut chunk = ChunkComponent::new(key);
    chunk.neighbor_depth = stitch_neighbor_depths(key, stitch_neighbors);
    let entity = commands
        .spawn((
            chunk,
            MeshMaterial3d(material.clone()),
            Transform::from_translation(chunk_render_translation(key, radius, origin.world)),
            Visibility::Hidden,
            // The custom culler includes horizon, distance, and frustum tests.
            // Avoid repeating Bevy's AABB frustum walk for every terrain chunk.
            NoFrustumCulling,
        ))
        .id();

    queue_chunk_mesh(
        queued_meshes,
        entity,
        key,
        radius,
        elevation_scale,
        field,
        origin.generation,
        stitch_neighbors,
    );

    entity
}

fn queue_chunk_mesh(
    queued_meshes: &mut QueuedChunkMeshes,
    entity: Entity,
    key: CellKey,
    radius: f64,
    elevation_scale: f32,
    field: Arc<dyn TerrainField>,
    origin_generation: u64,
    stitch_neighbors: StitchNeighbors,
) {
    let source_anchor = cell_to_dir(key) * radius;
    queued_meshes.0.insert(
        entity,
        ChunkMeshRequest {
            entity,
            key,
            radius,
            elevation_scale,
            field,
            origin_generation,
            source_anchor,
            stitch_neighbors,
        },
    );
}

fn dispatch_chunk_meshes(
    mut queued_meshes: ResMut<QueuedChunkMeshes>,
    mut pending: ResMut<PendingChunkMeshes>,
    terrain_state: Res<TerrainState>,
    camera_world: Res<CameraWorldPosition>,
    chunk_query: Query<&ChunkComponent>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let available_slots =
        mesh_dispatch_slots(terrain_state.max_inflight_terrain_meshes, pending.0.len());
    if available_slots == 0 || queued_meshes.is_empty() {
        profiler.record("dispatch_meshes", t0.elapsed());
        return;
    }

    let mut requests: Vec<ChunkMeshRequest> = std::mem::take(&mut queued_meshes.0)
        .into_values()
        .filter(|request| {
            chunk_query
                .get(request.entity)
                .is_ok_and(|chunk| pending_mesh_matches_chunk(chunk.key, request.key))
        })
        .collect();
    sort_chunk_mesh_requests(&mut requests, camera_world.0, terrain_state.planet_radius);

    for (index, request) in requests.into_iter().enumerate() {
        if index >= available_slots {
            queued_meshes.0.insert(request.entity, request);
            continue;
        }

        let ChunkMeshRequest {
            entity,
            key,
            radius,
            elevation_scale,
            field,
            origin_generation,
            source_anchor,
            stitch_neighbors,
        } = request;
        let task = AsyncComputeTaskPool::get().spawn(async move {
            PendingMeshPayload {
                mesh: generate_chunk_mesh_stitched(
                    key,
                    radius,
                    elevation_scale,
                    field.as_ref(),
                    stitch_neighbors,
                ),
                key,
                source_anchor,
                origin_generation,
            }
        });
        pending.0.insert(entity, task);
    }

    profiler.record("dispatch_meshes", t0.elapsed());
}

fn mesh_dispatch_slots(max_inflight: usize, inflight: usize) -> usize {
    max_inflight.saturating_sub(inflight)
}

fn can_admit_split_meshes(
    required_for_focus: bool,
    outstanding_meshes: usize,
    max_inflight: usize,
) -> bool {
    required_for_focus || outstanding_meshes.saturating_add(4) <= max_inflight.saturating_mul(2)
}

/// The four siblings replacing the camera-containing parent are one critical
/// handoff: prioritize all of them so the visible parent can retire promptly.
fn sort_chunk_mesh_requests(
    requests: &mut [ChunkMeshRequest],
    camera_pos: DVec3,
    planet_radius: f64,
) {
    let camera_dir = camera_pos.normalize_or(DVec3::Y);
    requests.sort_by(|a, b| {
        let critical_a = cell_contains_direction(a.key, camera_dir)
            || parent_of(a.key).is_some_and(|parent| cell_contains_direction(parent, camera_dir));
        let critical_b = cell_contains_direction(b.key, camera_dir)
            || parent_of(b.key).is_some_and(|parent| cell_contains_direction(parent, camera_dir));
        let err_a = screen_error(a.key, camera_pos, planet_radius);
        let err_b = screen_error(b.key, camera_pos, planet_radius);
        critical_b
            .cmp(&critical_a)
            .then_with(|| err_b.total_cmp(&err_a))
            .then_with(|| a.key.face.cmp(&b.key.face))
            .then_with(|| a.key.lod.cmp(&b.key.lod))
            .then_with(|| a.key.i.cmp(&b.key.i))
            .then_with(|| a.key.j.cmp(&b.key.j))
    });
}

fn stitch_neighbors(key: CellKey, active_chunks: &ActiveChunks) -> StitchNeighbors {
    NEIGHBOR_SIDES.map(|side| {
        coarser_neighbor_across_edge(key, side, active_chunks)
            .filter(|neighbor| neighbor.lod + 1 == key.lod)
    })
}

/// Return the coarsest adjacent leaf that must split before `key` can split.
/// This preserves the 2:1 leaf-depth invariant incrementally, including across
/// cube-face edges, without increasing the active chunk cap.
fn balancing_split_for(key: CellKey, active_chunks: &ActiveChunks) -> Option<CellKey> {
    NEIGHBOR_SIDES
        .into_iter()
        .filter_map(|side| coarser_neighbor_across_edge(key, side, active_chunks))
        .min_by_key(|neighbor| (neighbor.lod, neighbor.face, neighbor.i, neighbor.j))
}

fn stitch_neighbor_depths(key: CellKey, neighbors: StitchNeighbors) -> [u8; 4] {
    neighbors.map(|neighbor| neighbor.map_or(key.lod, |neighbor| neighbor.lod))
}

#[allow(clippy::too_many_arguments)]
fn queue_stitch_rebuilds(
    mut chunks: Query<(&mut ChunkComponent, Option<&Mesh3d>)>,
    active_chunks: Res<ActiveChunks>,
    terrain_state: Res<TerrainState>,
    origin: Res<RenderOrigin>,
    mut queued_meshes: ResMut<QueuedChunkMeshes>,
    pending: Res<PendingChunkMeshes>,
) {
    for (&key, &entity) in &active_chunks.chunks {
        if pending.0.contains_key(&entity) || queued_meshes.contains(entity) {
            continue;
        }
        let Ok((mut chunk, mesh)) = chunks.get_mut(entity) else {
            continue;
        };
        if mesh.is_none() {
            continue;
        }

        let neighbors = stitch_neighbors(key, &active_chunks);
        let depths = stitch_neighbor_depths(key, neighbors);
        if chunk.neighbor_depth == depths {
            continue;
        }

        chunk.neighbor_depth = depths;
        queue_chunk_mesh(
            &mut queued_meshes,
            entity,
            key,
            terrain_state.planet_radius,
            terrain_state.elevation_scale,
            Arc::clone(&terrain_state.field),
            origin.generation,
            neighbors,
        );
    }
}

fn despawn_chunk(commands: &mut Commands, entity: Entity) {
    if let Ok(mut ec) = commands.get_entity(entity) {
        ec.despawn();
    }
}

fn apply_pending_chunk_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut pending: ResMut<PendingChunkMeshes>,
    chunk_query: Query<&ChunkComponent>,
    origin: Res<RenderOrigin>,
    mut debug: ResMut<TerrainDebugInfo>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let mut done = Vec::new();
    let mut meshes_applied = 0usize;
    let mut cross_gen_attaches = 0usize;
    for (&entity, task) in &mut pending.0 {
        if let Some(payload) = check_ready(task) {
            let generation_is_current = payload.origin_generation == origin.generation;
            if chunk_query
                .get(entity)
                .is_ok_and(|chunk| pending_mesh_matches_chunk(chunk.key, payload.key))
            {
                let handle = meshes.add(payload.mesh);
                if let Ok(mut e) = commands.get_entity(entity) {
                    // Mesh vertices are anchor-local, so a product generated
                    // before an origin shift remains valid when attached using
                    // the current origin-relative anchor transform.
                    e.insert((
                        Mesh3d(handle),
                        Transform::from_translation(anchor_render_translation(
                            payload.source_anchor,
                            origin.world,
                        )),
                    ));
                    meshes_applied += 1;
                    if !generation_is_current {
                        cross_gen_attaches += 1;
                        trace!(
                            entity = ?entity,
                            queued_generation = payload.origin_generation,
                            current_generation = origin.generation,
                            "Attached origin-invariant chunk mesh after rebasing its anchor"
                        );
                    }
                }
            }
            done.push(entity);
        }
    }
    for entity in done {
        pending.0.remove(&entity);
    }
    debug.meshes_built = meshes_applied;
    debug.cross_generation_mesh_attaches = cross_gen_attaches;
    profiler.record("apply_meshes", t0.elapsed());
}

fn pending_mesh_matches_chunk(entity_key: CellKey, payload_key: CellKey) -> bool {
    entity_key == payload_key
}

fn anchor_render_translation(source_anchor: DVec3, render_origin: DVec3) -> Vec3 {
    (source_anchor - render_origin).as_vec3()
}

/// Finalize retained splits: once all four children of a retained parent have
/// meshes, reveal the children and despawn the parent atomically (same frame).
/// This is the no-gap / no-overlap handoff — the parent was the visible
/// fallback right up to the frame its replacement coverage is ready.
fn finalize_retirements(
    mut commands: Commands,
    mut retained: ResMut<RetainedSplits>,
    mesh_query: Query<(), With<Mesh3d>>,
    chunk_query: Query<&ChunkComponent>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let mut done = Vec::new();
    // First pass: collect which entries can finalize and which parents must be
    // despawned. We need the despawn set before queueing any reveal commands so
    // we don't try to reveal an entity that is being despawned in the same
    // command batch.
    let mut despawn_parents = std::collections::HashSet::<Entity>::new();
    for (&key, entry) in retained.map.iter() {
        if entry.children.iter().all(|&c| mesh_query.get(c).is_ok()) {
            if chunk_query.get(entry.parent_entity).is_ok() && !is_minimum_coverage_chunk(key) {
                despawn_parents.insert(entry.parent_entity);
            }
            done.push(key);
        }
    }

    for &key in &done {
        let entry = retained.map.get(&key).unwrap();
        if despawn_parents.contains(&entry.parent_entity) {
            if let Ok(mut parent_cmd) = commands.get_entity(entry.parent_entity) {
                parent_cmd.despawn();
            }
        } else if is_minimum_coverage_chunk(key) {
            if let Ok(mut parent_cmd) = commands.get_entity(entry.parent_entity) {
                parent_cmd.insert(FarCoverageRoot);
            }
        }
        for &c in &entry.children {
            if !despawn_parents.contains(&c) {
                if let Ok(mut child_cmd) = commands.get_entity(c) {
                    child_cmd.remove::<HoldHidden>().insert(Visibility::Visible);
                }
            }
        }
    }

    for key in done {
        retained.map.remove(&key);
    }
    profiler.record("finalize_retirements", t0.elapsed());
}

/// Finalize retained merges: once the newly-spawned (coarser) parent has a
/// mesh, reveal it and despawn the four fallback children atomically in the
/// same frame. The children were kept visible as the fallback while the
/// parent's mesh generated, so there is no gap.
fn finalize_retained_merges(
    mut commands: Commands,
    mut retained_merges: ResMut<RetainedMerges>,
    mesh_query: Query<(), With<Mesh3d>>,
    chunk_query: Query<&ChunkComponent>,
    mut active_chunks: ResMut<ActiveChunks>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let mut done = Vec::new();
    let mut despawn_children: HashSet<Entity> = HashSet::new();

    for (&key, entry) in retained_merges.map.iter() {
        if mesh_query.get(entry.parent_entity).is_ok() {
            if chunk_query.get(entry.parent_entity).is_ok() {
                for &c in &entry.children {
                    if c != Entity::PLACEHOLDER {
                        despawn_children.insert(c);
                    }
                }
            }
            done.push(key);
        }
    }

    for &key in &done {
        let entry = retained_merges.map.get(&key).unwrap();
        // An inner merge can finalize alongside its coarser parent. The inner
        // parent is then a child scheduled for despawn, so revealing it would
        // queue an insert/remove command against a dead entity.
        if !despawn_children.contains(&entry.parent_entity) {
            if let Ok(mut e) = commands.get_entity(entry.parent_entity) {
                e.remove::<HoldForMerge>().insert(Visibility::Visible);
            }
            if !active_chunks.contains(&key) {
                active_chunks.insert(key, entry.parent_entity);
            }
        }
        // Despawn the fallback children.
        for &c in &entry.children {
            if c != Entity::PLACEHOLDER && despawn_children.contains(&c) {
                if let Ok(mut child_cmd) = commands.get_entity(c) {
                    child_cmd.despawn();
                }
            }
        }
    }

    for key in done {
        retained_merges.map.remove(&key);
    }
    profiler.record("finalize_retained_merges", t0.elapsed());
}

fn has_merged_ancestor(key: CellKey, merged_keys: &HashSet<CellKey>) -> bool {
    merged_keys
        .iter()
        .any(|&merged_key| is_ancestor_of(merged_key, key))
}

fn is_child_of_retained_split(key: CellKey, retained: &RetainedSplits) -> bool {
    parent_of(key).is_some_and(|parent| retained.map.contains_key(&parent))
}

fn is_ancestor_of(ancestor_key: CellKey, key: CellKey) -> bool {
    let mut ancestor = parent_of(key);
    while let Some(parent) = ancestor {
        if parent == ancestor_key {
            return true;
        }
        ancestor = parent_of(parent);
    }
    false
}

fn has_conflicting_merge(key: CellKey, merged_keys: &HashSet<CellKey>) -> bool {
    has_merged_ancestor(key, merged_keys)
        || merged_keys
            .iter()
            .any(|&merged_key| is_ancestor_of(key, merged_key))
}

/// Sort pending splits by descending `screen_error` with deterministic
/// `(face, lod, i, j)` tie-breaks so the highest-visual-benefit splits are
/// processed first within each frame's budget, independent of hash-map order.
fn sort_pending_splits_by_screen_error(
    pending: &mut Vec<CellKey>,
    camera_pos: DVec3,
    planet_radius: f64,
) {
    let camera_dir = camera_pos.normalize_or(DVec3::Y);
    pending.sort_by(|a, b| {
        let focus_a = cell_contains_direction(*a, camera_dir);
        let focus_b = cell_contains_direction(*b, camera_dir);
        let err_a = screen_error(*a, camera_pos, planet_radius);
        let err_b = screen_error(*b, camera_pos, planet_radius);
        focus_b
            .cmp(&focus_a)
            .then_with(|| err_b.total_cmp(&err_a))
            .then_with(|| a.face.cmp(&b.face))
            .then_with(|| a.lod.cmp(&b.lod))
            .then_with(|| a.i.cmp(&b.i))
            .then_with(|| a.j.cmp(&b.j))
    });
}

fn cell_contains_direction(key: CellKey, direction: DVec3) -> bool {
    er_core::math::dir_to_cell(direction, key.lod) == key
}

/// Walk the quadtree from the finest cell containing `cam_dir` upward, returning
/// the deepest ancestor present in `active_chunks` or `retained_splits`. Falls
/// back to the face root when no active ancestor covers the camera direction.
fn containing_ancestor_key(
    cam_dir: DVec3,
    max_depth: u8,
    active_chunks: &ActiveChunks,
    retained_splits: &RetainedSplits,
) -> CellKey {
    let finest = er_core::math::dir_to_cell(cam_dir, max_depth);
    let mut current = Some(finest);
    while let Some(key) = current {
        if active_chunks.contains(&key) || retained_splits.map.contains_key(&key) {
            return key;
        }
        current = parent_of(key);
    }
    CellKey {
        face: finest.face,
        i: 0,
        j: 0,
        lod: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        anchor_render_translation, balancing_split_for, breadth_chunk_cap, camera_clip_planes,
        can_admit_split_meshes, chunk_render_translation, containing_ancestor_key,
        has_conflicting_merge, has_merged_ancestor, hide_far_coverage_root,
        is_child_of_retained_split, mesh_dispatch_slots, pending_mesh_matches_chunk,
        queue_chunk_mesh, sort_chunk_mesh_requests, sort_pending_splits_by_screen_error,
        stitch_neighbors, ChunkMeshRequest, QueuedChunkMeshes, RenderOrigin, TerrainPlugin,
        TerrainState,
    };
    use crate::culling::frustum_cull_sphere;
    use crate::lod::screen_error;
    use crate::quadtree::{parent_of, ActiveChunks, RetainedSplit, RetainedSplits};
    use bevy::prelude::{Entity, World};
    use er_core::config::{
        PlanetPreset, CHUNK_QUADS_PER_EDGE, EARTH_RADIUS_M, MAX_INFLIGHT_TERRAIN_MESHES,
    };
    use er_core::math::{
        cell_neighbor, cell_size, cell_to_dir, dir_to_cell, CellKey, NeighborSide,
    };
    use glam::{DVec3, Vec3};
    use std::collections::HashSet;

    fn mesh_request(entity: Entity, key: CellKey, state: &TerrainState) -> ChunkMeshRequest {
        ChunkMeshRequest {
            entity,
            key,
            radius: state.planet_radius,
            elevation_scale: state.elevation_scale,
            field: state.field.clone(),
            origin_generation: 0,
            source_anchor: cell_to_dir(key) * state.planet_radius,
            stitch_neighbors: [None; 4],
        }
    }

    #[test]
    fn mesh_dispatch_slots_enforce_the_inflight_cap() {
        assert_eq!(mesh_dispatch_slots(MAX_INFLIGHT_TERRAIN_MESHES, 0), 64);
        assert_eq!(mesh_dispatch_slots(MAX_INFLIGHT_TERRAIN_MESHES, 63), 1);
        assert_eq!(mesh_dispatch_slots(MAX_INFLIGHT_TERRAIN_MESHES, 64), 0);
        assert_eq!(mesh_dispatch_slots(MAX_INFLIGHT_TERRAIN_MESHES, 80), 0);
    }

    #[test]
    fn breadth_mesh_admission_applies_backpressure_but_never_blocks_focus() {
        assert!(can_admit_split_meshes(false, 124, 64));
        assert!(!can_admit_split_meshes(false, 125, 64));
        assert!(can_admit_split_meshes(true, usize::MAX, 64));
    }

    #[test]
    fn queued_mesh_request_for_an_entity_is_superseded() {
        let state = TerrainState::for_preset(
            PlanetPreset::EarthScale,
            1000.0,
            er_core::seed::PlanetSeed(7),
        );
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let first = CellKey {
            face: 0,
            i: 2,
            j: 3,
            lod: 4,
        };
        let second = CellKey { i: 3, ..first };
        let mut queue = QueuedChunkMeshes::default();

        queue_chunk_mesh(
            &mut queue,
            entity,
            first,
            state.planet_radius,
            state.elevation_scale,
            state.field.clone(),
            1,
            [None; 4],
        );
        queue_chunk_mesh(
            &mut queue,
            entity,
            second,
            state.planet_radius,
            state.elevation_scale,
            state.field.clone(),
            2,
            [None; 4],
        );

        assert_eq!(queue.len(), 1);
        let request = queue.0.get(&entity).unwrap();
        assert_eq!(request.key, second);
        assert_eq!(request.origin_generation, 2);
    }

    #[test]
    fn mesh_dispatch_prioritizes_all_camera_handoff_siblings() {
        let state = TerrainState::for_preset(
            PlanetPreset::EarthScale,
            1000.0,
            er_core::seed::PlanetSeed(11),
        );
        let camera_pos = DVec3::new(EARTH_RADIUS_M + 100.0, 0.0, 0.0);
        let focused_parent = dir_to_cell(camera_pos, 8);
        let focused_children = crate::quadtree::children_of(focused_parent);
        let breadth = CellKey {
            face: 1,
            i: 0,
            j: 0,
            lod: 1,
        };
        let mut world = World::new();
        let breadth_entity = world.spawn_empty().id();
        let mut requests = vec![mesh_request(breadth_entity, breadth, &state)];
        requests.extend(focused_children.into_iter().map(|key| {
            let entity = world.spawn_empty().id();
            mesh_request(entity, key, &state)
        }));

        sort_chunk_mesh_requests(&mut requests, camera_pos, EARTH_RADIUS_M);

        assert!(requests[..4]
            .iter()
            .all(|request| parent_of(request.key) == Some(focused_parent)));
        assert_eq!(requests[4].key, breadth);
    }

    #[test]
    fn render_origin_cell_size_is_configurable() {
        let origin = RenderOrigin::with_cell_size(250.0);
        let plugin = TerrainPlugin::default().with_render_origin_cell_size(500.0);

        assert_eq!(origin.cell_size_m, 250.0);
        assert_eq!(plugin.render_origin_cell_size_m, 500.0);
    }

    #[test]
    #[should_panic]
    fn render_origin_rejects_non_positive_cell_size() {
        let _ = RenderOrigin::with_cell_size(0.0);
    }

    #[test]
    fn detects_merged_ancestor_for_nested_merge() {
        let grandparent = CellKey {
            face: 0,
            i: 0,
            j: 0,
            lod: 0,
        };
        let parent = CellKey {
            face: 0,
            i: 1,
            j: 0,
            lod: 1,
        };
        let mut merged = HashSet::new();
        merged.insert(grandparent);

        assert!(has_merged_ancestor(parent, &merged));
        assert!(!has_merged_ancestor(grandparent, &merged));

        merged.clear();
        merged.insert(parent);
        assert!(has_conflicting_merge(grandparent, &merged));
    }

    #[test]
    fn nested_split_waits_for_parent_handoff() {
        let parent = CellKey {
            face: 0,
            i: 2,
            j: 3,
            lod: 4,
        };
        let children = crate::quadtree::children_of(parent);
        let mut retained = RetainedSplits::default();
        retained.map.insert(
            parent,
            RetainedSplit {
                parent_entity: Entity::PLACEHOLDER,
                children: [Entity::PLACEHOLDER; 4],
            },
        );

        assert!(is_child_of_retained_split(children[0], &retained));
        assert!(!is_child_of_retained_split(parent, &retained));
        retained.map.remove(&parent);
        assert!(!is_child_of_retained_split(children[0], &retained));
    }

    #[test]
    fn chunk_anchor_round_trips_through_nearby_render_origin() {
        let key = CellKey {
            face: 0,
            i: 65_535,
            j: 65_536,
            lod: 17,
        };
        let radius = 6_371_000.0;
        let anchor = cell_to_dir(key) * radius;
        let origin = (anchor / 1000.0).floor() * 1000.0;
        let render = chunk_render_translation(key, radius, origin);
        let reconstructed = origin + render.as_dvec3();
        assert!((reconstructed - anchor).length() < 0.001);
    }

    #[test]
    fn frustum_result_is_invariant_under_common_origin_shift() {
        let camera = Vec3::new(100.0, 0.0, 0.0);
        let sphere = Vec3::new(50.0, 0.0, -500.0);
        let origin = Vec3::new(6_371_000.0, 2_000.0, -3_000.0);
        let args = (Vec3::NEG_Z, Vec3::X, Vec3::Y, 0.8, 16.0 / 9.0);
        let world_result = frustum_cull_sphere(
            sphere + origin,
            25.0,
            camera + origin,
            args.0,
            args.1,
            args.2,
            args.3,
            args.4,
        );
        let render_result = frustum_cull_sphere(
            sphere,
            camera.x * 0.0 + 25.0,
            camera,
            args.0,
            args.1,
            args.2,
            args.3,
            args.4,
        );
        assert_eq!(world_result, render_result);
    }

    #[test]
    fn pending_mesh_key_must_match_live_entity_key() {
        let key = CellKey {
            face: 2,
            i: 7,
            j: 9,
            lod: 5,
        };
        let other = CellKey { i: 8, ..key };
        assert!(pending_mesh_matches_chunk(key, key));
        assert!(!pending_mesh_matches_chunk(key, other));

        let stale_origin = DVec3::new(1000.0, 0.0, 0.0);
        let current_origin = DVec3::new(2000.0, 0.0, 0.0);
        assert_ne!(
            chunk_render_translation(key, 6_371_000.0, stale_origin),
            chunk_render_translation(key, 6_371_000.0, current_origin)
        );
    }

    #[test]
    fn stale_generation_anchor_rebases_to_the_same_absolute_position() {
        let source_anchor = DVec3::new(6_371_123.25, -4_500.5, 8_900.75);
        let queued_origin = DVec3::new(6_371_000.0, -5_000.0, 8_000.0);
        let current_origin = DVec3::new(6_372_000.0, -4_000.0, 9_000.0);

        assert_ne!(
            anchor_render_translation(source_anchor, queued_origin),
            anchor_render_translation(source_anchor, current_origin)
        );
        let reconstructed =
            current_origin + anchor_render_translation(source_anchor, current_origin).as_dvec3();
        assert!((reconstructed - source_anchor).length() < 0.001);
    }

    #[test]
    fn far_coverage_roots_are_hidden_until_normal_coverage_expires() {
        let max_render_distance = 50_968_000.0;
        assert!(hide_far_coverage_root(6_371_010.0, max_render_distance));
        assert!(hide_far_coverage_root(
            max_render_distance,
            max_render_distance
        ));
        assert!(!hide_far_coverage_root(
            max_render_distance + 1_024.0,
            max_render_distance
        ));
    }

    #[test]
    fn stale_refresh_key_is_not_compatible_with_reused_entity() {
        let queued = CellKey {
            face: 1,
            i: 3,
            j: 4,
            lod: 6,
        };
        let current = CellKey { j: 5, ..queued };
        assert!(!pending_mesh_matches_chunk(current, queued));
    }

    #[test]
    fn focused_split_selects_coarse_neighbor_before_deepening() {
        let key = CellKey {
            face: 0,
            i: 0,
            j: 2,
            lod: 3,
        };
        let ancestor = parent_of(parent_of(key).unwrap()).unwrap();
        let coarse = cell_neighbor(ancestor, NeighborSide::NegU);
        let mut active = ActiveChunks::default();
        active.insert(key, Entity::PLACEHOLDER);
        active.insert(coarse, Entity::PLACEHOLDER);

        assert_eq!(balancing_split_for(key, &active), Some(coarse));
    }

    #[test]
    fn stitch_contract_rejects_multi_level_neighbor() {
        let key = CellKey {
            face: 0,
            i: 0,
            j: 2,
            lod: 3,
        };
        let ancestor = parent_of(parent_of(key).unwrap()).unwrap();
        let coarse = cell_neighbor(ancestor, NeighborSide::NegU);
        let mut active = ActiveChunks::default();
        active.insert(key, Entity::PLACEHOLDER);
        active.insert(coarse, Entity::PLACEHOLDER);

        assert!(stitch_neighbors(key, &active)
            .into_iter()
            .all(|n| n.is_none()));
    }

    #[test]
    fn earth_scale_close_view_keeps_the_limb_inside_the_far_plane() {
        let earth_radius = 6_371_000.0;
        let (_, far) = camera_clip_planes(earth_radius, 6_400_000.0);

        assert!(far > 12_000_000.0);
    }

    #[test]
    fn sorts_splits_by_descending_screen_error_with_deterministic_ties() {
        let camera_pos = DVec3::new(EARTH_RADIUS_M * 1.1, 0.0, 0.0);
        let radius = EARTH_RADIUS_M;

        let high_error = CellKey {
            face: 0,
            i: 0,
            j: 0,
            lod: 10,
        };
        let low_error = CellKey {
            face: 0,
            i: 0,
            j: 0,
            lod: 12,
        };
        assert!(
            screen_error(high_error, camera_pos, radius)
                > screen_error(low_error, camera_pos, radius),
            "coarser LOD must have higher screen error"
        );

        for order in [vec![low_error, high_error], vec![high_error, low_error]] {
            let mut pending = order.clone();
            sort_pending_splits_by_screen_error(&mut pending, camera_pos, radius);
            assert_eq!(
                pending[0], high_error,
                "high-error key sorts first (input {order:?})"
            );
            assert_eq!(pending.len(), 2, "all inputs remain");
        }

        // tie-break: equal error (symmetric faces from Y-axis camera)
        let sym_cam = DVec3::new(0.0, EARTH_RADIUS_M * 1.1, 0.0);
        let a = CellKey {
            face: 0,
            i: 0,
            j: 0,
            lod: 8,
        };
        let b = CellKey {
            face: 1,
            i: 0,
            j: 0,
            lod: 8,
        };
        let err_a = screen_error(a, sym_cam, radius);
        let err_b = screen_error(b, sym_cam, radius);
        assert!(
            (err_a - err_b).abs() < 1e-4,
            "symmetric faces must have equal error"
        );

        let mut pending = vec![b, a];
        sort_pending_splits_by_screen_error(&mut pending, sym_cam, radius);
        assert_eq!(pending[0], a, "lower face sorts first on tie");
    }

    #[test]
    fn containing_camera_path_preempts_higher_error_breadth_split() {
        let camera_pos = DVec3::X * (EARTH_RADIUS_M + 10_000.0);
        let focused = dir_to_cell(camera_pos, 17);
        let breadth = CellKey {
            face: 0,
            i: 15,
            j: 16,
            lod: 5,
        };
        assert!(
            screen_error(breadth, camera_pos, EARTH_RADIUS_M)
                > screen_error(focused, camera_pos, EARTH_RADIUS_M)
        );

        let mut pending = vec![breadth, focused];
        sort_pending_splits_by_screen_error(&mut pending, camera_pos, EARTH_RADIUS_M);
        assert_eq!(pending[0], focused);
    }

    #[test]
    fn breadth_admission_reserves_a_complete_camera_refinement_path() {
        let hard_cap = 5000;
        let max_depth = 17;
        let breadth_cap = breadth_chunk_cap(hard_cap, max_depth);

        assert_eq!(breadth_cap, 4949);
        assert_eq!(breadth_cap + usize::from(max_depth) * 3, hard_cap);
    }

    #[test]
    fn breadth_admission_saturates_for_tiny_caps() {
        assert_eq!(breadth_chunk_cap(32, 17), 0);
    }

    #[test]
    fn lod17_earth_vertex_spacing_is_at_most_5m() {
        let radius = EARTH_RADIUS_M;
        let lod = 17u8;
        let cs = cell_size(lod, radius);
        let vs = cs / CHUNK_QUADS_PER_EDGE as f64;
        assert!(vs <= 5.0, "LOD17 vertex spacing {vs:.3} > 5 m");
        assert!(vs > 0.0);
    }

    #[test]
    fn containing_ancestor_finds_active_ancestor_under_largest_detail() {
        let mut active = ActiveChunks::default();
        let parent = CellKey {
            face: 0,
            i: 12,
            j: 8,
            lod: 11,
        };
        active.insert(parent, Entity::PLACEHOLDER);
        let retained = RetainedSplits::default();

        let cam_dir = cell_to_dir(parent);
        let result = containing_ancestor_key(cam_dir, 17, &active, &retained);
        assert_eq!(result, parent);
    }

    #[test]
    fn containing_ancestor_walks_to_face_root_when_nothing_is_active() {
        let dir = DVec3::new(EARTH_RADIUS_M, 1000.0, -5000.0).normalize();
        let active = ActiveChunks::default();
        let retained = RetainedSplits::default();

        let result = containing_ancestor_key(dir, 17, &active, &retained);
        // Must land on a face root (lod=0, i=0, j=0)
        assert_eq!(result.lod, 0);
        assert_eq!(result.i, 0);
        assert_eq!(result.j, 0);
        assert!(result.face < 6);
        // The face must match the direction's face
        assert_eq!(result.face, dir_to_cell(dir, 0).face);
    }

    #[test]
    fn retained_split_parent_counts_as_containing_ancestor() {
        let parent = CellKey {
            face: 2,
            i: 5,
            j: 5,
            lod: 6,
        };
        let mut retained = RetainedSplits::default();
        retained.map.insert(
            parent,
            RetainedSplit {
                parent_entity: Entity::PLACEHOLDER,
                children: [
                    Entity::PLACEHOLDER,
                    Entity::PLACEHOLDER,
                    Entity::PLACEHOLDER,
                    Entity::PLACEHOLDER,
                ],
            },
        );
        let active = ActiveChunks::default();

        let cam_dir = cell_to_dir(parent);
        let result = containing_ancestor_key(cam_dir, 10, &active, &retained);
        assert_eq!(result, parent);
    }
}
