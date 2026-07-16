use bevy::camera::visibility::NoFrustumCulling;
use bevy::ecs::schedule::ApplyDeferred;
use bevy::ecs::schedule::SystemSet;
use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;
use bevy::shader::Shader;
use bevy::tasks::{futures::check_ready, AsyncComputeTaskPool, Task};
use er_core::config::{
    PlanetPreset, ACTIVE_CHUNK_CAP, LOD_SPLIT_BUDGET_PER_FRAME, MERGE_HYSTERESIS,
    PLANET_RADIUS_DEFAULT, SCREEN_ERROR_THRESHOLD,
};
use er_core::math::{cell_size, cell_to_dir, CellKey};
use er_core::seed::PlanetSeed;
use er_world::cache::WorldCache;
use er_world::elevation::{elevation_params, ElevationParams};
use er_world::params::{planet_params as make_planet_params, PlanetParams};
use er_world::terrain_field::{
    HybridTerrainField, MacroTerrainField, ProceduralTerrainField, TerrainField, TerrainSourceMode,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use crate::chunk::{ChunkComponent, HoldForMerge, HoldHidden};
use crate::culling::{
    frustum_cull_sphere, is_below_horizon, is_beyond_render_distance, is_minimum_coverage_chunk,
};
use crate::debug::TerrainDebugInfo;
use crate::lod::{chunk_camera_distance, should_merge_parent, should_split};
use crate::material::{TerrainMaterial, TerrainMaterialUniform, FRAGMENT_SHADER, VERTEX_SHADER};
use crate::mesh_gen::generate_chunk_mesh;
use crate::ocean::{setup_ocean, update_ocean_time, OceanMaterial};
use crate::quadtree::{
    children_of, parent_of, root_chunks, ActiveChunks, RetainedMerge, RetainedMerges,
    RetainedSplit, RetainedSplits,
};

#[derive(Resource, Clone, Copy)]
pub struct SunDirection(pub Vec3);

impl Default for SunDirection {
    fn default() -> Self {
        Self(Vec3::new(0.5, 0.8, 0.3).normalize())
    }
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
pub struct PendingChunkMeshes(pub HashMap<Entity, Task<Mesh>>);

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
    macro_field: Option<Arc<dyn MacroTerrainField>>,
}

impl Default for TerrainPlugin {
    fn default() -> Self {
        Self {
            planet_radius: PLANET_RADIUS_DEFAULT,
            elevation_scale: 1000.0,
            seed: PlanetSeed(0xC0FFEE),
            preset: PlanetPreset::default(),
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

    /// Use `PlanetPreset` parameters (radius, LOD depth, render distance) while
    /// keeping the caller's `elevation_scale` and `seed`.
    pub fn from_preset(preset: PlanetPreset, elevation_scale: f32, seed: PlanetSeed) -> Self {
        Self {
            planet_radius: preset.radius_m(),
            elevation_scale,
            seed,
            preset,
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
        .insert_resource(TerrainFieldRefresh::default())
        .insert_resource(crate::profiler::FrameProfiler::default())
        .insert_resource(SunDirection::default())
        .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
        .add_plugins(MaterialPlugin::<OceanMaterial>::default())
        .add_systems(Startup, (setup_terrain, setup_ocean))
        .add_systems(PreUpdate, profiler_clear)
        .add_systems(
            Update,
            (
                update_lod,
                queue_resident_field_refreshes,
                process_lod_queue,
                ApplyDeferred,
                apply_pending_chunk_meshes,
                ApplyDeferred,
                finalize_retirements,
                ApplyDeferred,
                finalize_retained_merges,
                ApplyDeferred,
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
    mut pending: ResMut<PendingChunkMeshes>,
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
            &mut pending,
            &material,
            key,
            terrain_state.planet_radius,
            terrain_state.elevation_scale,
            Arc::clone(&terrain_state.field),
        );
        active_chunks.insert(key, entity);
    }
}

fn profiler_clear(mut profiler: ResMut<crate::profiler::FrameProfiler>) {
    profiler.clear();
}

fn queue_resident_field_refreshes(
    terrain_state: Res<TerrainState>,
    mut refresh: ResMut<TerrainFieldRefresh>,
    chunk_query: Query<(Entity, &ChunkComponent)>,
    mut pending: ResMut<PendingChunkMeshes>,
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
        if pending.0.contains_key(&entity) {
            // Let the in-flight mesh finish, then rebuild it from the newest
            // resident field data on a later frame.
            refresh.queued.push_back((entity, key));
            continue;
        }
        queue_chunk_mesh(
            &mut pending,
            entity,
            key,
            terrain_state.planet_radius,
            terrain_state.elevation_scale,
            Arc::clone(&terrain_state.field),
        );
    }
}

fn update_lod(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut active_chunks: ResMut<ActiveChunks>,
    terrain_state: Res<TerrainState>,
    retained_splits: Res<RetainedSplits>,
    retained_merges: Res<RetainedMerges>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let Ok(camera_transform) = camera_query.single() else {
        profiler.record("update_lod", t0.elapsed());
        return;
    };
    let camera_pos = camera_transform.translation().as_dvec3();

    let keys: Vec<CellKey> = active_chunks.chunks.keys().copied().collect();
    active_chunks.clear_pending();

    for &key in &keys {
        if is_below_horizon(key, camera_pos, terrain_state.planet_radius) {
            continue;
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
            active_chunks.pending_splits.push(key);
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
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mesh_query: Query<(), With<Mesh3d>>,
    mut debug: ResMut<TerrainDebugInfo>,
    mut pending: ResMut<PendingChunkMeshes>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };
    let camera_pos = camera_transform.translation().as_dvec3();
    let budget = terrain_state.lod_split_budget_per_frame;

    let mut splits_done = 0usize;
    let pending_splits: Vec<CellKey> = std::mem::take(&mut active_chunks.pending_splits);
    for key in pending_splits {
        if splits_done >= budget {
            break;
        }
        // A split replaces one active parent with four children. Enforce the
        // budget before the replacement so every active terrain region remains
        // covered; evicting live chunks after the fact creates permanent holes.
        if active_chunks.len() + 3 > terrain_state.active_chunk_cap {
            break;
        }
        if !active_chunks.contains(&key) {
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
            let entity = spawn_chunk_entity(
                &mut commands,
                &mut pending,
                &terrain_material.0,
                child,
                terrain_state.planet_radius,
                terrain_state.elevation_scale,
                Arc::clone(&terrain_state.field),
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
            &mut pending,
            &terrain_material.0,
            parent_key,
            terrain_state.planet_radius,
            terrain_state.elevation_scale,
            Arc::clone(&terrain_state.field),
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
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let Ok((camera_transform, mut projection)) = camera_query.single_mut() else {
        profiler.record("cull_chunks", t0.elapsed());
        return;
    };
    let camera_pos = camera_transform.translation().as_dvec3();

    let planet_radius = terrain_state.planet_radius as f32;
    let cam_dist = camera_transform.translation().length();
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
        if far_coverage_root.is_some() && cam_dist <= terrain_state.max_render_distance as f32 {
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
            let sphere_center = chunk_center.as_vec3();
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

fn update_debug_info(
    active_chunks: Res<ActiveChunks>,
    pending: Res<PendingChunkMeshes>,
    chunk_query: Query<&Visibility, With<ChunkComponent>>,
    mesh_query: Query<(), (With<ChunkComponent>, With<Mesh3d>)>,
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
    debug.pending_meshes = pending.0.len();
    debug.visible_chunks = chunk_query
        .iter()
        .filter(|visibility| matches!(visibility, Visibility::Visible))
        .count();
    debug.estimated_mesh_bytes =
        mesh_query.iter().count() * crate::debug::ESTIMATED_BYTES_PER_CHUNK_MESH;
    debug.frame_time_ms = profiler.total().as_secs_f32() * 1000.0;
}

#[allow(clippy::too_many_arguments)]
fn spawn_chunk_entity(
    commands: &mut Commands,
    pending: &mut PendingChunkMeshes,
    material: &Handle<TerrainMaterial>,
    key: CellKey,
    radius: f64,
    elevation_scale: f32,
    field: Arc<dyn TerrainField>,
) -> Entity {
    // Spawn entity without mesh — it will be added by apply_pending_chunk_meshes
    // when the async mesh generation task completes.
    let entity = commands
        .spawn((
            ChunkComponent::new(key),
            MeshMaterial3d(material.clone()),
            Transform::default(),
            Visibility::Hidden,
            // The custom culler includes horizon, distance, and frustum tests.
            // Avoid repeating Bevy's AABB frustum walk for every terrain chunk.
            NoFrustumCulling,
        ))
        .id();

    queue_chunk_mesh(pending, entity, key, radius, elevation_scale, field);

    entity
}

fn queue_chunk_mesh(
    pending: &mut PendingChunkMeshes,
    entity: Entity,
    key: CellKey,
    radius: f64,
    elevation_scale: f32,
    field: Arc<dyn TerrainField>,
) {
    let task = AsyncComputeTaskPool::get()
        .spawn(async move { generate_chunk_mesh(key, radius, elevation_scale, field.as_ref()) });
    pending.0.insert(entity, task);
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
    mut debug: ResMut<TerrainDebugInfo>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let mut done = Vec::new();
    let mut meshes_applied = 0usize;
    for (&entity, task) in &mut pending.0 {
        if let Some(mesh) = check_ready(task) {
            if chunk_query.get(entity).is_ok() {
                let handle = meshes.add(mesh);
                if let Ok(mut e) = commands.get_entity(entity) {
                    e.insert(Mesh3d(handle));
                    meshes_applied += 1;
                }
            }
            done.push(entity);
        }
    }
    for entity in done {
        pending.0.remove(&entity);
    }
    debug.meshes_built = meshes_applied;
    profiler.record("apply_meshes", t0.elapsed());
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

#[cfg(test)]
mod tests {
    use super::{camera_clip_planes, has_conflicting_merge, has_merged_ancestor};
    use er_core::math::CellKey;
    use std::collections::HashSet;

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
    fn earth_scale_close_view_keeps_the_limb_inside_the_far_plane() {
        let earth_radius = 6_371_000.0;
        let (_, far) = camera_clip_planes(earth_radius, 6_400_000.0);

        assert!(far > 12_000_000.0);
    }
}
