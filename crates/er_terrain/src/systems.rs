use bevy::ecs::schedule::ApplyDeferred;
use bevy::ecs::schedule::SystemSet;
use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;
use bevy::shader::Shader;
use bevy::tasks::{futures::check_ready, AsyncComputeTaskPool, Task};
use er_core::config::{
    ACTIVE_CHUNK_CAP, LOD_SPLIT_BUDGET_PER_FRAME, MAX_QUADTREE_DEPTH, MAX_RENDER_DISTANCE,
    MERGE_HYSTERESIS, PLANET_RADIUS_DEFAULT, SCREEN_ERROR_THRESHOLD,
};
use er_core::math::{cell_size, cell_to_dir, CellKey};
use er_core::seed::PlanetSeed;
use er_world::cache::WorldCache;
use er_world::elevation::{elevation_params, ElevationNoise, ElevationParams};
use er_world::params::{climate_noise as make_climate_noise, planet_params as make_planet_params, ClimateNoise, PlanetParams};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::chunk::{ChunkComponent, HoldHidden};
use crate::culling::{frustum_cull_sphere, is_below_horizon, is_outside_render_distance};
use crate::debug::TerrainDebugInfo;
use crate::lod::{chunk_camera_distance, should_merge_parent, should_split};
use crate::material::{TerrainMaterial, TerrainMaterialUniform, FRAGMENT_SHADER, VERTEX_SHADER};
use crate::mesh_gen::generate_chunk_mesh;
use crate::ocean::{OceanMaterial, setup_ocean, update_ocean_time};
use crate::quadtree::{children_of, parent_of, root_chunks, ActiveChunks, RetainedSplit, RetainedSplits};

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
    pub noise: ElevationNoise,
    pub planet_params: PlanetParams,
    pub climate_noise: ClimateNoise,
    pub base_uniform: TerrainMaterialUniform,
    pub max_quadtree_depth: u8,
    pub screen_error_threshold: f32,
    pub merge_hysteresis: f32,
    pub max_render_distance: f64,
    pub active_chunk_cap: usize,
    pub lod_split_budget_per_frame: usize,
    pub cache: Arc<WorldCache>,
}

impl TerrainState {
    pub fn new(planet_radius: f64, elevation_scale: f32, seed: PlanetSeed) -> Self {
        let params = elevation_params(seed);
        let noise = ElevationNoise::new(&params);
        let planet_params = make_planet_params(seed);
        let climate_noise = make_climate_noise(&planet_params);
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
            noise,
            planet_params,
            climate_noise,
            base_uniform,
            max_quadtree_depth: MAX_QUADTREE_DEPTH,
            screen_error_threshold: SCREEN_ERROR_THRESHOLD,
            merge_hysteresis: MERGE_HYSTERESIS,
            max_render_distance: MAX_RENDER_DISTANCE,
            active_chunk_cap: ACTIVE_CHUNK_CAP,
            lod_split_budget_per_frame: LOD_SPLIT_BUDGET_PER_FRAME,
            cache: Arc::new(WorldCache::new(262144)),
        }
    }
}

#[derive(Resource, Default)]
pub struct PendingChunkMeshes(pub HashMap<Entity, Task<Mesh>>);

pub struct TerrainPlugin {
    pub planet_radius: f64,
    pub elevation_scale: f32,
    pub seed: PlanetSeed,
}

impl Default for TerrainPlugin {
    fn default() -> Self {
        Self {
            planet_radius: PLANET_RADIUS_DEFAULT,
            elevation_scale: 1000.0,
            seed: PlanetSeed(0xC0FFEE),
        }
    }
}

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(TerrainState::new(
            self.planet_radius,
            self.elevation_scale,
            self.seed,
        ))
        .insert_resource(ActiveChunks::default())
        .insert_resource(RetainedSplits::default())
        .insert_resource(TerrainDebugInfo::default())
        .insert_resource(PendingChunkMeshes::default())
        .insert_resource(crate::profiler::FrameProfiler::default())
        .insert_resource(SunDirection::default())
        .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
        .add_plugins(MaterialPlugin::<OceanMaterial>::default())
        .add_plugins(MaterialPlugin::<crate::ocean::OcclusionMaterial>::default())
        .add_systems(Startup, (setup_terrain, setup_ocean))
        .add_systems(PreUpdate, profiler_clear)
        .add_systems(
            Update,
            (
                update_lod,
                process_lod_queue,
                ApplyDeferred,
                apply_pending_chunk_meshes,
                ApplyDeferred,
                finalize_retirements,
                ApplyDeferred,
                update_neighbor_lod,
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

    for key in root_chunks() {
        let entity = spawn_chunk_entity(
            &mut commands,
            &mut materials,
            &mut pending,
            &terrain_state.base_uniform,
            key,
            terrain_state.planet_radius,
            &terrain_state.params,
            &terrain_state.planet_params,
            &terrain_state.cache,
        );
        active_chunks.insert(key, entity);
    }
}

fn profiler_clear(mut profiler: ResMut<crate::profiler::FrameProfiler>) {
    profiler.clear();
}

fn update_lod(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut active_chunks: ResMut<ActiveChunks>,
    terrain_state: Res<TerrainState>,
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
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut active_chunks: ResMut<ActiveChunks>,
    mut retained: ResMut<RetainedSplits>,
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
    let base_uniform = terrain_state.base_uniform;
    let budget = terrain_state.lod_split_budget_per_frame;

    let mut splits_done = 0usize;
    let pending_splits: Vec<CellKey> = std::mem::take(&mut active_chunks.pending_splits);
    for key in pending_splits {
        if splits_done >= budget {
            break;
        }
        if active_chunks.len() >= terrain_state.active_chunk_cap {
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
                &mut materials,
                &mut pending,
                &base_uniform,
                child,
                terrain_state.planet_radius,
                &terrain_state.params,
                &terrain_state.planet_params,
                &terrain_state.cache,
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

    let over_cap = active_chunks.len().saturating_sub(terrain_state.active_chunk_cap);
    let merge_budget = if over_cap > 0 {
        (budget * 4).max(over_cap * 4)
    } else {
        budget
    };
    let mut merges_done = 0usize;
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
            merges_done += 1;
            continue;
        }

        // Non-retained merge: despawn children immediately, spawn parent.
        // There will be a brief gap (1-2 frames) until parent mesh arrives,
        // but this is better than persistent black holes.
        for child in &children {
            if let Some(entity) = active_chunks.remove(child) {
                despawn_chunk(&mut commands, entity);
            }
        }

        let entity = spawn_chunk_entity(
            &mut commands,
            &mut materials,
            &mut pending,
            &base_uniform,
            parent_key,
            terrain_state.planet_radius,
            &terrain_state.params,
            &terrain_state.planet_params,
            &terrain_state.cache,
        );
        active_chunks.insert(parent_key, entity);
        merges_done += 1;
    }
    debug.pending_merges = merges_done;

    if active_chunks.len() > terrain_state.active_chunk_cap {
        let mut distances: Vec<(CellKey, f64)> = active_chunks
            .chunks
            .keys()
            .map(|&k| {
                (
                    k,
                    chunk_camera_distance(k, camera_pos, terrain_state.planet_radius),
                )
            })
            .collect();
        distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let to_evict = active_chunks.len() - terrain_state.active_chunk_cap;
        for (key, _) in distances.into_iter().take(to_evict) {
            if active_chunks.contains(&key) {
                if let Some(entity) = active_chunks.remove(&key) {
                    despawn_chunk(&mut commands, entity);
                }
            }
        }
    }
    profiler.record("process_lod_queue", t0.elapsed());
}

fn cull_chunks(
    mut camera_query: Query<(&GlobalTransform, &mut Projection), With<Camera3d>>,
    mut chunk_query: Query<(&ChunkComponent, &mut Visibility), (With<Mesh3d>, Without<HoldHidden>)>,
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
    let (near, far) = if cam_dist < planet_radius * 1.1 {
        (0.1, 500000.0)
    } else if cam_dist < planet_radius * 5.0 {
        (1.0, 500000.0)
    } else {
        (10.0, 5000000.0)
    };
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

    for (chunk, mut visibility) in &mut chunk_query {
        let key = chunk.key;

        if is_outside_render_distance(
            key,
            camera_pos,
            terrain_state.planet_radius,
            terrain_state.max_render_distance,
        ) {
            *visibility = Visibility::Hidden;
            continue;
        }

        if is_below_horizon(key, camera_pos, terrain_state.planet_radius) {
            *visibility = Visibility::Hidden;
            continue;
        }

        if let Some((cam_pos, forward, right, up, fov_cos, aspect)) = frustum {
            let sphere_center = (cell_to_dir(key) * terrain_state.planet_radius).as_vec3();
            let sphere_radius =
                cell_size(key.lod, terrain_state.planet_radius) as f32 + terrain_state.elevation_scale * 3.0;
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
                *visibility = Visibility::Hidden;
                continue;
            }
        }

        *visibility = Visibility::Visible;
    }
    profiler.record("cull_chunks", t0.elapsed());
}

fn update_debug_info(
    active_chunks: Res<ActiveChunks>,
    mut debug: ResMut<TerrainDebugInfo>,
    profiler: Res<crate::profiler::FrameProfiler>,
) {
    debug.active_chunks = active_chunks.len();
    debug.max_depth = active_chunks.chunks.keys().map(|k| k.lod).max().unwrap_or(0);
    debug.frame_time_ms = profiler.total().as_secs_f32() * 1000.0;
}

#[allow(clippy::too_many_arguments)]
fn spawn_chunk_entity(
    commands: &mut Commands,
    materials: &mut Assets<TerrainMaterial>,
    pending: &mut PendingChunkMeshes,
    base_uniform: &TerrainMaterialUniform,
    key: CellKey,
    radius: f64,
    elev_params: &ElevationParams,
    planet_params: &PlanetParams,
    cache: &Arc<WorldCache>,
) -> Entity {
    let material = materials.add(TerrainMaterial {
        uniform: base_uniform.for_chunk(key),
    });

    // Spawn entity without mesh — it will be added by apply_pending_chunk_meshes
    // when the async mesh generation task completes.
    let entity = commands
        .spawn((
            ChunkComponent::new(key),
            MeshMaterial3d(material),
            Transform::default(),
            Visibility::Hidden,
        ))
        .id();

    // Copy/clone data for the async task.
    // ElevationNoise and ClimateNoise don't derive Clone (FastNoiseLite
    // isn't Clone), so we reconstruct both from their params inside the task
    // — both constructors are deterministic from their params.
    let elev_params = *elev_params;
    let planet_params = *planet_params;
    let cache = Arc::clone(cache);

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let noise = ElevationNoise::new(&elev_params);
        let climate_noise = make_climate_noise(&planet_params);
        generate_chunk_mesh(
            key,
            radius,
            &noise,
            &elev_params,
            &planet_params,
            &climate_noise,
            Some(&cache),
        )
    });

    pending.0.insert(entity, task);

    entity
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
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let mut done = Vec::new();
    for (&entity, task) in &mut pending.0 {
        if let Some(mesh) = check_ready(task) {
            if chunk_query.get(entity).is_ok() {
                let handle = meshes.add(mesh);
                if let Ok(mut e) = commands.get_entity(entity) {
                    e.insert(Mesh3d(handle));
                }
            }
            done.push(entity);
        }
    }
    for entity in done {
        pending.0.remove(&entity);
    }
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
            if chunk_query.get(entry.parent_entity).is_ok() {
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

fn update_neighbor_lod(
    active_chunks: Res<ActiveChunks>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut chunk_query: Query<(&mut ChunkComponent, &MeshMaterial3d<TerrainMaterial>, &Visibility), With<Mesh3d>>,
    mut profiler: ResMut<crate::profiler::FrameProfiler>,
) {
    let t0 = Instant::now();
    let sides = [
        er_core::math::NeighborSide::NegU,
        er_core::math::NeighborSide::PosU,
        er_core::math::NeighborSide::NegV,
        er_core::math::NeighborSide::PosV,
    ];
    for (mut chunk, mat_handle, vis) in &mut chunk_query {
        if *vis == Visibility::Hidden {
            continue;
        }
        let key = chunk.key;
        let mut nd = [key.lod; 4];
        for (i, side) in sides.iter().enumerate() {
            nd[i] = crate::quadtree::neighbor_lod_across_edge(key, *side, &active_chunks);
        }
        let prev = chunk.neighbor_depth;
        chunk.neighbor_depth = nd;
        if nd != prev {
            if let Some(mut mat) = materials.get_mut(&mat_handle.0) {
                mat.uniform.neighbor_depth_0 = nd[0] as f32;
                mat.uniform.neighbor_depth_1 = nd[1] as f32;
                mat.uniform.neighbor_depth_2 = nd[2] as f32;
                mat.uniform.neighbor_depth_3 = nd[3] as f32;
            }
        }
    }
    profiler.record("neighbor_lod", t0.elapsed());
}
