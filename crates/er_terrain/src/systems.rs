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
use er_world::elevation::{elevation_params, ElevationParams};
use std::collections::{HashMap, HashSet};

use crate::chunk::ChunkComponent;
use crate::culling::{frustum_cull_sphere, is_below_horizon, is_outside_render_distance};
use crate::debug::TerrainDebugInfo;
use crate::lod::{chunk_camera_distance, should_merge_parent, should_split};
use crate::material::{TerrainMaterial, TerrainMaterialUniform, FRAGMENT_SHADER, VERTEX_SHADER};
use crate::mesh_gen::generate_chunk_mesh;
use crate::quadtree::{children_of, parent_of, root_chunks, ActiveChunks};

#[derive(Resource)]
pub struct TerrainState {
    pub planet_radius: f64,
    pub elevation_scale: f32,
    pub params: ElevationParams,
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
        let params = elevation_params(seed);
        let base_uniform = TerrainMaterialUniform::from_params(
            &params,
            planet_radius as f32,
            elevation_scale,
        );
        Self {
            planet_radius,
            elevation_scale,
            params,
            base_uniform,
            max_quadtree_depth: MAX_QUADTREE_DEPTH,
            screen_error_threshold: SCREEN_ERROR_THRESHOLD,
            merge_hysteresis: MERGE_HYSTERESIS,
            max_render_distance: MAX_RENDER_DISTANCE,
            active_chunk_cap: ACTIVE_CHUNK_CAP,
            lod_split_budget_per_frame: LOD_SPLIT_BUDGET_PER_FRAME,
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
        .insert_resource(TerrainDebugInfo::default())
        .insert_resource(PendingChunkMeshes::default())
        .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
        .add_systems(Startup, setup_terrain)
        .add_systems(
            Update,
            (
                update_lod,
                process_lod_queue,
                apply_pending_chunk_meshes,
                update_neighbor_lod,
                cull_chunks,
                update_debug_info,
            )
                .chain(),
        );
    }
}

fn setup_terrain(
    mut commands: Commands,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut terrain_state: ResMut<TerrainState>,
    mut active_chunks: ResMut<ActiveChunks>,
    mut pending_meshes: ResMut<PendingChunkMeshes>,
) {
    let vertex_source = format!(
        "{}\n{}\n{}",
        include_str!("../../er_world/assets/shaders/elevation.wgsl"),
        include_str!("../assets/shaders/spherify.wgsl"),
        include_str!("../assets/shaders/terrain_vertex.wgsl")
    );
    let vertex_handle = shaders.add(Shader::from_wgsl(vertex_source, "terrain_vertex"));
    let _ = VERTEX_SHADER.set(vertex_handle);

    let fragment_source = include_str!("../assets/shaders/terrain_fragment.wgsl");
    let fragment_handle = shaders.add(Shader::from_wgsl(fragment_source.to_string(), "terrain_fragment"));
    let _ = FRAGMENT_SHADER.set(fragment_handle);

    let uniform = TerrainMaterialUniform::from_params(
        &terrain_state.params,
        terrain_state.planet_radius as f32,
        terrain_state.elevation_scale,
    );
    terrain_state.base_uniform = uniform;

    for key in root_chunks() {
        let entity = spawn_chunk_entity(
            &mut commands,
            &mut materials,
            &terrain_state.base_uniform,
            &mut pending_meshes,
            key,
            terrain_state.planet_radius,
        );
        active_chunks.insert(key, entity);
    }
}

fn update_lod(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut active_chunks: ResMut<ActiveChunks>,
    terrain_state: Res<TerrainState>,
) {
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };
    let camera_pos = camera_transform.translation().as_dvec3();

    let keys: Vec<CellKey> = active_chunks.chunks.keys().copied().collect();
    active_chunks.clear_pending();

    for &key in &keys {
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
}

#[allow(clippy::too_many_arguments)]
fn process_lod_queue(
    mut commands: Commands,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut active_chunks: ResMut<ActiveChunks>,
    mut pending_meshes: ResMut<PendingChunkMeshes>,
    terrain_state: Res<TerrainState>,
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
) {
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };
    let camera_pos = camera_transform.translation().as_dvec3();
    let base_uniform = terrain_state.base_uniform;
    let budget = terrain_state.lod_split_budget_per_frame;

    let mut splits_done = 0usize;
    let pending_splits: Vec<CellKey> = active_chunks.pending_splits.clone();
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

        if let Some(entity) = active_chunks.remove(&key) {
            despawn_chunk(&mut commands, entity);
        }

        for child in children_of(key) {
            let entity = spawn_chunk_entity(
                &mut commands,
                &mut materials,
                &base_uniform,
                &mut pending_meshes,
                child,
                terrain_state.planet_radius,
            );
            active_chunks.insert(child, entity);
        }
        splits_done += 1;
    }

    let mut merges_done = 0usize;
    let pending_merges: Vec<CellKey> = active_chunks.pending_merges.clone();
    for parent_key in pending_merges {
        if merges_done >= budget {
            break;
        }
        let children = children_of(parent_key);
        if !children.iter().all(|c| active_chunks.contains(c)) {
            continue;
        }

        for child in &children {
            if let Some(entity) = active_chunks.remove(child) {
                despawn_chunk(&mut commands, entity);
            }
        }

        let entity = spawn_chunk_entity(
            &mut commands,
            &mut materials,
            &base_uniform,
            &mut pending_meshes,
            parent_key,
            terrain_state.planet_radius,
        );
        active_chunks.insert(parent_key, entity);
        merges_done += 1;
    }

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
}

fn cull_chunks(
    camera_query: Query<(&GlobalTransform, &Projection), With<Camera3d>>,
    mut chunk_query: Query<(&ChunkComponent, &mut Visibility)>,
    terrain_state: Res<TerrainState>,
) {
    let Ok((camera_transform, projection)) = camera_query.single() else {
        return;
    };
    let camera_pos = camera_transform.translation().as_dvec3();

    let frustum = match projection {
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

        if let Some((cam_pos, forward, right, up, fov_cos, aspect)) = frustum {
            let sphere_center = (cell_to_dir(key) * terrain_state.planet_radius).as_vec3();
            let sphere_radius =
                cell_size(key.lod, terrain_state.planet_radius) as f32 + terrain_state.elevation_scale;
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

        if is_below_horizon(key, camera_pos, terrain_state.planet_radius) {
            *visibility = Visibility::Hidden;
            continue;
        }

        if is_outside_render_distance(
            key,
            camera_pos,
            terrain_state.planet_radius,
            terrain_state.max_render_distance,
        ) {
            *visibility = Visibility::Hidden;
            continue;
        }

        *visibility = Visibility::Visible;
    }
}

fn update_debug_info(active_chunks: Res<ActiveChunks>, mut debug: ResMut<TerrainDebugInfo>) {
    debug.active_chunks = active_chunks.len();
    debug.max_depth = active_chunks.chunks.keys().map(|k| k.lod).max().unwrap_or(0);
    debug.pending_splits = active_chunks.pending_splits.len();
    debug.pending_merges = active_chunks.pending_merges.len();
}

fn spawn_chunk_entity(
    commands: &mut Commands,
    materials: &mut Assets<TerrainMaterial>,
    base_uniform: &TerrainMaterialUniform,
    pending_meshes: &mut PendingChunkMeshes,
    key: CellKey,
    radius: f64,
) -> Entity {
    let material = materials.add(TerrainMaterial {
        uniform: base_uniform.for_chunk(key),
    });
    let entity = commands
        .spawn((
            ChunkComponent::new(key),
            MeshMaterial3d(material),
            Transform::default(),
            Visibility::Hidden,
        ))
        .id();
    let task = AsyncComputeTaskPool::get().spawn(async move {
        generate_chunk_mesh(key, radius)
    });
    pending_meshes.0.insert(entity, task);
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
) {
    let mut done = Vec::new();
    for (&entity, task) in &mut pending.0 {
        if let Some(mesh) = check_ready(task) {
            if chunk_query.get(entity).is_ok() {
                let handle = meshes.add(mesh);
                commands.entity(entity).insert(Mesh3d(handle));
            }
            done.push(entity);
        }
    }
    for entity in done {
        pending.0.remove(&entity);
    }
}

fn update_neighbor_lod(
    active_chunks: Res<ActiveChunks>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut chunk_query: Query<(&mut ChunkComponent, &MeshMaterial3d<TerrainMaterial>)>,
) {
    for (mut chunk, mat_handle) in &mut chunk_query {
        let key = chunk.key;
        let mut nd = [key.lod; 4];
        for (i, nb) in chunk.neighbors.iter().enumerate() {
            if active_chunks.contains(nb) {
                nd[i] = nb.lod;
            }
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
}
