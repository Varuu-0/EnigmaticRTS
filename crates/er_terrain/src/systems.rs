use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;
use bevy::shader::Shader;
use er_core::config::{
    ACTIVE_CHUNK_CAP, LOD_SPLIT_BUDGET_PER_FRAME, MAX_QUADTREE_DEPTH, MAX_RENDER_DISTANCE,
    MERGE_HYSTERESIS, PLANET_RADIUS_DEFAULT, SCREEN_ERROR_THRESHOLD,
};
use er_core::math::CellKey;
use er_core::seed::PlanetSeed;
use er_world::elevation::{elevation_params, ElevationParams};
use std::collections::HashSet;

use crate::chunk::ChunkComponent;
use crate::culling::{is_below_horizon, is_outside_render_distance};
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
    pub material: Option<Handle<TerrainMaterial>>,
    pub max_quadtree_depth: u8,
    pub screen_error_threshold: f32,
    pub merge_hysteresis: f32,
    pub max_render_distance: f64,
    pub active_chunk_cap: usize,
    pub lod_split_budget_per_frame: usize,
}

impl TerrainState {
    pub fn new(planet_radius: f64, elevation_scale: f32, seed: PlanetSeed) -> Self {
        Self {
            planet_radius,
            elevation_scale,
            params: elevation_params(seed),
            material: None,
            max_quadtree_depth: MAX_QUADTREE_DEPTH,
            screen_error_threshold: SCREEN_ERROR_THRESHOLD,
            merge_hysteresis: MERGE_HYSTERESIS,
            max_render_distance: MAX_RENDER_DISTANCE,
            active_chunk_cap: ACTIVE_CHUNK_CAP,
            lod_split_budget_per_frame: LOD_SPLIT_BUDGET_PER_FRAME,
        }
    }
}

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
        .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
        .add_systems(Startup, setup_terrain)
        .add_systems(
            Update,
            (update_lod, process_lod_queue, cull_chunks, update_debug_info).chain(),
        );
    }
}

fn setup_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut terrain_state: ResMut<TerrainState>,
    mut active_chunks: ResMut<ActiveChunks>,
) {
    let vertex_source = format!(
        "{}\n{}",
        include_str!("../../er_world/assets/shaders/elevation.wgsl"),
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
    let material = materials.add(TerrainMaterial { uniform });
    terrain_state.material = Some(material.clone());

    for key in root_chunks() {
        let mesh = meshes.add(generate_chunk_mesh(key, terrain_state.planet_radius));
        let entity = commands
            .spawn((
                ChunkComponent::new(key),
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                Transform::default(),
                Visibility::default(),
            ))
            .id();
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut active_chunks: ResMut<ActiveChunks>,
    terrain_state: Res<TerrainState>,
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
) {
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };
    let camera_pos = camera_transform.translation().as_dvec3();
    let material = match &terrain_state.material {
        Some(m) => m.clone(),
        None => return,
    };
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
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.despawn();
            }
        }

        for child in children_of(key) {
            let mesh = meshes.add(generate_chunk_mesh(child, terrain_state.planet_radius));
            let entity = commands
                .spawn((
                    ChunkComponent::new(child),
                    Mesh3d(mesh),
                    MeshMaterial3d(material.clone()),
                    Transform::default(),
                    Visibility::default(),
                ))
                .id();
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
                if let Ok(mut ec) = commands.get_entity(entity) {
                    ec.despawn();
                }
            }
        }

        let mesh = meshes.add(generate_chunk_mesh(parent_key, terrain_state.planet_radius));
        let entity = commands
            .spawn((
                ChunkComponent::new(parent_key),
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                Transform::default(),
                Visibility::default(),
            ))
            .id();
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
                    if let Ok(mut ec) = commands.get_entity(entity) {
                        ec.despawn();
                    }
                }
            }
        }
    }
}

fn cull_chunks(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut chunk_query: Query<(&ChunkComponent, &mut Visibility)>,
    terrain_state: Res<TerrainState>,
) {
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };
    let camera_pos = camera_transform.translation().as_dvec3();

    for (chunk, mut visibility) in &mut chunk_query {
        let key = chunk.key;

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

pub fn spawn_chunk_at(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &Handle<TerrainMaterial>,
    key: CellKey,
    radius: f64,
) -> Entity {
    let mesh = meshes.add(generate_chunk_mesh(key, radius));
    commands
        .spawn((
            ChunkComponent::new(key),
            Mesh3d(mesh),
            MeshMaterial3d(material.clone()),
            Transform::default(),
            Visibility::default(),
        ))
        .id()
}
