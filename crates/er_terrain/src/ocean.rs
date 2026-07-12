use bevy::asset::{Asset, Handle, RenderAssetUsages};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::reflect::TypePath;
use bevy::render::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef};
use bevy::render::render_resource::{
    AsBindGroup, BlendState, ColorWrites, Face, PrimitiveTopology,
    RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use bevy::ecs::system::{Commands, Res, ResMut};
use bevy::ecs::resource::Resource;
use bevy::prelude::*;
use std::sync::OnceLock;
use er_world::elevation::ElevationParams;
use er_world::params::PlanetParams;

pub static OCEAN_VERTEX_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();
pub static OCEAN_FRAGMENT_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

/// Depth-only material for occlusion sphere - writes depth but not color
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct OcclusionMaterial {
    #[uniform(0)]
    pub _dummy: u32,
}

impl Material for OcclusionMaterial {
    fn fragment_shader() -> ShaderRef {
        // Use a minimal shader that just returns transparent
        ShaderRef::Default
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        // Render both sides to create full occlusion sphere
        descriptor.primitive.cull_mode = None;
        // Don't write color, only depth
        if let Some(fragment) = &mut descriptor.fragment {
            for target in &mut fragment.targets {
                if let Some(color_target) = target {
                    color_target.write_mask = ColorWrites::empty();
                }
            }
        }
        Ok(())
    }
}

#[derive(encase::ShaderType, Clone, Copy)]
pub struct OceanMaterialUniform {
    pub seed: i32,
    pub sea_level: f32,
    pub continental_freq: f32,
    pub continental_amp: f32,
    pub continental_octaves: i32,
    pub mountain_freq: f32,
    pub mountain_amp: f32,
    pub mountain_octaves: i32,
    pub hill_freq: f32,
    pub hill_amp: f32,
    pub hill_octaves: i32,
    pub detail_freq: f32,
    pub detail_amp: f32,
    pub detail_octaves: i32,
    pub warp_freq: f32,
    pub warp_amp: f32,
    pub lacunarity: f32,
    pub gain: f32,
    _pad0: f32,
    _pad1: f32,
    pub planet_radius: f32,
    pub elevation_scale: f32,
    pub sun_dir_x: f32,
    pub sun_dir_y: f32,
    pub sun_dir_z: f32,
    pub time: f32,
    pub camera_pos_x: f32,
    pub camera_pos_y: f32,
    pub camera_pos_z: f32,
}

impl OceanMaterialUniform {
    pub fn from_params(
        params: &ElevationParams,
        planet_radius: f32,
        elevation_scale: f32,
        climate: &PlanetParams,
    ) -> Self {
        Self {
            seed: params.seed,
            sea_level: climate.sea_level as f32,
            continental_freq: params.continental_freq,
            continental_amp: params.continental_amp,
            continental_octaves: params.continental_octaves,
            mountain_freq: params.mountain_freq,
            mountain_amp: params.mountain_amp,
            mountain_octaves: params.mountain_octaves,
            hill_freq: params.hill_freq,
            hill_amp: params.hill_amp,
            hill_octaves: params.hill_octaves,
            detail_freq: params.detail_freq,
            detail_amp: params.detail_amp,
            detail_octaves: params.detail_octaves,
            warp_freq: params.warp_freq,
            warp_amp: params.warp_amp,
            lacunarity: params.lacunarity,
            gain: params.gain,
            _pad0: 0.0,
            _pad1: 0.0,
            planet_radius,
            elevation_scale,
            sun_dir_x: 0.5,
            sun_dir_y: 0.8,
            sun_dir_z: 0.3,
            time: 0.0,
            camera_pos_x: 0.0,
            camera_pos_y: 0.0,
            camera_pos_z: 0.0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct OceanMaterial {
    #[uniform(0)]
    pub uniform: OceanMaterialUniform,
}

impl Material for OceanMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(OCEAN_VERTEX_SHADER.get().expect("ocean vertex shader not initialized").clone())
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(OCEAN_FRAGMENT_SHADER.get().expect("ocean fragment shader not initialized").clone())
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = Some(Face::Back);
        // Enable alpha blending so transparent pixels (over land) don't write color
        // but still write depth to occlude far-side terrain
        descriptor.fragment.as_mut().unwrap().targets[0].as_mut().unwrap().blend = Some(BlendState::ALPHA_BLENDING);
        Ok(())
    }
}

#[derive(Component)]
pub struct OceanComponent;

#[derive(Resource)]
pub struct OceanState {
    pub material: Handle<OceanMaterial>,
}

pub fn generate_ocean_sphere(radius: f32, segments: usize, rings: usize) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((rings + 1) * (segments + 1));
    let mut indices: Vec<u32> = Vec::with_capacity(rings * segments * 6);

    for i in 0..=rings {
        let theta = (i as f32 / rings as f32) * std::f32::consts::PI;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        for j in 0..=segments {
            let phi = (j as f32 / segments as f32) * 2.0 * std::f32::consts::PI;
            let x = radius * sin_t * phi.cos();
            let y = radius * cos_t;
            let z = radius * sin_t * phi.sin();
            positions.push([x, y, z]);
        }
    }

    for i in 0..rings {
        for j in 0..segments {
            let a = (i * (segments + 1) + j) as u32;
            let b = a + (segments + 1) as u32;
            indices.extend_from_slice(&[a, b, a + 1]);
            indices.extend_from_slice(&[a + 1, b, b + 1]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

pub fn setup_ocean(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OceanMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    terrain_state: Res<crate::systems::TerrainState>,
) {
    let vertex_source = format!(
        "{}\n{}",
        include_str!("../assets/shaders/ocean_uniform.wgsl"),
        include_str!("../assets/shaders/ocean_vertex.wgsl"),
    );
    let vertex_handle = shaders.add(Shader::from_wgsl(vertex_source, "ocean_vertex"));
    let _ = OCEAN_VERTEX_SHADER.set(vertex_handle);

    let fragment_source = format!(
        "{}\n{}\n{}",
        include_str!("../../er_world/assets/shaders/elevation.wgsl"),
        include_str!("../assets/shaders/ocean_uniform.wgsl"),
        include_str!("../assets/shaders/ocean_fragment.wgsl"),
    );
    let fragment_handle = shaders.add(Shader::from_wgsl(fragment_source, "ocean_fragment"));
    let _ = OCEAN_FRAGMENT_SHADER.set(fragment_handle);

    let ocean_radius = terrain_state.planet_radius as f32
        + terrain_state.elevation_scale
        + 100.0;

    let uniform = OceanMaterialUniform::from_params(
        &terrain_state.params,
        terrain_state.planet_radius as f32,
        terrain_state.elevation_scale,
        &terrain_state.planet_params,
    );

    let material = materials.add(OceanMaterial { uniform });
    let mesh = generate_ocean_sphere(ocean_radius, 128, 64);
    let mesh_handle = meshes.add(mesh);

    commands.spawn((
        OceanComponent,
        MeshMaterial3d(material.clone()),
        Mesh3d(mesh_handle),
        Transform::default(),
        Visibility::Visible,
    ));

    commands.insert_resource(OceanState { material });
}

pub fn update_ocean_time(
    time: Res<Time>,
    mut materials: ResMut<Assets<OceanMaterial>>,
    ocean_state: Res<OceanState>,
) {
    if let Some(mut mat) = materials.get_mut(&ocean_state.material) {
        mat.uniform.time = time.elapsed_secs();
    }
}
