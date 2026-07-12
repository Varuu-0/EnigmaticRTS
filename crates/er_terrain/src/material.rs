use bevy::asset::{Asset, Handle};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::reflect::TypePath;
use bevy::render::mesh::{Mesh, MeshVertexBufferLayoutRef};
use bevy::render::render_resource::{
    AsBindGroup, Face, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use er_core::math::{cells_per_edge, CellKey};
use er_world::elevation::ElevationParams;
use std::sync::OnceLock;

use crate::mesh_gen::{
    ATTRIBUTE_ELEVATION, ATTRIBUTE_GRID, ATTRIBUTE_LOW_FREQ_ELEV, ATTRIBUTE_MOISTURE_LOW,
    ATTRIBUTE_MORPH, ATTRIBUTE_NORMAL, ATTRIBUTE_TEMPERATURE, ATTRIBUTE_WARPED_DIR,
};

pub static VERTEX_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();
pub static FRAGMENT_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

#[derive(encase::ShaderType, Clone, Copy)]
pub struct TerrainMaterialUniform {
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
    pub planet_radius: f32,
    pub elevation_scale: f32,
    pub face: i32,
    pub u_min: f32,
    pub u_max: f32,
    pub v_min: f32,
    pub v_max: f32,
    pub chunk_depth: i32,
    pub neighbor_depth_0: f32,
    pub neighbor_depth_1: f32,
    pub neighbor_depth_2: f32,
    pub neighbor_depth_3: f32,
    pub sea_level_climate: f32,
    pub lapse_rate: f32,
    pub temp_gradient: f32,
    pub temp_noise_freq: f32,
    pub temp_noise_amp: f32,
    pub moisture_noise_freq: f32,
    pub moisture_noise_amp: f32,
    pub rain_shadow_strength: f32,
    pub high_alt_threshold: f32,
    pub beach_threshold: f32,
    pub volcanic_threshold: f32,
    pub toxic_moisture_threshold: f32,
    pub toxic_temp_threshold: f32,
    pub temp_noise_seed: i32,
    pub moisture_noise_seed: i32,
    pub sun_dir_x: f32,
    pub sun_dir_y: f32,
    pub sun_dir_z: f32,
    pub camera_pos_x: f32,
    pub camera_pos_y: f32,
    pub camera_pos_z: f32,
    pub debug_skirt_highlight: f32,
}

impl TerrainMaterialUniform {
    pub fn from_params(
        params: &ElevationParams,
        planet_radius: f32,
        elevation_scale: f32,
        climate: &er_world::params::PlanetParams,
    ) -> Self {
        Self {
            seed: params.seed,
            sea_level: params.sea_level,
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
            planet_radius,
            elevation_scale,
            face: 0,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            chunk_depth: 0,
            neighbor_depth_0: 0.0,
            neighbor_depth_1: 0.0,
            neighbor_depth_2: 0.0,
            neighbor_depth_3: 0.0,
            sea_level_climate: climate.sea_level as f32,
            lapse_rate: climate.lapse_rate as f32,
            temp_gradient: climate.temp_gradient as f32,
            temp_noise_freq: climate.temp_noise_freq,
            temp_noise_amp: climate.temp_noise_amp,
            moisture_noise_freq: climate.moisture_noise_freq,
            moisture_noise_amp: climate.moisture_noise_amp,
            rain_shadow_strength: climate.rain_shadow_strength,
            high_alt_threshold: climate.high_alt_threshold as f32,
            beach_threshold: climate.beach_threshold as f32,
            volcanic_threshold: climate.volcanic_threshold as f32,
            toxic_moisture_threshold: climate.toxic_moisture_threshold as f32,
            toxic_temp_threshold: climate.toxic_temp_threshold as f32,
            temp_noise_seed: climate.temp_noise_seed,
            moisture_noise_seed: climate.moisture_noise_seed,
            sun_dir_x: 0.5,
            sun_dir_y: 0.8,
            sun_dir_z: 0.3,
            camera_pos_x: 0.0,
            camera_pos_y: 0.0,
            camera_pos_z: 0.0,
            debug_skirt_highlight: 0.0,
        }
    }

    pub fn for_chunk(&self, key: CellKey) -> Self {
        let cells = cells_per_edge(key.lod) as f32;
        let depth = key.lod as f32;
        let mut u = *self;
        u.face = key.face as i32;
        u.u_min = key.i as f32 / cells;
        u.u_max = (key.i + 1) as f32 / cells;
        u.v_min = key.j as f32 / cells;
        u.v_max = (key.j + 1) as f32 / cells;
        u.chunk_depth = key.lod as i32;
        u.neighbor_depth_0 = depth;
        u.neighbor_depth_1 = depth;
        u.neighbor_depth_2 = depth;
        u.neighbor_depth_3 = depth;
        u
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct TerrainMaterial {
    #[uniform(0)]
    pub uniform: TerrainMaterialUniform,
}

impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(
            VERTEX_SHADER
                .get()
                .expect("terrain vertex shader not initialized")
                .clone(),
        )
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(
            FRAGMENT_SHADER
                .get()
                .expect("terrain fragment shader not initialized")
                .clone(),
        )
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            ATTRIBUTE_MORPH.at_shader_location(1),
            ATTRIBUTE_GRID.at_shader_location(2),
            ATTRIBUTE_LOW_FREQ_ELEV.at_shader_location(3),
            ATTRIBUTE_WARPED_DIR.at_shader_location(4),
            ATTRIBUTE_MOISTURE_LOW.at_shader_location(5),
            ATTRIBUTE_ELEVATION.at_shader_location(6),
            ATTRIBUTE_NORMAL.at_shader_location(7),
            ATTRIBUTE_TEMPERATURE.at_shader_location(8),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = Some(Face::Back);
        Ok(())
    }
}
