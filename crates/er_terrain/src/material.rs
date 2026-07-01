use std::sync::OnceLock;
use bevy::asset::{Asset, Handle};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::reflect::TypePath;
use bevy::render::mesh::{Mesh, MeshVertexBufferLayoutRef};
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use er_world::elevation::ElevationParams;

use crate::mesh_gen::ATTRIBUTE_MORPH;

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
}

impl TerrainMaterialUniform {
    pub fn from_params(params: &ElevationParams, planet_radius: f32, elevation_scale: f32) -> Self {
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
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct TerrainMaterial {
    #[uniform(0)]
    pub uniform: TerrainMaterialUniform,
}

impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(VERTEX_SHADER.get().expect("terrain vertex shader not initialized").clone())
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(FRAGMENT_SHADER.get().expect("terrain fragment shader not initialized").clone())
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
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}
