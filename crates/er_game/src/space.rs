//! Space rendering: atmosphere (Rayleigh shell), starfield (procedural),
//! and sun (emissive sphere). Phase 6.

use bevy::asset::{Asset, RenderAssetUsages};
use bevy::pbr::{Material, MaterialPlugin, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::material::AlphaMode;
use bevy::reflect::TypePath;
use bevy::render::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef};
use bevy::render::render_resource::{
    AsBindGroup, Face, PrimitiveTopology, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use std::sync::OnceLock;

static ATMOSPHERE_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();
static STARFIELD_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();
static SUN_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

// ---------------------------------------------------------------------------
// Atmosphere
// ---------------------------------------------------------------------------

#[derive(encase::ShaderType, Clone, Copy)]
pub struct AtmosphereUniform {
    pub camera_x: f32,
    pub camera_y: f32,
    pub camera_z: f32,
    pub sun_x: f32,
    pub sun_y: f32,
    pub sun_z: f32,
    pub planet_radius: f32,
    pub atmosphere_radius: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct AtmosphereMaterial {
    #[uniform(0)]
    pub uniform: AtmosphereUniform,
}

impl Material for AtmosphereMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(ATMOSPHERE_SHADER.get().expect("atmosphere shader").clone())
    }
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(ATMOSPHERE_SHADER.get().expect("atmosphere shader").clone())
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout =
            layout.0.get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = Some(Face::Front);
        Ok(())
    }
}

#[derive(Component)]
pub struct AtmosphereComponent;

#[derive(Resource)]
pub struct AtmosphereState {
    pub material: Handle<AtmosphereMaterial>,
}

// ---------------------------------------------------------------------------
// Starfield
// ---------------------------------------------------------------------------

#[derive(encase::ShaderType, Clone, Copy)]
pub struct StarfieldUniform {
    pub seed: f32,
    pub brightness: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct StarfieldMaterial {
    #[uniform(0)]
    pub uniform: StarfieldUniform,
}

impl Material for StarfieldMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(STARFIELD_SHADER.get().expect("starfield shader").clone())
    }
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(STARFIELD_SHADER.get().expect("starfield shader").clone())
    }
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout =
            layout.0.get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = Some(Face::Front);
        Ok(())
    }
}

#[derive(Component)]
pub struct StarfieldComponent;

// ---------------------------------------------------------------------------
// Sun
// ---------------------------------------------------------------------------

#[derive(encase::ShaderType, Clone, Copy)]
pub struct SunUniform {
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    pub intensity: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct SunMaterial {
    #[uniform(0)]
    pub uniform: SunUniform,
}

impl Material for SunMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(SUN_SHADER.get().expect("sun shader").clone())
    }
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(SUN_SHADER.get().expect("sun shader").clone())
    }
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout =
            layout.0.get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = Some(Face::Back);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Mesh generation
// ---------------------------------------------------------------------------

fn make_sphere(radius: f32, segments: usize, rings: usize) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((rings + 1) * (segments + 1));
    let mut indices: Vec<u32> = Vec::with_capacity(rings * segments * 6);

    for i in 0..=rings {
        let theta = (i as f32 / rings as f32) * std::f32::consts::PI;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        for j in 0..=segments {
            let phi = (j as f32 / segments as f32) * 2.0 * std::f32::consts::PI;
            positions.push([
                radius * sin_t * phi.cos(),
                radius * cos_t,
                radius * sin_t * phi.sin(),
            ]);
        }
    }
    for i in 0..rings {
        for j in 0..segments {
            let a = (i * (segments + 1) + j) as u32;
            let b = a + (segments + 1) as u32;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct SpacePlugin;

impl Plugin for SpacePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<AtmosphereMaterial>::default())
            .add_plugins(MaterialPlugin::<StarfieldMaterial>::default())
            .add_plugins(MaterialPlugin::<SunMaterial>::default())
            .add_systems(Startup, setup_space)
            .add_systems(Update, update_space);
    }
}

fn setup_space(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AtmosphereMaterial>>,
    mut star_materials: ResMut<Assets<StarfieldMaterial>>,
    mut sun_materials: ResMut<Assets<SunMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    terrain_state: Res<er_terrain::TerrainState>,
) {
    let atm_source = include_str!("../assets/shaders/atmosphere.wgsl");
    let atm_handle = shaders.add(Shader::from_wgsl(atm_source, "atmosphere"));
    let _ = ATMOSPHERE_SHADER.set(atm_handle);

    let star_source = include_str!("../assets/shaders/starfield.wgsl");
    let star_handle = shaders.add(Shader::from_wgsl(star_source, "starfield"));
    let _ = STARFIELD_SHADER.set(star_handle);

    let sun_source = include_str!("../assets/shaders/sun.wgsl");
    let sun_handle = shaders.add(Shader::from_wgsl(sun_source, "sun"));
    let _ = SUN_SHADER.set(sun_handle);

    let planet_radius = terrain_state.planet_radius as f32;

    // Atmosphere shell
    let atm_radius = planet_radius * 1.025;
    let atm_uniform = AtmosphereUniform {
        camera_x: 0.0,
        camera_y: 0.0,
        camera_z: 90000.0,
        sun_x: 0.5,
        sun_y: 0.8,
        sun_z: 0.3,
        planet_radius,
        atmosphere_radius: atm_radius,
    };
    let atm_material = materials.add(AtmosphereMaterial {
        uniform: atm_uniform,
    });
    let atm_mesh = meshes.add(make_sphere(atm_radius, 64, 32));
    commands.spawn((
        AtmosphereComponent,
        MeshMaterial3d(atm_material.clone()),
        Mesh3d(atm_mesh),
        Transform::default(),
        Visibility::Visible,
    ));
    commands.insert_resource(AtmosphereState {
        material: atm_material,
    });

    // Starfield
    let star_uniform = StarfieldUniform {
        seed: 42.0,
        brightness: 1.0,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let star_mat = star_materials.add(StarfieldMaterial {
        uniform: star_uniform,
    });
    let star_mesh = meshes.add(make_sphere(400000.0, 128, 64));
    commands.spawn((
        StarfieldComponent,
        MeshMaterial3d(star_mat),
        Mesh3d(star_mesh),
        Transform::default(),
        Visibility::Visible,
    ));

    // Sun
    let sun_dir = Vec3::new(0.5, 0.8, 0.3).normalize();
    let sun_distance = 300000.0;
    let sun_radius = 15000.0;
    let sun_pos = sun_dir * sun_distance;
    let sun_uniform = SunUniform {
        color_r: 1.0,
        color_g: 0.95,
        color_b: 0.8,
        intensity: 2.0,
    };
    let sun_mat = sun_materials.add(SunMaterial {
        uniform: sun_uniform,
    });
    let sun_mesh = meshes.add(make_sphere(sun_radius, 32, 16));
    commands.spawn((
        MeshMaterial3d(sun_mat),
        Mesh3d(sun_mesh),
        Transform::from_translation(sun_pos),
        Visibility::Visible,
    ));
}

fn update_space(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut atm_materials: ResMut<Assets<AtmosphereMaterial>>,
    atm_state: Res<AtmosphereState>,
    mut starfield_query: Query<&mut Transform, With<StarfieldComponent>>,
) {
    let Ok(cam) = camera_query.single() else {
        return;
    };
    let cam_pos = cam.translation();

    if let Some(mut mat) = atm_materials.get_mut(&atm_state.material) {
        mat.uniform.camera_x = cam_pos.x;
        mat.uniform.camera_y = cam_pos.y;
        mat.uniform.camera_z = cam_pos.z;
    }

    for mut tf in &mut starfield_query {
        tf.translation = cam_pos;
    }
}
