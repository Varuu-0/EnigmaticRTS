//! Space rendering: atmosphere (Rayleigh shell), starfield (procedural),
//! and sun (emissive sphere). Phase 6.

use bevy::asset::{Asset, RenderAssetUsages};
use bevy::material::AlphaMode;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef};
use bevy::render::render_resource::{
    AsBindGroup, Face, PrimitiveTopology, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use er_core::config::DEFAULT_DAY_LENGTH_SEC;
use er_terrain::{
    ocean::OceanComponent, RenderOrigin, SharedTerrainMaterial, SunDirection, TerrainMaterial,
};
use std::sync::OnceLock;

static STARFIELD_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();
static SUN_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();
static CLOUD_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

fn sun_distance(planet_radius: f32) -> f32 {
    planet_radius * 8.333334
}

fn sun_radius(planet_radius: f32) -> f32 {
    planet_radius * 0.4167
}

fn starfield_radius(planet_radius: f32) -> f32 {
    planet_radius * 11.1112
}

/// Time scale for the simulation (1.0 = real-time, 0.0 = paused).
#[derive(Resource, Clone, Copy)]
pub struct TimeScale {
    pub current: f32,
    pub resume: f32,
}

impl Default for TimeScale {
    fn default() -> Self {
        Self {
            current: 1.0,
            resume: 1.0,
        }
    }
}

/// Accumulated simulation time (seconds), advanced by `TimeScale`.
#[derive(Resource, Default, Clone, Copy)]
pub struct SimTime(pub f32);

#[derive(Component)]
pub struct SunLight;

#[derive(Component)]
pub struct SunSphere;

// ---------------------------------------------------------------------------
// Starfield
// ---------------------------------------------------------------------------

#[derive(encase::ShaderType, Clone, Copy)]
pub struct StarfieldUniform {
    pub seed: f32,
    pub brightness: f32,
    pub sun_dir_x: f32,
    pub sun_dir_y: f32,
    pub sun_dir_z: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
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
        let vertex_layout = layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
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
    pub time: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
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
        let vertex_layout = layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = Some(Face::Back);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Clouds
// ---------------------------------------------------------------------------

#[derive(encase::ShaderType, Clone, Copy)]
pub struct CloudUniform {
    pub sun_dir_x: f32,
    pub sun_dir_y: f32,
    pub sun_dir_z: f32,
    pub time: f32,
    pub camera_pos_x: f32,
    pub camera_pos_y: f32,
    pub camera_pos_z: f32,
    pub planet_radius: f32,
    pub render_origin_x: f32,
    pub render_origin_y: f32,
    pub render_origin_z: f32,
    pub _pad: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct CloudMaterial {
    #[uniform(0)]
    pub uniform: CloudUniform,
}

impl Material for CloudMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(CLOUD_SHADER.get().expect("cloud shader").clone())
    }
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(CLOUD_SHADER.get().expect("cloud shader").clone())
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
        let vertex_layout = layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = Some(Face::Front);
        Ok(())
    }
}

#[derive(Component)]
pub struct CloudComponent;

#[derive(Resource)]
pub struct CloudState {
    pub material: Handle<CloudMaterial>,
}

// ---------------------------------------------------------------------------
// Mesh generation
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct StarfieldState {
    pub material: Handle<StarfieldMaterial>,
}

#[derive(Resource)]
pub struct SunState {
    pub material: Handle<SunMaterial>,
}

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
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct SpacePlugin;

#[derive(SystemSet, Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SpaceUpdate;

impl Plugin for SpacePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(TimeScale::default())
            .insert_resource(SimTime::default())
            .add_plugins(MaterialPlugin::<StarfieldMaterial>::default())
            .add_plugins(MaterialPlugin::<SunMaterial>::default())
            .add_plugins(MaterialPlugin::<CloudMaterial>::default())
            .add_systems(Startup, setup_space)
            .add_systems(
                Update,
                (
                    handle_time_controls,
                    update_sun,
                    update_terrain_uniforms,
                    update_space,
                )
                    .chain()
                    .in_set(SpaceUpdate)
                    .before(er_terrain::TerrainUpdate),
            );
    }
}

fn setup_space(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut star_materials: ResMut<Assets<StarfieldMaterial>>,
    mut sun_materials: ResMut<Assets<SunMaterial>>,
    mut cloud_materials: ResMut<Assets<CloudMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    terrain_state: Res<er_terrain::TerrainState>,
    render_origin: Res<er_terrain::RenderOrigin>,
) {
    let star_source = include_str!("../assets/shaders/starfield.wgsl");
    let star_handle = shaders.add(Shader::from_wgsl(star_source, "starfield"));
    let _ = STARFIELD_SHADER.set(star_handle);

    let sun_source = include_str!("../assets/shaders/sun.wgsl");
    let sun_handle = shaders.add(Shader::from_wgsl(sun_source, "sun"));
    let _ = SUN_SHADER.set(sun_handle);

    let cloud_source = include_str!("../assets/shaders/clouds.wgsl");
    let cloud_handle = shaders.add(Shader::from_wgsl(cloud_source, "clouds"));
    let _ = CLOUD_SHADER.set(cloud_handle);

    let planet_radius = terrain_state.planet_radius as f32;

    let origin_off = render_origin.to_vec3();

    // Starfield
    let star_uniform = StarfieldUniform {
        seed: 42.0,
        brightness: 1.0,
        sun_dir_x: 0.0,
        sun_dir_y: 1.0,
        sun_dir_z: 0.0,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };
    let star_mat = star_materials.add(StarfieldMaterial {
        uniform: star_uniform,
    });
    let star_mesh = meshes.add(make_sphere(starfield_radius(planet_radius), 128, 64));
    commands.spawn((
        StarfieldComponent,
        MeshMaterial3d(star_mat.clone()),
        Mesh3d(star_mesh),
        Transform::default(),
        Visibility::Visible,
    ));
    commands.insert_resource(StarfieldState { material: star_mat });

    // Sun
    let sun_dir = Vec3::new(0.0, 1.0, 0.0);
    let sun_pos = sun_dir * sun_distance(planet_radius) - origin_off;
    let sun_radius = sun_radius(planet_radius);
    let sun_uniform = SunUniform {
        color_r: 1.0,
        color_g: 0.95,
        color_b: 0.8,
        intensity: 2.0,
        time: 0.0,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };
    let sun_mat = sun_materials.add(SunMaterial {
        uniform: sun_uniform,
    });
    let sun_mesh = meshes.add(make_sphere(sun_radius, 32, 16));
    commands.spawn((
        SunSphere,
        MeshMaterial3d(sun_mat.clone()),
        Mesh3d(sun_mesh),
        Transform::from_translation(sun_pos),
        Visibility::Visible,
    ));
    commands.insert_resource(SunState { material: sun_mat });

    // Cloud layer
    let cloud_radius = planet_radius * 1.08;
    let cloud_uniform = CloudUniform {
        sun_dir_x: 0.0,
        sun_dir_y: 1.0,
        sun_dir_z: 0.0,
        time: 0.0,
        camera_pos_x: 0.0,
        camera_pos_y: 0.0,
        camera_pos_z: 90000.0,
        planet_radius,
        render_origin_x: 0.0,
        render_origin_y: 0.0,
        render_origin_z: 0.0,
        _pad: 0.0,
    };
    let cloud_mat = cloud_materials.add(CloudMaterial {
        uniform: cloud_uniform,
    });
    let cloud_mesh = meshes.add(make_sphere(cloud_radius, 256, 128));
    commands.spawn((
        CloudComponent,
        MeshMaterial3d(cloud_mat.clone()),
        Mesh3d(cloud_mesh),
        Transform::from_translation(-origin_off),
        Visibility::Hidden,
    ));
    commands.insert_resource(CloudState {
        material: cloud_mat,
    });

    // Directional light (illuminates PBR terrain/ocean materials)
    let light_transform = Transform::from_translation(sun_pos).looking_to(-sun_dir, Vec3::Y);
    commands.spawn((
        SunLight,
        DirectionalLight {
            color: Color::srgb(1.0, 0.95, 0.85),
            illuminance: 50000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        light_transform,
    ));
}

fn update_space(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut starfield_query: Query<&mut Transform, With<StarfieldComponent>>,
    mut cloud_query: Query<
        &mut Transform,
        (
            With<CloudComponent>,
            Without<StarfieldComponent>,
            Without<OceanComponent>,
        ),
    >,
    mut ocean_query: Query<
        &mut Transform,
        (
            With<OceanComponent>,
            Without<StarfieldComponent>,
            Without<CloudComponent>,
        ),
    >,
    render_origin: Res<RenderOrigin>,
) {
    let Ok(cam) = camera_query.single() else {
        return;
    };
    let cam_pos = cam.translation();

    for mut tf in &mut starfield_query {
        tf.translation = cam_pos;
    }

    if render_origin.is_changed() {
        let planet_center = -render_origin.to_vec3();
        for mut tf in &mut cloud_query {
            tf.translation = planet_center;
        }
        for mut tf in &mut ocean_query {
            tf.translation = planet_center;
        }
    }
}

fn handle_time_controls(keys: Res<ButtonInput<KeyCode>>, mut time_scale: ResMut<TimeScale>) {
    if keys.just_pressed(KeyCode::KeyP) {
        if time_scale.current == 0.0 {
            time_scale.current = time_scale.resume;
            info!("Time resumed: {:.1}x", time_scale.current);
        } else {
            time_scale.resume = time_scale.current;
            time_scale.current = 0.0;
            info!("Time paused");
        }
    }
    if keys.just_pressed(KeyCode::Digit1) {
        time_scale.current = 1.0;
        time_scale.resume = 1.0;
        info!("Time speed: 1x");
    }
    if keys.just_pressed(KeyCode::Digit2) {
        time_scale.current = 5.0;
        time_scale.resume = 5.0;
        info!("Time speed: 5x");
    }
    if keys.just_pressed(KeyCode::Digit3) {
        time_scale.current = 20.0;
        time_scale.resume = 20.0;
        info!("Time speed: 20x");
    }
}

#[allow(clippy::too_many_arguments)]
fn update_sun(
    time: Res<Time>,
    time_scale: Res<TimeScale>,
    mut sim_time: ResMut<SimTime>,
    mut sun_direction: ResMut<SunDirection>,
    light_query: Query<Entity, With<SunLight>>,
    sphere_query: Query<Entity, With<SunSphere>>,
    mut commands: Commands,
    mut star_materials: ResMut<Assets<StarfieldMaterial>>,
    starfield_state: Res<StarfieldState>,
    mut sun_materials: ResMut<Assets<SunMaterial>>,
    sun_state: Res<SunState>,
    mut cloud_materials: ResMut<Assets<CloudMaterial>>,
    cloud_state: Res<CloudState>,
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    terrain_state: Res<er_terrain::TerrainState>,
    render_origin: Res<er_terrain::RenderOrigin>,
) {
    sim_time.0 += time.delta_secs() * time_scale.current;

    let day_length = DEFAULT_DAY_LENGTH_SEC as f32;
    let t = (sim_time.0 % day_length) * (2.0 * std::f32::consts::PI / day_length);
    let sun_dir = Vec3::new(t.sin() * 0.8, t.cos(), t.sin() * 0.3).normalize();

    sun_direction.0 = sun_dir;

    let planet_radius = terrain_state.planet_radius as f32;
    let sun_pos = sun_dir * sun_distance(planet_radius);
    let origin_off = render_origin.to_vec3();
    let sun_pos_rebased = sun_pos - origin_off;

    let forward = -sun_dir;
    let up = if forward.y.abs() > 0.99 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let light_tf = Transform::from_translation(sun_pos_rebased).looking_to(-sun_dir, up);

    for entity in &light_query {
        if let Ok(mut e) = commands.get_entity(entity) {
            e.insert(light_tf);
        }
    }

    for entity in &sphere_query {
        if let Ok(mut e) = commands.get_entity(entity) {
            e.insert(Transform::from_translation(sun_pos_rebased));
        }
    }

    if let Some(mut mat) = star_materials.get_mut(&starfield_state.material) {
        mat.uniform.sun_dir_x = sun_dir.x;
        mat.uniform.sun_dir_y = sun_dir.y;
        mat.uniform.sun_dir_z = sun_dir.z;
    }

    if let Some(mut mat) = sun_materials.get_mut(&sun_state.material) {
        mat.uniform.time = sim_time.0;
    }

    let cam_pos = camera_query
        .single()
        .map(|c| c.translation())
        .unwrap_or(Vec3::ZERO);
    if let Some(mut mat) = cloud_materials.get_mut(&cloud_state.material) {
        mat.uniform.sun_dir_x = sun_dir.x;
        mat.uniform.sun_dir_y = sun_dir.y;
        mat.uniform.sun_dir_z = sun_dir.z;
        mat.uniform.time = sim_time.0;
        mat.uniform.camera_pos_x = cam_pos.x;
        mat.uniform.camera_pos_y = cam_pos.y;
        mat.uniform.camera_pos_z = cam_pos.z;
        mat.uniform.render_origin_x = origin_off.x;
        mat.uniform.render_origin_y = origin_off.y;
        mat.uniform.render_origin_z = origin_off.z;
    }
}

fn update_terrain_uniforms(
    sim_time: Res<SimTime>,
    camera_world: Res<er_terrain::CameraWorldPosition>,
    terrain_material: Res<SharedTerrainMaterial>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
    mut ocean_materials: ResMut<Assets<er_terrain::ocean::OceanMaterial>>,
    render_origin: Res<er_terrain::RenderOrigin>,
) {
    let cam_pos = (camera_world.0 - render_origin.world).as_vec3();

    let day_length = DEFAULT_DAY_LENGTH_SEC as f32;
    let t = (sim_time.0 % day_length) * (2.0 * std::f32::consts::PI / day_length);
    let sun_dir = Vec3::new(t.sin() * 0.8, t.cos(), t.sin() * 0.3).normalize();

    let sx = sun_dir.x;
    let sy = sun_dir.y;
    let sz = sun_dir.z;
    let cx = cam_pos.x;
    let cy = cam_pos.y;
    let cz = cam_pos.z;
    let ox = render_origin.world.x as f32;
    let oy = render_origin.world.y as f32;
    let oz = render_origin.world.z as f32;

    if let Some(mut mat) = terrain_materials.get_mut(&terrain_material.0) {
        let u = &mut mat.uniform;
        u.sun_dir_x = sx;
        u.sun_dir_y = sy;
        u.sun_dir_z = sz;
        u.camera_pos_x = cx;
        u.camera_pos_y = cy;
        u.camera_pos_z = cz;
        u.render_origin_x = ox;
        u.render_origin_y = oy;
        u.render_origin_z = oz;
    }

    for (_, mat) in ocean_materials.iter_mut() {
        let u = &mut mat.uniform;
        u.sun_dir_x = sx;
        u.sun_dir_y = sy;
        u.sun_dir_z = sz;
        u.camera_pos_x = cx;
        u.camera_pos_y = cy;
        u.camera_pos_z = cz;
        u.render_origin_x = ox;
        u.render_origin_y = oy;
        u.render_origin_z = oz;
    }
}
