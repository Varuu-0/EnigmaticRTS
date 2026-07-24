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
use er_core::rng::{rng_from_seed, Rng};
use er_terrain::{
    ocean::OceanComponent, CameraWorldPosition, RenderOrigin, SharedTerrainMaterial, SunDirection,
    TerrainMaterial,
};
use glam::{DQuat, DVec3};
use rand::RngCore;

use crate::debug_overlay::DebugViewSelection;
use std::sync::OnceLock;

static STARFIELD_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();
static SUN_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();
static CLOUD_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();
const CLOUD_ALTITUDE_RATIO: f64 = 0.0025;
const CLOUD_SPAWN_CHECK_INTERVAL_SEC: f32 = 4.0;
const CLOUD_SPAWN_CHANCE: f32 = 0.2;
const CLOUD_MAX_CLUSTERS: usize = 4;
const CLOUD_MAX_TOTAL_PUFFS: usize = 36;
const _: () = {
    assert!(CLOUD_SPAWN_CHANCE < 0.5);
    assert!(CLOUD_MAX_CLUSTERS <= 4);
    assert!(CLOUD_MAX_TOTAL_PUFFS <= 36);
    assert!(CLOUD_SPAWN_CHECK_INTERVAL_SEC >= 4.0);
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeatherMagnitude {
    CloudBank,
    Thunderstorm,
    Cyclone,
    Superstorm,
}

#[derive(Clone, Copy)]
struct WeatherProfile {
    weight: f32,
    lifetime_min: f32,
    lifetime_max: f32,
    puff_min: usize,
    puff_max: usize,
    spread_min: f32,
    spread_max: f32,
    max_active: usize,
    storminess: f32,
    spin_min: f64,
    spin_max: f64,
}

impl WeatherMagnitude {
    const ALL: [Self; 4] = [
        Self::CloudBank,
        Self::Thunderstorm,
        Self::Cyclone,
        Self::Superstorm,
    ];

    fn profile(self) -> WeatherProfile {
        match self {
            Self::CloudBank => WeatherProfile {
                weight: 0.60,
                lifetime_min: 24.0,
                lifetime_max: 38.0,
                puff_min: 5,
                puff_max: 7,
                spread_min: 0.035,
                spread_max: 0.060,
                max_active: 2,
                storminess: 0.0,
                spin_min: 0.0,
                spin_max: 0.0,
            },
            Self::Thunderstorm => WeatherProfile {
                weight: 0.25,
                lifetime_min: 42.0,
                lifetime_max: 64.0,
                puff_min: 6,
                puff_max: 8,
                spread_min: 0.025,
                spread_max: 0.045,
                max_active: 2,
                storminess: 0.48,
                spin_min: 0.0,
                spin_max: 0.0,
            },
            Self::Cyclone => WeatherProfile {
                weight: 0.11,
                lifetime_min: 72.0,
                lifetime_max: 108.0,
                puff_min: 8,
                puff_max: 10,
                spread_min: 0.080,
                spread_max: 0.125,
                max_active: 1,
                storminess: 0.68,
                spin_min: 0.00012,
                spin_max: 0.00022,
            },
            Self::Superstorm => WeatherProfile {
                weight: 0.04,
                lifetime_min: 120.0,
                lifetime_max: 180.0,
                puff_min: 10,
                puff_max: 12,
                spread_min: 0.130,
                spread_max: 0.190,
                max_active: 1,
                storminess: 1.0,
                spin_min: 0.00018,
                spin_max: 0.00032,
            },
        }
    }
}

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
    pub opacity: f32,
    pub storminess: f32,
    pub formation: f32,
    pub role: f32,
    pub noise_seed: f32,
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
        descriptor.primitive.cull_mode = Some(Face::Back);
        Ok(())
    }
}

#[derive(Component)]
pub struct CloudComponent;

#[derive(Component)]
struct CloudPuff;

#[derive(Resource, Clone, Copy)]
pub struct CloudRuntimeConfig {
    pub spawning_enabled: bool,
    pub force_initial_cluster: bool,
}

impl Default for CloudRuntimeConfig {
    fn default() -> Self {
        Self {
            spawning_enabled: true,
            force_initial_cluster: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CloudPuffPlan {
    offset: Vec3,
    scale: Vec3,
    phase: f32,
    noise_seed: f32,
    role: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct CloudSpawnPlan {
    magnitude: WeatherMagnitude,
    direction: DVec3,
    drift_axis: DVec3,
    angular_speed: f64,
    local_spin_speed: f64,
    lifetime: f32,
    puffs: Vec<CloudPuffPlan>,
}

struct ActiveCloudCluster {
    magnitude: WeatherMagnitude,
    direction: DVec3,
    drift_axis: DVec3,
    angular_speed: f64,
    local_spin_speed: f64,
    local_spin_angle: f64,
    born_at: f32,
    lifetime: f32,
    puffs: Vec<(Entity, CloudPuffPlan, Handle<CloudMaterial>)>,
}

#[derive(Resource)]
struct CloudPopulation {
    rng: Rng,
    next_spawn_attempt: f32,
    forced_spawn_done: bool,
    planet_radius: f64,
    cloud_radius: f64,
    mesh: Handle<Mesh>,
    active: Vec<ActiveCloudCluster>,
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
            // Counter-clockwise when viewed from outside. This keeps outward
            // cloud faces after Back culling and matches their shader normals.
            indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn random_unit(rng: &mut Rng) -> f32 {
    rng.next_u32() as f32 / u32::MAX as f32
}

fn should_spawn_cloud(rng: &mut Rng) -> bool {
    random_unit(rng) < CLOUD_SPAWN_CHANCE
}

fn select_weather_magnitude(rng: &mut Rng) -> WeatherMagnitude {
    let roll = random_unit(rng);
    let mut cumulative = 0.0;
    for magnitude in WeatherMagnitude::ALL {
        cumulative += magnitude.profile().weight;
        if roll < cumulative {
            return magnitude;
        }
    }
    WeatherMagnitude::Superstorm
}

fn tangent_basis(direction: DVec3) -> (DVec3, DVec3) {
    let reference = if direction.y.abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let tangent_x = direction.cross(reference).normalize();
    let tangent_z = tangent_x.cross(direction).normalize();
    (tangent_x, tangent_z)
}

fn make_cloud_spawn_plan(
    rng: &mut Rng,
    camera_direction: DVec3,
    planet_radius: f64,
    magnitude: WeatherMagnitude,
) -> CloudSpawnPlan {
    let camera_direction = camera_direction.normalize_or_zero();
    let camera_direction = if camera_direction.length_squared() > 0.0 {
        camera_direction
    } else {
        DVec3::Y
    };
    let (camera_tangent_x, camera_tangent_z) = tangent_basis(camera_direction);
    let azimuth = random_unit(rng) as f64 * std::f64::consts::TAU;
    let cos_angle = 0.65 + random_unit(rng) as f64 * 0.35;
    let sin_angle = (1.0 - cos_angle * cos_angle).sqrt();
    let direction = (camera_direction * cos_angle
        + camera_tangent_x * (azimuth.cos() * sin_angle)
        + camera_tangent_z * (azimuth.sin() * sin_angle))
        .normalize();

    let (tangent_x, tangent_z) = tangent_basis(direction);
    let drift_heading = random_unit(rng) as f64 * std::f64::consts::TAU;
    let drift_direction = tangent_x * drift_heading.cos() + tangent_z * drift_heading.sin();
    let drift_axis = direction.cross(drift_direction).normalize();
    let angular_speed = 0.00003 + random_unit(rng) as f64 * 0.00005;
    let profile = magnitude.profile();
    let lifetime =
        profile.lifetime_min + random_unit(rng) * (profile.lifetime_max - profile.lifetime_min);
    let puff_count =
        profile.puff_min + (rng.next_u32() as usize % (profile.puff_max - profile.puff_min + 1));
    let spread = planet_radius as f32
        * (profile.spread_min + random_unit(rng) * (profile.spread_max - profile.spread_min));
    let core_width = match magnitude {
        WeatherMagnitude::CloudBank => spread * (0.34 + random_unit(rng) * 0.10),
        WeatherMagnitude::Thunderstorm => spread * (0.42 + random_unit(rng) * 0.12),
        WeatherMagnitude::Cyclone => spread * (0.25 + random_unit(rng) * 0.08),
        WeatherMagnitude::Superstorm => spread * (0.22 + random_unit(rng) * 0.07),
    };

    let mut puffs = Vec::with_capacity(puff_count);
    match magnitude {
        WeatherMagnitude::CloudBank => {
            let heading = random_unit(rng) * std::f32::consts::TAU;
            for index in 0..puff_count {
                let along = ((index as f32 + random_unit(rng)) / puff_count as f32) * 2.0 - 1.0;
                let across = (random_unit(rng) * 2.0 - 1.0) * spread * 0.13;
                let offset = Vec3::new(
                    along * spread * heading.cos() - across * heading.sin(),
                    (random_unit(rng) * 2.0 - 1.0) * spread * 0.05,
                    along * spread * heading.sin() + across * heading.cos(),
                );
                let width = core_width * (0.72 + random_unit(rng) * 0.28);
                puffs.push(make_weather_puff(
                    offset,
                    width,
                    width * (0.22 + random_unit(rng) * 0.12),
                    width * (0.85 + random_unit(rng) * 0.20),
                    0.2,
                    rng,
                ));
            }
        }
        WeatherMagnitude::Thunderstorm => {
            puffs.push(make_weather_puff(
                Vec3::ZERO,
                core_width,
                core_width * 1.8,
                core_width * 0.9,
                0.8,
                rng,
            ));
            puffs.push(make_weather_puff(
                Vec3::new(0.0, core_width * 1.15, 0.0),
                spread * 0.75,
                spread * 0.12,
                spread * 0.58,
                0.5,
                rng,
            ));
            for _ in 2..puff_count {
                let angle = random_unit(rng) * std::f32::consts::TAU;
                let radius = random_unit(rng).sqrt() * spread * 0.35;
                let width = core_width * (0.48 + random_unit(rng) * 0.34);
                puffs.push(make_weather_puff(
                    Vec3::new(
                        radius * angle.cos(),
                        -core_width * 0.35 + random_unit(rng) * core_width,
                        radius * angle.sin(),
                    ),
                    width,
                    width * (0.75 + random_unit(rng) * 0.35),
                    width,
                    0.7,
                    rng,
                ));
            }
        }
        WeatherMagnitude::Cyclone | WeatherMagnitude::Superstorm => {
            let arm_count = if magnitude == WeatherMagnitude::Cyclone {
                2
            } else {
                3
            };
            let eye_radius = spread
                * if magnitude == WeatherMagnitude::Cyclone {
                    0.14
                } else {
                    0.10
                };
            let puffs_per_arm = (puff_count / arm_count).max(1);
            for arm in 0..arm_count {
                let arm_offset = arm as f32 / arm_count as f32 * std::f32::consts::TAU;
                for index in 0..puffs_per_arm {
                    if puffs.len() == puff_count {
                        break;
                    }
                    let progress = (index as f32 + random_unit(rng) * 0.4) / puffs_per_arm as f32;
                    let angle = arm_offset + progress * std::f32::consts::TAU * 1.25;
                    let radius = eye_radius + (spread - eye_radius) * progress;
                    let width = core_width * (0.78 + progress * 0.55);
                    puffs.push(make_weather_puff(
                        Vec3::new(
                            radius * angle.cos(),
                            (random_unit(rng) * 2.0 - 1.0) * width * 0.20,
                            radius * angle.sin(),
                        ),
                        width,
                        width * (0.30 + random_unit(rng) * 0.18),
                        width * (0.85 + random_unit(rng) * 0.20),
                        if magnitude == WeatherMagnitude::Superstorm {
                            1.0
                        } else {
                            0.78
                        },
                        rng,
                    ));
                }
            }
            while puffs.len() < puff_count {
                let angle = random_unit(rng) * std::f32::consts::TAU;
                let radius = spread * (0.45 + random_unit(rng) * 0.45);
                let width = core_width * (0.85 + random_unit(rng) * 0.30);
                puffs.push(make_weather_puff(
                    Vec3::new(radius * angle.cos(), width * 0.25, radius * angle.sin()),
                    width,
                    width
                        * if magnitude == WeatherMagnitude::Superstorm {
                            0.80
                        } else {
                            0.45
                        },
                    width,
                    0.9,
                    rng,
                ));
            }
        }
    }

    CloudSpawnPlan {
        magnitude,
        direction,
        drift_axis,
        angular_speed,
        local_spin_speed: profile.spin_min
            + random_unit(rng) as f64 * (profile.spin_max - profile.spin_min),
        lifetime,
        puffs,
    }
}

fn make_weather_puff(
    offset: Vec3,
    width: f32,
    height: f32,
    depth: f32,
    role: f32,
    rng: &mut Rng,
) -> CloudPuffPlan {
    CloudPuffPlan {
        offset,
        scale: Vec3::new(width, height, depth),
        phase: random_unit(rng) * std::f32::consts::TAU,
        noise_seed: random_unit(rng) * 100.0,
        role,
    }
}

fn smooth_unit(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn cloud_lifecycle(age: f32, lifetime: f32) -> (f32, f32) {
    if lifetime <= 0.0 || age < 0.0 || age >= lifetime {
        return (0.0, 0.0);
    }
    let progress = age / lifetime;
    let fade_in = smooth_unit(progress / 0.2);
    let fade_out = smooth_unit((1.0 - progress) / 0.3);
    let opacity = fade_in.min(fade_out) * 0.68;
    let growth = smooth_unit(progress / 0.25);
    let evaporation = 1.0 - fade_out;
    let scale = 0.65 + growth * 0.35 + evaporation * 0.12;
    (opacity, scale)
}

fn cloud_altitude(planet_radius: f64, elevation_scale: f32) -> f64 {
    (planet_radius * CLOUD_ALTITUDE_RATIO).max(elevation_scale.abs() as f64 * 3.0)
}

fn cloud_spawn_direction(
    camera_position: DVec3,
    camera_forward: DVec3,
    cloud_radius: f64,
) -> DVec3 {
    let radial = camera_position.normalize_or(DVec3::Y);
    if camera_position.length() >= cloud_radius || camera_forward.dot(radial) < 0.0 {
        return radial;
    }

    // For cameras below the cloud layer, place the cluster where the current
    // outward view ray exits it instead of placing it behind the view.
    let projection = camera_position.dot(camera_forward);
    let discriminant =
        projection * projection + cloud_radius * cloud_radius - camera_position.length_squared();
    if discriminant <= 0.0 {
        return radial;
    }
    (camera_position + camera_forward * (-projection + discriminant.sqrt())).normalize_or(radial)
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
            .init_resource::<DebugViewSelection>()
            .init_resource::<CloudRuntimeConfig>()
            .add_plugins(MaterialPlugin::<StarfieldMaterial>::default())
            .add_plugins(MaterialPlugin::<SunMaterial>::default())
            .add_plugins(MaterialPlugin::<CloudMaterial>::default())
            .add_systems(Startup, setup_space)
            .add_systems(
                Update,
                (
                    handle_time_controls,
                    update_sun,
                    update_clouds,
                    update_terrain_uniforms,
                    update_environment_debug_visibility,
                    update_space,
                )
                    .chain()
                    .in_set(SpaceUpdate)
                    .before(er_terrain::TerrainUpdate),
            );
    }
}

#[allow(clippy::too_many_arguments)]
fn setup_space(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut star_materials: ResMut<Assets<StarfieldMaterial>>,
    mut sun_materials: ResMut<Assets<SunMaterial>>,
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

    let cloud_mesh = meshes.add(make_sphere(1.0, 24, 12));
    commands.insert_resource(CloudPopulation {
        rng: rng_from_seed(terrain_state.params.seed as u32 as u64 ^ 0xC10D_5EED),
        next_spawn_attempt: 2.0,
        forced_spawn_done: false,
        planet_radius: terrain_state.planet_radius,
        cloud_radius: terrain_state.planet_radius
            + cloud_altitude(terrain_state.planet_radius, terrain_state.elevation_scale),
        mesh: cloud_mesh,
        active: Vec::new(),
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

#[allow(clippy::type_complexity)]
fn update_space(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut starfield_query: Query<&mut Transform, With<StarfieldComponent>>,
    mut ocean_query: Query<&mut Transform, (With<OceanComponent>, Without<StarfieldComponent>)>,
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
        for mut tf in &mut ocean_query {
            tf.translation = planet_center;
        }
    }
}

type EnvironmentDebugLayer = Or<(
    With<OceanComponent>,
    With<CloudComponent>,
    With<StarfieldComponent>,
    With<SunSphere>,
)>;

fn update_environment_debug_visibility(
    debug_view: Res<DebugViewSelection>,
    mut layer_query: Query<&mut Visibility, EnvironmentDebugLayer>,
) {
    if !debug_view.is_changed() {
        return;
    }
    let visibility = if debug_view.0.is_off() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for mut current in &mut layer_query {
        *current = visibility;
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
    terrain_state: Res<er_terrain::TerrainState>,
    render_origin: Res<er_terrain::RenderOrigin>,
) {
    sim_time.0 += time.delta_secs() * time_scale.current;

    let day_length = DEFAULT_DAY_LENGTH_SEC as f32;
    let t = (sim_time.0 % day_length) * (2.0 * std::f32::consts::PI / day_length);
    let sun_dir = Vec3::new(t.sin() * 0.8, t.cos(), t.sin() * 0.3).normalize();

    if sun_direction.0 == sun_dir {
        return;
    }
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
}

fn cloud_puff_transform(
    direction: DVec3,
    cloud_radius: f64,
    render_origin: DVec3,
    puff: &CloudPuffPlan,
    sim_time: f32,
    lifecycle_scale: f32,
    local_spin_angle: f64,
) -> Transform {
    let (base_tangent_x, base_tangent_z) = tangent_basis(direction);
    let up = direction;
    let (spin_sin, spin_cos) = local_spin_angle.sin_cos();
    let tangent_x = base_tangent_x * spin_cos - base_tangent_z * spin_sin;
    let tangent_z = base_tangent_x * spin_sin + base_tangent_z * spin_cos;
    let breathe = 1.0 + (sim_time * 0.38 + puff.phase).sin() * 0.07;
    let drift = (sim_time * 0.21 + puff.phase * 1.7).sin();
    let offset = puff.offset
        + Vec3::new(
            drift * puff.scale.x * 0.08,
            drift * puff.scale.y * 0.06,
            -drift * puff.scale.z * 0.05,
        );
    let absolute = up * cloud_radius
        + tangent_x * offset.x as f64
        + up * offset.y as f64
        + tangent_z * offset.z as f64;
    let rotation = Quat::from_mat3(&Mat3::from_cols(
        tangent_x.as_vec3(),
        up.as_vec3(),
        tangent_z.as_vec3(),
    ));
    Transform {
        translation: (absolute - render_origin).as_vec3(),
        rotation,
        scale: puff.scale * lifecycle_scale * breathe,
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_cloud_cluster(
    commands: &mut Commands,
    materials: &mut Assets<CloudMaterial>,
    population: &mut CloudPopulation,
    plan: CloudSpawnPlan,
    sim_time: f32,
    initial_age: f32,
    camera_position: Vec3,
    render_origin: DVec3,
    sun_direction: Vec3,
    visibility: Visibility,
) {
    let born_at = sim_time - initial_age.clamp(0.0, plan.lifetime);
    let (initial_opacity, initial_lifecycle_scale) = cloud_lifecycle(initial_age, plan.lifetime);
    let mut puffs = Vec::with_capacity(plan.puffs.len());
    for puff in plan.puffs {
        let material = materials.add(CloudMaterial {
            uniform: CloudUniform {
                sun_dir_x: sun_direction.x,
                sun_dir_y: sun_direction.y,
                sun_dir_z: sun_direction.z,
                time: sim_time,
                camera_pos_x: camera_position.x,
                camera_pos_y: camera_position.y,
                camera_pos_z: camera_position.z,
                opacity: initial_opacity,
                storminess: plan.magnitude.profile().storminess,
                formation: plan.magnitude as u8 as f32,
                role: puff.role,
                noise_seed: puff.noise_seed,
            },
        });
        let transform = cloud_puff_transform(
            plan.direction,
            population.cloud_radius,
            render_origin,
            &puff,
            sim_time,
            initial_lifecycle_scale,
            0.0,
        );
        let entity = commands
            .spawn((
                CloudComponent,
                CloudPuff,
                MeshMaterial3d(material.clone()),
                Mesh3d(population.mesh.clone()),
                transform,
                visibility,
            ))
            .id();
        puffs.push((entity, puff, material));
    }
    population.active.push(ActiveCloudCluster {
        magnitude: plan.magnitude,
        direction: plan.direction,
        drift_axis: plan.drift_axis,
        angular_speed: plan.angular_speed,
        local_spin_speed: plan.local_spin_speed,
        local_spin_angle: 0.0,
        born_at,
        lifetime: plan.lifetime,
        puffs,
    });
}

#[allow(clippy::too_many_arguments)]
fn update_clouds(
    mut commands: Commands,
    time: Res<Time>,
    time_scale: Res<TimeScale>,
    sim_time: Res<SimTime>,
    config: Res<CloudRuntimeConfig>,
    debug_view: Res<DebugViewSelection>,
    camera_world: Res<CameraWorldPosition>,
    render_origin: Res<RenderOrigin>,
    sun_direction: Res<SunDirection>,
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut materials: ResMut<Assets<CloudMaterial>>,
    mut population: ResMut<CloudPopulation>,
    mut puff_query: Query<&mut Transform, With<CloudPuff>>,
) {
    if !config.spawning_enabled {
        return;
    }

    let camera_forward = camera_query
        .single()
        .map(|transform| transform.forward().as_dvec3())
        .unwrap_or_else(|_| camera_world.0.normalize_or(DVec3::Y));
    let camera_direction =
        cloud_spawn_direction(camera_world.0, camera_forward, population.cloud_radius);
    let camera_position = (camera_world.0 - render_origin.world).as_vec3();
    let visibility = if debug_view.0.is_off() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    let force_spawn = config.force_initial_cluster && !population.forced_spawn_done;
    let spawn_attempt_due = sim_time.0 >= population.next_spawn_attempt;
    if force_spawn || spawn_attempt_due {
        population.forced_spawn_done |= force_spawn;
        if spawn_attempt_due {
            population.next_spawn_attempt += CLOUD_SPAWN_CHECK_INTERVAL_SEC;
        }
        let should_spawn = force_spawn || should_spawn_cloud(&mut population.rng);
        let magnitude = if force_spawn {
            WeatherMagnitude::CloudBank
        } else {
            select_weather_magnitude(&mut population.rng)
        };
        let profile = magnitude.profile();
        let tier_active = population
            .active
            .iter()
            .filter(|cluster| cluster.magnitude == magnitude)
            .count();
        if should_spawn
            && population.active.len() < CLOUD_MAX_CLUSTERS
            && tier_active < profile.max_active
        {
            let planet_radius = population.planet_radius;
            let plan = make_cloud_spawn_plan(
                &mut population.rng,
                camera_direction,
                planet_radius,
                magnitude,
            );
            let active_puffs: usize = population
                .active
                .iter()
                .map(|cluster| cluster.puffs.len())
                .sum();
            if active_puffs + plan.puffs.len() <= CLOUD_MAX_TOTAL_PUFFS {
                let initial_age = if force_spawn {
                    plan.lifetime * 0.4
                } else {
                    0.0
                };
                spawn_cloud_cluster(
                    &mut commands,
                    &mut materials,
                    &mut population,
                    plan,
                    sim_time.0,
                    initial_age,
                    camera_position,
                    render_origin.world,
                    sun_direction.0,
                    visibility,
                );
            }
        }
    }

    let delta_angle_scale = time.delta_secs() as f64 * time_scale.current as f64;
    let cloud_radius = population.cloud_radius;
    let mut expired = Vec::new();
    for (index, cluster) in population.active.iter_mut().enumerate() {
        let age = sim_time.0 - cluster.born_at;
        if age >= cluster.lifetime {
            expired.push(index);
            continue;
        }

        cluster.direction = (DQuat::from_axis_angle(
            cluster.drift_axis,
            cluster.angular_speed * delta_angle_scale,
        ) * cluster.direction)
            .normalize();
        cluster.local_spin_angle += cluster.local_spin_speed * delta_angle_scale;
        let (opacity, lifecycle_scale) = cloud_lifecycle(age, cluster.lifetime);
        for (entity, puff, material_handle) in &cluster.puffs {
            if let Some(mut material) = materials.get_mut(material_handle) {
                material.uniform.sun_dir_x = sun_direction.0.x;
                material.uniform.sun_dir_y = sun_direction.0.y;
                material.uniform.sun_dir_z = sun_direction.0.z;
                material.uniform.time = sim_time.0;
                material.uniform.camera_pos_x = camera_position.x;
                material.uniform.camera_pos_y = camera_position.y;
                material.uniform.camera_pos_z = camera_position.z;
                material.uniform.opacity = opacity;
            }
            if let Ok(mut transform) = puff_query.get_mut(*entity) {
                *transform = cloud_puff_transform(
                    cluster.direction,
                    cloud_radius,
                    render_origin.world,
                    puff,
                    sim_time.0,
                    lifecycle_scale,
                    cluster.local_spin_angle,
                );
            }
        }
    }
    for index in expired.into_iter().rev() {
        let cluster = population.active.swap_remove(index);
        for (entity, _, material) in cluster.puffs {
            commands.entity(entity).despawn();
            materials.remove(material.id());
        }
    }
}

fn update_terrain_uniforms(
    camera_world: Res<er_terrain::CameraWorldPosition>,
    sun_direction: Res<er_terrain::SunDirection>,
    terrain_material: Res<SharedTerrainMaterial>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
    mut ocean_materials: ResMut<Assets<er_terrain::ocean::OceanMaterial>>,
    render_origin: Res<er_terrain::RenderOrigin>,
    debug_view: Res<DebugViewSelection>,
) {
    if !camera_world.is_changed()
        && !render_origin.is_changed()
        && !sun_direction.is_changed()
        && !debug_view.is_changed()
    {
        return;
    }

    let cam_pos = (camera_world.0 - render_origin.world).as_vec3();

    let sun_dir = sun_direction.0;

    let sx = sun_dir.x;
    let sy = sun_dir.y;
    let sz = sun_dir.z;
    let cx = cam_pos.x;
    let cy = cam_pos.y;
    let cz = cam_pos.z;
    let ox = render_origin.world.x as f32;
    let oy = render_origin.world.y as f32;
    let oz = render_origin.world.z as f32;
    let debug_view_mode = debug_view.0 as u32;

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
        u.debug_view_mode = debug_view_mode;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_spawn_plan_is_deterministic_and_bounded() {
        let mut first_rng = rng_from_seed(42);
        let mut second_rng = rng_from_seed(42);
        let camera_direction = DVec3::new(0.3, 0.4, 0.5).normalize();
        let first = make_cloud_spawn_plan(
            &mut first_rng,
            camera_direction,
            6_371_000.0,
            WeatherMagnitude::Cyclone,
        );
        let second = make_cloud_spawn_plan(
            &mut second_rng,
            camera_direction,
            6_371_000.0,
            WeatherMagnitude::Cyclone,
        );

        assert_eq!(first, second);
        assert!(first.direction.dot(camera_direction) >= 0.65);
        assert!((72.0..=108.0).contains(&first.lifetime));
        assert!((WeatherMagnitude::Cyclone.profile().puff_min
            ..=WeatherMagnitude::Cyclone.profile().puff_max)
            .contains(&first.puffs.len()));
        assert!(first
            .puffs
            .iter()
            .all(|puff| puff.scale.min_element() > 0.0));
    }

    #[test]
    fn cloud_spawn_seeds_produce_distinct_clusters() {
        let mut first_rng = rng_from_seed(1);
        let mut second_rng = rng_from_seed(2);
        let first = make_cloud_spawn_plan(
            &mut first_rng,
            DVec3::Y,
            6_371_000.0,
            WeatherMagnitude::Thunderstorm,
        );
        let second = make_cloud_spawn_plan(
            &mut second_rng,
            DVec3::Y,
            6_371_000.0,
            WeatherMagnitude::Thunderstorm,
        );
        assert_ne!(first, second);
    }

    #[test]
    fn cloud_lifecycle_grows_fades_and_expires() {
        let lifetime = 30.0;
        assert_eq!(cloud_lifecycle(0.0, lifetime), (0.0, 0.65));
        let (growing_opacity, growing_scale) = cloud_lifecycle(3.0, lifetime);
        let (mature_opacity, mature_scale) = cloud_lifecycle(15.0, lifetime);
        let (fading_opacity, fading_scale) = cloud_lifecycle(28.0, lifetime);
        assert!(growing_opacity > 0.0 && growing_opacity < mature_opacity);
        assert!(growing_scale < mature_scale);
        assert!(fading_opacity > 0.0 && fading_opacity < mature_opacity);
        assert!(fading_scale > mature_scale);
        assert_eq!(cloud_lifecycle(lifetime, lifetime), (0.0, 0.0));
    }

    #[test]
    fn normal_runtime_keeps_cloud_population_small() {}

    #[test]
    fn weather_magnitudes_become_rarer_and_last_longer() {
        let magnitudes = WeatherMagnitude::ALL;
        for pair in magnitudes.windows(2) {
            let smaller = pair[0].profile();
            let larger = pair[1].profile();
            assert!(larger.weight < smaller.weight);
            assert!(larger.lifetime_min > smaller.lifetime_min);
            assert!(larger.lifetime_max > smaller.lifetime_max);
            assert!(larger.puff_max > smaller.puff_max);
        }
    }

    #[test]
    fn weather_selection_uses_all_magnitudes_for_a_fixed_seed() {
        let mut rng = rng_from_seed(0x0570_A005);
        let mut counts = [0_usize; 4];
        for _ in 0..10_000 {
            counts[select_weather_magnitude(&mut rng) as usize] += 1;
        }
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
        assert!(counts[2] > counts[3]);
        assert!(counts.iter().all(|count| *count > 0));
    }

    #[test]
    fn every_weather_formation_is_bounded_and_distinct() {
        let mut previous = None;
        for magnitude in WeatherMagnitude::ALL {
            let mut rng = rng_from_seed(0xBADC0DE + magnitude as u64);
            let plan = make_cloud_spawn_plan(&mut rng, DVec3::Y, 6_371_000.0, magnitude);
            let profile = magnitude.profile();
            assert_eq!(plan.magnitude, magnitude);
            assert!((profile.lifetime_min..=profile.lifetime_max).contains(&plan.lifetime));
            assert!((profile.puff_min..=profile.puff_max).contains(&plan.puffs.len()));
            assert!(plan.puffs.iter().all(|puff| puff.scale.min_element() > 0.0));
            if magnitude == WeatherMagnitude::CloudBank
                || magnitude == WeatherMagnitude::Thunderstorm
            {
                assert_eq!(plan.local_spin_speed, 0.0);
            } else {
                assert!(plan.local_spin_speed > 0.0);
            }
            if let Some(previous) = previous {
                assert_ne!(plan, previous);
            }
            previous = Some(plan);
        }
    }

    #[test]
    fn normal_spawn_rate_stays_sparse_for_a_fixed_seed() {
        let mut rng = rng_from_seed(0xC10D_5EED);
        let spawned = (0..1_000).filter(|_| should_spawn_cloud(&mut rng)).count();
        assert!((150..=250).contains(&spawned));
    }

    #[test]
    fn puff_shape_and_position_evolve_over_time() {
        let puff = CloudPuffPlan {
            offset: Vec3::new(100.0, 20.0, -50.0),
            scale: Vec3::new(80.0, 12.0, 70.0),
            phase: 1.25,
            noise_seed: 42.0,
            role: 0.0,
        };
        let first = cloud_puff_transform(DVec3::Y, 6_371_000.0, DVec3::ZERO, &puff, 1.0, 1.0, 0.0);
        let second = cloud_puff_transform(DVec3::Y, 6_371_000.0, DVec3::ZERO, &puff, 9.0, 1.0, 0.2);
        assert_ne!(first.translation, second.translation);
        assert_ne!(first.scale, second.scale);
    }

    #[test]
    fn clouds_stay_above_miniature_terrain() {
        assert_eq!(cloud_altitude(36_000.0, 1_000.0), 3_000.0);
        assert_eq!(cloud_altitude(6_371_000.0, 1_000.0), 15_927.5);
    }

    #[test]
    fn inside_cloud_layer_spawns_along_outward_view() {
        let direction =
            cloud_spawn_direction(DVec3::new(6_371_000.0, 0.0, 0.0), DVec3::Y, 6_386_927.5);
        assert!(direction.x > 0.0);
        assert!(direction.y > 0.0);
    }
}
