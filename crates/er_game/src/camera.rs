use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use er_terrain::{FrameProfiler, TerrainState};
use std::time::Instant;

#[derive(Component)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub min_altitude: f32,
    pub max_altitude: f32,
    pub smoothed_distance: f32,
    pub smoothed_target: Vec3,
    pub smoothing: f32,
}

impl OrbitCamera {
    pub fn for_planet(radius: f32, elevation_scale: f32) -> Self {
        let min_altitude = 10.0_f32.max(elevation_scale.abs() * 0.01);
        let max_altitude = radius * 29.0;
        let distance = radius * 2.5;
        Self {
            yaw: 0.0,
            pitch: 0.3,
            distance,
            target: Vec3::ZERO,
            min_altitude,
            max_altitude,
            smoothed_distance: distance,
            smoothed_target: Vec3::ZERO,
            smoothing: 10.0,
        }
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        let radius = 36000.0;
        Self::for_planet(radius, 1000.0)
    }
}

pub struct CameraPlugin;

impl Default for CameraPlugin {
    fn default() -> Self {
        Self
    }
}

#[derive(SystemSet, Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct CameraUpdate;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (orbit_camera_input, orbit_camera_update)
                .chain()
                .in_set(CameraUpdate),
        );
    }
}

use crate::space::{StarfieldComponent, SunLight, SunSphere};

fn orbit_camera_update(
    mut query: Query<
        (&mut OrbitCamera, &mut Transform),
        (
            With<Camera3d>,
            Without<SunLight>,
            Without<SunSphere>,
            Without<StarfieldComponent>,
        ),
    >,
    time: Res<Time>,
    terrain_state: Res<TerrainState>,
    mut profiler: ResMut<FrameProfiler>,
) {
    let t0 = Instant::now();
    let dt = time.delta_secs();
    for (mut orbit, mut transform) in &mut query {
        clamp_orbit_to_terrain(&mut orbit, &terrain_state);
        let alpha = 1.0 - (-orbit.smoothing * dt).exp();
        orbit.smoothed_distance = orbit.smoothed_distance.lerp(orbit.distance, alpha);
        orbit.smoothed_target = orbit.smoothed_target.lerp(orbit.target, alpha);

        let cp = orbit.pitch.cos();
        let direction = Vec3::new(
            cp * orbit.yaw.sin(),
            orbit.pitch.sin(),
            cp * orbit.yaw.cos(),
        )
        .normalize();

        transform.translation = orbit.smoothed_target + direction * orbit.smoothed_distance;
        let up = if direction.abs().dot(Vec3::Y) > 0.999 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        transform.look_at(orbit.smoothed_target, up);
    }
    profiler.record("camera_update", t0.elapsed());
}

fn orbit_camera_input(
    accumulated_motion: Res<AccumulatedMouseMotion>,
    accumulated_scroll: Res<AccumulatedMouseScroll>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut OrbitCamera>,
    time: Res<Time>,
    terrain_state: Res<TerrainState>,
) {
    let mut orbit = match query.single_mut() {
        Ok(o) => o,
        Err(_) => return,
    };

    let dt = time.delta_secs();
    let orbit_speed = 2.5 * (orbit.distance / 50000.0).clamp(0.1, 3.0);
    let zoom_speed = orbit.distance * 0.5;

    if keys.pressed(KeyCode::KeyA) {
        orbit.yaw += orbit_speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        orbit.yaw -= orbit_speed * dt;
    }
    if keys.pressed(KeyCode::KeyW) {
        orbit.pitch = (orbit.pitch + orbit_speed * dt).min(1.5);
    }
    if keys.pressed(KeyCode::KeyS) {
        orbit.pitch = (orbit.pitch - orbit_speed * dt).max(-1.5);
    }

    if buttons.pressed(MouseButton::Left) {
        orbit.yaw -= accumulated_motion.delta.x * 0.005;
        orbit.pitch = (orbit.pitch + accumulated_motion.delta.y * 0.005).clamp(-1.5, 1.5);
    }

    orbit.distance *= 1.0 - accumulated_scroll.delta.y * 0.1;

    if keys.pressed(KeyCode::ArrowUp) {
        orbit.distance -= zoom_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        orbit.distance += zoom_speed * dt;
    }

    clamp_orbit_to_terrain(&mut orbit, &terrain_state);
}

fn orbit_direction(orbit: &OrbitCamera) -> Vec3 {
    let cp = orbit.pitch.cos();
    Vec3::new(
        cp * orbit.yaw.sin(),
        orbit.pitch.sin(),
        cp * orbit.yaw.cos(),
    )
    .normalize()
}

fn clamp_orbit_to_terrain(orbit: &mut OrbitCamera, terrain_state: &TerrainState) {
    // Orbit targets are planet-center-relative. Sample the same synchronous
    // field used by mesh workers so the camera cannot descend through a ridge.
    let direction = orbit_direction(orbit).as_dvec3();
    let terrain_elevation =
        terrain_state.field.sample(direction).elevation * terrain_state.elevation_scale as f64;
    let minimum_distance =
        (terrain_state.planet_radius + terrain_elevation).max(0.0) as f32 + orbit.min_altitude;
    let maximum_distance = terrain_state.planet_radius as f32 + orbit.max_altitude;
    orbit.distance = orbit.distance.clamp(minimum_distance, maximum_distance);
    orbit.smoothed_distance = orbit.smoothed_distance.max(minimum_distance);
}
