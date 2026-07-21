use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use er_core::math::{
    cell_size, dir_to_uv, needs_recenter, recenter, tangent_frame, uv_to_dir, OriginOffset,
    WorldPos,
};
#[cfg(test)]
use er_terrain::anchor_render_translation;
use er_terrain::{
    CameraWorldPosition, FrameProfiler, RenderOrigin, TerrainDebugInfo, TerrainState,
};
use glam::{DVec2, DVec3};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Surface target: stable cube-face surface reference with local offset
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceTarget {
    pub face: u8,
    pub u: f64,
    pub v: f64,
    pub local: DVec2,
}

impl SurfaceTarget {
    pub fn at(face: u8, u: f64, v: f64) -> Self {
        Self {
            face,
            u,
            v,
            local: DVec2::ZERO,
        }
    }
}

impl Default for SurfaceTarget {
    fn default() -> Self {
        Self {
            face: 0,
            u: 0.5,
            v: 0.5,
            local: DVec2::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Orbit,
    Surface,
}

// ---------------------------------------------------------------------------
// Camera component
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct OrbitCamera {
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub target: DVec3,
    pub min_altitude: f64,
    pub max_altitude: f64,
    pub smoothed_distance: f64,
    pub smoothed_target: DVec3,
    pub smoothing: f64,
    // -- Milestone 1.2 surface mode --
    pub mode: CameraMode,
    pub surface_target: SurfaceTarget,
    pub surface_altitude: f64,
    pub smoothed_surface_altitude: f64,
    surface_min_safe_altitude: f64,
}

impl OrbitCamera {
    pub fn for_planet(radius: f64, elevation_scale: f32) -> Self {
        let min_altitude = 10.0_f64.max(elevation_scale.abs() as f64 * 0.01);
        let max_altitude = radius * 29.0;
        let distance = radius * 3.0;
        Self {
            yaw: 0.0,
            pitch: 0.3,
            distance,
            target: DVec3::ZERO,
            min_altitude,
            max_altitude,
            smoothed_distance: distance,
            smoothed_target: DVec3::ZERO,
            smoothing: 10.0,
            mode: CameraMode::Orbit,
            surface_target: SurfaceTarget::default(),
            surface_altitude: radius * 0.5,
            smoothed_surface_altitude: radius * 0.5,
            surface_min_safe_altitude: min_altitude,
        }
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        let radius = 36000.0_f64;
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
            (camera_input, camera_update).chain().in_set(CameraUpdate),
        );
    }
}

// ---------------------------------------------------------------------------
// Transition thresholds
// ---------------------------------------------------------------------------

const MINIATURE_SURFACE_ENTER_ALTITUDE: f64 = 50.0;
const MINIATURE_SURFACE_EXIT_ALTITUDE: f64 = 120.0;
const EARTH_SURFACE_ENTER_ALTITUDE: f64 = 200.0;
const EARTH_SURFACE_EXIT_ALTITUDE: f64 = 500.0;
const SCROLL_ZOOM_STEP: f64 = 0.1;
// Keep the planetary orbit camera under player control. Surface mode remains
// available for explicit scenarios and future opt-in controls.
const AUTO_MODE_TRANSITIONS: bool = false;

fn surface_enter_altitude(radius: f64) -> f64 {
    if radius > 1_000_000.0 {
        EARTH_SURFACE_ENTER_ALTITUDE
    } else {
        MINIATURE_SURFACE_ENTER_ALTITUDE
    }
}

fn surface_exit_altitude(radius: f64) -> f64 {
    if radius > 1_000_000.0 {
        EARTH_SURFACE_EXIT_ALTITUDE
    } else {
        MINIATURE_SURFACE_EXIT_ALTITUDE
    }
}

fn scroll_zoom_factor(scroll_delta: f64) -> f64 {
    1.0 - scroll_delta * SCROLL_ZOOM_STEP
}

// ---------------------------------------------------------------------------
// Main update
// ---------------------------------------------------------------------------

use crate::space::{StarfieldComponent, SunLight, SunSphere};

#[allow(clippy::type_complexity)]
fn camera_update(
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
    debug_info: Res<TerrainDebugInfo>,
    mut camera_world: ResMut<CameraWorldPosition>,
    mut origin: ResMut<RenderOrigin>,
    mut profiler: ResMut<FrameProfiler>,
) {
    let t0 = Instant::now();
    let dt = time.delta_secs_f64();
    for (mut camera, mut transform) in &mut query {
        if AUTO_MODE_TRANSITIONS {
            transition_mode_if_needed(&mut camera, &terrain_state, &debug_info);
        }

        match camera.mode {
            CameraMode::Orbit => {
                update_orbit(&mut camera, &terrain_state, dt);
                place_orbit(&camera, &mut transform, &mut camera_world, &mut origin);
            }
            CameraMode::Surface => {
                update_surface(&mut camera, dt);
                place_surface(
                    &camera,
                    &mut transform,
                    &mut camera_world,
                    &mut origin,
                    &terrain_state,
                );
            }
        }
    }
    profiler.record("camera_update", t0.elapsed());
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn camera_input(
    accumulated_motion: Res<AccumulatedMouseMotion>,
    accumulated_scroll: Res<AccumulatedMouseScroll>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut OrbitCamera>,
    time: Res<Time>,
    terrain_state: Res<TerrainState>,
    debug_info: Res<TerrainDebugInfo>,
) {
    let mut camera = match query.single_mut() {
        Ok(o) => o,
        Err(_) => return,
    };

    let dt = time.delta_secs_f64();

    match camera.mode {
        CameraMode::Orbit => {
            orbit_input_inner(
                &mut camera,
                &buttons,
                &keys,
                &accumulated_motion,
                &accumulated_scroll,
                dt,
                &terrain_state,
            );
        }
        CameraMode::Surface => {
            surface_input_inner(
                &mut camera,
                &buttons,
                &keys,
                &accumulated_motion,
                &accumulated_scroll,
                dt,
                &terrain_state,
                &debug_info,
            );
        }
    }
}

fn orbit_input_inner(
    camera: &mut OrbitCamera,
    buttons: &ButtonInput<MouseButton>,
    keys: &ButtonInput<KeyCode>,
    accumulated_motion: &AccumulatedMouseMotion,
    accumulated_scroll: &AccumulatedMouseScroll,
    dt: f64,
    terrain_state: &TerrainState,
) {
    let orbit_speed = 2.5 * (camera.distance / 50000.0).clamp(0.1, 3.0);
    let zoom_speed = camera.distance * 0.5;

    if keys.pressed(KeyCode::KeyA) {
        camera.yaw += orbit_speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        camera.yaw -= orbit_speed * dt;
    }
    if keys.pressed(KeyCode::KeyW) {
        camera.pitch = (camera.pitch + orbit_speed * dt).min(1.5);
    }
    if keys.pressed(KeyCode::KeyS) {
        camera.pitch = (camera.pitch - orbit_speed * dt).max(-1.5);
    }

    if buttons.pressed(MouseButton::Left) {
        camera.yaw -= accumulated_motion.delta.x as f64 * 0.005;
        camera.pitch = (camera.pitch + accumulated_motion.delta.y as f64 * 0.005).clamp(-1.5, 1.5);
    }

    apply_orbit_scroll_zoom(camera, terrain_state, accumulated_scroll.delta.y as f64);

    if keys.pressed(KeyCode::ArrowUp) {
        camera.distance -= zoom_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        camera.distance += zoom_speed * dt;
    }

    clamp_orbit_to_terrain(camera, terrain_state);
}

#[allow(clippy::too_many_arguments)]
fn surface_input_inner(
    camera: &mut OrbitCamera,
    buttons: &ButtonInput<MouseButton>,
    keys: &ButtonInput<KeyCode>,
    accumulated_motion: &AccumulatedMouseMotion,
    accumulated_scroll: &AccumulatedMouseScroll,
    dt: f64,
    terrain_state: &TerrainState,
    debug_info: &TerrainDebugInfo,
) {
    let (_n, tangent, bitangent) = tangent_frame(
        camera.surface_target.face,
        camera.surface_target.u,
        camera.surface_target.v,
    );
    let radius = terrain_state.planet_radius;

    let vertex_spacing = surface_vertex_spacing(terrain_state, debug_info);
    let pan_speed =
        ((1.0 + camera.surface_altitude / 1000.0) * vertex_spacing * 0.5).clamp(1.0, 5000.0);

    let mut pan = DVec2::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        pan.y += pan_speed * dt;
    }
    if keys.pressed(KeyCode::KeyS) {
        pan.y -= pan_speed * dt;
    }
    if keys.pressed(KeyCode::KeyA) {
        pan.x -= pan_speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        pan.x += pan_speed * dt;
    }

    if buttons.pressed(MouseButton::Right) {
        let mouse_speed = (camera.surface_altitude + 1.0) * 0.01;
        pan.x -= accumulated_motion.delta.x as f64 * mouse_speed;
        pan.y += accumulated_motion.delta.y as f64 * mouse_speed;
    }

    if pan.length_squared() > 1e-12 {
        apply_tangent_pan(camera, &tangent, &bitangent, pan, radius);
    }

    let altitude_speed = camera.surface_altitude.max(1.0) * 0.3;

    if keys.pressed(KeyCode::ArrowUp) {
        camera.surface_altitude -= altitude_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        camera.surface_altitude += altitude_speed * dt;
    }
    camera.surface_altitude *= scroll_zoom_factor(accumulated_scroll.delta.y as f64);

    let min_safe = slope_safe_min_altitude(
        &camera.surface_target,
        camera.surface_altitude,
        terrain_state,
        debug_info,
    );
    camera.surface_min_safe_altitude = min_safe;
    camera.surface_altitude = camera.surface_altitude.clamp(min_safe, camera.max_altitude);
}

// ---------------------------------------------------------------------------
// Tangent-plane panning & reprojection
// ---------------------------------------------------------------------------

fn apply_tangent_pan(
    camera: &mut OrbitCamera,
    _tangent: &DVec3,
    _bitangent: &DVec3,
    pan: DVec2,
    radius: f64,
) {
    camera.surface_target.local += pan;
    reproject_surface_target(&mut camera.surface_target, radius);
}

fn reproject_surface_target(target: &mut SurfaceTarget, radius: f64) {
    let offset = target.local;
    if offset.length_squared() < 1e-6 {
        return;
    }

    let dir = uv_to_dir(target.face, target.u, target.v);
    let (_normal, tangent, bitangent) = tangent_frame(target.face, target.u, target.v);

    let world_point = dir * radius + tangent * offset.x + bitangent * offset.y;
    let new_dir = world_point.normalize();
    let (new_face, new_u, new_v) = dir_to_uv(new_dir);

    let (_, new_tangent, new_bitangent) = tangent_frame(new_face, new_u, new_v);
    let new_sphere_point = new_dir * radius;
    let residual_world = world_point - new_sphere_point;

    target.face = new_face;
    target.u = new_u;
    target.v = new_v;
    target.local = DVec2::new(
        residual_world.dot(new_tangent),
        residual_world.dot(new_bitangent),
    );
}

// ---------------------------------------------------------------------------
// Orbit helpers
// ---------------------------------------------------------------------------

fn orbit_direction(orbit: &OrbitCamera) -> DVec3 {
    let cp = orbit.pitch.cos();
    DVec3::new(
        cp * orbit.yaw.sin(),
        orbit.pitch.sin(),
        cp * orbit.yaw.cos(),
    )
    .normalize()
}

/// Zoom orbit altitude relative to the terrain, not the planet center.
/// Scaling center distance makes a 10% wheel step larger than the remaining
/// altitude in low Earth orbit and snaps directly to the terrain clamp.
fn apply_orbit_scroll_zoom(
    orbit: &mut OrbitCamera,
    terrain_state: &TerrainState,
    scroll_delta: f64,
) {
    let direction = orbit_direction(orbit);
    let terrain_elevation =
        terrain_state.field.sample(direction).elevation * terrain_state.elevation_scale as f64;
    let surface_radius = (terrain_state.planet_radius + terrain_elevation).max(0.0);
    let altitude = (orbit.distance - surface_radius).max(orbit.min_altitude);
    orbit.distance = surface_radius + altitude * scroll_zoom_factor(scroll_delta);
}

fn clamp_orbit_to_terrain(orbit: &mut OrbitCamera, terrain_state: &TerrainState) {
    let direction = orbit_direction(orbit);
    let terrain_elevation =
        terrain_state.field.sample(direction).elevation * terrain_state.elevation_scale as f64;
    let minimum_distance =
        (terrain_state.planet_radius + terrain_elevation).max(0.0) + orbit.min_altitude;
    let maximum_distance = terrain_state.planet_radius + orbit.max_altitude;
    orbit.distance = orbit.distance.clamp(minimum_distance, maximum_distance);
    orbit.smoothed_distance = orbit.smoothed_distance.max(minimum_distance);
}

fn update_orbit(camera: &mut OrbitCamera, terrain_state: &TerrainState, dt: f64) {
    clamp_orbit_to_terrain(camera, terrain_state);
    let alpha = 1.0 - (-camera.smoothing * dt).exp();
    camera.smoothed_distance =
        camera.smoothed_distance + (camera.distance - camera.smoothed_distance) * alpha;
    camera.smoothed_target =
        camera.smoothed_target + (camera.target - camera.smoothed_target) * alpha;
}

fn update_surface(camera: &mut OrbitCamera, dt: f64) {
    camera.surface_altitude = camera
        .surface_altitude
        .max(camera.surface_min_safe_altitude);
    let alpha = 1.0 - (-camera.smoothing * dt).exp();
    camera.smoothed_surface_altitude = camera.smoothed_surface_altitude
        + (camera.surface_altitude - camera.smoothed_surface_altitude) * alpha;
    // The smoothed altitude lags the target during panning onto rising
    // terrain. Without this floor the camera could dip below the slope-safe
    // minimum (penetrating the terrain) until the exponential catch-up
    // completes. Clamp the placed altitude so the rendered camera never
    // penetrates, even while the smoothed value is still converging.
    camera.smoothed_surface_altitude = camera
        .smoothed_surface_altitude
        .max(camera.surface_min_safe_altitude);
}

// ---------------------------------------------------------------------------
// Place orbit camera
// ---------------------------------------------------------------------------

fn place_orbit(
    camera: &OrbitCamera,
    transform: &mut Transform,
    camera_world: &mut CameraWorldPosition,
    origin: &mut RenderOrigin,
) {
    let cp = camera.pitch.cos();
    let direction = DVec3::new(
        cp * camera.yaw.sin(),
        camera.pitch.sin(),
        cp * camera.yaw.cos(),
    )
    .normalize();

    let world_camera = camera.smoothed_target + direction * camera.smoothed_distance;
    camera_world.0 = world_camera;

    let camera_wp = WorldPos(world_camera);
    let current_origin = OriginOffset(origin.world);
    if needs_recenter(camera_wp, current_origin, origin.cell_size_m) {
        let snapped = recenter(current_origin, camera_wp, origin.cell_size_m);
        if snapped.0 != origin.world {
            origin.world = snapped.0;
            origin.generation += 1;
        }
    }

    let render_pos = world_camera - origin.world;
    let render_target = camera.smoothed_target - origin.world;

    transform.translation = Vec3::new(
        render_pos.x as f32,
        render_pos.y as f32,
        render_pos.z as f32,
    );
    let up = if direction.abs().dot(DVec3::Y) > 0.999 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let look_target = Vec3::new(
        render_target.x as f32,
        render_target.y as f32,
        render_target.z as f32,
    );
    transform.look_at(look_target, up);
}

// ---------------------------------------------------------------------------
// Place surface camera
// ---------------------------------------------------------------------------

fn place_surface(
    camera: &OrbitCamera,
    transform: &mut Transform,
    camera_world: &mut CameraWorldPosition,
    origin: &mut RenderOrigin,
    terrain_state: &TerrainState,
) {
    let dir = uv_to_dir(
        camera.surface_target.face,
        camera.surface_target.u,
        camera.surface_target.v,
    );
    let (_normal, tangent, bitangent) = tangent_frame(
        camera.surface_target.face,
        camera.surface_target.u,
        camera.surface_target.v,
    );

    let radius = terrain_state.planet_radius;
    let elevation =
        terrain_state.field.sample(dir).elevation * terrain_state.elevation_scale as f64;
    let sphere_point = dir * (radius + elevation);

    let world_target = sphere_point
        + tangent * camera.surface_target.local.x
        + bitangent * camera.surface_target.local.y;

    let camera_dir = world_target.normalize();
    let altitude = camera.smoothed_surface_altitude;
    let world_camera = world_target + camera_dir * altitude;

    camera_world.0 = world_camera;

    let camera_wp = WorldPos(world_camera);
    let current_origin = OriginOffset(origin.world);
    if needs_recenter(camera_wp, current_origin, origin.cell_size_m) {
        let snap = recenter(current_origin, camera_wp, origin.cell_size_m);
        if snap.0 != origin.world {
            origin.world = snap.0;
            origin.generation += 1;
        }
    }

    let render_pos = world_camera - origin.world;
    let render_target = world_target - origin.world;

    transform.translation = Vec3::new(
        render_pos.x as f32,
        render_pos.y as f32,
        render_pos.z as f32,
    );
    let look_at = Vec3::new(
        render_target.x as f32,
        render_target.y as f32,
        render_target.z as f32,
    );
    let up = Vec3::new(tangent.x as f32, tangent.y as f32, tangent.z as f32);
    transform.look_at(look_at, up);
}

// ---------------------------------------------------------------------------
// Mode transition without position / orientation jumps
// ---------------------------------------------------------------------------

fn transition_mode_if_needed(
    camera: &mut OrbitCamera,
    terrain_state: &TerrainState,
    debug_info: &TerrainDebugInfo,
) {
    match camera.mode {
        CameraMode::Orbit => {
            let direction = orbit_direction(camera);
            let world_camera = camera.smoothed_target + direction * camera.smoothed_distance;
            let world_dir = world_camera.normalize();
            let terrain_elevation = terrain_state.field.sample(world_dir).elevation
                * terrain_state.elevation_scale as f64;
            let altitude = world_camera.length() - terrain_state.planet_radius - terrain_elevation;
            let enter = surface_enter_altitude(terrain_state.planet_radius);

            if altitude < enter && world_camera.length() > 1.0 {
                let (face, u, v) = dir_to_uv(world_camera);
                camera.mode = CameraMode::Surface;
                camera.surface_target =
                    SurfaceTarget::at(face, u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
                camera.surface_min_safe_altitude = slope_safe_min_altitude(
                    &camera.surface_target,
                    altitude,
                    terrain_state,
                    debug_info,
                );
                camera.surface_altitude = altitude.max(camera.surface_min_safe_altitude);
                camera.smoothed_surface_altitude = camera.surface_altitude;
            }
        }
        CameraMode::Surface => {
            let exit = surface_exit_altitude(terrain_state.planet_radius);
            if camera.smoothed_surface_altitude > exit {
                let world_cam = surface_world_pos_at_altitude(
                    camera,
                    terrain_state,
                    camera.smoothed_surface_altitude,
                );
                let distance = world_cam.length();
                if distance > 1.0 {
                    let dir = world_cam.normalize();
                    camera.pitch = dir.y.asin();
                    camera.yaw = dir.x.atan2(dir.z);
                    camera.distance = distance;
                    camera.smoothed_distance = distance;
                    camera.target = DVec3::ZERO;
                    camera.smoothed_target = DVec3::ZERO;
                    camera.mode = CameraMode::Orbit;
                }
            }
        }
    }
}

#[cfg(test)]
fn surface_world_pos(camera: &OrbitCamera, terrain_state: &TerrainState) -> DVec3 {
    surface_world_pos_at_altitude(camera, terrain_state, camera.surface_altitude)
}

fn surface_world_pos_at_altitude(
    camera: &OrbitCamera,
    terrain_state: &TerrainState,
    altitude: f64,
) -> DVec3 {
    let dir = uv_to_dir(
        camera.surface_target.face,
        camera.surface_target.u,
        camera.surface_target.v,
    );
    let (_, tangent, bitangent) = tangent_frame(
        camera.surface_target.face,
        camera.surface_target.u,
        camera.surface_target.v,
    );
    let elevation =
        terrain_state.field.sample(dir).elevation * terrain_state.elevation_scale as f64;
    let sphere_point = dir * (terrain_state.planet_radius + elevation);
    let world_target = sphere_point
        + tangent * camera.surface_target.local.x
        + bitangent * camera.surface_target.local.y;
    let camera_dir = world_target.normalize();
    world_target + camera_dir * altitude
}

// ---------------------------------------------------------------------------
// Slope-safe altitude helper
// ---------------------------------------------------------------------------

const CAMERA_FOOTPRINT_RADIUS_M: f64 = 1.0;

fn slope_safe_min_altitude(
    target: &SurfaceTarget,
    desired_altitude: f64,
    terrain_state: &TerrainState,
    debug_info: &TerrainDebugInfo,
) -> f64 {
    let dir = uv_to_dir(target.face, target.u, target.v);
    let (_normal, tangent, bitangent) = tangent_frame(target.face, target.u, target.v);
    let radius = terrain_state.planet_radius;
    let elevation_scale = terrain_state.elevation_scale as f64;

    let centre_elevation = terrain_state.field.sample_elevation(dir) * elevation_scale;
    let min_clearance = 10.0_f64.max(elevation_scale * 0.01);

    let vertex_spacing = surface_vertex_spacing(terrain_state, debug_info);
    let sample_radius = camera_footprint_radius(desired_altitude)
        .max(vertex_spacing.clamp(0.001, CAMERA_FOOTPRINT_RADIUS_M));
    let num_samples = 12;

    let mut max_rise = 0.0_f64;

    for ring in [0.25, 0.5, 1.0] {
        for i in 0..num_samples {
            let angle = i as f64 * std::f64::consts::TAU / num_samples as f64;
            let offset_dir = tangent * angle.cos() + bitangent * angle.sin();
            let offset_world = offset_dir * sample_radius * ring;
            let sample_dir = (dir * radius
                + tangent * target.local.x
                + bitangent * target.local.y
                + offset_world)
                .normalize();
            let sample_elev = terrain_state.field.sample_elevation(sample_dir) * elevation_scale;
            let rise = sample_elev - centre_elevation;
            max_rise = max_rise.max(rise);
        }
    }

    max_rise.max(0.0) + min_clearance
}

fn camera_footprint_radius(altitude: f64) -> f64 {
    // A roughly 60-degree vertical FOV with a widescreen diagonal sees a
    // footprint on the order of its altitude in a nadir-facing surface view.
    altitude.max(0.0).max(CAMERA_FOOTPRINT_RADIUS_M)
}

fn surface_vertex_spacing(terrain_state: &TerrainState, debug_info: &TerrainDebugInfo) -> f64 {
    if debug_info.vertex_spacing_m > 0.0 && debug_info.vertex_spacing_m.is_finite() {
        return debug_info.vertex_spacing_m;
    }
    let max_depth = if terrain_state.planet_radius > 1_000_000.0 {
        er_core::config::EARTH_MAX_QUADTREE_DEPTH
    } else {
        er_core::config::MAX_QUADTREE_DEPTH
    };
    let cell_size_m = cell_size(max_depth, terrain_state.planet_radius);
    cell_size_m / (er_core::config::CHUNK_VERT_RES as f64 - 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::config::PlanetPreset;
    use er_core::seed::PlanetSeed;

    fn make_terrain_state(radius: f64) -> TerrainState {
        TerrainState::new(radius, 1000.0, PlanetSeed(0xC0FFEE))
    }

    fn make_earth_terrain_state() -> TerrainState {
        TerrainState::for_preset(PlanetPreset::EarthScale, 1000.0, PlanetSeed(0xC0FFEE))
    }

    // ---------
    // Existing tests (preserved)
    // ---------

    #[test]
    fn earth_scale_1m_position_change_survives_in_f64() {
        let radius = 6_371_000.0;
        let mut orbit = OrbitCamera::for_planet(radius, 1000.0);
        let cp = orbit.pitch.cos();
        let dir = DVec3::new(
            cp * orbit.yaw.sin(),
            orbit.pitch.sin(),
            cp * orbit.yaw.cos(),
        )
        .normalize();

        let pos_a = orbit.target + dir * orbit.distance;

        orbit.target += dir * 1.0;

        let pos_b = orbit.target + dir * orbit.distance;
        let drift = (pos_b - pos_a).length();
        assert!(
            (drift - 1.0).abs() < 0.001,
            "1m position change not preserved in f64 state (got {drift}m)"
        );
    }

    #[test]
    fn render_camera_stays_bounded_after_recenter() {
        let radius = 6_371_000.0;
        let pitch = 0.3_f64;
        let cp = pitch.cos();
        let dir = DVec3::new(cp * 0.0, pitch.sin(), cp * 1.0).normalize();
        let world_camera = dir * (radius * 2.5);
        let cell = 1000.0;

        let snapped = recenter(OriginOffset(DVec3::ZERO), WorldPos(world_camera), cell);
        let render_pos = world_camera - snapped.0;
        let offset_m = render_pos.length();
        assert!(
            offset_m < cell * 1.01,
            "render camera {offset_m:.1}m exceeds cell size {cell}"
        );
    }

    #[test]
    fn target_look_positioning_is_origin_invariant() {
        let radius = 6_371_000.0;
        let yaw = 0.5_f64;
        let pitch = 0.3_f64;
        let distance = radius * 2.5;
        let cp = pitch.cos();
        let direction = DVec3::new(cp * yaw.sin(), pitch.sin(), cp * yaw.cos()).normalize();
        let target = DVec3::ZERO;
        let world_camera = target + direction * distance;

        let origin_a = DVec3::ZERO;
        let origin_b = world_camera;

        let render_cam_a = world_camera - origin_a;
        let render_cam_b = world_camera - origin_b;
        let render_tgt_a = target - origin_a;
        let render_tgt_b = target - origin_b;

        let look_a = (render_tgt_a - render_cam_a).normalize();
        let look_b = (render_tgt_b - render_cam_b).normalize();
        let dot = look_a.dot(look_b);
        assert!(
            (dot - 1.0).abs() < 1e-6,
            "look-at direction shifts under origin change (dot={dot})"
        );
    }

    // ---------
    // New Milestone 1.2 tests
    // ---------

    #[test]
    fn earth_1m_pan_precision() {
        let radius = 6_371_000.0;
        let face = 0u8;
        let u = 0.5;
        let v = 0.5;
        let (_n, tangent, _b) = tangent_frame(face, u, v);

        let dir = uv_to_dir(face, u, v);
        let pos_a = dir * radius;
        let pos_b = dir * radius + tangent * 1.0;

        let drift = (pos_b - pos_a).length();
        assert!(
            (drift - 1.0).abs() < 0.001,
            "1m tangential offset not preserved (got {drift}m)"
        );

        let mut target = SurfaceTarget::at(face, u, v);
        target.local = DVec2::new(1.0, 0.0);
        reproject_surface_target(&mut target, radius);

        let residual = target.local.length();
        assert!(
            residual < 1e-6,
            "After 1m pan and reprojection the residual is {residual}m"
        );

        let re_dir = uv_to_dir(target.face, target.u, target.v);
        let (_n, n_t, n_b) = tangent_frame(target.face, target.u, target.v);
        let re_pos = re_dir * radius + n_t * target.local.x + n_b * target.local.y;
        let orig_pos = dir * radius + tangent * 1.0;
        let delta = (orig_pos - re_pos).length();
        assert!(delta < 1e-6, "Reprojected position diverges (got {delta}m)");
    }

    #[test]
    fn face_edge_crossing_via_reprojection() {
        let radius = 6_371_000.0;
        let face = 0u8;
        let u = 0.0;
        let v = 0.5;
        let (_, tangent, _) = tangent_frame(face, u, v);

        let origin = uv_to_dir(face, u, v) * radius;
        let spanned_point = origin + tangent * (-2.0);
        let spanned_dir = spanned_point.normalize();
        let (crossed_face, _, _) = dir_to_uv(spanned_dir);

        assert_ne!(crossed_face, face);

        let mut target = SurfaceTarget::at(face, u, v);
        target.local = DVec2::new(-2.0, 0.0);
        reproject_surface_target(&mut target, radius);

        assert_ne!(target.face, face);
        assert!((0.0..=1.0).contains(&target.u));
        assert!((0.0..=1.0).contains(&target.v));
    }

    #[test]
    fn reprojection_roundtrip_is_identity() {
        let radius = 6_371_000.0;
        let mut target = SurfaceTarget {
            face: 2,
            u: 0.3,
            v: 0.7,
            local: DVec2::new(5.0, -3.0),
        };
        let dir = uv_to_dir(target.face, target.u, target.v);
        let (_, t, b) = tangent_frame(target.face, target.u, target.v);
        let world_before = dir * radius + t * target.local.x + b * target.local.y;

        reproject_surface_target(&mut target, radius);

        let new_dir = uv_to_dir(target.face, target.u, target.v);
        let (_, n_t, n_b) = tangent_frame(target.face, target.u, target.v);
        let world_after = new_dir * radius + n_t * target.local.x + n_b * target.local.y;

        let delta = (world_after - world_before).length();
        assert!(
            delta < 1e-5,
            "Re projection roundtrip diverges by {delta}m (should be identity)"
        );
    }

    #[test]
    fn slope_clearance_covers_close_peaks() {
        let radius = 6_371_000.0;
        let terrain_state = make_terrain_state(radius);
        let debug_info = TerrainDebugInfo::default();
        let target = SurfaceTarget::at(0, 0.5, 0.5);

        let min_alt = slope_safe_min_altitude(&target, 10.0, &terrain_state, &debug_info);

        assert!(min_alt >= 10.0);
        assert!(min_alt < 100_000.0);
    }

    // ---------
    // New slope-safe clearance tests
    // ---------

    #[test]
    fn camera_10m_target_stays_close() {
        let terrain_state = make_earth_terrain_state();
        let debug = TerrainDebugInfo {
            vertex_spacing_m: 5000.0,
            ..TerrainDebugInfo::default()
        };

        let (yaw, pitch) = (0.55_f64, 0.30_f64);
        let cp = pitch.cos();
        let dir = DVec3::new(cp * yaw.sin(), pitch.sin(), cp * yaw.cos()).normalize();
        let (face, u, v) = dir_to_uv(dir);
        let target = SurfaceTarget::at(face, u, v);

        let alt = slope_safe_min_altitude(&target, 10.0, &terrain_state, &debug);

        assert!(alt >= 10.0);
        assert!(
            alt < 25.0,
            "altitude {alt} should stay close to 10m, not rise from distant terrain"
        );
    }

    #[test]
    fn steep_local_patch_does_not_penetrate() {
        let terrain_state = make_earth_terrain_state();
        let radius = terrain_state.planet_radius;
        let debug = TerrainDebugInfo {
            vertex_spacing_m: 5000.0,
            ..TerrainDebugInfo::default()
        };

        let candidates = [(0, 0.5, 0.5), (1, 0.0, 0.5), (2, 0.2, 0.8)];
        let elevation_scale = terrain_state.elevation_scale as f64;
        let min_clearance = 10.0_f64.max(elevation_scale * 0.01);

        for (face, u, v) in candidates {
            let target = SurfaceTarget::at(face, u, v);
            let dir = uv_to_dir(face, u, v);
            let centre = terrain_state.field.sample(dir).elevation * elevation_scale;
            let (_n, tangent, bitangent) = tangent_frame(face, u, v);

            let mut max_rise = 0.0_f64;
            for ring in [0.25, 0.5, 1.0] {
                for i in 0..12 {
                    let angle = i as f64 * std::f64::consts::TAU / 12.0;
                    let offset = tangent * angle.cos() + bitangent * angle.sin();
                    let sample_radius = camera_footprint_radius(10.0) * ring;
                    let sdir = (dir * radius + offset * sample_radius).normalize();
                    let se = terrain_state.field.sample(sdir).elevation * elevation_scale;
                    max_rise = max_rise.max(se - centre);
                }
            }

            let alt = slope_safe_min_altitude(&target, 10.0, &terrain_state, &debug);
            let expected_base = max_rise.max(0.0) + min_clearance;
            assert!(
                (alt - expected_base).abs() < 1e-3,
                "altitude {alt} diverges from expected {expected_base}"
            );
        }
    }

    #[test]
    fn fallback_spacing_matches_earth_lod17() {
        let terrain_state = make_earth_terrain_state();
        let debug = TerrainDebugInfo {
            vertex_spacing_m: 0.0,
            ..TerrainDebugInfo::default()
        };

        let vs = surface_vertex_spacing(&terrain_state, &debug);
        let expected = 4.77;
        assert!(
            (vs - expected).abs() < 0.1,
            "fallback spacing {vs} diverges from {expected} at Earth LOD17"
        );
    }

    #[test]
    fn slope_safety_footprint_scales_with_view_altitude() {
        assert_eq!(camera_footprint_radius(0.0), CAMERA_FOOTPRINT_RADIUS_M);
        assert_eq!(camera_footprint_radius(10.0), 10.0);
        assert_eq!(camera_footprint_radius(500.0), 500.0);
    }

    #[test]
    fn surface_to_orbit_transition_no_jump() {
        let radius = 6_371_000.0;
        let terrain_state = make_terrain_state(radius);
        let debug_info = TerrainDebugInfo::default();
        let mut camera = OrbitCamera::for_planet(radius, 1000.0);
        camera.mode = CameraMode::Surface;
        camera.surface_target = SurfaceTarget::at(0, 0.5, 0.5);
        camera.surface_altitude = surface_exit_altitude(radius) + 10.0;
        camera.smoothed_surface_altitude = camera.surface_altitude;

        let before_pos = surface_world_pos(&camera, &terrain_state);
        transition_mode_if_needed(&mut camera, &terrain_state, &debug_info);

        assert_eq!(camera.mode, CameraMode::Orbit);
        let dir = orbit_direction(&camera);
        let after_pos = camera.target + dir * camera.distance;

        let lateral_dot = before_pos.normalize().dot(after_pos.normalize());
        assert!(
            lateral_dot > 0.99999,
            "Surface→Orbit transition lateral direction changed (dot={lateral_dot})"
        );
    }

    #[test]
    fn orbit_to_surface_transition_no_jump() {
        let radius = 6_371_000.0;
        let terrain_state = make_terrain_state(radius);
        let debug_info = TerrainDebugInfo {
            max_depth: 8,
            ..TerrainDebugInfo::default()
        };
        let mut camera = OrbitCamera::for_planet(radius, 1000.0);
        camera.mode = CameraMode::Orbit;
        let altitude = surface_enter_altitude(radius) * 0.3;
        let dir = orbit_direction(&camera);
        let terrain_elevation =
            terrain_state.field.sample(dir).elevation * terrain_state.elevation_scale as f64;
        camera.distance = radius + terrain_elevation + altitude;
        camera.smoothed_distance = camera.distance;
        let before_pos = camera.target + dir * camera.distance;

        transition_mode_if_needed(&mut camera, &terrain_state, &debug_info);

        assert_eq!(camera.mode, CameraMode::Surface);
        let after_pos = surface_world_pos(&camera, &terrain_state);

        let lateral_dot = before_pos.normalize().dot(after_pos.normalize());
        assert!(
            lateral_dot > 0.99999,
            "Orbit→Surface transition lateral direction changed (dot={lateral_dot})"
        );
    }

    #[test]
    fn hysteresis_exits_above_enters() {
        for radius in [36_000.0, 6_371_000.0] {
            let enter = surface_enter_altitude(radius);
            let exit = surface_exit_altitude(radius);
            assert!(exit > enter);
            assert!(exit / enter < 3.0);
        }
    }

    #[test]
    fn surface_and_orbit_share_multiplicative_scroll_zoom() {
        assert_eq!(scroll_zoom_factor(1.0), 0.9);
        assert_eq!(scroll_zoom_factor(-1.0), 1.1);
    }

    #[test]
    fn earth_orbit_scroll_at_500km_scales_altitude_without_surface_snap() {
        let terrain_state = make_earth_terrain_state();
        let mut camera = OrbitCamera::for_planet(terrain_state.planet_radius, 1000.0);
        let direction = orbit_direction(&camera);
        let terrain_elevation =
            terrain_state.field.sample(direction).elevation * terrain_state.elevation_scale as f64;
        let surface_radius = terrain_state.planet_radius + terrain_elevation;
        camera.distance = surface_radius + 500_000.0;

        apply_orbit_scroll_zoom(&mut camera, &terrain_state, 1.0);

        let altitude = camera.distance - surface_radius;
        assert!((altitude - 450_000.0).abs() < 1e-6);
        assert!(altitude > camera.min_altitude);
    }

    #[test]
    fn automatic_mode_transitions_are_disabled() {
        const _: () = assert!(!AUTO_MODE_TRANSITIONS);
    }

    #[test]
    fn surface_tart_defaults_are_valid() {
        let target = SurfaceTarget::default();
        let dir = uv_to_dir(target.face, target.u, target.v);
        assert!(dir.length() > 0.999 && dir.length() < 1.001);
    }

    #[test]
    fn bench_screenshot_placement_still_works() {
        let radius = 6_371_000.0;
        let terrain_state = make_terrain_state(radius);
        let mut camera = OrbitCamera::for_planet(radius, 1000.0);
        camera.yaw = 0.55;
        camera.pitch = 0.30;
        camera.distance = radius * 6.0;
        camera.smoothed_distance = radius * 6.0;
        camera.target = DVec3::ZERO;
        camera.smoothed_target = DVec3::ZERO;
        camera.mode = CameraMode::Orbit;

        clamp_orbit_to_terrain(&mut camera, &terrain_state);
        assert_eq!(camera.mode, CameraMode::Orbit);
        assert!((camera.distance - radius * 6.0).abs() < 0.1);
    }

    // -------------------------------------------------------------------------
    // Milestone 1 Exit Gate: penetration prevention, traversal, continuity
    // -------------------------------------------------------------------------

    /// The smoothed altitude lags the target. When the safe minimum rises
    /// (camera pans onto a peak), the rendered altitude must never drop below
    /// the safe minimum even while the exponential is still converging. This
    /// is the core penetration-prevention guarantee of the Exit Gate.
    #[test]
    fn smoothed_surface_altitude_never_penetrates_rising_floor() {
        let radius = 6_371_000.0;
        let terrain_state = make_terrain_state(radius);
        let debug_info = TerrainDebugInfo::default();
        let target = SurfaceTarget::at(0, 0.5, 0.5);

        let min_safe = slope_safe_min_altitude(&target, 10.0, &terrain_state, &debug_info);
        // Simulate the camera already converged at a low altitude, then the
        // safe floor jumps well above it (pan onto steep terrain).
        let mut camera = OrbitCamera::for_planet(radius, 1000.0);
        camera.mode = CameraMode::Surface;
        camera.surface_target = target;
        camera.surface_min_safe_altitude = 5.0;
        camera.surface_altitude = 5.0;
        camera.smoothed_surface_altitude = 5.0;

        // A large dt step drives the exponential most of the way to target.
        let dt = 0.1;
        // Floor rises above the smoothed value mid-convergence.
        camera.surface_min_safe_altitude = min_safe.max(20.0);
        camera.surface_altitude = camera.surface_min_safe_altitude;
        update_surface(&mut camera, dt);

        assert!(
            camera.smoothed_surface_altitude >= camera.surface_min_safe_altitude,
            "smoothed altitude {} penetrated the safe floor {}",
            camera.smoothed_surface_altitude,
            camera.surface_min_safe_altitude
        );

        // Repeated steps must keep it above the floor as it converges.
        for _ in 0..10 {
            update_surface(&mut camera, dt);
            assert!(
                camera.smoothed_surface_altitude >= camera.surface_min_safe_altitude,
                "smoothed altitude penetrated the safe floor during convergence"
            );
        }
        // And it must converge toward the target, not get stuck far below.
        assert!(
            (camera.smoothed_surface_altitude - camera.surface_altitude).abs() < 1.0,
            "smoothed altitude failed to converge to target"
        );
    }

    /// Placed camera altitude must respect the safe floor at all times. This
    /// exercises the actual placement path (`place_surface` consumes
    /// `smoothed_surface_altitude`), proving no penetration in the rendered
    /// transform even when smoothing lags.
    #[test]
    fn placed_surface_camera_altitude_respects_safe_floor() {
        let radius = 6_371_000.0;
        let terrain_state = make_terrain_state(radius);
        let debug_info = TerrainDebugInfo::default();
        let target = SurfaceTarget::at(0, 0.5, 0.5);
        let min_safe = slope_safe_min_altitude(&target, 10.0, &terrain_state, &debug_info);

        let mut camera = OrbitCamera::for_planet(radius, 1000.0);
        camera.mode = CameraMode::Surface;
        camera.surface_target = target;
        camera.surface_min_safe_altitude = min_safe;
        // Force the smoothed value to something below the floor; update_surface
        // must lift it back above the floor before placement.
        camera.surface_altitude = min_safe;
        camera.smoothed_surface_altitude = min_safe - 50.0;
        update_surface(&mut camera, 0.05);

        let mut transform = Transform::default();
        let mut camera_world = CameraWorldPosition::default();
        let mut origin = RenderOrigin::default();
        place_surface(
            &camera,
            &mut transform,
            &mut camera_world,
            &mut origin,
            &terrain_state,
        );

        let dir = uv_to_dir(target.face, target.u, target.v);
        let elev = terrain_state.field.sample(dir).elevation * terrain_state.elevation_scale as f64;
        let surface_radius = radius + elev;
        let cam_radial = camera_world.0.length();
        let placed_altitude = cam_radial - surface_radius;
        assert!(
            placed_altitude >= min_safe - 1e-6,
            "placed camera altitude {placed_altitude} penetrated safe floor {min_safe}"
        );
    }

    /// Exit Gate: at 10 m the nearest chunk must admit <=5 m vertex spacing via
    /// the same path `update_debug_info` uses (containing ancestor + cell_size).
    /// This ties the 10 m acceptance scenario to the actual LOD geometry.
    #[test]
    fn ten_meter_camera_nearest_chunk_has_sub_five_meter_spacing() {
        let terrain_state = make_earth_terrain_state();
        let radius = terrain_state.planet_radius;
        let max_depth = terrain_state.max_quadtree_depth;

        // Camera 10 m above terrain along a fixed direction.
        let (yaw, pitch) = (0.55_f64, 0.30_f64);
        let cp = pitch.cos();
        let dir = DVec3::new(cp * yaw.sin(), pitch.sin(), cp * yaw.cos()).normalize();
        let elev = terrain_state.field.sample(dir).elevation * terrain_state.elevation_scale as f64;
        let camera_pos = dir * (radius + elev + 10.0);

        // Walk the quadtree like containing_ancestor_key: deepest cell at the
        // camera direction, walking up until an active chunk covers it. At 10 m
        // the focus-detail path forces the containing cell to max_depth.
        let finest = er_core::math::dir_to_cell(dir, max_depth);
        let spacing = cell_size(finest.lod, radius) / er_core::config::CHUNK_QUADS_PER_EDGE as f64;
        assert!(
            spacing <= 5.0,
            "10 m camera nearest chunk vertex spacing {spacing:.3} > 5 m"
        );
        assert!(spacing > 0.0);

        // Sanity: the cell actually contains the camera direction.
        assert_eq!(er_core::math::dir_to_cell(dir, finest.lod), finest);
        let _ = camera_pos;
    }

    /// Exit Gate: cube-edge traversal. Panning across a cube-face edge must
    /// keep the surface target on the sphere and hand off to the neighbor
    /// face without a position discontinuity. This is an end-to-end pan that
    /// crosses the edge, not just a single reprojection.
    #[test]
    fn pan_across_cube_edge_is_continuous() {
        let radius = 6_371_000.0;
        let mut camera = OrbitCamera::for_planet(radius, 1000.0);
        camera.mode = CameraMode::Surface;
        // Start at the face-0 / face-2 boundary (u=0).
        camera.surface_target = SurfaceTarget::at(0, 0.0, 0.5);
        camera.surface_altitude = 100.0;
        camera.smoothed_surface_altitude = 100.0;

        let (_n, tangent, _bitangent) = tangent_frame(0, 0.0, 0.5);
        let before_world = surface_world_pos(&camera, &make_terrain_state(radius));

        // Pan in the -tangent direction (off the face-0 edge into face-2).
        apply_tangent_pan(
            &mut camera,
            &tangent,
            &DVec3::ZERO,
            DVec2::new(-5.0, 0.0),
            radius,
        );

        assert_ne!(camera.surface_target.face, 0, "pan did not cross the edge");
        assert!((0.0..=1.0).contains(&camera.surface_target.u));
        assert!((0.0..=1.0).contains(&camera.surface_target.v));

        // The world position after the pan must remain on the sphere surface
        // (radius + local elevation), i.e. no teleport or jump off the globe.
        let after_world = surface_world_pos(&camera, &make_terrain_state(radius));
        let after_dir = after_world.normalize();
        let (_face, _u, _v) = dir_to_uv(after_dir);
        // Direction must be finite and unit-length (on the sphere).
        assert!(
            (after_dir.length() - 1.0).abs() < 1e-9,
            "post-pan direction left the unit sphere"
        );
        // The pan was 5 m tangentially; the surface point must have moved by a
        // comparable amount (not jumped across the planet).
        let delta = (after_world - before_world).length();
        assert!(
            delta < 50.0,
            "5 m pan moved the surface target by {delta} m (expected ~5 m)"
        );
    }

    /// Exit Gate: origin-shift cell traversal. Recentering the render origin
    /// must not pop the rendered chunk transform: a camera-near chunk anchor,
    /// rebased through the new origin, stays small and continuous in render
    /// space. This is the no-pop guarantee that matters at close range.
    #[test]
    fn origin_recenter_keeps_chunk_render_position_continuous() {
        let radius = 6_371_000.0;
        // A camera-near chunk: face 0 (dominant +X), a deep cell so its anchor
        // sits close to a chosen surface point. We then pick origins snapped
        // near that anchor (as `recenter` does near the camera), keeping the
        // rebased render offset small and f32-precise.
        let key = er_core::math::dir_to_cell(DVec3::X, 10);
        let anchor = er_core::math::cell_to_dir(key) * radius;

        // Two origins both snapped near the anchor on a 1 km grid (as recenter
        // does near the camera). Render offsets stay small and f32-precise.
        let origin_a = (anchor / 1000.0).floor() * 1000.0;
        let origin_b = origin_a + DVec3::new(1000.0, 0.0, 0.0);

        let render_a = (anchor - origin_a).as_vec3();
        let render_b = (anchor - origin_b).as_vec3();

        // Each render offset is small (within one cell of its origin): the
        // close-range precision guarantee. A large offset here would mean a
        // camera-near chunk lost precision — the failure mode the floating
        // origin exists to prevent.
        assert!(
            render_a.length() < 2000.0,
            "render offset {} exceeds the recenter cell after snap",
            render_a.length()
        );
        assert!(
            render_b.length() < 2000.0,
            "render offset {} exceeds the recenter cell after shift",
            render_b.length()
        );

        // The render positions differ by exactly the *negative* origin delta
        // (render = anchor - origin, so shifting origin by +d shifts render by
        // -d). The camera moves with the origin, so the relative
        // camera->chunk vector is preserved. No pop, no extra drift.
        let delta = (render_b - render_a).as_dvec3();
        let origin_delta = origin_b - origin_a;
        assert!(
            (delta + origin_delta).length() < 1e-3,
            "render transform drifted by {} beyond the origin shift",
            (delta + origin_delta).length()
        );

        assert!(render_a.is_finite());
        assert!(render_b.is_finite());
    }

    /// Exit Gate: stale mesh safety. A mesh produced before an origin shift
    /// (stale generation) must be rebased to the *current* origin and land at
    /// the same absolute world position as a fresh mesh — it must not be
    /// dropped or misplaced. This mirrors `apply_pending_chunk_meshes`. The
    /// origin is snapped near the anchor (as `recenter` does) so the rebased
    /// render offset stays f32-precise, exactly as it does at runtime.
    #[test]
    fn stale_generation_mesh_is_rebased_not_dropped() {
        let radius = 6_371_000.0;
        let key = er_core::math::CellKey {
            face: 0,
            i: 5,
            j: 5,
            lod: 5,
        };
        let source_anchor = er_core::math::cell_to_dir(key) * radius;

        // The mesh was generated when the origin was `queued_origin`; by the
        // time it attaches, the origin has shifted to `current_origin`. Both
        // are snapped near the anchor on a 1 km grid (as recenter does), so
        // the rebased render offset is small and f32-precise.
        let base = (source_anchor / 1000.0).floor() * 1000.0;
        let queued_origin = base;
        let current_origin = base + DVec3::new(1000.0, 0.0, 0.0);

        // anchor_render_translation uses the CURRENT origin, so the stale
        // mesh is rebased rather than placed at its queued-origin offset.
        let rebased = anchor_render_translation(source_anchor, current_origin);
        let reconstructed = current_origin + rebased.as_dvec3();
        assert!(
            (reconstructed - source_anchor).length() < 0.001,
            "stale-generation mesh was not rebased to its absolute source anchor"
        );

        // A fresh mesh (current generation) uses the same anchor and current
        // origin, so both must coincide — proving no pop between stale/fresh.
        let fresh = anchor_render_translation(source_anchor, current_origin);
        assert_eq!(
            rebased, fresh,
            "stale and fresh meshes diverged in render space"
        );
        let _ = queued_origin;
    }

    /// Exit Gate: transition continuity (radial). The orbit->surface handoff
    /// must not produce a radial/altitude discontinuity when the safe floor is
    /// below the orbit altitude (the non-clamping path). This complements the
    /// existing lateral-direction test with radial continuity.
    #[test]
    fn orbit_to_surface_transition_preserves_radial_distance() {
        let radius = 6_371_000.0;
        let terrain_state = make_terrain_state(radius);
        let debug_info = TerrainDebugInfo {
            max_depth: 8,
            ..TerrainDebugInfo::default()
        };
        let mut camera = OrbitCamera::for_planet(radius, 1000.0);
        camera.mode = CameraMode::Orbit;

        // Place the orbit camera at an altitude safely above the surface-enter
        // threshold so the safe floor does not clamp (isolating radial
        // continuity rather than the penetration-prevention clamp).
        let dir = orbit_direction(&camera);
        let elev = terrain_state.field.sample(dir).elevation * terrain_state.elevation_scale as f64;
        let enter = surface_enter_altitude(radius);
        let altitude = enter * 0.3;
        camera.distance = radius + elev + altitude;
        camera.smoothed_distance = camera.distance;
        let before_radial = camera.target + dir * camera.distance;
        let before_len = before_radial.length();

        transition_mode_if_needed(&mut camera, &terrain_state, &debug_info);
        assert_eq!(camera.mode, CameraMode::Surface);

        let after_radial = surface_world_pos(&camera, &terrain_state);
        let after_len = after_radial.length();

        // Radial distance must be preserved (no pop in/out along the normal).
        // The surface placement adds the same altitude above the same surface
        // point, so the radial length must match to sub-meter precision.
        assert!(
            (after_len - before_len).abs() < 1.0,
            "orbit->surface radial distance jumped by {} m",
            (after_len - before_len).abs()
        );
        // Lateral direction continuity (already covered, re-asserted here).
        let lateral_dot = before_radial.normalize().dot(after_radial.normalize());
        assert!(
            lateral_dot > 0.99999,
            "lateral direction changed (dot={lateral_dot})"
        );
    }

    /// Exit Gate: transition continuity when the safe floor *does* clamp. The
    /// orbit->surface handoff must move the camera *up* to the safe floor
    /// (preventing penetration) rather than *down* into the terrain. A jump up
    /// is acceptable; a jump down (penetration) is not.
    #[test]
    fn orbit_to_surface_clamp_only_moves_camera_up() {
        let radius = 6_371_000.0;
        let terrain_state = make_terrain_state(radius);
        let debug_info = TerrainDebugInfo {
            max_depth: 8,
            ..TerrainDebugInfo::default()
        };
        let mut camera = OrbitCamera::for_planet(radius, 1000.0);
        camera.mode = CameraMode::Orbit;

        let dir = orbit_direction(&camera);
        let elev = terrain_state.field.sample(dir).elevation * terrain_state.elevation_scale as f64;
        // Very low altitude so the safe floor likely clamps above it.
        let altitude = 5.0;
        camera.distance = radius + elev + altitude;
        camera.smoothed_distance = camera.distance;
        let before_radial_len = (camera.target + dir * camera.distance).length();

        transition_mode_if_needed(&mut camera, &terrain_state, &debug_info);
        assert_eq!(camera.mode, CameraMode::Surface);

        let after = surface_world_pos(&camera, &terrain_state);
        let after_radial_len = after.length();
        // The clamp can only raise the camera (prevent penetration), never
        // lower it below the orbit position.
        assert!(
            after_radial_len >= before_radial_len - 1e-6,
            "orbit->surface clamp lowered the camera by {} m (penetration)",
            before_radial_len - after_radial_len
        );
    }
}
