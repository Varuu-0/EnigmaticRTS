use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use er_core::math::{
    cell_size, dir_to_uv, needs_recenter, recenter, tangent_frame, uv_to_dir, OriginOffset,
    WorldPos,
};
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
}

impl OrbitCamera {
    pub fn for_planet(radius: f64, elevation_scale: f32) -> Self {
        let min_altitude = 10.0_f64.max(elevation_scale.abs() as f64 * 0.01);
        let max_altitude = radius * 29.0;
        let distance = radius * 2.5;
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

const MINIATURE_SURFACE_ENTER_ALTITUDE: f64 = 150.0;
const MINIATURE_SURFACE_EXIT_ALTITUDE: f64 = 600.0;
const EARTH_SURFACE_ENTER_ALTITUDE: f64 = 1000.0;
const EARTH_SURFACE_EXIT_ALTITUDE: f64 = 5000.0;

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

// ---------------------------------------------------------------------------
// Main update
// ---------------------------------------------------------------------------

use crate::space::{StarfieldComponent, SunLight, SunSphere};

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
        transition_mode_if_needed(&mut camera, &terrain_state, &debug_info);

        match camera.mode {
            CameraMode::Orbit => {
                update_orbit(&mut camera, &terrain_state, dt);
                place_orbit(&camera, &mut transform, &mut camera_world, &mut origin);
            }
            CameraMode::Surface => {
                update_surface(&mut camera, &terrain_state, &debug_info, dt);
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

    let scroll_factor = 1.0 - accumulated_scroll.delta.y as f64 * 0.1;
    camera.distance *= scroll_factor;

    if keys.pressed(KeyCode::ArrowUp) {
        camera.distance -= zoom_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        camera.distance += zoom_speed * dt;
    }

    clamp_orbit_to_terrain(camera, terrain_state);
}

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
    let scroll_delta = accumulated_scroll.delta.y as f64;
    if scroll_delta.abs() > 0.0 {
        camera.surface_altitude -= altitude_speed * scroll_delta * 0.05;
    }

    let min_safe = slope_safe_min_altitude(
        &camera.surface_target,
        camera.surface_altitude,
        terrain_state,
        debug_info,
    );
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

fn update_surface(
    camera: &mut OrbitCamera,
    terrain_state: &TerrainState,
    debug_info: &TerrainDebugInfo,
    dt: f64,
) {
    let min_safe = slope_safe_min_altitude(
        &camera.surface_target,
        camera.surface_altitude,
        terrain_state,
        debug_info,
    );
    camera.surface_altitude = camera.surface_altitude.max(min_safe);
    let alpha = 1.0 - (-camera.smoothing * dt).exp();
    camera.smoothed_surface_altitude = camera.smoothed_surface_altitude
        + (camera.surface_altitude - camera.smoothed_surface_altitude) * alpha;
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

            if altitude < enter {
                if world_camera.length() > 1.0 {
                    let (face, u, v) = dir_to_uv(world_camera);
                    camera.mode = CameraMode::Surface;
                    camera.surface_target =
                        SurfaceTarget::at(face, u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
                    camera.surface_altitude = altitude.max(slope_safe_min_altitude(
                        &camera.surface_target,
                        altitude,
                        terrain_state,
                        debug_info,
                    ));
                    camera.smoothed_surface_altitude = camera.surface_altitude;
                }
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

    let sample = terrain_state.field.sample(dir);
    let centre_elevation = sample.elevation * elevation_scale;
    let min_clearance = 10.0_f64.max(elevation_scale * 0.01);

    let vertex_spacing = surface_vertex_spacing(terrain_state, debug_info);
    let sample_radius = camera_footprint_radius(desired_altitude)
        .max(vertex_spacing.max(0.001).min(CAMERA_FOOTPRINT_RADIUS_M));
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
            let sample_elev = terrain_state.field.sample(sample_dir).elevation * elevation_scale;
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
        let base_elev = terrain_state.field.sample(uv_to_dir(0, 0.5, 0.5)).elevation * 1000.0;

        assert!((min_alt - base_elev) >= 0.0);
        assert!((min_alt - base_elev) < 100_000.0);
    }

    // ---------
    // New slope-safe clearance tests
    // ---------

    #[test]
    fn camera_10m_target_stays_close() {
        let terrain_state = make_earth_terrain_state();
        let mut debug = TerrainDebugInfo::default();
        debug.vertex_spacing_m = 5000.0; // coarse LOD should not inflate clearance

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
        let mut debug = TerrainDebugInfo::default();
        debug.vertex_spacing_m = 5000.0; // simulate coarse pre-LOD load

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
        let mut debug = TerrainDebugInfo::default();
        debug.vertex_spacing_m = 0.0; // fallback active

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
        let mut debug_info = TerrainDebugInfo::default();
        debug_info.max_depth = 8; // fine enough that safe_altitude < enter
        let mut camera = OrbitCamera::for_planet(radius, 1000.0);
        camera.mode = CameraMode::Orbit;
        let altitude = surface_enter_altitude(radius) * 0.3;
        camera.distance = radius + altitude;
        camera.smoothed_distance = camera.distance;
        let dir = orbit_direction(&camera);
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
        let radius = 6_371_000.0;
        let enter = surface_enter_altitude(radius);
        let exit = surface_exit_altitude(radius);
        assert!(exit > enter);
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
}
