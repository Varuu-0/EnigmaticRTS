use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
use er_terrain::{
    ocean::OceanComponent, PendingChunkMeshes, TerrainDebugInfo, TerrainMaterial, TerrainUpdate,
};
use std::path::PathBuf;

use crate::space::{CloudComponent, SimTime, StarfieldComponent, SunSphere, TimeScale};

/// Consecutive frames with no pending chunk meshes required before the scene is
/// considered settled enough for a screenshot. This also gives extracted meshes
/// several render frames to become drawable before the capture request.
const SETTLED_FRAMES_THRESHOLD: u32 = 5;

/// Hard cap: never wait more than this many frames for a scenario, even if the
/// terrain is still churning. At 60 fps this is ~10 s per scenario.
const MAX_FRAMES_PER_SCENARIO: u32 = 600;

#[derive(Resource, Default)]
pub struct ScreenshotTestConfig {
    pub output_dir: PathBuf,
    pub scenarios: Vec<ScreenshotScenario>,
    pub current_index: usize,
    /// Minimum frames to wait after moving the camera before stabilization can
    /// be declared. This is kept low because the real gate is the pending-mesh
    /// count.
    pub frames_to_wait: u32,
    pub frames_waited: u32,
    pub completed: bool,
    pub pending_screenshots: u32,
    /// Consecutive frames the terrain has reported no pending meshes.
    pub settled_frames: u32,
    /// Hide non-terrain layers to isolate terrain silhouette diagnostics.
    pub terrain_only: bool,
    /// Highlight skirts magenta in terrain material diagnostics.
    pub debug_skirts: bool,
    /// Freeze the solar simulation at this time for reproducible captures.
    pub fixed_sim_time: Option<f32>,
}

#[derive(Clone)]
pub struct ScreenshotScenario {
    pub name: String,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_distance: f32,
    pub camera_target: Vec3,
}

pub struct ScreenshotTestPlugin;

impl Plugin for ScreenshotTestPlugin {
    fn build(&self, app: &mut App) {
        // Run after terrain systems so pending_meshes reflects the latest work.
        app.add_systems(
            Update,
            (
                apply_screenshot_diagnostics,
                run_screenshot_test
                    .after(crate::camera::CameraUpdate)
                    .after(TerrainUpdate),
            ),
        );
    }
}

fn apply_screenshot_diagnostics(
    config: Res<ScreenshotTestConfig>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut applied_debug_skirts: Local<Option<bool>>,
    mut sim_time: ResMut<SimTime>,
    mut time_scale: ResMut<TimeScale>,
    mut non_terrain_layers: Query<
        &mut Visibility,
        Or<(
            With<OceanComponent>,
            With<CloudComponent>,
            With<StarfieldComponent>,
            With<SunSphere>,
        )>,
    >,
) {
    if let Some(fixed_time) = config.fixed_sim_time {
        sim_time.0 = fixed_time;
        time_scale.current = 0.0;
        time_scale.resume = 0.0;
    }

    if config.terrain_only {
        for mut visibility in &mut non_terrain_layers {
            *visibility = Visibility::Hidden;
        }
    }

    if *applied_debug_skirts != Some(config.debug_skirts) {
        let highlight = if config.debug_skirts { 1.0 } else { 0.0 };
        for (_, material) in materials.iter_mut() {
            material.uniform.debug_skirt_highlight = highlight;
        }
        *applied_debug_skirts = Some(config.debug_skirts);
    }
}

fn run_screenshot_test(
    mut config: ResMut<ScreenshotTestConfig>,
    mut camera_query: Query<(&mut crate::camera::OrbitCamera, &mut Transform), With<Camera3d>>,
    mut commands: Commands,
    capturing: Query<Entity, With<Capturing>>,
    pending_meshes: Res<PendingChunkMeshes>,
    debug_info: Res<TerrainDebugInfo>,
    mut exit: MessageWriter<AppExit>,
) {
    // Check if any screenshots are still being captured
    let capturing_count = capturing.iter().count() as u32;
    if capturing_count != config.pending_screenshots {
        config.pending_screenshots = capturing_count;
    }

    // If all scenarios are done and no pending screenshots, exit
    if config.completed {
        if config.pending_screenshots == 0 {
            info!("All screenshots saved, exiting.");
            exit.write(AppExit::Success);
        }
        return;
    }

    if config.scenarios.is_empty() {
        config.completed = true;
        return;
    }

    if config.current_index >= config.scenarios.len() {
        info!(
            "Screenshot test completed: {} scenarios captured",
            config.scenarios.len()
        );
        config.completed = true;
        return;
    }

    let scenario = config.scenarios[config.current_index].clone();

    if config.frames_waited == 0 {
        info!(
            "Capturing scenario: {} (index {})",
            scenario.name, config.current_index
        );

        if let Ok((mut orbit, mut transform)) = camera_query.single_mut() {
            orbit.yaw = scenario.camera_yaw;
            orbit.pitch = scenario.camera_pitch;
            orbit.distance = scenario.camera_distance;
            orbit.target = scenario.camera_target;
            orbit.smoothed_distance = scenario.camera_distance;
            orbit.smoothed_target = scenario.camera_target;

            let cp = orbit.pitch.cos();
            let direction = Vec3::new(
                cp * orbit.yaw.sin(),
                orbit.pitch.sin(),
                cp * orbit.yaw.cos(),
            )
            .normalize();

            transform.translation = orbit.target + direction * orbit.distance;
            let up = if direction.abs().dot(Vec3::Y) > 0.999 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            transform.look_at(orbit.target, up);
        }
    }

    config.frames_waited += 1;

    // CPU completion alone is insufficient: meshes must have been visible for
    // several render frames so they are extracted, uploaded, and rasterized.
    let terrain_settled = pending_meshes.0.is_empty()
        && debug_info.pending_splits == 0
        && debug_info.pending_merges == 0
        && debug_info.visible_chunks > 0;
    if terrain_settled {
        config.settled_frames += 1;
    } else {
        config.settled_frames = 0;
        if config.frames_waited % 30 == 0 {
            info!(
                "Scenario {} waiting for terrain: {} pending meshes, {} active chunks, {} visible chunks",
                scenario.name,
                debug_info.pending_meshes,
                debug_info.active_chunks,
                debug_info.visible_chunks,
            );
        }
    }

    let ready_to_capture = config.frames_waited >= config.frames_to_wait
        && config.settled_frames >= SETTLED_FRAMES_THRESHOLD;
    let timed_out = config.frames_waited >= MAX_FRAMES_PER_SCENARIO;

    if ready_to_capture || timed_out {
        if timed_out && !ready_to_capture {
            warn!(
                "Scenario {} timed out after {} frames with {} pending meshes still in flight",
                scenario.name, config.frames_waited, debug_info.pending_meshes,
            );
        } else {
            info!(
                "Scenario {} settled after {} frames ({} active chunks)",
                scenario.name, config.frames_waited, debug_info.active_chunks,
            );
        }

        let filename = format!("{}.png", scenario.name);
        let path = config.output_dir.join(&filename);

        info!("Taking screenshot: {:?}", path);

        std::fs::create_dir_all(&config.output_dir).ok();

        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));

        config.current_index += 1;
        config.frames_waited = 0;
        config.settled_frames = 0;
        config.pending_screenshots += 1;
    }
}

pub fn parse_test_args() -> Option<ScreenshotTestConfig> {
    let args: Vec<String> = std::env::args().collect();

    let mut output_dir = PathBuf::from("screenshots");
    let mut scenarios = Vec::new();
    // Minimum frames before stabilization is allowed; the real gate is the
    // pending mesh count.
    let mut frames_to_wait = 5;
    let mut terrain_only = false;
    let mut debug_skirts = false;
    let mut fixed_sim_time = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--screenshot-test" => {
                i += 1;
                if i < args.len() {
                    output_dir = PathBuf::from(&args[i]);
                }
            }
            "--frames" => {
                i += 1;
                if i < args.len() {
                    frames_to_wait = args[i].parse().unwrap_or(5);
                }
            }
            "--terrain-only" => terrain_only = true,
            "--debug-skirts" => debug_skirts = true,
            "--sim-time" => {
                i += 1;
                if i < args.len() {
                    fixed_sim_time = args[i].parse().ok();
                }
            }
            "--scenario" => {
                i += 1;
                if i < args.len() {
                    let parts: Vec<&str> = args[i].split(',').collect();
                    if parts.len() >= 4 {
                        scenarios.push(ScreenshotScenario {
                            name: parts[0].to_string(),
                            camera_yaw: parts[1].parse().unwrap_or(0.0),
                            camera_pitch: parts[2].parse().unwrap_or(0.3),
                            camera_distance: parts[3].parse().unwrap_or(90000.0),
                            camera_target: Vec3::ZERO,
                        });
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    if scenarios.is_empty() {
        return None;
    }

    Some(ScreenshotTestConfig {
        output_dir,
        scenarios,
        current_index: 0,
        frames_to_wait,
        frames_waited: 0,
        completed: false,
        pending_screenshots: 0,
        settled_frames: 0,
        terrain_only,
        debug_skirts,
        fixed_sim_time,
    })
}
