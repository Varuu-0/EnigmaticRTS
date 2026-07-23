use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
use er_terrain::{
    ocean::OceanComponent, CameraWorldPosition, FrameProfiler, PendingChunkMeshes,
    QueuedChunkMeshes, RenderOrigin, TerrainDebugInfo, TerrainMaterial, TerrainState,
    TerrainUpdate,
};
use glam::DVec3;
use std::{path::PathBuf, time::Instant};

use crate::space::{CloudComponent, SimTime, StarfieldComponent, SunSphere, TimeScale};
use crate::{diagnostics::PerformanceSnapshot, telemetry};

/// Consecutive frames with no pending chunk meshes required before the scene is
/// considered settled enough for a screenshot. This also gives extracted meshes
/// several render frames to become drawable before the capture request.
const SETTLED_FRAMES_THRESHOLD: u32 = 5;

/// Minimum frame budget before the wall-clock timeout may end a scenario.
/// Requiring both budgets gives background mesh workers real time to drain
/// without letting low-FPS runs fail after only a handful of rendered frames.
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
    /// True when an acceptance capture timed out. The run exits non-zero once
    /// pending screenshot writes finish unless explicitly marked exploratory.
    pub acceptance_failed: bool,
    pub pending_screenshots: u32,
    /// Consecutive frames the terrain has reported no pending meshes.
    pub settled_frames: u32,
    /// Hide non-terrain layers to isolate terrain silhouette diagnostics.
    pub terrain_only: bool,
    /// Highlight skirts magenta in terrain material diagnostics.
    pub debug_skirts: bool,
    /// Freeze the solar simulation at this time for reproducible captures.
    pub fixed_sim_time: Option<f32>,
    /// Seed passed to the terrain generator and recorded in every sidecar.
    pub fixed_seed: u64,
    /// Exploratory captures retain timeout output but do not fail the process.
    pub exploratory: bool,
    /// Wall-clock timeout complements the frame floor so high-FPS runs still
    /// give background mesh workers enough real time to drain.
    pub max_wait_seconds: f64,
    pub scenario_started_at: Option<Instant>,
}

#[derive(Clone)]
pub struct ScreenshotScenario {
    pub name: String,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_distance: f32,
    pub camera_target: Vec3,
    pub target_altitude_m: Option<f32>,
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

#[allow(clippy::type_complexity)]
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

#[allow(clippy::too_many_arguments)]
fn run_screenshot_test(
    mut config: ResMut<ScreenshotTestConfig>,
    mut camera_query: Query<(&mut crate::camera::OrbitCamera, &mut Transform), With<Camera3d>>,
    camera_world: Res<CameraWorldPosition>,
    origin: Res<RenderOrigin>,
    mut commands: Commands,
    capturing: Query<Entity, With<Capturing>>,
    pending_meshes: Res<PendingChunkMeshes>,
    queued_meshes: Res<QueuedChunkMeshes>,
    debug_info: Res<TerrainDebugInfo>,
    terrain_state: Res<TerrainState>,
    performance: Res<PerformanceSnapshot>,
    profiler: Res<FrameProfiler>,
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
            if config.acceptance_failed {
                error!("Acceptance screenshot run failed because one or more scenarios timed out.");
                exit.write(AppExit::error());
            } else {
                info!("All screenshots saved, exiting.");
                exit.write(AppExit::Success);
            }
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
        config.scenario_started_at = Some(Instant::now());
        info!(
            "Capturing scenario: {} (index {})",
            scenario.name, config.current_index
        );

        if let Ok((mut orbit, _transform)) = camera_query.single_mut() {
            orbit.yaw = scenario.camera_yaw as f64;
            orbit.pitch = scenario.camera_pitch as f64;
            if let Some(target_alt) = scenario.target_altitude_m {
                let direction = DVec3::new(
                    orbit.pitch.cos() * orbit.yaw.sin(),
                    orbit.pitch.sin(),
                    orbit.pitch.cos() * orbit.yaw.cos(),
                )
                .normalize();
                let (face, u, v) = er_core::math::dir_to_uv(direction);
                orbit.mode = crate::camera::CameraMode::Surface;
                orbit.surface_target = crate::camera::SurfaceTarget::at(face, u, v);
                orbit.surface_altitude = target_alt as f64;
                orbit.smoothed_surface_altitude = target_alt as f64;
            } else {
                orbit.mode = crate::camera::CameraMode::Orbit;
                orbit.distance = scenario.camera_distance as f64;
                orbit.smoothed_distance = scenario.camera_distance as f64;
            }
            orbit.target = scenario.camera_target.as_dvec3();
            orbit.smoothed_target = scenario.camera_target.as_dvec3();
        }
    }

    if config.frames_waited > 0 && scenario.target_altitude_m.is_some() {
        if let Ok((orbit, mut transform)) = camera_query.single_mut() {
            let dir = er_core::math::uv_to_dir(
                orbit.surface_target.face,
                orbit.surface_target.u,
                orbit.surface_target.v,
            );
            let (_, tangent, _) = er_core::math::tangent_frame(
                orbit.surface_target.face,
                orbit.surface_target.u,
                orbit.surface_target.v,
            );
            let elevation =
                terrain_state.field.sample(dir).elevation * terrain_state.elevation_scale as f64;
            let surface = dir * (terrain_state.planet_radius + elevation);
            if scenario.name == "coastline" {
                let render_target = (surface - origin.world).as_vec3();
                transform.look_at(render_target, tangent.as_vec3());
            } else {
                let look_ahead_m =
                    (scenario.target_altitude_m.unwrap_or(10.0) * 2.0).max(50.0) as f64;
                let render_target = (surface + tangent * look_ahead_m - origin.world).as_vec3();
                transform.look_at(render_target, dir.as_vec3());
            }
        }
    }

    config.frames_waited += 1;

    // CPU completion alone is insufficient: meshes must have been visible for
    // several render frames so they are extracted, uploaded, and rasterized.
    let terrain_settled = pending_meshes.0.is_empty()
        && queued_meshes.is_empty()
        && debug_info.pending_splits == 0
        && debug_info.pending_merges == 0
        && debug_info.visible_chunks > 0
        && detail_target_met(&scenario, &debug_info)
        && altitude_target_met(&scenario, &debug_info);
    if terrain_settled {
        config.settled_frames += 1;
    } else {
        config.settled_frames = 0;
        if config.frames_waited.is_multiple_of(30) {
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
    let elapsed_seconds = config
        .scenario_started_at
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let timed_out = scenario_timed_out(
        config.frames_waited,
        elapsed_seconds,
        config.max_wait_seconds,
    );

    if ready_to_capture || timed_out {
        if timed_out && !ready_to_capture {
            warn!(
                "Scenario {} timed out after {} frames with {} pending meshes still in flight",
                scenario.name, config.frames_waited, debug_info.pending_meshes,
            );
            if acceptance_timeout_fails(config.exploratory) {
                config.acceptance_failed = true;
            }
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

        let telemetry_path = config
            .output_dir
            .join(format!("{}.telemetry.json", scenario.name));
        if let Err(error) = telemetry::write_scenario_telemetry(
            &telemetry_path,
            telemetry::ScenarioTelemetry::capture(
                &scenario,
                &config,
                timed_out && !ready_to_capture,
                &terrain_state,
                &debug_info,
                &camera_world,
                &origin,
                &performance,
                &profiler,
            ),
        ) {
            error!(path = ?telemetry_path, %error, "Could not write scenario telemetry");
        }

        config.current_index += 1;
        config.frames_waited = 0;
        config.settled_frames = 0;
        config.scenario_started_at = None;
        config.pending_screenshots += 1;
    }
}

pub fn parse_test_args(planet_radius: f64) -> Option<ScreenshotTestConfig> {
    let args: Vec<String> = std::env::args().collect();

    let mut output_dir = PathBuf::from("screenshots");
    let mut scenarios = Vec::new();
    // Minimum frames before stabilization is allowed; the real gate is the
    // pending mesh count.
    let mut frames_to_wait = 5;
    let mut terrain_only = false;
    let mut debug_skirts = false;
    let mut fixed_sim_time = None;
    let mut fixed_coverage = false;
    let mut fixed_seed = 0xC0FFEE;
    let mut exploratory = false;
    let mut max_wait_seconds = if planet_radius >= 1_000_000.0 {
        30.0
    } else {
        10.0
    };

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
            "--seed" => {
                i += 1;
                if i < args.len() {
                    fixed_seed = crate::cli::parse_seed_value(&args[i]).unwrap_or(fixed_seed);
                }
            }
            "--fixed-seed-coverage" => fixed_coverage = true,
            "--exploratory" => exploratory = true,
            "--timeout-seconds" => {
                i += 1;
                if i < args.len() {
                    max_wait_seconds = args[i].parse().unwrap_or(max_wait_seconds);
                }
            }
            "--altitude-scenario" => {
                i += 1;
                if i < args.len() {
                    let parts: Vec<&str> = args[i].split(',').collect();
                    if parts.len() >= 4 {
                        scenarios.push(ScreenshotScenario {
                            name: parts[0].to_string(),
                            camera_yaw: parts[1].parse().unwrap_or(0.0),
                            camera_pitch: parts[2].parse().unwrap_or(0.3),
                            camera_distance: planet_radius as f32,
                            camera_target: Vec3::ZERO,
                            target_altitude_m: parts[3].parse().ok(),
                        });
                    }
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
                            target_altitude_m: None,
                        });
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    if fixed_coverage && scenarios.is_empty() {
        scenarios = fixed_coverage_scenarios(planet_radius);
        terrain_only = true;
        fixed_sim_time = Some(0.0);
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
        acceptance_failed: false,
        pending_screenshots: 0,
        settled_frames: 0,
        terrain_only,
        debug_skirts,
        fixed_sim_time,
        fixed_seed,
        exploratory,
        max_wait_seconds,
        scenario_started_at: None,
    })
}

/// Fixed-coverage scenarios for Earth-scale planets. Altitude targets are
/// chosen to cover the full visible range: close-range surface (10 m, 100 m),
/// mid-range ground (1 km, 10 km), and orbital overview. Two pairs of
/// boundary scenarios exercise camera stability at cube-face edges and
/// across recenter events.
#[allow(clippy::vec_init_then_push)]
fn fixed_coverage_scenarios(radius: f64) -> Vec<ScreenshotScenario> {
    let mut scenarios = Vec::new();

    scenarios.push(ScreenshotScenario {
        name: "altitude_10m".into(),
        camera_yaw: 1.8,
        camera_pitch: 0.8,
        camera_distance: radius as f32,
        camera_target: Vec3::ZERO,
        target_altitude_m: Some(10.0),
    });
    scenarios.push(ScreenshotScenario {
        name: "altitude_100m".into(),
        camera_yaw: 0.55,
        camera_pitch: 0.30,
        camera_distance: radius as f32,
        camera_target: Vec3::ZERO,
        target_altitude_m: Some(100.0),
    });
    scenarios.push(ScreenshotScenario {
        name: "altitude_1km".into(),
        camera_yaw: 0.55,
        camera_pitch: 0.30,
        camera_distance: radius as f32,
        camera_target: Vec3::ZERO,
        target_altitude_m: Some(1000.0),
    });
    scenarios.push(ScreenshotScenario {
        name: "altitude_10km".into(),
        camera_yaw: 0.55,
        camera_pitch: 0.30,
        camera_distance: radius as f32,
        camera_target: Vec3::ZERO,
        target_altitude_m: Some(10000.0),
    });
    scenarios.push(ScreenshotScenario {
        name: "orbit".into(),
        camera_yaw: 0.0,
        camera_pitch: 0.30,
        camera_distance: (radius * 2.2) as f32,
        camera_target: Vec3::ZERO,
        target_altitude_m: None,
    });
    scenarios.extend([
        ScreenshotScenario {
            name: "globe".into(),
            camera_yaw: 0.55,
            camera_pitch: 0.30,
            camera_distance: (radius * 6.0) as f32,
            camera_target: Vec3::ZERO,
            target_altitude_m: None,
        },
        ScreenshotScenario {
            name: "surface".into(),
            camera_yaw: 1.10,
            camera_pitch: 0.08,
            camera_distance: radius as f32,
            camera_target: Vec3::ZERO,
            target_altitude_m: Some(100.0),
        },
        ScreenshotScenario {
            name: "coastline".into(),
            camera_yaw: 5.218_547_3,
            camera_pitch: -0.331_653_15,
            camera_distance: radius as f32,
            camera_target: Vec3::ZERO,
            target_altitude_m: Some(2_000.0),
        },
        ScreenshotScenario {
            name: "mountain".into(),
            camera_yaw: 2.68,
            camera_pitch: 0.76,
            camera_distance: radius as f32,
            camera_target: Vec3::ZERO,
            target_altitude_m: Some(10_000.0),
        },
        ScreenshotScenario {
            name: "cube_corner".into(),
            camera_yaw: std::f32::consts::FRAC_PI_4,
            camera_pitch: 0.615_479_7,
            camera_distance: (radius * 1.3) as f32,
            camera_target: Vec3::ZERO,
            target_altitude_m: None,
        },
    ]);
    // Cube-edge before/after: look near a face boundary, then just after it.
    scenarios.push(ScreenshotScenario {
        name: "cube_edge_before".into(),
        camera_yaw: std::f32::consts::FRAC_PI_4 - 0.01,
        camera_pitch: 0.0,
        camera_distance: (radius * 1.2) as f32,
        camera_target: Vec3::ZERO,
        target_altitude_m: None,
    });
    scenarios.push(ScreenshotScenario {
        name: "cube_edge_after".into(),
        camera_yaw: std::f32::consts::FRAC_PI_4 + 0.01,
        camera_pitch: 0.0,
        camera_distance: (radius * 1.2) as f32,
        camera_target: Vec3::ZERO,
        target_altitude_m: None,
    });
    // Origin-shift before/after: move about 1.2 km along the same latitude so
    // the view remains geographically comparable while crossing a 1 km cell.
    scenarios.push(ScreenshotScenario {
        name: "origin_shift_before".into(),
        camera_yaw: 1.8,
        camera_pitch: 0.8,
        camera_distance: radius as f32,
        camera_target: Vec3::ZERO,
        target_altitude_m: Some(10_000.0),
    });
    scenarios.push(ScreenshotScenario {
        name: "origin_shift_after".into(),
        camera_yaw: 1.8003,
        camera_pitch: 0.8,
        camera_distance: radius as f32,
        camera_target: Vec3::ZERO,
        target_altitude_m: Some(10_000.0),
    });
    scenarios
}

fn acceptance_timeout_fails(exploratory: bool) -> bool {
    !exploratory
}

fn scenario_timed_out(frames: u32, elapsed_seconds: f64, max_wait_seconds: f64) -> bool {
    frames >= MAX_FRAMES_PER_SCENARIO && elapsed_seconds >= max_wait_seconds
}

fn detail_target_met(scenario: &ScreenshotScenario, debug: &TerrainDebugInfo) -> bool {
    scenario.target_altitude_m.is_none_or(|altitude| {
        altitude > 10.0 || (debug.vertex_spacing_m > 0.0 && debug.vertex_spacing_m <= 5.0)
    })
}

fn altitude_target_met(scenario: &ScreenshotScenario, debug: &TerrainDebugInfo) -> bool {
    scenario.target_altitude_m.is_none_or(|requested| {
        let tolerance = if requested <= 10.0 {
            2.0
        } else {
            (requested as f64 * 0.01).max(0.5)
        };
        if scenario.name.starts_with("altitude_") {
            (debug.camera_altitude_m - requested as f64).abs() <= tolerance
        } else {
            debug.camera_altitude_m + tolerance >= requested as f64
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_meter_scenario_is_slope_safe() {
        let radius = 6_371_000.0;
        let seed = er_core::seed::PlanetSeed(0xC0FFEE);
        let climate = er_world::planet_params(seed);
        let field = er_world::ProceduralTerrainField::new_metric(
            er_world::elevation::elevation_params(seed),
            climate,
            radius,
        );
        let scenario = fixed_coverage_scenarios(radius)
            .into_iter()
            .find(|scenario| scenario.name == "altitude_10m")
            .unwrap();
        let yaw = scenario.camera_yaw as f64;
        let pitch = scenario.camera_pitch as f64;
        let cp = pitch.cos();
        let dir = DVec3::new(cp * yaw.sin(), pitch.sin(), cp * yaw.cos());
        let (face, u, v) = er_core::math::dir_to_uv(dir);
        let (_, tangent, bitangent) = er_core::math::tangent_frame(face, u, v);
        let center_sample = er_world::TerrainField::sample(&field, dir);
        let center = center_sample.elevation * 1000.0;
        let mut max_rise = 0.0_f64;

        for ring in [0.25, 0.5, 1.0] {
            for sample_index in 0..12 {
                let angle = sample_index as f64 * std::f64::consts::TAU / 12.0;
                let offset = (tangent * angle.cos() + bitangent * angle.sin()) * 10.0 * ring;
                let sample_dir = (dir * radius + offset).normalize();
                let elevation =
                    er_world::TerrainField::sample(&field, sample_dir).elevation * 1000.0;
                max_rise = max_rise.max(elevation - center);
            }
        }

        assert_eq!(scenario.target_altitude_m, Some(10.0));
        assert!(center > 0.0, "10 m gate should render over land");
        assert!(
            center_sample.low_freq_elev as f64 > climate.sea_level + 0.2,
            "10 m gate must remain unambiguously above the macro shoreline"
        );
        assert!(
            max_rise <= 2.0,
            "10 m footprint rises {max_rise:.3} m above its center"
        );
    }

    #[test]
    fn mountain_scenario_is_land_with_kilometer_scale_relief() {
        let radius = 6_371_000.0;
        let seed = er_core::seed::PlanetSeed(0xC0FFEE);
        let field = er_world::ProceduralTerrainField::new_metric(
            er_world::elevation::elevation_params(seed),
            er_world::planet_params(seed),
            radius,
        );
        let scenario = fixed_coverage_scenarios(radius)
            .into_iter()
            .find(|scenario| scenario.name == "mountain")
            .unwrap();
        let yaw = scenario.camera_yaw as f64;
        let pitch = scenario.camera_pitch as f64;
        let cp = pitch.cos();
        let center_direction = DVec3::new(cp * yaw.sin(), pitch.sin(), cp * yaw.cos());
        let center_elevation =
            er_world::TerrainField::sample(&field, center_direction).elevation * 1000.0;
        let mut min_elevation = center_elevation;
        let mut max_elevation = center_elevation;

        for radius_m in [2_000.0, 5_000.0] {
            for sample_index in 0..8 {
                let angle = sample_index as f64 * std::f64::consts::TAU / 8.0;
                let angular_radius = radius_m / radius;
                let sample_yaw = yaw + angular_radius * angle.cos() / pitch.cos();
                let sample_pitch = pitch + angular_radius * angle.sin();
                let cp = sample_pitch.cos();
                let direction = DVec3::new(
                    cp * sample_yaw.sin(),
                    sample_pitch.sin(),
                    cp * sample_yaw.cos(),
                );
                let elevation =
                    er_world::TerrainField::sample(&field, direction).elevation * 1000.0;
                min_elevation = min_elevation.min(elevation);
                max_elevation = max_elevation.max(elevation);
            }
        }

        assert_eq!(scenario.target_altitude_m, Some(10_000.0));
        assert!(min_elevation > 0.0, "mountain view must remain over land");
        assert!(
            max_elevation - min_elevation > 1_500.0,
            "mountain view needs kilometer-scale relief"
        );
    }

    #[test]
    fn coastline_scenario_crosses_macro_sea_level() {
        let radius = 6_371_000.0;
        let seed = er_core::seed::PlanetSeed(0xC0FFEE);
        let climate = er_world::planet_params(seed);
        let field = er_world::ProceduralTerrainField::new_metric(
            er_world::elevation::elevation_params(seed),
            climate,
            radius,
        );
        let scenario = fixed_coverage_scenarios(radius)
            .into_iter()
            .find(|scenario| scenario.name == "coastline")
            .unwrap();
        let yaw = scenario.camera_yaw as f64;
        let pitch = scenario.camera_pitch as f64;
        let cp = pitch.cos();
        let direction = DVec3::new(cp * yaw.sin(), pitch.sin(), cp * yaw.cos());
        let (face, u, v) = er_core::math::dir_to_uv(direction);
        let (_, tangent, bitangent) = er_core::math::tangent_frame(face, u, v);
        let center = er_world::TerrainField::sample(&field, direction);
        let mut min_macro = center.low_freq_elev as f64;
        let mut max_macro = min_macro;

        for sample_index in 0..16 {
            let angle = sample_index as f64 * std::f64::consts::TAU / 16.0;
            let offset = (tangent * angle.cos() + bitangent * angle.sin()) * 2_000.0;
            let sample_direction = (direction * radius + offset).normalize();
            let macro_elevation =
                er_world::TerrainField::sample(&field, sample_direction).low_freq_elev as f64;
            min_macro = min_macro.min(macro_elevation);
            max_macro = max_macro.max(macro_elevation);
        }

        assert_eq!(scenario.target_altitude_m, Some(2_000.0));
        assert!(
            center.low_freq_elev as f64 > climate.sea_level,
            "coastline camera center must remain on the macro-land side"
        );
        assert!(
            center.elevation as f64 * 1_000.0 > climate.sea_level * 1_000.0 + 500.0,
            "coastline camera center must remain safely above the water surface"
        );
        assert!(
            min_macro < climate.sea_level && max_macro > climate.sea_level,
            "2 km footprint must cross macro sea level: min={min_macro}, max={max_macro}, sea={}",
            climate.sea_level
        );
    }

    #[test]
    fn fixed_coverage_has_all_required_views() {
        let scenarios = fixed_coverage_scenarios(6_371_000.0);
        assert_eq!(scenarios.len(), 14);
        let names: Vec<&str> = scenarios.iter().map(|s| s.name.as_str()).collect();
        for required in [
            "altitude_10m",
            "altitude_100m",
            "altitude_1km",
            "altitude_10km",
            "orbit",
            "globe",
            "surface",
            "coastline",
            "mountain",
            "cube_corner",
            "cube_edge_before",
            "cube_edge_after",
            "origin_shift_before",
            "origin_shift_after",
        ] {
            assert!(
                names.contains(&required),
                "missing required scenario '{required}'"
            );
        }
    }

    #[test]
    fn altitude_scenarios_have_correct_targets() {
        let scenarios = fixed_coverage_scenarios(6_371_000.0);
        let alt_10m = scenarios.iter().find(|s| s.name == "altitude_10m").unwrap();
        assert_eq!(alt_10m.target_altitude_m, Some(10.0));
        let alt_100m = scenarios
            .iter()
            .find(|s| s.name == "altitude_100m")
            .unwrap();
        assert_eq!(alt_100m.target_altitude_m, Some(100.0));
        let alt_1km = scenarios.iter().find(|s| s.name == "altitude_1km").unwrap();
        assert_eq!(alt_1km.target_altitude_m, Some(1000.0));
        let alt_10km = scenarios
            .iter()
            .find(|s| s.name == "altitude_10km")
            .unwrap();
        assert_eq!(alt_10km.target_altitude_m, Some(10000.0));
        let orbit = scenarios.iter().find(|s| s.name == "orbit").unwrap();
        assert_eq!(orbit.target_altitude_m, None);
    }

    #[test]
    fn origin_shift_pair_stays_in_the_same_local_area() {
        let scenarios = fixed_coverage_scenarios(6_371_000.0);
        let before = scenarios
            .iter()
            .find(|scenario| scenario.name == "origin_shift_before")
            .unwrap();
        let after = scenarios
            .iter()
            .find(|scenario| scenario.name == "origin_shift_after")
            .unwrap();
        let angular_delta = (after.camera_yaw - before.camera_yaw).abs() as f64;
        let surface_distance = angular_delta * 6_371_000.0 * before.camera_pitch.cos() as f64;

        assert!((1_000.0..1_500.0).contains(&surface_distance));
        assert_eq!(before.camera_pitch, after.camera_pitch);
        assert_eq!(before.target_altitude_m, after.target_altitude_m);
    }

    #[test]
    fn altitude_from_composed_field_produces_more_than_radius() {
        let radius = 6_371_000.0;
        let direction = DVec3::new(0.0, 0.5, 0.866).normalize();
        let state = er_terrain::TerrainState::for_preset(
            er_core::config::PlanetPreset::EarthScale,
            1000.0,
            er_core::seed::PlanetSeed(0xC0FFEE),
        );
        let elevation = state.field.sample(direction).elevation * state.elevation_scale as f64;
        let target_alt = 100.0;
        let radial = radius + elevation + target_alt;
        let altitude = radial - radius - elevation;
        assert!(
            (altitude - target_alt).abs() < 0.001,
            "computed altitude {altitude} should match target {target_alt}"
        );
        assert!(radial > 0.0, "radial {radial} must be positive");
    }

    #[test]
    fn acceptance_timeout_fails_unless_exploratory() {
        assert!(acceptance_timeout_fails(false));
        assert!(!acceptance_timeout_fails(true));
    }

    #[test]
    fn timeout_requires_frame_and_wall_clock_budgets() {
        assert!(!scenario_timed_out(599, 60.0, 30.0));
        assert!(!scenario_timed_out(600, 29.9, 30.0));
        assert!(scenario_timed_out(600, 30.0, 30.0));
    }

    #[test]
    fn ten_meter_acceptance_requires_five_meter_vertex_spacing() {
        let mut scenario = fixed_coverage_scenarios(6_371_000.0).remove(0);
        let mut debug = TerrainDebugInfo {
            vertex_spacing_m: 5.01,
            ..TerrainDebugInfo::default()
        };
        assert!(!detail_target_met(&scenario, &debug));
        debug.vertex_spacing_m = 4.99;
        assert!(detail_target_met(&scenario, &debug));

        scenario.target_altitude_m = Some(100.0);
        debug.vertex_spacing_m = 1000.0;
        assert!(detail_target_met(&scenario, &debug));
    }

    #[test]
    fn altitude_acceptance_rejects_clearance_drift() {
        let scenario = fixed_coverage_scenarios(6_371_000.0).remove(0);
        let mut debug = TerrainDebugInfo {
            camera_altitude_m: 330.7,
            ..TerrainDebugInfo::default()
        };
        assert!(!altitude_target_met(&scenario, &debug));

        debug.camera_altitude_m = 10.4;
        assert!(altitude_target_met(&scenario, &debug));

        debug.camera_altitude_m = 11.3;
        assert!(altitude_target_met(&scenario, &debug));
    }

    #[test]
    fn altitude_calculation_yields_reasonable_earth_distances() {
        let radius = 6_371_000.0;
        let ter = er_terrain::TerrainState::for_preset(
            er_core::config::PlanetPreset::EarthScale,
            1000.0,
            er_core::seed::PlanetSeed(0xC0FFEE),
        );
        let direction = DVec3::new(0.52, 0.30, 0.80).normalize();
        let elev = ter.field.sample(direction).elevation * ter.elevation_scale as f64;
        for alt_m in [10.0f64, 100.0, 1000.0, 10000.0] {
            let radial = radius + elev + alt_m;
            assert!(
                radial > radius - 12_000.0,
                "alt {alt_m} radial {radial} implausibly low"
            );
            assert!(
                radial < radius * 50.0,
                "alt {} radial {} suspiciously large",
                alt_m,
                radial
            );
        }
    }
}
