//! er_game entry point.
//!
//! Phase 3: terrain quadtree LOD with GPU-displaced chunks, orbital camera,
//! and a debug overlay. ESC still opens the settings menu.

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    log::{Level, LogPlugin},
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        settings::{WgpuFeatures, WgpuSettings},
        RenderPlugin,
    },
};
use er_core::config::PlanetPreset;
use std::num::NonZeroU32;
#[cfg(feature = "terrain_diffusion")]
use std::sync::Arc;

mod baseline_manifest;
mod bench;
mod camera;
mod cli;
mod crash;
mod debug_overlay;
mod diagnostics;
mod frame_timing;
#[cfg(feature = "terrain_diffusion")]
mod learned_stress;
#[cfg(feature = "terrain_diffusion")]
mod m5_surface_test;
mod menu;
mod projection;
mod screenshot_test;
mod settings;
mod space;
mod telemetry;
#[cfg(feature = "terrain_diffusion")]
mod terrain_diffusion;
#[cfg(feature = "terrain_diffusion")]
mod terrain_diffusion_stress;

use baseline_manifest::BaselineManifestPlugin;
#[cfg(feature = "terrain_diffusion")]
use baseline_manifest::TerrainDiffusionManifest;
use camera::{CameraPlugin, CameraUpdate, OrbitCamera};
use debug_overlay::DebugOverlayPlugin;
use diagnostics::PerformanceDiagnosticsPlugin;
use er_terrain::TerrainPlugin;
use frame_timing::FrameTimingPlugin;
use menu::SettingsMenuPlugin;
use screenshot_test::{parse_test_args, ScreenshotTestPlugin};
use settings::GraphicsSettings;
use space::{CloudRuntimeConfig, SpacePlugin, SpaceUpdate};

fn main() {
    crash::install_crash_hook();
    configure_log_filter();

    let planet_preset = detect_planet_preset();
    let planet_seed = detect_planet_seed();
    let environment_preset = detect_environment_preset();
    let test_config = parse_test_args(planet_preset.radius_m());
    let bench_config = bench::parse_bench_args(planet_preset.radius_m());
    let is_test_mode = test_config.is_some();
    let is_bench_mode = bench_config.is_some();
    #[cfg(feature = "terrain_diffusion")]
    let stress_config = terrain_diffusion_stress::StressConfig::parse_args();
    #[cfg(feature = "terrain_diffusion")]
    let is_stress_mode = stress_config.is_some();
    #[cfg(not(feature = "terrain_diffusion"))]
    let is_stress_mode = false;
    #[cfg(feature = "terrain_diffusion")]
    let learned_stress_config = learned_stress::LearnedStressConfig::parse_args();
    #[cfg(feature = "terrain_diffusion")]
    let is_learned_stress_mode = learned_stress_config.is_some();
    #[cfg(not(feature = "terrain_diffusion"))]
    let is_learned_stress_mode = false;
    #[cfg(feature = "terrain_diffusion")]
    let m5_test_config = m5_surface_test::M5TestConfig::parse_args();
    #[cfg(feature = "terrain_diffusion")]
    let is_m5_test_mode = m5_test_config.is_some();
    #[cfg(not(feature = "terrain_diffusion"))]
    let is_m5_test_mode = false;
    let gpu_diagnostics = has_gpu_diagnostics_flag();
    let cloud_test_mode = std::env::args().any(|arg| arg == "--cloud-test");
    let dump_manifest_path = baseline_manifest::parse_dump_path();
    let exit_after_manifest_dump = dump_manifest_path.is_some();
    let headless = is_bench_mode || is_stress_mode || is_learned_stress_mode || is_m5_test_mode;

    let settings = settings::load_settings();
    let present_mode = settings.present_mode();

    #[cfg(feature = "terrain_diffusion")]
    let terrain_diffusion = terrain_diffusion::startup_from_args(
        // Screenshot mode can exercise hybrid terrain when explicitly requested.
        // Benchmark mode remains procedural so its measurements stay repeatable.
        is_bench_mode,
        planet_preset.radius_m(),
        planet_seed,
    );
    #[cfg(feature = "terrain_diffusion")]
    let terrain_diffusion_manifest =
        terrain_diffusion
            .as_ref()
            .map(|startup| TerrainDiffusionManifest {
                endpoint: startup.config.endpoint.to_string(),
                native_resolution: startup.config.metadata.native_resolution,
                native_pixel_scale_m: startup.config.metadata.native_pixel_scale_m,
                api_scale: startup.config.metadata.api_scale,
                halo_samples: startup.config.metadata.halo_samples,
                tiles_per_face_edge: startup.config.metadata.tiles_per_face_edge,
                is_upsampled: startup.config.metadata.is_upsampled(),
            });
    #[cfg(not(feature = "terrain_diffusion"))]
    let terrain_diffusion_manifest = None;

    #[cfg(feature = "terrain_diffusion")]
    let terrain_plugin = terrain_diffusion
        .as_ref()
        .map(|startup| {
            let rebuild_source = Arc::new(terrain_diffusion::StreamingRebuildSource::new(
                Arc::clone(&startup.streaming_queue),
            ));
            TerrainPlugin::from_preset(planet_preset, 1000.0, planet_seed)
                .with_environment_preset(environment_preset)
                .with_hybrid_macro_field(startup.chart_field.clone())
                .with_halo_checker(startup.chart_field.clone())
                .with_rebuild_source(rebuild_source)
        })
        .unwrap_or_else(|| {
            TerrainPlugin::from_preset(planet_preset, 1000.0, planet_seed)
                .with_environment_preset(environment_preset)
        });
    #[cfg(not(feature = "terrain_diffusion"))]
    let terrain_plugin = TerrainPlugin::from_preset(planet_preset, 1000.0, planet_seed)
        .with_environment_preset(environment_preset);
    #[cfg(not(feature = "terrain_diffusion"))]
    if std::env::args().any(|arg| arg == "--terrain-diffusion") {
        eprintln!(
            "Terrain Diffusion support is not compiled in. Run with --features terrain_diffusion."
        );
    }

    let mut app = App::new();

    let mut wgpu_settings = WgpuSettings::default();
    if gpu_diagnostics {
        wgpu_settings.features |=
            WgpuFeatures::TIMESTAMP_QUERY | WgpuFeatures::PIPELINE_STATISTICS_QUERY;
    }

    let render_plugin = RenderPlugin {
        // Screenshot, benchmark, and learned-stress modes need deterministic
        // shader readiness before their first captured frame/report.
        synchronous_pipeline_compilation: is_test_mode
            || is_bench_mode
            || is_learned_stress_mode
            || is_m5_test_mode,
        render_creation: wgpu_settings.into(),
        ..default()
    };
    if headless {
        app.add_plugins(
            DefaultPlugins
                .set(log_plugin())
                .set(render_plugin)
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Planet Solar Sim (Bench)".into(),
                        present_mode,
                        visible: false,
                        ..default()
                    }),
                    ..default()
                }),
        );
    } else if is_test_mode {
        app.add_plugins(
            DefaultPlugins
                .set(log_plugin())
                .set(render_plugin)
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Planet Solar Sim (Test Mode)".into(),
                        present_mode,
                        visible: true,
                        ..default()
                    }),
                    ..default()
                }),
        );
    } else {
        app.add_plugins(
            DefaultPlugins
                .set(log_plugin())
                .set(render_plugin)
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Planet Solar Sim".into(),
                        present_mode,
                        // Mailbox uses this to size the Vulkan swapchain. Three
                        // frames in flight prevents the main thread from
                        // stalling on surface acquisition while the render
                        // thread presents the previous frame.
                        desired_maximum_frame_latency: NonZeroU32::new(3),
                        ..default()
                    }),
                    ..default()
                }),
        );
    }

    app.add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(RenderDiagnosticsPlugin)
        .add_plugins(PerformanceDiagnosticsPlugin)
        .insert_resource(ClearColor(Color::srgb(0.02, 0.03, 0.05)))
        .insert_resource(settings)
        .insert_resource(CloudRuntimeConfig {
            spawning_enabled: cloud_test_mode
                || (!is_test_mode
                    && !is_bench_mode
                    && !is_stress_mode
                    && !is_learned_stress_mode
                    && !is_m5_test_mode),
            // Normal gameplay starts with one mature, visible cluster. Later
            // clusters still use the sparse stochastic scheduler.
            force_initial_cluster: cloud_test_mode
                || (!is_test_mode
                    && !is_bench_mode
                    && !is_stress_mode
                    && !is_learned_stress_mode
                    && !is_m5_test_mode),
        })
        .add_plugins(CameraPlugin)
        .add_plugins(terrain_plugin)
        .add_plugins(SpacePlugin)
        .configure_sets(Update, SpaceUpdate.after(CameraUpdate))
        .configure_sets(Update, er_terrain::TerrainUpdate.after(CameraUpdate));

    if !is_bench_mode {
        app.add_plugins(LogDiagnosticsPlugin::default());
    }

    let manifest_path = dump_manifest_path.or_else(|| {
        test_config
            .as_ref()
            .map(|config| config.output_dir.join("baseline_manifest.json"))
    });
    app.add_plugins(BaselineManifestPlugin::new(
        manifest_path,
        exit_after_manifest_dump,
        terrain_diffusion_manifest,
        detect_hardware_profile(),
        planet_seed.0,
    ));

    #[cfg(feature = "terrain_diffusion")]
    if let Some(startup) = terrain_diffusion {
        let recorder = Arc::new(terrain_diffusion::StreamingSampleRecorder::new(Arc::clone(
            &startup.streaming_queue,
        )));
        app.insert_resource(er_terrain::TerrainSampleSourceRecorder(recorder));
        app.add_plugins(terrain_diffusion::TerrainDiffusionPlugin::new(
            startup.cache,
            startup.chart_field,
            startup.streaming_queue,
            startup.config,
        ));
    }

    if is_bench_mode {
        let config = bench_config.unwrap();
        app.insert_resource(config);
        app.add_plugins(bench::BenchPlugin);
    } else if is_test_mode {
        let config = test_config.unwrap();
        app.insert_resource(debug_overlay::DebugViewSelection(config.debug_view_mode));
        app.insert_resource(config);
        app.add_plugins(ScreenshotTestPlugin);
    } else {
        app.add_plugins(SettingsMenuPlugin)
            .add_plugins(FrameTimingPlugin)
            .add_plugins(DebugOverlayPlugin);
    }

    #[cfg(feature = "terrain_diffusion")]
    if is_stress_mode {
        let config = stress_config.unwrap();
        app.insert_resource(config);
        app.add_plugins(terrain_diffusion_stress::TerrainDiffusionStressPlugin);
    }

    #[cfg(feature = "terrain_diffusion")]
    if is_learned_stress_mode {
        let config = learned_stress_config.unwrap();
        app.insert_resource(config);
        app.add_plugins(learned_stress::LearnedStressPlugin);
    }

    #[cfg(feature = "terrain_diffusion")]
    if is_m5_test_mode {
        let config = m5_test_config.unwrap();
        app.insert_resource(config);
        app.add_plugins(m5_surface_test::M5TestPlugin);
    }

    app.add_systems(Startup, (setup, apply_startup_window_mode));

    app.run();
}

fn has_gpu_diagnostics_flag() -> bool {
    std::env::args().any(|arg| arg == "--gpu-diagnostics")
}

/// Detect the active hardware profile name from `--hardware-profile <name>` or
/// by matching the primary Optimus laptop profile when the adapter is NVIDIA on
/// a hybrid system. Returns `None` when no profile can be inferred so the
/// manifest records `null` rather than a fabricated name.
fn detect_hardware_profile() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(index) = args.iter().position(|arg| arg == "--hardware-profile") {
        return args.get(index + 1).cloned();
    }
    let gpu = er_game::gpu_telemetry::sample();
    if gpu.description.to_lowercase().contains("nvidia") {
        Some("rtx3060_optimus".to_owned())
    } else {
        None
    }
}

fn detect_planet_preset() -> PlanetPreset {
    if std::env::args().any(|arg| arg == "--miniature") {
        PlanetPreset::MiniatureDebug
    } else {
        PlanetPreset::default()
    }
}

fn detect_planet_seed() -> er_core::seed::PlanetSeed {
    let args: Vec<String> = std::env::args().collect();
    let seed = args
        .windows(2)
        .find(|pair| pair[0] == "--seed")
        .and_then(|pair| cli::parse_seed_value(&pair[1]))
        .unwrap_or(0xC0FFEE);
    er_core::seed::PlanetSeed(seed)
}

/// Parse `--environment-preset <name>` from the command line.
///
/// Accepts `earth-like` (default), `hot-arid`, and `cold-icehouse`. Unknown
/// values log a warning and conservatively fall back to `EarthLike` so the
/// game still runs — consistent with how other optional CLI flags degrade.
fn detect_environment_preset() -> er_world::environment::EnvironmentPreset {
    parse_environment_preset_from_args(&std::env::args().collect::<Vec<_>>())
}

/// Pure helper for CLI environment-preset parsing (testable without process
/// argv). Returns the resolved preset, emitting a warning to stderr for
/// unrecognized values.
pub(crate) fn parse_environment_preset_from_args(
    args: &[String],
) -> er_world::environment::EnvironmentPreset {
    let Some(value) = args
        .windows(2)
        .find(|pair| pair[0] == "--environment-preset")
        .map(|pair| pair[1].as_str())
    else {
        return er_world::environment::EnvironmentPreset::EarthLike;
    };
    match er_world::environment::EnvironmentPreset::parse(value) {
        Some(preset) => preset,
        None => {
            eprintln!(
                "Unrecognized --environment-preset '{value}'; falling back to EarthLike. \
                 Valid values: earth-like, hot-arid, cold-icehouse."
            );
            er_world::environment::EnvironmentPreset::EarthLike
        }
    }
}

fn log_plugin() -> LogPlugin {
    LogPlugin {
        filter: "info,wgpu=warn,naga=warn".to_owned(),
        level: Level::INFO,
        ..default()
    }
}

fn configure_log_filter() {
    let args: Vec<String> = std::env::args().collect();
    let mut index = 1;

    while index + 1 < args.len() {
        if args[index] == "--log-level" {
            std::env::set_var("RUST_LOG", &args[index + 1]);
            return;
        }
        index += 1;
    }
}

fn setup(
    mut commands: Commands,
    settings: Res<GraphicsSettings>,
    terrain_state: Res<er_terrain::TerrainState>,
) {
    let radius = terrain_state.planet_radius;
    commands.spawn((
        Camera3d::default(),
        projection::ProjectionPolicy::for_placement(radius, None).to_projection(),
        OrbitCamera::for_planet(radius, terrain_state.elevation_scale),
        Transform::default(),
        settings.msaa(),
    ));
}

fn apply_startup_window_mode(settings: Res<GraphicsSettings>, mut windows: Query<&mut Window>) {
    if !settings.fullscreen {
        return;
    }
    for mut window in &mut windows {
        if window.mode != settings.window_mode() {
            info!("Applying window mode: {:?}", settings.window_mode());
            window.mode = settings.window_mode();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        std::iter::once("er_game".to_owned())
            .chain(parts.iter().map(|s| (*s).to_owned()))
            .collect()
    }

    #[test]
    fn environment_preset_defaults_to_earth_like() {
        assert_eq!(
            parse_environment_preset_from_args(&args(&[])),
            er_world::environment::EnvironmentPreset::EarthLike
        );
    }

    #[test]
    fn environment_preset_parses_known_values() {
        assert_eq!(
            parse_environment_preset_from_args(&args(&["--environment-preset", "earth-like"])),
            er_world::environment::EnvironmentPreset::EarthLike
        );
        assert_eq!(
            parse_environment_preset_from_args(&args(&["--environment-preset", "hot-arid"])),
            er_world::environment::EnvironmentPreset::HotArid
        );
        assert_eq!(
            parse_environment_preset_from_args(&args(&["--environment-preset", "cold-icehouse"])),
            er_world::environment::EnvironmentPreset::ColdIcehouse
        );
    }

    #[test]
    fn environment_preset_accepts_compact_forms() {
        assert_eq!(
            parse_environment_preset_from_args(&args(&["--environment-preset", "hotarid"])),
            er_world::environment::EnvironmentPreset::HotArid
        );
        assert_eq!(
            parse_environment_preset_from_args(&args(&["--environment-preset", "coldicehouse"])),
            er_world::environment::EnvironmentPreset::ColdIcehouse
        );
    }

    #[test]
    fn environment_preset_is_case_insensitive() {
        assert_eq!(
            parse_environment_preset_from_args(&args(&["--environment-preset", "EARTH-LIKE"])),
            er_world::environment::EnvironmentPreset::EarthLike
        );
    }

    #[test]
    fn environment_preset_falls_back_on_unknown() {
        assert_eq!(
            parse_environment_preset_from_args(&args(&["--environment-preset", "mars"])),
            er_world::environment::EnvironmentPreset::EarthLike
        );
    }
}
