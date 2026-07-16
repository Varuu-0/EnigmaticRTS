//! er_game entry point.
//!
//! Phase 3: terrain quadtree LOD with GPU-displaced chunks, orbital camera,
//! and a debug overlay. ESC still opens the settings menu.

use bevy::camera::{PerspectiveProjection, Projection};
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

mod bench;
mod camera;
mod chunk_cap_controller;
mod crash;
mod debug_overlay;
mod diagnostics;
mod frame_timing;
mod menu;
mod screenshot_test;
mod settings;
mod space;
#[cfg(feature = "terrain_diffusion")]
mod terrain_diffusion;

use camera::{CameraPlugin, CameraUpdate, OrbitCamera};
use chunk_cap_controller::DynamicChunkCapControllerPlugin;
use debug_overlay::DebugOverlayPlugin;
use diagnostics::PerformanceDiagnosticsPlugin;
use er_terrain::TerrainPlugin;
use frame_timing::FrameTimingPlugin;
use menu::SettingsMenuPlugin;
use screenshot_test::{parse_test_args, ScreenshotTestPlugin};
use settings::GraphicsSettings;
use space::SpacePlugin;

fn main() {
    crash::install_crash_hook();
    configure_log_filter();

    let planet_preset = detect_planet_preset();
    let test_config = parse_test_args();
    let bench_config = bench::parse_bench_args();
    let is_test_mode = test_config.is_some();
    let is_bench_mode = bench_config.is_some();
    let gpu_diagnostics = has_gpu_diagnostics_flag();
    let headless = is_bench_mode;

    let settings = settings::load_settings();
    let present_mode = settings.present_mode();

    #[cfg(feature = "terrain_diffusion")]
    let terrain_diffusion = terrain_diffusion::startup_from_args(
        // Screenshot mode can exercise hybrid terrain when explicitly requested.
        // Benchmark mode remains procedural so its measurements stay repeatable.
        is_bench_mode,
        planet_preset.radius_m(),
    );
    #[cfg(feature = "terrain_diffusion")]
    let terrain_plugin = terrain_diffusion
        .as_ref()
        .map(|startup| {
            TerrainPlugin::from_preset(planet_preset, 1000.0, er_core::seed::PlanetSeed(0xC0FFEE))
                .with_hybrid_macro_field(startup.cache.clone())
        })
        .unwrap_or_else(|| {
            TerrainPlugin::from_preset(planet_preset, 1000.0, er_core::seed::PlanetSeed(0xC0FFEE))
        });
    #[cfg(not(feature = "terrain_diffusion"))]
    let terrain_plugin =
        TerrainPlugin::from_preset(planet_preset, 1000.0, er_core::seed::PlanetSeed(0xC0FFEE));
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
        // Screenshot and benchmark modes need deterministic shader readiness before
        // their first captured frame.
        synchronous_pipeline_compilation: is_test_mode || is_bench_mode,
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
        .add_plugins(LogDiagnosticsPlugin::default())
        .add_plugins(RenderDiagnosticsPlugin)
        .add_plugins(PerformanceDiagnosticsPlugin)
        .insert_resource(ClearColor(Color::srgb(0.02, 0.03, 0.05)))
        .insert_resource(settings)
        .add_plugins(CameraPlugin::default())
        .add_plugins(terrain_plugin)
        .add_plugins(SpacePlugin)
        .configure_sets(Update, er_terrain::TerrainUpdate.after(CameraUpdate));

    #[cfg(feature = "terrain_diffusion")]
    if let Some(startup) = terrain_diffusion {
        app.add_plugins(terrain_diffusion::TerrainDiffusionPlugin::new(
            startup.cache,
            startup.config,
        ));
    }

    if is_bench_mode {
        let config = bench_config.unwrap();
        app.insert_resource(config);
        app.add_plugins(bench::BenchPlugin);
    } else if is_test_mode {
        let config = test_config.unwrap();
        app.insert_resource(config);
        app.add_plugins(ScreenshotTestPlugin);
    } else {
        app.add_plugins(SettingsMenuPlugin)
            .add_plugins(DynamicChunkCapControllerPlugin)
            .add_plugins(FrameTimingPlugin)
            .add_plugins(DebugOverlayPlugin);
    }

    app.add_systems(Startup, (setup, apply_startup_window_mode));

    app.run();
}

fn has_gpu_diagnostics_flag() -> bool {
    std::env::args().any(|arg| arg == "--gpu-diagnostics")
}

fn detect_planet_preset() -> PlanetPreset {
    if std::env::args().any(|arg| arg == "--earth-scale") {
        PlanetPreset::EarthScale
    } else {
        PlanetPreset::MiniatureDebug
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
    let radius = terrain_state.planet_radius as f32;
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            near: 1.0,
            far: (radius * 50.0).max(500000.0),
            fov: 60.0_f32.to_radians(),
            ..default()
        }),
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
