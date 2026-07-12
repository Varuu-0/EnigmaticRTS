//! er_game entry point.
//!
//! Phase 3: terrain quadtree LOD with GPU-displaced chunks, orbital camera,
//! and a debug overlay. ESC still opens the settings menu.

use bevy::camera::{PerspectiveProjection, Projection};
use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
    render::RenderPlugin,
};

mod bench;
mod camera;
mod debug_overlay;
mod menu;
mod screenshot_test;
mod settings;
mod space;

use camera::{CameraPlugin, CameraUpdate, OrbitCamera};
use debug_overlay::DebugOverlayPlugin;
use er_terrain::TerrainPlugin;
use menu::SettingsMenuPlugin;
use screenshot_test::{parse_test_args, ScreenshotTestPlugin};
use settings::GraphicsSettings;
use space::SpacePlugin;

fn main() {
    let test_config = parse_test_args();
    let bench_config = bench::parse_bench_args();
    let is_test_mode = test_config.is_some();
    let is_bench_mode = bench_config.is_some();
    let headless = is_bench_mode;

    let settings = settings::load_settings();
    let present_mode = settings.present_mode();

    let mut app = App::new();

    let render_plugin = RenderPlugin {
        // Screenshot and benchmark modes need deterministic shader readiness before
        // their first captured frame.
        synchronous_pipeline_compilation: is_test_mode || is_bench_mode,
        ..default()
    };

    if headless {
        app.add_plugins(DefaultPlugins.set(render_plugin).set(WindowPlugin {
            primary_window: Some(Window {
                title: "Planet Solar Sim (Bench)".into(),
                present_mode,
                visible: false,
                ..default()
            }),
            ..default()
        }));
    } else if is_test_mode {
        app.add_plugins(DefaultPlugins.set(render_plugin).set(WindowPlugin {
            primary_window: Some(Window {
                title: "Planet Solar Sim (Test Mode)".into(),
                present_mode,
                visible: true,
                ..default()
            }),
            ..default()
        }));
    } else {
        app.add_plugins(DefaultPlugins.set(render_plugin).set(WindowPlugin {
            primary_window: Some(Window {
                title: "Planet Solar Sim".into(),
                present_mode,
                ..default()
            }),
            ..default()
        }));
    }

    app.add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(LogDiagnosticsPlugin::default())
        .insert_resource(ClearColor(Color::srgb(0.02, 0.03, 0.05)))
        .insert_resource(settings)
        .add_plugins(CameraPlugin)
        .add_plugins(TerrainPlugin::default())
        .add_plugins(SpacePlugin)
        .configure_sets(Update, er_terrain::TerrainUpdate.after(CameraUpdate));

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
            .add_plugins(DebugOverlayPlugin);
    }

    app.add_systems(Startup, (setup, apply_startup_window_mode));

    app.run();
}

fn setup(mut commands: Commands, settings: Res<GraphicsSettings>) {
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            near: 1.0,
            far: 500000.0,
            fov: 60.0_f32.to_radians(),
            ..default()
        }),
        OrbitCamera::default(),
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
