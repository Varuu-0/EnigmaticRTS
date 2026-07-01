//! er_game entry point.
//!
//! Phase 3: terrain quadtree LOD with GPU-displaced chunks, orbital camera,
//! and a debug overlay. ESC still opens the settings menu.

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
    render::camera::{PerspectiveProjection, Projection},
};

mod camera;
mod debug_overlay;
mod menu;
mod settings;

use camera::{CameraPlugin, OrbitCamera};
use debug_overlay::DebugOverlayPlugin;
use menu::SettingsMenuPlugin;
use settings::GraphicsSettings;
use er_terrain::TerrainPlugin;

fn main() {
    let settings = settings::load_settings();
    let present_mode = settings.present_mode();

    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Planet Solar Sim".into(),
                    present_mode,
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(LogDiagnosticsPlugin::default())
        .insert_resource(ClearColor(Color::srgb(0.02, 0.03, 0.05)))
        .insert_resource(settings)
        .add_plugins(SettingsMenuPlugin)
        .add_plugins(CameraPlugin)
        .add_plugins(DebugOverlayPlugin)
        .add_plugins(TerrainPlugin::default())
        .add_systems(Startup, (setup, apply_startup_window_mode))
        .run();
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
