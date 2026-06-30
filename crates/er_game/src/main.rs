//! er_game entry point.
//!
//! Phase 0 bootstrap: opens a window ("Planet Solar Sim"), logs FPS, and
//! provides an ESC settings menu (VSync / Fullscreen / MSAA / quit). VSync and
//! Fullscreen are applied at startup from a persisted file (changing them at
//! runtime is unsafe on hybrid graphics); MSAA applies live. Planet/terrain
//! systems get wired in by later phases.

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
};

mod menu;
mod settings;

use menu::SettingsMenuPlugin;
use settings::GraphicsSettings;

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
        .add_systems(Startup, (setup, apply_startup_window_mode))
        .run();
}

fn setup(mut commands: Commands, settings: Res<GraphicsSettings>) {
    commands.spawn((Camera3d::default(), settings.msaa()));
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
