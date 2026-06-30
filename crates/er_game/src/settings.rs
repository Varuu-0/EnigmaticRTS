//! Graphics settings: persisted (present mode / window mode apply at startup; MSAA applies live).
//!
//! On hybrid (Optimus) graphics, changing `present_mode` or `window.mode` at runtime recreates
//! the Vulkan swapchain, and repeated recreations lose the GPU device (`DeviceLost`). So
//! VSync/Fullscreen are only applied at startup from a persisted file ("restart to apply"),
//! while MSAA (a per-camera component, not a swapchain change) is applied live.

use std::fs;
use std::path::PathBuf;

use bevy::{
    prelude::*,
    window::{MonitorSelection, PresentMode, VideoModeSelection, WindowMode},
};

const SETTINGS_FILE: &str = "er_game_settings.txt";

#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct GraphicsSettings {
    pub vsync: bool,
    pub msaa: u32,
    pub fullscreen: bool,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            vsync: false,
            msaa: 4,
            fullscreen: false,
        }
    }
}

impl GraphicsSettings {
    pub fn present_mode(&self) -> PresentMode {
        if self.vsync {
            PresentMode::AutoVsync
        } else {
            PresentMode::AutoNoVsync
        }
    }

    pub fn window_mode(&self) -> WindowMode {
        if self.fullscreen {
            WindowMode::Fullscreen(MonitorSelection::Current, VideoModeSelection::Current)
        } else {
            WindowMode::Windowed
        }
    }

    pub fn msaa(&self) -> Msaa {
        Msaa::from_samples(self.msaa)
    }
}

fn settings_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(SETTINGS_FILE)))
        .unwrap_or_else(|| PathBuf::from(SETTINGS_FILE))
}

pub fn load_settings() -> GraphicsSettings {
    let mut s = GraphicsSettings::default();
    if let Ok(text) = fs::read_to_string(settings_path()) {
        for line in text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                match k.trim() {
                    "vsync" => s.vsync = v.trim() == "1",
                    "msaa" => s.msaa = v.trim().parse().unwrap_or(s.msaa),
                    "fullscreen" => s.fullscreen = v.trim() == "1",
                    _ => {}
                }
            }
        }
    }
    s
}

pub fn save_settings(s: &GraphicsSettings) {
    let text = format!(
        "vsync={}\nmsaa={}\nfullscreen={}\n",
        s.vsync as u8,
        s.msaa,
        s.fullscreen as u8
    );
    let _ = fs::write(settings_path(), text);
}

pub fn apply_graphics_settings(
    settings: Res<GraphicsSettings>,
    mut cameras: Query<&mut Msaa, With<Camera3d>>,
) {
    // Live apply for MSAA only (per-camera component, no swapchain recreation). Present mode /
    // window mode are applied at startup only — see apply_startup_window_mode + the window config.
    if !settings.is_changed() {
        return;
    }
    let want_msaa = settings.msaa();
    for mut cam_msaa in &mut cameras {
        if *cam_msaa != want_msaa {
            info!("Applying MSAA: {:?}", want_msaa);
            *cam_msaa = want_msaa;
        }
    }
}
