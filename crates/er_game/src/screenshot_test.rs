use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
use std::path::PathBuf;

#[derive(Resource, Default)]
pub struct ScreenshotTestConfig {
    pub output_dir: PathBuf,
    pub scenarios: Vec<ScreenshotScenario>,
    pub current_index: usize,
    pub frames_to_wait: u32,
    pub frames_waited: u32,
    pub completed: bool,
    pub pending_screenshots: u32,
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
        app.add_systems(Update, run_screenshot_test);
    }
}

fn run_screenshot_test(
    mut config: ResMut<ScreenshotTestConfig>,
    mut camera_query: Query<(&mut crate::camera::OrbitCamera, &mut Transform), With<Camera3d>>,
    mut commands: Commands,
    capturing: Query<Entity, With<Capturing>>,
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
            std::process::exit(0);
        }
        return;
    }

    if config.scenarios.is_empty() {
        config.completed = true;
        return;
    }

    if config.current_index >= config.scenarios.len() {
        info!("Screenshot test completed: {} scenarios captured", config.scenarios.len());
        config.completed = true;
        return;
    }

    let scenario = config.scenarios[config.current_index].clone();

    if config.frames_waited == 0 {
        info!("Capturing scenario: {} (index {})", scenario.name, config.current_index);
        
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

    if config.frames_waited >= config.frames_to_wait {
        let filename = format!("{}.png", scenario.name);
        let path = config.output_dir.join(&filename);
        
        info!("Taking screenshot: {:?}", path);
        
        std::fs::create_dir_all(&config.output_dir).ok();
        
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));
        
        config.current_index += 1;
        config.frames_waited = 0;
        config.pending_screenshots += 1;
    }
}

pub fn parse_test_args() -> Option<ScreenshotTestConfig> {
    let args: Vec<String> = std::env::args().collect();
    
    let mut output_dir = PathBuf::from("screenshots");
    let mut scenarios = Vec::new();
    let mut frames_to_wait = 10;
    
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
                    frames_to_wait = args[i].parse().unwrap_or(10);
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
    })
}

pub fn default_test_scenarios() -> Vec<ScreenshotScenario> {
    vec![
        ScreenshotScenario {
            name: "orbit_far_0deg".to_string(),
            camera_yaw: 0.0,
            camera_pitch: 0.3,
            camera_distance: 150000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "orbit_far_90deg".to_string(),
            camera_yaw: std::f32::consts::FRAC_PI_2,
            camera_pitch: 0.3,
            camera_distance: 150000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "orbit_far_180deg".to_string(),
            camera_yaw: std::f32::consts::PI,
            camera_pitch: 0.3,
            camera_distance: 150000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "orbit_far_270deg".to_string(),
            camera_yaw: std::f32::consts::PI * 1.5,
            camera_pitch: 0.3,
            camera_distance: 150000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "orbit_mid_0deg".to_string(),
            camera_yaw: 0.0,
            camera_pitch: 0.3,
            camera_distance: 70000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "orbit_mid_90deg".to_string(),
            camera_yaw: std::f32::consts::FRAC_PI_2,
            camera_pitch: 0.3,
            camera_distance: 70000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "orbit_close_0deg".to_string(),
            camera_yaw: 0.0,
            camera_pitch: 0.3,
            camera_distance: 45000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "orbit_close_45deg".to_string(),
            camera_yaw: std::f32::consts::FRAC_PI_4,
            camera_pitch: 0.3,
            camera_distance: 45000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "orbit_very_close_0deg".to_string(),
            camera_yaw: 0.0,
            camera_pitch: 0.3,
            camera_distance: 38000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "top_down".to_string(),
            camera_yaw: 0.0,
            camera_pitch: 1.5,
            camera_distance: 100000.0,
            camera_target: Vec3::ZERO,
        },
        ScreenshotScenario {
            name: "equator_view".to_string(),
            camera_yaw: 0.0,
            camera_pitch: 0.0,
            camera_distance: 60000.0,
            camera_target: Vec3::ZERO,
        },
    ]
}
