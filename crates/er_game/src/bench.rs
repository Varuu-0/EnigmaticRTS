//! Headless/hidden-window benchmark mode.
//!
//! Moves the main OrbitCamera through a series of distances (far → surface),
//! collects per-frame durations and TerrainDebugInfo, writes a concise text
//! report to `--bench-output` (or stdout), and exits successfully.
//!
//! CLI flags:
//!   --bench                 enable benchmark mode
//!   --bench-output <path>   write results to this file (default: stdout)
//!   --bench-warmup <frames> warmup frames per scenario (default: 30)
//!   --bench-measure <frames> measured frames per scenario (default: 60)

use std::path::PathBuf;

use bevy::prelude::*;

use er_terrain::{TerrainDebugInfo, TerrainUpdate};

use crate::camera::OrbitCamera;

const DEFAULT_WARMUP: u32 = 30;
const DEFAULT_MEASURE: u32 = 60;

const SCENARIO_DISTANCES: &[(&str, f32)] = &[
    ("far", 150000.0),
    ("mid", 70000.0),
    ("close", 45000.0),
    ("very_close", 38000.0),
    ("surface", 37000.0),
];

#[derive(Resource)]
pub struct BenchConfig {
    pub output_path: Option<PathBuf>,
    pub warmup_frames: u32,
    pub measure_frames: u32,
    pub scenarios: Vec<BenchScenario>,
    pub current_index: usize,
    pub frames_in_phase: u32,
    pub phase: BenchPhase,
    pub frame_durations: Vec<f32>,
    pub results: Vec<BenchResult>,
    pub completed: bool,
}

#[derive(Clone)]
pub struct BenchScenario {
    pub name: String,
    pub distance: f32,
}

#[derive(Clone)]
pub struct BenchResult {
    pub name: String,
    pub distance: f32,
    pub durations: Vec<f32>,
    pub debug: BenchDebugSnapshot,
}

#[derive(Clone, Copy)]
pub struct BenchDebugSnapshot {
    pub active_chunks: usize,
    pub max_depth: u8,
    pub pending_splits: usize,
    pub pending_merges: usize,
    pub pending_meshes: usize,
    pub visible_chunks: usize,
    pub frame_time_ms: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BenchPhase {
    Warmup,
    Measure,
}

pub struct BenchPlugin;

impl Plugin for BenchPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, run_benchmark.after(TerrainUpdate));
    }
}

pub fn parse_bench_args() -> Option<BenchConfig> {
    let args: Vec<String> = std::env::args().collect();
    let mut bench = false;
    let mut output_path: Option<PathBuf> = None;
    let mut warmup = DEFAULT_WARMUP;
    let mut measure = DEFAULT_MEASURE;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bench" => {
                bench = true;
            }
            "--bench-output" => {
                i += 1;
                if i < args.len() {
                    output_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--bench-warmup" => {
                i += 1;
                if i < args.len() {
                    warmup = args[i].parse().unwrap_or(DEFAULT_WARMUP);
                }
            }
            "--bench-measure" => {
                i += 1;
                if i < args.len() {
                    measure = args[i].parse().unwrap_or(DEFAULT_MEASURE);
                }
            }
            _ => {}
        }
        i += 1;
    }

    if !bench {
        return None;
    }

    let scenarios = SCENARIO_DISTANCES
        .iter()
        .map(|(name, dist)| BenchScenario {
            name: name.to_string(),
            distance: *dist,
        })
        .collect();

    Some(BenchConfig {
        output_path,
        warmup_frames: warmup,
        measure_frames: measure,
        scenarios,
        current_index: 0,
        frames_in_phase: 0,
        phase: BenchPhase::Warmup,
        frame_durations: Vec::new(),
        results: Vec::new(),
        completed: false,
    })
}

fn run_benchmark(
    mut config: ResMut<BenchConfig>,
    time: Res<Time>,
    mut camera_query: Query<(&mut OrbitCamera, &mut Transform), With<Camera3d>>,
    debug_info: Res<TerrainDebugInfo>,
) {
    if config.completed {
        return;
    }

    if config.current_index >= config.scenarios.len() {
        let report = generate_report(&config);
        write_report(&config, &report);
        config.completed = true;
        info!("Benchmark completed.\n{}", report);
        std::process::exit(0);
    }

    let scenario_name = config.scenarios[config.current_index].name.clone();
    let scenario_dist = config.scenarios[config.current_index].distance;
    let dist = scenario_dist;

    // Set camera position at the start of each scenario (frame 0 of warmup).
    if config.frames_in_phase == 0 && config.phase == BenchPhase::Warmup {
        info!(
            "Benchmark scenario: {} (distance={})",
            scenario_name, dist
        );
        if let Ok((mut orbit, mut transform)) = camera_query.single_mut() {
            orbit.distance = dist;
            orbit.smoothed_distance = dist;
            orbit.target = Vec3::ZERO;
            orbit.smoothed_target = Vec3::ZERO;

            let cp = orbit.pitch.cos();
            let direction = Vec3::new(
                cp * orbit.yaw.sin(),
                orbit.pitch.sin(),
                cp * orbit.yaw.cos(),
            )
            .normalize();

            transform.translation = orbit.target + direction * dist;
            let up = if direction.abs().dot(Vec3::Y) > 0.999 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            transform.look_at(orbit.target, up);
        }
    }

    let dt_ms = time.delta_secs() * 1000.0;

    match config.phase {
        BenchPhase::Warmup => {
            config.frames_in_phase += 1;
            if config.frames_in_phase >= config.warmup_frames {
                config.phase = BenchPhase::Measure;
                config.frames_in_phase = 0;
                config.frame_durations.clear();
            }
        }
        BenchPhase::Measure => {
            config.frame_durations.push(dt_ms);
            config.frames_in_phase += 1;
            if config.frames_in_phase >= config.measure_frames {
                let durations = std::mem::take(&mut config.frame_durations);
                let avg = durations.iter().sum::<f32>() / durations.len().max(1) as f32;
                info!(
                    "Scenario {} measured: {} frames, avg {:.2}ms",
                    scenario_name,
                    durations.len(),
                    avg,
                );
                config.results.push(BenchResult {
                    name: scenario_name,
                    distance: scenario_dist,
                    durations,
                    debug: BenchDebugSnapshot {
                        active_chunks: debug_info.active_chunks,
                        max_depth: debug_info.max_depth,
                        pending_splits: debug_info.pending_splits,
                        pending_merges: debug_info.pending_merges,
                        pending_meshes: debug_info.pending_meshes,
                        visible_chunks: debug_info.visible_chunks,
                        frame_time_ms: debug_info.frame_time_ms,
                    },
                });
                config.current_index += 1;
                config.phase = BenchPhase::Warmup;
                config.frames_in_phase = 0;
            }
        }
    }
}

fn generate_report(config: &BenchConfig) -> String {
    let mut lines = Vec::new();
    lines.push("EnigmaticRTS Benchmark Report".to_string());
    lines.push("============================".to_string());
    lines.push(format!(
        "Warmup: {} frames | Measure: {} frames per scenario",
        config.warmup_frames, config.measure_frames
    ));
    lines.push(String::new());
    lines.push(format!(
        "{:<12} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6}",
        "Scenario", "avg_ms", "min_ms", "max_ms", "p95_ms", "fps", "chunks", "depth"
    ));
    lines.push("-".repeat(74));

    for result in &config.results {
        let avg = result.durations.iter().sum::<f32>() / result.durations.len().max(1) as f32;
        let min = result.durations.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result.durations.iter().cloned().fold(0.0_f32, f32::max);
        let p95 = percentile(&result.durations, 95.0);
        let fps = 1000.0 / avg.max(0.001);

        lines.push(format!(
            "{:<12} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>6.1} {:>6} {:>6}",
            result.name,
            avg,
            min,
            max,
            p95,
            fps,
            result.debug.active_chunks,
            result.debug.max_depth,
        ));
    }

    lines.push(String::new());
    lines.push("TerrainDebugInfo at end of each scenario:".to_string());
    for result in &config.results {
        lines.push(format!(
            "  {:<12} dist={} active={} depth={} visible={} pending(s/m/mesh)={}/{}/{} frame_time_ms={:.2}",
            result.name,
            result.distance,
            result.debug.active_chunks,
            result.debug.max_depth,
            result.debug.visible_chunks,
            result.debug.pending_splits,
            result.debug.pending_merges,
            result.debug.pending_meshes,
            result.debug.frame_time_ms,
        ));
    }

    lines.push(String::new());
    lines.join("\n")
}

fn percentile(data: &[f32], pct: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((pct / 100.0) * (sorted.len() as f32 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn write_report(config: &BenchConfig, report: &str) {
    if let Some(path) = &config.output_path {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, report);
    }
}
