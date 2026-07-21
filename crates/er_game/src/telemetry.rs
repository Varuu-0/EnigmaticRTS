//! Scenario Telemetry evidence written next to deterministic captures.

use er_terrain::{
    CameraWorldPosition, FrameProfiler, RenderOrigin, TerrainDebugInfo, TerrainState,
};
use serde::Serialize;
use std::{io, path::Path};

use crate::{
    diagnostics::PerformanceSnapshot,
    screenshot_test::{ScreenshotScenario, ScreenshotTestConfig},
};

#[derive(Serialize)]
pub struct ScenarioTelemetry {
    format: &'static str,
    scenario: String,
    seed: u64,
    camera: CameraTelemetry,
    camera_world: CameraWorldTelemetry,
    origin: OriginTelemetry,
    capture: CaptureTelemetry,
    performance: PerformanceTelemetry,
    terrain: TerrainTelemetry,
    source_mode: String,
}

#[derive(Serialize)]
struct CameraTelemetry {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

#[derive(Serialize)]
struct CameraWorldTelemetry {
    x: f64,
    y: f64,
    z: f64,
    altitude_above_terrain_m: f64,
    target_altitude_requested: Option<f32>,
}

#[derive(Serialize)]
struct OriginTelemetry {
    world_x: f64,
    world_y: f64,
    world_z: f64,
    generation: u64,
    cell_size_m: f64,
}

#[derive(Serialize)]
struct CaptureTelemetry {
    frames_waited: u32,
    settled_frames: u32,
    timed_out: bool,
}

#[derive(Serialize)]
struct PerformanceTelemetry {
    frame_ms: f32,
    frame_p50_ms: f32,
    frame_p95_ms: f32,
    frame_p99_ms: f32,
    one_percent_low_fps: f32,
    process_cpu_percent: Option<f64>,
    process_memory_gib: Option<f64>,
    opaque_render_cpu_ms: Option<f64>,
    opaque_render_gpu_ms: Option<f64>,
    gpu_vram_usage_bytes: Option<u64>,
    gpu_vram_budget_bytes: Option<u64>,
    mesh_allocator_bytes: Option<f64>,
    visible_mesh_draw_estimate: usize,
}

#[derive(Serialize)]
struct TerrainTelemetry {
    active_chunks: usize,
    visible_chunks: usize,
    max_depth: u8,
    pending_splits: usize,
    pending_merges: usize,
    pending_meshes: usize,
    meshes_built: usize,
    estimated_mesh_bytes: usize,
    nearest_chunk_lod: u8,
    nearest_chunk_width_m: f64,
    nearest_vertex_spacing_m: f64,
    normal_diff_spacing_m: f64,
    normal_difference_span_m: f64,
    normal_diff_epsilon_radians: f64,
    procedural_source_coverage_percent: f32,
    learned_source_coverage_percent: f32,
    cross_generation_mesh_attaches: usize,
    terrain_step_ms: Vec<(String, f64)>,
}

impl ScenarioTelemetry {
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        scenario: &ScreenshotScenario,
        config: &ScreenshotTestConfig,
        timed_out: bool,
        terrain: &TerrainState,
        debug: &TerrainDebugInfo,
        camera_world: &CameraWorldPosition,
        origin: &RenderOrigin,
        performance: &PerformanceSnapshot,
        profiler: &FrameProfiler,
    ) -> Self {
        let altitude = camera_world.0.length()
            - terrain.planet_radius
            - terrain.field.sample(camera_world.0.normalize()).elevation
                * terrain.elevation_scale as f64;
        Self {
            format: "enigmatic-rts-scenario-telemetry/v2",
            scenario: scenario.name.clone(),
            seed: config.fixed_seed,
            camera: CameraTelemetry {
                yaw: scenario.camera_yaw,
                pitch: scenario.camera_pitch,
                distance: scenario.camera_distance,
            },
            camera_world: CameraWorldTelemetry {
                x: camera_world.0.x,
                y: camera_world.0.y,
                z: camera_world.0.z,
                altitude_above_terrain_m: altitude,
                target_altitude_requested: scenario.target_altitude_m,
            },
            origin: OriginTelemetry {
                world_x: origin.world.x,
                world_y: origin.world.y,
                world_z: origin.world.z,
                generation: origin.generation,
                cell_size_m: origin.cell_size_m,
            },
            capture: CaptureTelemetry {
                frames_waited: config.frames_waited,
                settled_frames: config.settled_frames,
                timed_out,
            },
            performance: PerformanceTelemetry {
                frame_ms: performance.frame_ms,
                frame_p50_ms: performance.frame_p50_ms,
                frame_p95_ms: performance.frame_p95_ms,
                frame_p99_ms: performance.frame_p99_ms,
                one_percent_low_fps: performance.one_percent_low_fps,
                process_cpu_percent: performance.process_cpu_percent,
                process_memory_gib: performance.process_memory_gib,
                opaque_render_cpu_ms: performance.opaque_render_cpu_ms,
                opaque_render_gpu_ms: performance.opaque_render_gpu_ms,
                gpu_vram_usage_bytes: performance.gpu_vram_usage_bytes,
                gpu_vram_budget_bytes: performance.gpu_vram_budget_bytes,
                mesh_allocator_bytes: performance.mesh_allocator_bytes,
                visible_mesh_draw_estimate: performance.visible_mesh_draw_estimate,
            },
            terrain: TerrainTelemetry {
                active_chunks: debug.active_chunks,
                visible_chunks: debug.visible_chunks,
                max_depth: debug.max_depth,
                pending_splits: debug.pending_splits,
                pending_merges: debug.pending_merges,
                pending_meshes: debug.pending_meshes,
                meshes_built: debug.meshes_built,
                estimated_mesh_bytes: debug.estimated_mesh_bytes,
                nearest_chunk_lod: debug.nearest_chunk_lod,
                nearest_chunk_width_m: debug.nearest_chunk_width_m,
                nearest_vertex_spacing_m: debug.vertex_spacing_m,
                normal_diff_spacing_m: debug.normal_diff_spacing_m,
                normal_difference_span_m: debug.normal_difference_span_m,
                normal_diff_epsilon_radians: debug.normal_diff_epsilon_radians,
                procedural_source_coverage_percent: debug.procedural_source_coverage_percent,
                learned_source_coverage_percent: debug.learned_source_coverage_percent,
                cross_generation_mesh_attaches: debug.cross_generation_mesh_attaches,
                terrain_step_ms: profiler
                    .timings
                    .iter()
                    .map(|(name, duration)| (name.to_string(), duration.as_secs_f64() * 1000.0))
                    .collect(),
            },
            source_mode: format!("{:?}", terrain.source_mode),
        }
    }
}

pub fn write_scenario_telemetry(path: &Path, telemetry: ScenarioTelemetry) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(&telemetry).map_err(io::Error::other)?;
    std::fs::write(path, bytes)
}
