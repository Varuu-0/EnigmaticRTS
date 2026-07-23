//! Runtime performance diagnostics shared by interactive, screenshot, and benchmark modes.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use bevy::diagnostic::{
    DiagnosticsStore, EntityCountDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::prelude::*;
use bevy::render::diagnostic::MeshAllocatorDiagnosticPlugin;
use bevy::tasks::{futures::check_ready, IoTaskPool, Task};

use er_terrain::TerrainDebugInfo;

use er_game::gpu_telemetry::{self, GpuTelemetryStatus};

use crate::frame_timing::MainWorldFrameTimings;

const FRAME_HISTORY_CAPACITY: usize = 600;
const SYSTEM_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// Latest normalized measurements for the current run.
///
/// Values from the Bevy system-information plugin are sampled asynchronously,
/// so `None` means the platform has not provided a measurement yet.
#[derive(Clone, Debug, Default, Resource)]
pub struct PerformanceSnapshot {
    pub sample_revision: u64,
    pub rolling_fps: f32,
    pub frame_ms: f32,
    pub frame_p50_ms: f32,
    pub frame_p95_ms: f32,
    pub frame_p99_ms: f32,
    pub one_percent_low_fps: f32,
    pub hitch_16ms_count: u64,
    pub hitch_33ms_count: u64,
    pub hitch_50ms_count: u64,
    /// Time from a newly observed keyboard or mouse-button input until the end
    /// of the main-world frame. This excludes render queueing, present, and
    /// display scanout; it is not input-to-photon latency.
    pub input_to_cpu_frame_end_ms: Option<f32>,
    pub process_cpu_percent: Option<f64>,
    pub system_cpu_percent: Option<f64>,
    /// Resident process memory reported by Bevy's sysinfo plugin in GiB.
    pub process_memory_gib: Option<f64>,
    /// System-wide used physical memory percentage.
    pub system_memory_percent: Option<f64>,
    pub entity_count: Option<f64>,
    pub terrain_mesh_bytes: usize,
    pub terrain_meshes_built: usize,
    /// GPU slab allocation bytes reported by Bevy's mesh allocator.
    pub mesh_allocator_bytes: Option<f64>,
    pub mesh_allocator_allocations: Option<f64>,
    /// Visible mesh entities. This is a draw-work estimate, not an exact GPU
    /// draw-call count because Bevy may batch or multi-draw these entities.
    pub visible_mesh_draw_estimate: usize,
    /// CPU time spent recording Bevy's primary opaque 3D render pass.
    pub opaque_render_cpu_ms: Option<f64>,
    /// GPU elapsed time for the primary opaque 3D render pass when timestamp
    /// queries were requested with `--gpu-diagnostics` and are supported.
    pub opaque_render_gpu_ms: Option<f64>,
    /// Per-pass render-thread CPU timings reported by Bevy's render diagnostics.
    /// These overlap the main-world schedules and are not additive frame time.
    pub render_cpu_spans: Vec<(String, f64)>,
    pub gpu_vram_usage_bytes: Option<u64>,
    pub gpu_vram_budget_bytes: Option<u64>,
    pub gpu_name: Option<String>,
    pub gpu_telemetry_available: bool,
}

#[derive(Resource)]
struct PerformanceHistory {
    frame_ms: VecDeque<f32>,
    sorted_frame_ms: Vec<f32>,
    hitch_16ms_count: u64,
    hitch_33ms_count: u64,
    hitch_50ms_count: u64,
    frames_until_refresh: u8,
    last_timing_revision: u64,
}

#[derive(Resource)]
struct GpuTelemetryWorker {
    task: Option<Task<gpu_telemetry::GpuTelemetrySample>>,
    last_started: Instant,
}

#[derive(Default, Resource)]
struct InputLatencyProbe {
    observed_at: Option<Instant>,
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self {
            frame_ms: VecDeque::with_capacity(FRAME_HISTORY_CAPACITY),
            sorted_frame_ms: Vec::with_capacity(FRAME_HISTORY_CAPACITY),
            hitch_16ms_count: 0,
            hitch_33ms_count: 0,
            hitch_50ms_count: 0,
            frames_until_refresh: 0,
            last_timing_revision: 0,
        }
    }
}

impl Default for GpuTelemetryWorker {
    fn default() -> Self {
        Self {
            task: None,
            last_started: Instant::now() - SYSTEM_SAMPLE_INTERVAL,
        }
    }
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PerformanceSnapshotUpdate;

pub struct PerformanceDiagnosticsPlugin;

impl Plugin for PerformanceDiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            EntityCountDiagnosticsPlugin::default(),
            SystemInformationDiagnosticsPlugin,
            MeshAllocatorDiagnosticPlugin,
        ))
        .insert_resource(PerformanceSnapshot::default())
        .insert_resource(PerformanceHistory::default())
        .insert_resource(InputLatencyProbe::default())
        .insert_resource(GpuTelemetryWorker::default())
        .add_systems(First, observe_input)
        .add_systems(
            PostUpdate,
            (poll_gpu_telemetry, update_performance_snapshot).in_set(PerformanceSnapshotUpdate),
        );
    }
}

fn update_performance_snapshot(
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time>,
    terrain: Res<TerrainDebugInfo>,
    main_world_timings: Option<Res<MainWorldFrameTimings>>,
    mut input_probe: ResMut<InputLatencyProbe>,
    mut history: ResMut<PerformanceHistory>,
    mut snapshot: ResMut<PerformanceSnapshot>,
) {
    let frame_ms = time.delta_secs() * 1000.0;
    push_frame_sample(&mut history, frame_ms);

    let published_frame_ms = if let Some(timings) = main_world_timings.as_ref() {
        if timings.sample_revision == 0 || timings.sample_revision == history.last_timing_revision {
            return;
        }
        history.last_timing_revision = timings.sample_revision;
        timings
            .frame_duration
            .map(|duration| duration.as_secs_f32() * 1000.0)
            .unwrap_or(frame_ms)
    } else {
        if history.frames_until_refresh > 0 {
            history.frames_until_refresh -= 1;
            return;
        }
        history.frames_until_refresh = 7;
        frame_ms
    };

    let PerformanceHistory {
        frame_ms,
        sorted_frame_ms,
        ..
    } = &mut *history;
    sorted_frame_ms.clear();
    sorted_frame_ms.extend(frame_ms.iter().copied());
    sorted_frame_ms.sort_by(|a, b| a.total_cmp(b));
    snapshot.frame_p50_ms = percentile_sorted(&history.sorted_frame_ms, 50.0);
    snapshot.frame_p95_ms = percentile_sorted(&history.sorted_frame_ms, 95.0);
    snapshot.frame_p99_ms = percentile_sorted(&history.sorted_frame_ms, 99.0);
    snapshot.one_percent_low_fps = one_percent_low_fps_sorted(&history.sorted_frame_ms);
    snapshot.sample_revision = snapshot.sample_revision.wrapping_add(1);
    snapshot.rolling_fps = rolling_fps(&history.frame_ms);
    snapshot.frame_ms = published_frame_ms;
    snapshot.hitch_16ms_count = history.hitch_16ms_count;
    snapshot.hitch_33ms_count = history.hitch_33ms_count;
    snapshot.hitch_50ms_count = history.hitch_50ms_count;
    snapshot.input_to_cpu_frame_end_ms = input_probe
        .observed_at
        .take()
        .map(|observed_at| observed_at.elapsed().as_secs_f32() * 1000.0);
    snapshot.process_cpu_percent = diagnostic_value(
        &diagnostics,
        &SystemInformationDiagnosticsPlugin::PROCESS_CPU_USAGE,
    );
    snapshot.system_cpu_percent = diagnostic_value(
        &diagnostics,
        &SystemInformationDiagnosticsPlugin::SYSTEM_CPU_USAGE,
    );
    snapshot.process_memory_gib = diagnostic_value(
        &diagnostics,
        &SystemInformationDiagnosticsPlugin::PROCESS_MEM_USAGE,
    );
    snapshot.system_memory_percent = diagnostic_value(
        &diagnostics,
        &SystemInformationDiagnosticsPlugin::SYSTEM_MEM_USAGE,
    );
    snapshot.entity_count =
        diagnostic_value(&diagnostics, &EntityCountDiagnosticsPlugin::ENTITY_COUNT);
    snapshot.terrain_mesh_bytes = terrain.estimated_mesh_bytes;
    snapshot.terrain_meshes_built = terrain.meshes_built;
    snapshot.mesh_allocator_bytes = diagnostic_value(
        &diagnostics,
        MeshAllocatorDiagnosticPlugin::slabs_size_diagnostic_path(),
    );
    snapshot.mesh_allocator_allocations = diagnostic_value(
        &diagnostics,
        MeshAllocatorDiagnosticPlugin::allocations_diagnostic_path(),
    );
    snapshot.visible_mesh_draw_estimate = terrain.visible_chunks;
    snapshot.opaque_render_cpu_ms = render_diagnostic_value(&diagnostics, "elapsed_cpu");
    snapshot.opaque_render_gpu_ms = render_diagnostic_value(&diagnostics, "elapsed_gpu");
    snapshot.render_cpu_spans = render_cpu_spans(&diagnostics);
}

fn observe_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut probe: ResMut<InputLatencyProbe>,
) {
    if keys.get_just_pressed().next().is_some() || mouse_buttons.get_just_pressed().next().is_some()
    {
        probe.observed_at = Some(Instant::now());
    }
}

fn render_diagnostic_value(diagnostics: &DiagnosticsStore, metric: &str) -> Option<f64> {
    diagnostics.iter().find_map(|diagnostic| {
        let path = diagnostic.path().as_str();
        (path.contains("main_opaque_pass_3d") && path.ends_with(metric))
            .then(|| diagnostic.value())
            .flatten()
    })
}

fn render_cpu_spans(diagnostics: &DiagnosticsStore) -> Vec<(String, f64)> {
    let mut spans: Vec<_> = diagnostics
        .iter()
        .filter_map(|diagnostic| {
            let path = diagnostic.path().as_str();
            (path.starts_with("render/") && path.ends_with("/elapsed_cpu"))
                .then(|| diagnostic.value().map(|value| (path.to_owned(), value)))
                .flatten()
        })
        .collect();
    spans.sort_by(|left, right| right.1.total_cmp(&left.1));
    spans
}

fn diagnostic_value(
    diagnostics: &DiagnosticsStore,
    path: &bevy::diagnostic::DiagnosticPath,
) -> Option<f64> {
    diagnostics
        .get(path)
        .and_then(|diagnostic| diagnostic.smoothed().or(diagnostic.value()))
}

fn poll_gpu_telemetry(
    mut worker: ResMut<GpuTelemetryWorker>,
    mut snapshot: ResMut<PerformanceSnapshot>,
) {
    if let Some(task) = worker.task.as_mut() {
        if let Some(gpu) = check_ready(task) {
            update_gpu_snapshot(&mut snapshot, gpu);
            worker.task = None;
        }
    }

    if worker.task.is_none() && worker.last_started.elapsed() >= SYSTEM_SAMPLE_INTERVAL {
        worker.task = Some(IoTaskPool::get().spawn(async { gpu_telemetry::sample() }));
        worker.last_started = Instant::now();
    }
}

fn update_gpu_snapshot(snapshot: &mut PerformanceSnapshot, gpu: gpu_telemetry::GpuTelemetrySample) {
    snapshot.gpu_telemetry_available = matches!(gpu.status, GpuTelemetryStatus::Available);
    snapshot.gpu_name = (!gpu.description.is_empty()).then_some(gpu.description);
    snapshot.gpu_vram_usage_bytes = (gpu.vram_usage_bytes > 0).then_some(gpu.vram_usage_bytes);
    snapshot.gpu_vram_budget_bytes = (gpu.vram_budget_bytes > 0).then_some(gpu.vram_budget_bytes);
}

fn rolling_fps(frame_ms: &VecDeque<f32>) -> f32 {
    if frame_ms.is_empty() {
        return 0.0;
    }
    let average_frame_ms = frame_ms.iter().sum::<f32>() / frame_ms.len() as f32;
    1000.0 / average_frame_ms.max(0.001)
}

fn push_frame_sample(history: &mut PerformanceHistory, frame_ms: f32) {
    if history.frame_ms.len() == FRAME_HISTORY_CAPACITY {
        history.frame_ms.pop_front();
    }
    history.frame_ms.push_back(frame_ms);
    if frame_ms > 16.7 {
        history.hitch_16ms_count += 1;
    }
    if frame_ms > 33.3 {
        history.hitch_33ms_count += 1;
    }
    if frame_ms > 50.0 {
        history.hitch_50ms_count += 1;
    }
}

fn percentile_sorted(sorted: &[f32], percentile: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((percentile / 100.0) * (sorted.len() as f32 - 1.0)).round() as usize;
    sorted[index]
}

fn one_percent_low_fps_sorted(sorted: &[f32]) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let sample_count = ((sorted.len() as f32 * 0.01).ceil() as usize).max(1);
    let average_frame_ms =
        sorted.iter().rev().take(sample_count).sum::<f32>() / sample_count as f32;
    1000.0 / average_frame_ms.max(0.001)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_ranked_frame_samples() {
        let mut history = PerformanceHistory::default();
        for frame_ms in [10.0, 20.0, 30.0, 40.0, 50.0] {
            push_frame_sample(&mut history, frame_ms);
        }
        history
            .sorted_frame_ms
            .extend(history.frame_ms.iter().copied());
        history.sorted_frame_ms.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(percentile_sorted(&history.sorted_frame_ms, 50.0), 30.0);
        assert_eq!(percentile_sorted(&history.sorted_frame_ms, 95.0), 50.0);
    }

    #[test]
    fn one_percent_low_uses_slowest_frame_samples() {
        let mut history = PerformanceHistory::default();
        for frame_ms in [10.0, 20.0, 40.0, 100.0] {
            push_frame_sample(&mut history, frame_ms);
        }
        history
            .sorted_frame_ms
            .extend(history.frame_ms.iter().copied());
        history.sorted_frame_ms.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(one_percent_low_fps_sorted(&history.sorted_frame_ms), 10.0);
    }

    #[test]
    fn rolling_fps_uses_average_frame_time() {
        let mut history = PerformanceHistory::default();
        push_frame_sample(&mut history, 10.0);
        push_frame_sample(&mut history, 30.0);
        assert_eq!(rolling_fps(&history.frame_ms), 50.0);
    }

    #[test]
    fn hitches_are_classified_by_threshold() {
        let mut history = PerformanceHistory::default();
        push_frame_sample(&mut history, 20.0);
        push_frame_sample(&mut history, 40.0);
        push_frame_sample(&mut history, 60.0);
        assert_eq!(history.hitch_16ms_count, 3);
        assert_eq!(history.hitch_33ms_count, 2);
        assert_eq!(history.hitch_50ms_count, 1);
    }
}
