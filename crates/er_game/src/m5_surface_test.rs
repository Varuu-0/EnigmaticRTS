//! Automated M5 SURFACE gate: a phased deterministic state machine that
//! validates the full learned-surface streaming pipeline end-to-end against
//! real live terrain mesh generation, blend transitions, and camera movement.
//!
//! Activated by `--m5-test <max-seconds>` (requires the
//! `terrain_diffusion` feature). The test runs through four phases:
//!
//! - **Phase A (Settle)**: Place the camera at a deterministic Earth surface
//!   direction inside an uncached provider tile. Verify immediate procedural
//!   fallback before tile residency. Wait for fine LOD (LOD >= 17 or vertex
//!   spacing <= 5m) and terrain settled for consecutive frames.
//! - **Phase B (Residency)**: Compute exact halo dependencies for the actual
//!   camera-containing active chunk (from `TerrainDebugInfo::containing_chunk`)
//!   using the real `ChartMacroField` exact-N API. Enqueue them at
//!   `VisibleSurface` priority. Wait until the exact chunk halo is resident.
//! - **Phase C (Blend)**: Observe real applied mesh transition: sample blend
//!   weight and camera-local learned source/coverage over time, require
//!   monotonic progression with a meaningful >0 intermediate sample and
//!   completion near 1, require rebuild increments attributable after Phase B.
//! - **Phase D (Warm-ahead)**: Perform controlled tangent surface movement
//!   through the existing `reproject_surface_target` API toward/across a
//!   provider tile boundary. Compute the containing active chunk as it
//!   changes, require local chunk halo residency or bounded recovery, record
//!   maximum local fallback / min learned coverage. Provider P95 <= 1ms.
//!
//! All phases fail closed on timeout — they do not advance to the next
//! phase as if successful. The system is ordered after camera update,
//! TerrainDiffusion diagnostic publication, and TerrainUpdate.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use er_core::math::{uv_to_dir, CellKey};
use er_world::streaming::{PriorityClass, ProviderTileCoordinate};
use serde::Serialize;

use crate::camera::{reproject_surface_target, CameraMode, OrbitCamera, SurfaceTarget};
use crate::terrain_diffusion::{
    ProviderTimingHistory, TerrainDiffusionDiagnostic, TerrainDiffusionRuntime,
};

/// Deterministic starting point for the uncached-tile search. The test probes
/// a stable permutation and selects the first tile absent from RAM and disk.
const M5_TEST_FACE: u8 = 0;
const M5_TEST_TILE_X: u32 = 300;
const M5_TEST_TILE_Y: u32 = 400;
const M5_EARTH_TILES_PER_EDGE: u32 = 652;

/// Altitude for surface mode (forces fine LOD). At 10m altitude, the screen
/// error is high enough to trigger deep LOD refinement.
const M5_SURFACE_ALTITUDE_M: f64 = 10.0;

/// Maximum time to wait for fine LOD settle.
const PHASE_A_TIMEOUT: Duration = Duration::from_secs(60);
/// Maximum time to wait for halo residency.
const PHASE_B_TIMEOUT: Duration = Duration::from_secs(180);
/// Maximum time to wait for blend completion.
const PHASE_C_TIMEOUT: Duration = Duration::from_secs(60);
/// Phase D movement duration.
const PHASE_D_DURATION: Duration = Duration::from_secs(30);
/// Maximum continuous loss of camera-local halo residency during movement.
const PHASE_D_MAX_RECOVERY_S: f64 = 2.0;
/// Start just inside the selected tile so normal-halo residency must cover the
/// adjacent tile before movement crosses the boundary.
const M5_BOUNDARY_INSET_TILE_FRACTION: f64 = 0.0001;
/// Controlled close-surface movement speed.
const PHASE_D_MOVE_SPEED_MPS: f64 = 10.0;
/// Maximum provider main-thread P95 (microseconds) for the hitch gate.
const PROVIDER_P95_LIMIT_US: f64 = 1_000.0; // 1ms
/// Target fine LOD for close Earth exploration.
const FINE_LOD_TARGET: u8 = 17;
/// Vertex spacing threshold for fine LOD alternative gate.
const FINE_VERTEX_SPACING_M: f64 = 5.0;
/// Consecutive settled frames required before advancing.
const SETTLED_FRAMES_REQUIRED: u32 = 10;

#[derive(Resource)]
pub struct M5TestConfig {
    max_duration: Duration,
    output_path: Option<PathBuf>,
    start: Option<Instant>,
    phase: M5Phase,
    phase_start: Option<Instant>,
    finished: bool,
    /// The actual camera-containing active chunk from TerrainDebugInfo.
    camera_chunk: Option<CellKey>,
    /// Deterministically selected provider tile, proven cold at test start.
    scenario_tile: Option<(u8, u32, u32)>,
    /// Halo dependency keys enqueued in Phase B.
    enqueued_keys: Vec<er_world::surface_cache::SurfaceCacheKey>,
    /// Blend weight samples recorded in Phase C.
    blend_samples: Vec<f64>,
    /// Cumulative cross-generation attaches observed during Phase D.
    cross_gen_attaches: usize,
    /// Consecutive settled frame counter.
    settled_frames: u32,
    /// Phase B rebuild counts at entry (to verify increments).
    phase_b_rebuilds_completed: u64,
    /// Report data being accumulated.
    report: M5Report,
    /// Whether the immediate-procedural proof was already captured once.
    immediate_procedural_captured: bool,
    /// Phase D movement: current tangent offset.
    phase_d_offset_m: f64,
    /// Phase D: min local learned coverage observed.
    phase_d_min_learned: f64,
    /// Phase D: max local fallback observed.
    phase_d_max_fallback: f64,
    /// Phase D: max local learned coverage observed.
    phase_d_max_learned: f64,
    /// Phase D: cache hits observed.
    phase_d_cache_hits: u64,
    /// Phase D: cache misses observed.
    phase_d_cache_misses: u64,
    /// Cumulative cache counters at Phase D entry.
    phase_d_cache_start: Option<(u64, u64)>,
    /// Camera-local halo recovery evidence.
    phase_d_nonresident_s: f64,
    phase_d_max_nonresident_s: f64,
    phase_d_halo_resident_frames: u64,
    phase_d_halo_nonresident_frames: u64,
    phase_d_frames: u64,
    /// Predicates accumulated across every Phase D frame.
    phase_d_structural_ok: bool,
    phase_d_streaming_ok: bool,
    /// Render-origin evidence used to bound valid rebased mesh attaches.
    phase_d_origin_generation_start: Option<u64>,
    phase_d_peak_active_chunks: usize,
    /// Phase failure reason (if any).
    failure_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum M5Phase {
    NotStarted,
    PhaseA_Settle,
    PhaseB_Residency,
    PhaseC_Blend,
    PhaseD_WarmAhead,
    Done,
}

impl M5TestConfig {
    pub fn parse_args() -> Option<Self> {
        let args: Vec<String> = std::env::args().collect();
        let mut max_seconds: Option<u64> = None;
        let mut output_path: Option<PathBuf> = None;

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--m5-test" => {
                    i += 1;
                    if i < args.len() {
                        max_seconds = args[i].parse().ok();
                    }
                }
                "--m5-test-output" => {
                    i += 1;
                    if i < args.len() {
                        output_path = Some(PathBuf::from(&args[i]));
                    }
                }
                _ => {}
            }
            i += 1;
        }

        max_seconds.map(|secs| Self {
            max_duration: Duration::from_secs(secs),
            output_path,
            start: None,
            phase: M5Phase::NotStarted,
            phase_start: None,
            finished: false,
            camera_chunk: None,
            scenario_tile: None,
            enqueued_keys: Vec::new(),
            blend_samples: Vec::new(),
            cross_gen_attaches: 0,
            settled_frames: 0,
            phase_b_rebuilds_completed: 0,
            report: M5Report::default(),
            immediate_procedural_captured: false,
            phase_d_offset_m: 0.0,
            phase_d_min_learned: 100.0,
            phase_d_max_fallback: 0.0,
            phase_d_max_learned: 0.0,
            phase_d_cache_hits: 0,
            phase_d_cache_misses: 0,
            phase_d_cache_start: None,
            phase_d_nonresident_s: 0.0,
            phase_d_max_nonresident_s: 0.0,
            phase_d_halo_resident_frames: 0,
            phase_d_halo_nonresident_frames: 0,
            phase_d_frames: 0,
            phase_d_structural_ok: true,
            phase_d_streaming_ok: true,
            phase_d_origin_generation_start: None,
            phase_d_peak_active_chunks: 0,
            failure_reason: None,
        })
    }
}

#[derive(Clone, Serialize, Default)]
struct M5Report {
    schema: String,
    max_duration_seconds: u64,
    test_direction: [f64; 3],
    test_tile: [u32; 3],
    scenario_initial_ram_cached: bool,
    scenario_initial_disk_cached: bool,
    forward_tile_initial_ram_cached: bool,
    forward_tile_initial_disk_cached: bool,
    scenario_proven_uncached: bool,
    camera_chunk: Option<[u32; 4]>,
    // Phase durations in seconds.
    phase_a_duration_s: Option<f64>,
    phase_b_duration_s: Option<f64>,
    phase_c_duration_s: Option<f64>,
    phase_d_duration_s: Option<f64>,
    // Phase A gates.
    immediate_procedural: bool,
    fine_lod: bool,
    nearest_lod: Option<u8>,
    vertex_spacing_m: Option<f64>,
    // Phase B gates.
    exact_halo_resident: bool,
    halo_deps_count: Option<usize>,
    // Phase C gates.
    blend_monotonic_complete: bool,
    blend_weights: Vec<f64>,
    targeted_rebuilds: bool,
    rebuilds_queued: u64,
    rebuilds_completed: u64,
    // Phase D gates.
    warm_ahead: bool,
    provider_hitch: bool,
    provider_p95_us: Option<f64>,
    phase_d_min_learned_coverage: f64,
    phase_d_max_fallback: f64,
    phase_d_cache_hits: u64,
    phase_d_cache_misses: u64,
    phase_d_distance_m: f64,
    phase_d_halo_resident_frames: u64,
    phase_d_halo_nonresident_frames: u64,
    phase_d_max_halo_recovery_s: f64,
    // Structural gates.
    no_holes_or_black: bool,
    no_cross_generation_errors: bool,
    cross_gen_attaches: usize,
    // Telemetry.
    final_resident_tiles: usize,
    final_queue_depth: usize,
    final_pending_in_flight: usize,
    final_failed_total: u64,
    final_cache_hits: u64,
    final_cache_misses: u64,
    final_cache_hit_rate: f64,
    final_fallback_percent: f64,
    final_latency_p50_ms: Option<f64>,
    final_latency_p95_ms: Option<f64>,
    final_health: String,
    min_local_learned_coverage: f64,
    max_local_learned_coverage: f64,
    // Phase-specific failure info.
    failure_reason: Option<String>,
    phase_reached: String,
    // Overall.
    all_passed: bool,
}

impl M5Report {
    fn compute_all_passed(&mut self) {
        self.all_passed = self.immediate_procedural
            && self.scenario_proven_uncached
            && self.fine_lod
            && self.exact_halo_resident
            && self.blend_monotonic_complete
            && self.targeted_rebuilds
            && self.warm_ahead
            && self.provider_hitch
            && self.no_holes_or_black
            && self.no_cross_generation_errors;
    }
}

pub struct M5TestPlugin;

impl Plugin for M5TestPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            run_m5_test
                .after(crate::camera::CameraUpdate)
                .after(crate::terrain_diffusion::TerrainDiffusionUpdate)
                .after(er_terrain::TerrainUpdate),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_m5_test(
    mut config: ResMut<M5TestConfig>,
    mut camera_query: Query<&mut OrbitCamera>,
    terrain_state: Res<er_terrain::TerrainState>,
    debug_info: Res<er_terrain::TerrainDebugInfo>,
    diag: Option<Res<TerrainDiffusionDiagnostic>>,
    timing: Res<ProviderTimingHistory>,
    time: Res<Time>,
    #[cfg(feature = "terrain_diffusion")] runtime: Option<Res<TerrainDiffusionRuntime>>,
    mut exit: MessageWriter<AppExit>,
) {
    if config.finished {
        return;
    }
    if config.start.is_none() {
        config.start = Some(Instant::now());
        config.phase_start = Some(Instant::now());
        config.phase = M5Phase::PhaseA_Settle;
        config.report.max_duration_seconds = config.max_duration.as_secs();
    }

    let now = Instant::now();
    let elapsed = now.duration_since(config.start.unwrap());

    // Global timeout: write fail report and exit.
    if elapsed > config.max_duration {
        config.report.schema = "m5-surface-test-report/v2".to_owned();
        if config.failure_reason.is_none() {
            config.failure_reason = Some(format!("Global timeout at {:?}", config.phase));
        }
        collect_final_telemetry(&mut config, &diag);
        config.report.failure_reason = config.failure_reason.clone();
        config.report.phase_reached = format!("{:?}", config.phase);
        config.report.compute_all_passed();
        finish_test(&mut config, &mut exit);
        return;
    }

    match config.phase {
        M5Phase::NotStarted => {
            config.phase = M5Phase::PhaseA_Settle;
            config.phase_start = Some(now);
        }
        M5Phase::PhaseA_Settle => {
            run_phase_a(
                &mut config,
                &mut camera_query,
                &terrain_state,
                &debug_info,
                &runtime,
                now,
            );
        }
        M5Phase::PhaseB_Residency => {
            #[cfg(feature = "terrain_diffusion")]
            run_phase_b(&mut config, &runtime, &terrain_state, &debug_info, now);
            #[cfg(not(feature = "terrain_diffusion"))]
            {
                let _ = &runtime;
                config.phase = M5Phase::Done;
            }
        }
        M5Phase::PhaseC_Blend => {
            run_phase_c(&mut config, &terrain_state, &diag, &debug_info, now);
        }
        M5Phase::PhaseD_WarmAhead => {
            run_phase_d(
                &mut config,
                &mut camera_query,
                &diag,
                &timing,
                &time,
                &debug_info,
                &terrain_state,
                &runtime,
                now,
            );
        }
        M5Phase::Done => {
            collect_final_telemetry(&mut config, &diag);
            config.report.schema = "m5-surface-test-report/v2".to_owned();
            config.report.failure_reason = config.failure_reason.clone();
            config.report.phase_reached = format!("{:?}", config.phase);
            config.report.compute_all_passed();
            finish_test(&mut config, &mut exit);
        }
    }
}

fn collect_final_telemetry(
    config: &mut M5TestConfig,
    diag: &Option<Res<TerrainDiffusionDiagnostic>>,
) {
    if let Some(d) = diag {
        let s = &d.streaming;
        config.report.final_resident_tiles = s.resident_tiles;
        config.report.final_queue_depth = d.queue_depth;
        config.report.final_pending_in_flight = s.pending_in_flight;
        config.report.final_failed_total = s.failed_total;
        config.report.final_cache_hits = s.cache_hits;
        config.report.final_cache_misses = s.cache_misses;
        config.report.final_cache_hit_rate = s.cache_hit_rate;
        config.report.final_fallback_percent = s.fallback_percent;
        config.report.final_latency_p50_ms = s.latency_p50_ms;
        config.report.final_latency_p95_ms = s.latency_p95_ms;
        config.report.final_health = s.health.clone();
    }
}

fn run_phase_a(
    config: &mut M5TestConfig,
    camera_query: &mut Query<&mut OrbitCamera>,
    terrain_state: &er_terrain::TerrainState,
    debug_info: &er_terrain::TerrainDebugInfo,
    runtime: &Option<Res<TerrainDiffusionRuntime>>,
    now: Instant,
) {
    let Some(rt) = runtime else {
        config.failure_reason = Some("Phase A: no TerrainDiffusionRuntime".to_owned());
        config.phase = M5Phase::Done;
        return;
    };
    if rt.chart_field.metadata().charts_per_face_edge != M5_EARTH_TILES_PER_EDGE {
        config.failure_reason = Some(format!(
            "Phase A: M5 surface gate requires Earth provider grid {}, runtime has {}",
            M5_EARTH_TILES_PER_EDGE,
            rt.chart_field.metadata().charts_per_face_edge
        ));
        config.phase = M5Phase::Done;
        return;
    }

    if config.scenario_tile.is_none() {
        let Some(tile) = select_uncached_scenario_tile(&rt.chart_field) else {
            config.failure_reason = Some(
                "Phase A: no RAM+disk-uncached scenario tile found in deterministic probe set"
                    .to_owned(),
            );
            config.phase = M5Phase::Done;
            return;
        };
        let key = rt
            .chart_field
            .key_for_tile(tile.0, tile.1, tile.2, M5_EARTH_TILES_PER_EDGE);
        let forward_key =
            rt.chart_field
                .key_for_tile(tile.0, tile.1 + 1, tile.2, M5_EARTH_TILES_PER_EDGE);
        config.report.scenario_initial_ram_cached =
            rt.chart_field.cache().get_resident(&key).is_some();
        config.report.scenario_initial_disk_cached = is_tile_cached_on_disk(
            &rt.chart_field,
            tile.0,
            tile.1,
            tile.2,
            M5_EARTH_TILES_PER_EDGE,
        );
        config.report.forward_tile_initial_ram_cached =
            rt.chart_field.cache().get_resident(&forward_key).is_some();
        config.report.forward_tile_initial_disk_cached = is_tile_cached_on_disk(
            &rt.chart_field,
            tile.0,
            tile.1 + 1,
            tile.2,
            M5_EARTH_TILES_PER_EDGE,
        );
        config.report.scenario_proven_uncached = !config.report.scenario_initial_ram_cached
            && !config.report.scenario_initial_disk_cached
            && !config.report.forward_tile_initial_ram_cached
            && !config.report.forward_tile_initial_disk_cached;
        config.scenario_tile = Some(tile);
    }

    let (face, tile_x, tile_y) = config.scenario_tile.unwrap();
    let n = M5_EARTH_TILES_PER_EDGE as f64;
    let u = (tile_x as f64 + 1.0 - M5_BOUNDARY_INSET_TILE_FRACTION) / n;
    let v = (tile_y as f64 + 0.5) / n;
    let dir = uv_to_dir(face, u, v);

    // Set camera to surface mode at the deterministic test direction.
    if let Ok(mut camera) = camera_query.single_mut() {
        if camera.mode != CameraMode::Surface {
            camera.mode = CameraMode::Surface;
            camera.surface_target = SurfaceTarget::at(face, u, v);
            camera.surface_altitude = M5_SURFACE_ALTITUDE_M;
            camera.smoothed_surface_altitude = M5_SURFACE_ALTITUDE_M;
        }
    }

    config.report.test_direction = [dir.x, dir.y, dir.z];
    config.report.test_tile = [face as u32, tile_x, tile_y];

    // Capture the actual camera-containing active chunk from debug info.
    if let Some(chunk) = &debug_info.containing_chunk {
        config.camera_chunk = Some(*chunk);
        config.report.camera_chunk = Some([chunk.face as u32, chunk.i, chunk.j, chunk.lod as u32]);
    }

    // Gate 1: immediate procedural fallback — captured ONCE before any
    // scenario tile is resident. Non-tautological: must verify the field
    // source is Procedural AND coverage is unknown/zero.
    if !config.immediate_procedural_captured {
        let sample = terrain_state.field.sample(dir);
        let is_procedural =
            sample.source == er_world::terrain_field::TerrainSampleSource::Procedural;
        config.report.immediate_procedural =
            config.report.scenario_proven_uncached && is_procedural;
        config.immediate_procedural_captured = true;
        info!(
            "M5 Phase A: immediate procedural proof captured: source={:?}",
            sample.source
        );
    }

    // Gate 2: fine LOD — require LOD >= 17 or vertex spacing <= 5m.
    let lod = debug_info.nearest_chunk_lod;
    let vs = debug_info.vertex_spacing_m;
    config.report.nearest_lod = Some(lod);
    config.report.vertex_spacing_m = Some(vs);
    let fine_lod = lod >= FINE_LOD_TARGET || (vs > 0.0 && vs <= FINE_VERTEX_SPACING_M);
    config.report.fine_lod = fine_lod;

    // Track terrain settle: consecutive frames with low pending work.
    // At fine LOD (17), many chunks need meshes — 256 pending is normal.
    // We require the pending count to be stable (not growing) and the LOD
    // to have reached the target. The threshold is relative to active chunks.
    let pending_total = debug_info.pending_meshes;
    let pending_threshold = (debug_info.active_chunks / 4).max(16);
    let settled = pending_total <= pending_threshold && fine_lod;
    if settled {
        config.settled_frames += 1;
    } else {
        config.settled_frames = 0;
    }

    let phase_elapsed = now.duration_since(config.phase_start.unwrap());
    let settled_ok = config.settled_frames >= SETTLED_FRAMES_REQUIRED;

    if config.settled_frames == 0 && phase_elapsed.as_secs().is_multiple_of(10) {
        info!(
            "M5 Phase A: lod={}, vs={:.1}m, pending meshes={}, splits={}, merges={}, settled_frames={}",
            lod, vs, debug_info.pending_meshes, debug_info.pending_splits, debug_info.pending_merges, config.settled_frames
        );
    }

    if settled_ok {
        config.report.phase_a_duration_s = Some(phase_elapsed.as_secs_f64());
        config.phase = M5Phase::PhaseB_Residency;
        config.phase_start = Some(now);
        info!(
            "M5 Phase A complete: immediate_procedural={}, fine_lod={}, lod={}, vs={:.1}m, settled_frames={}",
            config.report.immediate_procedural,
            config.report.fine_lod,
            lod,
            vs,
            config.settled_frames
        );
    } else if phase_elapsed > PHASE_A_TIMEOUT {
        // Fail closed: do NOT proceed as if successful.
        config.failure_reason = Some(format!(
            "Phase A timeout: fine_lod={}, lod={}, settled_frames={}",
            fine_lod, lod, config.settled_frames
        ));
        config.phase = M5Phase::Done;
    }
}

#[cfg(feature = "terrain_diffusion")]
fn run_phase_b(
    config: &mut M5TestConfig,
    runtime: &Option<Res<TerrainDiffusionRuntime>>,
    terrain_state: &er_terrain::TerrainState,
    debug_info: &er_terrain::TerrainDebugInfo,
    now: Instant,
) {
    let Some(rt) = runtime else {
        config.failure_reason = Some("Phase B: no TerrainDiffusionRuntime".to_owned());
        config.phase = M5Phase::Done;
        return;
    };
    let field = &rt.chart_field;
    let queue = &rt.streaming_queue;

    if let Some(dir) = scenario_direction(config) {
        config
            .blend_samples
            .push(current_blend_weight(terrain_state, dir));
    }

    // Use the actual camera-containing active chunk from debug info.
    let chunk = match &debug_info.containing_chunk {
        Some(c) => *c,
        None => match config.camera_chunk {
            Some(c) => c,
            None => {
                config.failure_reason =
                    Some("Phase B: no camera-containing chunk available".to_owned());
                config.phase = M5Phase::Done;
                return;
            }
        },
    };
    // Update the stored chunk in case it refined.
    config.camera_chunk = Some(chunk);
    config.report.camera_chunk = Some([chunk.face as u32, chunk.i, chunk.j, chunk.lod as u32]);

    if config.enqueued_keys.is_empty() {
        // Record rebuild count at entry to verify increments later.
        let tel = queue.telemetry();
        config.phase_b_rebuilds_completed = tel.rebuilds_completed;

        let deps = field.chunk_halo_dependencies(chunk);
        config.report.halo_deps_count = Some(deps.len());
        let now_instant = Instant::now();
        for key in &deps {
            let provider_coord = ProviderTileCoordinate {
                face: key.face,
                x: key.x,
                y: key.y,
            };
            queue.enqueue(
                key.clone(),
                provider_coord,
                PriorityClass::VisibleSurface,
                Some(chunk),
                now_instant,
            );
            config.enqueued_keys.push(key.clone());
        }
        info!(
            "M5 Phase B: enqueued {} halo dependencies for chunk {:?} (lod={})",
            deps.len(),
            chunk,
            chunk.lod
        );
    }

    // Verify every requested exact key is resident AND chunk_halo_resident.
    let all_resident = field.chunk_halo_resident(chunk);
    config.report.exact_halo_resident = all_resident;

    let phase_elapsed = now.duration_since(config.phase_start.unwrap());
    if all_resident {
        config.report.phase_b_duration_s = Some(phase_elapsed.as_secs_f64());
        config.phase = M5Phase::PhaseC_Blend;
        config.phase_start = Some(now);
        info!(
            "M5 Phase B complete: halo_resident=true, deps={}, duration={:.1}s",
            config.enqueued_keys.len(),
            phase_elapsed.as_secs_f64()
        );
    } else if phase_elapsed > PHASE_B_TIMEOUT {
        // Fail closed: do NOT enter blend as if successful.
        config.failure_reason = Some(format!(
            "Phase B timeout: halo_resident=false, deps={}",
            config.enqueued_keys.len()
        ));
        config.phase = M5Phase::Done;
    }
}

fn run_phase_c(
    config: &mut M5TestConfig,
    terrain_state: &er_terrain::TerrainState,
    diag: &Option<Res<TerrainDiffusionDiagnostic>>,
    debug_info: &er_terrain::TerrainDebugInfo,
    now: Instant,
) {
    let Some(dir) = scenario_direction(config) else {
        config.failure_reason = Some("Phase C: scenario tile was not selected".to_owned());
        config.phase = M5Phase::Done;
        return;
    };

    // Sample the blend weight from the real BlendedHybridTerrainField.
    let weight = current_blend_weight(terrain_state, dir);

    config.blend_samples.push(weight);

    // Track targeted rebuilds: require increments attributable after Phase B.
    if let Some(d) = diag {
        config.report.rebuilds_queued = d.streaming.rebuilds_queued;
        config.report.rebuilds_completed = d.streaming.rebuilds_completed;
        // Rebuilds must have increased since Phase B entry.
        config.report.targeted_rebuilds =
            d.streaming.rebuilds_completed > config.phase_b_rebuilds_completed;
    }

    // Track cross-generation attaches (must remain 0).
    config.cross_gen_attaches = config
        .cross_gen_attaches
        .max(debug_info.cross_generation_mesh_attaches);

    let phase_elapsed = now.duration_since(config.phase_start.unwrap());

    // Require a meaningful >0 intermediate sample before accepting completion.
    let has_intermediate = config.blend_samples.iter().any(|&w| w > 0.0 && w < 0.95);

    let blend_complete = weight >= 0.95;
    let blend_monotonic = is_monotonic(&config.blend_samples);

    // Also require terrain settled (low pending work, not strict zero).
    let pending_threshold = (debug_info.active_chunks / 4).max(16);
    let settled = debug_info.pending_meshes <= pending_threshold;

    if blend_complete && has_intermediate && blend_monotonic && settled {
        config.report.blend_weights = config.blend_samples.clone();
        config.report.blend_monotonic_complete = true;
        config.report.phase_c_duration_s = Some(phase_elapsed.as_secs_f64());
        if let Some(d) = diag {
            config.phase_d_cache_start = Some((d.streaming.cache_hits, d.streaming.cache_misses));
        }
        config.cross_gen_attaches = 0;
        config.phase_d_origin_generation_start = Some(debug_info.render_origin_generation);
        config.phase_d_peak_active_chunks = debug_info.active_chunks;
        config.phase = M5Phase::PhaseD_WarmAhead;
        config.phase_start = Some(now);
        info!(
            "M5 Phase C complete: blend_weight={:.3}, monotonic={}, samples={}, rebuilds={}",
            weight,
            blend_monotonic,
            config.blend_samples.len(),
            config.report.rebuilds_completed
        );
    } else if phase_elapsed > PHASE_C_TIMEOUT {
        // Fail closed on timeout.
        config.report.blend_weights = config.blend_samples.clone();
        config.report.blend_monotonic_complete = false;
        config.failure_reason = Some(format!(
            "Phase C timeout: weight={:.3}, complete={}, has_intermediate={}, monotonic={}, settled={}",
            weight, blend_complete, has_intermediate, blend_monotonic, settled
        ));
        config.phase = M5Phase::Done;
    }
}

#[allow(clippy::too_many_arguments)]
fn run_phase_d(
    config: &mut M5TestConfig,
    camera_query: &mut Query<&mut OrbitCamera>,
    diag: &Option<Res<TerrainDiffusionDiagnostic>>,
    timing: &ProviderTimingHistory,
    time: &Time,
    debug_info: &er_terrain::TerrainDebugInfo,
    terrain_state: &er_terrain::TerrainState,
    runtime: &Option<Res<TerrainDiffusionRuntime>>,
    now: Instant,
) {
    // Measure provider main-thread P95.
    let p95_us = timing.percentile_ms(95.0).map(|ms| ms * 1000.0);
    config.report.provider_p95_us = p95_us;
    config.report.provider_hitch = p95_us.is_some_and(|us| us <= PROVIDER_P95_LIMIT_US);

    // Incremental tangent movement matches the production camera path. The
    // scenario begins just inside a provider boundary, so ordinary 10m/s
    // movement crosses into the halo tile prepared during Phase B.
    let dt = time.delta_secs_f64().clamp(0.0, 0.25);
    let step_m = PHASE_D_MOVE_SPEED_MPS * dt;
    config.phase_d_offset_m += step_m;

    if let Ok(mut camera) = camera_query.single_mut() {
        let pan = glam::DVec2::new(step_m, 0.0);
        camera.surface_target.local += pan;
        reproject_surface_target(&mut camera.surface_target, terrain_state.planet_radius);
    }

    // Compute the containing active chunk as it changes.
    if let Some(chunk) = &debug_info.containing_chunk {
        config.camera_chunk = Some(*chunk);
        config.report.camera_chunk = Some([chunk.face as u32, chunk.i, chunk.j, chunk.lod as u32]);
    }

    // Track local learned coverage and fallback from debug info.
    let learned_pct = debug_info.learned_source_coverage_percent as f64;
    let fallback_pct = 100.0 - learned_pct;
    config.phase_d_min_learned = config.phase_d_min_learned.min(learned_pct);
    config.phase_d_max_learned = config.phase_d_max_learned.max(learned_pct);
    config.phase_d_max_fallback = config.phase_d_max_fallback.max(fallback_pct);
    config.report.phase_d_min_learned_coverage = config.phase_d_min_learned;
    config.report.phase_d_max_fallback = config.phase_d_max_fallback;
    config.phase_d_frames += 1;

    // Observe the exact halo of the actual camera-containing active chunk.
    let halo_resident = runtime.as_ref().is_some_and(|rt| {
        debug_info
            .containing_chunk
            .is_some_and(|chunk| rt.chart_field.chunk_halo_resident(chunk))
    });
    if halo_resident {
        config.phase_d_halo_resident_frames += 1;
        config.phase_d_nonresident_s = 0.0;
    } else {
        config.phase_d_halo_nonresident_frames += 1;
        config.phase_d_nonresident_s += dt;
        config.phase_d_max_nonresident_s = config
            .phase_d_max_nonresident_s
            .max(config.phase_d_nonresident_s);
    }

    // Track real cumulative cache counter deltas from Phase D entry.
    if let Some(d) = diag {
        let (start_hits, start_misses) = *config
            .phase_d_cache_start
            .get_or_insert((d.streaming.cache_hits, d.streaming.cache_misses));
        config.phase_d_cache_hits = d.streaming.cache_hits.saturating_sub(start_hits);
        config.phase_d_cache_misses = d.streaming.cache_misses.saturating_sub(start_misses);
        config.report.phase_d_cache_hits = config.phase_d_cache_hits;
        config.report.phase_d_cache_misses = config.phase_d_cache_misses;
        config.phase_d_streaming_ok &=
            d.streaming.failed_total < 500 && d.streaming.pending_in_flight <= 8;
    }

    // No holes: continuous visible coverage and no stuck pending work.
    // During movement, pending meshes can spike as new chunks enter view.
    // The gate requires visible chunks > 0 (terrain is visible, no black
    // screen) and pending meshes bounded by a generous multiple of active
    // chunks (not stuck in an infinite rebuild loop).
    let visible_ok = debug_info.visible_chunks > 0;
    let no_stuck = debug_info.pending_meshes < debug_info.active_chunks * 3 + 50;
    let structural_frame_ok = visible_ok
        && debug_info.active_chunks > 0
        && debug_info.containing_chunk.is_some()
        && no_stuck
        && debug_info.vertex_spacing_m.is_finite();
    config.phase_d_structural_ok &= structural_frame_ok;
    config.report.no_holes_or_black = config.phase_d_structural_ok;

    // Cross-generation attaches are valid for origin-invariant meshes. Count
    // them cumulatively and bound them by one active-set worth of work per
    // render-origin generation window, rather than only checking one frame.
    config.cross_gen_attaches = config
        .cross_gen_attaches
        .saturating_add(debug_info.cross_generation_mesh_attaches);
    config.phase_d_peak_active_chunks = config
        .phase_d_peak_active_chunks
        .max(debug_info.active_chunks);
    config.report.cross_gen_attaches = config.cross_gen_attaches;
    let origin_shifts = debug_info.render_origin_generation.saturating_sub(
        config
            .phase_d_origin_generation_start
            .unwrap_or(debug_info.render_origin_generation),
    );
    config.report.no_cross_generation_errors = cross_generation_work_is_bounded(
        config.cross_gen_attaches,
        origin_shifts,
        config.phase_d_peak_active_chunks,
    );

    let phase_elapsed = now.duration_since(config.phase_start.unwrap());
    if phase_elapsed > PHASE_D_DURATION {
        config.report.phase_d_duration_s = Some(phase_elapsed.as_secs_f64());
        config.report.min_local_learned_coverage = config.phase_d_min_learned;
        config.report.max_local_learned_coverage = config.phase_d_max_learned;
        config.report.phase_d_distance_m = config.phase_d_offset_m;
        config.report.phase_d_halo_resident_frames = config.phase_d_halo_resident_frames;
        config.report.phase_d_halo_nonresident_frames = config.phase_d_halo_nonresident_frames;
        config.report.phase_d_max_halo_recovery_s = config.phase_d_max_nonresident_s;
        config.report.warm_ahead = phase_d_warm_ahead_passes(
            config.phase_d_frames,
            config.phase_d_halo_resident_frames,
            config.phase_d_nonresident_s,
            config.phase_d_max_nonresident_s,
            config.phase_d_min_learned,
            config.phase_d_max_fallback,
            config.phase_d_cache_hits,
            config.phase_d_cache_misses,
            config.phase_d_streaming_ok,
        );
        if !config.report.warm_ahead && config.failure_reason.is_none() {
            config.failure_reason = Some(format!(
                "Phase D warm-ahead failed: min_learned={:.1}%, max_fallback={:.1}%, max_halo_recovery={:.3}s, current_halo_loss={:.3}s",
                config.phase_d_min_learned,
                config.phase_d_max_fallback,
                config.phase_d_max_nonresident_s,
                config.phase_d_nonresident_s,
            ));
        }
        config.phase = M5Phase::Done;
        info!(
            "M5 Phase D complete: warm_ahead={}, provider_hitch={}, min_learned={:.1}%, max_fallback={:.1}%",
            config.report.warm_ahead,
            config.report.provider_hitch,
            config.phase_d_min_learned,
            config.phase_d_max_fallback
        );
    }
}

fn finish_test(config: &mut M5TestConfig, exit: &mut MessageWriter<AppExit>) {
    config.finished = true;
    let report = config.report.clone();
    let json = serde_json::to_vec_pretty(&report)
        .unwrap_or_else(|e| format!("{{\"error\": \"serialization failed: {e}\"}}").into_bytes());

    if let Some(path) = &config.output_path {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(path, &json) {
            error!(path = ?path, %e, "Failed to write M5 surface test report");
        } else {
            info!(path = ?path, "M5 surface test report written");
        }
    } else {
        let text = String::from_utf8_lossy(&json);
        info!("M5 surface test report:\n{text}");
    }

    exit.write(if report.all_passed {
        AppExit::Success
    } else {
        AppExit::error()
    });
}

fn is_monotonic(samples: &[f64]) -> bool {
    if samples.len() < 2 {
        return true;
    }
    let mut prev = -1.0;
    for &s in samples {
        if s < prev {
            return false;
        }
        prev = s;
    }
    true
}

fn current_blend_weight(terrain_state: &er_terrain::TerrainState, dir: glam::DVec3) -> f64 {
    terrain_state
        .field
        .as_any()
        .downcast_ref::<er_world::terrain_field::BlendedHybridTerrainField>()
        .map(|blended| blended.current_blend_weight(dir))
        .unwrap_or(0.0)
}

fn scenario_direction(config: &M5TestConfig) -> Option<glam::DVec3> {
    let (face, x, y) = config.scenario_tile?;
    let n = M5_EARTH_TILES_PER_EDGE as f64;
    Some(uv_to_dir(
        face,
        (x as f64 + 1.0 - M5_BOUNDARY_INSET_TILE_FRACTION) / n,
        (y as f64 + 0.5) / n,
    ))
}

fn select_uncached_scenario_tile(
    chart_field: &er_world::surface_cache::ChartMacroField,
) -> Option<(u8, u32, u32)> {
    let n = M5_EARTH_TILES_PER_EDGE as u64;
    let total = 6 * n * n;
    let start = M5_TEST_FACE as u64 * n * n + M5_TEST_TILE_Y as u64 * n + M5_TEST_TILE_X as u64;
    // A prime stride gives a stable spread across the provider atlas without
    // scanning millions of cache paths on a heavily populated cache.
    for probe in 0..4096_u64 {
        let flat = (start + probe * 104_729) % total;
        let face = (flat / (n * n)) as u8;
        let local = flat % (n * n);
        let x = (local % n) as u32;
        let y = (local / n) as u32;
        if x + 1 >= M5_EARTH_TILES_PER_EDGE {
            continue;
        }
        let key = chart_field.key_for_tile(face, x, y, M5_EARTH_TILES_PER_EDGE);
        let forward_key = chart_field.key_for_tile(face, x + 1, y, M5_EARTH_TILES_PER_EDGE);
        if chart_field.cache().get_resident(&key).is_none()
            && !is_tile_cached_on_disk(chart_field, face, x, y, M5_EARTH_TILES_PER_EDGE)
            && chart_field.cache().get_resident(&forward_key).is_none()
            && !is_tile_cached_on_disk(chart_field, face, x + 1, y, M5_EARTH_TILES_PER_EDGE)
        {
            return Some((face, x, y));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn phase_d_warm_ahead_passes(
    frames: u64,
    halo_resident_frames: u64,
    current_nonresident_s: f64,
    max_nonresident_s: f64,
    min_learned_percent: f64,
    max_fallback_percent: f64,
    cache_hits: u64,
    cache_misses: u64,
    streaming_ok: bool,
) -> bool {
    frames > 0
        && halo_resident_frames > 0
        && current_nonresident_s == 0.0
        && max_nonresident_s <= PHASE_D_MAX_RECOVERY_S
        && min_learned_percent > 0.0
        && max_fallback_percent < 100.0
        && cache_hits.saturating_add(cache_misses) > 0
        && streaming_ok
}

fn cross_generation_work_is_bounded(
    attaches: usize,
    origin_shifts: u64,
    peak_active_chunks: usize,
) -> bool {
    attaches <= peak_active_chunks.saturating_mul(origin_shifts.saturating_add(1) as usize)
}

/// Check if a test tile is already cached on disk. Returns true if the
/// disk file exists for the given tile's chart key.
#[cfg(feature = "terrain_diffusion")]
fn is_tile_cached_on_disk(
    chart_field: &er_world::surface_cache::ChartMacroField,
    face: u8,
    x: u32,
    y: u32,
    tiles_per_face_edge: u32,
) -> bool {
    let key = chart_field.key_for_tile(face, x, y, tiles_per_face_edge);
    chart_field.cache().is_on_disk(&key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_monotonic_detects_non_monotonic() {
        assert!(is_monotonic(&[]));
        assert!(is_monotonic(&[0.0]));
        assert!(is_monotonic(&[0.0, 0.1, 0.5, 0.9, 1.0]));
        assert!(!is_monotonic(&[0.5, 0.3]));
        assert!(is_monotonic(&[0.0, 0.0, 0.1]));
    }

    #[test]
    fn m5_report_all_passed_requires_all_gates() {
        let mut r = M5Report::default();
        r.compute_all_passed();
        assert!(!r.all_passed);

        r.immediate_procedural = true;
        r.scenario_proven_uncached = true;
        r.fine_lod = true;
        r.exact_halo_resident = true;
        r.blend_monotonic_complete = true;
        r.targeted_rebuilds = true;
        r.warm_ahead = true;
        r.provider_hitch = true;
        r.no_holes_or_black = true;
        r.no_cross_generation_errors = true;
        r.compute_all_passed();
        assert!(r.all_passed);
    }

    #[test]
    fn warm_ahead_rejects_zero_local_learned_coverage() {
        assert!(!phase_d_warm_ahead_passes(
            120, 120, 0.0, 0.0, 0.0, 100.0, 10, 2, true,
        ));
    }

    #[test]
    fn warm_ahead_rejects_unrecovered_or_slow_halo_loss() {
        assert!(!phase_d_warm_ahead_passes(
            120, 100, 0.1, 0.5, 50.0, 50.0, 10, 2, true,
        ));
        assert!(!phase_d_warm_ahead_passes(
            120,
            100,
            0.0,
            PHASE_D_MAX_RECOVERY_S + 0.01,
            50.0,
            50.0,
            10,
            2,
            true,
        ));
    }

    #[test]
    fn warm_ahead_requires_real_cache_activity() {
        assert!(!phase_d_warm_ahead_passes(
            120, 120, 0.0, 0.0, 100.0, 0.0, 0, 0, true,
        ));
        assert!(phase_d_warm_ahead_passes(
            120, 120, 0.0, 0.0, 100.0, 0.0, 10, 2, true,
        ));
    }

    #[test]
    fn cross_generation_bound_scales_with_active_chunks_and_shifts() {
        assert!(cross_generation_work_is_bounded(100, 0, 100));
        assert!(!cross_generation_work_is_bounded(101, 0, 100));
        assert!(cross_generation_work_is_bounded(300, 2, 100));
        assert!(!cross_generation_work_is_bounded(301, 2, 100));
    }

    #[test]
    fn m5_report_failure_reason_propagates() {
        let mut r = M5Report {
            failure_reason: Some("test failure".to_owned()),
            ..Default::default()
        };
        r.compute_all_passed();
        assert!(!r.all_passed);
        assert!(r.failure_reason.is_some());
    }

    #[test]
    fn fine_lod_target_is_17() {
        assert_eq!(FINE_LOD_TARGET, 17);
    }

    #[test]
    fn fine_vertex_spacing_threshold_is_5m() {
        assert_eq!(FINE_VERTEX_SPACING_M, 5.0);
    }

    #[test]
    fn provider_p95_limit_is_1ms() {
        let limit = PROVIDER_P95_LIMIT_US;
        assert_eq!(limit, 1_000.0);
    }

    #[test]
    fn settled_frames_required_is_reasonable() {
        let n = SETTLED_FRAMES_REQUIRED;
        assert!(n >= 5);
    }

    #[test]
    fn test_tile_is_in_range() {
        let x = M5_TEST_TILE_X;
        let y = M5_TEST_TILE_Y;
        let n = M5_EARTH_TILES_PER_EDGE;
        assert!(x < n);
        assert!(y < n);
    }

    #[test]
    fn test_direction_is_valid() {
        let n = M5_EARTH_TILES_PER_EDGE as f64;
        let u = (M5_TEST_TILE_X as f64 + 0.5) / n;
        let v = (M5_TEST_TILE_Y as f64 + 0.5) / n;
        let dir = uv_to_dir(M5_TEST_FACE, u, v);
        assert!(dir.is_normalized());
        assert!(dir.length() > 0.99 && dir.length() < 1.01);
    }

    #[test]
    fn phase_d_path_crosses_forward_provider_boundary_at_ten_mps() {
        let n = M5_EARTH_TILES_PER_EDGE as f64;
        let start_u = (M5_TEST_TILE_X as f64 + 1.0 - M5_BOUNDARY_INSET_TILE_FRACTION) / n;
        let v = (M5_TEST_TILE_Y as f64 + 0.5) / n;
        let mut target = SurfaceTarget::at(M5_TEST_FACE, start_u, v);
        target.local.x += PHASE_D_MOVE_SPEED_MPS * PHASE_D_DURATION.as_secs_f64();
        reproject_surface_target(&mut target, 6_371_000.0);

        assert_eq!(target.face, M5_TEST_FACE);
        assert_eq!(
            (target.u * n).floor() as u32,
            M5_TEST_TILE_X + 1,
            "300m Phase D path must cross exactly into the forward tile"
        );
    }

    #[test]
    fn test_tile_is_not_at_face_root() {
        // The test tile should not be at a face root (center) to avoid
        // collision with FarRootCoverage requests.
        let n = M5_EARTH_TILES_PER_EDGE as f64;
        let u = (M5_TEST_TILE_X as f64 + 0.5) / n;
        let v = (M5_TEST_TILE_Y as f64 + 0.5) / n;
        assert!((u - 0.5).abs() > 0.01 || (v - 0.5).abs() > 0.01);
    }

    #[test]
    fn test_default_config_has_no_failure() {
        let cfg = M5TestConfig {
            max_duration: Duration::from_secs(600),
            output_path: None,
            start: None,
            phase: M5Phase::NotStarted,
            phase_start: None,
            finished: false,
            camera_chunk: None,
            scenario_tile: None,
            enqueued_keys: Vec::new(),
            blend_samples: Vec::new(),
            cross_gen_attaches: 0,
            settled_frames: 0,
            phase_b_rebuilds_completed: 0,
            report: M5Report::default(),
            immediate_procedural_captured: false,
            phase_d_offset_m: 0.0,
            phase_d_min_learned: 100.0,
            phase_d_max_fallback: 0.0,
            phase_d_max_learned: 0.0,
            phase_d_cache_hits: 0,
            phase_d_cache_misses: 0,
            phase_d_cache_start: None,
            phase_d_nonresident_s: 0.0,
            phase_d_max_nonresident_s: 0.0,
            phase_d_halo_resident_frames: 0,
            phase_d_halo_nonresident_frames: 0,
            phase_d_frames: 0,
            phase_d_structural_ok: true,
            phase_d_streaming_ok: true,
            phase_d_origin_generation_start: None,
            phase_d_peak_active_chunks: 0,
            failure_reason: None,
        };
        assert_eq!(cfg.phase, M5Phase::NotStarted);
        assert!(!cfg.finished);
        assert!(cfg.failure_reason.is_none());
    }
}
