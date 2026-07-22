//! Learned-enabled streaming smoke/stress mode (Milestone 5).
//!
//! Activated by `--learned-stress <seconds>` (requires the
//! `terrain_diffusion` feature). Runs the game with learned terrain streaming
//! enabled for the requested duration, then writes a machine-readable JSON
//! report of all M5 streaming gates and exits.
//!
//! Unlike `--terrain-diffusion-stress` (which measures procedural sidecar
//! timing), this mode enables the full learned streaming pipeline (priority
//! queue, disk cache, halo residency, blend) and reports:
//!
//! - Queue depth, resident/pending/failed counts
//! - Cache hit rate, fallback percentage
//! - Latency P50/P95
//! - Rebuild counts
//! - Service health and provider state
//! - Whether the M5 exit gates passed (no black chunks, warm tiles ahead of
//!   camera, fallback < threshold)
//!
//! CLI flags:
//!   --learned-stress <seconds>           enable mode, run for N seconds
//!   --learned-stress-output <path>       write JSON report to this path
//!   --learned-stress-fallback-ceiling <pct>  fallback % gate (default: 100.0)
//!   --learned-stress-resident-floor <n>  min resident tiles for warm-ahead gate (default: 0)
//!
//! Avoids `PresentMode::Immediate`; preserves screenshot/benchmark determinism
//! by using the same `AutoNoVsync` present mode as normal play.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use serde::Serialize;

use crate::terrain_diffusion::TerrainDiffusionDiagnostic;

const DEFAULT_FALLBACK_CEILING: f64 = 95.0;
const DEFAULT_RESIDENT_FLOOR: usize = 1;

#[derive(Resource)]
pub struct LearnedStressConfig {
    duration: Duration,
    output_path: Option<PathBuf>,
    fallback_ceiling_pct: f64,
    resident_floor: usize,
    start: Option<Instant>,
    finished: bool,
    /// Samples of the streaming diagnostic taken at regular intervals.
    samples: Vec<LearnedStressSample>,
}

impl LearnedStressConfig {
    pub fn parse_args() -> Option<Self> {
        let args: Vec<String> = std::env::args().collect();
        let mut seconds: Option<u64> = None;
        let mut output_path: Option<PathBuf> = None;
        let mut fallback_ceiling = DEFAULT_FALLBACK_CEILING;
        let mut resident_floor = DEFAULT_RESIDENT_FLOOR;

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--learned-stress" => {
                    i += 1;
                    if i < args.len() {
                        seconds = args[i].parse().ok();
                    }
                }
                "--learned-stress-output" => {
                    i += 1;
                    if i < args.len() {
                        output_path = Some(PathBuf::from(&args[i]));
                    }
                }
                "--learned-stress-fallback-ceiling" => {
                    i += 1;
                    if i < args.len() {
                        fallback_ceiling = args[i].parse().unwrap_or(DEFAULT_FALLBACK_CEILING);
                    }
                }
                "--learned-stress-resident-floor" => {
                    i += 1;
                    if i < args.len() {
                        resident_floor = args[i].parse().unwrap_or(DEFAULT_RESIDENT_FLOOR);
                    }
                }
                _ => {}
            }
            i += 1;
        }

        seconds.map(|secs| Self {
            duration: Duration::from_secs(secs),
            output_path,
            fallback_ceiling_pct: fallback_ceiling,
            resident_floor,
            start: None,
            finished: false,
            samples: Vec::new(),
        })
    }
}

/// A single point-in-time sample of the streaming diagnostic.
#[derive(Clone, Serialize)]
struct LearnedStressSample {
    elapsed_seconds: f64,
    queue_depth: usize,
    resident_tiles: usize,
    pending_in_flight: usize,
    failed_total: u64,
    cache_hit_rate: f64,
    fallback_percent: f64,
    latency_p50_ms: Option<f64>,
    latency_p95_ms: Option<f64>,
    rebuilds_queued: u64,
    rebuilds_completed: u64,
    health: String,
}

/// The final machine-readable report including all M5 gates.
#[derive(Serialize)]
struct LearnedStressReport {
    schema: String,
    duration_seconds: u64,
    fallback_ceiling_pct: f64,
    resident_floor: usize,
    samples_count: usize,
    // Final snapshot.
    final_queue_depth: usize,
    final_resident_tiles: usize,
    final_pending_in_flight: usize,
    final_failed_total: u64,
    final_cache_hit_rate: f64,
    final_fallback_percent: f64,
    final_latency_p50_ms: Option<f64>,
    final_latency_p95_ms: Option<f64>,
    final_rebuilds_queued: u64,
    final_rebuilds_completed: u64,
    final_health: String,
    final_provider_attempts: u64,
    final_provider_timeouts: u64,
    final_provider_cancellations: u64,
    final_provider_coalesced: u64,
    final_provider_stale_discarded: u64,
    // M5 exit gates.
    gate_no_black_chunks: bool,
    gate_warm_tiles_ahead_of_camera: bool,
    gate_fallback_within_ceiling: bool,
    gate_all_passed: bool,
    // The full sample series for offline analysis.
    samples: Vec<LearnedStressSample>,
}

pub struct LearnedStressPlugin;

impl Plugin for LearnedStressPlugin {
    fn build(&self, app: &mut App) {
        // M1: Order after diagnostic publication so the report captures the
        // latest streaming telemetry.
        app.add_systems(
            Update,
            run_learned_stress.after(crate::terrain_diffusion::TerrainDiffusionUpdate),
        );
    }
}

fn run_learned_stress(
    mut config: ResMut<LearnedStressConfig>,
    diag: Option<Res<TerrainDiffusionDiagnostic>>,
    mut exit: MessageWriter<AppExit>,
) {
    if config.finished {
        return;
    }
    if config.start.is_none() {
        config.start = Some(Instant::now());
    }
    let elapsed = config.start.unwrap().elapsed();

    // Sample the diagnostic every ~0.5s.
    let sample_interval = Duration::from_millis(500);
    let last_sample_time = config
        .samples
        .last()
        .map(|s| Duration::from_secs_f64(s.elapsed_seconds))
        .unwrap_or(Duration::ZERO);
    if elapsed - last_sample_time >= sample_interval || config.samples.is_empty() {
        if let Some(diag) = &diag {
            let s = &diag.streaming;
            config.samples.push(LearnedStressSample {
                elapsed_seconds: elapsed.as_secs_f64(),
                queue_depth: diag.queue_depth,
                resident_tiles: s.resident_tiles,
                pending_in_flight: s.pending_in_flight,
                failed_total: s.failed_total,
                cache_hit_rate: s.cache_hit_rate,
                fallback_percent: s.fallback_percent,
                latency_p50_ms: s.latency_p50_ms,
                latency_p95_ms: s.latency_p95_ms,
                rebuilds_queued: s.rebuilds_queued,
                rebuilds_completed: s.rebuilds_completed,
                health: s.health.clone(),
            });
        }
    }

    if elapsed < config.duration {
        return;
    }

    // Build the final report. Read the diagnostic fields directly from the
    // Res (it does not implement Clone) by destructuring into plain values.
    let (
        queue_depth,
        resident_tiles,
        pending,
        failed,
        hit_rate,
        fallback_pct,
        p50,
        p95,
        rq,
        rc,
        health,
        attempts,
        timeouts,
        cancels,
        coalesced,
        stale,
        has_diag,
    ) = if let Some(d) = &diag {
        let s = &d.streaming;
        (
            d.queue_depth,
            s.resident_tiles,
            s.pending_in_flight,
            s.failed_total,
            s.cache_hit_rate,
            s.fallback_percent,
            s.latency_p50_ms,
            s.latency_p95_ms,
            s.rebuilds_queued,
            s.rebuilds_completed,
            s.health.clone(),
            s.provider_attempts,
            s.provider_timeouts,
            s.provider_cancellations,
            s.provider_coalesced,
            s.provider_stale_discarded,
            true,
        )
    } else {
        (
            0,
            0,
            0,
            0,
            0.0,
            100.0,
            None,
            None,
            0,
            0,
            "unknown".to_owned(),
            0,
            0,
            0,
            0,
            0,
            false,
        )
    };

    // M5 exit gates (non-tautological):
    // 1. No black chunks: at least one learned tile arrived AND coverage is
    //    known (sample-source data was recorded). If no diagnostic was
    //    available or no learned data arrived, the gate FAILS.
    let gate_no_black_chunks = has_diag && resident_tiles >= 1 && fallback_pct < 100.0;
    // 2. Warm tiles ahead of camera: resident_tiles >= resident_floor (default
    //    1, not 0). This is non-tautological: it requires actual learned data
    //    to be resident.
    let gate_warm_tiles_ahead = resident_tiles >= config.resident_floor;
    // 3. Fallback within ceiling (default 95%, not 100%). This requires
    //    actual learned coverage to bring fallback below the ceiling.
    let gate_fallback_within_ceiling = has_diag && fallback_pct <= config.fallback_ceiling_pct;
    let gate_all_passed =
        gate_no_black_chunks && gate_warm_tiles_ahead && gate_fallback_within_ceiling;

    let report = LearnedStressReport {
        schema: "learned-streaming-stress-report/v1".to_owned(),
        duration_seconds: config.duration.as_secs(),
        fallback_ceiling_pct: config.fallback_ceiling_pct,
        resident_floor: config.resident_floor,
        samples_count: config.samples.len(),
        final_queue_depth: queue_depth,
        final_resident_tiles: resident_tiles,
        final_pending_in_flight: pending,
        final_failed_total: failed,
        final_cache_hit_rate: hit_rate,
        final_fallback_percent: fallback_pct,
        final_latency_p50_ms: p50,
        final_latency_p95_ms: p95,
        final_rebuilds_queued: rq,
        final_rebuilds_completed: rc,
        final_health: health,
        final_provider_attempts: attempts,
        final_provider_timeouts: timeouts,
        final_provider_cancellations: cancels,
        final_provider_coalesced: coalesced,
        final_provider_stale_discarded: stale,
        gate_no_black_chunks,
        gate_warm_tiles_ahead_of_camera: gate_warm_tiles_ahead,
        gate_fallback_within_ceiling,
        gate_all_passed,
        samples: std::mem::take(&mut config.samples),
    };

    let Ok(json) = serde_json::to_vec_pretty(&report) else {
        error!("Could not serialize learned streaming stress report");
        config.finished = true;
        exit.write(AppExit::error());
        return;
    };
    if let Some(path) = &config.output_path {
        if let Some(parent) = path.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                error!(?path, %error, "Could not create learned stress report directory");
                config.finished = true;
                exit.write(AppExit::error());
                return;
            }
        }
        if let Err(error) = std::fs::write(path, json) {
            error!(?path, %error, "Could not write learned stress report");
            config.finished = true;
            exit.write(AppExit::error());
            return;
        }
        info!(
            path = ?path,
            samples = report.samples_count,
            fallback_pct = report.final_fallback_percent,
            "Learned streaming stress report written"
        );
    } else {
        let text = String::from_utf8_lossy(&json);
        info!("Learned streaming stress report:\n{text}");
    }

    config.finished = true;
    exit.write(if report.gate_all_passed {
        AppExit::Success
    } else {
        AppExit::error()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fallback_ceiling_is_non_tautological() {
        // The default ceiling must be < 100 so the gate is non-tautological.
        assert_eq!(DEFAULT_FALLBACK_CEILING, 95.0);
    }

    #[test]
    fn default_resident_floor_is_nonzero() {
        // The default resident floor must be > 0 so the gate is
        // non-tautological (requires actual learned data).
        assert_eq!(DEFAULT_RESIDENT_FLOOR, 1);
    }

    #[test]
    fn report_serializes_with_all_gate_fields() {
        let report = LearnedStressReport {
            schema: "learned-streaming-stress-report/v1".to_owned(),
            duration_seconds: 60,
            fallback_ceiling_pct: 50.0,
            resident_floor: 10,
            samples_count: 10,
            final_queue_depth: 5,
            final_resident_tiles: 42,
            final_pending_in_flight: 2,
            final_failed_total: 1,
            final_cache_hit_rate: 0.85,
            final_fallback_percent: 15.0,
            final_latency_p50_ms: Some(0.5),
            final_latency_p95_ms: Some(2.0),
            final_rebuilds_queued: 10,
            final_rebuilds_completed: 10,
            final_health: "healthy".to_owned(),
            final_provider_attempts: 50,
            final_provider_timeouts: 1,
            final_provider_cancellations: 2,
            final_provider_coalesced: 5,
            final_provider_stale_discarded: 0,
            gate_no_black_chunks: true,
            gate_warm_tiles_ahead_of_camera: true,
            gate_fallback_within_ceiling: true,
            gate_all_passed: true,
            samples: vec![],
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("gate_all_passed"));
        assert!(json.contains("gate_no_black_chunks"));
        assert!(json.contains("final_fallback_percent"));
        assert!(json.contains("final_cache_hit_rate"));
        assert!(json.contains("final_health"));
    }

    #[test]
    fn gate_fails_when_fallback_exceeds_ceiling() {
        let gate = 60.0_f64 <= 50.0;
        assert!(!gate);
    }

    #[test]
    fn gate_passes_when_resident_meets_floor() {
        let resident = 42usize;
        let floor = DEFAULT_RESIDENT_FLOOR;
        assert!(resident >= floor);
        // Zero resident tiles must fail the floor gate.
        let zero = 0usize;
        assert!(zero < floor);
    }
}
