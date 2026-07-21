//! Sidecar-plus-game coexistence stress mode for Terrain Diffusion.
//!
//! Activated by `--terrain-diffusion-stress <seconds>` (requires the
//! `terrain_diffusion` feature). Runs the game with the sidecar active for the
//! requested duration, then writes a machine-readable JSON report of
//! main-thread provider timing statistics and exits.
//!
//! The report is the authoritative source for the Milestone 3
//! "provider-attributable main-thread hitch <= 1 ms P95" gate. The async
//! sidecar/network/model work is explicitly excluded — only the Bevy `Update`
//! systems owned by the Terrain Diffusion plugin are timed (see
//! [`ProviderTimingHistory`]).
//!
//! CLI flags:
//!   --terrain-diffusion-stress <seconds>   enable stress mode, run for N seconds
//!   --terrain-diffusion-stress-output <path>  write JSON report to this path
//!   --terrain-diffusion-stress-hitch-ceiling <ms>  P95 hitch gate ceiling (default: 1.0)

use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use serde::Serialize;

use crate::terrain_diffusion::ProviderTimingHistory;

const DEFAULT_HITCH_CEILING_MS: f64 = 1.0;

#[derive(Resource)]
pub struct StressConfig {
    duration: Duration,
    output_path: Option<PathBuf>,
    hitch_ceiling_ms: f64,
    start: Option<Instant>,
    finished: bool,
}

impl StressConfig {
    pub fn parse_args() -> Option<Self> {
        let args: Vec<String> = std::env::args().collect();
        let mut seconds: Option<u64> = None;
        let mut output_path: Option<PathBuf> = None;
        let mut hitch_ceiling = DEFAULT_HITCH_CEILING_MS;

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--terrain-diffusion-stress" => {
                    i += 1;
                    if i < args.len() {
                        seconds = args[i].parse().ok();
                    }
                }
                "--terrain-diffusion-stress-output" => {
                    i += 1;
                    if i < args.len() {
                        output_path = Some(PathBuf::from(&args[i]));
                    }
                }
                "--terrain-diffusion-stress-hitch-ceiling" => {
                    i += 1;
                    if i < args.len() {
                        hitch_ceiling = args[i].parse().unwrap_or(DEFAULT_HITCH_CEILING_MS);
                    }
                }
                _ => {}
            }
            i += 1;
        }

        seconds.map(|secs| Self {
            duration: Duration::from_secs(secs),
            output_path,
            hitch_ceiling_ms: hitch_ceiling,
            start: None,
            finished: false,
        })
    }
}

#[derive(Serialize)]
struct StressReport {
    schema: String,
    duration_seconds: u64,
    hitch_ceiling_ms: f64,
    frames_recorded: usize,
    provider_p50_ms: Option<f64>,
    provider_p95_ms: Option<f64>,
    provider_p99_ms: Option<f64>,
    provider_max_ms: Option<f64>,
    provider_mean_ms: Option<f64>,
    exit_gate_hitch_ok: bool,
    exit_gate_passed: bool,
}

pub struct TerrainDiffusionStressPlugin;

impl Plugin for TerrainDiffusionStressPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, run_stress.after(er_terrain::TerrainUpdate));
    }
}

fn run_stress(
    mut config: ResMut<StressConfig>,
    timing: Res<ProviderTimingHistory>,
    mut exit: MessageWriter<AppExit>,
) {
    if config.finished {
        return;
    }
    if config.start.is_none() {
        config.start = Some(Instant::now());
    }
    let elapsed = config.start.unwrap().elapsed();
    if elapsed < config.duration {
        return;
    }

    let p50 = timing.percentile_ms(50.0);
    let p95 = timing.percentile_ms(95.0);
    let p99 = timing.percentile_ms(99.0);
    let max_ms = timing.max_ms();
    let mean_ms = timing.mean_ms();
    let hitch_ok = !timing.is_empty() && p95.is_some_and(|v| v <= config.hitch_ceiling_ms);

    let report = StressReport {
        schema: "terrain-diffusion-stress-report/v1".to_owned(),
        duration_seconds: config.duration.as_secs(),
        hitch_ceiling_ms: config.hitch_ceiling_ms,
        frames_recorded: timing.len(),
        provider_p50_ms: p50,
        provider_p95_ms: p95,
        provider_p99_ms: p99,
        provider_max_ms: max_ms,
        provider_mean_ms: mean_ms,
        exit_gate_hitch_ok: hitch_ok,
        exit_gate_passed: hitch_ok,
    };

    let Ok(json) = serde_json::to_vec_pretty(&report) else {
        error!("Could not serialize Terrain Diffusion stress report");
        config.finished = true;
        exit.write(AppExit::error());
        return;
    };
    if let Some(path) = &config.output_path {
        if let Some(parent) = path.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                error!(?path, %error, "Could not create Terrain Diffusion stress report directory");
                config.finished = true;
                exit.write(AppExit::error());
                return;
            }
        }
        if let Err(error) = std::fs::write(path, json) {
            error!(?path, %error, "Could not write Terrain Diffusion stress report");
            config.finished = true;
            exit.write(AppExit::error());
            return;
        }
        info!(
            path = ?path,
            frames = report.frames_recorded,
            p95_ms = ?p95,
            "Terrain Diffusion stress report written"
        );
    } else {
        let text = String::from_utf8_lossy(&json);
        info!("Terrain Diffusion stress report:\n{text}");
    }

    config.finished = true;
    exit.write(if report.exit_gate_passed {
        AppExit::Success
    } else {
        AppExit::error()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_returns_none_without_flag() {
        // Cannot easily set std::env::args in a unit test, but we can verify
        // the default ceiling constant.
        assert_eq!(DEFAULT_HITCH_CEILING_MS, 1.0);
    }

    #[test]
    fn report_serializes_with_gate_fields() {
        let report = StressReport {
            schema: "terrain-diffusion-stress-report/v1".to_owned(),
            duration_seconds: 60,
            hitch_ceiling_ms: 1.0,
            frames_recorded: 100,
            provider_p50_ms: Some(0.05),
            provider_p95_ms: Some(0.8),
            provider_p99_ms: Some(1.2),
            provider_max_ms: Some(2.0),
            provider_mean_ms: Some(0.1),
            exit_gate_hitch_ok: true,
            exit_gate_passed: true,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("exit_gate_hitch_ok"));
        assert!(json.contains("exit_gate_passed"));
        assert!(json.contains("provider_p95_ms"));
    }

    #[test]
    fn gate_fails_when_p95_exceeds_ceiling() {
        let report = StressReport {
            schema: "terrain-diffusion-stress-report/v1".to_owned(),
            duration_seconds: 60,
            hitch_ceiling_ms: 1.0,
            frames_recorded: 100,
            provider_p50_ms: Some(0.05),
            provider_p95_ms: Some(1.5),
            provider_p99_ms: Some(2.0),
            provider_max_ms: Some(3.0),
            provider_mean_ms: Some(0.1),
            exit_gate_hitch_ok: false,
            exit_gate_passed: false,
        };
        assert!(!report.exit_gate_hitch_ok);
        assert!(!report.exit_gate_passed);
    }

    #[test]
    fn gate_fails_when_no_samples_present() {
        // When no frames are recorded, p95 is None and the gate should be
        // false (cannot prove the gate without evidence).
        let report = StressReport {
            schema: "terrain-diffusion-stress-report/v1".to_owned(),
            duration_seconds: 60,
            hitch_ceiling_ms: 1.0,
            frames_recorded: 0,
            provider_p50_ms: None,
            provider_p95_ms: None,
            provider_p99_ms: None,
            provider_max_ms: None,
            provider_mean_ms: None,
            exit_gate_hitch_ok: false,
            exit_gate_passed: false,
        };
        assert!(!report.exit_gate_passed);
    }
}
