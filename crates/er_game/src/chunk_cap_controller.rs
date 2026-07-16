//! Conservative frame-budget controller for the terrain active-chunk cap.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use er_core::config::ACTIVE_CHUNK_CAP;
use er_terrain::{TerrainDebugInfo, TerrainState};

use crate::settings::GraphicsSettings;

const MIN_CHUNK_CAP: usize = 1_000;
const MAX_CHUNK_CAP: usize = 10_000;
const INCREASE_STEP: usize = 250;
const DECREASE_STEP: usize = 500;
const TARGET_FRAME_MS: f32 = 13.0;
const REGRESSION_FRAME_MS: f32 = 18.0;
const EVALUATION_INTERVAL: Duration = Duration::from_millis(750);
const STARTUP_GRACE: Duration = Duration::from_secs(5);
const FRAME_SAMPLE_CAPACITY: usize = 180;
const MIN_SAMPLES_PER_EVALUATION: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CapDecision {
    Hold,
    Increase,
    Decrease,
}

/// Dynamically raises terrain detail only when the game has demonstrated
/// sustained frame-time headroom. A decrease stops subsequent splits but never
/// evicts active chunks, preserving complete terrain coverage.
#[derive(Resource)]
pub struct DynamicChunkCapController {
    pub current_cap: usize,
    pub last_p95_ms: f32,
    pub last_change: i32,
    good_windows: u8,
    frame_samples: VecDeque<f32>,
    started_at: Instant,
    last_evaluation: Instant,
}

impl Default for DynamicChunkCapController {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            current_cap: ACTIVE_CHUNK_CAP,
            last_p95_ms: 0.0,
            last_change: 0,
            good_windows: 0,
            frame_samples: VecDeque::with_capacity(FRAME_SAMPLE_CAPACITY),
            started_at: now,
            last_evaluation: now,
        }
    }
}

pub struct DynamicChunkCapControllerPlugin;

impl Plugin for DynamicChunkCapControllerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DynamicChunkCapController>()
            .add_systems(PreUpdate, update_dynamic_chunk_cap);
    }
}

fn update_dynamic_chunk_cap(
    time: Res<Time>,
    settings: Res<GraphicsSettings>,
    terrain_debug: Res<TerrainDebugInfo>,
    mut terrain_state: ResMut<TerrainState>,
    mut controller: ResMut<DynamicChunkCapController>,
) {
    let frame_ms = time.delta_secs() * 1000.0;
    push_sample(&mut controller.frame_samples, frame_ms);

    if controller.last_evaluation.elapsed() < EVALUATION_INTERVAL {
        return;
    }
    controller.last_evaluation = Instant::now();
    controller.last_change = 0;

    if controller.started_at.elapsed() < STARTUP_GRACE
        || controller.frame_samples.len() < MIN_SAMPLES_PER_EVALUATION
    {
        controller.frame_samples.clear();
        return;
    }

    let p95_ms = percentile(&controller.frame_samples, 95.0);
    controller.last_p95_ms = p95_ms;
    controller.frame_samples.clear();

    let decision = decide_cap_change(
        p95_ms,
        terrain_debug.active_chunks,
        terrain_debug.pending_meshes,
        settings.vsync,
        controller.current_cap,
        controller.good_windows,
    );
    match decision {
        CapDecision::Increase => {
            controller.good_windows = 0;
            let new_cap = (controller.current_cap + INCREASE_STEP).min(MAX_CHUNK_CAP);
            controller.last_change = new_cap as i32 - controller.current_cap as i32;
            controller.current_cap = new_cap;
        }
        CapDecision::Decrease => {
            controller.good_windows = 0;
            let new_cap = controller
                .current_cap
                .saturating_sub(DECREASE_STEP)
                .max(MIN_CHUNK_CAP);
            controller.last_change = new_cap as i32 - controller.current_cap as i32;
            controller.current_cap = new_cap;
        }
        CapDecision::Hold => {
            if !settings.vsync
                && p95_ms >= 1.0
                && p95_ms < TARGET_FRAME_MS
                && terrain_debug.pending_meshes == 0
                && near_cap(terrain_debug.active_chunks, controller.current_cap, 90)
            {
                controller.good_windows = controller.good_windows.saturating_add(1);
            } else {
                controller.good_windows = 0;
            }
        }
    }

    // Terrain's split guard reads this every frame. Lowering it blocks new
    // splits but does not despawn coverage that is already active.
    terrain_state.active_chunk_cap = controller.current_cap;
}

fn decide_cap_change(
    p95_ms: f32,
    active_chunks: usize,
    pending_meshes: usize,
    vsync: bool,
    current_cap: usize,
    good_windows: u8,
) -> CapDecision {
    if p95_ms < 1.0 {
        return CapDecision::Hold;
    }
    if p95_ms > REGRESSION_FRAME_MS
        && pending_meshes == 0
        && current_cap > MIN_CHUNK_CAP
        && near_cap(active_chunks, current_cap, 70)
    {
        return CapDecision::Decrease;
    }
    if !vsync
        && p95_ms < TARGET_FRAME_MS
        && pending_meshes == 0
        && current_cap < MAX_CHUNK_CAP
        && good_windows >= 1
        && near_cap(active_chunks, current_cap, 90)
    {
        return CapDecision::Increase;
    }
    CapDecision::Hold
}

fn near_cap(active_chunks: usize, cap: usize, percent: usize) -> bool {
    active_chunks.saturating_mul(100) >= cap.saturating_mul(percent)
}

fn push_sample(samples: &mut VecDeque<f32>, frame_ms: f32) {
    if samples.len() == FRAME_SAMPLE_CAPACITY {
        samples.pop_front();
    }
    samples.push_back(frame_ms);
}

fn percentile(samples: &VecDeque<f32>, percentile: f32) -> f32 {
    let mut sorted: Vec<f32> = samples.iter().copied().collect();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let index = ((sorted.len() as f32 - 1.0) * percentile / 100.0).round() as usize;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increases_after_sustained_headroom_at_the_cap() {
        assert_eq!(
            decide_cap_change(12.0, 1_900, 0, false, 2_000, 1),
            CapDecision::Increase
        );
    }

    #[test]
    fn vsync_does_not_trigger_growth() {
        assert_eq!(
            decide_cap_change(8.0, 2_000, 0, true, 2_000, 1),
            CapDecision::Hold
        );
    }

    #[test]
    fn decreases_only_when_chunk_count_is_high() {
        assert_eq!(
            decide_cap_change(20.0, 1_500, 0, false, 2_000, 0),
            CapDecision::Decrease
        );
        assert_eq!(
            decide_cap_change(20.0, 500, 0, false, 2_000, 0),
            CapDecision::Hold
        );
    }

    #[test]
    fn mesh_generation_bursts_do_not_change_the_cap() {
        assert_eq!(
            decide_cap_change(20.0, 2_000, 1, false, 2_000, 1),
            CapDecision::Hold
        );
    }

    #[test]
    fn cap_bounds_are_respected() {
        assert_eq!(
            decide_cap_change(20.0, MIN_CHUNK_CAP, 0, false, MIN_CHUNK_CAP, 0),
            CapDecision::Hold
        );
        assert_eq!(
            decide_cap_change(8.0, MAX_CHUNK_CAP, 0, false, MAX_CHUNK_CAP, 1),
            CapDecision::Hold
        );
    }
}
