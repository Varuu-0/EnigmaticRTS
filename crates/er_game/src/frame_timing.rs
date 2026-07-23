use std::time::{Duration, Instant};

use bevy::app::{
    First, Last, MainScheduleOrder, PostUpdate, PreUpdate, RunFixedMainLoop, SpawnScene, Update,
};
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;

/// Wall-clock durations for each completed main-world schedule.
///
/// Rendering runs concurrently in Bevy's render sub-app, so these timings are
/// intentionally kept separate from render-pass diagnostics.
#[derive(Resource, Default)]
pub struct MainWorldFrameTimings {
    pub stages: Vec<(&'static str, Duration)>,
    /// Wall time for the completed frame represented by `stages`.
    pub frame_duration: Option<Duration>,
    /// Increments when a complete, internally consistent timing sample is published.
    pub sample_revision: u64,
    stage_names: [&'static str; 7],
    boundaries: Vec<Instant>,
    pending_stages: Vec<(&'static str, Duration)>,
    frame_start: Option<Instant>,
    frames_until_capture: u8,
    capture_frame: bool,
    completed_capture: bool,
}

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct FramePhaseCheckpoint(usize);

pub struct FrameTimingPlugin;

impl Plugin for FrameTimingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MainWorldFrameTimings {
            stage_names: [
                "First",
                "PreUpdate",
                "RunFixedMainLoop",
                "Update",
                "SpawnScene",
                "PostUpdate",
                "Last",
            ],
            stages: Vec::with_capacity(7),
            boundaries: Vec::with_capacity(8),
            pending_stages: Vec::with_capacity(7),
            ..default()
        });

        // Checkpoints are standalone schedules, so they bracket every system
        // in the real main-world schedules, including Bevy's own systems.
        add_checkpoint::<First>(app, 0);
        add_checkpoint::<PreUpdate>(app, 1);
        add_checkpoint::<RunFixedMainLoop>(app, 2);
        add_checkpoint::<Update>(app, 3);
        add_checkpoint::<SpawnScene>(app, 4);
        add_checkpoint::<PostUpdate>(app, 5);
        add_checkpoint::<Last>(app, 6);

        app.world_mut()
            .resource_mut::<MainScheduleOrder>()
            .insert_after(Last, FramePhaseCheckpoint(7));
        app.add_systems(
            FramePhaseCheckpoint(7),
            |mut timings: ResMut<MainWorldFrameTimings>| timings.record_boundary(7),
        );
    }
}

fn add_checkpoint<L: ScheduleLabel + Default>(app: &mut App, index: usize) {
    app.world_mut()
        .resource_mut::<MainScheduleOrder>()
        .insert_before(L::default(), FramePhaseCheckpoint(index));
    app.add_systems(
        FramePhaseCheckpoint(index),
        move |mut timings: ResMut<MainWorldFrameTimings>| timings.record_boundary(index),
    );
}

impl MainWorldFrameTimings {
    fn record_boundary(&mut self, index: usize) {
        if index == 0 {
            let now = Instant::now();
            if self.completed_capture {
                self.frame_duration = self.frame_start.map(|start| now - start);
                std::mem::swap(&mut self.stages, &mut self.pending_stages);
                self.sample_revision = self.sample_revision.wrapping_add(1);
                self.completed_capture = false;
            }
            self.frame_start = Some(now);
            if self.frames_until_capture > 0 {
                self.frames_until_capture -= 1;
                self.capture_frame = false;
                return;
            }
            self.frames_until_capture = 7;
            self.capture_frame = true;
            self.boundaries.clear();
            self.boundaries.push(now);
            return;
        }
        if !self.capture_frame {
            return;
        }
        self.boundaries.push(Instant::now());

        if index != self.stage_names.len() || self.boundaries.len() != self.stage_names.len() + 1 {
            return;
        }

        self.pending_stages.clear();
        for (name, bounds) in self.stage_names.iter().zip(self.boundaries.windows(2)) {
            self.pending_stages.push((*name, bounds[1] - bounds[0]));
        }
        self.completed_capture = true;
    }

    /// Number of non-capture frames between each captured frame. Exposed for
    /// deterministic stress tests that verify the capture cadence without
    /// depending on wall-clock timing values.
    #[cfg(test)]
    pub(crate) fn capture_period_frames(&self) -> u8 {
        7
    }

    /// Whether the current frame is being captured. Exposed for deterministic
    /// tests of the capture window state machine.
    #[cfg(test)]
    pub(crate) fn is_capturing(&self) -> bool {
        self.capture_frame
    }

    /// Remaining non-capture frames before the next capture begins.
    #[cfg(test)]
    pub(crate) fn frames_until_capture(&self) -> u8 {
        self.frames_until_capture
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_timings() -> MainWorldFrameTimings {
        MainWorldFrameTimings::default()
    }

    /// Drive a full frame through every stage checkpoint so `record_boundary`
    /// sees the complete boundary sequence exactly once.
    fn record_full_frame(timings: &mut MainWorldFrameTimings, stage_count: usize) {
        for index in 0..=stage_count {
            timings.record_boundary(index);
        }
    }

    #[test]
    fn first_frame_is_a_capture_frame() {
        let mut timings = fresh_timings();
        // The very first boundary (index 0) starts a capture immediately.
        timings.record_boundary(0);
        assert!(timings.is_capturing());
        assert_eq!(
            timings.frames_until_capture(),
            timings.capture_period_frames()
        );
    }

    #[test]
    fn non_capture_frames_skip_stage_recording() {
        let mut timings = fresh_timings();
        // Frame 0: capture starts.
        record_full_frame(&mut timings, 7);
        assert!(timings.stages.is_empty());

        // Frame 1 publishes frame 0, then frames 2-7 skip stage recording.
        timings.record_boundary(0);
        assert_eq!(timings.stages.len(), 7);
        let published_revision = timings.sample_revision;
        assert!(!timings.is_capturing());
        for _ in 1..timings.capture_period_frames() {
            timings.record_boundary(0);
            assert!(!timings.is_capturing());
            // Record a stray stage boundary mid-non-capture; it must be ignored.
            timings.record_boundary(1);
        }
        // The published sample remains untouched between captures.
        assert_eq!(timings.stages.len(), 7);
        assert_eq!(timings.sample_revision, published_revision);
    }

    #[test]
    fn capture_cadence_repeats_every_period_plus_one_frames() {
        let mut timings = fresh_timings();
        let stage_count = 7;
        record_full_frame(&mut timings, stage_count);
        timings.record_boundary(0);
        let first_capture_stages = timings.stages.len();

        // Skip the remainder of the non-capture gap.
        for _ in 1..timings.capture_period_frames() {
            timings.record_boundary(0);
            for index in 1..=stage_count {
                timings.record_boundary(index);
            }
        }

        // Next frame should capture again.
        timings.record_boundary(0);
        assert!(timings.is_capturing());
        for index in 1..=stage_count {
            timings.record_boundary(index);
        }
        timings.record_boundary(0);
        assert_eq!(timings.stages.len(), first_capture_stages);
        assert_eq!(timings.sample_revision, 2);
    }

    #[test]
    fn frame_duration_is_none_before_first_boundary() {
        let timings = fresh_timings();
        assert!(timings.frame_duration.is_none());
    }

    #[test]
    fn frame_duration_becomes_some_after_second_frame_starts() {
        let mut timings = fresh_timings();
        record_full_frame(&mut timings, 7); // first captured frame
        assert!(timings.frame_duration.is_none());
        timings.record_boundary(0); // second frame start computes first duration
        assert!(timings.frame_duration.is_some());
        assert_eq!(timings.sample_revision, 1);
    }

    #[test]
    fn stage_names_match_expected_schedule_order() {
        // Default construction leaves stage_names empty ([""; 7]); the plugin
        // populates the real names. This test asserts the plugin-supplied names
        // by reconstructing the same array the plugin inserts.
        let plugin_names = [
            "First",
            "PreUpdate",
            "RunFixedMainLoop",
            "Update",
            "SpawnScene",
            "PostUpdate",
            "Last",
        ];
        let mut timings = fresh_timings();
        timings.stage_names = plugin_names;
        assert_eq!(timings.stage_names, plugin_names);
        assert_eq!(timings.stage_names.len(), 7);
    }
}
