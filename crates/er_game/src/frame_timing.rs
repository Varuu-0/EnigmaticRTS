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
    pub stages: Vec<(String, Duration)>,
    /// Wall time for the completed frame represented by `stages`.
    pub frame_duration: Option<Duration>,
    stage_names: Vec<String>,
    boundaries: Vec<Instant>,
    frame_start: Option<Instant>,
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
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
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
            self.frame_duration = self.frame_start.map(|start| now - start);
            self.frame_start = Some(now);
            self.boundaries.clear();
            self.boundaries.push(now);
            return;
        }
        self.boundaries.push(Instant::now());

        if index != self.stage_names.len() || self.boundaries.len() != self.stage_names.len() + 1 {
            return;
        }

        self.stages.clear();
        for (name, bounds) in self.stage_names.iter().zip(self.boundaries.windows(2)) {
            self.stages.push((name.clone(), bounds[1] - bounds[0]));
        }
    }
}
