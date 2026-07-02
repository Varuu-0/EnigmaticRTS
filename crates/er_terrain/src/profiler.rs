use bevy::ecs::resource::Resource;
use std::time::{Duration, Instant};

#[derive(Resource)]
pub struct FrameProfiler {
    pub timings: Vec<(&'static str, Duration)>,
    frame_start: Instant,
}

impl Default for FrameProfiler {
    fn default() -> Self {
        Self {
            timings: Vec::with_capacity(16),
            frame_start: Instant::now(),
        }
    }
}

impl FrameProfiler {
    pub fn record(&mut self, name: &'static str, duration: Duration) {
        self.timings.push((name, duration));
    }

    pub fn clear(&mut self) {
        self.timings.clear();
        self.frame_start = Instant::now();
    }

    pub fn total(&self) -> Duration {
        self.frame_start.elapsed()
    }
}
