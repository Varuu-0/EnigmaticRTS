//! Tunable configuration constants for the planet/terrain system.

pub const PLANET_RADIUS_DEFAULT: f64 = 12000.0;
pub const MAX_QUADTREE_DEPTH: u8 = 12;
pub const CHUNK_VERT_RES: u32 = 17;
pub const CHUNK_QUADS_PER_EDGE: u32 = 16; // CHUNK_VERT_RES - 1
pub const LOD_SPLIT_BUDGET_PER_FRAME: usize = 4;
pub const SCREEN_ERROR_THRESHOLD: f32 = 3.0;
pub const MERGE_HYSTERESIS: f32 = 0.8;
pub const MAX_RENDER_DISTANCE: f64 = PLANET_RADIUS_DEFAULT * 8.0;
pub const ACTIVE_CHUNK_CAP: usize = 512;
pub const FIXED_TPS: i32 = 30;
pub const DEFAULT_DAY_LENGTH_SEC: f64 = 180.0;

/// Runtime-tunable parameters (defaults mirror the `const`s above). Game crates
/// wrap this in a Bevy `Resource`; `er_core` itself stays Bevy-free.
#[derive(Clone, Debug)]
pub struct Tunables {
    pub planet_radius: f64,
    pub max_quadtree_depth: u8,
    pub screen_error_threshold: f32,
    pub merge_hysteresis: f32,
    pub max_render_distance: f64,
    pub active_chunk_cap: usize,
    pub lod_split_budget_per_frame: usize,
    pub fixed_tps: i32,
    pub default_day_length_sec: f64,
}

impl Default for Tunables {
    fn default() -> Self {
        Self {
            planet_radius: PLANET_RADIUS_DEFAULT,
            max_quadtree_depth: MAX_QUADTREE_DEPTH,
            screen_error_threshold: SCREEN_ERROR_THRESHOLD,
            merge_hysteresis: MERGE_HYSTERESIS,
            max_render_distance: MAX_RENDER_DISTANCE,
            active_chunk_cap: ACTIVE_CHUNK_CAP,
            lod_split_budget_per_frame: LOD_SPLIT_BUDGET_PER_FRAME,
            fixed_tps: FIXED_TPS,
            default_day_length_sec: DEFAULT_DAY_LENGTH_SEC,
        }
    }
}
