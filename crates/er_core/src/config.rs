//! Tunable configuration constants for the planet/terrain system.

pub const PLANET_RADIUS_DEFAULT: f64 = 36000.0;
pub const MAX_QUADTREE_DEPTH: u8 = 12;
pub const CHUNK_VERT_RES: u32 = 17;
pub const CHUNK_QUADS_PER_EDGE: u32 = 16; // CHUNK_VERT_RES - 1
pub const LOD_SPLIT_BUDGET_PER_FRAME: usize = 48;
// Terrain displacement is at most a few percent of the planet radius, so the
// previous 150px threshold over-tessellated distant curved terrain. A 1600px
// geometric-error target keeps the silhouette continuous while avoiding
// thousands of small draw calls near the surface.
pub const SCREEN_ERROR_THRESHOLD: f32 = 1600.0;
pub const MERGE_HYSTERESIS: f32 = 0.5;
pub const MAX_RENDER_DISTANCE: f64 = PLANET_RADIUS_DEFAULT * 8.0;
pub const ACTIVE_CHUNK_CAP: usize = 4000;

/// Converts angular-size ratio (chunk_size / distance) to approximate pixel
/// error: viewport_height / (2 * tan(fov/2)) for 1080p + 60° FOV.
/// `1080 / (2 * tan(30°)) ≈ 935`. screen_error returns pixels; the threshold
/// is in the same unit (e.g. 20 = split when a chunk exceeds ~20px of error).
pub const LOD_PIXEL_SCALE: f32 = 935.0;
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
