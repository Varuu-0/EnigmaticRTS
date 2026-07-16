//! Tunable configuration constants for the planet/terrain system.

/// The original small planet is retained for fast visual diagnostics. It is
/// not a hidden scale factor for the Earth-scale preset.
pub const MINIATURE_PLANET_RADIUS_M: f64 = 36_000.0;
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;
pub const PLANET_RADIUS_DEFAULT: f64 = MINIATURE_PLANET_RADIUS_M;
pub const MAX_QUADTREE_DEPTH: u8 = 12;
pub const EARTH_MAX_QUADTREE_DEPTH: u8 = 17;
pub const CHUNK_VERT_RES: u32 = 17;
pub const CHUNK_QUADS_PER_EDGE: u32 = 16; // CHUNK_VERT_RES - 1
pub const LOD_SPLIT_BUDGET_PER_FRAME: usize = 48;
// Split once a chunk's projected geometric error exceeds 40 px. This keeps the
// default orbit dense enough to render a continuous planet disk instead of the
// sparse cube-face shell produced by coarser LODs.
pub const SCREEN_ERROR_THRESHOLD: f32 = 40.0;
pub const MERGE_HYSTERESIS: f32 = 0.6;
pub const MAX_RENDER_DISTANCE: f64 = PLANET_RADIUS_DEFAULT * 8.0;
// Never evict live terrain to satisfy this limit. Stop splitting before the
// cap instead, preserving complete coverage without unbounded close-view cost.
pub const ACTIVE_CHUNK_CAP: usize = 5000;
// The six cube-face roots are the persistent coarse coverage floor. They bypass
// only the distance cull at extreme zoom-out; horizon and frustum culling still
// prevent rendering the planet's hidden side.
pub const MINIMUM_TERRAIN_COVERAGE_LOD: u8 = 0;

/// Converts angular-size ratio (chunk_size / distance) to approximate pixel
/// error: viewport_height / (2 * tan(fov/2)) for 1080p + 60° FOV.
/// `1080 / (2 * tan(30°)) ≈ 935`. screen_error returns pixels; the threshold
/// is in the same unit (e.g. 20 = split when a chunk exceeds ~20px of error).
pub const LOD_PIXEL_SCALE: f32 = 935.0;
pub const FIXED_TPS: i32 = 30;
pub const DEFAULT_DAY_LENGTH_SEC: f64 = 180.0;

/// Explicit planet-scale configurations. The miniature planet remains the
/// default until camera-relative rendering is complete; `EarthScale` is
/// selected with the game's `--earth-scale` flag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlanetPreset {
    #[default]
    MiniatureDebug,
    EarthScale,
}

impl PlanetPreset {
    pub const fn radius_m(self) -> f64 {
        match self {
            Self::MiniatureDebug => MINIATURE_PLANET_RADIUS_M,
            Self::EarthScale => EARTH_RADIUS_M,
        }
    }

    pub const fn max_quadtree_depth(self) -> u8 {
        match self {
            Self::MiniatureDebug => MAX_QUADTREE_DEPTH,
            Self::EarthScale => EARTH_MAX_QUADTREE_DEPTH,
        }
    }

    pub const fn max_render_distance_m(self) -> f64 {
        self.radius_m() * 8.0
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_preset_uses_physical_radius_and_close_lod() {
        assert_eq!(PlanetPreset::EarthScale.radius_m(), EARTH_RADIUS_M);
        assert_eq!(PlanetPreset::EarthScale.max_quadtree_depth(), 17);
    }

    #[test]
    fn miniature_preset_preserves_existing_default() {
        assert_eq!(PlanetPreset::default(), PlanetPreset::MiniatureDebug);
        assert_eq!(PlanetPreset::default().radius_m(), PLANET_RADIUS_DEFAULT);
    }
}
