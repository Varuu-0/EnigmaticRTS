use er_core::config::LOD_PIXEL_SCALE;
use er_core::math::{cell_size, cell_to_dir, CellKey};
use glam::DVec3;

/// Approximate screen-space error in **pixels** for a chunk at `key`, given the
/// camera position. Computed as `(chunk_size / distance) * pixel_scale` where
/// `pixel_scale = viewport_height / (2 * tan(fov/2))` (see `LOD_PIXEL_SCALE`).
/// A threshold of ~20px means "split when a chunk's geometric error exceeds 20
/// pixels on screen."
pub fn screen_error(key: CellKey, camera_pos: DVec3, planet_radius: f64) -> f32 {
    let chunk_center = cell_to_dir(key) * planet_radius;
    let distance = (chunk_center - camera_pos).length().max(1.0);
    let chunk_size = cell_size(key.lod, planet_radius);
    ((chunk_size / distance) as f32) * LOD_PIXEL_SCALE
}

pub fn should_split(
    key: CellKey,
    camera_pos: DVec3,
    planet_radius: f64,
    max_depth: u8,
    threshold: f32,
) -> bool {
    key.lod < max_depth && screen_error(key, camera_pos, planet_radius) > threshold
}

pub fn should_merge_parent(
    parent_key: CellKey,
    camera_pos: DVec3,
    planet_radius: f64,
    threshold: f32,
    hysteresis: f32,
) -> bool {
    let merge_threshold = threshold * hysteresis;
    screen_error(parent_key, camera_pos, planet_radius) < merge_threshold
}

pub fn chunk_camera_distance(key: CellKey, camera_pos: DVec3, planet_radius: f64) -> f64 {
    let chunk_center = cell_to_dir(key) * planet_radius;
    (chunk_center - camera_pos).length()
}
