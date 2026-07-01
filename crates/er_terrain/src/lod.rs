use er_core::math::{cell_size, cell_to_dir, CellKey};
use glam::DVec3;

pub fn screen_error(key: CellKey, camera_pos: DVec3, planet_radius: f64) -> f32 {
    let chunk_center = cell_to_dir(key) * planet_radius;
    let distance = (chunk_center - camera_pos).length().max(1.0);
    let chunk_size = cell_size(key.lod, planet_radius);
    (chunk_size / distance) as f32
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
    let children = crate::quadtree::children_of(parent_key);
    children
        .iter()
        .all(|c| screen_error(*c, camera_pos, planet_radius) < merge_threshold)
}

pub fn chunk_camera_distance(key: CellKey, camera_pos: DVec3, planet_radius: f64) -> f64 {
    let chunk_center = cell_to_dir(key) * planet_radius;
    (chunk_center - camera_pos).length()
}
