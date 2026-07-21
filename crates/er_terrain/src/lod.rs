use er_core::config::LOD_PIXEL_SCALE;
use er_core::math::{cell_size, cell_to_dir, CellKey};
use glam::DVec3;

/// Approximate screen-space error in **pixels** for a chunk at `key`, given the
/// camera position. Computed as `(chunk_size / distance) * pixel_scale` where
/// `pixel_scale = viewport_height / (2 * tan(fov/2))` (see `LOD_PIXEL_SCALE`).
/// The chunk width remains the LOD geometric-error proxy: per-vertex spacing
/// under-refines the full visible terrain surface at normal viewing altitudes.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key(lod: u8) -> CellKey {
        CellKey {
            face: 0,
            i: 0,
            j: 0,
            lod,
        }
    }

    #[test]
    fn screen_error_falls_with_distance() {
        let radius = 36_000.0;
        let near = screen_error(key(4), DVec3::new(radius + 100.0, 0.0, 0.0), radius);
        let far = screen_error(key(4), DVec3::new(radius * 8.0, 0.0, 0.0), radius);
        assert!(near > far);
    }

    #[test]
    fn split_and_merge_have_hysteresis() {
        let radius = 36_000.0;
        let camera = DVec3::new(radius * 4.0, 0.0, 0.0);
        let error = screen_error(key(2), camera, radius);
        assert!(!should_split(key(2), camera, radius, 2, 0.0));
        assert!(!should_split(key(2), camera, radius, 10, error));
        assert!(!should_merge_parent(key(2), camera, radius, error, 0.6));
    }
}

#[cfg(test)]
mod deterministic_tests {
    use super::*;

    const RADIUS: f64 = 36_000.0;

    fn key(lod: u8) -> CellKey {
        CellKey {
            face: 0,
            i: 0,
            j: 0,
            lod,
        }
    }

    #[test]
    fn screen_error_is_monotonic_for_distance_and_lod() {
        let near = DVec3::new(RADIUS * 1.1, 0.0, 0.0);
        let far = DVec3::new(RADIUS * 8.0, 0.0, 0.0);
        assert!(screen_error(key(2), near, RADIUS) > screen_error(key(2), far, RADIUS));
        assert!(screen_error(key(2), near, RADIUS) > screen_error(key(3), near, RADIUS));
    }

    #[test]
    fn split_respects_threshold_and_depth_limit() {
        let camera = DVec3::new(RADIUS * 1.01, 0.0, 0.0);
        let error = screen_error(key(2), camera, RADIUS);
        assert!(should_split(key(2), camera, RADIUS, 3, error * 0.5));
        assert!(!should_split(key(2), camera, RADIUS, 2, 0.0));
        assert!(!should_split(key(2), camera, RADIUS, 3, error));
    }

    #[test]
    fn merge_uses_a_lower_hysteresis_threshold() {
        let camera = DVec3::new(RADIUS * 8.0, 0.0, 0.0);
        let error = screen_error(key(2), camera, RADIUS);
        assert!(should_merge_parent(
            key(2),
            camera,
            RADIUS,
            error * 2.0,
            1.0
        ));
        assert!(!should_merge_parent(
            key(2),
            camera,
            RADIUS,
            error * 2.0,
            0.25
        ));
    }

    #[test]
    fn camera_distance_matches_the_screen_error_geometry() {
        let camera = DVec3::new(RADIUS * 2.0, 0.0, 0.0);
        let distance = chunk_camera_distance(key(1), camera, RADIUS);
        assert!(distance.is_finite() && distance > 0.0);
        let expected = (er_core::math::cell_size(1, RADIUS) / distance) as f32 * LOD_PIXEL_SCALE;
        assert_eq!(screen_error(key(1), camera, RADIUS), expected);
    }

    #[test]
    fn close_earth_error_uses_chunk_width_geometric_proxy() {
        let radius = 6_371_000.0;
        let camera = DVec3::X * (radius + 200.0);
        let lod16 = er_core::math::dir_to_cell(DVec3::X, 16);
        let distance = chunk_camera_distance(lod16, camera, radius);
        let whole_chunk = cell_size(lod16.lod, radius);
        assert_eq!(
            screen_error(lod16, camera, radius),
            (whole_chunk / distance) as f32 * LOD_PIXEL_SCALE
        );
    }
}
