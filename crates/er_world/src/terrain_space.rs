//! Meter-based spatial scales for the deterministic procedural terrain field.
//!
//! Noise is evaluated from a 3D point on the planet sphere, not a cube-face
//! coordinate, so its value is continuous across all cube-sphere face edges.

use glam::DVec3;

pub const CONTINENTAL_WAVELENGTH_M: f64 = 1_200_000.0;
pub const MOUNTAIN_WAVELENGTH_M: f64 = 120_000.0;
pub const FOOTHILL_WAVELENGTH_M: f64 = 8_000.0;
pub const RIDGE_WAVELENGTH_M: f64 = 900.0;
pub const WARP_WAVELENGTH_M: f64 = 700_000.0;

/// Converts a unit surface direction into a continuous metric coordinate for
/// 3D noise. The same direction has the same coordinate regardless of which
/// cube face generated it.
#[inline]
pub fn metric_surface_point(dir: DVec3, planet_radius_m: f64) -> DVec3 {
    dir.normalize() * planet_radius_m
}

#[inline]
pub fn vertex_spacing_m(chunk_width_m: f64, quads_per_edge: u32) -> f64 {
    chunk_width_m / quads_per_edge.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::math::uv_to_dir;

    #[test]
    fn shared_cube_edge_has_one_metric_coordinate() {
        let a = metric_surface_point(uv_to_dir(0, 1.0, 0.5), 6_371_000.0);
        let b = metric_surface_point(uv_to_dir(2, 1.0, 0.5), 6_371_000.0);
        assert!((a - b).length() < 1e-6);
    }

    #[test]
    fn vertex_spacing_is_a_metric_measurement() {
        assert_eq!(vertex_spacing_m(80.0, 16), 5.0);
    }
}
