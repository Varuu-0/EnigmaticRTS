//! Meter-based spatial scales for the deterministic procedural terrain field.
//!
//! Noise is evaluated from a 3D point on the planet sphere, not a cube-face
//! coordinate, so its value is continuous across all cube-sphere face edges.
//!
//! Wavelengths and amplitudes are centralized here so that the CPU elevation
//! functions, the biome split functions, and the WGSL parity shader all
//! reference a single source of truth.

use glam::DVec3;

// ---- Named metric wavelengths (meters per cycle) ----

pub const CONTINENTAL_WAVELENGTH_M: f64 = 1_200_000.0;
pub const MOUNTAIN_WAVELENGTH_M: f64 = 120_000.0;
pub const FOOTHILL_WAVELENGTH_M: f64 = 8_000.0;
pub const RIDGE_WAVELENGTH_M: f64 = 900.0;
pub const WARP_WAVELENGTH_M: f64 = 700_000.0;
/// Maximum metric domain-warp displacement before the seeded per-planet
/// `warp_amp` multiplier is applied.
pub const METRIC_WARP_AMP_M: f64 = WARP_WAVELENGTH_M * 0.1;

// ---- Landform mask wavelengths (Milestone 2.2) ----
//
// Each wavelength is chosen so its layer occupies a distinct spatial band
// without aliasing its neighbours.  All are sampled from the same
// continuous 3D metric surface point, preserving cube-edge continuity.

/// Broad tectonic uplift / subsidence that shifts where mountain belts
/// and continental shelves form.  Larger than the continental wavelength
/// so it modulates the macro field without dominating shoreline detail.
pub const TECTONIC_BELT_WAVELENGTH_M: f64 = 2_500_000.0;

/// Regional drainage / catchment approximation — medium-frequency
/// displacement representing broad river-basin relief.
pub const DRAINAGE_WAVELENGTH_M: f64 = 50_000.0;

/// Erosion-intensity mask wavelength.  Controls where valleys deepen and
/// ridges smooth.  Output is remapped to [0, 1] before masking other
/// layers, so this is a control field, not a direct displacement.
pub const EROSION_WAVELENGTH_M: f64 = 15_000.0;

/// Valley / canyon channel wavelength.  Ridged noise carved proportionally
/// to the erosion mask.
pub const VALLEY_WAVELENGTH_M: f64 = 6_000.0;

/// Ridge-detail wavelength.  Sharp ridge lines at a slightly broader scale
/// than the existing value-noise detail (900 m) to avoid duplication.
pub const RIDGE_DETAIL_WAVELENGTH_M: f64 = 2_500.0;

/// Slope-limited talus roughness wavelength.  High-frequency roughness
/// gated by a low-cost slope proxy so it appears mainly on steep terrain.
pub const TALUS_WAVELENGTH_M: f64 = 1_200.0;

/// Fine residual geometry wavelength. Material shading adds a separate
/// screen-filtered detail band below this scale.
pub const MICRO_DETAIL_WAVELENGTH_M: f64 = 200.0;

// ---- Named metric amplitudes (field-unit displacement) ----
//
// The signed theoretical bounds are approximately -8.4 to +12.3 field units,
// or -8.4 km to +12.3 km at the Earth preset's 1000 m elevation scale. Actual
// extrema are lower because independent layers do not peak together.

// --- Macro (low-frequency) amplitudes ---

pub const METRIC_CONTINENTAL_AMP: f64 = 7.0;
pub const METRIC_MOUNTAIN_AMP: f64 = 2.5;
/// Bounded broad macro displacement from the tectonic belt mask [0,1].
pub const METRIC_TECTONIC_AMP: f64 = 1.5;

// --- Residual (high-frequency) amplitudes ---
//
// Ridge uplift is nonnegative (0..amp).  Valley and drainage are subtracted
// (carving), so they only reduce elevation.  Talus is signed roughness gated
// by a steep-slope proxy.  Hill and detail are bidirectional.

pub const METRIC_HILL_AMP: f64 = 0.8;
pub const METRIC_DETAIL_AMP: f64 = 0.02;
/// Nonnegative ridge uplift (0..amp).
pub const METRIC_RIDGE_DETAIL_AMP: f64 = 0.3;
/// Maximum valley/canyon carving depth (subtracted, 0..amp).
pub const METRIC_VALLEY_AMP: f64 = 0.2;
/// Maximum drainage channel carving depth (subtracted, 0..amp).
pub const METRIC_DRAINAGE_AMP: f64 = 0.2;
/// Signed talus roughness magnitude, gated by steep-slope proxy.
pub const METRIC_TALUS_AMP: f64 = 0.08;

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

    #[test]
    fn named_wavelengths_are_ordered_and_positive() {
        assert!(TECTONIC_BELT_WAVELENGTH_M > CONTINENTAL_WAVELENGTH_M);
        assert!(CONTINENTAL_WAVELENGTH_M > WARP_WAVELENGTH_M);
        assert!(WARP_WAVELENGTH_M > MOUNTAIN_WAVELENGTH_M);
        assert!(MOUNTAIN_WAVELENGTH_M > DRAINAGE_WAVELENGTH_M);
        assert!(DRAINAGE_WAVELENGTH_M > EROSION_WAVELENGTH_M);
        assert!(EROSION_WAVELENGTH_M > FOOTHILL_WAVELENGTH_M);
        assert!(FOOTHILL_WAVELENGTH_M > VALLEY_WAVELENGTH_M);
        assert!(VALLEY_WAVELENGTH_M > RIDGE_DETAIL_WAVELENGTH_M);
        assert!(RIDGE_DETAIL_WAVELENGTH_M > TALUS_WAVELENGTH_M);
        assert!(TALUS_WAVELENGTH_M > RIDGE_WAVELENGTH_M);
        assert!(RIDGE_WAVELENGTH_M > MICRO_DETAIL_WAVELENGTH_M);
        assert!(MICRO_DETAIL_WAVELENGTH_M > 0.0);
    }

    #[test]
    fn metric_amplitudes_have_expected_values() {
        assert_eq!(METRIC_CONTINENTAL_AMP, 7.0);
        assert_eq!(METRIC_MOUNTAIN_AMP, 2.5);
        assert_eq!(METRIC_TECTONIC_AMP, 1.5);
        assert_eq!(METRIC_HILL_AMP, 0.8);
        assert_eq!(METRIC_DETAIL_AMP, 0.02);
        assert_eq!(METRIC_RIDGE_DETAIL_AMP, 0.3);
        assert_eq!(METRIC_VALLEY_AMP, 0.2);
        assert_eq!(METRIC_DRAINAGE_AMP, 0.2);
        assert_eq!(METRIC_TALUS_AMP, 0.08);
    }

    #[test]
    fn metric_amplitudes_match_signed_theoretical_bounds() {
        let max_displacement = METRIC_CONTINENTAL_AMP
            + METRIC_MOUNTAIN_AMP
            + METRIC_TECTONIC_AMP
            + METRIC_HILL_AMP
            + METRIC_DETAIL_AMP
            + METRIC_RIDGE_DETAIL_AMP
            + METRIC_TALUS_AMP;
        let min_displacement = -METRIC_CONTINENTAL_AMP
            - METRIC_HILL_AMP
            - METRIC_DETAIL_AMP
            - METRIC_VALLEY_AMP
            - METRIC_DRAINAGE_AMP
            - METRIC_TALUS_AMP;
        assert!((max_displacement - 12.2).abs() < 1e-9);
        assert!((min_displacement + 8.3).abs() < 1e-9);
    }

    #[test]
    fn micro_detail_is_smaller_than_ridge() {
        assert!(MICRO_DETAIL_WAVELENGTH_M < RIDGE_WAVELENGTH_M);
        assert!(MICRO_DETAIL_WAVELENGTH_M > 0.0);
    }
}
