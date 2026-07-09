//! Water identification: sea-level fill, lake detection, depth bands.
//!
//! All functions are pure (deterministic from inputs) and intended to be
//! routed through the `WorldCache` for memoization.

use crate::elevation::{elevation, ElevationNoise, ElevationParams};
use glam::DVec3;

/// Depth of water at a point. Positive = submerged depth (how deep below
/// sea level). Zero or negative = land (not underwater).
pub fn water_depth(elevation: f64, sea_level: f64) -> f64 {
    (sea_level - elevation).max(0.0)
}

/// True if this point is underwater (below sea level).
pub fn is_ocean(elevation: f64, sea_level: f64) -> bool {
    elevation < sea_level
}

/// The surface elevation of water at a point. If below sea, the water
/// surface is at `sea_level`. If above sea (land), the surface is the
/// terrain elevation itself.
pub fn water_surface_elev(elevation: f64, sea_level: f64) -> f64 {
    if elevation < sea_level {
        sea_level
    } else {
        elevation
    }
}

/// Depth band index for ocean coloring:
/// 0 = not water, 1 = shallow, 2 = mid, 3 = deep, 4 = abyss.
pub fn ocean_depth_band(elevation: f64, sea_level: f64) -> u8 {
    if elevation >= sea_level {
        return 0;
    }
    let depth = sea_level - elevation;
    if depth < 0.3 {
        1
    } else if depth < 0.6 {
        2
    } else if depth < 1.0 {
        3
    } else {
        4
    }
}

/// Lake detection: samples neighbors to determine if a point sits in a
/// local minimum above sea level. If so, returns the spill elevation
/// (the lowest neighbor — water would fill up to this level). Returns
/// `None` if the point is not in a depression (it's on a slope or below
/// sea).
///
/// This is a simplified heuristic: it samples 8 neighbors at a fixed
/// angular distance and checks if all are higher. True watershed analysis
/// is deferred to the RTS pass.
pub fn lake_surface_elev(
    dir: DVec3,
    elevation_val: f64,
    sea_level: f64,
    noise: &ElevationNoise,
    params: &ElevationParams,
) -> Option<f64> {
    if elevation_val < sea_level {
        return None;
    }

    let eps = 0.01_f64;
    let neighbors = [
        DVec3::new(dir.x + eps, dir.y, dir.z),
        DVec3::new(dir.x - eps, dir.y, dir.z),
        DVec3::new(dir.x, dir.y + eps, dir.z),
        DVec3::new(dir.x, dir.y - eps, dir.z),
        DVec3::new(dir.x, dir.y, dir.z + eps),
        DVec3::new(dir.x, dir.y, dir.z - eps),
    ];

    let mut min_neighbor = f64::MAX;
    for n in &neighbors {
        let n_dir = n.normalize();
        let n_elev = elevation(n_dir, noise, params);
        if n_elev <= elevation_val {
            return None;
        }
        min_neighbor = min_neighbor.min(n_elev);
    }

    if min_neighbor > elevation_val {
        Some(min_neighbor)
    } else {
        None
    }
}

/// True if this point is any kind of water (ocean or lake).
pub fn is_water(
    dir: DVec3,
    elevation_val: f64,
    sea_level: f64,
    noise: &ElevationNoise,
    params: &ElevationParams,
) -> bool {
    if is_ocean(elevation_val, sea_level) {
        return true;
    }
    lake_surface_elev(dir, elevation_val, sea_level, noise, params).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elevation::{elevation_params, ElevationNoise};
    use crate::params::planet_params;
    use er_core::rng::rng_from_seed;
    use er_core::seed::PlanetSeed;
    use rand::RngCore;

    fn rand_dirs(seed: u64, count: usize) -> Vec<DVec3> {
        let mut rng = rng_from_seed(seed);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let x = (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let y = (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let z = (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let v = DVec3::new(x, y, z);
            let l2 = v.length_squared();
            if l2 > 1e-6 && l2 < 1.0 {
                out.push(v.normalize());
            }
        }
        out
    }

    #[test]
    fn water_depth_basics() {
        assert_eq!(water_depth(-0.5, 0.0), 0.5);
        assert_eq!(water_depth(0.3, 0.0), 0.0);
        assert_eq!(water_depth(0.0, 0.0), 0.0);
    }

    #[test]
    fn is_ocean_basics() {
        assert!(is_ocean(-0.5, 0.0));
        assert!(!is_ocean(0.5, 0.0));
        assert!(!is_ocean(0.0, 0.0));
    }

    #[test]
    fn depth_bands() {
        let sl = 0.0;
        assert_eq!(ocean_depth_band(0.1, sl), 0);
        assert_eq!(ocean_depth_band(-0.1, sl), 1);
        assert_eq!(ocean_depth_band(-0.4, sl), 2);
        assert_eq!(ocean_depth_band(-0.8, sl), 3);
        assert_eq!(ocean_depth_band(-1.5, sl), 4);
    }

    #[test]
    fn water_surface_elev_basics() {
        assert_eq!(water_surface_elev(-0.5, 0.0), 0.0);
        assert_eq!(water_surface_elev(0.3, 0.0), 0.3);
    }

    #[test]
    fn is_water_consistent_with_elevation() {
        let params = elevation_params(PlanetSeed(0xC0FFEE));
        let noise = ElevationNoise::new(&params);
        let pp = planet_params(PlanetSeed(0xC0FFEE));
        let sea_level = pp.sea_level;

        let dirs = rand_dirs(0x1234, 500);
        for d in &dirs {
            let elev = elevation(*d, &noise, &params);
            let water = is_water(*d, elev, sea_level, &noise, &params);
            if elev < sea_level {
                assert!(water, "elevation below sea should be water");
            }
        }
    }

    #[test]
    fn lake_detection_determinism() {
        let params = elevation_params(PlanetSeed(0xC0FFEE));
        let noise = ElevationNoise::new(&params);
        let pp = planet_params(PlanetSeed(0xC0FFEE));
        let sea_level = pp.sea_level;

        let dirs = rand_dirs(0xBEEF, 100);
        for d in &dirs {
            let elev = elevation(*d, &noise, &params);
            let lake1 = lake_surface_elev(*d, elev, sea_level, &noise, &params);
            let lake2 = lake_surface_elev(*d, elev, sea_level, &noise, &params);
            assert_eq!(lake1, lake2, "lake detection must be deterministic");
        }
    }
}
