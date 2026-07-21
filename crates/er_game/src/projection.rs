//! Reusable camera projection near/far policy.
//!
//! Centralizes the near/far plane selection so it can be validated structurally
//! at miniature, Earth-orbit, and Earth-close altitudes without depending on
//! the interactive camera controller. The policy keeps far planes large enough
//! to render the whole planet from orbit while keeping the near plane stable
//! enough to avoid z-fighting at close range, and never shrinks far below the
//! render distance required for the six cube-face root coverage.

use bevy::camera::{PerspectiveProjection, Projection};
use bevy::prelude::*;
use er_core::config::PlanetPreset;

/// Field of view shared by all presets. 60 degrees matches the original setup
/// and keeps the LOD pixel-error scale in `er_core::config::LOD_PIXEL_SCALE`
/// valid.
pub const PROJECTION_FOV_RADIANS: f32 = 60.0_f32.to_radians();

/// Near plane used when no terrain-relative altitude is known yet. Matches the
/// legacy default so the first frame is identical before the camera updates.
const DEFAULT_NEAR_M: f32 = 1.0;

/// Minimum far plane so the six cube-face roots stay visible at extreme
/// zoom-out regardless of preset.
const MIN_FAR_M: f32 = 500_000.0;

/// Near plane floor at close range. Going below 0.1 m at Earth scale wastes
/// depth precision without any visible benefit.
const CLOSE_NEAR_FLOOR_M: f32 = 0.1;

/// Computed projection policy for a given camera placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectionPolicy {
    pub near: f32,
    pub far: f32,
    pub fov_radians: f32,
}

impl ProjectionPolicy {
    /// Resolve a near/far pair for a planet of the given radius and an optional
    /// terrain-relative camera altitude in meters.
    ///
    /// `altitude_m` is the camera height above the local composed terrain
    /// surface. When `None`, the legacy default near plane is used and the far
    /// plane is sized from the radius alone (suitable for the initial spawn
    /// before the camera has placed itself).
    pub fn for_placement(planet_radius_m: f64, altitude_m: Option<f64>) -> Self {
        let radius_f32 = planet_radius_m as f32;
        let far = (radius_f32 * 50.0).max(MIN_FAR_M);
        let near = match altitude_m {
            None => DEFAULT_NEAR_M,
            Some(altitude) => {
                // Keep the near plane a small fraction of the altitude so close
                // exploration does not z-fight, but never below the floor.
                let proportional = (altitude as f32 * 0.01).max(CLOSE_NEAR_FLOOR_M);
                proportional.min(DEFAULT_NEAR_M)
            }
        };
        Self {
            near,
            far,
            fov_radians: PROJECTION_FOV_RADIANS,
        }
    }

    /// Build a Bevy `Projection` from this policy.
    pub fn to_projection(self) -> Projection {
        Projection::Perspective(PerspectiveProjection {
            near: self.near,
            far: self.far,
            fov: self.fov_radians,
            ..default()
        })
    }
}

/// Convenience wrapper that takes a `PlanetPreset` plus optional altitude.
#[allow(dead_code)]
pub fn projection_for_preset(preset: PlanetPreset, altitude_m: Option<f64>) -> ProjectionPolicy {
    ProjectionPolicy::for_placement(preset.radius_m(), altitude_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::config::{EARTH_RADIUS_M, MINIATURE_PLANET_RADIUS_M};
    fn earth_radius() -> f64 {
        EARTH_RADIUS_M
    }

    // ---- Roadmap 0.2.2: miniature, Earth orbit, Earth close altitude ----

    #[test]
    fn miniature_debug_far_covers_render_distance_and_near_is_default() {
        let policy = projection_for_preset(PlanetPreset::MiniatureDebug, None);
        // Far must at least cover the preset render distance (radius * 8).
        let render_distance = PlanetPreset::MiniatureDebug.max_render_distance_m() as f32;
        assert_eq!(
            PlanetPreset::MiniatureDebug.radius_m(),
            MINIATURE_PLANET_RADIUS_M
        );
        assert!(
            policy.far >= render_distance,
            "far {} < render distance {}",
            policy.far,
            render_distance
        );
        assert_eq!(policy.near, DEFAULT_NEAR_M);
        assert_eq!(policy.fov_radians, PROJECTION_FOV_RADIANS);
    }

    #[test]
    fn earth_orbit_far_covers_planet_disk_from_six_radii() {
        // Globe scenario is at radius * 6.0; far must extend past that.
        let orbit_altitude = earth_radius() * 5.0; // ~5 radii above surface
        let policy = projection_for_preset(PlanetPreset::EarthScale, Some(orbit_altitude));
        let orbit_distance = (earth_radius() * 6.0) as f32;
        assert!(
            policy.far > orbit_distance,
            "far {} must exceed globe distance {}",
            policy.far,
            orbit_distance
        );
        // Earth far is radius * 50 which is well beyond the globe view.
        assert_eq!(policy.far, (earth_radius() as f32) * 50.0);
    }

    #[test]
    fn earth_close_altitude_near_does_not_z_fight_at_ten_meters() {
        let policy = projection_for_preset(PlanetPreset::EarthScale, Some(10.0));
        // 1% of 10 m = 0.1 m, which is the close floor.
        assert_eq!(policy.near, CLOSE_NEAR_FLOOR_M);
        assert!(policy.far >= MIN_FAR_M);
    }

    #[test]
    fn earth_close_altitude_near_scales_with_altitude_in_mid_range() {
        // At 1 km, 1% = 10 m near plane, capped at the default 1.0 m.
        let policy = projection_for_preset(PlanetPreset::EarthScale, Some(1_000.0));
        assert_eq!(policy.near, DEFAULT_NEAR_M);
        // At 50 m, 1% = 0.5 m, which is above the floor and below the cap.
        let policy = projection_for_preset(PlanetPreset::EarthScale, Some(50.0));
        assert_eq!(policy.near, 0.5);
    }

    #[test]
    fn far_never_drops_below_minimum_regardless_of_radius() {
        let tiny_radius = 1.0_f64;
        let policy = ProjectionPolicy::for_placement(tiny_radius, None);
        assert_eq!(policy.far, MIN_FAR_M);
    }

    #[test]
    fn near_floors_at_close_range_and_never_zero() {
        for altitude in [0.0_f64, 0.001, 0.05, 1.0] {
            let policy = ProjectionPolicy::for_placement(earth_radius(), Some(altitude));
            assert!(
                policy.near >= CLOSE_NEAR_FLOOR_M,
                "altitude {} near {}",
                altitude,
                policy.near
            );
            assert!(policy.near.is_finite());
        }
    }

    #[test]
    fn to_projection_round_trips_policy_values() {
        let policy = projection_for_preset(PlanetPreset::EarthScale, Some(100.0));
        let Projection::Perspective(perspective) = policy.to_projection() else {
            panic!("expected perspective projection");
        };
        assert_eq!(perspective.near, policy.near);
        assert_eq!(perspective.far, policy.far);
        assert_eq!(perspective.fov, policy.fov_radians);
    }

    #[test]
    fn policy_is_origin_invariant() {
        // Projection policy depends only on radius and altitude, not on world
        // origin, which is the whole point of the f32 render boundary.
        let a = ProjectionPolicy::for_placement(earth_radius(), Some(100.0));
        let b = ProjectionPolicy::for_placement(earth_radius(), Some(100.0));
        assert_eq!(a, b);
    }

    #[test]
    fn miniature_and_earth_produce_distinct_far_planes() {
        let mini = projection_for_preset(PlanetPreset::MiniatureDebug, None);
        let earth = projection_for_preset(PlanetPreset::EarthScale, None);
        assert_ne!(mini.far, earth.far);
        assert!(earth.far > mini.far);
    }
}
