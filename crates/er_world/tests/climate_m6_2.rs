//! Milestone 6.2 integration tests for the physical climate atlas.
//!
//! These tests exercise the intended public climate API:
//!
//! - `er_world::ClimateAtlas`
//! - `er_world::ClimateSample`
//! - `er_world::CLIMATE_ATLAS_LOD`
//! - `er_world::EnvironmentPreset`
//! - `er_core::seed::PlanetSeed`
//!
//! The tests target *scientifically defensible invariants and budgets* rather
//! than brittle golden values. Broad tolerances are used throughout; exact
//! climate statistics are intentionally avoided.
//!
//! If the types above are not yet exported, the compiler errors below identify
//! the exact public exports that must be wired up in `er_world/src/lib.rs`.

use std::time::Instant;

use er_core::seed::PlanetSeed;
use er_world::{ClimateAtlas, ClimateSample, EnvironmentPreset, CLIMATE_ATLAS_LOD};
use glam::DVec3;

const R: f64 = 6_371_000.0;
const TEST_SEED: PlanetSeed = PlanetSeed(0xC0FFEE);
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653_5;
const _: () = assert!(CLIMATE_ATLAS_LOD <= 12);

// Project budgets from `EnvironmentPresetExpectations`.
const MAX_ATLAS_BUILD_SECS: f64 = 30.0;
const MAX_ATLAS_MEMORY_MIB: f64 = 500.0;

// Loose tolerances used for scientific invariants, not golden values.
const TEMP_EPSILON_C: f64 = 2.0;
const PRECIP_EPSILON_MM: f64 = 50.0;
const MONTHLY_TEMP_MIN_C: f32 = -100.0;
const MONTHLY_TEMP_MAX_C: f32 = 70.0;
const MONTHLY_PRECIP_MAX_MM: f32 = 3000.0;

/// Returns `n` approximately evenly distributed directions on the unit sphere.
fn fibonacci_directions(n: usize) -> Vec<DVec3> {
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * ((i as f64 + 0.5) / n as f64);
            let radial = (1.0 - y * y).max(0.0).sqrt();
            let theta = GOLDEN_ANGLE * i as f64;
            DVec3::new(theta.cos() * radial, y, theta.sin() * radial).normalize()
        })
        .collect()
}

/// The eight spherified-cube corners.
fn cube_corners() -> Vec<DVec3> {
    let s = 1.0 / 3.0f64.sqrt();
    let mut dirs = Vec::with_capacity(8);
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                dirs.push(DVec3::new(sx * s, sy * s, sz * s));
            }
        }
    }
    dirs
}

/// The twelve edge-centers of the spherified cube.
fn cube_edge_centers() -> Vec<DVec3> {
    let h = 1.0 / 2.0f64.sqrt();
    vec![
        DVec3::new(h, h, 0.0),
        DVec3::new(h, -h, 0.0),
        DVec3::new(-h, h, 0.0),
        DVec3::new(-h, -h, 0.0),
        DVec3::new(h, 0.0, h),
        DVec3::new(h, 0.0, -h),
        DVec3::new(-h, 0.0, h),
        DVec3::new(-h, 0.0, -h),
        DVec3::new(0.0, h, h),
        DVec3::new(0.0, h, -h),
        DVec3::new(0.0, -h, h),
        DVec3::new(0.0, -h, -h),
    ]
}

/// Pairs of directions separated by a small angle across cube edges/corners.
fn seam_probe_pairs() -> Vec<(DVec3, DVec3)> {
    let mut pairs = Vec::new();
    let delta = 1.0e-4;

    for base in cube_corners().iter().chain(cube_edge_centers().iter()) {
        // Perturb along two orthogonal tangents.
        let tangent = if base.x.abs() < base.y.abs() && base.x.abs() < base.z.abs() {
            DVec3::X
        } else if base.y.abs() < base.z.abs() {
            DVec3::Y
        } else {
            DVec3::Z
        };
        let t = (*base - base.dot(tangent) * tangent).normalize();
        let a = (*base + delta * t).normalize();
        let b = (*base - delta * t).normalize();
        pairs.push((a, b));
    }

    pairs
}

fn sample_all(atlas: &ClimateAtlas, dirs: &[DVec3]) -> Vec<ClimateSample> {
    dirs.iter().map(|d| atlas.sample(*d)).collect()
}

fn mean_annual_temperature_c(samples: &[ClimateSample]) -> f64 {
    let sum: f64 = samples
        .iter()
        .map(|s| s.annual_mean_temperature_c() as f64)
        .sum();
    sum / samples.len().max(1) as f64
}

fn mean_annual_precipitation_mm(samples: &[ClimateSample]) -> f64 {
    let sum: f64 = samples
        .iter()
        .map(|s| s.annual_precipitation_mm() as f64)
        .sum();
    sum / samples.len().max(1) as f64
}

fn mean_snow_fraction(samples: &[ClimateSample]) -> f64 {
    let sum: f64 = samples.iter().map(|s| s.snow_fraction() as f64).sum();
    sum / samples.len().max(1) as f64
}

/// Assert the basic physical validity of a single climate sample.
fn assert_sample_physical(sample: &ClimateSample) {
    assert!(
        sample.annual_mean_temperature_c().is_finite(),
        "annual mean temperature must be finite"
    );
    assert!(
        sample.warmest_month_temperature_c().is_finite(),
        "warmest month temperature must be finite"
    );
    assert!(
        sample.coldest_month_temperature_c().is_finite(),
        "coldest month temperature must be finite"
    );
    assert!(
        sample.coldest_month_temperature_c() <= sample.annual_mean_temperature_c(),
        "coldest month must not exceed annual mean"
    );
    assert!(
        sample.annual_mean_temperature_c() <= sample.warmest_month_temperature_c(),
        "annual mean must not exceed warmest month"
    );

    assert!(
        sample.annual_precipitation_mm().is_finite() && sample.annual_precipitation_mm() >= 0.0,
        "annual precipitation must be finite and non-negative"
    );
    assert!(
        sample.driest_month_precipitation_mm().is_finite()
            && sample.driest_month_precipitation_mm() >= 0.0,
        "driest month precipitation must be finite and non-negative"
    );
    assert!(
        sample.wettest_month_precipitation_mm().is_finite()
            && sample.wettest_month_precipitation_mm() >= sample.driest_month_precipitation_mm(),
        "wettest month precipitation must be at least driest month"
    );

    assert!(
        sample.pet_mm().is_finite() && sample.pet_mm() >= 0.0,
        "PET must be finite and non-negative"
    );

    let water_balance = sample.water_balance_mm();
    assert!(water_balance.is_finite(), "water balance must be finite");
    let expected_water_balance = sample.annual_precipitation_mm() - sample.pet_mm();
    assert!(
        (water_balance - expected_water_balance).abs()
            <= 1.0 + 0.01 * sample.annual_precipitation_mm(),
        "water balance must approximate precipitation - PET within 1 mm + 1%"
    );

    assert!(
        (-1.0..=10.0).contains(&sample.aridity_index()),
        "aridity index must lie within the declared [-1, 10] range"
    );
    assert!(
        (-1.0..=1.0).contains(&sample.moisture_index()),
        "moisture index must lie within the declared [-1, 1] range"
    );
    assert!(
        (0.0..=1.0).contains(&sample.snow_fraction()),
        "snow fraction must be a unit fraction"
    );

    assert!(
        sample.growing_season_days() <= 366,
        "growing season cannot exceed 366 days"
    );
    assert!(
        sample.growing_degree_days_c().is_finite() && sample.growing_degree_days_c() >= 0.0,
        "growing degree days must be finite and non-negative"
    );

    for month in 0..12 {
        let t = sample.monthly_temperature_c(month);
        assert!(
            t.is_finite() && (MONTHLY_TEMP_MIN_C..=MONTHLY_TEMP_MAX_C).contains(&t),
            "monthly temperature {month} must be finite and physically plausible"
        );
        let p = sample.monthly_precipitation_mm(month);
        assert!(
            p.is_finite() && (0.0..=MONTHLY_PRECIP_MAX_MM).contains(&p),
            "monthly precipitation {month} must be finite and physically plausible"
        );
        let pet = sample.monthly_pet_mm(month);
        assert!(
            pet.is_finite() && pet >= 0.0,
            "monthly PET {month} must be finite and non-negative"
        );
    }
}

#[test]
fn climate_atlas_build_is_deterministic() {
    let a = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::EarthLike, R);
    let b = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::EarthLike, R);

    let dirs = fibonacci_directions(512);
    let sa = sample_all(&a, &dirs);
    let sb = sample_all(&b, &dirs);

    assert_eq!(sa.len(), sb.len());
    for (left, right) in sa.iter().zip(sb.iter()) {
        assert_eq!(
            left.annual_mean_temperature_c().to_bits(),
            right.annual_mean_temperature_c().to_bits()
        );
        assert_eq!(
            left.annual_precipitation_mm().to_bits(),
            right.annual_precipitation_mm().to_bits()
        );
        assert_eq!(left.pet_mm().to_bits(), right.pet_mm().to_bits());
        assert_eq!(
            left.water_balance_mm().to_bits(),
            right.water_balance_mm().to_bits()
        );
        assert_eq!(
            left.aridity_index().to_bits(),
            right.aridity_index().to_bits()
        );
        assert_eq!(
            left.moisture_index().to_bits(),
            right.moisture_index().to_bits()
        );
        assert_eq!(
            left.snow_fraction().to_bits(),
            right.snow_fraction().to_bits()
        );
        assert_eq!(left.growing_season_days(), right.growing_season_days());
    }
}

#[test]
fn climate_presets_are_distinct() {
    let earth = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::EarthLike, R);
    let hot = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::HotArid, R);
    let cold = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::ColdIcehouse, R);

    let dirs = fibonacci_directions(2048);

    let earth_temp = mean_annual_temperature_c(&sample_all(&earth, &dirs));
    let hot_temp = mean_annual_temperature_c(&sample_all(&hot, &dirs));
    let cold_temp = mean_annual_temperature_c(&sample_all(&cold, &dirs));

    let earth_precip = mean_annual_precipitation_mm(&sample_all(&earth, &dirs));
    let hot_precip = mean_annual_precipitation_mm(&sample_all(&hot, &dirs));
    let _cold_precip = mean_annual_precipitation_mm(&sample_all(&cold, &dirs));

    let earth_snow = mean_snow_fraction(&sample_all(&earth, &dirs));
    let hot_snow = mean_snow_fraction(&sample_all(&hot, &dirs));
    let cold_snow = mean_snow_fraction(&sample_all(&cold, &dirs));

    assert!(
        hot_temp > earth_temp + TEMP_EPSILON_C,
        "HotArid global mean temperature should exceed EarthLike"
    );
    assert!(
        cold_temp < earth_temp - TEMP_EPSILON_C,
        "ColdIcehouse global mean temperature should be below EarthLike"
    );

    assert!(
        hot_precip < earth_precip - PRECIP_EPSILON_MM,
        "HotArid global precipitation should be lower than EarthLike"
    );

    assert!(
        cold_snow > earth_snow + 0.05,
        "ColdIcehouse should have more snow cover than EarthLike"
    );
    assert!(
        hot_snow < earth_snow,
        "HotArid should have less snow cover than EarthLike"
    );
}

#[test]
fn climate_samples_respect_physical_invariants() {
    let atlas = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::EarthLike, R);
    let dirs = fibonacci_directions(1024);

    for dir in dirs {
        let sample = atlas.sample(dir);
        assert_sample_physical(&sample);
    }
}

#[test]
fn climate_is_continuous_across_cube_seams_and_corners() {
    let atlas = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::EarthLike, R);

    for (a, b) in seam_probe_pairs() {
        let sa = atlas.sample(a);
        let sb = atlas.sample(b);

        assert!(
            (sa.annual_mean_temperature_c() - sb.annual_mean_temperature_c()).abs() as f64
                <= 2.0 * TEMP_EPSILON_C,
            "temperature jump across seam exceeds broad tolerance"
        );
        assert!(
            (sa.annual_precipitation_mm() - sb.annual_precipitation_mm()).abs() as f64
                <= 2.0 * PRECIP_EPSILON_MM,
            "precipitation jump across seam exceeds broad tolerance"
        );
        assert!(
            (sa.pet_mm() - sb.pet_mm()).abs() as f64 <= 2.0 * PRECIP_EPSILON_MM,
            "PET jump across seam exceeds broad tolerance"
        );
    }

    // The exact cube corners and edge centers themselves must be valid.
    for dir in cube_corners().iter().chain(cube_edge_centers().iter()) {
        let sample = atlas.sample(*dir);
        assert_sample_physical(&sample);
    }
}

#[test]
fn climate_sampling_is_independent_of_sample_density() {
    let atlas = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::EarthLike, R);

    let fine_dirs = fibonacci_directions(4096);
    let coarse_dirs = fibonacci_directions(256);

    let fine_temp = mean_annual_temperature_c(&sample_all(&atlas, &fine_dirs));
    let coarse_temp = mean_annual_temperature_c(&sample_all(&atlas, &coarse_dirs));
    let fine_precip = mean_annual_precipitation_mm(&sample_all(&atlas, &fine_dirs));
    let coarse_precip = mean_annual_precipitation_mm(&sample_all(&atlas, &coarse_dirs));

    assert!(
        (fine_temp - coarse_temp).abs() <= 2.0 * TEMP_EPSILON_C,
        "mean annual temperature should not depend on sampling density"
    );
    assert!(
        (fine_precip - coarse_precip).abs() <= 2.0 * PRECIP_EPSILON_MM,
        "mean annual precipitation should not depend on sampling density"
    );
}

#[test]
fn climate_atlas_memory_fits_budget() {
    let atlas = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::EarthLike, R);

    // If the atlas exposes a sample count, use it to derive a conservative
    // upper bound. Otherwise fall back to a bound based on the public LOD constant.
    let sample_count = atlas.sample_count();
    let bytes_per_sample = std::mem::size_of::<ClimateSample>() as u64;
    let estimated_bytes = sample_count as u64 * bytes_per_sample;

    let mib = estimated_bytes as f64 / (1024.0 * 1024.0);
    assert!(
        mib <= MAX_ATLAS_MEMORY_MIB,
        "estimated atlas memory {mib:.1} MiB exceeds budget {MAX_ATLAS_MEMORY_MIB} MiB"
    );
}

#[test]
fn climate_atlas_build_time_is_reported_within_budget() {
    let start = Instant::now();
    let atlas = ClimateAtlas::new(TEST_SEED, EnvironmentPreset::EarthLike, R);
    let elapsed = start.elapsed().as_secs_f64();

    // Force the atlas to be used so the build is not dead-code eliminated.
    let _ = atlas.sample(DVec3::Y);

    eprintln!(
        "ClimateAtlas build time (EarthLike, seed=0xC0FFEE): {elapsed:.3}s (budget {MAX_ATLAS_BUILD_SECS}s)"
    );

    assert!(
        elapsed <= MAX_ATLAS_BUILD_SECS,
        "atlas build time {elapsed:.3}s exceeded project budget {MAX_ATLAS_BUILD_SECS}s"
    );
}
