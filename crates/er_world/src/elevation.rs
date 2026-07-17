use crate::brushes::BrushSet;
use er_core::math::{dir_to_surface, WorldPos};
use er_core::rng::{rng_child, rng_from_seed, Rng};
use er_core::seed::PlanetSeed;
use fastnoise_lite::{DomainWarpType, FastNoiseLite, FractalType, NoiseType};
use glam::{DVec3, Vec3};
use rand::RngCore;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ElevationParams {
    pub seed: i32,
    pub sea_level: f32,
    pub continental_freq: f32,
    pub continental_amp: f32,
    pub continental_octaves: i32,
    pub mountain_freq: f32,
    pub mountain_amp: f32,
    pub mountain_octaves: i32,
    pub hill_freq: f32,
    pub hill_amp: f32,
    pub hill_octaves: i32,
    pub detail_freq: f32,
    pub detail_amp: f32,
    pub detail_octaves: i32,
    pub warp_freq: f32,
    pub warp_amp: f32,
    pub lacunarity: f32,
    pub gain: f32,
    _pad0: f32,
    _pad1: f32,
}

pub fn elevation_params(seed: PlanetSeed) -> ElevationParams {
    let mut rng = rng_from_seed(seed.0);
    let mut r = rng_child(&mut rng, 0);

    fn rand_unit(r: &mut Rng) -> f64 {
        (r.next_u32() as f64) / (u32::MAX as f64)
    }

    let noise_seed = r.next_u32() as i32;

    ElevationParams {
        seed: noise_seed,
        sea_level: 0.0,
        continental_freq: (0.5 + rand_unit(&mut r) * 0.3) as f32,
        continental_amp: (0.8 + rand_unit(&mut r) * 0.4) as f32,
        continental_octaves: 4,
        mountain_freq: (0.7 + rand_unit(&mut r) * 0.4) as f32,
        mountain_amp: (0.6 + rand_unit(&mut r) * 0.6) as f32,
        mountain_octaves: 4,
        hill_freq: (1.5 + rand_unit(&mut r) * 0.8) as f32,
        hill_amp: (0.25 + rand_unit(&mut r) * 0.3) as f32,
        hill_octaves: 3,
        detail_freq: (8.0 + rand_unit(&mut r) * 6.0) as f32,
        detail_amp: (0.10 + rand_unit(&mut r) * 0.12) as f32,
        detail_octaves: 2,
        warp_freq: (0.3 + rand_unit(&mut r) * 0.2) as f32,
        warp_amp: (0.3 + rand_unit(&mut r) * 0.4) as f32,
        lacunarity: 2.0,
        gain: 0.5,
        _pad0: 0.0,
        _pad1: 0.0,
    }
}

pub struct ElevationNoise {
    pub(crate) warp: FastNoiseLite,
    pub(crate) continental: FastNoiseLite,
    pub(crate) mountain: FastNoiseLite,
    pub(crate) hill: FastNoiseLite,
    pub(crate) detail: FastNoiseLite,
    // --- Milestone 2.2 landform layers (metric path only) ---
    pub(crate) tectonic_belt: FastNoiseLite,
    pub(crate) drainage: FastNoiseLite,
    pub(crate) ridge_detail: FastNoiseLite,
    pub(crate) valley: FastNoiseLite,
    pub(crate) erosion: FastNoiseLite,
    pub(crate) talus: FastNoiseLite,
    // --- Milestone 2.2 brush landforms (metric path only) ---
    pub(crate) brushes: BrushSet,
}

impl ElevationNoise {
    pub fn new(params: &ElevationParams) -> Self {
        Self::init_layers(
            params.seed,
            params,
            params.continental_freq,
            params.mountain_freq,
            params.hill_freq,
            params.detail_freq,
            params.warp_freq,
            params.warp_amp,
            params.continental_freq * 0.5,
            params.hill_freq * 0.5,
            params.mountain_freq * 0.5,
            params.mountain_freq * 0.5,
            params.hill_freq * 0.5,
            params.detail_freq * 0.5,
            false,
        )
    }

    pub fn new_metric(params: &ElevationParams) -> Self {
        use crate::terrain_space::*;
        Self::init_layers(
            params.seed,
            params,
            (1.0 / CONTINENTAL_WAVELENGTH_M) as f32,
            (1.0 / MOUNTAIN_WAVELENGTH_M) as f32,
            (1.0 / FOOTHILL_WAVELENGTH_M) as f32,
            (1.0 / MICRO_DETAIL_WAVELENGTH_M) as f32,
            (1.0 / WARP_WAVELENGTH_M) as f32,
            (METRIC_WARP_AMP_M * params.warp_amp as f64) as f32,
            (1.0 / TECTONIC_BELT_WAVELENGTH_M) as f32,
            (1.0 / DRAINAGE_WAVELENGTH_M) as f32,
            (1.0 / RIDGE_DETAIL_WAVELENGTH_M) as f32,
            (1.0 / VALLEY_WAVELENGTH_M) as f32,
            (1.0 / EROSION_WAVELENGTH_M) as f32,
            (1.0 / TALUS_WAVELENGTH_M) as f32,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn init_layers(
        seed: i32,
        params: &ElevationParams,
        continental_freq: f32,
        mountain_freq: f32,
        hill_freq: f32,
        detail_freq: f32,
        warp_freq: f32,
        warp_amp: f32,
        tectonic_freq: f32,
        drainage_freq: f32,
        ridge_detail_freq: f32,
        valley_freq: f32,
        erosion_freq: f32,
        talus_freq: f32,
        decorrelate_layers: bool,
    ) -> Self {
        let layer_seed = |salt: i32| {
            if decorrelate_layers {
                seed ^ salt
            } else {
                seed
            }
        };

        let mut warp = FastNoiseLite::with_seed(seed);
        warp.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
        warp.set_frequency(Some(warp_freq));
        warp.set_domain_warp_amp(Some(warp_amp));

        let continental = Self::make_fbm_opensimplex(
            layer_seed(0x0001_3579),
            continental_freq,
            params.continental_octaves,
            params.lacunarity,
            params.gain,
        );
        let mountain = Self::make_ridged_opensimplex(
            layer_seed(0x0002_468b),
            mountain_freq,
            params.mountain_octaves,
            params.lacunarity,
            params.gain,
        );
        let hill = Self::make_fbm_opensimplex(
            layer_seed(0x0003_5a7d),
            hill_freq,
            params.hill_octaves,
            params.lacunarity,
            params.gain,
        );
        let detail = Self::make_fbm_value(
            layer_seed(0x0004_7c91),
            detail_freq,
            params.detail_octaves,
            params.lacunarity,
            params.gain,
        );

        // New landform layers reuse the same noise primitives.
        let tectonic_belt = Self::make_fbm_opensimplex(
            layer_seed(0x0005_8da3),
            tectonic_freq,
            params.continental_octaves,
            params.lacunarity,
            params.gain,
        );
        let drainage = Self::make_fbm_opensimplex(
            layer_seed(0x0006_9eb5),
            drainage_freq,
            params.hill_octaves,
            params.lacunarity,
            params.gain,
        );
        let ridge_detail = Self::make_ridged_opensimplex(
            layer_seed(0x0007_afc7),
            ridge_detail_freq,
            params.mountain_octaves,
            params.lacunarity,
            params.gain,
        );
        let valley = Self::make_ridged_opensimplex(
            layer_seed(0x0008_b0d9),
            valley_freq,
            params.mountain_octaves,
            params.lacunarity,
            params.gain,
        );
        let erosion = Self::make_fbm_opensimplex(
            layer_seed(0x0009_c1eb),
            erosion_freq,
            params.hill_octaves,
            params.lacunarity,
            params.gain,
        );
        let talus = Self::make_fbm_value(
            layer_seed(0x000a_d2fd),
            talus_freq,
            params.detail_octaves,
            params.lacunarity,
            params.gain,
        );

        Self {
            warp,
            continental,
            mountain,
            hill,
            detail,
            tectonic_belt,
            drainage,
            ridge_detail,
            valley,
            erosion,
            talus,
            brushes: BrushSet::from_seed(seed as u32),
        }
    }

    fn make_fbm_opensimplex(
        seed: i32,
        freq: f32,
        octaves: i32,
        lac: f32,
        gain: f32,
    ) -> FastNoiseLite {
        let mut n = FastNoiseLite::with_seed(seed);
        n.set_noise_type(Some(NoiseType::OpenSimplex2));
        n.set_fractal_type(Some(FractalType::FBm));
        n.set_fractal_octaves(Some(octaves));
        n.set_frequency(Some(freq));
        n.set_fractal_lacunarity(Some(lac));
        n.set_fractal_gain(Some(gain));
        n.set_fractal_weighted_strength(Some(0.0));
        n
    }

    fn make_ridged_opensimplex(
        seed: i32,
        freq: f32,
        octaves: i32,
        lac: f32,
        gain: f32,
    ) -> FastNoiseLite {
        let mut n = FastNoiseLite::with_seed(seed);
        n.set_noise_type(Some(NoiseType::OpenSimplex2));
        n.set_fractal_type(Some(FractalType::Ridged));
        n.set_fractal_octaves(Some(octaves));
        n.set_frequency(Some(freq));
        n.set_fractal_lacunarity(Some(lac));
        n.set_fractal_gain(Some(gain));
        n.set_fractal_weighted_strength(Some(0.0));
        n
    }

    fn make_fbm_value(seed: i32, freq: f32, octaves: i32, lac: f32, gain: f32) -> FastNoiseLite {
        let mut n = FastNoiseLite::with_seed(seed);
        n.set_noise_type(Some(NoiseType::Value));
        n.set_fractal_type(Some(FractalType::FBm));
        n.set_fractal_octaves(Some(octaves));
        n.set_frequency(Some(freq));
        n.set_fractal_lacunarity(Some(lac));
        n.set_fractal_gain(Some(gain));
        n.set_fractal_weighted_strength(Some(0.0));
        n
    }
}

/// Elevation in normalized field units, sampled from a unit sphere direction.
///
/// Units are scaled by `TerrainState` (currently 1000 m per field unit) during
/// mesh generation. The backwards-compatible path stays within [-3.5, +3.5].
pub fn elevation(dir: DVec3, noise: &ElevationNoise, params: &ElevationParams) -> f64 {
    let (wx, wy, wz) = noise.warp.domain_warp_3d(dir.x, dir.y, dir.z);

    let continental = noise.continental.get_noise_3d(wx, wy, wz);

    let mountain_raw = noise.mountain.get_noise_3d(wx, wy, wz);
    let mountain_mask = continental.max(0.0);
    let mountains = mountain_raw * mountain_mask;

    let hills = noise.hill.get_noise_3d(wx, wy, wz);

    let detail = noise.detail.get_noise_3d(wx, wy, wz);

    (continental * params.continental_amp
        + mountains * params.mountain_amp
        + hills * params.hill_amp
        + detail * params.detail_amp) as f64
}

/// Deterministic metric landform sample exposing every composed layer so
/// that macro / residual responsibilities are explicit and cannot drift.
///
/// **Macro displacement** (low-frequency): continental + mountain (gated by
/// tectonic belt) + tectonic belt uplift.  This owns shoreline, broad biome
/// classification, and stable water classification.
///
/// **Residual displacement** (high-frequency): hill + detail + ridge uplift
/// − valley carving − drainage carving + talus roughness.  Added on top of
/// the macro to produce the composed terrain elevation.
#[derive(Clone, Copy, Debug)]
pub struct MetricLandformSample {
    /// Low-frequency macro displacement (continental + mountain + tectonic).
    pub macro_displacement: f64,
    /// High-frequency residual displacement (hill + detail + ridge − valley
    /// − drainage + talus).
    pub residual_displacement: f64,
    /// Full composed elevation = macro + residual.
    pub full_elevation: f64,
    /// Warped metric position (named for legacy compatibility).
    pub warped_dir: DVec3,
    /// Mountain influence for rain-shadow climate — reflects the gated belt.
    pub mountain_influence: f64,
    // --- Bounded mask values (all in [0, 1]) ---
    /// Tectonic belt corridor mask [0, 1].
    pub tectonic_belt: f32,
    /// Drainage catchment/channel mask [0, 1].
    pub drainage: f32,
    /// Erosion intensity mask [0, 1].
    pub erosion: f32,
    /// Ridge uplift mask [0, 1].
    pub ridge_mask: f32,
    /// Valley/canyon channel mask [0, 1].
    pub valley_mask: f32,
    /// Raw signed talus noise before ridge/steepness gating.
    pub talus_raw: f32,
    /// Brush landform displacement (Milestone 2.2).  Included in macro.
    pub brush_displacement: f32,
}

/// Single shared helper that computes every metric landform layer once.
/// `elevation_at`, `elevation_low_freq_metric`, and `elevation_split_metric`
/// all delegate here so their results cannot drift.
///
/// # Mask formulas (CPU = WGSL)
///
/// **tectonic_belt** [0,1]: sharpened inverse-absolute band of broad FBm.
/// `belt = clamp(1 - |raw| * sharpness, 0, 1)` where `raw` is OpenSimplex2
/// FBm at `TECTONIC_BELT_WAVELENGTH_M`.  Corridors form where raw ≈ 0.
/// The belt gates mountain uplift and contributes a bounded macro uplift.
///
/// **erosion** [0,1]: `clamp(raw * 0.5 + 0.5, 0, 1)` from OpenSimplex2 FBm.
///
/// **ridge_mask** [0,1]: `clamp(ridged_raw * 0.5 + 0.5, 0, 1)`, then gated
/// by `(1 - erosion*0.5)`.  Nonnegative uplift.
///
/// **valley_mask** [0,1]: `clamp(ridged_raw * 0.5 + 0.5, 0, 1)`, gated by
/// `erosion`.  Subtracted (carving).
///
/// **drainage** [0,1]: `clamp(raw * 0.5 + 0.5, 0, 1)` from OpenSimplex2 FBm.
/// Subtracted (shallow carving).
///
/// **talus**: signed [-1,1] Value FBm, gated by `ridge_mask` (steep proxy).
pub fn metric_landform_sample(pos: DVec3, noise: &ElevationNoise) -> MetricLandformSample {
    use crate::terrain_space::*;

    let (wx, wy, wz) = noise.warp.domain_warp_3d(pos.x, pos.y, pos.z);
    let warped_dir = DVec3::new(wx, wy, wz);

    // ---- Macro layers (low-frequency) ----
    let continental = noise.continental.get_noise_3d(wx, wy, wz);
    let mountain_raw = noise.mountain.get_noise_3d(wx, wy, wz);

    // Tectonic belt corridor mask [0, 1]: sharpened inverse-absolute band.
    // Where the broad noise crosses zero, a corridor forms (belt → 1).
    let tectonic_raw = noise.tectonic_belt.get_noise_3d(wx, wy, wz);
    let belt_sharpness = 2.0_f64;
    let tectonic_belt =
        ((1.0 - (tectonic_raw as f64).abs() * belt_sharpness).max(0.0)).min(1.0) as f32;

    // Mountain uplift is gated by the tectonic belt corridor (in addition
    // to the continental land mask).  Mountains concentrate in belt zones.
    let mountain_mask = (continental.max(0.0) * tectonic_belt).min(1.0) as f32;
    let mountains = mountain_raw * mountain_mask;

    // ---- Brush landform displacement (Milestone 2.2) ----
    // Evaluated on the unwarped unit direction (seam-safe spherical), added
    // to the macro so mesh edges/normals/water use the same composed field.
    let dir = pos.normalize();
    let dir_f32 = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32).normalize();
    let brush_disp = noise.brushes.displacement_indexed(dir_f32);

    let macro_displacement = continental as f64 * METRIC_CONTINENTAL_AMP
        + mountains as f64 * METRIC_MOUNTAIN_AMP
        + tectonic_belt as f64 * METRIC_TECTONIC_AMP
        + brush_disp as f64;
    // Mountain influence reflects the gated belt for rain-shadow climate.
    let mountain_influence = (mountain_raw.max(0.0) as f64 * mountain_mask as f64).min(1.0);

    // ---- Residual layers (high-frequency) ----
    let hills = noise.hill.get_noise_3d(wx, wy, wz);
    let detail = noise.detail.get_noise_3d(wx, wy, wz);

    // Erosion mask [0, 1]: controls valley deepening and ridge smoothing.
    let erosion_raw = noise.erosion.get_noise_3d(wx, wy, wz);
    let erosion_mask = ((erosion_raw as f64) * 0.5 + 0.5).clamp(0.0, 1.0) as f32;

    // Ridge uplift: nonnegative mask [0,1], smoothed where erosion is high.
    let ridge_raw = noise.ridge_detail.get_noise_3d(wx, wy, wz);
    let ridge_mask =
        ((ridge_raw as f64 * 0.5 + 0.5).clamp(0.0, 1.0) * (1.0 - erosion_mask as f64 * 0.5)) as f32;
    let ridge = ridge_mask as f64 * METRIC_RIDGE_DETAIL_AMP;

    // Valley / canyon channel: nonnegative mask [0,1], gated by erosion.
    // Subtracted (carving) — always reduces elevation.
    let valley_raw = noise.valley.get_noise_3d(wx, wy, wz);
    let valley_mask =
        ((valley_raw as f64 * 0.5 + 0.5).clamp(0.0, 1.0) * erosion_mask as f64) as f32;
    let valley = valley_mask as f64 * METRIC_VALLEY_AMP;

    // Drainage catchment: nonnegative mask [0,1].  Subtracted (shallow
    // carving) — always reduces elevation.
    let drainage_raw = noise.drainage.get_noise_3d(wx, wy, wz);
    let drainage_mask = ((drainage_raw as f64 * 0.5 + 0.5).clamp(0.0, 1.0)) as f32;
    let drainage = drainage_mask as f64 * METRIC_DRAINAGE_AMP;

    // Talus roughness: signed [-1,1], gated by ridge_mask as a low-cost
    // steep-slope proxy (ridges = steep terrain).  No nested central
    // differences inside the elevation sample.
    let talus_raw = noise.talus.get_noise_3d(wx, wy, wz);
    let talus = talus_raw as f64 * METRIC_TALUS_AMP * ridge_mask as f64;

    let residual_displacement =
        hills as f64 * METRIC_HILL_AMP + detail as f64 * METRIC_DETAIL_AMP + ridge
            - valley
            - drainage
            + talus;

    let full_elevation = macro_displacement + residual_displacement;

    MetricLandformSample {
        macro_displacement,
        residual_displacement,
        full_elevation,
        warped_dir,
        mountain_influence,
        tectonic_belt,
        drainage: drainage_mask,
        erosion: erosion_mask,
        ridge_mask,
        valley_mask,
        talus_raw,
        brush_displacement: brush_disp,
    }
}

/// Elevation in normalized field units, sampled from a continuous 3D metric
/// coordinate on the planet sphere surface (in meters).
///
/// This function must be called with noise built via `new_metric()`. The result
/// is multiplied by 1000 (the `TerrainState` elevation scale) during mesh
/// generation to obtain meter displacement. Amplitudes are calibrated so the
/// aggregate range is ~‑11 to +9 field units, approximating a credible 11 km
/// abyss / 9 km mountain ceiling on Earth-like planets.
pub fn elevation_at(pos: DVec3, noise: &ElevationNoise) -> f64 {
    metric_landform_sample(pos, noise).full_elevation
}

/// Full metric landform sample — the public entry point that exposes every
/// composed layer for diagnostic, profiling, and future brush-admission use.
pub fn elevation_metric_full_eval(pos: DVec3, noise: &ElevationNoise) -> MetricLandformSample {
    metric_landform_sample(pos, noise)
}

/// Backward-compatible world-space surface position from a unit direction.
pub fn surface_pos(
    dir: DVec3,
    radius: f64,
    noise: &ElevationNoise,
    params: &ElevationParams,
) -> WorldPos {
    let elev = elevation(dir, noise, params);
    dir_to_surface(dir, radius, elev)
}

/// World-space surface position from a metric point on the sphere surface.
pub fn surface_pos_at(pos: DVec3, radius: f64, noise: &ElevationNoise) -> WorldPos {
    let elev = elevation_at(pos, noise);
    let n = pos.normalize();
    dir_to_surface(n, radius, elev)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_dirs(seed: u64, count: usize) -> Vec<DVec3> {
        let mut rng = rng_from_seed(seed);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let x: f64 = u2f(rng.next_u64());
            let y: f64 = u2f(rng.next_u64());
            let z: f64 = u2f(rng.next_u64());
            let v = DVec3::new(x, y, z);
            let l2 = v.length_squared();
            if l2 > 1e-6 && l2 < 1.0 {
                out.push(v.normalize());
            }
        }
        out
    }

    fn u2f(u: u64) -> f64 {
        (u as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    // ---- legacy backward-compat tests (unchanged) ----

    #[test]
    fn determinism_bit_identical() {
        let params = elevation_params(PlanetSeed(0xC0FFEE));
        let noise = ElevationNoise::new(&params);
        let dirs = rand_dirs(0xABCDEF, 1000);

        let pass1: Vec<f64> = dirs
            .iter()
            .map(|d| elevation(*d, &noise, &params))
            .collect();
        let pass2: Vec<f64> = dirs
            .iter()
            .map(|d| elevation(*d, &noise, &params))
            .collect();

        for (a, b) in pass1.iter().zip(pass2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "elevation not bit-identical");
        }
    }

    #[test]
    fn elevation_in_bounds() {
        let params = elevation_params(PlanetSeed(0xC0FFEE));
        let noise = ElevationNoise::new(&params);
        let dirs = rand_dirs(0x1234, 1000);

        let amp_sum = params.continental_amp as f64
            + params.mountain_amp as f64
            + params.hill_amp as f64
            + params.detail_amp as f64;

        for d in &dirs {
            let e = elevation(*d, &noise, &params);
            assert!(
                e >= -3.5 && e <= 3.5,
                "elevation {e} out of [-3.5, 3.5] (amp_sum={amp_sum})"
            );
        }
    }

    #[test]
    fn different_seeds_differ() {
        let dirs = rand_dirs(0xBEEF, 200);

        let pa = elevation_params(PlanetSeed(0xC0FFEE));
        let na = ElevationNoise::new(&pa);
        let pb = elevation_params(PlanetSeed(0xDEADBEEF));
        let nb = ElevationNoise::new(&pb);

        let mut any_diff = false;
        for d in &dirs {
            let ea = elevation(*d, &na, &pa);
            let eb = elevation(*d, &nb, &pb);
            if ea.to_bits() != eb.to_bits() {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "different seeds produced identical elevations");
    }

    // ---- Metric tests ----

    const TEST_R: f64 = 6_371_000.0;

    fn metric_points(seed: u64, count: usize) -> Vec<DVec3> {
        rand_dirs(seed, count)
            .iter()
            .map(|d| crate::terrain_space::metric_surface_point(*d, TEST_R))
            .collect()
    }

    #[test]
    fn metric_fixed_seed_determinism() {
        let params = elevation_params(PlanetSeed(0x1818));
        let noise = ElevationNoise::new_metric(&params);
        let points = metric_points(0x234, 500);

        let pass1: Vec<f64> = points.iter().map(|p| elevation_at(*p, &noise)).collect();
        let pass2: Vec<f64> = points.iter().map(|p| elevation_at(*p, &noise)).collect();
        for (a, b) in pass1.iter().zip(pass2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "metric elevation not deterministic"
            );
        }
    }

    #[test]
    fn metric_elevation_finite_and_bounded() {
        let params = elevation_params(PlanetSeed(0xDEAD));
        let noise = ElevationNoise::new_metric(&params);
        let points = metric_points(0x511, 10000);

        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for p in &points {
            let e = elevation_at(*p, &noise);
            assert!(e.is_finite(), "non-finite elevation at {p:?}");
            min = min.min(e);
            max = max.max(e);
        }
        assert!(min >= -25.0, "min {min} below reasonable");
        assert!(max <= 25.0, "max {max} above reasonable");
        assert!(min < max, "all elevations equal");
    }

    #[test]
    fn metric_different_seeds_differ() {
        let pnts = metric_points(0xF00D, 200);
        let pa = elevation_params(PlanetSeed(0xCAFE));
        let na = ElevationNoise::new_metric(&pa);
        let pb = elevation_params(PlanetSeed(0xBABE));
        let nb = ElevationNoise::new_metric(&pb);

        let mut any_diff = false;
        for p in &pnts {
            let ea = elevation_at(*p, &na);
            let eb = elevation_at(*p, &nb);
            if ea.to_bits() != eb.to_bits() {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "different seeds must diverge");
    }

    #[test]
    fn cube_edge_continuity() {
        use er_core::math::uv_to_dir;

        let params = elevation_params(PlanetSeed(0xACC));
        let noise = ElevationNoise::new_metric(&params);

        // shared edge: +X posU == +Y negU
        for i in 0..32 {
            let t = (i as f64 + 0.5) / 32.0;
            let d_x = uv_to_dir(0, 1.0, t); // +X face, posU edge
            let d_y = uv_to_dir(2, 1.0, t); // +Y face, posU edge (shared with +X)
            let diff_dir = (d_x - d_y).length();
            assert!(
                diff_dir < 1e-6,
                "edge dirs must be identical: diff={diff_dir} i={i}"
            );
            let px = crate::terrain_space::metric_surface_point(d_x, TEST_R);
            let py = crate::terrain_space::metric_surface_point(d_y, TEST_R);
            let ex = elevation_at(px, &noise);
            let ey = elevation_at(py, &noise);
            assert!(
                (ex - ey).abs() < 1e-9,
                "edge discontinuity at {i}: {ex} vs {ey}"
            );
        }

        // shared edge: +X negV === -Z posU
        for j in 0..32 {
            let t = (j as f64 + 0.5) / 32.0;
            let d_xn = uv_to_dir(0, t, 0.0); // +X face, negV edge
            let d_zn = uv_to_dir(5, 1.0, t); // -Z face, posU edge
            let diff_dir = (d_xn - d_zn).length();
            assert!(
                diff_dir < 1e-6,
                "edge dir mismatch negV/faceX: j={j} diff={diff_dir}"
            );

            let px = crate::terrain_space::metric_surface_point(d_xn, TEST_R);
            let pz = crate::terrain_space::metric_surface_point(d_zn, TEST_R);
            let diff_pos = (px - pz).length();
            assert!(diff_pos < 1.0, "coordinate discontinuity: {diff_pos}");

            let ex = elevation_at(px, &noise);
            let ez = elevation_at(pz, &noise);
            assert!(
                (ex - ez).abs() < 1e-9,
                "edge discontinuity at j={j}: {ex} vs {ez}"
            );
        }
    }

    #[test]
    fn metric_surface_pos_coherent() {
        let params = elevation_params(PlanetSeed(0xF00DBAE));
        let noise = ElevationNoise::new_metric(&params);
        let pos = crate::terrain_space::metric_surface_point(
            er_core::math::uv_to_dir(1, 0.4, 0.3),
            TEST_R,
        );
        let sp = surface_pos_at(pos, TEST_R, &noise);
        assert!(sp.0.length() > TEST_R - 15.0);
        assert!(sp.0.length() < TEST_R + 15.0);

        let recovered_dir = sp.0.normalize();
        let re_elev = elevation_at(
            crate::terrain_space::metric_surface_point(recovered_dir, TEST_R),
            &noise,
        );
        assert!((sp.0.length() - TEST_R - re_elev).abs() < 1e-3);
    }
}
