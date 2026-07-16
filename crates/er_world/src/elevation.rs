use er_core::math::{dir_to_surface, WorldPos};
use er_core::rng::{rng_child, rng_from_seed, Rng};
use er_core::seed::PlanetSeed;
use fastnoise_lite::{DomainWarpType, FastNoiseLite, FractalType, NoiseType};
use glam::DVec3;
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
            (1.0 / RIDGE_WAVELENGTH_M) as f32,
            (1.0 / WARP_WAVELENGTH_M) as f32,
        )
    }

    fn init_layers(
        seed: i32,
        params: &ElevationParams,
        continental_freq: f32,
        mountain_freq: f32,
        hill_freq: f32,
        detail_freq: f32,
        warp_freq: f32,
    ) -> Self {
        let mut warp = FastNoiseLite::with_seed(seed);
        warp.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
        warp.set_frequency(Some(warp_freq));
        warp.set_domain_warp_amp(Some(params.warp_amp));

        let mut continental = FastNoiseLite::with_seed(seed);
        continental.set_noise_type(Some(NoiseType::OpenSimplex2));
        continental.set_fractal_type(Some(FractalType::FBm));
        continental.set_fractal_octaves(Some(params.continental_octaves));
        continental.set_frequency(Some(continental_freq));
        continental.set_fractal_lacunarity(Some(params.lacunarity));
        continental.set_fractal_gain(Some(params.gain));
        continental.set_fractal_weighted_strength(Some(0.0));

        let mut mountain = FastNoiseLite::with_seed(seed);
        mountain.set_noise_type(Some(NoiseType::OpenSimplex2));
        mountain.set_fractal_type(Some(FractalType::Ridged));
        mountain.set_fractal_octaves(Some(params.mountain_octaves));
        mountain.set_frequency(Some(mountain_freq));
        mountain.set_fractal_lacunarity(Some(params.lacunarity));
        mountain.set_fractal_gain(Some(params.gain));
        mountain.set_fractal_weighted_strength(Some(0.0));

        let mut hill = FastNoiseLite::with_seed(seed);
        hill.set_noise_type(Some(NoiseType::OpenSimplex2));
        hill.set_fractal_type(Some(FractalType::FBm));
        hill.set_fractal_octaves(Some(params.hill_octaves));
        hill.set_frequency(Some(hill_freq));
        hill.set_fractal_lacunarity(Some(params.lacunarity));
        hill.set_fractal_gain(Some(params.gain));
        hill.set_fractal_weighted_strength(Some(0.0));

        let mut detail = FastNoiseLite::with_seed(seed);
        detail.set_noise_type(Some(NoiseType::Value));
        detail.set_fractal_type(Some(FractalType::FBm));
        detail.set_fractal_octaves(Some(params.detail_octaves));
        detail.set_frequency(Some(detail_freq));
        detail.set_fractal_lacunarity(Some(params.lacunarity));
        detail.set_fractal_gain(Some(params.gain));
        detail.set_fractal_weighted_strength(Some(0.0));

        Self {
            warp,
            continental,
            mountain,
            hill,
            detail,
        }
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

/// Elevation in normalized field units, sampled from a continuous 3D metric
/// coordinate on the planet sphere surface (in meters).
///
/// This function must be called with noise built via `new_metric()`. The result
/// is multiplied by 1000 (the `TerrainState` elevation scale) during mesh
/// generation to obtain meter displacement. Amplitudes are hardcoded so the
/// aggregate range is ~‑11 to +9 field units, approximating a credible 11 km
/// abyss / 9 km mountain ceiling on Earth-like planets.
pub fn elevation_at(pos: DVec3, noise: &ElevationNoise) -> f64 {
    const METRIC_CONTINENTAL_AMP: f64 = 8.0;
    const METRIC_MOUNTAIN_AMP: f64 = 3.5;
    const METRIC_HILL_AMP: f64 = 2.0;
    const METRIC_DETAIL_AMP: f64 = 0.5;

    let (wx, wy, wz) = noise.warp.domain_warp_3d(pos.x, pos.y, pos.z);

    let continental = noise.continental.get_noise_3d(wx, wy, wz);

    let mountain_raw = noise.mountain.get_noise_3d(wx, wy, wz);
    let mountain_mask = continental.max(0.0);
    let mountains = mountain_raw * mountain_mask;

    let hills = noise.hill.get_noise_3d(wx, wy, wz);

    let detail = noise.detail.get_noise_3d(wx, wy, wz);

    (continental as f64 * METRIC_CONTINENTAL_AMP
        + mountains as f64 * METRIC_MOUNTAIN_AMP
        + hills as f64 * METRIC_HILL_AMP
        + detail as f64 * METRIC_DETAIL_AMP) as f64
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
