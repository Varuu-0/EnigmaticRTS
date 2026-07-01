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
        mountain_freq: (1.0 + rand_unit(&mut r) * 0.5) as f32,
        mountain_amp: (0.3 + rand_unit(&mut r) * 0.4) as f32,
        mountain_octaves: 4,
        hill_freq: (2.0 + rand_unit(&mut r) * 1.0) as f32,
        hill_amp: (0.15 + rand_unit(&mut r) * 0.2) as f32,
        hill_octaves: 3,
        detail_freq: (4.0 + rand_unit(&mut r) * 2.0) as f32,
        detail_amp: (0.05 + rand_unit(&mut r) * 0.1) as f32,
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
    warp: FastNoiseLite,
    continental: FastNoiseLite,
    mountain: FastNoiseLite,
    hill: FastNoiseLite,
    detail: FastNoiseLite,
}

impl ElevationNoise {
    pub fn new(params: &ElevationParams) -> Self {
        let seed = params.seed;

        let mut warp = FastNoiseLite::with_seed(seed);
        warp.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
        warp.set_frequency(Some(params.warp_freq));
        warp.set_domain_warp_amp(Some(params.warp_amp));

        let mut continental = FastNoiseLite::with_seed(seed);
        continental.set_noise_type(Some(NoiseType::OpenSimplex2));
        continental.set_fractal_type(Some(FractalType::FBm));
        continental.set_fractal_octaves(Some(params.continental_octaves));
        continental.set_frequency(Some(params.continental_freq));
        continental.set_fractal_lacunarity(Some(params.lacunarity));
        continental.set_fractal_gain(Some(params.gain));
        continental.set_fractal_weighted_strength(Some(0.0));

        let mut mountain = FastNoiseLite::with_seed(seed);
        mountain.set_noise_type(Some(NoiseType::OpenSimplex2));
        mountain.set_fractal_type(Some(FractalType::Ridged));
        mountain.set_fractal_octaves(Some(params.mountain_octaves));
        mountain.set_frequency(Some(params.mountain_freq));
        mountain.set_fractal_lacunarity(Some(params.lacunarity));
        mountain.set_fractal_gain(Some(params.gain));
        mountain.set_fractal_weighted_strength(Some(0.0));

        let mut hill = FastNoiseLite::with_seed(seed);
        hill.set_noise_type(Some(NoiseType::OpenSimplex2));
        hill.set_fractal_type(Some(FractalType::FBm));
        hill.set_fractal_octaves(Some(params.hill_octaves));
        hill.set_frequency(Some(params.hill_freq));
        hill.set_fractal_lacunarity(Some(params.lacunarity));
        hill.set_fractal_gain(Some(params.gain));
        hill.set_fractal_weighted_strength(Some(0.0));

        let mut detail = FastNoiseLite::with_seed(seed);
        detail.set_noise_type(Some(NoiseType::Value));
        detail.set_fractal_type(Some(FractalType::FBm));
        detail.set_fractal_octaves(Some(params.detail_octaves));
        detail.set_frequency(Some(params.detail_freq));
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

pub fn surface_pos(
    dir: DVec3,
    radius: f64,
    noise: &ElevationNoise,
    params: &ElevationParams,
) -> WorldPos {
    let elev = elevation(dir, noise, params);
    dir_to_surface(dir, radius, elev)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_dirs(seed: u64, count: usize) -> Vec<DVec3> {
        let mut rng = rng_from_seed(seed);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let x = u2f(rng.next_u64());
            let y = u2f(rng.next_u64());
            let z = u2f(rng.next_u64());
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

    #[test]
    fn determinism_bit_identical() {
        let params = elevation_params(PlanetSeed(0xC0FFEE));
        let noise = ElevationNoise::new(&params);
        let dirs = rand_dirs(0xABCDEF, 1000);

        let pass1: Vec<f64> = dirs.iter().map(|d| elevation(*d, &noise, &params)).collect();
        let pass2: Vec<f64> = dirs.iter().map(|d| elevation(*d, &noise, &params)).collect();

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
                e >= -3.0 && e <= 3.0,
                "elevation {e} out of [-3, 3] (amp_sum={amp_sum})"
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
}
