use er_core::rng::{rng_child, rng_from_seed, Rng};
use er_core::seed::PlanetSeed;
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use rand::RngCore;

#[derive(Clone, Copy, Debug)]
pub struct PlanetParams {
    pub axial_tilt: f64,
    pub sea_level: f64,
    pub lapse_rate: f64,
    pub temp_gradient: f64,
    pub temp_noise_freq: f32,
    pub temp_noise_amp: f32,
    pub moisture_noise_freq: f32,
    pub moisture_noise_amp: f32,
    pub rain_shadow_strength: f32,
    pub high_alt_threshold: f64,
    pub beach_threshold: f64,
    pub volcanic_threshold: f64,
    pub toxic_moisture_threshold: f64,
    pub toxic_temp_threshold: f64,
    pub temp_noise_seed: i32,
    pub moisture_noise_seed: i32,
}

pub fn planet_params(seed: PlanetSeed) -> PlanetParams {
    let mut rng = rng_from_seed(seed.0);
    let mut r = rng_child(&mut rng, 1);

    fn rand_unit(r: &mut Rng) -> f64 {
        (r.next_u32() as f64) / (u32::MAX as f64)
    }

    let temp_noise_seed = r.next_u32() as i32;
    let moisture_noise_seed = r.next_u32() as i32;

    PlanetParams {
        axial_tilt: 0.262 + rand_unit(&mut r) * (0.611 - 0.262),
        sea_level: -0.1 + rand_unit(&mut r) * 0.2,
        lapse_rate: 0.05 + rand_unit(&mut r) * 0.1,
        temp_gradient: 0.7 + rand_unit(&mut r) * 0.3,
        temp_noise_freq: (0.3 + rand_unit(&mut r) * 1.2) as f32,
        temp_noise_amp: (0.05 + rand_unit(&mut r) * 0.35) as f32,
        moisture_noise_freq: (0.3 + rand_unit(&mut r) * 1.2) as f32,
        moisture_noise_amp: (0.05 + rand_unit(&mut r) * 0.35) as f32,
        rain_shadow_strength: (0.3 + rand_unit(&mut r) * 0.4) as f32,
        high_alt_threshold: 0.5 + rand_unit(&mut r) * 0.3,
        beach_threshold: 0.02 + rand_unit(&mut r) * 0.03,
        volcanic_threshold: 0.8 + rand_unit(&mut r) * 0.4,
        toxic_moisture_threshold: 0.85,
        toxic_temp_threshold: 0.7,
        temp_noise_seed,
        moisture_noise_seed,
    }
}

pub struct ClimateNoise {
    pub temp_noise: FastNoiseLite,
    pub moisture_noise: FastNoiseLite,
}

pub fn climate_noise(params: &PlanetParams) -> ClimateNoise {
    let mut temp_noise = FastNoiseLite::with_seed(params.temp_noise_seed);
    temp_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    temp_noise.set_fractal_type(Some(FractalType::FBm));
    temp_noise.set_fractal_octaves(Some(3));
    temp_noise.set_frequency(Some(params.temp_noise_freq));

    let mut moisture_noise = FastNoiseLite::with_seed(params.moisture_noise_seed);
    moisture_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    moisture_noise.set_fractal_type(Some(FractalType::FBm));
    moisture_noise.set_fractal_octaves(Some(3));
    moisture_noise.set_frequency(Some(params.moisture_noise_freq));

    ClimateNoise {
        temp_noise,
        moisture_noise,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planet_params_determinism() {
        let a = planet_params(PlanetSeed(0xC0FFEE));
        let b = planet_params(PlanetSeed(0xC0FFEE));
        assert_eq!(a.axial_tilt.to_bits(), b.axial_tilt.to_bits());
        assert_eq!(a.sea_level.to_bits(), b.sea_level.to_bits());
        assert_eq!(a.lapse_rate.to_bits(), b.lapse_rate.to_bits());
        assert_eq!(a.temp_gradient.to_bits(), b.temp_gradient.to_bits());
        assert_eq!(a.temp_noise_freq.to_bits(), b.temp_noise_freq.to_bits());
        assert_eq!(a.temp_noise_seed, b.temp_noise_seed);
        assert_eq!(a.moisture_noise_seed, b.moisture_noise_seed);
    }

    #[test]
    fn planet_params_in_expected_ranges() {
        let p = planet_params(PlanetSeed(0xC0FFEE));
        assert!(p.axial_tilt >= 0.262 && p.axial_tilt <= 0.611);
        assert!(p.sea_level >= -0.1 && p.sea_level <= 0.1);
        assert!(p.lapse_rate >= 0.05 && p.lapse_rate <= 0.15);
        assert!(p.temp_gradient >= 0.7 && p.temp_gradient <= 1.0);
        assert!(p.temp_noise_freq >= 0.3 && p.temp_noise_freq <= 1.5);
        assert!(p.temp_noise_amp >= 0.05 && p.temp_noise_amp <= 0.4);
        assert!(p.moisture_noise_freq >= 0.3 && p.moisture_noise_freq <= 1.5);
        assert!(p.moisture_noise_amp >= 0.05 && p.moisture_noise_amp <= 0.4);
        assert!(p.high_alt_threshold >= 0.5 && p.high_alt_threshold <= 0.8);
        assert!(p.beach_threshold >= 0.02 && p.beach_threshold <= 0.05);
        assert!(p.volcanic_threshold >= 0.8 && p.volcanic_threshold <= 1.2);
    }

    #[test]
    fn different_seeds_differ() {
        let a = planet_params(PlanetSeed(0xC0FFEE));
        let b = planet_params(PlanetSeed(0xDEADBEEF));
        assert_ne!(a.temp_noise_seed, b.temp_noise_seed);
    }
}
