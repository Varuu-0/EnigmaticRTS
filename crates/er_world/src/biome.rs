use crate::elevation::{ElevationNoise, ElevationParams};
use crate::params::{ClimateNoise, PlanetParams};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[repr(u8)]
pub enum Biome {
    OceanShallow = 0,
    OceanMid = 1,
    OceanDeep = 2,
    OceanAbyss = 3,
    Beach = 4,
    Grassland = 5,
    Forest = 6,
    Jungle = 7,
    Desert = 8,
    Tundra = 9,
    Snow = 10,
    Mountains = 11,
    ToxicBog = 12,
    Volcanic = 13,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BiomeData {
    pub palette: [[f32; 3]; 4],
    pub traversability_cost: f32,
    pub water_behavior: String,
    pub weather_profile: Option<String>,
    pub hazard_set: Option<String>,
}

pub struct BiomeRegistry {
    entries: HashMap<Biome, BiomeData>,
}

impl BiomeRegistry {
    pub fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("Failed to read biome data file: {}", path));
        let entries: Vec<(Biome, BiomeData)> = ron::from_str(&text)
            .unwrap_or_else(|e| panic!("Failed to parse biome data RON: {}", e));
        let map: HashMap<Biome, BiomeData> = entries.into_iter().collect();
        Self { entries: map }
    }

    pub fn get(&self, biome: Biome) -> &BiomeData {
        self.entries.get(&biome).expect("missing biome data")
    }
}

pub fn temperature(dir: DVec3, elevation: f64, params: &PlanetParams, noise: &ClimateNoise) -> f64 {
    let latitude = dir.y.abs();
    let temp_noise = noise.temp_noise.get_noise_3d(dir.x, dir.y, dir.z);
    let temp = 1.0 - latitude * params.temp_gradient - elevation * params.lapse_rate
        + (temp_noise * params.temp_noise_amp) as f64;
    temp.clamp(0.0, 1.0)
}

pub fn moisture(
    dir: DVec3,
    mountain_influence: f64,
    params: &PlanetParams,
    noise: &ClimateNoise,
) -> f64 {
    let m_noise = noise.moisture_noise.get_noise_3d(dir.x, dir.y, dir.z);
    let m = (m_noise * params.moisture_noise_amp) as f64 * 0.5 + 0.5
        - mountain_influence * params.rain_shadow_strength as f64;
    m.clamp(0.0, 1.0)
}

pub fn classify_biome(
    elevation: f64,
    temperature: f64,
    moisture: f64,
    low_freq_elev: f64,
    params: &PlanetParams,
) -> Biome {
    if elevation < params.sea_level {
        let depth = params.sea_level - elevation;
        if depth < 0.3 {
            return Biome::OceanShallow;
        }
        if depth < 0.6 {
            return Biome::OceanMid;
        }
        if depth < 1.0 {
            return Biome::OceanDeep;
        }
        return Biome::OceanAbyss;
    }

    if (elevation - params.sea_level).abs() < params.beach_threshold {
        return Biome::Beach;
    }

    if low_freq_elev > params.volcanic_threshold {
        return Biome::Volcanic;
    }

    if elevation > params.high_alt_threshold {
        return Biome::Mountains;
    }

    if temperature > params.toxic_temp_threshold && moisture > params.toxic_moisture_threshold {
        return Biome::ToxicBog;
    }

    if temperature < 0.25 {
        if moisture < 0.35 {
            return Biome::Tundra;
        }
        if moisture < 0.7 {
            return Biome::Forest;
        }
        return Biome::Snow;
    } else if temperature < 0.6 {
        if moisture < 0.35 {
            return Biome::Grassland;
        }
        if moisture < 0.7 {
            return Biome::Grassland;
        }
        return Biome::Forest;
    } else {
        if moisture < 0.35 {
            return Biome::Desert;
        }
        if moisture < 0.7 {
            return Biome::Grassland;
        }
        return Biome::Jungle;
    }
}

pub fn biome(
    dir: DVec3,
    elevation: f64,
    low_freq_elev: f64,
    mountain_influence: f64,
    params: &PlanetParams,
    noise: &ClimateNoise,
) -> Biome {
    let temp = temperature(dir, elevation, params, noise);
    let moist = moisture(dir, mountain_influence, params, noise);
    classify_biome(elevation, temp, moist, low_freq_elev, params)
}

pub struct LowFreqElevation {
    pub low_freq_elev: f64,
    pub warped_dir: DVec3,
    pub mountain_influence: f64,
}

pub fn elevation_low_freq(
    dir: DVec3,
    noise: &ElevationNoise,
    params: &ElevationParams,
) -> LowFreqElevation {
    let (wx, wy, wz) = noise.warp.domain_warp_3d(dir.x, dir.y, dir.z);
    let warped_dir = DVec3::new(wx, wy, wz);

    let continental = noise.continental.get_noise_3d(wx, wy, wz);
    let mountain_raw = noise.mountain.get_noise_3d(wx, wy, wz);
    let mountain_mask = continental.max(0.0);
    let mountains = mountain_raw * mountain_mask;

    let low_freq_elev =
        (continental * params.continental_amp + mountains * params.mountain_amp) as f64;
    let mountain_influence = (mountain_raw.max(0.0) * mountain_mask) as f64;

    LowFreqElevation {
        low_freq_elev,
        warped_dir,
        mountain_influence,
    }
}

pub struct ElevationSplit {
    pub low_freq_elev: f64,
    pub warped_dir: DVec3,
    pub mountain_influence: f64,
    pub full_elev: f64,
}

pub fn elevation_split(
    dir: DVec3,
    noise: &ElevationNoise,
    params: &ElevationParams,
) -> ElevationSplit {
    let (wx, wy, wz) = noise.warp.domain_warp_3d(dir.x, dir.y, dir.z);
    let warped_dir = DVec3::new(wx, wy, wz);

    let continental = noise.continental.get_noise_3d(wx, wy, wz);
    let mountain_raw = noise.mountain.get_noise_3d(wx, wy, wz);
    let mountain_mask = continental.max(0.0);
    let mountains = mountain_raw * mountain_mask;

    let hills = noise.hill.get_noise_3d(wx, wy, wz);
    let detail = noise.detail.get_noise_3d(wx, wy, wz);

    let low_freq_elev =
        (continental * params.continental_amp + mountains * params.mountain_amp) as f64;
    let high_freq = (hills * params.hill_amp + detail * params.detail_amp) as f64;
    let full_elev = low_freq_elev + high_freq;

    let mountain_influence = (mountain_raw.max(0.0) * mountain_mask) as f64;

    ElevationSplit {
        low_freq_elev,
        warped_dir,
        mountain_influence,
        full_elev,
    }
}

pub fn temperature_at(
    pos: DVec3,
    elevation: f64,
    params: &PlanetParams,
    noise: &ClimateNoise,
) -> f64 {
    let dir = pos.normalize();
    let latitude = dir.y.abs();
    let temp_noise = noise.temp_noise.get_noise_3d(pos.x, pos.y, pos.z);
    let temp = 1.0 - latitude * params.temp_gradient - elevation * params.lapse_rate
        + (temp_noise * params.temp_noise_amp) as f64;
    temp.clamp(0.0, 1.0)
}

pub fn moisture_at(
    pos: DVec3,
    mountain_influence: f64,
    params: &PlanetParams,
    noise: &ClimateNoise,
) -> f64 {
    let m_noise = noise.moisture_noise.get_noise_3d(pos.x, pos.y, pos.z);
    let m = (m_noise * params.moisture_noise_amp) as f64 * 0.5 + 0.5
        - mountain_influence * params.rain_shadow_strength as f64;
    m.clamp(0.0, 1.0)
}

pub fn biome_metric(
    pos: DVec3,
    elevation: f64,
    low_freq_elev: f64,
    mountain_influence: f64,
    params: &PlanetParams,
    noise: &ClimateNoise,
) -> Biome {
    let temp = temperature_at(pos, elevation, params, noise);
    let moist = moisture_at(pos, mountain_influence, params, noise);
    classify_biome(elevation, temp, moist, low_freq_elev, params)
}

pub fn elevation_low_freq_metric(pos: DVec3, noise: &ElevationNoise) -> LowFreqElevation {
    let lf = crate::elevation::metric_landform_sample(pos, noise);
    LowFreqElevation {
        low_freq_elev: lf.macro_displacement,
        warped_dir: lf.warped_dir,
        mountain_influence: lf.mountain_influence,
    }
}

pub fn elevation_split_metric(pos: DVec3, noise: &ElevationNoise) -> ElevationSplit {
    let lf = crate::elevation::metric_landform_sample(pos, noise);
    ElevationSplit {
        low_freq_elev: lf.macro_displacement,
        warped_dir: lf.warped_dir,
        mountain_influence: lf.mountain_influence,
        full_elev: lf.full_elevation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elevation::{elevation, elevation_at, elevation_params, ElevationNoise};
    use crate::params::{climate_noise, planet_params};
    use er_core::rng::rng_from_seed;
    use er_core::seed::PlanetSeed;
    use rand::RngCore;

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
    fn temperature_in_bounds() {
        let params = planet_params(PlanetSeed(0xC0FFEE));
        let noise = climate_noise(&params);
        let dirs = rand_dirs(0x1234, 1000);
        for d in &dirs {
            let t = temperature(*d, 0.5, &params, &noise);
            assert!(t >= 0.0 && t <= 1.0, "temperature {t} out of [0,1]");
        }
    }

    #[test]
    fn moisture_in_bounds() {
        let params = planet_params(PlanetSeed(0xC0FFEE));
        let noise = climate_noise(&params);
        let dirs = rand_dirs(0x5678, 1000);
        for d in &dirs {
            let m = moisture(*d, 0.0, &params, &noise);
            assert!(m >= 0.0 && m <= 1.0, "moisture {m} out of [0,1]");
        }
    }

    #[test]
    fn classify_biome_ocean_bands() {
        let params = planet_params(PlanetSeed(0xC0FFEE));
        let sl = params.sea_level;
        assert_eq!(
            classify_biome(sl - 0.1, 0.5, 0.5, 0.0, &params),
            Biome::OceanShallow
        );
        assert_eq!(
            classify_biome(sl - 0.4, 0.5, 0.5, 0.0, &params),
            Biome::OceanMid
        );
        assert_eq!(
            classify_biome(sl - 0.8, 0.5, 0.5, 0.0, &params),
            Biome::OceanDeep
        );
        assert_eq!(
            classify_biome(sl - 1.5, 0.5, 0.5, 0.0, &params),
            Biome::OceanAbyss
        );
    }

    #[test]
    fn classify_biome_beach() {
        let params = planet_params(PlanetSeed(0xC0FFEE));
        let sl = params.sea_level;
        assert_eq!(
            classify_biome(sl + 0.01, 0.5, 0.5, 0.0, &params),
            Biome::Beach
        );
    }

    #[test]
    fn classify_biome_whittaker() {
        let params = planet_params(PlanetSeed(0xC0FFEE));
        let el = params.sea_level + 0.2;
        assert_eq!(classify_biome(el, 0.1, 0.1, 0.0, &params), Biome::Tundra);
        assert_eq!(classify_biome(el, 0.1, 0.5, 0.0, &params), Biome::Forest);
        assert_eq!(classify_biome(el, 0.1, 0.8, 0.0, &params), Biome::Snow);
        assert_eq!(classify_biome(el, 0.4, 0.1, 0.0, &params), Biome::Grassland);
        assert_eq!(classify_biome(el, 0.4, 0.8, 0.0, &params), Biome::Forest);
        assert_eq!(classify_biome(el, 0.8, 0.1, 0.0, &params), Biome::Desert);
        assert_eq!(classify_biome(el, 0.8, 0.5, 0.0, &params), Biome::Grassland);
        assert_eq!(classify_biome(el, 0.8, 0.8, 0.0, &params), Biome::Jungle);
    }

    #[test]
    fn classify_biome_overrides() {
        let params = planet_params(PlanetSeed(0xC0FFEE));
        let el = params.sea_level + 0.2;
        assert_eq!(classify_biome(el, 0.8, 0.9, 0.0, &params), Biome::ToxicBog);
        assert_eq!(classify_biome(el, 0.5, 0.5, 1.5, &params), Biome::Volcanic);
        assert_eq!(
            classify_biome(0.9, 0.5, 0.5, 0.0, &params),
            Biome::Mountains
        );
    }

    #[test]
    fn biome_determinism() {
        let params = planet_params(PlanetSeed(0xC0FFEE));
        let noise = climate_noise(&params);
        let dirs = rand_dirs(0xABCD, 500);
        let pass1: Vec<Biome> = dirs
            .iter()
            .map(|d| biome(*d, 0.3, 0.2, 0.0, &params, &noise))
            .collect();
        let pass2: Vec<Biome> = dirs
            .iter()
            .map(|d| biome(*d, 0.3, 0.2, 0.0, &params, &noise))
            .collect();
        assert_eq!(pass1, pass2);
    }

    #[test]
    fn elevation_split_matches_elevation() {
        let params = elevation_params(PlanetSeed(0xC0FFEE));
        let noise = ElevationNoise::new(&params);
        let dirs = rand_dirs(0xBEEF, 500);
        for d in &dirs {
            let split = elevation_split(*d, &noise, &params);
            let elev = elevation(*d, &noise, &params);
            assert!(
                (split.full_elev - elev).abs() < 1e-5,
                "split full_elev {} != elevation {} (diff {})",
                split.full_elev,
                elev,
                (split.full_elev - elev).abs()
            );
        }
    }

    #[test]
    fn different_seeds_differ() {
        let pa = planet_params(PlanetSeed(0xC0FFEE));
        let na = climate_noise(&pa);
        let pb = planet_params(PlanetSeed(0xDEADBEEF));
        let nb = climate_noise(&pb);
        let dirs = rand_dirs(0xFEED, 200);
        let mut any_diff = false;
        for d in &dirs {
            let ta = temperature(*d, 0.3, &pa, &na);
            let tb = temperature(*d, 0.3, &pb, &nb);
            if ta.to_bits() != tb.to_bits() {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "different seeds produced identical temperatures");
    }

    // ---- metric tests ----

    const TEST_R: f64 = 6_371_000.0;

    fn metric_points(seed: u64, count: usize) -> Vec<DVec3> {
        rand_dirs(seed, count)
            .iter()
            .map(|d| crate::terrain_space::metric_surface_point(*d, TEST_R))
            .collect()
    }

    #[test]
    fn metric_temperature_at_in_bounds() {
        let pp = planet_params(PlanetSeed(0xC0FFEE));
        let cn = crate::params::climate_noise_metric(&pp, TEST_R);
        let points = metric_points(0x1111, 1000);
        for p in &points {
            let t = temperature_at(*p, 5.0, &pp, &cn);
            assert!(t >= 0.0 && t <= 1.0, "temperature_at {t} out of [0,1]");
        }
    }

    #[test]
    fn metric_moisture_at_in_bounds() {
        let params = planet_params(PlanetSeed(0xC0FFEE));
        let noise = crate::params::climate_noise_metric(&params, TEST_R);
        let points = metric_points(0x2222, 1000);
        for p in &points {
            let m = moisture_at(*p, 0.0, &params, &noise);
            assert!(m >= 0.0 && m <= 1.0, "moisture_at {m} out of [0,1]");
        }
    }

    #[test]
    fn metric_split_matches_elevation_at() {
        let params = crate::elevation::elevation_params(PlanetSeed(0xBEEF));
        let noise = ElevationNoise::new_metric(&params);
        let points = metric_points(0x5555, 500);
        for p in &points {
            let split = elevation_split_metric(*p, &noise);
            let full = elevation_at(*p, &noise);
            assert!(
                (split.full_elev - full).abs() < 1e-10,
                "diff {s} - {e} = {}",
                (split.full_elev - full).abs(),
                s = split.full_elev,
                e = full
            );
        }
    }

    #[test]
    fn metric_biome_determinism() {
        let pp = planet_params(PlanetSeed(0xC0FFEE));
        let cn = crate::params::climate_noise_metric(&pp, TEST_R);
        let ep = crate::elevation::elevation_params(PlanetSeed(0xCAFE));
        let en = ElevationNoise::new_metric(&ep);
        let points = metric_points(0x4444, 500);

        let pass1: Vec<Biome> = points
            .iter()
            .map(|p| {
                let low = elevation_low_freq_metric(*p, &en);
                let e = elevation_at(*p, &en);
                biome_metric(*p, e, low.low_freq_elev, low.mountain_influence, &pp, &cn)
            })
            .collect();

        let pass2: Vec<Biome> = points
            .iter()
            .map(|p| {
                let low = elevation_low_freq_metric(*p, &en);
                let e = elevation_at(*p, &en);
                biome_metric(*p, e, low.low_freq_elev, low.mountain_influence, &pp, &cn)
            })
            .collect();

        assert_eq!(pass1, pass2);
    }
}
