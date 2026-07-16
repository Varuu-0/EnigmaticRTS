//! Synchronous terrain samples consumed by terrain mesh workers.
//!
//! Learned terrain is allowed to refine only already-resident macro samples.
//! Fetching, decoding, and inference belong to a higher-level streaming system;
//! this module must always be safe to call from a mesh worker.

use crate::biome::{
    biome, biome_metric, elevation_low_freq, elevation_low_freq_metric, moisture, moisture_at,
    temperature, temperature_at, Biome,
};
use crate::cache::{CachedWorldData, WorldCache};
use crate::elevation::{elevation, elevation_at, ElevationNoise, ElevationParams};
use crate::params::{climate_noise, climate_noise_metric, ClimateNoise, PlanetParams};
use glam::DVec3;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerrainSourceMode {
    #[default]
    Procedural,
    HybridLearned,
    LearnedOnlyDiagnostic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerrainSampleSource {
    #[default]
    Procedural,
    LearnedMacro,
}

#[derive(Clone, Copy, Debug)]
pub struct TerrainSample {
    pub elevation: f64,
    pub low_freq_elev: f32,
    pub warped_dir: [f32; 3],
    pub moisture: f32,
    pub biome: Biome,
    pub mountain_influence: f32,
    pub temperature: f32,
    pub source: TerrainSampleSource,
}

impl From<CachedWorldData> for TerrainSample {
    fn from(value: CachedWorldData) -> Self {
        Self {
            elevation: value.elevation,
            low_freq_elev: value.low_freq_elev,
            warped_dir: value.warped_dir,
            moisture: value.moisture,
            biome: value.biome,
            mountain_influence: value.mountain_influence,
            temperature: value.temperature,
            source: TerrainSampleSource::Procedural,
        }
    }
}

pub trait TerrainField: Send + Sync {
    fn sample(&self, dir: DVec3) -> TerrainSample;
    fn revision(&self) -> u64 {
        0
    }
}

pub struct ProceduralTerrainField {
    elevation_params: ElevationParams,
    planet_params: PlanetParams,
    noise: ElevationNoise,
    climate_noise: ClimateNoise,
    cache: Option<Arc<WorldCache>>,
    metric_radius_m: Option<f64>,
}

impl ProceduralTerrainField {
    pub fn new(elevation_params: ElevationParams, planet_params: PlanetParams) -> Self {
        let noise = ElevationNoise::new(&elevation_params);
        let climate_noise = climate_noise(&planet_params);
        Self {
            elevation_params,
            planet_params,
            noise,
            climate_noise,
            cache: None,
            metric_radius_m: None,
        }
    }

    pub fn with_cache(
        elevation_params: ElevationParams,
        planet_params: PlanetParams,
        cache: Arc<WorldCache>,
    ) -> Self {
        let noise = ElevationNoise::new(&elevation_params);
        let climate_noise = climate_noise(&planet_params);
        Self {
            elevation_params,
            planet_params,
            noise,
            climate_noise,
            cache: Some(cache),
            metric_radius_m: None,
        }
    }

    pub fn new_metric(
        elevation_params: ElevationParams,
        planet_params: PlanetParams,
        planet_radius_m: f64,
    ) -> Self {
        let noise = ElevationNoise::new_metric(&elevation_params);
        let climate_noise = climate_noise_metric(&planet_params, planet_radius_m);
        Self {
            elevation_params,
            planet_params,
            noise,
            climate_noise,
            cache: None,
            metric_radius_m: Some(planet_radius_m),
        }
    }

    pub fn with_cache_metric(
        elevation_params: ElevationParams,
        planet_params: PlanetParams,
        cache: Arc<WorldCache>,
        planet_radius_m: f64,
    ) -> Self {
        let noise = ElevationNoise::new_metric(&elevation_params);
        let climate_noise = climate_noise_metric(&planet_params, planet_radius_m);
        Self {
            elevation_params,
            planet_params,
            noise,
            climate_noise,
            cache: Some(cache),
            metric_radius_m: Some(planet_radius_m),
        }
    }

    fn compute_sample(&self, dir: DVec3) -> TerrainSample {
        if let Some(radius) = self.metric_radius_m {
            let pos = crate::terrain_space::metric_surface_point(dir, radius);
            let split = elevation_low_freq_metric(pos, &self.noise);
            let moist = moisture_at(
                pos,
                split.mountain_influence,
                &self.planet_params,
                &self.climate_noise,
            );
            let elev = elevation_at(pos, &self.noise);
            let temp = temperature_at(pos, elev, &self.planet_params, &self.climate_noise);
            let biom = biome_metric(
                pos,
                elev,
                split.low_freq_elev,
                split.mountain_influence,
                &self.planet_params,
                &self.climate_noise,
            );

            TerrainSample {
                elevation: elev,
                low_freq_elev: split.low_freq_elev as f32,
                warped_dir: [
                    split.warped_dir.x as f32,
                    split.warped_dir.y as f32,
                    split.warped_dir.z as f32,
                ],
                moisture: moist as f32,
                biome: biom,
                mountain_influence: split.mountain_influence as f32,
                temperature: temp as f32,
                source: TerrainSampleSource::Procedural,
            }
        } else {
            let split = elevation_low_freq(dir, &self.noise, &self.elevation_params);
            let moist = moisture(
                dir,
                split.mountain_influence,
                &self.planet_params,
                &self.climate_noise,
            );
            let elev = elevation(dir, &self.noise, &self.elevation_params);
            let temp = temperature(dir, elev, &self.planet_params, &self.climate_noise);
            let biom = biome(
                dir,
                elev,
                split.low_freq_elev,
                split.mountain_influence,
                &self.planet_params,
                &self.climate_noise,
            );

            TerrainSample {
                elevation: elev,
                low_freq_elev: split.low_freq_elev as f32,
                warped_dir: [
                    split.warped_dir.x as f32,
                    split.warped_dir.y as f32,
                    split.warped_dir.z as f32,
                ],
                moisture: moist as f32,
                biome: biom,
                mountain_influence: split.mountain_influence as f32,
                temperature: temp as f32,
                source: TerrainSampleSource::Procedural,
            }
        }
    }
}

impl TerrainField for ProceduralTerrainField {
    fn sample(&self, dir: DVec3) -> TerrainSample {
        let compute = || {
            let sample = self.compute_sample(dir);
            CachedWorldData {
                elevation: sample.elevation,
                low_freq_elev: sample.low_freq_elev,
                warped_dir: sample.warped_dir,
                moisture: sample.moisture,
                biome: sample.biome,
                mountain_influence: sample.mountain_influence,
                temperature: sample.temperature,
            }
        };

        match &self.cache {
            Some(cache) => cache.get_or_insert(dir, compute).into(),
            None => compute().into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MacroTerrainSample {
    pub elevation: f64,
}

pub trait MacroTerrainField: Send + Sync {
    fn sample_resident(&self, dir: DVec3) -> Option<MacroTerrainSample>;
    fn revision(&self) -> u64 {
        0
    }
}

pub struct HybridTerrainField {
    fallback: Arc<dyn TerrainField>,
    macro_field: Arc<dyn MacroTerrainField>,
}

impl HybridTerrainField {
    pub fn new(fallback: Arc<dyn TerrainField>, macro_field: Arc<dyn MacroTerrainField>) -> Self {
        Self {
            fallback,
            macro_field,
        }
    }
}

impl TerrainField for HybridTerrainField {
    fn sample(&self, dir: DVec3) -> TerrainSample {
        let fallback = self.fallback.sample(dir);
        let Some(macro_sample) = self.macro_field.sample_resident(dir) else {
            return fallback;
        };
        if !macro_sample.elevation.is_finite() {
            return fallback;
        }

        TerrainSample {
            elevation: macro_sample.elevation
                + (fallback.elevation - fallback.low_freq_elev as f64),
            // The learned tile supplies local relief, not a globally coherent
            // continent mask. Keep the procedural macro field for shoreline,
            // biome, and ambient-occlusion decisions.
            low_freq_elev: fallback.low_freq_elev,
            source: TerrainSampleSource::LearnedMacro,
            ..fallback
        }
    }

    fn revision(&self) -> u64 {
        self.fallback.revision().max(self.macro_field.revision())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elevation::{elevation, elevation_params};
    use crate::params::planet_params;
    use er_core::math::uv_to_dir;
    use er_core::seed::PlanetSeed;

    struct ConstantMacro(f64);

    impl MacroTerrainField for ConstantMacro {
        fn sample_resident(&self, _: DVec3) -> Option<MacroTerrainSample> {
            Some(MacroTerrainSample { elevation: self.0 })
        }
    }

    #[test]
    fn procedural_field_matches_direct_elevation() {
        let seed = PlanetSeed(0xC0FFEE);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new(ep, pp);
        let noise = ElevationNoise::new(&ep);
        let dir = uv_to_dir(3, 0.31, 0.72);

        let sample = field.sample(dir);
        assert_eq!(
            sample.elevation.to_bits(),
            elevation(dir, &noise, &ep).to_bits()
        );
    }

    #[test]
    fn hybrid_uses_resident_relief_and_preserves_procedural_macro() {
        let seed = PlanetSeed(0xC0FFEE);
        let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
            elevation_params(seed),
            planet_params(seed),
        ));
        let hybrid = HybridTerrainField::new(fallback.clone(), Arc::new(ConstantMacro(1.25)));
        let dir = uv_to_dir(1, 0.21, 0.61);

        let pro = fallback.sample(dir);
        let learned = hybrid.sample(dir);
        assert_eq!(learned.source, TerrainSampleSource::LearnedMacro);
        assert_eq!(learned.low_freq_elev, pro.low_freq_elev);
        assert_eq!(
            learned.elevation.to_bits(),
            (1.25 + pro.elevation - pro.low_freq_elev as f64).to_bits()
        );
    }

    #[test]
    fn metric_field_creates_fallen() {
        const TEST_R: f64 = 6_371_000.0;

        let seed = PlanetSeed(0xACC);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new_metric(ep, pp, TEST_R);
        for face in 0..6u8 {
            for i in 0..4 {
                for j in 0..4 {
                    let u = (i as f64 + 0.5) / 4.0;
                    let v = (j as f64 + 0.5) / 4.0;
                    let dir = uv_to_dir(face, u, v);
                    let s = field.sample(dir);
                    assert!(s.elevation.is_finite());
                    assert!(s.moisture >= 0.0 && s.moisture <= 1.0);
                    assert!(s.temperature >= 0.0 && s.temperature <= 1.0);
                }
            }
        }
    }
}
