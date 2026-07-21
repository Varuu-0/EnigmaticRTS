//! Synchronous terrain samples consumed by terrain mesh workers.
//!
//! Learned terrain is allowed to refine only already-resident macro samples.
//! Fetching, decoding, and inference belong to a higher-level streaming system;
//! this module must always be safe to call from a mesh worker.

use crate::biome::{
    classify_biome, elevation_low_freq, elevation_split, moisture, moisture_at, temperature,
    temperature_at, Biome,
};
use crate::cache::{CachedWorldData, WorldCache};
use crate::elevation::{elevation, metric_landform_sample, ElevationNoise, ElevationParams};
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
    /// Regional drainage/channel mask used by terrain materials.
    pub drainage: f32,
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
            drainage: value.drainage,
            source: TerrainSampleSource::Procedural,
        }
    }
}

pub trait TerrainField: Send + Sync {
    fn sample(&self, dir: DVec3) -> TerrainSample;
    fn sample_elevation(&self, dir: DVec3) -> f64 {
        self.sample(dir).elevation
    }
    /// Maximum safe spacing for interpolated CPU mesh samples. Fields with
    /// learned or unknown high-frequency content keep exact per-vertex sampling.
    fn mesh_sample_spacing_m(&self) -> Option<f64> {
        None
    }
    fn revision(&self) -> u64 {
        0
    }
}

/// Low-frequency macro elevation sample from the procedural global macro
/// field. This controls shoreline, broad biome classification, and stable
/// water classification.
#[derive(Clone, Copy, Debug)]
pub struct MacroSample {
    pub elevation: f64,
    pub warped_dir: DVec3,
    pub mountain_influence: f64,
}

/// High-frequency residual displacement from the procedural residual field.
/// Added on top of the macro elevation to produce the composed terrain
/// elevation.
#[derive(Clone, Copy, Debug)]
pub struct ResidualSample {
    pub elevation: f64,
}

/// Synchronous, `Send + Sync` pure-sampling contract for the global macro
/// elevation field.
///
/// The macro field provides the low-frequency continent/mountain elevation
/// that controls shoreline, broad biome logic, and stable water
/// classification. It is continuous across cube-face boundaries because it
/// samples from a 3D metric surface point, not a face-local coordinate.
pub trait GlobalMacroField: Send + Sync {
    fn sample_macro(&self, dir: DVec3) -> MacroSample;
    fn revision(&self) -> u64 {
        0
    }
}

/// Synchronous, `Send + Sync` pure-sampling contract for the procedural
/// residual field.
///
/// The residual field provides the high-frequency hill/detail displacement
/// that is added on top of the macro elevation to produce the composed
/// terrain elevation.
pub trait ProceduralResidualField: Send + Sync {
    fn sample_residual(&self, dir: DVec3) -> ResidualSample;
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
            // Compute the full landform sample once so macro and full
            // elevation cannot drift and we avoid duplicate noise work.
            let lf = metric_landform_sample(pos, &self.noise);
            let moist = moisture_at(
                pos,
                lf.mountain_influence,
                &self.planet_params,
                &self.climate_noise,
            );
            let elev = lf.full_elevation;
            let temp = temperature_at(pos, elev, &self.planet_params, &self.climate_noise);
            let biom = classify_biome(
                elev,
                temp,
                moist,
                lf.macro_displacement,
                &self.planet_params,
            );

            TerrainSample {
                elevation: elev,
                low_freq_elev: lf.macro_displacement as f32,
                warped_dir: [
                    lf.warped_dir.x as f32,
                    lf.warped_dir.y as f32,
                    lf.warped_dir.z as f32,
                ],
                moisture: moist as f32,
                biome: biom,
                mountain_influence: lf.mountain_influence as f32,
                temperature: temp as f32,
                drainage: lf.drainage,
                source: TerrainSampleSource::Procedural,
            }
        } else {
            // Compute macro and residual layers once; the previous separate
            // low-frequency and full calls repeated warp/continent/mountain.
            let split = elevation_split(dir, &self.noise, &self.elevation_params);
            let moist = moisture(
                dir,
                split.mountain_influence,
                &self.planet_params,
                &self.climate_noise,
            );
            let elev = split.full_elev;
            let temp = temperature(dir, elev, &self.planet_params, &self.climate_noise);
            let biom = classify_biome(elev, temp, moist, split.low_freq_elev, &self.planet_params);

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
                drainage: 0.0,
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
                drainage: sample.drainage,
            }
        };

        match &self.cache {
            Some(cache) => cache.get_or_insert(dir, compute).into(),
            None => compute().into(),
        }
    }

    fn sample_elevation(&self, dir: DVec3) -> f64 {
        let compute = || {
            if let Some(radius) = self.metric_radius_m {
                let pos = crate::terrain_space::metric_surface_point(dir, radius);
                metric_landform_sample(pos, &self.noise).full_elevation
            } else {
                elevation(dir, &self.noise, &self.elevation_params)
            }
        };
        match &self.cache {
            Some(cache) => cache.get_or_insert_elevation(dir, compute),
            None => compute(),
        }
    }

    fn mesh_sample_spacing_m(&self) -> Option<f64> {
        self.metric_radius_m.map(|_| 80.0)
    }
}

impl GlobalMacroField for ProceduralTerrainField {
    fn sample_macro(&self, dir: DVec3) -> MacroSample {
        if let Some(radius) = self.metric_radius_m {
            let pos = crate::terrain_space::metric_surface_point(dir, radius);
            let lf = metric_landform_sample(pos, &self.noise);
            MacroSample {
                elevation: lf.macro_displacement,
                warped_dir: lf.warped_dir,
                mountain_influence: lf.mountain_influence,
            }
        } else {
            let low = elevation_low_freq(dir, &self.noise, &self.elevation_params);
            MacroSample {
                elevation: low.low_freq_elev,
                warped_dir: low.warped_dir,
                mountain_influence: low.mountain_influence,
            }
        }
    }
}

impl ProceduralResidualField for ProceduralTerrainField {
    fn sample_residual(&self, dir: DVec3) -> ResidualSample {
        if let Some(radius) = self.metric_radius_m {
            let pos = crate::terrain_space::metric_surface_point(dir, radius);
            let lf = metric_landform_sample(pos, &self.noise);
            ResidualSample {
                elevation: lf.residual_displacement,
            }
        } else {
            let full = elevation(dir, &self.noise, &self.elevation_params);
            let macro_elev =
                elevation_low_freq(dir, &self.noise, &self.elevation_params).low_freq_elev;
            ResidualSample {
                elevation: full - macro_elev,
            }
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

        // The learned tile supplies macro elevation; the procedural
        // fallback supplies the residual (high-frequency detail).
        let procedural_residual = fallback.elevation - fallback.low_freq_elev as f64;

        TerrainSample {
            elevation: macro_sample.elevation + procedural_residual,
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
    use crate::biome::{elevation_low_freq, elevation_low_freq_metric};
    use crate::elevation::{elevation, elevation_params, ElevationNoise};
    use crate::params::planet_params;
    use crate::terrain_space::metric_surface_point;
    use er_core::math::uv_to_dir;
    use er_core::rng::rng_from_seed;
    use er_core::seed::PlanetSeed;
    use rand::RngCore;

    struct ConstantMacro(f64);

    impl MacroTerrainField for ConstantMacro {
        fn sample_resident(&self, _: DVec3) -> Option<MacroTerrainSample> {
            Some(MacroTerrainSample { elevation: self.0 })
        }
    }

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
        assert_eq!(
            field.sample_elevation(dir).to_bits(),
            sample.elevation.to_bits()
        );
    }

    #[test]
    fn sparse_mesh_sampling_is_procedural_metric_only() {
        let seed = PlanetSeed(0xC0FFEE);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let procedural = Arc::new(ProceduralTerrainField::new_metric(ep, pp, 6_371_000.0));
        assert_eq!(procedural.mesh_sample_spacing_m(), Some(80.0));

        let fallback: Arc<dyn TerrainField> = procedural;
        let hybrid = HybridTerrainField::new(fallback, Arc::new(ConstantMacro(0.0)));
        assert_eq!(hybrid.mesh_sample_spacing_m(), None);
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
    fn hybrid_shoreline_mask_is_invariant_over_learned_relief() {
        let seed = PlanetSeed(0xC0FFEE);
        let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
            elevation_params(seed),
            planet_params(seed),
        ));
        let dir = uv_to_dir(1, 0.21, 0.61);
        let procedural = fallback.sample(dir);
        let procedural_water = procedural.low_freq_elev < 0.0;

        for learned_macro in [-5.0, -1.0, 0.0, 1.0, 5.0] {
            let hybrid =
                HybridTerrainField::new(fallback.clone(), Arc::new(ConstantMacro(learned_macro)));
            let sample = hybrid.sample(dir);
            assert_eq!(sample.low_freq_elev, procedural.low_freq_elev);
            assert_eq!(sample.low_freq_elev < 0.0, procedural_water);
        }
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

    // ---- Macro + residual = composed elevation ----

    #[test]
    fn macro_plus_residual_equals_composed_legacy() {
        let seed = PlanetSeed(0xC0FFEE);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new(ep, pp);
        let dirs = rand_dirs(0xBEEF, 500);

        for dir in &dirs {
            let sample = field.sample(*dir);
            let macro_s = field.sample_macro(*dir);
            let residual = field.sample_residual(*dir);
            let composed = macro_s.elevation + residual.elevation;
            assert!(
                (composed - sample.elevation).abs() < 1e-5,
                "legacy macro+residual != composed: {composed} vs {} (diff {})",
                sample.elevation,
                (composed - sample.elevation).abs()
            );
        }
    }

    #[test]
    fn macro_plus_residual_equals_composed_metric() {
        const TEST_R: f64 = 6_371_000.0;
        let seed = PlanetSeed(0xCAFE);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new_metric(ep, pp, TEST_R);
        let dirs = rand_dirs(0xF00D, 500);

        for dir in &dirs {
            let sample = field.sample(*dir);
            let macro_s = field.sample_macro(*dir);
            let residual = field.sample_residual(*dir);
            let composed = macro_s.elevation + residual.elevation;
            assert!(
                (composed - sample.elevation).abs() < 1e-10,
                "metric macro+residual != composed: {composed} vs {} (diff {})",
                sample.elevation,
                (composed - sample.elevation).abs()
            );
        }
    }

    // ---- Procedural macro standalone sampling ----

    #[test]
    fn procedural_macro_standalone_matches_low_freq_legacy() {
        let seed = PlanetSeed(0xC0FFEE);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new(ep, pp);
        let noise = ElevationNoise::new(&ep);
        let dirs = rand_dirs(0x1234, 200);

        for dir in &dirs {
            let macro_s = field.sample_macro(*dir);
            let low = elevation_low_freq(*dir, &noise, &ep);
            assert_eq!(macro_s.elevation.to_bits(), low.low_freq_elev.to_bits());
            assert!((macro_s.warped_dir - low.warped_dir).length() < 1e-12);
            assert_eq!(
                macro_s.mountain_influence.to_bits(),
                low.mountain_influence.to_bits()
            );
        }
    }

    #[test]
    fn procedural_macro_standalone_matches_low_freq_metric() {
        const TEST_R: f64 = 6_371_000.0;
        let seed = PlanetSeed(0xDEAD);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new_metric(ep, pp, TEST_R);
        let noise = ElevationNoise::new_metric(&ep);
        let dirs = rand_dirs(0x5678, 200);

        for dir in &dirs {
            let pos = metric_surface_point(*dir, TEST_R);
            let macro_s = field.sample_macro(*dir);
            let low = elevation_low_freq_metric(pos, &noise);
            assert_eq!(macro_s.elevation.to_bits(), low.low_freq_elev.to_bits());
            assert!((macro_s.warped_dir - low.warped_dir).length() < 1e-12);
            assert_eq!(
                macro_s.mountain_influence.to_bits(),
                low.mountain_influence.to_bits()
            );
        }
    }

    // ---- Hybrid shoreline invariance with explicit residual ----

    #[test]
    fn hybrid_residual_is_procedural_detail() {
        let seed = PlanetSeed(0xC0FFEE);
        let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
            elevation_params(seed),
            planet_params(seed),
        ));
        let dir = uv_to_dir(2, 0.41, 0.59);
        let pro = fallback.sample(dir);
        let expected_residual = pro.elevation - pro.low_freq_elev as f64;

        for learned_macro in [-5.0, -1.0, 0.0, 1.0, 5.0] {
            let hybrid =
                HybridTerrainField::new(fallback.clone(), Arc::new(ConstantMacro(learned_macro)));
            let sample = hybrid.sample(dir);
            // Composed = learned macro + procedural residual.
            assert_eq!(
                sample.elevation.to_bits(),
                (learned_macro + expected_residual).to_bits()
            );
            // Shoreline ownership stays with the procedural macro.
            assert_eq!(sample.low_freq_elev, pro.low_freq_elev);
        }
    }

    // ---- Deterministic behavior ----

    #[test]
    fn macro_field_is_deterministic() {
        let seed = PlanetSeed(0xBEEF);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new(ep, pp);
        let dirs = rand_dirs(0xCAFE, 100);

        for dir in &dirs {
            let a = field.sample_macro(*dir);
            let b = field.sample_macro(*dir);
            assert_eq!(a.elevation.to_bits(), b.elevation.to_bits());
        }
    }

    #[test]
    fn residual_field_is_deterministic() {
        const TEST_R: f64 = 6_371_000.0;
        let seed = PlanetSeed(0xF00D);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new_metric(ep, pp, TEST_R);
        let dirs = rand_dirs(0xBEEF, 100);

        for dir in &dirs {
            let a = field.sample_residual(*dir);
            let b = field.sample_residual(*dir);
            assert_eq!(a.elevation.to_bits(), b.elevation.to_bits());
        }
    }

    // ---- Metric spherical continuity ----

    #[test]
    fn metric_macro_continuous_across_cube_face_edges() {
        const TEST_R: f64 = 6_371_000.0;
        let seed = PlanetSeed(0xACC);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new_metric(ep, pp, TEST_R);

        // shared edge: +X posU == +Y posU
        for i in 0..32 {
            let t = (i as f64 + 0.5) / 32.0;
            let d_x = uv_to_dir(0, 1.0, t);
            let d_y = uv_to_dir(2, 1.0, t);
            let mx = field.sample_macro(d_x);
            let my = field.sample_macro(d_y);
            assert!(
                (mx.elevation - my.elevation).abs() < 1e-9,
                "macro edge discontinuity at {i}: {} vs {}",
                mx.elevation,
                my.elevation
            );
        }

        // shared edge: +X negV === -Z posU
        for j in 0..32 {
            let t = (j as f64 + 0.5) / 32.0;
            let d_xn = uv_to_dir(0, t, 0.0);
            let d_zn = uv_to_dir(5, 1.0, t);
            let mx = field.sample_macro(d_xn);
            let mz = field.sample_macro(d_zn);
            assert!(
                (mx.elevation - mz.elevation).abs() < 1e-9,
                "macro edge discontinuity at j={j}: {} vs {}",
                mx.elevation,
                mz.elevation
            );
        }
    }

    #[test]
    fn metric_residual_continuous_across_cube_face_edges() {
        const TEST_R: f64 = 6_371_000.0;
        let seed = PlanetSeed(0xACC);
        let ep = elevation_params(seed);
        let pp = planet_params(seed);
        let field = ProceduralTerrainField::new_metric(ep, pp, TEST_R);

        // shared edge: +X posU == +Y posU
        for i in 0..32 {
            let t = (i as f64 + 0.5) / 32.0;
            let d_x = uv_to_dir(0, 1.0, t);
            let d_y = uv_to_dir(2, 1.0, t);
            let rx = field.sample_residual(d_x);
            let ry = field.sample_residual(d_y);
            assert!(
                (rx.elevation - ry.elevation).abs() < 1e-9,
                "residual edge discontinuity at {i}: {} vs {}",
                rx.elevation,
                ry.elevation
            );
        }
    }
}
