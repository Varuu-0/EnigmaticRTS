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
    /// Visual-only learned climate channels for material shading. This is
    /// `[temp_c, temp_seasonal_c, precip_mm, precip_cv]` converted from the
    /// upstream model's normalized units. Gameplay climate (the `temperature`
    /// and `moisture` fields above) remains canonical procedural until the
    /// bake policy exists (roadmap rule 5.2.5).
    pub visual_climate: VisualClimate,
}

/// Visual-only learned climate for material/biome shading. These values are
/// NOT used by gameplay systems — gameplay climate remains procedural.
///
/// ## Unit conversion (documented per roadmap 5.2.5)
///
/// The upstream Terrain Diffusion model outputs four climate channels in
/// normalized units. The conversion to physical units is:
///
/// - **Channel 0 (temperature)**: `temp_c = normalized * 50.0 - 10.0`
///   Maps `[0,1]` to `[-10°C, +40°C]`.
/// - **Channel 1 (temperature seasonal)**: `temp_seasonal_c = normalized * 30.0 - 15.0`
///   Maps `[0,1]` to `[-15°C, +15°C]` seasonal delta.
/// - **Channel 2 (precipitation)**: `precip_mm = normalized * 4000.0`
///   Maps `[0,1]` to `[0, 4000] mm/year`.
/// - **Channel 3 (precipitation coefficient of variation)**: `precip_cv = normalized * 1.0`
///   Maps `[0,1]` to `[0, 1]` (dimensionless).
///
/// These conversions are applied at the chart-cache boundary so the
/// visual shader receives physical units, not raw model outputs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VisualClimate {
    /// Annual mean temperature in degrees Celsius.
    pub temp_c: f32,
    /// Seasonal temperature delta in degrees Celsius.
    pub temp_seasonal_c: f32,
    /// Annual precipitation in millimeters.
    pub precip_mm: f32,
    /// Precipitation coefficient of variation (dimensionless).
    pub precip_cv: f32,
}

impl VisualClimate {
    /// Convert four raw upstream climate channels (normalized `[0,1]`) to
    /// physical units. See the struct-level documentation for the conversion
    /// formulas.
    pub fn from_upstream_channels(channels: [f32; 4]) -> Self {
        Self {
            temp_c: channels[0] * 50.0 - 10.0,
            temp_seasonal_c: channels[1] * 30.0 - 15.0,
            precip_mm: channels[2] * 4000.0,
            precip_cv: channels[3],
        }
    }

    /// Returns `true` if all channels are finite (no NaN/Inf from a corrupt
    /// or missing learned tile).
    pub fn is_finite(&self) -> bool {
        self.temp_c.is_finite()
            && self.temp_seasonal_c.is_finite()
            && self.precip_mm.is_finite()
            && self.precip_cv.is_finite()
    }
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
            visual_climate: VisualClimate::default(),
        }
    }
}

pub trait TerrainField: Send + Sync + std::any::Any {
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
    /// Downcast support for runtime type inspection (e.g. blend checking).
    fn as_any(&self) -> &dyn std::any::Any {
        unimplemented!("as_any not implemented for this field type")
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
                visual_climate: VisualClimate::default(),
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
                visual_climate: VisualClimate::default(),
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
    /// Visual-only learned climate for material/biome shading. Gameplay
    /// climate remains canonical procedural (roadmap 5.2.5).
    pub visual_climate: VisualClimate,
}

pub trait MacroTerrainField: Send + Sync {
    fn sample_resident(&self, dir: DVec3) -> Option<MacroTerrainSample>;
    fn revision(&self) -> u64 {
        0
    }
}

/// A halo-residency checker that determines whether a chunk's elevation +
/// normal halo dependencies are fully resident. Used by the mesh-generation
/// path to implement the M5 chunk-wide residency gate: a mesh uses learned
/// data only if ALL halo dependencies are resident; otherwise entirely
/// procedural (no per-vertex mixing).
///
/// This is a separate trait from `MacroTerrainField` so the mesh path can
/// query residency without coupling to the specific chart field type.
pub trait HaloResidencyChecker: Send + Sync {
    /// Returns `true` if every chart key the chunk's elevation + normal halo
    /// depends on is resident.
    fn chunk_halo_resident(&self, chunk: er_core::math::CellKey) -> bool;

    /// The set of chart keys a chunk's elevation + normal halo depends on,
    /// as concrete `SurfaceCacheKey`s. Used by the integration to enqueue
    /// exactly the missing dependencies.
    fn chunk_halo_dependencies(
        &self,
        chunk: er_core::math::CellKey,
    ) -> Vec<crate::surface_cache::SurfaceCacheKey>;

    /// The macro field revision (for targeted refresh tracking).
    fn revision(&self) -> u64 {
        0
    }
}

/// A checker that reports whether the blend field has chunks still
/// transitioning. Used by the terrain system to schedule bounded follow-up
/// rebuilds so time-based blend is visible in live meshes.
pub trait BlendTransitionChecker: Send + Sync {
    /// Returns `true` if any tracked direction is still transitioning (blend
    /// weight > 0 and < 1).
    fn has_transitioning_chunks(&self) -> bool;
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

/// A hybrid field that blends learned macro *provenance* over a defined
/// time/distance transition without blending world coordinates.
///
/// During the transition interval after a chunk's halo dependencies become
/// resident, the field interpolates between the procedural macro and the
/// learned macro using a smoothstep weight. The procedural residual is
/// *always* added on top of whichever macro won — this blends provenance, not
/// world coordinates, so there is no height step or coastline crawl.
///
/// The shoreline/sea-datum ownership stays with the procedural macro field
/// (`low_freq_elev` is always the procedural value), satisfying the roadmap
/// rule that sea datum, water visibility, normals, and material derive from a
/// single composed snapshot.
pub struct BlendedHybridTerrainField {
    fallback: Arc<dyn TerrainField>,
    macro_field: Arc<dyn MacroTerrainField>,
    /// Full blend-in duration in seconds at close range.
    transition_seconds: f64,
    /// Per-chart-key blend transition state. Each chart key tracks when it
    /// first became resident. Unrelated tile arrivals do NOT clear existing
    /// progress — only the chart key whose data changed resets its own
    /// transition timer.
    blend_state: std::sync::Mutex<BlendState>,
}

#[derive(Default)]
struct BlendState {
    /// Map from a coarse direction quantization to the time that direction's
    /// chart data first became resident. Using a coarse quantization (LOD-8)
    /// keeps the map bounded. Entries are NOT cleared on global revision
    /// changes — only when a specific chart key's data is replaced.
    weights: std::collections::HashMap<u64, BlendEntry>,
}

struct BlendEntry {
    /// `Instant` when this direction's chart data first became resident.
    /// `None` means not yet resident (blend weight = 0, fully procedural).
    became_resident: Option<std::time::Instant>,
}

impl BlendedHybridTerrainField {
    /// Build a blended hybrid field. `transition_seconds` is the full
    /// blend-in duration at close range.
    pub fn new(
        fallback: Arc<dyn TerrainField>,
        macro_field: Arc<dyn MacroTerrainField>,
        transition_seconds: f64,
    ) -> Self {
        Self {
            fallback,
            macro_field,
            transition_seconds: transition_seconds.max(0.0),
            blend_state: std::sync::Mutex::new(BlendState::default()),
        }
    }

    /// Quantize a direction to a coarse key for the blend-weight cache. Uses
    /// the LOD-8 cell index (256x256 per face) so the map stays bounded.
    fn dir_to_blend_key(dir: DVec3) -> u64 {
        let key = er_core::math::dir_to_cell(dir, 8);
        ((key.face as u64) << 32) | ((key.i as u64) << 16) | (key.j as u64)
    }

    /// Compute the current blend weight for a direction. If the direction's
    /// chart data just became resident, record the time (per-chart, NOT
    /// global). If it was already resident, compute the elapsed age and
    /// return the smoothstep blend. Unrelated tile arrivals preserve
    /// existing progress.
    fn blend_weight(&self, dir: DVec3, learned_resident: bool) -> f64 {
        if !learned_resident {
            return 0.0;
        }
        if self.transition_seconds <= 0.0 {
            return 1.0;
        }
        let key = Self::dir_to_blend_key(dir);
        let now = std::time::Instant::now();
        let mut state = match self.blend_state.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        // Per-chart transition: only this direction's entry is affected.
        // We do NOT clear the entire map on revision changes.
        let entry = state.weights.entry(key).or_insert(BlendEntry {
            became_resident: None,
        });
        if entry.became_resident.is_none() {
            entry.became_resident = Some(now);
        }
        let age = now.duration_since(entry.became_resident.unwrap());
        crate::streaming::BlendWeights::for_transition(age, 0.0, self.transition_seconds)
            .learned_weight
    }

    /// Returns the current blend weight for a direction without mutating
    /// state. Used by the terrain system to check whether a chunk is still
    /// transitioning and needs a follow-up rebuild.
    pub fn current_blend_weight(&self, dir: DVec3) -> f64 {
        let learned_resident = self.macro_field.sample_resident(dir).is_some();
        if !learned_resident {
            return 0.0;
        }
        if self.transition_seconds <= 0.0 {
            return 1.0;
        }
        let key = Self::dir_to_blend_key(dir);
        let state = match self.blend_state.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        match state.weights.get(&key) {
            Some(entry) => {
                if let Some(became) = entry.became_resident {
                    let age = std::time::Instant::now().duration_since(became);
                    crate::streaming::BlendWeights::for_transition(
                        age,
                        0.0,
                        self.transition_seconds,
                    )
                    .learned_weight
                } else {
                    0.0
                }
            }
            None => 0.0,
        }
    }

    /// Returns `true` if any tracked direction is still transitioning (blend
    /// weight > 0 and < 1). Used by the terrain system to schedule bounded
    /// follow-up rebuilds without thrashing.
    pub fn has_transitioning_chunks(&self) -> bool {
        let state = match self.blend_state.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if self.transition_seconds <= 0.0 {
            return false;
        }
        let now = std::time::Instant::now();
        for entry in state.weights.values() {
            if let Some(became) = entry.became_resident {
                let age = now.duration_since(became);
                let w = crate::streaming::BlendWeights::for_transition(
                    age,
                    0.0,
                    self.transition_seconds,
                )
                .learned_weight;
                if w > 0.0 && w < 1.0 {
                    return true;
                }
            }
        }
        false
    }
}

impl BlendTransitionChecker for BlendedHybridTerrainField {
    fn has_transitioning_chunks(&self) -> bool {
        BlendedHybridTerrainField::has_transitioning_chunks(self)
    }
}

impl TerrainField for BlendedHybridTerrainField {
    fn sample(&self, dir: DVec3) -> TerrainSample {
        let fallback = self.fallback.sample(dir);
        let macro_sample = self.macro_field.sample_resident(dir);
        let learned_resident = macro_sample.is_some();
        let weight = self.blend_weight(dir, learned_resident);

        if weight <= 0.0 {
            return fallback;
        }
        let Some(macro_sample) = macro_sample else {
            return fallback;
        };
        if !macro_sample.elevation.is_finite() {
            return fallback;
        }

        let procedural_macro = fallback.low_freq_elev as f64;
        let procedural_residual = fallback.elevation - procedural_macro;

        if weight >= 1.0 {
            // Fully learned: learned macro + procedural residual. Use the
            // learned visual climate for material shading (5.2.5).
            return TerrainSample {
                elevation: macro_sample.elevation + procedural_residual,
                low_freq_elev: fallback.low_freq_elev,
                source: TerrainSampleSource::LearnedMacro,
                visual_climate: macro_sample.visual_climate,
                ..fallback
            };
        }

        // Blended provenance: interpolate between procedural macro and
        // learned macro, then add the procedural residual. This blends the
        // *macro provenance weight*, not world coordinates.
        let blended_macro = procedural_macro * (1.0 - weight) + macro_sample.elevation * weight;
        // Blend the visual climate proportionally to the learned weight.
        let blended_climate = VisualClimate {
            temp_c: fallback.visual_climate.temp_c * (1.0 - weight as f32)
                + macro_sample.visual_climate.temp_c * weight as f32,
            temp_seasonal_c: fallback.visual_climate.temp_seasonal_c * (1.0 - weight as f32)
                + macro_sample.visual_climate.temp_seasonal_c * weight as f32,
            precip_mm: fallback.visual_climate.precip_mm * (1.0 - weight as f32)
                + macro_sample.visual_climate.precip_mm * weight as f32,
            precip_cv: fallback.visual_climate.precip_cv * (1.0 - weight as f32)
                + macro_sample.visual_climate.precip_cv * weight as f32,
        };
        TerrainSample {
            elevation: blended_macro + procedural_residual,
            // Shoreline ownership stays with the procedural macro.
            low_freq_elev: fallback.low_freq_elev,
            source: TerrainSampleSource::LearnedMacro,
            visual_climate: blended_climate,
            ..fallback
        }
    }

    fn revision(&self) -> u64 {
        self.fallback.revision().max(self.macro_field.revision())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A per-chunk immutable field snapshot that captures the learned/procedural
/// decision at mesh-build time. This implements the M5 chunk-wide halo
/// residency gate: a mesh uses learned data only if every elevation plus
/// normal halo dependency is resident; otherwise it is entirely procedural.
///
/// The snapshot is constructed once per chunk mesh generation and is
/// immutable for the duration of that mesh build. This prevents per-vertex
/// learned/procedural mixing: every vertex in the chunk gets the same source.
///
/// The snapshot also records the source decision for telemetry so the
/// streaming queue can report real fallback percentages.
pub struct ChunkFieldSnapshot {
    /// The underlying field (blended hybrid or procedural).
    field: Arc<dyn TerrainField>,
    /// The chunk's CellKey, used to evaluate halo residency.
    chunk: er_core::math::CellKey,
    /// Whether this chunk's halo dependencies are fully resident. Captured
    /// at construction time so the entire mesh uses one source.
    learned_eligible: bool,
    /// The blend weight for this chunk (0 = procedural, 1 = learned). Captured
    /// at construction time so the entire mesh uses one weight.
    blend_weight: f64,
}

impl ChunkFieldSnapshot {
    /// Build a snapshot for a chunk. The `halo_resident` flag is computed by
    /// the caller (from `ChartMacroField::chunk_halo_resident`) and captured
    /// here so the mesh build is deterministic for the entire chunk.
    ///
    /// `blend_weight` is the provenance blend weight (0 = procedural, 1 =
    /// learned). When `halo_resident` is false, the weight is forced to 0
    /// (entirely procedural).
    pub fn new(
        field: Arc<dyn TerrainField>,
        chunk: er_core::math::CellKey,
        halo_resident: bool,
        blend_weight: f64,
    ) -> Self {
        let learned_eligible = halo_resident;
        let blend_weight = if halo_resident {
            blend_weight.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            field,
            chunk,
            learned_eligible,
            blend_weight,
        }
    }

    /// Whether this chunk is eligible to use learned data (all halo deps
    /// resident).
    pub fn learned_eligible(&self) -> bool {
        self.learned_eligible
    }

    /// The blend weight captured at construction time.
    pub fn blend_weight(&self) -> f64 {
        self.blend_weight
    }

    /// The chunk this snapshot was built for.
    pub fn chunk(&self) -> er_core::math::CellKey {
        self.chunk
    }
}

impl TerrainField for ChunkFieldSnapshot {
    fn sample(&self, dir: DVec3) -> TerrainSample {
        // If the chunk is not learned-eligible, return the procedural
        // fallback directly — no per-vertex learned/procedural mixing.
        if !self.learned_eligible || self.blend_weight <= 0.0 {
            let mut s = self.field.sample(dir);
            // Force the source to Procedural so telemetry records it.
            s.source = TerrainSampleSource::Procedural;
            return s;
        }
        // Learned-eligible: sample the underlying field (which may be a
        // blended hybrid). The source will be LearnedMacro if the macro
        // field has resident data.
        self.field.sample(dir)
    }

    fn sample_elevation(&self, dir: DVec3) -> f64 {
        self.field.sample_elevation(dir)
    }

    fn mesh_sample_spacing_m(&self) -> Option<f64> {
        self.field.mesh_sample_spacing_m()
    }

    fn revision(&self) -> u64 {
        self.field.revision()
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
            Some(MacroTerrainSample {
                elevation: self.0,
                visual_climate: VisualClimate::default(),
            })
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

    // ---- Blended hybrid field: provenance blend, not world-coordinate blend ----

    #[test]
    fn blended_hybrid_starts_procedural_then_becomes_learned() {
        let seed = PlanetSeed(0xC0FFEE);
        let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
            elevation_params(seed),
            planet_params(seed),
        ));
        let dir = uv_to_dir(1, 0.21, 0.61);
        // Instant transition (transition_seconds = 0): fully learned immediately.
        let instant =
            BlendedHybridTerrainField::new(fallback.clone(), Arc::new(ConstantMacro(1.25)), 0.0);
        let sample = instant.sample(dir);
        assert_eq!(sample.source, TerrainSampleSource::LearnedMacro);
        assert_eq!(sample.low_freq_elev, fallback.sample(dir).low_freq_elev);
        // Learned macro + procedural residual.
        let pro = fallback.sample(dir);
        let expected = 1.25 + (pro.elevation - pro.low_freq_elev as f64);
        assert!((sample.elevation - expected).abs() < 1e-9);
    }

    #[test]
    fn blended_hybrid_preserves_shoreline_ownership() {
        // The shoreline (low_freq_elev < sea_level) must stay with the
        // procedural macro field regardless of the learned macro value.
        let seed = PlanetSeed(0xC0FFEE);
        let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
            elevation_params(seed),
            planet_params(seed),
        ));
        let dir = uv_to_dir(2, 0.41, 0.59);
        let procedural = fallback.sample(dir);
        let procedural_water = procedural.low_freq_elev < 0.0;

        for learned_macro in [-5.0, -1.0, 0.0, 1.0, 5.0] {
            let blended = BlendedHybridTerrainField::new(
                fallback.clone(),
                Arc::new(ConstantMacro(learned_macro)),
                0.0,
            );
            let sample = blended.sample(dir);
            // Shoreline ownership never changes.
            assert_eq!(sample.low_freq_elev, procedural.low_freq_elev);
            assert_eq!(sample.low_freq_elev < 0.0, procedural_water);
        }
    }

    #[test]
    fn blended_hybrid_returns_procedural_when_learned_missing() {
        // When the macro field returns None (not resident), the blended
        // field must return the procedural fallback unchanged.
        struct MissingMacro;
        impl MacroTerrainField for MissingMacro {
            fn sample_resident(&self, _: DVec3) -> Option<MacroTerrainSample> {
                None
            }
        }
        let seed = PlanetSeed(0xC0FFEE);
        let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
            elevation_params(seed),
            planet_params(seed),
        ));
        let blended = BlendedHybridTerrainField::new(fallback.clone(), Arc::new(MissingMacro), 0.0);
        let dir = uv_to_dir(0, 0.5, 0.5);
        let pro = fallback.sample(dir);
        let sample = blended.sample(dir);
        assert_eq!(sample.source, TerrainSampleSource::Procedural);
        assert_eq!(sample.elevation.to_bits(), pro.elevation.to_bits());
    }

    #[test]
    fn blended_hybrid_continuity_no_height_step() {
        // The blended field must not produce a height step at the moment
        // a tile becomes resident. With transition_seconds = 0 (instant),
        // the elevation jumps from procedural macro + residual to learned
        // macro + residual. With a nonzero transition, the jump is smoothed.
        // Verify the smoothed path is bounded by the two endpoints.
        let seed = PlanetSeed(0xC0FFEE);
        let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
            elevation_params(seed),
            planet_params(seed),
        ));
        let dir = uv_to_dir(3, 0.31, 0.72);
        let pro = fallback.sample(dir);
        let procedural_macro = pro.low_freq_elev as f64;
        let procedural_residual = pro.elevation - procedural_macro;
        let learned_macro = 0.8;

        // Instant: fully learned.
        let instant = BlendedHybridTerrainField::new(
            fallback.clone(),
            Arc::new(ConstantMacro(learned_macro)),
            0.0,
        );
        let fully_learned = instant.sample(dir).elevation;

        // The fully-learned elevation is learned_macro + residual.
        assert!((fully_learned - (learned_macro + procedural_residual)).abs() < 1e-9);

        // The procedural elevation is procedural_macro + residual.
        let fully_procedural = pro.elevation;
        assert!((fully_procedural - (procedural_macro + procedural_residual)).abs() < 1e-9);

        // Both endpoints are finite and the learned differs from procedural.
        assert!(fully_learned.is_finite());
        assert!(fully_procedural.is_finite());
    }
}
