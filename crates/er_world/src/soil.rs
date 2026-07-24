//! M6.3 functional soil sampler.
//!
//! Deterministic CLORPT-style soil functional model:
//!   * steep, exposed slopes lose soil and expose rock/scree
//!   * stable, vegetated climates deepen soils and raise organic matter
//!   * warm, wet climates accelerate weathering (clay production, reduced rock)
//!   * arid basins accumulate salt
//!   * fresh volcanic material is fertile but barren; it weathers with age
//!   * cold, wet climates build peat and permafrost
//!
//! Integration: `crate::lib.rs` needs `pub mod geology;` and `pub mod soil;`, plus
//! a public re-export of the soil API (e.g. `SoilDrainageClass`,
//! `SoilFunctionalSample`, `sample_soil`).

use er_core::rng::rng_from_seed;
use glam::DVec3;
use rand::Rng;
use serde::Serialize;

use crate::climate::ClimateSample;
use crate::geology::{SubstrateClass, SubstrateSample};
use crate::landform::LandformSampleV2;

// =============================================================================
// Output types
// =============================================================================

/// Drainage class derived from slope, texture, wetness and valley position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub enum SoilDrainageClass {
    ExcessivelyDrained,
    #[default]
    WellDrained,
    ModeratelyWellDrained,
    PoorlyDrained,
    VeryPoorlyDrained,
    Saturated,
}

/// Validity bitmask for a functional soil sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SoilFunctionalValidity {
    pub finite: bool,
    pub texture_normalized: bool,
    pub exposure_normalized: bool,
    pub drainage_class_valid: bool,
    pub depth_positive: bool,
    pub climate_input: bool,
    pub landform_input: bool,
    pub substrate_input: bool,
}

impl SoilFunctionalValidity {
    pub const fn all_valid() -> Self {
        Self {
            finite: true,
            texture_normalized: true,
            exposure_normalized: true,
            drainage_class_valid: true,
            depth_positive: true,
            climate_input: true,
            landform_input: true,
            substrate_input: true,
        }
    }

    pub fn is_valid(self) -> bool {
        self.finite
            && self.texture_normalized
            && self.exposure_normalized
            && self.drainage_class_valid
            && self.depth_positive
            && self.climate_input
            && self.landform_input
            && self.substrate_input
    }
}

/// Functional soil sample.
///
/// Contains all fields of the legacy `SoilSample` plus the drainage class and
/// normalized surface-cover masks for rock, scree, sediment, sand, mud, peat and
/// salt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SoilFunctionalSample {
    // Legacy SoilSample fields
    pub depth_m: f32,
    pub sand_fraction: f32,
    pub silt_fraction: f32,
    pub clay_fraction: f32,
    pub water_capacity_mm: f32,
    pub fertility: f32,
    pub organic_fraction: f32,
    pub permafrost_fraction: f32,
    pub salinity_ds_m: f32,
    pub bare_substrate_fraction: f32,

    // New M6.3 fields
    pub drainage_class: SoilDrainageClass,
    pub rock_exposure: f32,
    pub scree_exposure: f32,
    pub sediment_exposure: f32,
    pub sand_exposure: f32,
    pub mud_exposure: f32,
    pub peat_exposure: f32,
    pub salt_exposure: f32,

    pub validity: SoilFunctionalValidity,
}

impl SoilFunctionalSample {
    pub fn is_valid(&self) -> bool {
        self.validity.is_valid()
            && sample_finite(self)
            && self.texture_normalized()
            && self.exposure_normalized()
            && self.depth_m >= 0.0
            && self.depth_m.is_finite()
    }

    fn texture_sum(&self) -> f32 {
        self.sand_fraction + self.silt_fraction + self.clay_fraction
    }

    fn texture_normalized(&self) -> bool {
        (self.texture_sum() - 1.0).abs() <= 1.0e-3
    }

    fn exposure_sum(&self) -> f32 {
        self.rock_exposure
            + self.scree_exposure
            + self.sediment_exposure
            + self.sand_exposure
            + self.mud_exposure
            + self.peat_exposure
            + self.salt_exposure
    }

    fn exposure_normalized(&self) -> bool {
        (self.exposure_sum() - 1.0).abs() <= 1.0e-3
    }
}

// =============================================================================
// Internal substrate model derived from crate::geology::SubstrateSample
// =============================================================================

/// Soil-relevant substrate parameters extracted from the geology sample.
///
/// The geology API describes classes, grain size and chemistry; this internal
/// model translates those into the fertility, weathering, volcanic and cover
/// fractions needed by the CLORPT soil sampler.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SubstrateInputs {
    parent_fertility: f32,
    weathering_susceptibility: f32,
    surface_age_kyr: f32,
    volcanic_fraction: f32,
    initial_rock_fraction: f32,
    initial_scree_fraction: f32,
    initial_sand_fraction: f32,
    initial_silt_fraction: f32,
    initial_clay_fraction: f32,
    initial_mud_fraction: f32,
    initial_peat_fraction: f32,
    initial_salt_fraction: f32,
    initial_bare_fraction: f32,
}

fn is_volcanic(class: SubstrateClass) -> bool {
    matches!(
        class,
        SubstrateClass::VolcanicBasalt
            | SubstrateClass::VolcanicAndesite
            | SubstrateClass::VolcanicRhyolite
    )
}

fn class_fertility(class: SubstrateClass) -> f32 {
    use SubstrateClass::*;
    match class {
        AlluvialDeposit => 0.85,
        ContinentalSediment => 0.65,
        OceanicSediment => 0.45,
        CarbonatePlatform => 0.55,
        Karst => 0.45,
        VolcanicBasalt => 0.75,
        VolcanicAndesite => 0.70,
        VolcanicRhyolite => 0.55,
        GranitePlutonic => 0.30,
        MetamorphicSchist => 0.35,
        Sandstone => 0.30,
        Shale => 0.60,
        GlacialTill => 0.35,
        Regolith => 0.50,
        Laterite => 0.20,
    }
}

/// Texture and cover fractions implied by a substrate class.
/// Returns `(rock, scree, sand, silt, clay)` in [0, 1].
fn class_texture(class: SubstrateClass) -> (f32, f32, f32, f32, f32) {
    use SubstrateClass::*;
    match class {
        VolcanicBasalt | VolcanicAndesite | VolcanicRhyolite => (0.45, 0.30, 0.10, 0.08, 0.07),
        GranitePlutonic | MetamorphicSchist => (0.55, 0.25, 0.10, 0.05, 0.05),
        GlacialTill => (0.25, 0.40, 0.15, 0.12, 0.08),
        Sandstone => (0.05, 0.05, 0.75, 0.10, 0.05),
        ContinentalSediment => (0.05, 0.05, 0.35, 0.35, 0.20),
        AlluvialDeposit => (0.05, 0.05, 0.40, 0.30, 0.20),
        OceanicSediment => (0.05, 0.05, 0.30, 0.35, 0.25),
        CarbonatePlatform | Karst => (0.05, 0.05, 0.25, 0.35, 0.30),
        Regolith => (0.10, 0.10, 0.35, 0.25, 0.20),
        Shale => (0.05, 0.05, 0.20, 0.35, 0.35),
        Laterite => (0.10, 0.10, 0.20, 0.20, 0.40),
    }
}

fn derive_substrate_inputs(substrate: &SubstrateSample) -> SubstrateInputs {
    let (rock, scree, sand, silt, clay) = class_texture(substrate.class);
    let volcanic = is_volcanic(substrate.class);
    let volcanic_fraction = if volcanic {
        (1.0 - substrate.volcanic_age_normalized).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let surface_age_kyr = if volcanic {
        substrate.volcanic_age_normalized * 1000.0
    } else {
        1000.0
    };

    SubstrateInputs {
        parent_fertility: class_fertility(substrate.class),
        weathering_susceptibility: substrate.erodibility.clamp(0.0, 1.0),
        surface_age_kyr,
        volcanic_fraction,
        initial_rock_fraction: rock,
        initial_scree_fraction: scree,
        initial_sand_fraction: sand,
        initial_silt_fraction: silt,
        initial_clay_fraction: clay,
        initial_mud_fraction: 0.0,
        initial_peat_fraction: 0.0,
        initial_salt_fraction: 0.0,
        initial_bare_fraction: 0.0,
    }
}

// =============================================================================
// Sampling function
// =============================================================================

/// Deterministic functional soil sample at a world direction.
///
/// All randomness is derived from `seed` and `dir`; the same inputs always
/// produce the same output. Climate, landform and substrate inputs are read
/// but never mutated.
pub fn sample_soil(
    seed: u64,
    dir: DVec3,
    climate: &ClimateSample,
    landform: &LandformSampleV2,
    substrate: &SubstrateSample,
) -> SoilFunctionalSample {
    let inputs = derive_substrate_inputs(substrate);

    let mut rng = world_rng(seed, dir);
    let local_var: f32 = 0.9 + 0.2 * rng.random::<f32>();

    let mut validity = SoilFunctionalValidity::all_valid();
    if !climate.validity.is_valid() {
        validity.climate_input = false;
    }
    if !landform_valid(landform) {
        validity.landform_input = false;
    }
    if !substrate.validity.is_valid() {
        validity.substrate_input = false;
    }

    // Climate factors
    let temp_c = climate.annual_mean_temperature_c();
    let precip = climate.annual_precipitation_mm();
    let aridity = climate.aridity_index().clamp(0.0, 10.0);
    let moisture = climate.moisture_index().clamp(-1.0, 1.0);
    let wetness = (moisture + 1.0) * 0.5;
    let snow = climate.snow_fraction();
    let gs = (climate.growing_season_days() as f32 / 365.0).clamp(0.0, 1.0);
    let gdd = climate.growing_degree_days_c();

    // Landform factors
    let slope = landform.slope_degrees.clamp(0.0, 90.0);
    let slope_norm = slope / 90.0;
    let exposure = ((landform.exposure + 1.0) * 0.5).clamp(0.0, 1.0);
    let valley = (-landform.valley_position).clamp(0.0, 1.0);
    let ruggedness = landform.ruggedness.clamp(0.0, 1.0);
    let relief = landform.local_relief_m.max(0.0);
    let topographic_wetness_index = (valley * 20.0 * (1.0 - slope_norm) + 1.0).clamp(0.0, 30.0);

    // Substrate factors
    let age = inputs.surface_age_kyr.max(0.0);
    let volcanic = inputs.volcanic_fraction.clamp(0.0, 1.0);
    let parent_fertility = inputs.parent_fertility.clamp(0.0, 1.0);
    let weather_suscept = inputs.weathering_susceptibility.clamp(0.0, 1.0);

    // -------------------------------------------------------------------------
    // Texture: start from parent material and normalize.
    // -------------------------------------------------------------------------
    let mut sand = inputs.initial_sand_fraction.max(0.0);
    let mut silt = inputs.initial_silt_fraction.max(0.0);
    let mut clay = inputs.initial_clay_fraction.max(0.0);
    normalize3(&mut sand, &mut silt, &mut clay);

    // Warm, wet chemical weathering: sand -> silt -> clay.
    let precip_factor = (precip / 2500.0).clamp(0.0, 1.0);
    let age_factor = age / (age + 10.0);
    let temp_factor = ((temp_c + 10.0) / 40.0).clamp(0.0, 1.0);
    let weathering =
        (weather_suscept * temp_factor * wetness * precip_factor * age_factor).clamp(0.0, 1.0);
    let sand_to_silt = sand * weathering * 0.4;
    let silt_to_clay = silt * weathering * 0.3;
    sand -= sand_to_silt;
    silt += sand_to_silt - silt_to_clay;
    clay += silt_to_clay;
    normalize3(&mut sand, &mut silt, &mut clay);

    // -------------------------------------------------------------------------
    // Depth and organic matter
    // -------------------------------------------------------------------------
    let base_depth = 2.0 * local_var;
    let vegetation_productivity =
        (gs * wetness * (1.0 - aridity / 5.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let stability = ((1.0 - ruggedness) * (1.0 - slope_norm)).clamp(0.0, 1.0);

    let mut depth = base_depth
        * (1.0 + 2.0 * vegetation_productivity * stability)
        * (1.0 - slope_norm * (0.5 + 0.5 * exposure))
        * (1.0 - aridity * 0.3);
    depth = depth.clamp(0.0, 10.0);

    let mut organic =
        0.02 * vegetation_productivity * stability * local_var * (1.0 + (gdd / 3000.0).min(1.0));

    // Cold, wet peat / permafrost.
    let cold_wet = ((5.0 - temp_c) / 25.0).clamp(0.0, 1.0) * wetness;
    organic += cold_wet * 0.4 * (1.0 - slope_norm);

    // Volcanic young barren: very fresh deposits suppress organics and depth.
    let young_volcanic = volcanic * (1.0 - (age / 2.0).clamp(0.0, 1.0));
    organic *= 1.0 - young_volcanic * 0.8;
    depth *= 1.0 - young_volcanic * 0.7;
    organic = organic.clamp(0.0, 1.0);
    depth = depth.clamp(0.0, 10.0);

    let permafrost = if temp_c < 0.0 {
        ((-temp_c) / 20.0).clamp(0.0, 1.0) * wetness * (1.0 - slope_norm) * (1.0 + snow * 0.5)
    } else {
        0.0
    }
    .clamp(0.0, 1.0);

    // -------------------------------------------------------------------------
    // Salinity: arid basins accumulate salt; coasts add a little in dry climates.
    // -------------------------------------------------------------------------
    let basin_factor = valley * (1.0 - slope_norm) * (1.0 - (relief / 500.0).min(1.0));
    let coastal = 1.0 - (landform.coastal_proximity_m / 50_000.0).min(1.0);
    let mut salinity = inputs.initial_salt_fraction * 4.0
        + aridity * basin_factor * 12.0
        + aridity * coastal * 2.0;
    salinity = salinity.clamp(0.0, 60.0);

    // -------------------------------------------------------------------------
    // Fertility: parent + weathering + volcanic ash + organic; penalties.
    // -------------------------------------------------------------------------
    let weathering_fertility = weathering * 0.5;
    let volcanic_fertility = volcanic * (1.0 - (age / 10.0).clamp(0.0, 1.0)) * 0.6;
    let mut fertility = (parent_fertility + weathering_fertility + volcanic_fertility)
        * (1.0 + organic * 0.5)
        * (1.0 - salinity / 60.0)
        * (1.0 - permafrost * 0.6);
    fertility = fertility.clamp(0.0, 1.0);

    // -------------------------------------------------------------------------
    // Bare substrate: steep/exposed slopes and very young volcanic surfaces.
    // -------------------------------------------------------------------------
    let mut bare =
        slope_norm * exposure * 0.8 + young_volcanic * 0.9 + inputs.initial_bare_fraction;
    bare = bare.clamp(0.0, 1.0);

    // -------------------------------------------------------------------------
    // Surface-cover masks (rock, scree, sediment, sand, mud, peat, salt).
    // -------------------------------------------------------------------------
    let mut rock =
        inputs.initial_rock_fraction * (1.0 - weathering * 0.5) * (1.0 + slope_norm * exposure);
    let mut scree = inputs.initial_scree_fraction * (1.0 + slope_norm * 2.0);
    let mut sand_exp = inputs.initial_sand_fraction * (1.0 - weathering * 0.3);
    let mut mud_exp =
        (inputs.initial_silt_fraction + inputs.initial_clay_fraction) * (1.0 + weathering * 0.5);
    let mut peat = organic * wetness * (1.0 - permafrost * 0.3);
    let mut salt_exp = (salinity / 60.0).clamp(0.0, 1.0);

    rock = rock.clamp(0.0, 1.0);
    scree = scree.clamp(0.0, 1.0);
    sand_exp = sand_exp.clamp(0.0, 1.0);
    mud_exp = mud_exp.clamp(0.0, 1.0);
    peat = peat.clamp(0.0, 1.0);
    salt_exp = salt_exp.clamp(0.0, 1.0);

    let mut sediment = (1.0 - rock - scree - sand_exp - mud_exp - peat - salt_exp).max(0.0);
    let cover_sum = rock + scree + sediment + sand_exp + mud_exp + peat + salt_exp;
    if cover_sum > 0.0 {
        let inv = 1.0 / cover_sum;
        rock *= inv;
        scree *= inv;
        sediment *= inv;
        sand_exp *= inv;
        mud_exp *= inv;
        peat *= inv;
        salt_exp *= inv;
    } else {
        sediment = 1.0;
    }

    // -------------------------------------------------------------------------
    // Drainage class
    // -------------------------------------------------------------------------
    let drainage = classify_drainage(slope, clay, topographic_wetness_index, aridity, valley);

    // -------------------------------------------------------------------------
    // Water capacity
    // -------------------------------------------------------------------------
    let water_capacity = (depth
        * 1000.0
        * (clay * 0.22 + silt * 0.18 + sand * 0.07 + organic * 0.50)
        * (1.0 - permafrost * 0.8)
        * (1.0 - rock * 0.5))
        .clamp(0.0, 2000.0);

    let mut sample = SoilFunctionalSample {
        depth_m: depth,
        sand_fraction: sand,
        silt_fraction: silt,
        clay_fraction: clay,
        water_capacity_mm: water_capacity,
        fertility,
        organic_fraction: organic,
        permafrost_fraction: permafrost,
        salinity_ds_m: salinity,
        bare_substrate_fraction: bare,
        drainage_class: drainage,
        rock_exposure: rock,
        scree_exposure: scree,
        sediment_exposure: sediment,
        sand_exposure: sand_exp,
        mud_exposure: mud_exp,
        peat_exposure: peat,
        salt_exposure: salt_exp,
        validity,
    };

    // Validate outputs.
    sample.validity.finite = sample_finite(&sample);
    sample.validity.texture_normalized = sample.texture_normalized();
    sample.validity.exposure_normalized = sample.exposure_normalized();
    sample.validity.drainage_class_valid = true; // classify_drainage always returns a class
    sample.validity.depth_positive = sample.depth_m >= 0.0 && sample.depth_m.is_finite();

    sample
}

// =============================================================================
// Helpers
// =============================================================================

fn landform_valid(landform: &LandformSampleV2) -> bool {
    landform.slope_degrees.is_finite()
        && (0.0..=90.0).contains(&landform.slope_degrees)
        && landform.exposure.is_finite()
        && (-1.0..=1.0).contains(&landform.exposure)
        && landform.valley_position.is_finite()
        && (-1.0..=1.0).contains(&landform.valley_position)
        && landform.ruggedness.is_finite()
        && (0.0..=1.0).contains(&landform.ruggedness)
        && landform.confidence.is_finite()
        && (0.0..=1.0).contains(&landform.confidence)
}

fn world_rng(seed: u64, dir: DVec3) -> impl Rng {
    let d = dir.normalize_or_zero();
    let x = d.x.to_bits();
    let y = d.y.to_bits();
    let z = d.z.to_bits();
    let mixed = seed
        .wrapping_add(x)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(y)
        .wrapping_mul(0xC6BC_2796_92B5_CC83)
        .wrapping_add(z)
        .wrapping_mul(0x85EB_CA6B_F0D5_3E2D);
    rng_from_seed(mixed)
}

fn normalize3(a: &mut f32, b: &mut f32, c: &mut f32) {
    let sum = *a + *b + *c;
    if sum > 0.0 {
        let inv = 1.0 / sum;
        *a *= inv;
        *b *= inv;
        *c *= inv;
    } else {
        *a = 0.4;
        *b = 0.4;
        *c = 0.2;
    }
}

pub fn classify_drainage(
    slope_degrees: f32,
    clay_fraction: f32,
    topographic_wetness_index: f32,
    aridity_index: f32,
    valley_position: f32,
) -> SoilDrainageClass {
    let slope_norm = (slope_degrees / 90.0).clamp(0.0, 1.0);
    let clay = clay_fraction.clamp(0.0, 1.0);
    let twi_norm = (topographic_wetness_index / 30.0).clamp(0.0, 1.0);
    let aridity_norm = (aridity_index / 10.0).clamp(0.0, 1.0);
    let valley = valley_position.clamp(0.0, 1.0);

    // Higher score = better drainage.
    let score =
        slope_norm * 0.45 + (1.0 - clay) * 0.20 + aridity_norm * 0.20 + (1.0 - twi_norm) * 0.10
            - valley * 0.05;
    let score = score.clamp(0.0, 1.0);

    match score {
        s if s >= 0.85 => SoilDrainageClass::ExcessivelyDrained,
        s if s >= 0.60 => SoilDrainageClass::WellDrained,
        s if s >= 0.40 => SoilDrainageClass::ModeratelyWellDrained,
        s if s >= 0.20 => SoilDrainageClass::PoorlyDrained,
        s if s >= 0.08 => SoilDrainageClass::VeryPoorlyDrained,
        _ => SoilDrainageClass::Saturated,
    }
}

fn sample_finite(sample: &SoilFunctionalSample) -> bool {
    sample.depth_m.is_finite()
        && sample.sand_fraction.is_finite()
        && sample.silt_fraction.is_finite()
        && sample.clay_fraction.is_finite()
        && sample.water_capacity_mm.is_finite()
        && sample.fertility.is_finite()
        && sample.organic_fraction.is_finite()
        && sample.permafrost_fraction.is_finite()
        && sample.salinity_ds_m.is_finite()
        && sample.bare_substrate_fraction.is_finite()
        && sample.rock_exposure.is_finite()
        && sample.scree_exposure.is_finite()
        && sample.sediment_exposure.is_finite()
        && sample.sand_exposure.is_finite()
        && sample.mud_exposure.is_finite()
        && sample.peat_exposure.is_finite()
        && sample.salt_exposure.is_finite()
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::climate::ClimateSampleValidity;
    use crate::environment::ClimateNormals;
    use crate::geology::{SubstrateClass, SubstrateSampleValidity};
    use crate::landform::LandformClass;

    const TEST_SEED: u64 = 0x6B3D_5041_2026_0700;

    fn test_climate() -> ClimateSample {
        ClimateSample {
            normals: ClimateNormals {
                annual_mean_temperature_c: 12.0,
                warmest_month_temperature_c: 22.0,
                coldest_month_temperature_c: 2.0,
                growing_season_days: 180,
                growing_degree_days_c: 1500.0,
                annual_precipitation_mm: 800.0,
                driest_month_precipitation_mm: 40.0,
                wettest_month_precipitation_mm: 120.0,
                precipitation_seasonality: 0.5,
                snow_fraction: 0.1,
                pet_mm: 600.0,
                water_balance_mm: 200.0,
                aridity_index: 0.75,
                moisture_index: 0.2,
                monthly_temperature_c: [
                    2.0, 4.0, 7.0, 10.0, 14.0, 18.0, 22.0, 21.0, 17.0, 12.0, 7.0, 3.0,
                ],
                monthly_precipitation_mm: [
                    60.0, 55.0, 70.0, 80.0, 90.0, 100.0, 80.0, 70.0, 75.0, 85.0, 80.0, 65.0,
                ],
            },
            prevailing_wind_speed_m_s: 5.0,
            prevailing_wind_direction_radians: 0.0,
            continentality_c: 10.0,
            monthly_temperature_c: [
                2.0, 4.0, 7.0, 10.0, 14.0, 18.0, 22.0, 21.0, 17.0, 12.0, 7.0, 3.0,
            ],
            monthly_precipitation_mm: [
                60.0, 55.0, 70.0, 80.0, 90.0, 100.0, 80.0, 70.0, 75.0, 85.0, 80.0, 65.0,
            ],
            validity: ClimateSampleValidity::all_valid(),
        }
    }

    fn test_landform() -> LandformSampleV2 {
        LandformSampleV2 {
            elevation_above_sea_m: 200.0,
            slope_degrees: 5.0,
            aspect_radians: 0.0,
            profile_curvature_per_m: 0.0,
            plan_curvature_per_m: 0.0,
            local_relief_m: 50.0,
            ruggedness: 0.2,
            exposure: 0.3,
            valley_position: 0.3,
            plateau_likelihood: 0.1,
            coastal_proximity_m: 100_000.0,
            primary_class: LandformClass::RollingHill,
            confidence: 1.0,
        }
    }

    fn test_substrate(class: SubstrateClass) -> SubstrateSample {
        SubstrateSample {
            province_id: 0,
            class,
            grain_size_mm: 0.5,
            permeability: 0.4,
            erodibility: 0.5,
            carbonate_fraction: 0.1,
            silicate_fraction: 0.5,
            mineral_color_rgb: [0.5, 0.5, 0.5],
            volcanic_age_normalized: 0.5,
            validity: SubstrateSampleValidity::all_valid(),
        }
    }

    fn default_substrate() -> SubstrateSample {
        test_substrate(SubstrateClass::ContinentalSediment)
    }

    fn test_dir() -> DVec3 {
        DVec3::new(0.5, 0.5, 0.5).normalize()
    }

    fn sample() -> SoilFunctionalSample {
        sample_soil(
            TEST_SEED,
            test_dir(),
            &test_climate(),
            &test_landform(),
            &default_substrate(),
        )
    }

    // -------------------------------------------------------------------------
    // Causality
    // -------------------------------------------------------------------------

    #[test]
    fn steep_exposed_slopes_lose_soil() {
        let base = sample();
        let mut landform = test_landform();
        landform.slope_degrees = 60.0;
        landform.exposure = 0.9;
        landform.primary_class = LandformClass::MountainSlope;
        let steep = sample_soil(
            TEST_SEED,
            test_dir(),
            &test_climate(),
            &landform,
            &default_substrate(),
        );

        assert!(
            steep.depth_m < base.depth_m * 0.5,
            "steep slope should shallow soil"
        );
        assert!(
            steep.bare_substrate_fraction > base.bare_substrate_fraction,
            "steep slope should increase bare cover"
        );
        assert!(
            steep.rock_exposure + steep.scree_exposure > base.rock_exposure + base.scree_exposure,
            "steep slope should expose rock/scree"
        );
    }

    #[test]
    fn stable_vegetated_climates_deepen_soil() {
        let mut climate = test_climate();
        climate.normals.growing_season_days = 365;
        climate.normals.growing_degree_days_c = 3500.0;
        climate.normals.moisture_index = 0.8;
        climate.normals.aridity_index = 0.2;
        climate.normals.annual_precipitation_mm = 1600.0;

        let mut landform = test_landform();
        landform.slope_degrees = 1.0;
        landform.ruggedness = 0.0;
        landform.exposure = 0.0;
        landform.primary_class = LandformClass::Plain;

        let stable = sample_soil(
            TEST_SEED,
            test_dir(),
            &climate,
            &landform,
            &default_substrate(),
        );
        let base = sample();

        assert!(
            stable.depth_m > base.depth_m,
            "stable vegetated climate should deepen soil"
        );
        assert!(
            stable.organic_fraction > base.organic_fraction,
            "stable vegetated climate should raise organic matter"
        );
    }

    #[test]
    fn warm_wet_weathering_increases_clay() {
        let mut substrate = default_substrate();
        substrate.erodibility = 0.9;
        substrate.class = SubstrateClass::GranitePlutonic;

        let mut cold_dry = test_climate();
        cold_dry.normals.annual_mean_temperature_c = -5.0;
        cold_dry.normals.annual_precipitation_mm = 100.0;
        cold_dry.normals.moisture_index = -0.8;
        cold_dry.normals.aridity_index = 8.0;

        let mut warm_wet = test_climate();
        warm_wet.normals.annual_mean_temperature_c = 25.0;
        warm_wet.normals.annual_precipitation_mm = 2500.0;
        warm_wet.normals.moisture_index = 0.8;
        warm_wet.normals.aridity_index = 0.2;

        // Use a ridge landform so basin salinity does not dominate the normalized
        // cover fractions and the comparison is driven by texture weathering.
        let mut ridge = test_landform();
        ridge.slope_degrees = 10.0;
        ridge.valley_position = 0.0;
        ridge.primary_class = LandformClass::RollingHill;

        let cold = sample_soil(TEST_SEED, test_dir(), &cold_dry, &ridge, &substrate);
        let warm = sample_soil(TEST_SEED, test_dir(), &warm_wet, &ridge, &substrate);

        assert!(
            warm.clay_fraction > cold.clay_fraction,
            "warm wet weathering should raise clay"
        );
        assert!(
            warm.rock_exposure + warm.scree_exposure < cold.rock_exposure + cold.scree_exposure,
            "warm wet weathering should reduce rock/scree cover"
        );
    }

    #[test]
    fn arid_basins_accumulate_salt() {
        let mut climate = test_climate();
        climate.normals.aridity_index = 8.0;
        climate.normals.moisture_index = -0.8;
        climate.normals.annual_precipitation_mm = 50.0;

        let mut landform = test_landform();
        landform.slope_degrees = 1.0;
        landform.valley_position = -1.0;
        landform.local_relief_m = 10.0;
        landform.coastal_proximity_m = 200_000.0;
        landform.primary_class = LandformClass::Valley;

        let arid = sample_soil(
            TEST_SEED,
            test_dir(),
            &climate,
            &landform,
            &default_substrate(),
        );
        let base = sample();

        assert!(
            arid.salinity_ds_m > base.salinity_ds_m + 1.0,
            "arid basin should raise salinity"
        );
        assert!(
            arid.salt_exposure > base.salt_exposure,
            "arid basin should raise salt cover"
        );
    }

    #[test]
    fn volcanic_fertility_and_young_barren() {
        let mut substrate = default_substrate();
        substrate.class = SubstrateClass::VolcanicBasalt;
        substrate.volcanic_age_normalized = 0.0; // fresh
        substrate.erodibility = 0.4;

        let young = sample_soil(
            TEST_SEED,
            test_dir(),
            &test_climate(),
            &test_landform(),
            &substrate,
        );
        let base = sample();

        assert!(
            young.fertility > base.fertility,
            "fresh volcanic material should be fertile"
        );
        assert!(
            young.bare_substrate_fraction > base.bare_substrate_fraction + 0.2,
            "very young volcanic surface should be barren"
        );
        assert!(
            young.organic_fraction < base.organic_fraction,
            "very young volcanic surface should suppress organics"
        );
    }

    #[test]
    fn cold_wet_climates_build_peat_and_permafrost() {
        let mut climate = test_climate();
        climate.normals.annual_mean_temperature_c = -5.0;
        climate.normals.moisture_index = 0.8;
        climate.normals.snow_fraction = 0.8;
        climate.normals.annual_precipitation_mm = 1000.0;

        let mut landform = test_landform();
        landform.slope_degrees = 1.0;
        landform.valley_position = 0.8;
        landform.primary_class = LandformClass::Valley;

        let cold = sample_soil(
            TEST_SEED,
            test_dir(),
            &climate,
            &landform,
            &default_substrate(),
        );
        let base = sample();

        assert!(
            cold.peat_exposure > base.peat_exposure,
            "cold wet climate should build peat cover"
        );
        assert!(
            cold.permafrost_fraction > 0.1,
            "cold wet climate should produce permafrost"
        );
    }

    // -------------------------------------------------------------------------
    // Bounds and determinism
    // -------------------------------------------------------------------------

    #[test]
    fn outputs_are_within_bounds() {
        let climates = [
            (12.0, 0.75, 0.2, 800.0),
            (-10.0, 0.2, 0.8, 1200.0),
            (30.0, 8.0, -0.8, 20.0),
            (5.0, 1.5, 0.0, 400.0),
        ];
        let slopes = [0.0f32, 15.0, 45.0, 75.0, 90.0];
        let classes = [
            SubstrateClass::ContinentalSediment,
            SubstrateClass::VolcanicBasalt,
            SubstrateClass::Sandstone,
            SubstrateClass::GranitePlutonic,
        ];

        for &(temp, aridity, moisture, precip) in &climates {
            for &slope in &slopes {
                for &class in &classes {
                    let mut climate = test_climate();
                    climate.normals.annual_mean_temperature_c = temp;
                    climate.normals.aridity_index = aridity;
                    climate.normals.moisture_index = moisture;
                    climate.normals.annual_precipitation_mm = precip;

                    let mut landform = test_landform();
                    landform.slope_degrees = slope;

                    let s = sample_soil(
                        TEST_SEED,
                        test_dir(),
                        &climate,
                        &landform,
                        &test_substrate(class),
                    );
                    assert!(
                        s.is_valid(),
                        "sample should be valid for temp={temp} slope={slope} class={class:?}"
                    );
                    assert!((0.0..=10.0).contains(&s.depth_m));
                    assert!((0.0..=1.0).contains(&s.fertility));
                    assert!((0.0..=1.0).contains(&s.organic_fraction));
                    assert!((0.0..=1.0).contains(&s.permafrost_fraction));
                    assert!((0.0..=1.0).contains(&s.bare_substrate_fraction));
                    assert!((0.0..=60.0).contains(&s.salinity_ds_m));
                    assert!((0.0..=2000.0).contains(&s.water_capacity_mm));
                }
            }
        }
    }

    #[test]
    fn texture_fractions_normalize() {
        let s = sample();
        let sum = s.sand_fraction + s.silt_fraction + s.clay_fraction;
        assert!(
            (sum - 1.0).abs() < 1.0e-3,
            "texture fractions must sum to 1, got {sum}"
        );
    }

    #[test]
    fn exposure_masks_normalize() {
        let s = sample();
        let sum = s.rock_exposure
            + s.scree_exposure
            + s.sediment_exposure
            + s.sand_exposure
            + s.mud_exposure
            + s.peat_exposure
            + s.salt_exposure;
        assert!(
            (sum - 1.0).abs() < 1.0e-3,
            "exposure masks must sum to 1, got {sum}"
        );
    }

    #[test]
    fn sampling_is_deterministic() {
        let a = sample_soil(
            TEST_SEED,
            test_dir(),
            &test_climate(),
            &test_landform(),
            &default_substrate(),
        );
        let b = sample_soil(
            TEST_SEED,
            test_dir(),
            &test_climate(),
            &test_landform(),
            &default_substrate(),
        );
        assert_eq!(a, b, "same inputs must produce identical output");
    }

    #[test]
    fn world_space_variation_only() {
        // Changing only the seed (not the direction) must change the local variation.
        let a = sample_soil(
            0xAAA,
            test_dir(),
            &test_climate(),
            &test_landform(),
            &default_substrate(),
        );
        let b = sample_soil(
            0xBBB,
            test_dir(),
            &test_climate(),
            &test_landform(),
            &default_substrate(),
        );
        assert_ne!(a, b, "different seeds must produce different variation");
    }

    // -------------------------------------------------------------------------
    // Impossible / edge combinations
    // -------------------------------------------------------------------------

    #[test]
    fn hot_climates_have_no_permafrost() {
        let mut climate = test_climate();
        climate.normals.annual_mean_temperature_c = 30.0;
        climate.normals.moisture_index = 0.9;
        climate.normals.snow_fraction = 0.0;
        let s = sample_soil(
            TEST_SEED,
            test_dir(),
            &climate,
            &test_landform(),
            &default_substrate(),
        );
        assert!(
            s.permafrost_fraction < 1.0e-3,
            "hot climate cannot have permafrost"
        );
    }

    #[test]
    fn arid_mountain_ridges_are_not_saline() {
        let mut climate = test_climate();
        climate.normals.aridity_index = 8.0;
        climate.normals.moisture_index = -0.8;
        let mut landform = test_landform();
        landform.slope_degrees = 80.0;
        landform.valley_position = 0.0;
        landform.local_relief_m = 2000.0;
        landform.coastal_proximity_m = 200_000.0;
        landform.primary_class = LandformClass::MountainSlope;
        let s = sample_soil(
            TEST_SEED,
            test_dir(),
            &climate,
            &landform,
            &default_substrate(),
        );
        assert!(
            s.salinity_ds_m < 2.0,
            "arid ridge should not accumulate salt"
        );
        assert!(
            s.salt_exposure < 0.1,
            "arid ridge should have little salt cover"
        );
    }

    #[test]
    fn hot_deserts_have_no_peat() {
        let mut climate = test_climate();
        climate.normals.annual_mean_temperature_c = 35.0;
        climate.normals.moisture_index = -0.9;
        climate.normals.aridity_index = 9.0;
        let s = sample_soil(
            TEST_SEED,
            test_dir(),
            &climate,
            &test_landform(),
            &default_substrate(),
        );
        assert!(s.peat_exposure < 1.0e-3, "hot desert cannot have peat");
    }

    #[test]
    fn vertical_cliffs_have_negligible_soil() {
        let mut landform = test_landform();
        landform.slope_degrees = 90.0;
        landform.exposure = 1.0;
        landform.primary_class = LandformClass::Cliff;
        let s = sample_soil(
            TEST_SEED,
            test_dir(),
            &test_climate(),
            &landform,
            &default_substrate(),
        );
        assert!(
            s.depth_m < 0.15,
            "vertical cliff should have almost no soil"
        );
    }

    #[test]
    fn invalid_inputs_mark_sample_invalid() {
        let mut climate = test_climate();
        climate.validity.finite = false; // invalidate input
        let s = sample_soil(
            TEST_SEED,
            test_dir(),
            &climate,
            &test_landform(),
            &default_substrate(),
        );
        assert!(
            !s.is_valid(),
            "invalid climate input should make output invalid"
        );
    }

    #[test]
    fn manually_unnormalized_sample_is_invalid() {
        let mut s = sample();
        s.sand_fraction = 0.5;
        s.silt_fraction = 0.2;
        s.clay_fraction = 0.2;
        s.validity = SoilFunctionalValidity::all_valid();
        assert!(!s.is_valid(), "texture sum != 1 should fail validation");
    }
}
