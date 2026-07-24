//! Versioned contracts and baseline evidence for the generated environment.
//!
//! Milestone 6 generates these fields once and exposes an immutable snapshot.
//! The current M6.1 implementation defines identities, units, authority, and
//! bounded data shapes without running future climate or hydrology models.

use std::f64::consts::PI;

use er_core::seed::PlanetSeed;
use glam::DVec3;
use serde::Serialize;

use crate::biome::Biome;
use crate::elevation::elevation_params;
use crate::params::{planet_params, PlanetParams};
use crate::terrain_field::{ProceduralTerrainField, TerrainField};

pub const ENVIRONMENT_MODEL_VERSION: u32 = 2;
pub const MAX_BIOME_MEMBERSHIPS: usize = 4;
pub const DEFAULT_BASELINE_SAMPLE_COUNT: usize = 4096;
pub const LEGACY_BASELINE_SEED: u64 = 0xC0FFEE;

const LEGACY_BASELINE_MODEL_VERSION: EnvironmentModelVersion = EnvironmentModelVersion {
    model: 1,
    climate_algorithm: 1,
    landform_algorithm: 1,
    soil_algorithm: 1,
    hydrology_algorithm: 1,
    disturbance_algorithm: 1,
    biome_catalog: 1,
    field_resolution: 1,
    constants: 1,
};

/// Audited legacy limitations that M6.2-M6.7 replace in order.
pub const LEGACY_ENVIRONMENT_PIPELINE_GAPS: [&str; 7] = [
    "normalized latitude/noise climate without physical units",
    "hard Whittaker-like biome thresholds",
    "climate and landform conflated in one biome enum",
    "fragment shader independently reclassifies environmental fields",
    "biome registry data is not consumed by the live renderer",
    "learned visual climate is dropped before rendering",
    "drainage is heuristic rather than watershed-derived",
];

/// Scientific roles locked for the M6 climate implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct HybridClimateModelContract {
    pub koppen_geiger_role: &'static str,
    pub holdridge_role: &'static str,
    pub moisture_role: &'static str,
    pub realized_habitat_role: &'static str,
    pub pet_method: &'static str,
    pub learned_climate_authority: FieldAuthority,
}

pub const HYBRID_CLIMATE_MODEL: HybridClimateModelContract = HybridClimateModelContract {
    koppen_geiger_role: "diagnostic_and_regression_label",
    holdridge_role: "continuous_biotemperature_and_pet_ratio_envelope",
    moisture_role: "feddema_moisture_index_and_unep_aridity_index",
    realized_habitat_role: "climate_plus_landform_soil_hydrology_disturbance",
    pet_method: "thornthwaite_temperature_based_static_normal",
    learned_climate_authority: FieldAuthority::VisualOnly,
};

/// Cache identity for canonical M6 environmental atlases and golden evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct EnvironmentModelVersion {
    pub model: u32,
    pub climate_algorithm: u16,
    pub landform_algorithm: u16,
    pub soil_algorithm: u16,
    pub hydrology_algorithm: u16,
    pub disturbance_algorithm: u16,
    pub biome_catalog: u16,
    pub field_resolution: u16,
    pub constants: u16,
}

impl EnvironmentModelVersion {
    pub const fn current() -> Self {
        Self {
            model: ENVIRONMENT_MODEL_VERSION,
            climate_algorithm: 2,
            landform_algorithm: 1,
            soil_algorithm: 1,
            hydrology_algorithm: 1,
            disturbance_algorithm: 1,
            biome_catalog: 1,
            field_resolution: 2,
            constants: 2,
        }
    }

    pub const fn is_cache_compatible_with(self, other: Self) -> bool {
        self.model == other.model
            && self.climate_algorithm == other.climate_algorithm
            && self.landform_algorithm == other.landform_algorithm
            && self.soil_algorithm == other.soil_algorithm
            && self.hydrology_algorithm == other.hydrology_algorithm
            && self.disturbance_algorithm == other.disturbance_algorithm
            && self.biome_catalog == other.biome_catalog
            && self.field_resolution == other.field_resolution
            && self.constants == other.constants
    }
}

/// Environmental stress presets are independent of geometry/LOD presets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentPreset {
    #[default]
    EarthLike,
    HotArid,
    ColdIcehouse,
}

impl EnvironmentPreset {
    pub const ALL: [Self; 3] = [Self::EarthLike, Self::HotArid, Self::ColdIcehouse];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EarthLike => "earth-like",
            Self::HotArid => "hot-arid",
            Self::ColdIcehouse => "cold-icehouse",
        }
    }

    /// Resolves canonical names and CLI-friendly compact aliases.
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "earth-like" | "earthlike" => Some(Self::EarthLike),
            "hot-arid" | "hotarid" => Some(Self::HotArid),
            "cold-icehouse" | "coldicehouse" => Some(Self::ColdIcehouse),
            _ => None,
        }
    }

    /// Apply only climate stress parameters. EarthLike is an identity map.
    pub fn apply(self, mut params: PlanetParams) -> PlanetParams {
        match self {
            Self::EarthLike => params,
            Self::HotArid => {
                params.axial_tilt = 0.40;
                params.sea_level = -0.08;
                params.lapse_rate = 0.15;
                params.temp_gradient = 0.72;
                params.temp_noise_amp = 0.40;
                params.moisture_noise_amp = 0.05;
                params.rain_shadow_strength = 0.70;
                params
            }
            Self::ColdIcehouse => {
                params.axial_tilt = 0.61;
                params.sea_level = 0.08;
                params.lapse_rate = 0.05;
                params.temp_gradient = 1.0;
                params.temp_noise_amp = 0.05;
                params.moisture_noise_amp = 0.20;
                params.rain_shadow_strength = 0.30;
                params
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct EnvironmentPresetExpectations {
    pub preset: EnvironmentPreset,
    pub land_fraction_min: f32,
    pub land_fraction_max: f32,
    pub annual_temperature_c_min: f32,
    pub annual_temperature_c_max: f32,
    pub annual_precipitation_mm_min: f32,
    pub annual_precipitation_mm_max: f32,
    pub arid_land_fraction_min: f32,
    pub arid_land_fraction_max: f32,
    pub polar_land_fraction_min: f32,
    pub polar_land_fraction_max: f32,
    pub river_density_km_per_million_km2_min: f32,
    pub river_density_km_per_million_km2_max: f32,
    pub max_atlas_build_seconds: f32,
    pub max_snapshot_memory_mib: f32,
}

static PRESET_EXPECTATIONS: [EnvironmentPresetExpectations; 3] = [
    EnvironmentPresetExpectations {
        preset: EnvironmentPreset::EarthLike,
        land_fraction_min: 0.25,
        land_fraction_max: 0.35,
        annual_temperature_c_min: 5.0,
        annual_temperature_c_max: 15.0,
        annual_precipitation_mm_min: 800.0,
        annual_precipitation_mm_max: 1200.0,
        arid_land_fraction_min: 0.18,
        arid_land_fraction_max: 0.35,
        polar_land_fraction_min: 0.05,
        polar_land_fraction_max: 0.15,
        river_density_km_per_million_km2_min: 3000.0,
        river_density_km_per_million_km2_max: 10000.0,
        max_atlas_build_seconds: 30.0,
        max_snapshot_memory_mib: 500.0,
    },
    EnvironmentPresetExpectations {
        preset: EnvironmentPreset::HotArid,
        land_fraction_min: 0.40,
        land_fraction_max: 0.70,
        annual_temperature_c_min: 15.0,
        annual_temperature_c_max: 30.0,
        annual_precipitation_mm_min: 200.0,
        annual_precipitation_mm_max: 600.0,
        arid_land_fraction_min: 0.50,
        arid_land_fraction_max: 0.80,
        polar_land_fraction_min: 0.0,
        polar_land_fraction_max: 0.05,
        river_density_km_per_million_km2_min: 200.0,
        river_density_km_per_million_km2_max: 4000.0,
        max_atlas_build_seconds: 30.0,
        max_snapshot_memory_mib: 500.0,
    },
    EnvironmentPresetExpectations {
        preset: EnvironmentPreset::ColdIcehouse,
        land_fraction_min: 0.25,
        land_fraction_max: 0.60,
        annual_temperature_c_min: -15.0,
        annual_temperature_c_max: 5.0,
        annual_precipitation_mm_min: 300.0,
        annual_precipitation_mm_max: 800.0,
        arid_land_fraction_min: 0.05,
        arid_land_fraction_max: 0.20,
        polar_land_fraction_min: 0.30,
        polar_land_fraction_max: 0.60,
        river_density_km_per_million_km2_min: 1000.0,
        river_density_km_per_million_km2_max: 7000.0,
        max_atlas_build_seconds: 30.0,
        max_snapshot_memory_mib: 500.0,
    },
];

pub fn preset_expectations() -> &'static [EnvironmentPresetExpectations] {
    &PRESET_EXPECTATIONS
}

/// Frozen statistics from the pre-M6 classification pipeline. These are
/// regression locks, not targets for the future physical-unit climate model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LegacyBaselineExpectation {
    pub model_version: EnvironmentModelVersion,
    pub preset: EnvironmentPreset,
    pub seed: u64,
    pub sample_count: usize,
    pub land_fraction: f32,
    pub ocean_fraction: f32,
    pub mean_elevation_m: f32,
    pub min_elevation_m: f32,
    pub max_elevation_m: f32,
    pub mean_temperature: f32,
    pub min_temperature: f32,
    pub max_temperature: f32,
    pub mean_moisture: f32,
    pub min_moisture: f32,
    pub max_moisture: f32,
    pub biome_counts: [usize; Biome::COUNT],
}

static LEGACY_BASELINE_EXPECTATIONS: [LegacyBaselineExpectation; 3] = [
    LegacyBaselineExpectation {
        model_version: LEGACY_BASELINE_MODEL_VERSION,
        preset: EnvironmentPreset::EarthLike,
        seed: LEGACY_BASELINE_SEED,
        sample_count: DEFAULT_BASELINE_SAMPLE_COUNT,
        land_fraction: 0.665_039_06,
        ocean_fraction: 0.334_960_94,
        mean_elevation_m: 922.654_97,
        min_elevation_m: -4571.422,
        max_elevation_m: 7864.986,
        mean_temperature: 0.430_425_9,
        min_temperature: 0.0,
        max_temperature: 1.0,
        mean_moisture: 0.486_807_94,
        min_moisture: 0.156_438_01,
        max_moisture: 0.623_917_2,
        biome_counts: [207, 203, 249, 713, 18, 319, 105, 0, 0, 0, 0, 361, 0, 1921],
    },
    LegacyBaselineExpectation {
        model_version: LEGACY_BASELINE_MODEL_VERSION,
        preset: EnvironmentPreset::HotArid,
        seed: LEGACY_BASELINE_SEED,
        sample_count: DEFAULT_BASELINE_SAMPLE_COUNT,
        land_fraction: 0.672_851_56,
        ocean_fraction: 0.327_148_44,
        mean_elevation_m: 922.654_97,
        min_elevation_m: -4571.422,
        max_elevation_m: 7864.986,
        mean_temperature: 0.501_485_8,
        min_temperature: 0.0,
        max_temperature: 1.0,
        mean_moisture: 0.485_815_5,
        min_moisture: 0.201_965_44,
        max_moisture: 0.519_794_6,
        biome_counts: [207, 198, 252, 683, 17, 422, 35, 0, 0, 0, 0, 361, 0, 1921],
    },
    LegacyBaselineExpectation {
        model_version: LEGACY_BASELINE_MODEL_VERSION,
        preset: EnvironmentPreset::ColdIcehouse,
        seed: LEGACY_BASELINE_SEED,
        sample_count: DEFAULT_BASELINE_SAMPLE_COUNT,
        land_fraction: 0.644_775_4,
        ocean_fraction: 0.355_224_6,
        mean_elevation_m: 922.654_97,
        min_elevation_m: -4571.422,
        max_elevation_m: 7864.986,
        mean_temperature: 0.457_286_15,
        min_temperature: 0.0,
        max_temperature: 1.0,
        mean_moisture: 0.493_338_97,
        min_moisture: 0.316_076_55,
        max_moisture: 0.579_178_2,
        biome_counts: [210, 206, 271, 768, 21, 245, 93, 0, 0, 0, 0, 361, 0, 1921],
    },
];

pub fn legacy_baseline_expectations() -> &'static [LegacyBaselineExpectation] {
    &LEGACY_BASELINE_EXPECTATIONS
}

pub fn planet_params_with_preset(seed: PlanetSeed, preset: EnvironmentPreset) -> PlanetParams {
    preset.apply(planet_params(seed))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldProvenance {
    Procedural,
    LearnedVisual,
    Composed,
    Planned,
}

impl FieldProvenance {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Procedural => "procedural",
            Self::LearnedVisual => "learned_visual",
            Self::Composed => "composed",
            Self::Planned => "planned",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateFrequency {
    StaticBake,
    OnTerrainRevision,
}

impl UpdateFrequency {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StaticBake => "static_bake",
            Self::OnTerrainRevision => "on_terrain_revision",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldAuthority {
    Canonical,
    VisualOnly,
}

impl FieldAuthority {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::VisualOnly => "visual_only",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldAvailability {
    Legacy,
    M6_2,
    PlannedM6_3,
    PlannedM6_4,
    PlannedM6_5,
    PlannedM6_6,
    PlannedM6_7,
}

impl FieldAvailability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::M6_2 => "m6_2",
            Self::PlannedM6_3 => "planned_m6_3",
            Self::PlannedM6_4 => "planned_m6_4",
            Self::PlannedM6_5 => "planned_m6_5",
            Self::PlannedM6_6 => "planned_m6_6",
            Self::PlannedM6_7 => "planned_m6_7",
        }
    }
}

/// Machine-readable field documentation used by manifests and debug tooling.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FieldMetadata {
    pub name: &'static str,
    pub units: &'static str,
    pub range_min: f32,
    pub range_max: f32,
    pub spatial_scale: &'static str,
    pub provenance: FieldProvenance,
    pub update_frequency: UpdateFrequency,
    pub authority: FieldAuthority,
    pub availability: FieldAvailability,
}

macro_rules! field {
    ($name:literal, $units:literal, $min:expr, $max:expr, $scale:literal, $provenance:ident, $authority:ident, $availability:ident) => {
        FieldMetadata {
            name: $name,
            units: $units,
            range_min: $min,
            range_max: $max,
            spatial_scale: $scale,
            provenance: FieldProvenance::$provenance,
            update_frequency: UpdateFrequency::StaticBake,
            authority: FieldAuthority::$authority,
            availability: FieldAvailability::$availability,
        }
    };
}

static FIELD_METADATA: [FieldMetadata; 61] = [
    field!(
        "legacy.temperature",
        "normalized_0_1",
        0.0,
        1.0,
        "planetary_plus_regional",
        Procedural,
        Canonical,
        Legacy
    ),
    field!(
        "legacy.moisture",
        "normalized_0_1",
        0.0,
        1.0,
        "regional",
        Procedural,
        Canonical,
        Legacy
    ),
    field!(
        "climate.annual_mean_temperature_c",
        "degrees_celsius",
        -90.0,
        60.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.warmest_month_temperature_c",
        "degrees_celsius",
        -90.0,
        70.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.coldest_month_temperature_c",
        "degrees_celsius",
        -100.0,
        50.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.growing_season_days",
        "days_per_year",
        0.0,
        366.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.growing_degree_days_c",
        "degree_celsius_days",
        0.0,
        15000.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.annual_precipitation_mm",
        "millimetres_per_year",
        0.0,
        12000.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.driest_month_precipitation_mm",
        "millimetres_per_month",
        0.0,
        3000.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.wettest_month_precipitation_mm",
        "millimetres_per_month",
        0.0,
        5000.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.precipitation_seasonality",
        "coefficient_of_variation",
        0.0,
        5.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.snow_fraction",
        "fraction_0_1",
        0.0,
        1.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.pet_mm",
        "millimetres_per_year",
        0.0,
        5000.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.water_balance_mm",
        "millimetres_per_year",
        -5000.0,
        12000.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.aridity_index",
        "dimensionless",
        -1.0,
        10.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.moisture_index",
        "dimensionless",
        -1.0,
        1.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.prevailing_wind_speed",
        "metres_per_second",
        0.0,
        100.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.prevailing_wind_direction",
        "radians",
        0.0,
        6.2831855,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "climate.continentality",
        "degrees_celsius",
        0.0,
        80.0,
        "climate_atlas",
        Procedural,
        Canonical,
        M6_2
    ),
    field!(
        "landform.elevation_above_sea_m",
        "metres",
        -12000.0,
        12000.0,
        "terrain_field",
        Composed,
        Canonical,
        Legacy
    ),
    field!(
        "landform.slope_degrees",
        "degrees",
        0.0,
        90.0,
        "terrain_vertex",
        Composed,
        Canonical,
        Legacy
    ),
    field!(
        "landform.aspect_radians",
        "radians",
        0.0,
        6.2831855,
        "terrain_vertex",
        Composed,
        Canonical,
        Legacy
    ),
    field!(
        "landform.profile_curvature_per_m",
        "inverse_metres",
        -1.0,
        1.0,
        "landform_neighbourhood",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "landform.plan_curvature_per_m",
        "inverse_metres",
        -1.0,
        1.0,
        "landform_neighbourhood",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "landform.local_relief_m",
        "metres",
        0.0,
        12000.0,
        "landform_neighbourhood",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "landform.ruggedness",
        "dimensionless_0_1",
        0.0,
        1.0,
        "landform_neighbourhood",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "landform.exposure",
        "dimensionless_0_1",
        0.0,
        1.0,
        "landform_neighbourhood",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "landform.valley_position",
        "dimensionless_signed",
        -1.0,
        1.0,
        "landform_neighbourhood",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "landform.coastal_proximity_m",
        "metres",
        0.0,
        20000000.0,
        "landform_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.depth_m",
        "metres",
        0.0,
        20.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.sand_fraction",
        "fraction_0_1",
        0.0,
        1.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.silt_fraction",
        "fraction_0_1",
        0.0,
        1.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.clay_fraction",
        "fraction_0_1",
        0.0,
        1.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.water_capacity_mm",
        "millimetres",
        0.0,
        2000.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.fertility",
        "dimensionless_0_1",
        0.0,
        1.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.organic_fraction",
        "fraction_0_1",
        0.0,
        1.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.permafrost_fraction",
        "fraction_0_1",
        0.0,
        1.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.salinity_ds_m",
        "decisiemens_per_metre",
        0.0,
        100.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "soil.bare_substrate_fraction",
        "fraction_0_1",
        0.0,
        1.0,
        "soil_atlas",
        Planned,
        Canonical,
        PlannedM6_3
    ),
    field!(
        "hydrology.flow_accumulation_m2",
        "square_metres",
        0.0,
        5.2e14,
        "hydrology_atlas",
        Planned,
        Canonical,
        PlannedM6_4
    ),
    field!(
        "hydrology.channel_likelihood",
        "probability_0_1",
        0.0,
        1.0,
        "hydrology_atlas",
        Planned,
        Canonical,
        PlannedM6_4
    ),
    field!(
        "hydrology.intermittent_likelihood",
        "probability_0_1",
        0.0,
        1.0,
        "hydrology_atlas",
        Planned,
        Canonical,
        PlannedM6_4
    ),
    field!(
        "hydrology.discharge_class",
        "ordinal_class_0_255",
        0.0,
        255.0,
        "hydrology_atlas",
        Planned,
        Canonical,
        PlannedM6_4
    ),
    field!(
        "hydrology.river_width_m",
        "metres",
        0.0,
        50000.0,
        "hydrology_atlas",
        Planned,
        Canonical,
        PlannedM6_4
    ),
    field!(
        "hydrology.riparian_influence",
        "dimensionless_0_1",
        0.0,
        1.0,
        "hydrology_atlas",
        Planned,
        Canonical,
        PlannedM6_4
    ),
    field!(
        "hydrology.floodplain_potential",
        "dimensionless_0_1",
        0.0,
        1.0,
        "hydrology_atlas",
        Planned,
        Canonical,
        PlannedM6_4
    ),
    field!(
        "hydrology.topographic_wetness",
        "dimensionless_index",
        -20.0,
        30.0,
        "hydrology_atlas",
        Planned,
        Canonical,
        PlannedM6_4
    ),
    field!(
        "disturbance.fire_potential",
        "dimensionless_0_1",
        0.0,
        1.0,
        "disturbance_atlas",
        Planned,
        Canonical,
        PlannedM6_6
    ),
    field!(
        "disturbance.flood_potential",
        "dimensionless_0_1",
        0.0,
        1.0,
        "disturbance_atlas",
        Planned,
        Canonical,
        PlannedM6_6
    ),
    field!(
        "disturbance.wind_exposure",
        "dimensionless_0_1",
        0.0,
        1.0,
        "disturbance_atlas",
        Planned,
        Canonical,
        PlannedM6_6
    ),
    field!(
        "disturbance.avalanche_potential",
        "dimensionless_0_1",
        0.0,
        1.0,
        "disturbance_atlas",
        Planned,
        Canonical,
        PlannedM6_6
    ),
    field!(
        "disturbance.volcanic_succession",
        "dimensionless_0_1",
        0.0,
        1.0,
        "disturbance_atlas",
        Planned,
        Canonical,
        PlannedM6_6
    ),
    field!(
        "disturbance.glacial_disturbance",
        "dimensionless_0_1",
        0.0,
        1.0,
        "disturbance_atlas",
        Planned,
        Canonical,
        PlannedM6_6
    ),
    field!(
        "biome.membership_weight",
        "normalized_weight",
        0.0,
        1.0,
        "environment_snapshot",
        Planned,
        Canonical,
        PlannedM6_5
    ),
    field!(
        "biome.dominance_confidence",
        "dimensionless_0_1",
        0.0,
        1.0,
        "environment_snapshot",
        Planned,
        Canonical,
        PlannedM6_5
    ),
    field!(
        "biome.ecotone_strength",
        "dimensionless_0_1",
        0.0,
        1.0,
        "environment_snapshot",
        Planned,
        Canonical,
        PlannedM6_5
    ),
    field!(
        "material.final_masks",
        "normalized_fractions",
        0.0,
        1.0,
        "terrain_material",
        Planned,
        VisualOnly,
        PlannedM6_7
    ),
    field!(
        "visual_climate.temp_c",
        "degrees_celsius",
        -10.0,
        40.0,
        "learned_surface_chart",
        LearnedVisual,
        VisualOnly,
        Legacy
    ),
    field!(
        "visual_climate.temp_seasonal_c",
        "degrees_celsius",
        -15.0,
        15.0,
        "learned_surface_chart",
        LearnedVisual,
        VisualOnly,
        Legacy
    ),
    field!(
        "visual_climate.precip_mm",
        "millimetres_per_year",
        0.0,
        4000.0,
        "learned_surface_chart",
        LearnedVisual,
        VisualOnly,
        Legacy
    ),
    field!(
        "visual_climate.precip_cv",
        "coefficient_of_variation",
        0.0,
        1.0,
        "learned_surface_chart",
        LearnedVisual,
        VisualOnly,
        Legacy
    ),
];

pub fn field_metadata() -> &'static [FieldMetadata] {
    &FIELD_METADATA
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct ClimateNormals {
    pub annual_mean_temperature_c: f32,
    pub warmest_month_temperature_c: f32,
    pub coldest_month_temperature_c: f32,
    pub growing_season_days: u16,
    pub growing_degree_days_c: f32,
    pub annual_precipitation_mm: f32,
    pub driest_month_precipitation_mm: f32,
    pub wettest_month_precipitation_mm: f32,
    pub precipitation_seasonality: f32,
    pub snow_fraction: f32,
    pub pet_mm: f32,
    pub water_balance_mm: f32,
    pub aridity_index: f32,
    pub moisture_index: f32,
    pub monthly_temperature_c: [f32; 12],
    pub monthly_precipitation_mm: [f32; 12],
}

impl ClimateNormals {
    pub fn is_valid(self) -> bool {
        let temperature_ok = self.annual_mean_temperature_c.is_finite()
            && self.warmest_month_temperature_c.is_finite()
            && self.coldest_month_temperature_c.is_finite()
            && self.coldest_month_temperature_c <= self.annual_mean_temperature_c
            && self.annual_mean_temperature_c <= self.warmest_month_temperature_c
            && (-90.0..=60.0).contains(&self.annual_mean_temperature_c)
            && (-90.0..=70.0).contains(&self.warmest_month_temperature_c)
            && (-100.0..=50.0).contains(&self.coldest_month_temperature_c);

        let monthly_temperature_ok = self.monthly_temperature_c.iter().all(|t| {
            t.is_finite()
                && (-100.0..=70.0).contains(t)
                && self.coldest_month_temperature_c <= *t
                && *t <= self.warmest_month_temperature_c
        });

        let growing_ok = self.growing_season_days <= 366
            && self.growing_degree_days_c.is_finite()
            && self.growing_degree_days_c >= 0.0;

        let precipitation_ok = self.annual_precipitation_mm.is_finite()
            && (0.0..=12000.0).contains(&self.annual_precipitation_mm)
            && self.driest_month_precipitation_mm.is_finite()
            && (0.0..=3000.0).contains(&self.driest_month_precipitation_mm)
            && self.wettest_month_precipitation_mm.is_finite()
            && (0.0..=5000.0).contains(&self.wettest_month_precipitation_mm)
            && self.driest_month_precipitation_mm <= self.wettest_month_precipitation_mm;

        let monthly_precipitation_ok = self.monthly_precipitation_mm.iter().all(|p| {
            p.is_finite()
                && (0.0..=5000.0).contains(p)
                && self.driest_month_precipitation_mm <= *p
                && *p <= self.wettest_month_precipitation_mm
        });
        let monthly_precipitation_sum: f32 = self.monthly_precipitation_mm.iter().sum();
        let monthly_precipitation_sum_ok =
            (monthly_precipitation_sum - self.annual_precipitation_mm).abs()
                <= 0.01 * self.annual_precipitation_mm.max(1.0) + 1.0;

        let seasonality_ok = self.precipitation_seasonality.is_finite()
            && (0.0..=5.0).contains(&self.precipitation_seasonality);

        let snow_ok = (0.0..=1.0).contains(&self.snow_fraction);

        let pet_ok = self.pet_mm.is_finite() && (0.0..=5000.0).contains(&self.pet_mm);

        let water_balance_ok = self.water_balance_mm.is_finite()
            && (-5000.0..=12000.0).contains(&self.water_balance_mm)
            && (self.water_balance_mm - (self.annual_precipitation_mm - self.pet_mm)).abs()
                <= 0.05 * self.annual_precipitation_mm.max(self.pet_mm).max(1.0) + 5.0;

        let aridity_ok = self.aridity_index.is_finite()
            && (-1.0..=10.0).contains(&self.aridity_index)
            && (self.annual_precipitation_mm <= 0.0
                || (self.aridity_index - self.pet_mm / self.annual_precipitation_mm).abs() <= 0.5);

        let moisture_ok =
            self.moisture_index.is_finite() && (-1.0..=1.0).contains(&self.moisture_index);

        temperature_ok
            && monthly_temperature_ok
            && growing_ok
            && precipitation_ok
            && monthly_precipitation_ok
            && monthly_precipitation_sum_ok
            && seasonality_ok
            && snow_ok
            && pet_ok
            && water_balance_ok
            && aridity_ok
            && moisture_ok
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct LandformSample {
    pub elevation_above_sea_m: f64,
    pub slope_degrees: f32,
    pub aspect_radians: f32,
    pub profile_curvature_per_m: f32,
    pub plan_curvature_per_m: f32,
    pub local_relief_m: f32,
    pub ruggedness: f32,
    pub exposure: f32,
    pub valley_position: f32,
    pub coastal_proximity_m: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SoilSample {
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
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct HydrologySample {
    pub flow_accumulation_m2: f64,
    pub channel_likelihood: f32,
    pub intermittent_likelihood: f32,
    pub discharge_class: u8,
    pub river_width_m: f32,
    pub riparian_influence: f32,
    pub floodplain_potential: f32,
    pub topographic_wetness: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct DisturbanceSample {
    pub fire_potential: f32,
    pub flood_potential: f32,
    pub wind_exposure: f32,
    pub avalanche_potential: f32,
    pub volcanic_succession: f32,
    pub glacial_disturbance: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct EcosystemMembership {
    pub ecosystem_id: u32,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct BiomeMixture {
    pub memberships: [EcosystemMembership; MAX_BIOME_MEMBERSHIPS],
    pub dominance_confidence: f32,
    pub ecotone_strength: f32,
}

impl BiomeMixture {
    pub fn is_valid(self) -> bool {
        let sum: f32 = self.memberships.iter().map(|entry| entry.weight).sum();
        self.memberships
            .iter()
            .all(|entry| entry.weight.is_finite() && (0.0..=1.0).contains(&entry.weight))
            && (sum == 0.0 || (sum - 1.0).abs() <= 1.0e-4)
            && (0.0..=1.0).contains(&self.dominance_confidence)
            && (0.0..=1.0).contains(&self.ecotone_strength)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct EnvironmentSample {
    pub climate: ClimateNormals,
    pub landform: LandformSample,
    pub soil: SoilSample,
    pub hydrology: HydrologySample,
    pub disturbance: DisturbanceSample,
    pub biome: BiomeMixture,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct WorldEnvironmentIdentity {
    pub seed: u64,
    pub model_version: EnvironmentModelVersion,
    pub preset: EnvironmentPreset,
    pub planet_radius_m: u64,
    pub sea_level_datum_mm: i64,
}

/// Immutable identity/checksum foundation. Atlas storage arrives in M6.2+.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WorldEnvironmentSnapshot {
    identity: WorldEnvironmentIdentity,
    canonical_checksum: [u8; 32],
}

impl WorldEnvironmentSnapshot {
    pub const fn new(identity: WorldEnvironmentIdentity, canonical_checksum: [u8; 32]) -> Self {
        Self {
            identity,
            canonical_checksum,
        }
    }

    pub const fn identity(&self) -> &WorldEnvironmentIdentity {
        &self.identity
    }

    pub const fn canonical_checksum(&self) -> &[u8; 32] {
        &self.canonical_checksum
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LegacyEnvironmentBaseline {
    pub model_version: EnvironmentModelVersion,
    pub preset: EnvironmentPreset,
    pub seed: u64,
    pub sample_count: usize,
    pub land_fraction: f32,
    pub ocean_fraction: f32,
    pub mean_elevation_m: f32,
    pub min_elevation_m: f32,
    pub max_elevation_m: f32,
    pub mean_temperature: f32,
    pub min_temperature: f32,
    pub max_temperature: f32,
    pub mean_moisture: f32,
    pub min_moisture: f32,
    pub max_moisture: f32,
    pub biome_counts: Vec<(String, usize)>,
    pub biome_fractions: Vec<(String, f32)>,
}

impl LegacyEnvironmentBaseline {
    pub fn matches_golden(&self) -> bool {
        let Some(expected) = legacy_baseline_expectations()
            .iter()
            .find(|expected| expected.preset == self.preset)
        else {
            return false;
        };
        let counts_match = self.biome_counts.len() == expected.biome_counts.len()
            && self
                .biome_counts
                .iter()
                .zip(expected.biome_counts)
                .all(|((_, actual), expected)| *actual == expected);
        self.model_version == expected.model_version
            && self.seed == expected.seed
            && self.sample_count == expected.sample_count
            && (self.land_fraction - expected.land_fraction).abs() <= 1.0e-6
            && (self.ocean_fraction - expected.ocean_fraction).abs() <= 1.0e-6
            && (self.mean_elevation_m - expected.mean_elevation_m).abs() <= 0.01
            && (self.min_elevation_m - expected.min_elevation_m).abs() <= 0.01
            && (self.max_elevation_m - expected.max_elevation_m).abs() <= 0.01
            && (self.mean_temperature - expected.mean_temperature).abs() <= 1.0e-6
            && (self.min_temperature - expected.min_temperature).abs() <= 1.0e-6
            && (self.max_temperature - expected.max_temperature).abs() <= 1.0e-6
            && (self.mean_moisture - expected.mean_moisture).abs() <= 1.0e-6
            && (self.min_moisture - expected.min_moisture).abs() <= 1.0e-6
            && (self.max_moisture - expected.max_moisture).abs() <= 1.0e-6
            && counts_match
    }
}

pub fn collect_legacy_environment_baseline(
    seed: u64,
    preset: EnvironmentPreset,
    planet_radius_m: f64,
    elevation_scale_m: f32,
    sample_count: usize,
) -> LegacyEnvironmentBaseline {
    let sample_count = sample_count.max(1);
    let planet_seed = PlanetSeed(seed);
    let terrain = ProceduralTerrainField::new_metric(
        elevation_params(planet_seed),
        planet_params_with_preset(planet_seed, preset),
        planet_radius_m,
    );
    let mut biome_counts = [0usize; Biome::COUNT];
    let mut land_count = 0usize;
    let mut elevation_sum = 0.0f64;
    let mut elevation_min = f64::INFINITY;
    let mut elevation_max = f64::NEG_INFINITY;
    let mut temperature_sum = 0.0f64;
    let mut temperature_min = f32::INFINITY;
    let mut temperature_max = f32::NEG_INFINITY;
    let mut moisture_sum = 0.0f64;
    let mut moisture_min = f32::INFINITY;
    let mut moisture_max = f32::NEG_INFINITY;

    let golden_angle = PI * (3.0 - 5.0f64.sqrt());
    for index in 0..sample_count {
        let y = 1.0 - 2.0 * ((index as f64 + 0.5) / sample_count as f64);
        let radial = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden_angle * index as f64;
        let dir = DVec3::new(theta.cos() * radial, y, theta.sin() * radial);
        let sample = terrain.sample(dir);
        let elevation_m = sample.elevation * elevation_scale_m as f64;
        elevation_sum += elevation_m;
        elevation_min = elevation_min.min(elevation_m);
        elevation_max = elevation_max.max(elevation_m);
        temperature_sum += sample.temperature as f64;
        temperature_min = temperature_min.min(sample.temperature);
        temperature_max = temperature_max.max(sample.temperature);
        moisture_sum += sample.moisture as f64;
        moisture_min = moisture_min.min(sample.moisture);
        moisture_max = moisture_max.max(sample.moisture);
        let biome_index = sample.biome.as_index();
        biome_counts[biome_index] += 1;
        if !sample.biome.is_ocean() {
            land_count += 1;
        }
    }

    let count = sample_count as f32;
    LegacyEnvironmentBaseline {
        model_version: LEGACY_BASELINE_MODEL_VERSION,
        preset,
        seed,
        sample_count,
        land_fraction: land_count as f32 / count,
        ocean_fraction: (sample_count - land_count) as f32 / count,
        mean_elevation_m: (elevation_sum / sample_count as f64) as f32,
        min_elevation_m: elevation_min as f32,
        max_elevation_m: elevation_max as f32,
        mean_temperature: (temperature_sum / sample_count as f64) as f32,
        min_temperature: temperature_min,
        max_temperature: temperature_max,
        mean_moisture: (moisture_sum / sample_count as f64) as f32,
        min_moisture: moisture_min,
        max_moisture: moisture_max,
        biome_counts: Biome::ALL
            .iter()
            .zip(biome_counts)
            .map(|(biome, count)| (biome.as_str().to_owned(), count))
            .collect(),
        biome_fractions: Biome::ALL
            .iter()
            .zip(biome_counts)
            .map(|(biome, biome_count)| (biome.as_str().to_owned(), biome_count as f32 / count))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_version_compatibility_is_strict() {
        let current = EnvironmentModelVersion::current();
        assert_ne!(current.model, 0);
        assert_ne!(current.climate_algorithm, 0);
        assert_ne!(current.landform_algorithm, 0);
        assert_ne!(current.soil_algorithm, 0);
        assert_ne!(current.hydrology_algorithm, 0);
        assert_ne!(current.disturbance_algorithm, 0);
        assert_ne!(current.biome_catalog, 0);
        assert_ne!(current.field_resolution, 0);
        assert_ne!(current.constants, 0);
        assert!(current.is_cache_compatible_with(current));
        let mut changed = current;
        changed.climate_algorithm += 1;
        assert!(!current.is_cache_compatible_with(changed));
    }

    #[test]
    fn earthlike_preset_is_bit_identical() {
        let seed = PlanetSeed(0xC0FFEE);
        let original = planet_params(seed);
        let applied = planet_params_with_preset(seed, EnvironmentPreset::EarthLike);
        assert_eq!(original, applied);
    }

    #[test]
    fn stress_presets_are_deterministic_and_distinct() {
        let seed = PlanetSeed(0xC0FFEE);
        let hot_a = planet_params_with_preset(seed, EnvironmentPreset::HotArid);
        let hot_b = planet_params_with_preset(seed, EnvironmentPreset::HotArid);
        let cold = planet_params_with_preset(seed, EnvironmentPreset::ColdIcehouse);
        assert_eq!(hot_a, hot_b);
        assert_ne!(hot_a.temp_gradient.to_bits(), cold.temp_gradient.to_bits());
        assert_ne!(
            hot_a.moisture_noise_amp.to_bits(),
            cold.moisture_noise_amp.to_bits()
        );
    }

    #[test]
    fn environment_preset_names_and_aliases_are_complete() {
        for preset in EnvironmentPreset::ALL {
            assert_eq!(EnvironmentPreset::parse(preset.as_str()), Some(preset));
        }
        assert_eq!(
            EnvironmentPreset::parse("hotarid"),
            Some(EnvironmentPreset::HotArid)
        );
        assert_eq!(
            EnvironmentPreset::parse("coldicehouse"),
            Some(EnvironmentPreset::ColdIcehouse)
        );
        assert_eq!(EnvironmentPreset::parse("mars"), None);
    }

    #[test]
    fn field_metadata_is_complete_and_well_formed() {
        assert_eq!(field_metadata().len(), 61);
        let mut names = std::collections::HashSet::new();
        for metadata in field_metadata() {
            assert!(!metadata.name.is_empty());
            assert!(
                names.insert(metadata.name),
                "duplicate field {}",
                metadata.name
            );
            assert!(!metadata.units.is_empty());
            assert!(metadata.range_min.is_finite());
            assert!(metadata.range_max.is_finite());
            assert!(metadata.range_min <= metadata.range_max);
            assert!(!metadata.spatial_scale.is_empty());
        }
    }

    #[test]
    fn scientific_model_roles_and_legacy_gaps_are_locked() {
        assert_eq!(LEGACY_ENVIRONMENT_PIPELINE_GAPS.len(), 7);
        assert_eq!(
            HYBRID_CLIMATE_MODEL.learned_climate_authority,
            FieldAuthority::VisualOnly
        );
        assert!(HYBRID_CLIMATE_MODEL
            .koppen_geiger_role
            .contains("diagnostic"));
        assert!(HYBRID_CLIMATE_MODEL
            .holdridge_role
            .contains("biotemperature"));
    }

    #[test]
    fn every_environment_preset_has_bounded_expectations() {
        assert_eq!(preset_expectations().len(), 3);
        for expected in preset_expectations() {
            assert!(expected.land_fraction_min < expected.land_fraction_max);
            assert!(expected.annual_temperature_c_min < expected.annual_temperature_c_max);
            assert!(expected.annual_precipitation_mm_min < expected.annual_precipitation_mm_max);
            assert!(expected.arid_land_fraction_min < expected.arid_land_fraction_max);
            assert!(expected.polar_land_fraction_min < expected.polar_land_fraction_max);
            assert!(
                expected.river_density_km_per_million_km2_min
                    < expected.river_density_km_per_million_km2_max
            );
            assert!(expected.max_atlas_build_seconds > 0.0);
            assert!(expected.max_snapshot_memory_mib > 0.0);
        }
    }

    #[test]
    fn climate_normals_reject_non_finite_or_out_of_range_fields() {
        let valid = ClimateNormals {
            annual_mean_temperature_c: 12.0,
            warmest_month_temperature_c: 24.0,
            coldest_month_temperature_c: 0.0,
            growing_season_days: 180,
            growing_degree_days_c: 1200.0,
            annual_precipitation_mm: 900.0,
            driest_month_precipitation_mm: 40.0,
            wettest_month_precipitation_mm: 120.0,
            precipitation_seasonality: 0.4,
            snow_fraction: 0.1,
            pet_mm: 700.0,
            water_balance_mm: 200.0,
            aridity_index: 0.8,
            moisture_index: 0.2,
            monthly_temperature_c: [
                0.0, 2.0, 8.0, 12.0, 16.0, 20.0, 24.0, 22.0, 18.0, 12.0, 6.0, 2.0,
            ],
            monthly_precipitation_mm: [
                100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0,
            ],
        };
        assert!(valid.is_valid());

        let mut invalid = valid;
        invalid.wettest_month_precipitation_mm = 10.0;
        assert!(!invalid.is_valid());
        invalid = valid;
        invalid.growing_degree_days_c = f32::NAN;
        assert!(!invalid.is_valid());
        invalid = valid;
        invalid.aridity_index = 11.0;
        assert!(!invalid.is_valid());
        invalid = valid;
        invalid.water_balance_mm = 1000.0;
        assert!(!invalid.is_valid());
        invalid = valid;
        invalid.monthly_temperature_c[0] = f32::NAN;
        assert!(!invalid.is_valid());
        invalid = valid;
        invalid.monthly_precipitation_mm = [0.0; 12];
        assert!(!invalid.is_valid());
    }

    #[test]
    fn field_availability_m6_2_as_str() {
        assert_eq!(FieldAvailability::M6_2.as_str(), "m6_2");
    }

    #[test]
    fn canonical_climate_fields_are_m6_2_procedural_and_canonical() {
        let climate_fields: Vec<_> = field_metadata()
            .iter()
            .filter(|m| m.name.starts_with("climate."))
            .collect();
        assert!(!climate_fields.is_empty());
        for metadata in climate_fields {
            assert_eq!(metadata.provenance, FieldProvenance::Procedural);
            assert_eq!(metadata.availability, FieldAvailability::M6_2);
            assert_eq!(metadata.authority, FieldAuthority::Canonical);
        }
    }

    #[test]
    fn model_version_bumped_for_m6_2_and_legacy_frozen_at_v1() {
        let current = EnvironmentModelVersion::current();
        assert_eq!(current.model, 2);
        assert_eq!(current.climate_algorithm, 2);
        assert_eq!(current.field_resolution, 2);
        assert_eq!(current.constants, 2);
        assert_eq!(current.landform_algorithm, 1);
        assert_eq!(current.soil_algorithm, 1);
        assert_eq!(current.hydrology_algorithm, 1);
        assert_eq!(current.disturbance_algorithm, 1);
        assert_eq!(current.biome_catalog, 1);

        assert_eq!(LEGACY_BASELINE_MODEL_VERSION.model, 1);
        assert_eq!(LEGACY_BASELINE_MODEL_VERSION.climate_algorithm, 1);
        assert_eq!(LEGACY_BASELINE_MODEL_VERSION.field_resolution, 1);
        assert_eq!(LEGACY_BASELINE_MODEL_VERSION.constants, 1);
    }

    #[test]
    fn legacy_baseline_reports_frozen_v1_identity() {
        let baseline = collect_legacy_environment_baseline(
            LEGACY_BASELINE_SEED,
            EnvironmentPreset::EarthLike,
            6_371_000.0,
            1000.0,
            DEFAULT_BASELINE_SAMPLE_COUNT,
        );
        assert_eq!(baseline.model_version, LEGACY_BASELINE_MODEL_VERSION);
        assert!(baseline.matches_golden());
    }

    #[test]
    fn biome_mixture_requires_normalized_weights() {
        let valid = BiomeMixture {
            memberships: [
                EcosystemMembership {
                    ecosystem_id: 1,
                    weight: 0.7,
                },
                EcosystemMembership {
                    ecosystem_id: 2,
                    weight: 0.3,
                },
                EcosystemMembership::default(),
                EcosystemMembership::default(),
            ],
            dominance_confidence: 0.7,
            ecotone_strength: 0.3,
        };
        assert!(valid.is_valid());
        let mut invalid = valid;
        invalid.memberships[1].weight = 0.5;
        assert!(!invalid.is_valid());
    }

    #[test]
    fn snapshot_exposes_immutable_identity_and_checksum() {
        let identity = WorldEnvironmentIdentity {
            seed: 42,
            model_version: EnvironmentModelVersion::current(),
            preset: EnvironmentPreset::EarthLike,
            planet_radius_m: 6_371_000,
            sea_level_datum_mm: 0,
        };
        let snapshot = WorldEnvironmentSnapshot::new(identity, [7; 32]);
        assert_eq!(*snapshot.identity(), identity);
        assert_eq!(*snapshot.canonical_checksum(), [7; 32]);
    }

    #[test]
    fn legacy_baseline_is_deterministic_and_partitioned() {
        let a = collect_legacy_environment_baseline(
            0xC0FFEE,
            EnvironmentPreset::EarthLike,
            6_371_000.0,
            1000.0,
            256,
        );
        let b = collect_legacy_environment_baseline(
            0xC0FFEE,
            EnvironmentPreset::EarthLike,
            6_371_000.0,
            1000.0,
            256,
        );
        assert_eq!(a, b);
        assert!((a.land_fraction + a.ocean_fraction - 1.0).abs() <= f32::EPSILON);
        assert_eq!(
            a.biome_counts.iter().map(|(_, count)| count).sum::<usize>(),
            256
        );
        assert!(
            (a.biome_fractions
                .iter()
                .map(|(_, value)| value)
                .sum::<f32>()
                - 1.0)
                .abs()
                < 1.0e-5
        );
    }

    #[test]
    fn fixed_seed_legacy_baselines_match_frozen_golden_statistics() {
        for preset in [
            EnvironmentPreset::EarthLike,
            EnvironmentPreset::HotArid,
            EnvironmentPreset::ColdIcehouse,
        ] {
            let baseline = collect_legacy_environment_baseline(
                LEGACY_BASELINE_SEED,
                preset,
                6_371_000.0,
                1000.0,
                DEFAULT_BASELINE_SAMPLE_COUNT,
            );
            assert!(
                baseline.matches_golden(),
                "legacy baseline drifted for {preset:?}: {baseline:#?}"
            );
        }
    }
}
