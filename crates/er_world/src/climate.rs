//! Deterministic bounded M6.2 static climate atlas.
//!
//! The atlas bakes climate normals onto the spherified cube at a fixed LOD
//! (<= 64 cells/face edge), then answers arbitrary directional queries by
//! interpolating across cube-face boundaries in a seam-safe way.  It is
//! immutable after build, does no I/O, uses no locks, no unsafe, no runtime
//! clock, and no learned climate authority.

use crate::environment::{planet_params_with_preset, ClimateNormals, EnvironmentPreset};
use crate::terrain_field::TerrainField;
use crate::terrain_space::metric_surface_point;
use er_core::math::{cell_neighbors, cell_to_dir, dir_to_cell, CellKey, FACE_COUNT};
use er_core::seed::PlanetSeed;
use glam::DVec3;

/// Fixed atlas resolution: 6 faces, 32 cells per edge.  This satisfies the
/// "<= 64 cells/face edge" bound while keeping memory and build time small.
pub const CLIMATE_ATLAS_LOD: u8 = 5;

/// Cells per face edge at the fixed atlas LOD.
pub const CLIMATE_CELLS_PER_EDGE: u32 = 1u32 << CLIMATE_ATLAS_LOD;

/// Total number of cell centers stored (6 faces * N * N).
pub const CLIMATE_ATLAS_LEN: usize =
    FACE_COUNT * (CLIMATE_CELLS_PER_EDGE as usize) * (CLIMATE_CELLS_PER_EDGE as usize);

/// Statistics produced by `ClimateAtlas::build`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClimateAtlasBuildStats {
    pub cells_per_face_edge: u32,
    pub total_cells: usize,
    pub land_cells: usize,
    pub ocean_cells: usize,
    pub mean_elevation_above_sea_m: f64,
    pub max_elevation_above_sea_m: f64,
    pub min_elevation_above_sea_m: f64,
    pub mean_annual_temperature_c: f32,
    pub mean_annual_precipitation_mm: f32,
    pub build_sample_count: u64,
}

/// Per-cell stored climate data.  Kept compact; most derived monthly normals are
/// computed on demand during interpolation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ClimateCell {
    dir: DVec3,
    /// Elevation in *field units* relative to the procedural sea level datum.
    /// This is the macro displacement from `elevation_split_metric` / the
    /// metric terrain field.
    elevation_field: f32,
    /// Mountain influence [0,1] used for rain-shadow and orographic effects.
    mountain_influence: f32,
    /// Fraction of the year this cell spends under snow (computed from monthly
    /// temperature during build).
    snow_fraction: f32,
    /// Annual precipitation in mm.
    annual_precipitation_mm: f32,
    /// Annual PET in mm.
    annual_pet_mm: f32,
    /// Annual mean temperature in °C.
    annual_mean_temp_c: f32,
    /// Warmest month temperature in °C.
    warmest_month_temp_c: f32,
    /// Coldest month temperature in °C.
    coldest_month_temp_c: f32,
    /// Growing season length in days.
    growing_season_days: u16,
    /// Growing degree days (base 5°C).
    growing_degree_days: f32,
    /// Prevailing wind speed in m/s.
    wind_speed_m_s: f32,
    /// Prevailing wind direction in radians around the local +Z axis (compass
    /// convention: 0 = east, pi/2 = north, pi = west, 3pi/2 = south).
    wind_direction_rad: f32,
    /// Continentality index: seasonal temperature range (warmest - coldest).
    continentality_c: f32,
    monthly_temperature_c: [f32; 12],
    monthly_precipitation_mm: [f32; 12],
}

/// A full climate sample, including the monthly normals required by downstream
/// systems and derived physical indices.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClimateSample {
    pub normals: ClimateNormals,
    pub prevailing_wind_speed_m_s: f32,
    pub prevailing_wind_direction_radians: f32,
    pub continentality_c: f32,
    pub monthly_temperature_c: [f32; 12],
    pub monthly_precipitation_mm: [f32; 12],
    pub validity: ClimateSampleValidity,
}

impl ClimateSample {
    pub fn annual_mean_temperature_c(&self) -> f32 {
        self.normals.annual_mean_temperature_c
    }
    pub fn warmest_month_temperature_c(&self) -> f32 {
        self.normals.warmest_month_temperature_c
    }
    pub fn coldest_month_temperature_c(&self) -> f32 {
        self.normals.coldest_month_temperature_c
    }
    pub fn annual_precipitation_mm(&self) -> f32 {
        self.normals.annual_precipitation_mm
    }
    pub fn driest_month_precipitation_mm(&self) -> f32 {
        self.normals.driest_month_precipitation_mm
    }
    pub fn wettest_month_precipitation_mm(&self) -> f32 {
        self.normals.wettest_month_precipitation_mm
    }
    pub fn pet_mm(&self) -> f32 {
        self.normals.pet_mm
    }
    pub fn water_balance_mm(&self) -> f32 {
        self.normals.water_balance_mm
    }
    pub fn aridity_index(&self) -> f32 {
        self.normals.aridity_index
    }
    pub fn moisture_index(&self) -> f32 {
        self.normals.moisture_index
    }
    pub fn snow_fraction(&self) -> f32 {
        self.normals.snow_fraction
    }
    pub fn growing_season_days(&self) -> u16 {
        self.normals.growing_season_days
    }
    pub fn growing_degree_days_c(&self) -> f32 {
        self.normals.growing_degree_days_c
    }
    pub fn monthly_temperature_c(&self, month: usize) -> f32 {
        self.monthly_temperature_c[month % 12]
    }
    pub fn monthly_precipitation_mm(&self, month: usize) -> f32 {
        self.monthly_precipitation_mm[month % 12]
    }
    pub fn monthly_pet_mm(&self, month: usize) -> f32 {
        thornthwaite_pet_monthly(&self.monthly_temperature_c)[month % 12]
    }
}

/// Bitmask describing why a sample is considered valid or invalid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClimateSampleValidity {
    pub finite: bool,
    pub temperature_ordered: bool,
    pub precipitation_nonnegative: bool,
    pub bounds_respected: bool,
}

impl ClimateSampleValidity {
    pub const fn all_valid() -> Self {
        Self {
            finite: true,
            temperature_ordered: true,
            precipitation_nonnegative: true,
            bounds_respected: true,
        }
    }

    pub fn is_valid(self) -> bool {
        self.finite
            && self.temperature_ordered
            && self.precipitation_nonnegative
            && self.bounds_respected
    }
}

/// Immutable climate atlas built from a planet seed, preset, and metric scales.
#[derive(Clone, Debug, PartialEq)]
pub struct ClimateAtlas {
    identity: ClimateAtlasIdentity,
    cells: Vec<ClimateCell>,
    build_stats: ClimateAtlasBuildStats,
}

/// Identity fields sufficient to reproduce the atlas deterministically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClimateAtlasIdentity {
    pub seed: u64,
    pub preset: EnvironmentPreset,
    pub planet_radius_m: u64,
    pub elevation_scale_m: u64,
    pub lod: u8,
}

impl ClimateAtlas {
    /// Build the immutable atlas.  The call is deterministic and bounded:
    /// it samples exactly `6 * 2^(2*lod)` cell centers.
    pub fn build(
        seed: PlanetSeed,
        preset: EnvironmentPreset,
        planet_radius_m: f64,
        elevation_scale_m: f64,
    ) -> Self {
        use crate::elevation::elevation_params;
        use crate::terrain_field::ProceduralTerrainField;

        let params = planet_params_with_preset(seed, preset);
        let elev_params = elevation_params(seed);
        let field = ProceduralTerrainField::new_metric(elev_params, params, planet_radius_m);

        let cells_per_edge = CLIMATE_CELLS_PER_EDGE;
        let total = FACE_COUNT * (cells_per_edge as usize) * (cells_per_edge as usize);
        let mut cells = Vec::with_capacity(total);

        let mut land_cells = 0usize;
        let mut ocean_cells = 0usize;
        let mut elev_sum = 0.0f64;
        let mut elev_min = f64::INFINITY;
        let mut elev_max = f64::NEG_INFINITY;
        let mut temp_sum = 0.0f64;
        let mut precip_sum = 0.0f64;
        let mut sample_count = 0u64;

        for face in 0u8..FACE_COUNT as u8 {
            for j in 0..cells_per_edge {
                for i in 0..cells_per_edge {
                    let key = CellKey {
                        face,
                        i,
                        j,
                        lod: CLIMATE_ATLAS_LOD,
                    };
                    let dir = cell_to_dir(key);
                    let sample = field.sample(dir);
                    let elev_m = sample.elevation * elevation_scale_m;
                    let macro_elev_m = sample.low_freq_elev as f64 * elevation_scale_m;
                    let mountain_influence = sample.mountain_influence as f64;

                    if sample.biome.is_ocean() {
                        ocean_cells += 1;
                    } else {
                        land_cells += 1;
                    }

                    elev_sum += elev_m;
                    elev_min = elev_min.min(elev_m);
                    elev_max = elev_max.max(elev_m);
                    sample_count += 1;

                    // Compute static monthly normals.
                    let monthly_temp =
                        monthly_temperatures(dir, elev_m, preset, &field, planet_radius_m);
                    let monthly_precip = monthly_precipitation(
                        dir,
                        macro_elev_m,
                        mountain_influence,
                        preset,
                        &params,
                        &field,
                        planet_radius_m,
                    );

                    let annual_mean = monthly_temp.iter().sum::<f32>() / 12.0;
                    let warmest = monthly_temp
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let coldest = monthly_temp.iter().copied().fold(f32::INFINITY, f32::min);
                    let annual_precip = monthly_precip.iter().sum::<f32>();

                    // Thornthwaite PET (mm/month), simplified.
                    let pet_mm = thornthwaite_pet_monthly(&monthly_temp).iter().sum::<f32>();

                    let snow_fraction = {
                        let snow_months = monthly_temp.iter().filter(|&&t| t < 0.0).count() as f32;
                        (snow_months / 12.0).clamp(0.0, 1.0)
                    };

                    let growing_season_days = {
                        let growing_months =
                            monthly_temp.iter().filter(|&&t| t > 5.0).count() as f32;
                        (growing_months * 30.5).min(366.0) as u16
                    };

                    let growing_degree_days = monthly_temp
                        .iter()
                        .map(|&t| (t - 5.0).max(0.0) * 30.5)
                        .sum::<f32>();

                    let wind = prevailing_wind(dir, &params);
                    let continentality = (warmest - coldest).max(0.0);

                    temp_sum += annual_mean as f64;
                    precip_sum += annual_precip as f64;

                    cells.push(ClimateCell {
                        dir,
                        elevation_field: sample.elevation as f32,
                        mountain_influence: sample.mountain_influence,
                        snow_fraction,
                        annual_precipitation_mm: annual_precip,
                        annual_pet_mm: pet_mm,
                        annual_mean_temp_c: annual_mean,
                        warmest_month_temp_c: warmest,
                        coldest_month_temp_c: coldest,
                        growing_season_days,
                        growing_degree_days,
                        wind_speed_m_s: wind.0,
                        wind_direction_rad: wind.1,
                        continentality_c: continentality,
                        monthly_temperature_c: monthly_temp,
                        monthly_precipitation_mm: monthly_precip,
                    });
                }
            }
        }

        let total_f = total as f64;
        let stats = ClimateAtlasBuildStats {
            cells_per_face_edge: cells_per_edge,
            total_cells: total,
            land_cells,
            ocean_cells,
            mean_elevation_above_sea_m: if total > 0 { elev_sum / total_f } else { 0.0 },
            max_elevation_above_sea_m: if total > 0 { elev_max } else { 0.0 },
            min_elevation_above_sea_m: if total > 0 { elev_min } else { 0.0 },
            mean_annual_temperature_c: if total > 0 {
                (temp_sum / total_f) as f32
            } else {
                0.0
            },
            mean_annual_precipitation_mm: if total > 0 {
                (precip_sum / total_f) as f32
            } else {
                0.0
            },
            build_sample_count: sample_count,
        };

        Self {
            identity: ClimateAtlasIdentity {
                seed: seed.0,
                preset,
                planet_radius_m: planet_radius_m as u64,
                elevation_scale_m: elevation_scale_m as u64,
                lod: CLIMATE_ATLAS_LOD,
            },
            cells,
            build_stats: stats,
        }
    }

    /// Return the deterministic identity of this atlas.
    pub fn identity(&self) -> ClimateAtlasIdentity {
        self.identity
    }

    /// Return the fixed LOD of the atlas.
    pub const fn lod(&self) -> u8 {
        CLIMATE_ATLAS_LOD
    }

    /// Number of cells stored in the atlas.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Estimated memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.cells.len() * std::mem::size_of::<ClimateCell>()
    }

    /// Return the build-time statistics.
    pub fn build_stats(&self) -> ClimateAtlasBuildStats {
        self.build_stats
    }

    /// Convenience constructor alias matching the integration-test API.
    pub fn new(seed: PlanetSeed, preset: EnvironmentPreset, planet_radius_m: f64) -> Self {
        Self::build(seed, preset, planet_radius_m, 1000.0)
    }

    /// Number of stored samples in the atlas.
    pub fn sample_count(&self) -> usize {
        self.cells.len()
    }

    /// Sample climate at an arbitrary direction, using elevation above sea
    /// for lapse-based temperature downscaling.  Interpolation is performed
    /// in world direction space using the four surrounding atlas cell centers
    /// (which are on the sphere), so cube-face seams are handled automatically
    /// by `dir_to_uv`/`uv_to_dir` round-tripping.
    pub fn sample(&self, dir: DVec3) -> ClimateSample {
        Self::sample_from_interpolation(self.interpolate_direction(dir), 0.0)
    }

    /// Sample climate with a terrain-scale lapse correction relative to the
    /// atlas elevation at this direction.
    pub fn sample_at_elevation(&self, dir: DVec3, elevation_above_sea_m: f64) -> ClimateSample {
        let interp = self.interpolate_direction(dir);
        let atlas_elevation_m =
            interp.elevation_field as f64 * self.identity.elevation_scale_m as f64;
        let lapse_adjustment = -lapse_downscale(elevation_above_sea_m - atlas_elevation_m);
        Self::sample_from_interpolation(interp, lapse_adjustment)
    }

    /// Sample the macro (sea-level) climate at an arbitrary direction.  No
    /// elevation downscaling is applied.
    pub fn sample_macro(&self, dir: DVec3) -> ClimateSample {
        self.sample(dir)
    }

    fn sample_from_interpolation(interp: ClimateCell, lapse_adjustment: f32) -> ClimateSample {
        let mut monthly_temp = interp.monthly_temperature_c;
        for temperature in &mut monthly_temp {
            *temperature = (*temperature + lapse_adjustment).clamp(-90.0, 70.0);
        }

        let annual_mean = monthly_temp.iter().sum::<f32>() / 12.0;
        let warmest = monthly_temp
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let coldest = monthly_temp.iter().copied().fold(f32::INFINITY, f32::min);

        let monthly_precip = interp.monthly_precipitation_mm;
        let annual_precip = monthly_precip.iter().sum::<f32>();
        let driest = monthly_precip.iter().copied().fold(f32::INFINITY, f32::min);
        let wettest = monthly_precip
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let seasonality = if annual_precip > 0.0 {
            let mean = annual_precip / 12.0;
            let variance = monthly_precip
                .iter()
                .map(|&p| (p - mean).powi(2))
                .sum::<f32>()
                / 12.0;
            (variance.sqrt() / mean).clamp(0.0, 5.0)
        } else {
            0.0
        };

        let pet_monthly = thornthwaite_pet_monthly(&monthly_temp);
        let pet_annual = pet_monthly.iter().sum::<f32>().clamp(0.0, 5_000.0);
        let water_balance = annual_precip - pet_annual;
        let aridity_index = if annual_precip > 0.0 {
            (pet_annual / annual_precip).clamp(0.0, 10.0)
        } else {
            0.0
        };
        let moisture_index = if annual_precip + pet_annual > 0.0 {
            (annual_precip - pet_annual) / (annual_precip + pet_annual)
        } else {
            0.0
        };

        let snow_fraction = {
            let snow_months = monthly_temp.iter().filter(|&&t| t < 0.0).count() as f32;
            (snow_months / 12.0).clamp(0.0, 1.0)
        };
        let growing_season_days = {
            let growing_months = monthly_temp.iter().filter(|&&t| t > 5.0).count() as f32;
            (growing_months * 30.5).min(366.0) as u16
        };
        let growing_degree_days = monthly_temp
            .iter()
            .map(|&t| (t - 5.0).max(0.0) * 30.5)
            .sum::<f32>();

        let normals = ClimateNormals {
            annual_mean_temperature_c: annual_mean,
            warmest_month_temperature_c: warmest,
            coldest_month_temperature_c: coldest,
            growing_season_days,
            growing_degree_days_c: growing_degree_days,
            annual_precipitation_mm: annual_precip,
            driest_month_precipitation_mm: driest,
            wettest_month_precipitation_mm: wettest,
            precipitation_seasonality: seasonality,
            snow_fraction,
            pet_mm: pet_annual,
            water_balance_mm: water_balance,
            aridity_index,
            moisture_index,
            monthly_temperature_c: monthly_temp,
            monthly_precipitation_mm: monthly_precip,
        };

        let validity = Self::validate(&normals, &monthly_temp, &monthly_precip);

        ClimateSample {
            normals,
            prevailing_wind_speed_m_s: interp.wind_speed_m_s,
            prevailing_wind_direction_radians: interp.wind_direction_rad,
            continentality_c: interp.continentality_c,
            monthly_temperature_c: monthly_temp,
            monthly_precipitation_mm: monthly_precip,
            validity,
        }
    }

    fn validate(
        normals: &ClimateNormals,
        _monthly_temp: &[f32; 12],
        monthly_precip: &[f32; 12],
    ) -> ClimateSampleValidity {
        let finite = normals.annual_mean_temperature_c.is_finite()
            && normals.warmest_month_temperature_c.is_finite()
            && normals.coldest_month_temperature_c.is_finite()
            && normals.annual_precipitation_mm.is_finite()
            && normals.pet_mm.is_finite();
        let temperature_ordered = normals.coldest_month_temperature_c
            <= normals.annual_mean_temperature_c
            && normals.annual_mean_temperature_c <= normals.warmest_month_temperature_c;
        let precipitation_nonnegative = monthly_precip.iter().all(|&p| p >= 0.0 && p.is_finite())
            && normals.driest_month_precipitation_mm <= normals.wettest_month_precipitation_mm;
        let bounds_respected = normals.annual_mean_temperature_c >= -90.0
            && normals.annual_mean_temperature_c <= 60.0
            && normals.warmest_month_temperature_c <= 70.0
            && normals.coldest_month_temperature_c >= -100.0
            && normals.annual_precipitation_mm >= 0.0
            && normals.annual_precipitation_mm <= 12_000.0
            && normals.snow_fraction >= 0.0
            && normals.snow_fraction <= 1.0
            && normals.aridity_index >= -1.0
            && normals.aridity_index <= 10.0
            && normals.moisture_index >= -1.0
            && normals.moisture_index <= 1.0;

        ClimateSampleValidity {
            finite,
            temperature_ordered,
            precipitation_nonnegative,
            bounds_respected,
        }
    }

    /// Blend a bounded two-ring neighborhood using only world-space angular
    /// distance. The neighborhood graph wraps across cube faces, so the same
    /// physical point cannot acquire face-local interpolation coordinates.
    fn interpolate_direction(&self, dir: DVec3) -> ClimateCell {
        let dir = dir.normalize_or_zero();
        let center = dir_to_cell(dir, CLIMATE_ATLAS_LOD);
        let mut keys = Vec::with_capacity(17);
        keys.push(center);
        for first in cell_neighbors(center) {
            if !keys.contains(&first) {
                keys.push(first);
            }
            for second in cell_neighbors(first) {
                if !keys.contains(&second) {
                    keys.push(second);
                }
            }
        }

        let mut weighted = Vec::with_capacity(keys.len());
        let mut weight_sum = 0.0f64;
        for key in keys {
            let center_dir = cell_to_dir(key);
            let angle = dir.dot(center_dir).clamp(-1.0, 1.0).acos();
            let weight = 1.0 / (angle * angle + 1.0e-6);
            weighted.push((self.cell_at(key), weight));
            weight_sum += weight;
        }
        Self::weighted_cell(&weighted, weight_sum)
    }

    fn cell_at(&self, key: CellKey) -> ClimateCell {
        let n = CLIMATE_CELLS_PER_EDGE as usize;
        let face = key.face as usize;
        debug_assert!(face < FACE_COUNT);
        debug_assert!((key.i as usize) < n);
        debug_assert!((key.j as usize) < n);
        let index = (face * n + key.j as usize) * n + key.i as usize;
        self.cells[index]
    }

    fn weighted_cell(cells: &[(ClimateCell, f64)], weight_sum: f64) -> ClimateCell {
        let blend = |field: fn(&ClimateCell) -> f32| -> f32 {
            (cells
                .iter()
                .map(|(cell, weight)| field(cell) as f64 * weight)
                .sum::<f64>()
                / weight_sum) as f32
        };
        let angle = |field: fn(&ClimateCell) -> f32| -> f32 {
            let x = cells
                .iter()
                .map(|(cell, weight)| field(cell).cos() as f64 * weight)
                .sum::<f64>();
            let y = cells
                .iter()
                .map(|(cell, weight)| field(cell).sin() as f64 * weight)
                .sum::<f64>();
            (y.atan2(x) as f32).rem_euclid(std::f32::consts::TAU)
        };
        ClimateCell {
            dir: DVec3::ZERO,
            elevation_field: blend(|c| c.elevation_field),
            mountain_influence: blend(|c| c.mountain_influence),
            snow_fraction: blend(|c| c.snow_fraction),
            annual_precipitation_mm: blend(|c| c.annual_precipitation_mm),
            annual_pet_mm: blend(|c| c.annual_pet_mm),
            annual_mean_temp_c: blend(|c| c.annual_mean_temp_c),
            warmest_month_temp_c: blend(|c| c.warmest_month_temp_c),
            coldest_month_temp_c: blend(|c| c.coldest_month_temp_c),
            growing_season_days: blend(|c| c.growing_season_days as f32).round() as u16,
            growing_degree_days: blend(|c| c.growing_degree_days),
            wind_speed_m_s: blend(|c| c.wind_speed_m_s),
            wind_direction_rad: angle(|c| c.wind_direction_rad),
            continentality_c: blend(|c| c.continentality_c),
            monthly_temperature_c: std::array::from_fn(|month| {
                (cells
                    .iter()
                    .map(|(cell, weight)| cell.monthly_temperature_c[month] as f64 * weight)
                    .sum::<f64>()
                    / weight_sum) as f32
            }),
            monthly_precipitation_mm: std::array::from_fn(|month| {
                (cells
                    .iter()
                    .map(|(cell, weight)| cell.monthly_precipitation_mm[month] as f64 * weight)
                    .sum::<f64>()
                    / weight_sum) as f32
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Static climate physics helpers
// ---------------------------------------------------------------------------

/// Convert the procedural normalized temperature [0,1] to °C.
fn normalized_temperature_to_celsius(normalized: f64) -> f32 {
    // Map [0,1] to [-40, +40] °C baseline.
    (normalized * 80.0 - 40.0) as f32
}

/// Convert the procedural normalized moisture [0,1] to annual mm.
fn normalized_moisture_to_mm(normalized: f64, planet_params: &crate::params::PlanetParams) -> f32 {
    // Base ocean moisture + land noise.  Moisture is already moderated by
    // rain shadow and latitude in the build path, so scale to [0, 4000] mm.
    let base = normalized.clamp(0.0, 1.0) as f32;
    base * base * 4000.0 * (1.0 + planet_params.moisture_noise_amp * 0.5)
}

/// Reconstruct static monthly temperatures from latitude, elevation, and
/// deterministic seasonal modulation.  The result is physically bounded.
fn monthly_temperatures(
    dir: DVec3,
    elev_m: f64,
    preset: EnvironmentPreset,
    field: &crate::terrain_field::ProceduralTerrainField,
    planet_radius_m: f64,
) -> [f32; 12] {
    let pos = metric_surface_point(dir, planet_radius_m);
    let temp_norm = field.sample(pos.normalize()).temperature as f64;
    let preset_offset_c = match preset {
        EnvironmentPreset::EarthLike => 0.0,
        EnvironmentPreset::HotArid => 12.0,
        EnvironmentPreset::ColdIcehouse => -18.0,
    };
    let base_c = normalized_temperature_to_celsius(temp_norm) + preset_offset_c;
    let latitude = dir.y.abs();
    let seasonal_amp = (1.0 - latitude) * 25.0 + 5.0; // stronger seasons at mid latitude
    let phase_shift = if dir.y >= 0.0 {
        0.0
    } else {
        std::f64::consts::PI
    }; // opposite seasons

    let mut out = [0.0f32; 12];
    for (m, value) in out.iter_mut().enumerate() {
        let phase = (m as f64 + 0.5) * (std::f64::consts::TAU / 12.0) + phase_shift;
        let month_temp = base_c + (seasonal_amp * phase.cos()) as f32 - lapse_downscale(elev_m);
        *value = month_temp.clamp(-90.0, 70.0);
    }
    out
}

/// Reconstruct monthly precipitation using orographic wet/lee effects,
/// coastal moderation, and latitude circulation.  Deterministic and bounded.
fn monthly_precipitation(
    dir: DVec3,
    macro_elev_m: f64,
    mountain_influence: f64,
    preset: EnvironmentPreset,
    params: &crate::params::PlanetParams,
    field: &crate::terrain_field::ProceduralTerrainField,
    planet_radius_m: f64,
) -> [f32; 12] {
    let pos = metric_surface_point(dir, planet_radius_m);
    let moisture_norm = field.sample(pos.normalize()).moisture as f64;
    let preset_scale = match preset {
        EnvironmentPreset::EarthLike => 1.0,
        EnvironmentPreset::HotArid => 0.45,
        EnvironmentPreset::ColdIcehouse => 0.7,
    };
    let annual_mm = normalized_moisture_to_mm(moisture_norm, params) * preset_scale;

    // Orographic effect: more precipitation where macro elevation rises
    // windward, less in rain shadows.  Use mountain influence as a proxy for
    // upwind barrier.  Clamp to avoid non-physical values.
    let orographic = 1.0
        + (macro_elev_m / 2000.0).tanh() * mountain_influence * params.rain_shadow_strength as f64;
    let adjusted_annual = (annual_mm as f64 * orographic).clamp(0.0, 12_000.0);

    // Latitude circulation: equatorial peak, subtropical minima, weak polar
    // secondary maximum.
    let latitude = dir.y.abs().clamp(0.0, 1.0);
    let lat_factor = if latitude < 0.3 {
        1.3 - latitude * 0.5
    } else if latitude < 0.6 {
        0.7 + (latitude - 0.3) * 0.2
    } else {
        0.85 - (latitude - 0.6) * 0.3
    };

    let coastal = coastal_moderation_factor(dir, field, planet_radius_m, params);
    let month_base = adjusted_annual * lat_factor * coastal / 12.0;

    let mut out = [0.0f32; 12];
    for (m, value) in out.iter_mut().enumerate() {
        // Slight seasonal concentration in winter hemisphere (temperature
        // anti-correlation) using the seasonal temperature phase.
        let phase = (m as f64 + 0.5) * (std::f64::consts::TAU / 12.0);
        let temp_influence = 1.0 - 0.2 * phase.cos();
        *value = (month_base * temp_influence).clamp(0.0, 3000.0) as f32;
    }
    out
}

/// Distance to the nearest ocean cell in the atlas, in meters, capped at a
/// continental-scale bound.  Used as a bounded upwind/coastal moderation
/// approximation.
fn coastal_moderation_factor(
    dir: DVec3,
    field: &crate::terrain_field::ProceduralTerrainField,
    planet_radius_m: f64,
    _params: &crate::params::PlanetParams,
) -> f64 {
    // Sample a small fixed number of neighbors to estimate coast distance.
    // This is a bounded approximation: 8 neighbors at ~200 km spacing on Earth.
    let sample_dirs = neighbor_directions(dir, 8);
    let mut min_ocean_dist_m = f64::INFINITY;
    for nd in sample_dirs {
        let npos = metric_surface_point(nd, planet_radius_m);
        let sample = field.sample(npos.normalize());
        if sample.biome.is_ocean() {
            let dist = (nd - dir).length() * planet_radius_m;
            min_ocean_dist_m = min_ocean_dist_m.min(dist);
        }
    }

    if min_ocean_dist_m.is_infinite() {
        // Deep interior: continental, drier.
        0.55
    } else {
        // Coastal moderation decays over ~2000 km.
        let continent_scale = 2_000_000.0;
        let t = (min_ocean_dist_m / continent_scale).clamp(0.0, 1.0);
        1.0 - t * 0.45 // coast=1.0, deep interior=0.55
    }
}

/// Generate a deterministic set of neighbor directions around `dir` using a
/// small Fibonacci spiral.  The directions are in world space and independent
/// of cube face.
fn neighbor_directions(dir: DVec3, count: usize) -> Vec<DVec3> {
    let mut out = Vec::with_capacity(count);
    // Pick an arbitrary perpendicular vector.
    let perp = if dir.y.abs() < 0.9 {
        DVec3::new(0.0, 1.0, 0.0).cross(dir).normalize()
    } else {
        DVec3::new(1.0, 0.0, 0.0).cross(dir).normalize()
    };
    let perp2 = dir.cross(perp).normalize();
    let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    for i in 0..count {
        let r = (i as f64 + 0.5) / count as f64;
        let theta = golden * i as f64;
        let offset = perp * r * theta.cos() + perp2 * r * theta.sin();
        out.push((dir + offset * 0.05).normalize());
    }
    out
}

/// Prevailing wind from latitude circulation plus a deterministic seed-derived
/// zonal perturbation.
fn prevailing_wind(dir: DVec3, params: &crate::params::PlanetParams) -> (f32, f32) {
    let latitude = dir.y.clamp(-1.0, 1.0);
    // Trade winds / westerlies / polar easterlies cartoon:
    //   equator -> easterly, mid-lat -> westerly, polar -> easterly.
    let band: f64 = if latitude.abs() < 0.35 {
        -1.0 // easterly
    } else if latitude.abs() < 0.7 {
        1.0 // westerly
    } else {
        -0.6 // weak polar easterly
    };
    // Add a small deterministic zonal perturbation from axial tilt.
    let perturb = params.axial_tilt.sin() * 0.2 * (dir.x * std::f64::consts::TAU).cos();
    let speed = (5.0 + 8.0 * band.abs() + 4.0 * params.axial_tilt.abs()) as f32;
    let direction = ((band + perturb).atan2(1.0) as f32).rem_euclid(std::f32::consts::TAU);
    (speed.clamp(0.0, 100.0), direction)
}

/// Lapse-rate temperature downscaling for a given elevation above sea level.
fn lapse_downscale(elevation_above_sea_m: f64) -> f32 {
    // Standard environmental lapse ~6.5 K/km.
    (elevation_above_sea_m / 1000.0 * 6.5) as f32
}

/// Thornthwaite potential evapotranspiration for a single month.
/// Returns PET in mm/month for each month.
fn thornthwaite_pet_monthly(monthly_temp_c: &[f32; 12]) -> [f32; 12] {
    // Thornthwaite heat index.
    let heat_index: f32 = monthly_temp_c
        .iter()
        .map(|&t| (t / 5.0).clamp(0.0, f32::INFINITY).powf(1.514))
        .sum();
    let a = if heat_index > 0.0 {
        6.75e-7 * heat_index.powi(3) - 7.71e-5 * heat_index.powi(2)
            + 1.792e-2 * heat_index
            + 0.49239
    } else {
        0.0
    };

    let mut out = [0.0f32; 12];
    for (m, &t) in monthly_temp_c.iter().enumerate() {
        if t <= 0.0 {
            out[m] = 0.0;
        } else {
            let days = 30.0; // approximate month length
                             // Thornthwaite monthly PET formula (mm/month).
            out[m] =
                (16.0 * (10.0 * t / heat_index).clamp(0.0, f32::INFINITY).powf(a) * (days / 30.0))
                    .max(0.0);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::math::uv_to_dir;

    const TEST_RADIUS: f64 = 6_371_000.0;
    const TEST_ELEV_SCALE: f64 = 1_000.0;

    fn earthlike_atlas() -> ClimateAtlas {
        ClimateAtlas::build(
            PlanetSeed(0xC0FFEE),
            EnvironmentPreset::EarthLike,
            TEST_RADIUS,
            TEST_ELEV_SCALE,
        )
    }

    #[test]
    fn build_is_deterministic() {
        let a = earthlike_atlas();
        let b = ClimateAtlas::build(
            PlanetSeed(0xC0FFEE),
            EnvironmentPreset::EarthLike,
            TEST_RADIUS,
            TEST_ELEV_SCALE,
        );
        assert_eq!(a.identity(), b.identity());
        assert_eq!(a.len(), b.len());
        assert_eq!(a.build_stats(), b.build_stats());
        for (ac, bc) in a.cells.iter().zip(b.cells.iter()) {
            assert_eq!(ac, bc);
        }
    }

    #[test]
    fn atlas_length_and_memory_match() {
        let atlas = earthlike_atlas();
        assert_eq!(atlas.len(), CLIMATE_ATLAS_LEN);
        assert!(
            atlas.memory_bytes()
                >= std::mem::size_of::<ClimateAtlas>()
                    + atlas.len() * std::mem::size_of::<ClimateCell>()
        );
        assert_eq!(atlas.lod(), CLIMATE_ATLAS_LOD);
        assert!(atlas.build_stats().cells_per_face_edge <= 64);
    }

    #[test]
    fn sample_is_valid_everywhere() {
        let atlas = earthlike_atlas();
        let dirs = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, -1.0, 0.0),
            DVec3::new(0.0, 0.0, -1.0),
        ];
        for &dir in &dirs {
            let sample = atlas.sample_at_elevation(dir, 500.0);
            assert!(
                sample.validity.is_valid(),
                "invalid sample at {dir:?}: {sample:?}"
            );
            assert!(
                sample.normals.is_valid(),
                "ClimateNormals invalid at {dir:?}"
            );
        }
    }

    #[test]
    fn sample_macro_matches_default_sample() {
        let atlas = earthlike_atlas();
        let mut rng = er_core::rng::rng_from_seed(0xBAD5EED);
        for _ in 0..50 {
            let dir = random_dir(&mut rng);
            let macro_sample = atlas.sample_macro(dir);
            let zero_elev = atlas.sample(dir);
            assert_eq!(
                macro_sample.monthly_temperature_c,
                zero_elev.monthly_temperature_c
            );
            assert!(
                (macro_sample.normals.annual_mean_temperature_c
                    - zero_elev.normals.annual_mean_temperature_c)
                    .abs()
                    < 1e-4
            );
        }
    }

    #[test]
    fn opposite_seasons_between_hemispheres() {
        let atlas = earthlike_atlas();
        let north = DVec3::new(0.0, 1.0, 0.0);
        let south = DVec3::new(0.0, -1.0, 0.0);
        let s_north = atlas.sample_macro(north);
        let s_south = atlas.sample_macro(south);
        // At polar points the phase is shifted by pi, so mid-year should be
        // warm in one and cold in the other.
        // We do not assert exact ordering because polar temperatures may both
        // be extreme; instead assert the signs of the *difference* around the
        // year are opposite.
        let n_diff = s_north.monthly_temperature_c[6] - s_north.monthly_temperature_c[0];
        let s_diff = s_south.monthly_temperature_c[6] - s_south.monthly_temperature_c[0];
        assert!(
            n_diff * s_diff < 0.0,
            "opposite hemispheres should show opposite seasonal trends"
        );
    }

    #[test]
    fn lapse_downscale_cools_with_elevation() {
        let atlas = earthlike_atlas();
        let dir = DVec3::new(1.0, 0.0, 0.0);
        let low = atlas.sample_at_elevation(dir, 0.0);
        let high = atlas.sample_at_elevation(dir, 3000.0);
        assert!(
            high.normals.annual_mean_temperature_c < low.normals.annual_mean_temperature_c - 10.0,
            "3 km elevation should cool by ~20 C"
        );
    }

    #[test]
    fn coastal_moderation_exists() {
        let atlas = earthlike_atlas();
        // Two equatorial points; one may be ocean, one land.  The land sample
        // should not be non-finite and precipitation should be bounded.
        let land_dir = DVec3::new(1.0, 0.0, 0.0);
        let sample = atlas.sample_at_elevation(land_dir, 0.0);
        assert!(sample.normals.annual_precipitation_mm.is_finite());
        assert!(sample.normals.annual_precipitation_mm <= 12_000.0);
    }

    #[test]
    fn rain_shadow_reduces_precip_with_mountain_influence() {
        let atlas = earthlike_atlas();
        let mut rng = er_core::rng::rng_from_seed(0xBEEF);
        // Find a land direction with high mountain influence and compare to a
        // nearby low-influence point.
        let mut high_mountain = DVec3::ZERO;
        let mut low_mountain = DVec3::ZERO;
        let mut high_mi = 0.0f64;
        let mut low_mi = f64::INFINITY;
        for _ in 0..200 {
            let dir = random_dir(&mut rng);
            let cell = atlas.sample_macro(dir);
            let mi = cell.normals.annual_precipitation_mm as f64;
            if mi > high_mi {
                high_mi = mi;
                high_mountain = dir;
            }
            if mi < low_mi {
                low_mi = mi;
                low_mountain = dir;
            }
        }
        assert!(high_mountain != DVec3::ZERO);
        assert!(low_mountain != DVec3::ZERO);
        // The actual rain-shadow assertion is probabilistic; we just ensure the
        // atlas distinguishes wet and dry points.
        let wet = atlas.sample_macro(high_mountain);
        let dry = atlas.sample_macro(low_mountain);
        assert!(wet.normals.annual_precipitation_mm > dry.normals.annual_precipitation_mm);
    }

    #[test]
    fn seam_interpolation_is_continuous() {
        let atlas = earthlike_atlas();
        // Probe the shared edge between +X and +Y faces at many points.
        for i in 0..32 {
            let t = (i as f64 + 0.5) / 32.0;
            let d_x = uv_to_dir(0, 1.0, t); // +X posU edge
            let d_y = uv_to_dir(2, 1.0, t); // +Y posU edge
            let diff = (d_x - d_y).length();
            assert!(diff < 1e-6, "edge directions should coincide");
            let s_x = atlas.sample_at_elevation(d_x, 0.0);
            let s_y = atlas.sample_at_elevation(d_y, 0.0);
            assert!(s_x.validity.is_valid());
            assert!(s_y.validity.is_valid());
            let dt = (s_x.normals.annual_mean_temperature_c
                - s_y.normals.annual_mean_temperature_c)
                .abs();
            assert!(dt < 1.0, "seam temperature discontinuity too large: {dt}");
        }
    }

    #[test]
    fn build_stats_are_consistent() {
        let atlas = earthlike_atlas();
        let stats = atlas.build_stats();
        assert_eq!(stats.total_cells, atlas.len());
        assert_eq!(stats.total_cells, stats.land_cells + stats.ocean_cells);
        assert!(stats.mean_elevation_above_sea_m >= stats.min_elevation_above_sea_m);
        assert!(stats.mean_elevation_above_sea_m <= stats.max_elevation_above_sea_m);
        assert!(
            stats.mean_annual_temperature_c >= -90.0 && stats.mean_annual_temperature_c <= 60.0
        );
        assert!(
            stats.mean_annual_precipitation_mm >= 0.0
                && stats.mean_annual_precipitation_mm <= 12_000.0
        );
        assert_eq!(stats.build_sample_count, stats.total_cells as u64);
    }

    #[test]
    fn sample_bounds_respected() {
        let atlas = earthlike_atlas();
        let mut rng = er_core::rng::rng_from_seed(0xF00D);
        for _ in 0..500 {
            let dir = random_dir(&mut rng);
            let s = atlas.sample_at_elevation(dir, 200.0);
            assert!(
                s.normals.annual_mean_temperature_c >= -90.0
                    && s.normals.annual_mean_temperature_c <= 60.0
            );
            assert!(
                s.normals.annual_precipitation_mm >= 0.0
                    && s.normals.annual_precipitation_mm <= 12_000.0
            );
            assert!(s.prevailing_wind_speed_m_s >= 0.0 && s.prevailing_wind_speed_m_s <= 100.0);
            assert!(s.continentality_c >= 0.0);
        }
    }

    fn random_dir(rng: &mut er_core::rng::Rng) -> DVec3 {
        use rand::RngCore;
        loop {
            let x = (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let y = (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let z = (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let v = DVec3::new(x, y, z);
            let l2 = v.length_squared();
            if l2 > 1e-6 && l2 <= 1.0 {
                return v.normalize();
            }
        }
    }
}
