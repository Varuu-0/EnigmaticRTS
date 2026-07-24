//! Deterministic M6.3 sphere-native substrate geology atlas.
//!
//! The atlas bakes broad substrate provinces onto the spherified cube at a fixed
//! LOD (<= 64 cells/face edge), then answers arbitrary directional queries by
//! interpolating across cube-face boundaries.  It is immutable after build, does
//! no I/O, uses no locks, no unsafe, no runtime clock, and no chunk/LOD/camera
//! dependence.  Province IDs are stable Voronoi labels derived from the planet
//! seed; substrate classes are derived from the existing metric landform and
//! climate fields, deliberately aligned with tectonic/volcanic terrain
//! structure without claiming full plate tectonics.

use crate::biome::{moisture_at, temperature_at};
use crate::elevation::{elevation_params, metric_landform_sample, ElevationNoise};
use crate::params::{climate_noise_metric, planet_params, PlanetParams};
use crate::terrain_space::metric_surface_point;
use er_core::math::{cell_neighbors, cell_to_dir, dir_to_cell, CellKey, FACE_COUNT};
use er_core::rng::{rng_child, rng_from_seed};
use er_core::seed::PlanetSeed;
use glam::DVec3;
use rand::RngCore;
use serde::Serialize;
use std::collections::HashSet;

/// Fixed atlas resolution: 6 faces, 32 cells per edge.  This satisfies the
/// "<= 64 cells/face edge" bound while keeping memory and build time small.
pub const GEOLOGY_ATLAS_LOD: u8 = 5;

/// Cells per face edge at the fixed atlas LOD.
pub const GEOLOGY_CELLS_PER_EDGE: u32 = 1u32 << GEOLOGY_ATLAS_LOD;

/// Total number of cell centers stored (6 faces * N * N).
pub const GEOLOGY_ATLAS_LEN: usize =
    FACE_COUNT * (GEOLOGY_CELLS_PER_EDGE as usize) * (GEOLOGY_CELLS_PER_EDGE as usize);

/// Number of deterministic province seed points on the sphere.  More seeds give
/// finer provinces; the number is fixed so `province_id` values are stable.
const PROVINCE_SEED_COUNT: usize = 256;

/// Stable substrate classes.  The `u8` discriminant is part of the API; do not
/// renumber variants without a version bump.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[repr(u8)]
pub enum SubstrateClass {
    OceanicSediment = 0,
    ContinentalSediment = 1,
    AlluvialDeposit = 2,
    GlacialTill = 3,
    VolcanicBasalt = 4,
    VolcanicAndesite = 5,
    VolcanicRhyolite = 6,
    GranitePlutonic = 7,
    MetamorphicSchist = 8,
    Sandstone = 9,
    Shale = 10,
    CarbonatePlatform = 11,
    Karst = 12,
    Regolith = 13,
    Laterite = 14,
}

impl SubstrateClass {
    pub const ALL: [Self; 15] = [
        Self::OceanicSediment,
        Self::ContinentalSediment,
        Self::AlluvialDeposit,
        Self::GlacialTill,
        Self::VolcanicBasalt,
        Self::VolcanicAndesite,
        Self::VolcanicRhyolite,
        Self::GranitePlutonic,
        Self::MetamorphicSchist,
        Self::Sandstone,
        Self::Shale,
        Self::CarbonatePlatform,
        Self::Karst,
        Self::Regolith,
        Self::Laterite,
    ];

    pub const COUNT: usize = Self::ALL.len();

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OceanicSediment => "oceanic_sediment",
            Self::ContinentalSediment => "continental_sediment",
            Self::AlluvialDeposit => "alluvial_deposit",
            Self::GlacialTill => "glacial_till",
            Self::VolcanicBasalt => "volcanic_basalt",
            Self::VolcanicAndesite => "volcanic_andesite",
            Self::VolcanicRhyolite => "volcanic_rhyolite",
            Self::GranitePlutonic => "granite_plutonic",
            Self::MetamorphicSchist => "metamorphic_schist",
            Self::Sandstone => "sandstone",
            Self::Shale => "shale",
            Self::CarbonatePlatform => "carbonate_platform",
            Self::Karst => "karst",
            Self::Regolith => "regolith",
            Self::Laterite => "laterite",
        }
    }
}

/// Bitmask describing why a sample is considered valid or invalid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SubstrateSampleValidity {
    pub finite: bool,
    pub fractions_normalized: bool,
    pub fields_bounded: bool,
    pub class_resolved: bool,
}

impl SubstrateSampleValidity {
    pub const fn all_valid() -> Self {
        Self {
            finite: true,
            fractions_normalized: true,
            fields_bounded: true,
            class_resolved: true,
        }
    }

    pub fn is_valid(self) -> bool {
        self.finite && self.fractions_normalized && self.fields_bounded && self.class_resolved
    }
}

/// A complete substrate sample at a point on the sphere.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SubstrateSample {
    pub province_id: u32,
    pub class: SubstrateClass,
    pub grain_size_mm: f32,
    pub permeability: f32,
    pub erodibility: f32,
    pub carbonate_fraction: f32,
    pub silicate_fraction: f32,
    pub mineral_color_rgb: [f32; 3],
    pub volcanic_age_normalized: f32,
    pub validity: SubstrateSampleValidity,
}

/// Statistics produced by `GeologyAtlas::build`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeologyAtlasBuildStats {
    pub cells_per_face_edge: u32,
    pub total_cells: usize,
    pub province_count: usize,
    pub build_sample_count: u64,
    pub mean_grain_size_mm: f32,
    pub mean_permeability: f32,
    pub mean_erodibility: f32,
}

/// Identity fields sufficient to reproduce the atlas deterministically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeologyAtlasIdentity {
    pub seed: u64,
    pub planet_radius_m: u64,
    pub lod: u8,
}

/// Immutable geology atlas built from a planet seed and metric radius.
#[derive(Clone, Debug)]
pub struct GeologyAtlas {
    identity: GeologyAtlasIdentity,
    params: PlanetParams,
    planet_radius_m: f64,
    cells: Vec<GeologyCell>,
    province_seeds: Vec<DVec3>,
    build_stats: GeologyAtlasBuildStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct GeologyCell {
    dir: DVec3,
    elevation_m: f32,
    macro_elevation_m: f32,
    sea_level_m: f32,
    tectonic_belt: f32,
    mountain_influence: f32,
    erosion: f32,
    drainage: f32,
    valley_mask: f32,
    ridge_mask: f32,
    moisture: f32,
    temperature: f32,
    latitude: f32,
}

impl GeologyAtlas {
    /// Build the immutable atlas.  The call is deterministic and bounded:
    /// it samples exactly `6 * 2^(2*lod)` cell centers.
    pub fn build(seed: PlanetSeed, planet_radius_m: f64) -> Self {
        let params = planet_params(seed);
        let elev_params = elevation_params(seed);
        let noise = ElevationNoise::new_metric(&elev_params);
        let climate_noise = climate_noise_metric(&params, planet_radius_m);

        let province_seeds = province_seed_dirs(seed.0, PROVINCE_SEED_COUNT);
        let cells_per_edge = GEOLOGY_CELLS_PER_EDGE;
        let total = GEOLOGY_ATLAS_LEN;
        let mut cells = Vec::with_capacity(total);
        let mut province_ids_seen = HashSet::new();

        let mut grain_sum = 0.0f64;
        let mut perm_sum = 0.0f64;
        let mut erod_sum = 0.0f64;

        for face in 0u8..FACE_COUNT as u8 {
            for j in 0..cells_per_edge {
                for i in 0..cells_per_edge {
                    let key = CellKey {
                        face,
                        i,
                        j,
                        lod: GEOLOGY_ATLAS_LOD,
                    };
                    let dir = cell_to_dir(key);
                    let pos = metric_surface_point(dir, planet_radius_m);
                    let landform = metric_landform_sample(pos, &noise);
                    let moist = moisture_at(
                        pos,
                        landform.mountain_influence as f64,
                        &params,
                        &climate_noise,
                    );
                    let temp =
                        temperature_at(pos, landform.full_elevation, &params, &climate_noise);

                    let elevation_m = (landform.full_elevation * 1000.0) as f32;
                    let macro_elevation_m = (landform.macro_displacement * 1000.0) as f32;
                    let sea_level_m = (params.sea_level * 1000.0) as f32;

                    let province_id = nearest_province_id(dir, &province_seeds);
                    province_ids_seen.insert(province_id);

                    let cell = GeologyCell {
                        dir,
                        elevation_m,
                        macro_elevation_m,
                        sea_level_m,
                        tectonic_belt: landform.tectonic_belt,
                        mountain_influence: landform.mountain_influence as f32,
                        erosion: landform.erosion,
                        drainage: landform.drainage,
                        valley_mask: landform.valley_mask,
                        ridge_mask: landform.ridge_mask,
                        moisture: moist as f32,
                        temperature: temp as f32,
                        latitude: dir.y.abs() as f32,
                    };

                    let sample =
                        compute_substrate_sample(&cell, dir, province_id, &params, planet_radius_m);
                    grain_sum += sample.grain_size_mm as f64;
                    perm_sum += sample.permeability as f64;
                    erod_sum += sample.erodibility as f64;

                    cells.push(cell);
                }
            }
        }

        let total_f = total as f64;
        let stats = GeologyAtlasBuildStats {
            cells_per_face_edge: cells_per_edge,
            total_cells: total,
            province_count: province_ids_seen.len(),
            build_sample_count: total as u64,
            mean_grain_size_mm: if total > 0 {
                (grain_sum / total_f) as f32
            } else {
                0.0
            },
            mean_permeability: if total > 0 {
                (perm_sum / total_f) as f32
            } else {
                0.0
            },
            mean_erodibility: if total > 0 {
                (erod_sum / total_f) as f32
            } else {
                0.0
            },
        };

        Self {
            identity: GeologyAtlasIdentity {
                seed: seed.0,
                planet_radius_m: planet_radius_m as u64,
                lod: GEOLOGY_ATLAS_LOD,
            },
            params,
            planet_radius_m,
            cells,
            province_seeds,
            build_stats: stats,
        }
    }

    /// Return the deterministic identity of this atlas.
    pub fn identity(&self) -> GeologyAtlasIdentity {
        self.identity
    }

    /// Return the fixed LOD of the atlas.
    pub const fn lod(&self) -> u8 {
        GEOLOGY_ATLAS_LOD
    }

    /// Number of cells stored in the atlas.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Number of stored samples in the atlas (alias for `len`).
    pub fn sample_count(&self) -> usize {
        self.cells.len()
    }

    /// Estimated memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.cells.len() * std::mem::size_of::<GeologyCell>()
            + self.province_seeds.len() * std::mem::size_of::<DVec3>()
    }

    /// Return the build-time statistics.
    pub fn build_stats(&self) -> GeologyAtlasBuildStats {
        self.build_stats
    }

    /// Sample geology at an arbitrary world direction.  Interpolation is
    /// performed in world direction space using the four surrounding atlas cell
    /// centers, so cube-face seams are handled automatically.
    pub fn sample(&self, dir: DVec3) -> SubstrateSample {
        let dir = dir.normalize_or_zero();
        let interp = self.interpolate_direction(dir);
        let province_id = nearest_province_id(dir, &self.province_seeds);
        compute_substrate_sample(
            &interp,
            dir,
            province_id,
            &self.params,
            self.planet_radius_m,
        )
    }

    /// Blend a bounded two-ring neighborhood using only world-space angular
    /// distance. The neighborhood graph wraps across cube faces, so the same
    /// physical point cannot acquire face-local interpolation coordinates.
    fn interpolate_direction(&self, dir: DVec3) -> GeologyCell {
        let dir = dir.normalize_or_zero();
        let center = dir_to_cell(dir, GEOLOGY_ATLAS_LOD);
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
        Self::weighted_cell(&weighted, weight_sum, dir)
    }

    fn cell_at(&self, key: CellKey) -> GeologyCell {
        let n = GEOLOGY_CELLS_PER_EDGE as usize;
        let face = key.face as usize;
        debug_assert!(face < FACE_COUNT);
        debug_assert!((key.i as usize) < n);
        debug_assert!((key.j as usize) < n);
        let index = (face * n + key.j as usize) * n + key.i as usize;
        self.cells[index]
    }

    fn weighted_cell(cells: &[(GeologyCell, f64)], weight_sum: f64, dir: DVec3) -> GeologyCell {
        let blend = |field: fn(&GeologyCell) -> f32| -> f32 {
            (cells
                .iter()
                .map(|(cell, weight)| field(cell) as f64 * weight)
                .sum::<f64>()
                / weight_sum) as f32
        };
        GeologyCell {
            dir,
            elevation_m: blend(|c| c.elevation_m),
            macro_elevation_m: blend(|c| c.macro_elevation_m),
            sea_level_m: blend(|c| c.sea_level_m),
            tectonic_belt: blend(|c| c.tectonic_belt),
            mountain_influence: blend(|c| c.mountain_influence),
            erosion: blend(|c| c.erosion),
            drainage: blend(|c| c.drainage),
            valley_mask: blend(|c| c.valley_mask),
            ridge_mask: blend(|c| c.ridge_mask),
            moisture: blend(|c| c.moisture),
            temperature: blend(|c| c.temperature),
            latitude: blend(|c| c.latitude),
        }
    }
}

// ---------------------------------------------------------------------------
// Province generation
// ---------------------------------------------------------------------------

fn province_seed_dirs(seed: u64, count: usize) -> Vec<DVec3> {
    let mut rng = rng_from_seed(seed);
    let mut r = rng_child(&mut rng, 0x6d_33_6f_63_6b); // "m3ock" salt
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let x = u2f(r.next_u64());
        let y = u2f(r.next_u64());
        let z = u2f(r.next_u64());
        let v = DVec3::new(x, y, z);
        let l2 = v.length_squared();
        if l2 > 1e-6 && l2 < 1.0 {
            out.push(v.normalize());
        }
    }
    out
}

fn nearest_province_id(dir: DVec3, seeds: &[DVec3]) -> u32 {
    if seeds.is_empty() {
        return 0;
    }
    let dir = dir.normalize();
    let mut best = 0usize;
    let mut best_dot = -1.0f64;
    for (i, seed) in seeds.iter().enumerate() {
        let dot = dir.dot(*seed);
        if dot > best_dot {
            best_dot = dot;
            best = i;
        }
    }
    best as u32
}

// ---------------------------------------------------------------------------
// Substrate sample synthesis
// ---------------------------------------------------------------------------

fn compute_substrate_sample(
    fields: &GeologyCell,
    dir: DVec3,
    province_id: u32,
    params: &PlanetParams,
    planet_radius_m: f64,
) -> SubstrateSample {
    let elev = fields.elevation_m as f64;
    let sea_level = fields.sea_level_m as f64;
    let depth = sea_level - elev;
    let is_ocean = elev < sea_level;
    let macro_elev = fields.macro_elevation_m as f64;
    let tectonic = fields.tectonic_belt as f64;
    let mountain = fields.mountain_influence as f64;
    let erosion = fields.erosion as f64;
    let drainage = fields.drainage as f64;
    let valley = fields.valley_mask as f64;
    let _ridge = fields.ridge_mask as f64;
    let moisture = fields.moisture as f64;
    let temp = fields.temperature as f64;
    let latitude = fields.latitude as f64;

    let volcanic_threshold_m = params.volcanic_threshold * 1000.0;
    let high_alt_threshold_m = params.high_alt_threshold * 1000.0;

    let volcanic_activity = if macro_elev > volcanic_threshold_m {
        (tectonic * 0.8 + 0.2).clamp(0.0, 1.0)
    } else {
        (tectonic * 0.5).clamp(0.0, 1.0)
    };

    // Class selection is deliberately heuristic: broad sphere-native provinces
    // aligned with tectonic/volcanic terrain structure, not a full plate model.
    let class = if is_ocean {
        if depth < 200.0 && temp > 0.5 && latitude < 0.4 {
            SubstrateClass::CarbonatePlatform
        } else {
            SubstrateClass::OceanicSediment
        }
    } else if volcanic_activity > 0.65 {
        if macro_elev > volcanic_threshold_m * 1.2 {
            SubstrateClass::VolcanicRhyolite
        } else if macro_elev > volcanic_threshold_m {
            SubstrateClass::VolcanicAndesite
        } else {
            SubstrateClass::VolcanicBasalt
        }
    } else if tectonic > 0.65 && mountain > 0.5 {
        SubstrateClass::MetamorphicSchist
    } else if mountain > 0.7 && macro_elev > high_alt_threshold_m {
        SubstrateClass::GranitePlutonic
    } else if latitude > 0.75 && macro_elev > 1000.0 {
        SubstrateClass::GlacialTill
    } else if valley > 0.6 && drainage > 0.5 {
        SubstrateClass::AlluvialDeposit
    } else if temp > 0.65 && moisture > 0.65 {
        SubstrateClass::Laterite
    } else if temp > 0.5 && moisture < 0.25 {
        SubstrateClass::Sandstone
    } else if moisture > 0.65 && temp > 0.4 {
        SubstrateClass::Shale
    } else if depth > -300.0 && depth < 0.0 && temp > 0.4 {
        SubstrateClass::CarbonatePlatform
    } else if moisture > 0.45 && temp > 0.45 && erosion > 0.5 {
        SubstrateClass::Karst
    } else if macro_elev < sea_level + 100.0 {
        SubstrateClass::ContinentalSediment
    } else {
        SubstrateClass::Regolith
    };

    let (base_grain, base_perm, base_erod, base_carb, base_sil, base_color) =
        class_properties(class);

    let jitter = province_jitter(province_id, 0x9e3779b97f4a7c15);
    let grain_size_mm = (base_grain * (0.85 + jitter * 0.3)).clamp(0.001, 5000.0);

    let terrain_erosion = (erosion as f32 - 0.5) * 0.1;
    let terrain_drainage = (drainage as f32 - 0.5) * 0.1;
    let permeability =
        (base_perm + terrain_drainage * 0.15 - terrain_erosion * 0.1).clamp(0.0, 1.0);
    let erodibility = (base_erod + terrain_erosion * 0.15).clamp(0.0, 1.0);

    let carbonate_fraction = base_carb.clamp(0.0, 1.0);
    let silicate_fraction = base_sil.clamp(0.0, 1.0);

    let color = [
        (base_color[0] + (jitter - 0.5) * 0.08).clamp(0.0, 1.0),
        (base_color[1] + (province_jitter(province_id, 0x85ebca6b) - 0.5) * 0.08).clamp(0.0, 1.0),
        (base_color[2] + (province_jitter(province_id, 0xc2b2ae35) - 0.5) * 0.08).clamp(0.0, 1.0),
    ];

    let volcanic_age = {
        let age_noise = province_jitter(province_id, 0x7f4a7c15) as f64;
        ((1.0 - volcanic_activity) * 0.8 + age_noise * 0.2).clamp(0.0, 1.0)
    };

    let sample = SubstrateSample {
        province_id,
        class,
        grain_size_mm,
        permeability,
        erodibility,
        carbonate_fraction,
        silicate_fraction,
        mineral_color_rgb: color,
        volcanic_age_normalized: volcanic_age as f32,
        validity: SubstrateSampleValidity::all_valid(),
    };

    let validity = validate(&sample, dir, planet_radius_m);
    SubstrateSample { validity, ..sample }
}

fn class_properties(class: SubstrateClass) -> (f32, f32, f32, f32, f32, [f32; 3]) {
    // (grain_size_mm, permeability, erodibility, carbonate_fraction, silicate_fraction, rgb)
    match class {
        SubstrateClass::OceanicSediment => (0.05, 0.40, 0.50, 0.35, 0.35, [0.65, 0.70, 0.75]),
        SubstrateClass::ContinentalSediment => (0.10, 0.30, 0.55, 0.15, 0.55, [0.55, 0.45, 0.35]),
        SubstrateClass::AlluvialDeposit => (0.50, 0.60, 0.65, 0.10, 0.45, [0.50, 0.42, 0.30]),
        SubstrateClass::GlacialTill => (5.00, 0.15, 0.30, 0.05, 0.65, [0.75, 0.75, 0.78]),
        SubstrateClass::VolcanicBasalt => (1.00, 0.25, 0.40, 0.05, 0.75, [0.22, 0.20, 0.20]),
        SubstrateClass::VolcanicAndesite => (0.80, 0.20, 0.35, 0.05, 0.70, [0.30, 0.28, 0.27]),
        SubstrateClass::VolcanicRhyolite => (0.40, 0.15, 0.30, 0.05, 0.80, [0.55, 0.50, 0.48]),
        SubstrateClass::GranitePlutonic => (2.50, 0.12, 0.25, 0.05, 0.85, [0.60, 0.55, 0.50]),
        SubstrateClass::MetamorphicSchist => (1.50, 0.18, 0.30, 0.08, 0.78, [0.45, 0.42, 0.40]),
        SubstrateClass::Sandstone => (0.30, 0.55, 0.60, 0.10, 0.80, [0.70, 0.62, 0.45]),
        SubstrateClass::Shale => (0.02, 0.15, 0.45, 0.15, 0.55, [0.40, 0.40, 0.38]),
        SubstrateClass::CarbonatePlatform => (0.15, 0.45, 0.35, 0.75, 0.15, [0.80, 0.78, 0.72]),
        SubstrateClass::Karst => (0.20, 0.65, 0.50, 0.70, 0.20, [0.78, 0.76, 0.70]),
        SubstrateClass::Regolith => (0.08, 0.35, 0.55, 0.10, 0.50, [0.52, 0.45, 0.38]),
        SubstrateClass::Laterite => (0.06, 0.25, 0.50, 0.05, 0.55, [0.55, 0.30, 0.20]),
    }
}

/// Deterministic per-province jitter in [0, 1] from a 32-bit province id and a
/// 64-bit salt.  Uses a 64-bit splitmix-style mix so it needs no extra deps.
fn province_jitter(province_id: u32, salt: u64) -> f32 {
    let mut x = (province_id as u64).wrapping_add(salt);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    (x as f64 / u64::MAX as f64).clamp(0.0, 1.0) as f32
}

fn validate(
    sample: &SubstrateSample,
    _dir: DVec3,
    _planet_radius_m: f64,
) -> SubstrateSampleValidity {
    let finite = sample.grain_size_mm.is_finite()
        && sample.permeability.is_finite()
        && sample.erodibility.is_finite()
        && sample.carbonate_fraction.is_finite()
        && sample.silicate_fraction.is_finite()
        && sample.volcanic_age_normalized.is_finite()
        && sample.mineral_color_rgb.iter().all(|c| c.is_finite());
    let fractions_normalized = sample.carbonate_fraction >= 0.0
        && sample.carbonate_fraction <= 1.0
        && sample.silicate_fraction >= 0.0
        && sample.silicate_fraction <= 1.0
        && (sample.carbonate_fraction + sample.silicate_fraction) <= 1.0;
    let fields_bounded = sample.grain_size_mm >= 0.001
        && sample.grain_size_mm <= 5000.0
        && sample.permeability >= 0.0
        && sample.permeability <= 1.0
        && sample.erodibility >= 0.0
        && sample.erodibility <= 1.0
        && sample.volcanic_age_normalized >= 0.0
        && sample.volcanic_age_normalized <= 1.0
        && sample
            .mineral_color_rgb
            .iter()
            .all(|c| *c >= 0.0 && *c <= 1.0);
    let class_resolved = SubstrateClass::ALL.contains(&sample.class);

    SubstrateSampleValidity {
        finite,
        fractions_normalized,
        fields_bounded,
        class_resolved,
    }
}

fn u2f(u: u64) -> f64 {
    (u as f64 / u64::MAX as f64) * 2.0 - 1.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::math::uv_to_dir;
    use er_core::rng::Rng;

    const TEST_RADIUS: f64 = 6_371_000.0;

    fn earthlike_geology() -> GeologyAtlas {
        GeologyAtlas::build(PlanetSeed(0xC0FFEE), TEST_RADIUS)
    }

    fn random_dir(rng: &mut Rng) -> DVec3 {
        loop {
            let x = u2f(rng.next_u64());
            let y = u2f(rng.next_u64());
            let z = u2f(rng.next_u64());
            let v = DVec3::new(x, y, z);
            let l2 = v.length_squared();
            if l2 > 1e-6 && l2 < 1.0 {
                return v.normalize();
            }
        }
    }

    #[test]
    fn build_is_deterministic() {
        let a = GeologyAtlas::build(PlanetSeed(0xC0FFEE), TEST_RADIUS);
        let b = GeologyAtlas::build(PlanetSeed(0xC0FFEE), TEST_RADIUS);
        assert_eq!(a.identity(), b.identity());
        assert_eq!(a.build_stats(), b.build_stats());
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn sample_is_deterministic() {
        let a = earthlike_geology();
        let b = earthlike_geology();
        let mut rng = rng_from_seed(0xABCDEF);
        for _ in 0..200 {
            let dir = random_dir(&mut rng);
            let sa = a.sample(dir);
            let sb = b.sample(dir);
            assert_eq!(sa.province_id, sb.province_id, "province_id unstable");
            assert_eq!(sa.class, sb.class, "class unstable");
            assert_eq!(sa.grain_size_mm.to_bits(), sb.grain_size_mm.to_bits());
            assert_eq!(sa.permeability.to_bits(), sb.permeability.to_bits());
            assert_eq!(sa.erodibility.to_bits(), sb.erodibility.to_bits());
            assert_eq!(sa.validity, sb.validity);
        }
    }

    #[test]
    fn atlas_length_and_memory_match() {
        let atlas = earthlike_geology();
        assert_eq!(atlas.len(), GEOLOGY_ATLAS_LEN);
        assert!(
            atlas.memory_bytes()
                >= std::mem::size_of::<GeologyAtlas>()
                    + atlas.len() * std::mem::size_of::<GeologyCell>()
                    + atlas.province_seeds.len() * std::mem::size_of::<DVec3>()
        );
        assert_eq!(atlas.lod(), GEOLOGY_ATLAS_LOD);
        assert!(atlas.build_stats().cells_per_face_edge <= 64);
        assert_eq!(atlas.sample_count(), atlas.len());
    }

    #[test]
    fn sample_is_valid_everywhere() {
        let atlas = earthlike_geology();
        let dirs = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, -1.0, 0.0),
            DVec3::new(0.0, 0.0, -1.0),
        ];
        for &dir in &dirs {
            let s = atlas.sample(dir);
            assert!(s.validity.is_valid(), "invalid sample at {dir:?}: {s:?}");
        }
    }

    #[test]
    fn fractions_are_normalized() {
        let atlas = earthlike_geology();
        let mut rng = rng_from_seed(0xF00D);
        for _ in 0..500 {
            let dir = random_dir(&mut rng);
            let s = atlas.sample(dir);
            assert!(s.validity.is_valid());
            assert!(s.carbonate_fraction >= 0.0 && s.carbonate_fraction <= 1.0);
            assert!(s.silicate_fraction >= 0.0 && s.silicate_fraction <= 1.0);
            assert!(
                s.carbonate_fraction + s.silicate_fraction <= 1.0,
                "carbonate + silicate > 1 at {dir:?}: {} + {}",
                s.carbonate_fraction,
                s.silicate_fraction
            );
        }
    }

    #[test]
    fn fields_are_bounded() {
        let atlas = earthlike_geology();
        let mut rng = rng_from_seed(0xBAD);
        for _ in 0..500 {
            let dir = random_dir(&mut rng);
            let s = atlas.sample(dir);
            assert!(s.grain_size_mm >= 0.001 && s.grain_size_mm <= 5000.0);
            assert!(s.permeability >= 0.0 && s.permeability <= 1.0);
            assert!(s.erodibility >= 0.0 && s.erodibility <= 1.0);
            assert!(s.volcanic_age_normalized >= 0.0 && s.volcanic_age_normalized <= 1.0);
            assert!(s.mineral_color_rgb.iter().all(|c| *c >= 0.0 && *c <= 1.0));
        }
    }

    #[test]
    fn nearby_samples_are_autocorrelated() {
        let atlas = earthlike_geology();
        let mut rng = rng_from_seed(0xBEEF);
        let mut same_class = 0usize;
        let mut close_grain = 0usize;
        const TRIALS: usize = 100;

        for _ in 0..TRIALS {
            let dir = random_dir(&mut rng);
            let perp = if dir.y.abs() < 0.9 {
                DVec3::new(0.0, 1.0, 0.0).cross(dir).normalize()
            } else {
                DVec3::new(1.0, 0.0, 0.0).cross(dir).normalize()
            };
            let perp2 = dir.cross(perp).normalize();
            let offset = perp * 0.002 * (rng.next_u64() as f64 / u64::MAX as f64)
                + perp2 * 0.002 * (rng.next_u64() as f64 / u64::MAX as f64);
            let near = (dir + offset).normalize();

            let s1 = atlas.sample(dir);
            let s2 = atlas.sample(near);
            assert!(s1.validity.is_valid());
            assert!(s2.validity.is_valid());

            if s1.class == s2.class {
                same_class += 1;
            }
            if (s1.grain_size_mm - s2.grain_size_mm).abs() < 2.0 {
                close_grain += 1;
            }
        }

        assert!(
            same_class >= 70,
            "spatial autocorrelation too low: {same_class}/{TRIALS} same class"
        );
        assert!(
            close_grain >= 70,
            "grain size varies too rapidly: {close_grain}/{TRIALS} close"
        );
    }

    #[test]
    fn atlas_has_province_diversity() {
        let atlas = earthlike_geology();
        assert!(
            atlas.build_stats().province_count > 50,
            "too few provinces: {}",
            atlas.build_stats().province_count
        );
    }

    #[test]
    fn seam_samples_are_continuous() {
        let atlas = earthlike_geology();
        // Shared edge between +X and +Y faces.
        for i in 0..32 {
            let t = (i as f64 + 0.5) / 32.0;
            let d_x = uv_to_dir(0, 1.0, t); // +X posU edge
            let d_y = uv_to_dir(2, 1.0, t); // +Y posU edge
            assert!(
                (d_x - d_y).length() < 1e-6,
                "edge directions must coincide: i={i}"
            );
            let s_x = atlas.sample(d_x);
            let s_y = atlas.sample(d_y);
            assert!(s_x.validity.is_valid());
            assert!(s_y.validity.is_valid());
            assert_eq!(s_x.province_id, s_y.province_id, "province_id seam jump");
            assert_eq!(s_x.class, s_y.class, "class seam jump");
            assert!(
                (s_x.grain_size_mm - s_y.grain_size_mm).abs() < 1e-4,
                "grain size seam discontinuity at {i}: {} vs {}",
                s_x.grain_size_mm,
                s_y.grain_size_mm
            );
        }

        // Shared edge between +X negV and -Z posU.
        for j in 0..32 {
            let t = (j as f64 + 0.5) / 32.0;
            let d_xn = uv_to_dir(0, t, 0.0); // +X face, negV edge
            let d_zn = uv_to_dir(5, 1.0, t); // -Z face, posU edge
            assert!(
                (d_xn - d_zn).length() < 1e-6,
                "edge directions must coincide: j={j}"
            );
            let s_x = atlas.sample(d_xn);
            let s_z = atlas.sample(d_zn);
            assert!(s_x.validity.is_valid());
            assert!(s_z.validity.is_valid());
            assert_eq!(s_x.province_id, s_z.province_id, "province_id seam jump");
            assert_eq!(s_x.class, s_z.class, "class seam jump");
            assert!(
                (s_x.grain_size_mm - s_z.grain_size_mm).abs() < 1e-4,
                "grain size seam discontinuity at {j}: {} vs {}",
                s_x.grain_size_mm,
                s_z.grain_size_mm
            );
        }
    }

    #[test]
    fn different_seeds_produce_different_geology() {
        let a = GeologyAtlas::build(PlanetSeed(0xC0FFEE), TEST_RADIUS);
        let b = GeologyAtlas::build(PlanetSeed(0xDEADBEEF), TEST_RADIUS);
        assert_ne!(a.identity(), b.identity());
        let mut rng = rng_from_seed(0xBAD5EED);
        let mut any_diff = false;
        for _ in 0..100 {
            let dir = random_dir(&mut rng);
            if a.sample(dir).province_id != b.sample(dir).province_id {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "different seeds should produce different province IDs"
        );
    }
}
