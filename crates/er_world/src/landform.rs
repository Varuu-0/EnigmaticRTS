//! Deterministic bounded M6.3 sphere-native landform atlas.
//!
//! The atlas bakes geometric landform descriptors onto the spherified cube at a
//! fixed LOD (32 cells per face edge), then answers arbitrary directional
//! queries by world-space interpolation across cube-face boundaries.  It is
//! immutable after build, does no I/O, uses no locks, no unsafe, no runtime
//! clock, and does not depend on climate classes or learned data.
//!
//! All sampling is performed from 3-D metric surface coordinates, so every
//! probe is independent of cube-face topology and cube edges are continuous.

use crate::brushes::{BrushKind, BrushSet};
use crate::elevation::{
    elevation_params, metric_landform_sample, ElevationNoise, MetricLandformSample,
};
use crate::terrain_space::metric_surface_point;
use er_core::math::{cell_neighbors, cell_to_dir, dir_to_cell, CellKey, FACE_COUNT};
use er_core::seed::PlanetSeed;
use glam::{DVec3, Vec3};
use serde::Serialize;

/// Fixed atlas resolution: 6 faces, 32 cells per edge.  This satisfies the
/// "<= 64 cells/face edge" bound while keeping build cost deterministic and
/// modest (~6 k cells).
pub const LANDFORM_ATLAS_LOD: u8 = 5;

/// Cells per face edge at the fixed atlas LOD.
pub const LANDFORM_CELLS_PER_EDGE: u32 = 1u32 << LANDFORM_ATLAS_LOD;

/// Total number of cell centers stored (6 faces * N * N).
pub const LANDFORM_ATLAS_LEN: usize =
    FACE_COUNT * (LANDFORM_CELLS_PER_EDGE as usize) * (LANDFORM_CELLS_PER_EDGE as usize);

/// Algorithm version for identity / cache compatibility checks.
pub const LANDFORM_ATLAS_VERSION: u16 = 2;

/// Distance used for slope, aspect, and curvature probes (metres).
const SLOPE_PROBE_DISTANCE_M: f64 = 500.0;

/// Multi-scale relief/ruggedness/exposure ring distances (metres).
const RELIEF_DISTANCES_M: [f64; 4] = [200.0, 1_000.0, 5_000.0, 20_000.0];

/// Coastal search distances (metres).  The search caps at the last entry if no
/// ocean is found within the ring.
const COASTAL_DISTANCES_M: [f64; 8] = [
    1_000.0,
    5_000.0,
    20_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
    2_000_000.0,
];

/// Number of radial coastal-search directions.
const COASTAL_DIRECTION_COUNT: usize = 8;

/// Maximum reported coastal proximity (metres).  Used when no ocean is found
/// within the fixed coastal search radius.
const COASTAL_MAX_M: f32 = 20_000_000.0;

/// Roadmap landform classes.  The classifier is independent of climate / biome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum LandformClass {
    AbyssalPlain = 0,
    ShelfSlope = 1,
    CoastalPlain = 2,
    BeachDune = 3,
    Floodplain = 4,
    Plain = 5,
    RollingHill = 6,
    PlateauMesa = 7,
    Valley = 8,
    Canyon = 9,
    MountainSlope = 10,
    Summit = 11,
    Cliff = 12,
    KarstLike = 13,
    VolcanicEdifice = 14,
    CraterCaldera = 15,
}

impl LandformClass {
    pub const ALL: [Self; 16] = [
        Self::AbyssalPlain,
        Self::ShelfSlope,
        Self::CoastalPlain,
        Self::BeachDune,
        Self::Floodplain,
        Self::Plain,
        Self::RollingHill,
        Self::PlateauMesa,
        Self::Valley,
        Self::Canyon,
        Self::MountainSlope,
        Self::Summit,
        Self::Cliff,
        Self::KarstLike,
        Self::VolcanicEdifice,
        Self::CraterCaldera,
    ];

    pub const COUNT: usize = Self::ALL.len();

    pub const fn as_index(self) -> usize {
        self as usize
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AbyssalPlain => "abyssal_plain",
            Self::ShelfSlope => "shelf_slope",
            Self::CoastalPlain => "coastal_plain",
            Self::BeachDune => "beach_dune",
            Self::Floodplain => "floodplain",
            Self::Plain => "plain",
            Self::RollingHill => "rolling_hill",
            Self::PlateauMesa => "plateau_mesa",
            Self::Valley => "valley",
            Self::Canyon => "canyon",
            Self::MountainSlope => "mountain_slope",
            Self::Summit => "summit",
            Self::Cliff => "cliff",
            Self::KarstLike => "karst_like",
            Self::VolcanicEdifice => "volcanic_edifice",
            Self::CraterCaldera => "crater_caldera",
        }
    }
}

/// Statistics produced by `LandformAtlas::build`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LandformAtlasBuildStats {
    pub cells_per_face_edge: u32,
    pub total_cells: usize,
    pub land_cells: usize,
    pub ocean_cells: usize,
    pub mean_elevation_above_sea_m: f64,
    pub max_elevation_above_sea_m: f64,
    pub min_elevation_above_sea_m: f64,
    pub build_sample_count: u64,
}

/// Identity fields sufficient to reproduce the atlas deterministically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct LandformAtlasIdentity {
    pub seed: u64,
    pub planet_radius_m: u64,
    pub elevation_scale_m: u64,
    pub sea_level_mm: i64,
    pub version: u16,
}

/// Continuous geometric landform descriptors plus a primary classification and
/// confidence.  This is the M6.3 replacement for the legacy `LandformSample`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LandformSampleV2 {
    pub elevation_above_sea_m: f64,
    pub slope_degrees: f32,
    pub aspect_radians: f32,
    pub profile_curvature_per_m: f32,
    pub plan_curvature_per_m: f32,
    pub local_relief_m: f32,
    pub ruggedness: f32,
    pub exposure: f32,
    pub valley_position: f32,
    pub plateau_likelihood: f32,
    pub coastal_proximity_m: f32,
    pub primary_class: LandformClass,
    pub confidence: f32,
}

/// Compact per-cell baked landform data.  `dir` is stored only for the build
/// helpers; it is zeroed after interpolation so the cell does not retain a
/// stale direction in queries.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LandformCell {
    dir: DVec3,
    elevation_above_sea_m: f32,
    slope_degrees: f32,
    aspect_radians: f32,
    profile_curvature_per_m: f32,
    plan_curvature_per_m: f32,
    local_relief_m: f32,
    ruggedness: f32,
    exposure: f32,
    valley_position: f32,
    plateau_likelihood: f32,
    coastal_proximity_m: f32,
    tectonic_belt: f32,
    erosion: f32,
    ridge_mask: f32,
    valley_mask: f32,
    drainage: f32,
    mountain_influence: f32,
    brush_displacement: f32,
}

/// Immutable, deterministic, bounded landform atlas.
pub struct LandformAtlas {
    identity: LandformAtlasIdentity,
    cells: Vec<LandformCell>,
    build_stats: LandformAtlasBuildStats,
    brushes: BrushSet,
}

impl std::fmt::Debug for LandformAtlas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LandformAtlas")
            .field("identity", &self.identity)
            .field("cells", &self.cells.len())
            .field("build_stats", &self.build_stats)
            .field("lod", &LANDFORM_ATLAS_LOD)
            .finish()
    }
}

impl LandformAtlas {
    /// Build the immutable landform atlas.  The call is deterministic and
    /// bounded: it samples exactly `6 * 2^(2*lod)` cell centers, plus a fixed
    /// number of metric probes per cell.
    pub fn build(
        seed: PlanetSeed,
        planet_radius_m: f64,
        elevation_scale_m: f64,
        sea_level_m: f64,
    ) -> Self {
        let params = elevation_params(seed);
        let noise = ElevationNoise::new_metric(&params);
        let brushes = BrushSet::from_seed(params.seed as u32);

        let cells_per_edge = LANDFORM_CELLS_PER_EDGE;
        let total = LANDFORM_ATLAS_LEN;
        let mut cells = Vec::with_capacity(total);

        let mut land_cells = 0usize;
        let mut ocean_cells = 0usize;
        let mut elev_sum = 0.0f64;
        let mut elev_min = f64::INFINITY;
        let mut elev_max = f64::NEG_INFINITY;
        let mut sample_count = 0u64;

        for face in 0u8..FACE_COUNT as u8 {
            for j in 0..cells_per_edge {
                for i in 0..cells_per_edge {
                    let key = CellKey {
                        face,
                        i,
                        j,
                        lod: LANDFORM_ATLAS_LOD,
                    };
                    let dir = cell_to_dir(key);
                    let cell = build_cell(
                        dir,
                        &noise,
                        planet_radius_m,
                        elevation_scale_m,
                        sea_level_m,
                        &mut sample_count,
                    );

                    if cell.elevation_above_sea_m >= 0.0 {
                        land_cells += 1;
                    } else {
                        ocean_cells += 1;
                    }
                    elev_sum += cell.elevation_above_sea_m as f64;
                    elev_min = elev_min.min(cell.elevation_above_sea_m as f64);
                    elev_max = elev_max.max(cell.elevation_above_sea_m as f64);

                    cells.push(cell);
                }
            }
        }

        let total_f = total as f64;
        let stats = LandformAtlasBuildStats {
            cells_per_face_edge: cells_per_edge,
            total_cells: total,
            land_cells,
            ocean_cells,
            mean_elevation_above_sea_m: if total > 0 { elev_sum / total_f } else { 0.0 },
            max_elevation_above_sea_m: if total > 0 { elev_max } else { 0.0 },
            min_elevation_above_sea_m: if total > 0 { elev_min } else { 0.0 },
            build_sample_count: sample_count,
        };

        Self {
            identity: LandformAtlasIdentity {
                seed: seed.0,
                planet_radius_m: planet_radius_m as u64,
                elevation_scale_m: elevation_scale_m as u64,
                sea_level_mm: (sea_level_m * 1000.0) as i64,
                version: LANDFORM_ATLAS_VERSION,
            },
            cells,
            build_stats: stats,
            brushes,
        }
    }

    /// Return the deterministic identity of this atlas.
    pub fn identity(&self) -> LandformAtlasIdentity {
        self.identity
    }

    /// Return the fixed LOD of the atlas.
    pub const fn lod(&self) -> u8 {
        LANDFORM_ATLAS_LOD
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
        let base = std::mem::size_of::<Self>();
        let cells = self.cells.len() * std::mem::size_of::<LandformCell>();
        let brush_list = std::mem::size_of_val(self.brushes.brushes());
        // HashMap grid overhead: a small fixed estimate.
        let brush_overhead = 1024usize;
        base + cells + brush_list + brush_overhead
    }

    /// Return the build-time statistics.
    pub fn build_stats(&self) -> LandformAtlasBuildStats {
        self.build_stats
    }

    /// Convenience constructor alias using the same signature as the
    /// integration-test API.
    pub fn new(seed: PlanetSeed, planet_radius_m: f64, elevation_scale_m: f64) -> Self {
        Self::build(seed, planet_radius_m, elevation_scale_m, 0.0)
    }

    /// Sample landform at an arbitrary direction.  Interpolation is performed
    /// in world-space angular distance across the spherified cube, so cube-face
    /// seams are handled automatically.
    pub fn sample(&self, dir: DVec3) -> LandformSampleV2 {
        let dir = dir.normalize_or_zero();
        let cell = self.interpolate_direction(dir);
        let (primary_class, confidence) = self.classify(&cell, dir);
        LandformSampleV2 {
            elevation_above_sea_m: cell.elevation_above_sea_m as f64,
            slope_degrees: cell.slope_degrees,
            aspect_radians: cell.aspect_radians,
            profile_curvature_per_m: cell.profile_curvature_per_m,
            plan_curvature_per_m: cell.plan_curvature_per_m,
            local_relief_m: cell.local_relief_m,
            ruggedness: cell.ruggedness,
            exposure: cell.exposure,
            valley_position: cell.valley_position,
            plateau_likelihood: cell.plateau_likelihood,
            coastal_proximity_m: cell.coastal_proximity_m,
            primary_class,
            confidence,
        }
    }

    /// Classify a cell using the stored descriptors and the brush provenance at
    /// the query direction.  No climate or runtime data is consulted.
    fn classify(&self, cell: &LandformCell, dir: DVec3) -> (LandformClass, f32) {
        let dir_f = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32).normalize();
        let nearest = nearest_brush(&self.brushes, dir_f);
        let scores = class_scores(cell, nearest);

        let mut best_class = LandformClass::Plain;
        let mut best_score = -1.0f32;
        let mut second_score = -1.0f32;
        for (index, &score) in scores.iter().enumerate() {
            if score > best_score {
                second_score = best_score;
                best_score = score;
                best_class = LandformClass::ALL[index];
            } else if score > second_score {
                second_score = score;
            }
        }

        if best_score <= 0.0 {
            best_class = if cell.elevation_above_sea_m < 0.0 {
                LandformClass::AbyssalPlain
            } else {
                LandformClass::Plain
            };
            best_score = 0.0;
            second_score = 0.0;
        }

        let confidence =
            ((best_score - second_score) / (best_score + second_score + 1.0e-6)).clamp(0.0, 1.0);
        (best_class, confidence)
    }

    /// Blend a bounded two-ring neighborhood using only world-space angular
    /// distance.  The neighborhood graph wraps across cube faces, so the same
    /// physical point cannot acquire face-local interpolation coordinates.
    fn interpolate_direction(&self, dir: DVec3) -> LandformCell {
        let dir = dir.normalize_or_zero();
        let center = dir_to_cell(dir, LANDFORM_ATLAS_LOD);
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

    fn cell_at(&self, key: CellKey) -> LandformCell {
        let n = LANDFORM_CELLS_PER_EDGE as usize;
        let face = key.face as usize;
        debug_assert!(face < FACE_COUNT);
        debug_assert!((key.i as usize) < n);
        debug_assert!((key.j as usize) < n);
        let index = (face * n + key.j as usize) * n + key.i as usize;
        self.cells[index]
    }

    fn weighted_cell(cells: &[(LandformCell, f64)], weight_sum: f64) -> LandformCell {
        let _ = weight_sum; // captured by closures; explicit read silences lint
        let blend = |field: fn(&LandformCell) -> f32| -> f32 {
            (cells
                .iter()
                .map(|(cell, weight)| field(cell) as f64 * weight)
                .sum::<f64>()
                / weight_sum) as f32
        };

        LandformCell {
            dir: DVec3::ZERO,
            elevation_above_sea_m: blend(|c| c.elevation_above_sea_m),
            slope_degrees: blend(|c| c.slope_degrees),
            aspect_radians: blend_angle(cells, weight_sum, |c| c.aspect_radians),
            profile_curvature_per_m: blend(|c| c.profile_curvature_per_m),
            plan_curvature_per_m: blend(|c| c.plan_curvature_per_m),
            local_relief_m: blend(|c| c.local_relief_m),
            ruggedness: blend(|c| c.ruggedness),
            exposure: blend(|c| c.exposure),
            valley_position: blend(|c| c.valley_position),
            plateau_likelihood: blend(|c| c.plateau_likelihood),
            coastal_proximity_m: blend(|c| c.coastal_proximity_m),
            tectonic_belt: blend(|c| c.tectonic_belt),
            erosion: blend(|c| c.erosion),
            ridge_mask: blend(|c| c.ridge_mask),
            valley_mask: blend(|c| c.valley_mask),
            drainage: blend(|c| c.drainage),
            mountain_influence: blend(|c| c.mountain_influence),
            brush_displacement: blend(|c| c.brush_displacement),
        }
    }
}

// -----------------------------------------------------------------------------
// Private geometry / sampling helpers
// -----------------------------------------------------------------------------

/// Build an orthonormal tangent frame for a unit direction.  `east` and `north`
/// lie on the tangent plane and are continuous except at a handled polar
/// singularity.
fn tangent_frame(dir: DVec3) -> (DVec3, DVec3) {
    let dir = dir.normalize();
    let east = if dir.y.abs() < 0.9 {
        DVec3::new(0.0, 1.0, 0.0).cross(dir).normalize()
    } else {
        DVec3::new(1.0, 0.0, 0.0).cross(dir).normalize()
    };
    let north = dir.cross(east).normalize();
    (east, north)
}

/// Rotate a unit vector around a (not necessarily unit) axis by a signed angle.
fn rotate(dir: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    let axis = axis.normalize();
    let c = angle.cos();
    let s = angle.sin();
    let d = axis.dot(dir);
    let rotated = dir * c + axis.cross(dir) * s + axis * (d * (1.0 - c));
    rotated.normalize()
}

/// Sample elevation (in metres) at a metric offset from `dir` along the tangent
/// plane direction `(sx, sy)` in east/north coordinates.
#[allow(clippy::too_many_arguments)]
fn sample_elevation_m(
    dir: DVec3,
    east: &DVec3,
    north: &DVec3,
    sx: f64,
    sy: f64,
    dist: f64,
    radius: f64,
    elev_scale: f64,
    sample: &mut impl FnMut(DVec3) -> MetricLandformSample,
) -> f64 {
    let scale = (sx * sx + sy * sy).sqrt().max(1e-12);
    let axis = (*east * sx + *north * sy).normalize();
    let angle = dist * scale / radius;
    let offset_dir = rotate(dir, axis, angle);
    sample(offset_dir * radius).full_elevation * elev_scale
}

/// Build a single atlas cell.  All noise is evaluated through the existing
/// `metric_landform_sample` path so the result is bit-identical with the
/// composed terrain field.
fn build_cell(
    dir: DVec3,
    noise: &ElevationNoise,
    radius: f64,
    elev_scale: f64,
    sea_level: f64,
    counter: &mut u64,
) -> LandformCell {
    let mut sample = |pos: DVec3| {
        *counter += 1;
        metric_landform_sample(pos, noise)
    };

    let pos = metric_surface_point(dir, radius);
    let center_lf = sample(pos);
    let center_elev_m = center_lf.full_elevation * elev_scale;
    let elevation_above_sea_m = center_elev_m - sea_level;
    let (east, north) = tangent_frame(dir);

    // Slope and aspect from cardinal central differences.
    let d0 = SLOPE_PROBE_DISTANCE_M;
    let z_e = sample_elevation_m(
        dir,
        &east,
        &north,
        1.0,
        0.0,
        d0,
        radius,
        elev_scale,
        &mut sample,
    );
    let z_w = sample_elevation_m(
        dir,
        &east,
        &north,
        -1.0,
        0.0,
        d0,
        radius,
        elev_scale,
        &mut sample,
    );
    let z_n = sample_elevation_m(
        dir,
        &east,
        &north,
        0.0,
        1.0,
        d0,
        radius,
        elev_scale,
        &mut sample,
    );
    let z_s = sample_elevation_m(
        dir,
        &east,
        &north,
        0.0,
        -1.0,
        d0,
        radius,
        elev_scale,
        &mut sample,
    );
    let g_east = (z_e - z_w) / (2.0 * d0);
    let g_north = (z_n - z_s) / (2.0 * d0);
    let gradient = (g_east * g_east + g_north * g_north).sqrt();
    let slope_deg = gradient.atan().to_degrees().clamp(0.0, 90.0) as f32;
    let aspect = g_north.atan2(g_east).rem_euclid(std::f64::consts::TAU) as f32;

    // Curvature along the aspect direction and perpendicular to it.
    let (k_profile, k_plan) = if gradient > 1e-12 {
        let aspect_dir = (east * g_east + north * g_north).normalize();
        curvature_samples(
            dir,
            aspect_dir,
            d0,
            radius,
            center_elev_m,
            elev_scale,
            &mut sample,
        )
    } else {
        (0.0f32, 0.0f32)
    };

    // Multi-scale relief, ruggedness, and exposure.
    let (relief_m, ruggedness, exposure) = multi_scale_relief(
        dir,
        &east,
        &north,
        radius,
        center_elev_m,
        elev_scale,
        &mut sample,
    );

    // Valley / ridge position: negative = valley, positive = ridge.
    let valley_position = (exposure as f64 + center_lf.ridge_mask as f64
        - center_lf.valley_mask as f64
        - center_lf.drainage as f64)
        .clamp(-1.0, 1.0) as f32;

    let plateau_likelihood = compute_plateau_likelihood(
        elevation_above_sea_m,
        slope_deg,
        relief_m,
        center_lf.valley_mask,
        ruggedness,
    );

    let coastal_proximity_m = if elevation_above_sea_m < 0.0 {
        0.0f32
    } else {
        coastal_proximity(
            dir,
            &east,
            &north,
            radius,
            elev_scale,
            sea_level,
            &mut sample,
        )
    };

    LandformCell {
        dir,
        elevation_above_sea_m: elevation_above_sea_m as f32,
        slope_degrees: slope_deg,
        aspect_radians: aspect,
        profile_curvature_per_m: k_profile,
        plan_curvature_per_m: k_plan,
        local_relief_m: relief_m,
        ruggedness,
        exposure,
        valley_position,
        plateau_likelihood,
        coastal_proximity_m,
        tectonic_belt: center_lf.tectonic_belt,
        erosion: center_lf.erosion,
        ridge_mask: center_lf.ridge_mask,
        valley_mask: center_lf.valley_mask,
        drainage: center_lf.drainage,
        mountain_influence: center_lf.mountain_influence as f32,
        brush_displacement: center_lf.brush_displacement,
    }
}

/// Curvature probes along the aspect direction (profile) and perpendicular to it
/// (plan).  Values are clamped to the documented [-1, 1] per metre range.
fn curvature_samples(
    dir: DVec3,
    aspect_dir: DVec3,
    d0: f64,
    radius: f64,
    center_elev_m: f64,
    elev_scale: f64,
    sample: &mut impl FnMut(DVec3) -> MetricLandformSample,
) -> (f32, f32) {
    let angle = d0 / radius;

    // Profile: move along the aspect direction.
    let profile_axis = dir.cross(aspect_dir).normalize();
    let front_dir = rotate(dir, profile_axis, angle);
    let back_dir = rotate(dir, profile_axis, -angle);
    let front_e = sample(front_dir * radius).full_elevation * elev_scale;
    let back_e = sample(back_dir * radius).full_elevation * elev_scale;
    let k_profile = ((front_e + back_e - 2.0 * center_elev_m) / (d0 * d0)).clamp(-1.0, 1.0) as f32;

    // Plan: move perpendicular to the aspect direction.
    let perp_dir = dir.cross(aspect_dir).normalize();
    let plan_axis = dir.cross(perp_dir).normalize();
    let left_dir = rotate(dir, plan_axis, angle);
    let right_dir = rotate(dir, plan_axis, -angle);
    let left_e = sample(left_dir * radius).full_elevation * elev_scale;
    let right_e = sample(right_dir * radius).full_elevation * elev_scale;
    let k_plan = ((left_e + right_e - 2.0 * center_elev_m) / (d0 * d0)).clamp(-1.0, 1.0) as f32;

    (k_profile, k_plan)
}

/// Multi-scale relief (max elevation range), ruggedness (mean absolute
/// deviation normalised by relief), and exposure (relative height to the
/// neighbourhood mean) across a fixed ring set.
fn multi_scale_relief(
    dir: DVec3,
    east: &DVec3,
    north: &DVec3,
    radius: f64,
    center_elev_m: f64,
    elev_scale: f64,
    sample: &mut impl FnMut(DVec3) -> MetricLandformSample,
) -> (f32, f32, f32) {
    const RING_DIRS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    let mut local_relief = 0.0f64;
    let mut rugged_sum = 0.0f64;
    let mut exposure_sum = 0.0f64;
    let count = RELIEF_DISTANCES_M.len() as f64;

    for dist in RELIEF_DISTANCES_M {
        let mut min_e = f64::INFINITY;
        let mut max_e = f64::NEG_INFINITY;
        let mut sum_e = 0.0f64;
        let mut sum_abs = 0.0f64;
        for &(sx, sy) in &RING_DIRS {
            let e = sample_elevation_m(
                dir, east, north, sx as f64, sy as f64, dist, radius, elev_scale, sample,
            );
            min_e = min_e.min(e);
            max_e = max_e.max(e);
            sum_e += e;
            sum_abs += (e - center_elev_m).abs();
        }
        let relief = max_e - min_e;
        let mad = sum_abs / 8.0;
        local_relief = local_relief.max(relief);
        rugged_sum += (mad / (relief + 1.0)).clamp(0.0, 1.0);
        let mean = sum_e / 8.0;
        exposure_sum += ((center_elev_m - mean) / (relief + 1.0)).clamp(-1.0, 1.0);
    }

    let ruggedness = (rugged_sum / count).clamp(0.0, 1.0) as f32;
    let exposure = (exposure_sum / count).clamp(-1.0, 1.0) as f32;
    (local_relief as f32, ruggedness, exposure)
}

/// Plateau likelihood: high elevation, low slope, low relief, low valley signal.
fn compute_plateau_likelihood(
    elevation_above_sea_m: f64,
    slope_deg: f32,
    _relief_m: f32,
    valley_mask: f32,
    ruggedness: f32,
) -> f32 {
    if elevation_above_sea_m <= 500.0 {
        return 0.0;
    }
    // A plateau is locally flat and not rugged; it may sit within large-scale
    // relief, so only local slope, ruggedness, and valley signal matter.
    let slope_factor = (1.0 - slope_deg / 5.0).max(0.0);
    let valley_factor = (1.0 - valley_mask).max(0.0);
    let rugged_factor = (1.0 - ruggedness).max(0.0);
    (slope_factor * valley_factor * rugged_factor).clamp(0.0, 1.0)
}

/// Distance to the nearest ocean in metres, using a fixed set of radial metric
/// probes.  Capped at `COASTAL_MAX_M` if no ocean is found within the search
/// radius.
fn coastal_proximity(
    dir: DVec3,
    east: &DVec3,
    north: &DVec3,
    radius: f64,
    elev_scale: f64,
    sea_level: f64,
    sample: &mut impl FnMut(DVec3) -> MetricLandformSample,
) -> f32 {
    for dist in COASTAL_DISTANCES_M {
        for k in 0..COASTAL_DIRECTION_COUNT {
            let theta = (k as f64) * std::f64::consts::TAU / COASTAL_DIRECTION_COUNT as f64;
            let sx = theta.cos();
            let sy = theta.sin();
            let e = sample_elevation_m(dir, east, north, sx, sy, dist, radius, elev_scale, sample);
            if e < sea_level {
                return dist as f32;
            }
        }
    }
    COASTAL_MAX_M
}

/// Circular (vector) average of an angle field across weighted cells.
fn blend_angle(
    cells: &[(LandformCell, f64)],
    weight_sum: f64,
    field: fn(&LandformCell) -> f32,
) -> f32 {
    let _ = weight_sum;
    let x = cells
        .iter()
        .map(|(cell, weight)| (field(cell) as f64).cos() * weight)
        .sum::<f64>();
    let y = cells
        .iter()
        .map(|(cell, weight)| (field(cell) as f64).sin() * weight)
        .sum::<f64>();
    (y.atan2(x) as f32).rem_euclid(std::f32::consts::TAU)
}

// -----------------------------------------------------------------------------
// Classification helpers
// -----------------------------------------------------------------------------

/// Find the nearest active brush and its normalized closeness (1 = center, 0 = edge).
fn nearest_brush(brushes: &BrushSet, dir: Vec3) -> Option<(BrushKind, f32)> {
    let mut best_t = f32::INFINITY;
    let mut best_kind = BrushKind::Mountain;
    for brush in brushes.brushes() {
        if let Some(t) = brush.eval_t(dir) {
            if t < best_t {
                best_t = t;
                best_kind = brush.kind;
            }
        }
    }
    if best_t.is_finite() {
        Some((best_kind, (1.0 - best_t).clamp(0.0, 1.0)))
    } else {
        None
    }
}

/// Causal classification scores for every landform class.  Each score is in
/// [0, 1] and uses only the baked geometric descriptors and brush provenance.
fn class_scores(
    cell: &LandformCell,
    nearest: Option<(BrushKind, f32)>,
) -> [f32; LandformClass::COUNT] {
    let mut s = [0.0f32; LandformClass::COUNT];

    let e = cell.elevation_above_sea_m as f64;
    let ocean = if e < 0.0 { 1.0f32 } else { 0.0f32 };
    let land = 1.0 - ocean;
    let depth_m = (-e).max(0.0) as f32;

    let slope = cell.slope_degrees;
    let relief = cell.local_relief_m;
    let coastal = cell.coastal_proximity_m;
    let exposure = cell.exposure;
    let valley_pos = cell.valley_position;
    let plateau = cell.plateau_likelihood;
    let valley = cell.valley_mask;
    let erosion = cell.erosion;
    let drainage = cell.drainage;
    let mountain_inf = cell.mountain_influence;
    let tectonic = cell.tectonic_belt;

    let brush_kind = nearest.map(|(k, _)| k);
    let brush_close = nearest.map(|(_, c)| c).unwrap_or(0.0);
    let near_crater = brush_kind == Some(BrushKind::Crater);

    // 0. Abyssal plain: deep, flat ocean floor.
    s[0] = ocean
        * (1.0 - slope / 5.0).max(0.0)
        * (1.0 - relief / 200.0).max(0.0)
        * (depth_m / 3_000.0).min(1.0);

    // 1. Shelf / slope: shallow ocean near land.
    s[1] = ocean * (1.0 - coastal / 200_000.0).max(0.0) * ((4_000.0 - depth_m) / 4_000.0).max(0.0);

    // 2. Coastal plain: low, flat land close to the sea.
    s[2] = land
        * if e < 50.0 { 1.0 } else { 0.0 }
        * (1.0 - slope / 3.0).max(0.0)
        * (1.0 - coastal / 50_000.0).max(0.0);

    // 3. Beach / dune: very low, very flat, very close to the sea.
    s[3] = land
        * if e < 20.0 { 1.0 } else { 0.0 }
        * (1.0 - slope / 5.0).max(0.0)
        * (1.0 - coastal / 5_000.0).max(0.0);

    // 4. Floodplain: low, flat, close to water, with drainage signature.
    s[4] = land
        * if e < 100.0 { 1.0 } else { 0.0 }
        * (1.0 - slope / 2.0).max(0.0)
        * ((drainage + erosion) * 0.5).clamp(0.0, 1.0)
        * (1.0 - valley_pos.max(0.0)).max(0.0)
        * (1.0 - coastal / 200_000.0).max(0.0);

    // 5. Plain: low, flat, away from the coast.
    s[5] = land
        * if e < 200.0 { 1.0 } else { 0.0 }
        * (1.0 - slope / 2.0).max(0.0)
        * (coastal / 50_000.0).min(1.0);

    // 6. Rolling hill: low-moderate elevation, gentle-moderate slope, moderate relief.
    // Suppressed in strong valley positions because the negative valley signal
    // is a better explanation for the shape.
    s[6] = land
        * if e < 500.0 { 1.0 } else { 0.0 }
        * ((slope - 2.0) / 6.0).clamp(0.0, 1.0)
        * (1.0 - slope / 15.0).max(0.0)
        * (relief / 100.0).min(1.0)
        * if valley_pos < -0.2 { 0.2 } else { 1.0 };

    // 7. Plateau / mesa: high, flat, low relief.  Boosted so that genuine
    // flat highlands beat hillside scores, but discounted in negative valley
    // positions because a valley is a better explanation for a local low.
    s[7] = land
        * if e > 500.0 { 1.0 } else { 0.0 }
        * (plateau * 3.0).min(1.0)
        * if valley_pos < -0.2 { 0.3 } else { 1.0 };

    // 8. Valley: negative valley position with drainage / erosion signal.
    let valley_magnitude = (-valley_pos).max(0.0);
    s[8] = land
        * if valley_pos < -0.2 { 1.0 } else { 0.0 }
        * (1.0 - slope / 25.0).max(0.0)
        * ((drainage + valley + erosion) / 3.0).clamp(0.0, 1.0)
        * (1.0 + valley_magnitude).min(2.0)
        * 3.0
        * (1.0 - relief / 2_000.0).max(0.0)
        * if near_crater { 0.2 } else { 1.0 };

    // 9. Canyon: steep-sided valley with high relief and erosion.
    s[9] = land
        * if valley_pos < -0.2 { 1.0 } else { 0.0 }
        * (slope / 15.0).min(1.0)
        * (relief / 200.0).min(1.0)
        * ((valley + erosion) * 0.5).clamp(0.0, 1.0)
        * if brush_kind == Some(BrushKind::Canyon) {
            1.0
        } else if near_crater {
            0.2
        } else {
            0.7
        };

    // 10. Mountain slope: high, steep, on a hillside.  Suppressed in strong
    // valley positions and on flat highlands that are better explained as plateaus.
    s[10] = land
        * if e > 500.0 { 1.0 } else { 0.0 }
        * (slope / 8.0).min(1.0)
        * (1.0 - slope / 45.0).max(0.0)
        * (relief / 100.0).min(1.0)
        * if valley_pos > -0.2 { 1.0 } else { 0.2 }
        * (1.0 - plateau).max(0.2);

    // 11. Summit: high, locally flat, exposed.
    s[11] = land
        * if e > 500.0 { 1.0 } else { 0.0 }
        * (1.0 - slope / 5.0).max(0.0)
        * exposure.max(0.0)
        * (relief / 50.0).min(1.0);

    // 12. Cliff: very steep, high relief.  The threshold is relative to the
    // 500 m slope probe; real terrain rarely exceeds 30 deg at this scale.
    s[12] = land * if slope > 15.0 { 1.0 } else { 0.0 } * (relief / 100.0).min(1.0);

    // 13. Karst-like: high erosion + drainage, low slope, not a canyon.
    s[13] = land
        * (erosion * drainage).clamp(0.0, 1.0)
        * (1.0 - slope / 10.0).max(0.0)
        * (1.0 - relief / 100.0).max(0.0)
        * (1.0 - valley_pos.max(0.0)).max(0.0);

    // 14. Volcanic edifice: high, steep, high mountain/tectonic signal.
    s[14] = land
        * if e > 500.0 { 1.0 } else { 0.0 }
        * (slope / 10.0).min(1.0)
        * mountain_inf
        * tectonic
        * if brush_kind == Some(BrushKind::Mountain) {
            1.0
        } else {
            0.7
        };

    // 15. Crater / caldera: near a crater brush, with a depressed centre.
    s[15] = land
        * if near_crater { 1.0 } else { 0.0 }
        * brush_close
        * (1.0 + valley_magnitude).min(2.0)
        * (1.0 - slope / 20.0).max(0.0);

    s
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::math::uv_to_dir;
    use er_core::rng::rng_from_seed;
    use er_core::seed::PlanetSeed;
    use rand::RngCore;

    const TEST_RADIUS: f64 = 6_371_000.0;
    const TEST_ELEV_SCALE: f64 = 1_000.0;
    const TEST_SEA_LEVEL: f64 = 0.0;

    fn earthlike_atlas() -> LandformAtlas {
        LandformAtlas::build(
            PlanetSeed(0xC0FFEE),
            TEST_RADIUS,
            TEST_ELEV_SCALE,
            TEST_SEA_LEVEL,
        )
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
    fn class_catalog_is_complete_and_stable() {
        assert_eq!(LandformClass::ALL.len(), LandformClass::COUNT);
        assert_eq!(LandformClass::COUNT, 16);
        for (idx, class) in LandformClass::ALL.iter().enumerate() {
            assert_eq!(class.as_index(), idx);
            assert!(!class.as_str().is_empty());
        }
    }

    #[test]
    fn build_is_deterministic() {
        let a = earthlike_atlas();
        let b = LandformAtlas::build(
            PlanetSeed(0xC0FFEE),
            TEST_RADIUS,
            TEST_ELEV_SCALE,
            TEST_SEA_LEVEL,
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
        assert_eq!(atlas.len(), LANDFORM_ATLAS_LEN);
        assert!(
            atlas.memory_bytes()
                >= std::mem::size_of::<LandformAtlas>()
                    + atlas.len() * std::mem::size_of::<LandformCell>()
        );
        assert_eq!(atlas.lod(), LANDFORM_ATLAS_LOD);
        assert!(atlas.build_stats().cells_per_face_edge <= 64);
        assert!(atlas.memory_bytes() < 4_000_000);
    }

    #[test]
    fn build_sample_count_is_bounded() {
        let atlas = earthlike_atlas();
        let max_per_cell = 1u64
            + 8
            + (8 * RELIEF_DISTANCES_M.len() as u64)
            + (COASTAL_DIRECTION_COUNT as u64 * COASTAL_DISTANCES_M.len() as u64);
        let max_total = LANDFORM_ATLAS_LEN as u64 * max_per_cell;
        assert!(
            atlas.build_stats().build_sample_count <= max_total,
            "build cost exceeded bound: {} > {}",
            atlas.build_stats().build_sample_count,
            max_total
        );
        assert!(atlas.build_stats().build_sample_count > 0);
    }

    #[test]
    fn sample_is_finite_and_in_ranges() {
        let atlas = earthlike_atlas();
        let dirs = rand_dirs(0xABCD, 200);
        for dir in &dirs {
            let s = atlas.sample(*dir);
            assert!(s.elevation_above_sea_m.is_finite());
            assert!(
                s.slope_degrees.is_finite() && s.slope_degrees >= 0.0 && s.slope_degrees <= 90.0
            );
            assert!(
                s.aspect_radians.is_finite()
                    && s.aspect_radians >= 0.0
                    && s.aspect_radians <= std::f32::consts::TAU
            );
            assert!(
                s.profile_curvature_per_m.is_finite() && s.profile_curvature_per_m.abs() <= 1.0
            );
            assert!(s.plan_curvature_per_m.is_finite() && s.plan_curvature_per_m.abs() <= 1.0);
            assert!(s.local_relief_m.is_finite() && s.local_relief_m >= 0.0);
            assert!(s.ruggedness.is_finite() && s.ruggedness >= 0.0 && s.ruggedness <= 1.0);
            assert!(s.exposure.is_finite() && s.exposure >= -1.0 && s.exposure <= 1.0);
            assert!(
                s.valley_position.is_finite()
                    && s.valley_position >= -1.0
                    && s.valley_position <= 1.0
            );
            assert!(
                s.plateau_likelihood.is_finite()
                    && s.plateau_likelihood >= 0.0
                    && s.plateau_likelihood <= 1.0
            );
            assert!(
                s.coastal_proximity_m.is_finite()
                    && s.coastal_proximity_m >= 0.0
                    && s.coastal_proximity_m <= COASTAL_MAX_M
            );
            assert!(s.confidence >= 0.0 && s.confidence <= 1.0);
        }
    }

    #[test]
    fn ocean_samples_are_marine() {
        let atlas = earthlike_atlas();
        let dirs = rand_dirs(0xBEEF, 300);
        let mut found = 0usize;
        for dir in &dirs {
            let s = atlas.sample(*dir);
            if s.elevation_above_sea_m < 0.0 {
                found += 1;
                assert!(
                    s.primary_class == LandformClass::AbyssalPlain
                        || s.primary_class == LandformClass::ShelfSlope,
                    "ocean point classified as {:?}",
                    s.primary_class
                );
            }
        }
        assert!(found > 0, "no ocean samples found in deterministic atlas");
    }

    #[test]
    fn high_flat_samples_are_plateau_or_summit() {
        let atlas = earthlike_atlas();
        let dirs = rand_dirs(0xCAFE, 500);
        let mut found = 0usize;
        for dir in &dirs {
            let s = atlas.sample(*dir);
            if s.elevation_above_sea_m > 500.0
                && s.plateau_likelihood > 0.2
                && s.valley_position > -0.2
            {
                found += 1;
                assert!(
                    s.primary_class == LandformClass::PlateauMesa
                        || s.primary_class == LandformClass::Summit,
                    "high flat point classified as {:?}",
                    s.primary_class
                );
            }
        }
        assert!(found > 0, "no high-flat plateau samples found");
    }

    #[test]
    fn steep_samples_are_not_lowland() {
        let atlas = earthlike_atlas();
        let dirs = rand_dirs(0xF00D, 500);
        let lowlands = [
            LandformClass::CoastalPlain,
            LandformClass::BeachDune,
            LandformClass::Floodplain,
            LandformClass::Plain,
        ];
        let mut found = 0usize;
        for dir in &dirs {
            let s = atlas.sample(*dir);
            if s.slope_degrees > 5.0 {
                found += 1;
                assert!(
                    !lowlands.contains(&s.primary_class),
                    "steep point classified as lowland {:?}",
                    s.primary_class
                );
            }
        }
        assert!(found > 0, "no steep samples found");
    }

    #[test]
    fn valley_samples_are_negative_class() {
        let atlas = earthlike_atlas();
        let dirs = rand_dirs(0xBABE, 500);
        let negative_classes = [
            LandformClass::Valley,
            LandformClass::Canyon,
            LandformClass::Floodplain,
            LandformClass::KarstLike,
            LandformClass::CraterCaldera,
        ];
        let mut found = 0usize;
        for dir in &dirs {
            let s = atlas.sample(*dir);
            if s.elevation_above_sea_m > 0.0 && s.valley_position < -0.5 && s.slope_degrees < 15.0 {
                found += 1;
                assert!(
                    negative_classes.contains(&s.primary_class),
                    "strong valley point classified as {:?}",
                    s.primary_class
                );
            }
        }
        assert!(found > 0, "no strong valley samples found");
    }

    #[test]
    fn crater_brush_points_are_crater_or_volcanic() {
        let atlas = earthlike_atlas();
        let dirs = rand_dirs(0xDEAD, 600);
        let mut found = 0usize;
        for dir in &dirs {
            let dir_f = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32).normalize();
            if let Some((kind, close)) = nearest_brush(&atlas.brushes, dir_f) {
                if kind == BrushKind::Crater && close > 0.8 {
                    found += 1;
                    let s = atlas.sample(*dir);
                    assert!(
                        s.primary_class == LandformClass::CraterCaldera
                            || s.primary_class == LandformClass::VolcanicEdifice,
                        "crater-brush point classified as {:?}",
                        s.primary_class
                    );
                }
            }
        }
        assert!(found > 0, "no close crater-brush points found");
    }

    #[test]
    fn seam_continuity_across_cube_edges() {
        let atlas = earthlike_atlas();
        let pairs = [
            // Shared cube edges from the elevation continuity test.
            (0, 1.0, 0.5, 2, 1.0, 0.5), // +X posU == +Y posU
            (0, 0.5, 0.0, 5, 1.0, 0.5), // +X negV == -Z posU
        ];
        for (f1, u1, v1, f2, u2, v2) in pairs {
            let d1 = uv_to_dir(f1, u1, v1);
            let d2 = uv_to_dir(f2, u2, v2);
            let diff = (d1 - d2).length();
            assert!(diff < 1e-6, "shared edge dirs differ: {diff}");
            let s1 = atlas.sample(d1);
            let s2 = atlas.sample(d2);
            assert!(
                (s1.elevation_above_sea_m - s2.elevation_above_sea_m).abs() < 1.0,
                "elevation seam across edge"
            );
            assert_eq!(s1.primary_class, s2.primary_class, "class seam across edge");
            assert!(
                (s1.confidence - s2.confidence).abs() < 0.01,
                "confidence seam across edge"
            );
        }
    }

    #[test]
    fn different_seeds_produce_different_atlases() {
        let a = LandformAtlas::build(
            PlanetSeed(0xC0FFEE),
            TEST_RADIUS,
            TEST_ELEV_SCALE,
            TEST_SEA_LEVEL,
        );
        let b = LandformAtlas::build(
            PlanetSeed(0xDEADBEEF),
            TEST_RADIUS,
            TEST_ELEV_SCALE,
            TEST_SEA_LEVEL,
        );
        assert_ne!(a.identity(), b.identity());
        let dirs = rand_dirs(0xFEED, 50);
        let mut any_diff = false;
        for dir in &dirs {
            if a.sample(*dir).elevation_above_sea_m.to_bits()
                != b.sample(*dir).elevation_above_sea_m.to_bits()
            {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "different seeds produced identical samples");
    }
}
