//! Sphere-native canonical chart types for learned macro-terrain.
//!
//! This module defines the *contract* for sphere-native learned charts that
//! supersede the diagnostic four-tile-per-face atlas used by the
//! Milestone 3 sidecar evaluation path. The key invariants:
//!
//! - A chart is a canonical, bounded region of one cube face, addressed by a
//!   `SurfaceChartId` (face, level, x, y) analogous to `CellKey` but owned by
//!   the learned-cache layer rather than the render quadtree.
//! - All analytic fields are sampled from a 3D *metric* surface point
//!   (`terrain_space::metric_surface_point`) so the field value depends only
//!   on the sphere direction, not on which face produced it. This makes every
//!   cube edge and corner continuous by construction — the same property the
//!   procedural macro field already relies on
//!   (`metric_macro_continuous_across_cube_face_edges`).
//! - `SurfaceRegion` turns a quadtree chunk plus a halo into a canonical
//!   requested rectangle with meters-per-pixel and stable floor/ceil
//!   rounding, matching the upstream model's planar grid request contract
//!   while remaining anchored to the sphere.
//! - `SurfaceChartMetadata` carries every version-affecting field required by
//!   the roadmap's cache-key checklist and is the single source of truth for
//!   the cache record identity.
//!
//! The module is intentionally pure (`Send + Sync`, no I/O) so it can be
//! called from terrain mesh workers. Sidecar I/O and disk persistence live in
//! `surface_cache.rs` and the `er_game` adapter.

use crate::terrain_space::metric_surface_point;
use er_core::math::{dir_to_uv, uv_to_dir, CellKey};
use glam::DVec3;
use std::fmt;

/// Revision of the chart projection itself. Bumped whenever the mapping
/// between cube-face coordinates and the upstream planar grid changes.
/// Current mapping: metric 3D surface point (face-independent), revision 4
/// (supersedes revision 3 which used padded 2^level chart grids; revision 4
/// uses the exact provider `charts_per_face_edge` for non-power-of-two
/// tile counts like Earth's 652).
pub const SURFACE_CHART_PROJECTION_REVISION: u16 = 4;

/// Ownership rule for a chart: which layer "owns" the sea-level datum and
/// shoreline classification. The roadmap requires that sea datum, water
/// visibility, normals, and material masks derive from a single composed
/// field revision; learned charts supply local relief only, while the
/// procedural macro field retains shoreline ownership (Milestone 4 policy).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChartOwnership {
    /// Learned chart supplies local relief only; the procedural macro field
    /// owns the global sea datum and shoreline. This is the M4 default.
    LearnedReliefProceduralShoreline,
    /// Diagnostic mode where the learned chart owns the datum. Not used at
    /// runtime in M4 but reserved for the future bake-first workflow.
    LearnedOwnsDatum,
}

/// Canonical, versioned metadata describing how a family of learned charts
/// was generated and how it relates to the sphere. Two records with the same
/// `SurfaceChartMetadata` are interchangeable; any field change is a cache
/// migration boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceChartMetadata {
    /// World seed the upstream model was conditioned on.
    pub seed: u64,
    /// Revision of the cube-face <-> planar-grid projection. See
    /// `SURFACE_CHART_PROJECTION_REVISION`.
    pub projection_revision: u16,
    /// Upstream model revision string (pinned in the Milestone 3 manifest).
    pub model_revision: String,
    /// Revision of the conditioning map (coarse planetary stage) the model
    /// consumed. Changes here invalidate every tile.
    pub conditioning_revision: u32,
    /// Revision of the procedural residual field blended beneath the learned
    /// macro. Changes here do not invalidate the tile but invalidate the
    /// *composed* cache key when a record also stores composed values.
    pub residual_revision: u32,
    /// Global sea-level datum in meters. Tiles are stored relative to this
    /// datum so a single global value controls shoreline classification.
    pub sea_level_datum_m: i32,
    /// Native pixel scale of the upstream model in meters per pixel.
    pub pixel_scale_m: u32,
    /// Halo (samples per side) included in every stored tile so normals and
    /// material derivatives can be computed without cross-tile fetches.
    pub halo_samples: u32,
    /// Core tile resolution (samples per side, excluding halo).
    pub core_resolution: u32,
    /// Ownership rule for the sea datum and shoreline.
    pub ownership: ChartOwnership,
    /// Radius of the planet in meters. Part of the cache key because the
    /// metric surface point depends on it.
    pub planet_radius_m: u64,
    /// Exact number of charts (provider tiles) per cube-face edge. This is
    /// the provider's actual tile count (e.g. 4 for the miniature planet,
    /// 652 for Earth), NOT a padded power-of-two. All runtime chart
    /// identity, footprint math, and cache keys use this value. The `level`
    /// field in `SurfaceChartId` is kept only as informational metadata and
    /// is NEVER used to derive the chart grid width.
    pub charts_per_face_edge: u32,
}

impl SurfaceChartMetadata {
    /// Stored resolution (core + 2*halo) per side.
    pub fn stored_resolution(&self) -> u32 {
        self.core_resolution + self.halo_samples * 2
    }

    /// Effective meters-per-pixel at the planet surface for this chart.
    pub fn meters_per_pixel(&self) -> f64 {
        self.pixel_scale_m as f64
    }
}

/// Canonical address of one learned chart within a cube face.
///
/// `(face, x, y, charts_per_face_edge)` identifies a chart. `x`/`y` are in
/// `[0, charts_per_face_edge)`. The `level` field is retained for
/// informational/debugging purposes only and is NEVER used to derive the
/// chart grid width at runtime — `charts_per_face_edge` is the single source
/// of truth for the grid width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceChartId {
    pub face: u8,
    /// Informational only: the quadtree level that would produce
    /// `2^level >= charts_per_face_edge`. Never used for runtime math.
    pub level: u8,
    pub x: u32,
    pub y: u32,
    /// Exact charts per face edge (e.g. 652 for Earth, 4 for miniature).
    pub charts_per_face_edge: u32,
}

impl SurfaceChartId {
    /// Number of charts per face edge. This is the exact provider tile count,
    /// NOT a padded power-of-two.
    pub fn charts_per_edge(&self) -> u32 {
        self.charts_per_face_edge
    }

    /// Center uv of this chart within its face, in [0,1].
    pub fn center_uv(&self) -> (f64, f64) {
        let n = self.charts_per_edge() as f64;
        ((self.x as f64 + 0.5) / n, (self.y as f64 + 0.5) / n)
    }

    /// Unit direction at the chart center.
    pub fn center_dir(&self) -> DVec3 {
        let (u, v) = self.center_uv();
        uv_to_dir(self.face, u, v)
    }

    /// Build a chart id from a sphere direction. Uses the exact
    /// `charts_per_face_edge` (not a padded power-of-two) to map the
    /// direction to the containing chart.
    pub fn from_direction(dir: DVec3, charts_per_face_edge: u32) -> Self {
        let (face, u, v) = dir_to_uv(dir);
        Self::from_uv(face, u, v, charts_per_face_edge)
    }

    /// Build a chart id from face-local uv. Uses the exact
    /// `charts_per_face_edge` to floor the uv into the containing chart.
    pub fn from_uv(face: u8, u: f64, v: f64, charts_per_face_edge: u32) -> Self {
        let n = charts_per_face_edge.max(1) as f64;
        let max = n - 1.0;
        let x = (u * n).floor().max(0.0).min(max) as u32;
        let y = (v * n).floor().max(0.0).min(max) as u32;
        // Compute informational level: smallest power-of-two >= N.
        let level = if charts_per_face_edge <= 1 {
            0
        } else {
            (32 - (charts_per_face_edge - 1).leading_zeros()) as u8
        };
        SurfaceChartId {
            face,
            level,
            x,
            y,
            charts_per_face_edge,
        }
    }
}

impl fmt::Display for SurfaceChartId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "chart(f{},L{},{},{})",
            self.face, self.level, self.x, self.y
        )
    }
}

/// A chart id plus the halo/bounds that define a stored patch. The patch
/// bounds are the canonical request rectangle sent to the upstream model
/// (in face-local metric grid coordinates) plus the halo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfacePatchId {
    pub chart: SurfaceChartId,
    /// Halo in samples per side included in the stored patch.
    pub halo: u32,
}

impl SurfacePatchId {
    pub fn new(chart: SurfaceChartId, halo: u32) -> Self {
        Self { chart, halo }
    }

    /// Stored resolution per side for a patch with the given core resolution.
    pub fn stored_resolution(&self, core_resolution: u32) -> u32 {
        core_resolution + self.halo * 2
    }
}

/// A canonical requested rectangle derived from a render quadtree chunk
/// plus a normal-sampling halo. `SurfaceRegion` is the bridge between the
/// render LOD quadtree and the learned-chart quadtree: it snaps a chunk's
/// sphere footprint to a canonical, halo-padded rectangle in face-local uv,
/// then to chart coordinates at a fixed learned level.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceRegion {
    /// Source render cell.
    pub chunk: CellKey,
    /// Halo in *chart samples* required for normal/material computation.
    pub halo_samples: u32,
    /// Chart level the region maps onto (informational only).
    pub chart_level: u8,
    /// Exact charts per face edge (e.g. 652 for Earth, 4 for miniature).
    pub charts_per_face_edge: u32,
}

impl SurfaceRegion {
    pub fn new(
        chunk: CellKey,
        halo_samples: u32,
        chart_level: u8,
        charts_per_face_edge: u32,
    ) -> Self {
        Self {
            chunk,
            halo_samples,
            chart_level,
            charts_per_face_edge,
        }
    }

    /// The set of charts this region overlaps at `chart_level`. Returns at
    /// least one chart. Bounded by the region's uv extent so a small chunk
    /// near a chart boundary does not fan out to the whole face.
    pub fn overlapping_charts(&self) -> Vec<SurfaceChartId> {
        let chunk = self.chunk;
        let n_chunk = er_core::math::cells_per_edge(chunk.lod) as f64;
        // Chunk uv extent (half-open) in [0,1].
        let u0 = chunk.i as f64 / n_chunk;
        let v0 = chunk.j as f64 / n_chunk;
        let u1 = (chunk.i + 1) as f64 / n_chunk;
        let v1 = (chunk.j + 1) as f64 / n_chunk;
        // Expand by halo in chart-uv units (conservative: use the chart level
        // granularity so we always cover the halo even if the chunk is
        // coarser than a chart).
        let n_chart = self.charts_per_face_edge as f64;
        let halo_uv = (self.halo_samples as f64) / n_chart;
        let eu0 = (u0 - halo_uv).max(0.0);
        let ev0 = (v0 - halo_uv).max(0.0);
        let eu1 = (u1 + halo_uv).min(1.0);
        let ev1 = (v1 + halo_uv).min(1.0);
        let x0 = (eu0 * n_chart).floor() as u32;
        let x1 = ((eu1 * n_chart).ceil() as u32).saturating_sub(1).max(x0);
        let y0 = (ev0 * n_chart).floor() as u32;
        let y1 = ((ev1 * n_chart).ceil() as u32).saturating_sub(1).max(y0);
        let mut out = Vec::new();
        for x in x0..=x1 {
            for y in y0..=y1 {
                out.push(SurfaceChartId {
                    face: chunk.face,
                    level: self.chart_level,
                    x,
                    y,
                    charts_per_face_edge: self.charts_per_face_edge,
                });
            }
        }
        out
    }
}

/// A canonical, sphere-native field sample point. Because the field is a
/// pure function of the 3D metric surface point, the *same* direction yields
/// the *same* value regardless of which cube face or chart produced it. This
/// is the analytic-continuity guarantee required by the M4 exit gate.
#[derive(Clone, Copy, Debug)]
pub struct ChartSamplePoint {
    /// Unit sphere direction (face-independent identity).
    pub dir: DVec3,
    /// Metric surface point = dir * planet_radius_m.
    pub metric_pos: DVec3,
}

impl ChartSamplePoint {
    /// Build from a sphere direction and planet radius.
    pub fn from_dir(dir: DVec3, planet_radius_m: f64) -> Self {
        let n = dir.normalize();
        Self {
            dir: n,
            metric_pos: metric_surface_point(n, planet_radius_m),
        }
    }

    /// Build from a face/uv and planet radius.
    pub fn from_uv(face: u8, u: f64, v: f64, planet_radius_m: f64) -> Self {
        Self::from_dir(uv_to_dir(face, u, v), planet_radius_m)
    }
}

/// Trait for a sphere-native analytic field sampled through the chart
/// abstraction. Implementations must be pure functions of the metric surface
/// point so that cube-edge/corner continuity holds by construction.
pub trait SurfaceChartField: Send + Sync {
    /// Elevation in meters at the sample point.
    fn elevation_m(&self, point: ChartSamplePoint) -> f64;

    /// Four climate channels at the sample point, matching the upstream
    /// protocol order: `[temp, t_season, precip, p_cv]`.
    fn climate(&self, _point: ChartSamplePoint) -> [f32; 4] {
        [0.0; 4]
    }
}

/// An analytic synthetic field used to prove edge/corner continuity. The
/// field is a smooth function of the 3D metric position so it is continuous
/// across every cube face boundary by construction.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticGradientField {
    pub planet_radius_m: f64,
    pub amplitude_m: f64,
}

impl SyntheticGradientField {
    pub fn new(planet_radius_m: f64, amplitude_m: f64) -> Self {
        Self {
            planet_radius_m,
            amplitude_m,
        }
    }
}

impl SurfaceChartField for SyntheticGradientField {
    fn elevation_m(&self, point: ChartSamplePoint) -> f64 {
        // A smooth, low-frequency analytic function of the 3D position. The
        // exact form is irrelevant for the continuity proof; what matters is
        // that it depends only on `metric_pos`, which is identical for two
        // directions that map to the same sphere point.
        let p = point.metric_pos / self.planet_radius_m.max(1.0);
        self.amplitude_m * (0.5 * p.x + 0.3 * p.y + 0.2 * p.z + 0.1 * (p.x * p.y + p.z * 0.5))
    }

    fn climate(&self, point: ChartSamplePoint) -> [f32; 4] {
        let p = point.metric_pos / self.planet_radius_m.max(1.0);
        [
            (15.0 + 10.0 * p.y) as f32,
            (p.x * 5.0) as f32,
            (1000.0 + 500.0 * p.z) as f32,
            (0.2 + 0.1 * p.x) as f32,
        ]
    }
}

/// Iterate all 12 cube edges as `(face_a, side_a, face_b, side_b)` where
/// `side` is 0=NegU,1=PosU,2=NegV,3=PosV. The shared edge is the same set of
/// sphere directions seen from two faces, so an analytic field must agree to
/// machine precision there.
///
/// The pairs are derived from the already-validated cube topology in
/// `er_core::math` (`edge_neighbors_wrap`): each face has 4 sides, and the
/// neighbor across each side is the face that shares that edge. We compute
/// the neighbor's matching side by finding which of its 4 sides lands back
/// on the original face.
pub fn all_cube_edges() -> Vec<(u8, u8, u8, u8)> {
    use er_core::math::{cell_neighbor, cells_per_edge, NeighborSide};
    let sides = [
        NeighborSide::NegU,
        NeighborSide::PosU,
        NeighborSide::NegV,
        NeighborSide::PosV,
    ];
    let lod = 4u8;
    let n = cells_per_edge(lod);
    let mut edges: Vec<(u8, u8, u8, u8)> = Vec::with_capacity(12);
    for face in 0..6u8 {
        for (s_idx, side) in sides.iter().enumerate() {
            // Cell at the middle of this edge.
            let (i, j) = match s_idx {
                0 => (0u32, n / 2),
                1 => (n - 1, n / 2),
                2 => (n / 2, 0),
                _ => (n / 2, n - 1),
            };
            let key = er_core::math::CellKey { face, i, j, lod };
            let nb = cell_neighbor(key, *side);
            if nb.face <= face {
                // Already added from the neighbor's perspective, or same face.
                continue;
            }
            // Find which side of the neighbor points back to `face`.
            let nb_key = er_core::math::CellKey {
                face: nb.face,
                i: nb.i,
                j: nb.j,
                lod,
            };
            let mut nb_side = 0u8;
            for (k, ns) in sides.iter().enumerate() {
                let back = cell_neighbor(nb_key, *ns);
                if back.face == face {
                    nb_side = k as u8;
                    break;
                }
            }
            edges.push((face, s_idx as u8, nb.face, nb_side));
        }
    }
    edges
}

/// All 8 cube corners as unit directions. Each corner is shared by 3 faces,
/// so an analytic field must agree there to machine precision across all 3.
pub fn all_cube_corners() -> [DVec3; 8] {
    [
        DVec3::new(1.0, 1.0, 1.0).normalize(),
        DVec3::new(1.0, 1.0, -1.0).normalize(),
        DVec3::new(1.0, -1.0, 1.0).normalize(),
        DVec3::new(1.0, -1.0, -1.0).normalize(),
        DVec3::new(-1.0, 1.0, 1.0).normalize(),
        DVec3::new(-1.0, 1.0, -1.0).normalize(),
        DVec3::new(-1.0, -1.0, 1.0).normalize(),
        DVec3::new(-1.0, -1.0, -1.0).normalize(),
    ]
}

/// For a given (face, side), return the parametric uv along the shared edge
/// for parameter `t` in [0,1], plus the three faces that meet at each end
/// (used by corner checks). Returns `(u, v)` on `face` for the edge point.
pub fn edge_uv(_face: u8, side: u8, t: f64) -> (f64, f64) {
    match side {
        0 => (0.0, t), // NegU
        1 => (1.0, t), // PosU
        2 => (t, 0.0), // NegV
        _ => (t, 1.0), // PosV
    }
}

/// The three faces that share a cube corner, given one corner direction.
/// The corner is the point where three faces meet; this returns those faces.
pub fn faces_at_corner(dir: DVec3) -> [u8; 3] {
    let (f1, _, _) = dir_to_uv(dir);
    // The three faces meeting at a corner are the three axis-aligned faces
    // whose signs match the corner's dominant components. Build them from the
    // sign of each nonzero component.
    let mut faces = [0u8; 3];
    let mut idx = 0;
    let comps = [(dir.x, 0u8), (dir.y, 1u8), (dir.z, 2u8)];
    for (val, axis) in comps {
        if idx >= 3 {
            break;
        }
        if val.abs() > 1e-9 {
            // face = axis*2 + (sign<0 ? 1 : 0)
            faces[idx] = axis * 2 + if val < 0.0 { 1 } else { 0 };
            idx += 1;
        }
    }
    let _ = f1;
    faces
}

/// Normal at a chart sample point, computed by central differences on the
/// analytic field. Used by the seam gate to assert <0.1 degree continuity.
/// The step is in *chart uv* space so the normal is well-defined even when
/// the field is learned.
pub fn field_normal_deg(
    field: &dyn SurfaceChartField,
    face: u8,
    u: f64,
    v: f64,
    planet_radius_m: f64,
    step: f64,
) -> f64 {
    // Sample the field at the point and two neighbors, build a normal from
    // the elevation gradient, and return its angle from the radial normal.
    // For a pure-metric field the normal is continuous across edges because
    // the field is continuous and the tangent frame rotates smoothly.
    let p = ChartSamplePoint::from_uv(face, u, v, planet_radius_m);
    let p_u = ChartSamplePoint::from_uv(face, u + step, v, planet_radius_m);
    let p_v = ChartSamplePoint::from_uv(face, u, v + step, planet_radius_m);
    let h = field.elevation_m(p);
    let h_u = field.elevation_m(p_u);
    let h_v = field.elevation_m(p_v);
    // Radial normal is `dir`. Tangent vectors in uv space:
    let dir = p.dir;
    let du_dir = (p_u.dir - p.dir).normalize_or_zero();
    let dv_dir = (p_v.dir - p.dir).normalize_or_zero();
    // Surface normal ~ radial - (dh/du)*tangent_u - (dh/dv)*tangent_v
    let grad_u = (h_u - h) / step.max(1e-12);
    let grad_v = (h_v - h) / step.max(1e-12);
    let normal = (dir
        - du_dir * (grad_u / planet_radius_m.max(1.0))
        - dv_dir * (grad_v / planet_radius_m.max(1.0)))
    .normalize_or_zero();
    // Angle between the field normal and the radial normal, in degrees.
    let cos = normal.dot(dir).clamp(-1.0, 1.0);
    cos.acos().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::math::{uv_to_dir, FACE_COUNT};

    const R: f64 = 6_371_000.0;

    #[test]
    fn chart_id_round_trips_direction() {
        for face in 0..6u8 {
            for &(u, v) in &[(0.5, 0.5), (0.25, 0.75), (0.9, 0.1)] {
                let dir = uv_to_dir(face, u, v);
                let id = SurfaceChartId::from_direction(dir, 3);
                assert_eq!(id.face, face);
                assert!(id.x < id.charts_per_edge());
                assert!(id.y < id.charts_per_edge());
            }
        }
    }

    #[test]
    fn overlapping_charts_covers_chunk() {
        let chunk = CellKey {
            face: 0,
            i: 1,
            j: 1,
            lod: 3,
        };
        let region = SurfaceRegion::new(chunk, 2, 4, 16);
        let charts = region.overlapping_charts();
        assert!(!charts.is_empty());
        for c in &charts {
            assert_eq!(c.face, 0);
            assert!(c.x < c.charts_per_edge());
            assert!(c.y < c.charts_per_edge());
        }
    }

    #[test]
    fn synthetic_field_is_face_independent_at_shared_edge() {
        // The shared +X posU / +Y posU edge: the same direction must give
        // the same elevation and climate.
        let field = SyntheticGradientField::new(R, 1000.0);
        for i in 0..32 {
            let t = (i as f64 + 0.5) / 32.0;
            let (u, v) = edge_uv(0, 1, t); // +X PosU edge
            let (u2, v2) = edge_uv(2, 1, t); // +Y PosU edge
            let pa = ChartSamplePoint::from_uv(0, u, v, R);
            let pb = ChartSamplePoint::from_uv(2, u2, v2, R);
            assert!(
                (pa.metric_pos - pb.metric_pos).length() < 1e-3,
                "metric pos mismatch at edge t={t}"
            );
            assert!(
                (field.elevation_m(pa) - field.elevation_m(pb)).abs() < 1e-6,
                "elevation mismatch at edge t={t}"
            );
            let ca = field.climate(pa);
            let cb = field.climate(pb);
            for k in 0..4 {
                assert!(
                    (ca[k] - cb[k]).abs() < 1e-4,
                    "climate[{k}] mismatch at t={t}"
                );
            }
        }
    }

    #[test]
    fn all_12_edges_listed_and_valid() {
        let edges = all_cube_edges();
        assert_eq!(edges.len(), 12, "a cube has 12 edges");
        for &(fa, sa, fb, sb) in &edges {
            assert!(fa < FACE_COUNT as u8);
            assert!(fb < FACE_COUNT as u8);
            assert!(sa < 4);
            assert!(sb < 4);
            // The two (face,side) pairs must name the same shared edge: their
            // edge directions at t=0.5 must coincide.
            let (ua, va) = edge_uv(fa, sa, 0.5);
            let (ub, vb) = edge_uv(fb, sb, 0.5);
            let da = uv_to_dir(fa, ua, va);
            let db = uv_to_dir(fb, ub, vb);
            assert!(
                (da - db).length() < 1e-9,
                "edge ({fa},{sa}) <-> ({fb},{sb}) does not share a direction"
            );
        }
    }
    #[test]
    fn all_8_corners_listed_and_valid() {
        let corners = all_cube_corners();
        assert_eq!(corners.len(), 8);
        for c in corners {
            assert!((c.length() - 1.0).abs() < 1e-9);
            // Each corner is shared by exactly 3 faces.
            let faces = faces_at_corner(c);
            let unique: std::collections::HashSet<u8> = faces.iter().copied().collect();
            assert_eq!(unique.len(), 3, "corner {c:?} not shared by 3 faces");
        }
    }

    #[test]
    fn field_normal_is_small_for_smooth_field() {
        let field = SyntheticGradientField::new(R, 1000.0);
        // A smooth low-frequency field has a small normal deviation.
        let ang = field_normal_deg(&field, 0, 0.5, 0.5, R, 1e-4);
        assert!(
            ang < 90.0,
            "normal angle {ang} deg should be < 90 for smooth field"
        );
    }
}
