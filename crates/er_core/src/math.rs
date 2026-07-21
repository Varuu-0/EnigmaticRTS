//! er_core::math — coordinate & math foundation for the spherified-cube planet.
//!
//! f64 world math / f32 render boundary; spherified-cube projection (6 cube
//! faces -> unit sphere); per-point tangent frame; floating-origin
//! (origin-shifting); and the spherical quadtree cell key (CellKey) + neighbor
//! adjacency. Pure math, no rendering, no Bevy.
//!
//! Convention (faces 0..5 = +X,-X,+Y,-Y,+Z,-Z): each face fixes one axis at +-1
//! and spans the other two over [-1,1] via `u`,`v` in [0,1]. `FACE_U`/`FACE_V`
//! have magnitude 2 so `u` in [0,1] maps an axis coord to [-1,1]; `FACE_CORNER`
//! is the `u=v=0` corner. A point on the unit sphere is
//! `normalize(FACE_CORNER[f] + u*FACE_U[f] + v*FACE_V[f])`. The neighbor
//! adjacency is computed geometrically (no brittle edge table): step one cell
//! in uv, project, and re-derive the face — so wrapping across the 12 cube
//! edges is automatic. Validated against a Python reference + unit tests below.

use glam::{DVec3, Vec3};

// ---------------------------------------------------------------------------
// World / render positions
// ---------------------------------------------------------------------------

/// A point in f64 world space (planet/surface/system scale). Keeps full
/// precision; only converted to f32 at the render boundary (`to_render`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct WorldPos(pub DVec3);

/// A point in f32 render space (camera-relative, origin-shifted). This is the
/// last step before handing coordinates to the GPU; far from the origin so f32
/// precision is fine.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RenderPos(pub Vec3);

impl WorldPos {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self(DVec3::new(x, y, z))
    }

    /// World -> render: subtract the floating origin and cast to f32.
    pub fn to_render(self, origin: OriginOffset) -> RenderPos {
        let d = self.0 - origin.0;
        RenderPos(Vec3::new(d.x as f32, d.y as f32, d.z as f32))
    }
}

impl RenderPos {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3::new(x, y, z))
    }
}

// ---------------------------------------------------------------------------
// Spherified-cube face tables (6 faces: +X,-X,+Y,-Y,+Z,-Z)
// ---------------------------------------------------------------------------

pub const FACE_COUNT: usize = 6;

/// `u=v=0` corner of each face (the face plane coordinate is +-1; the two
/// spanned axes start at -1).
pub const FACE_CORNER: [DVec3; 6] = [
    DVec3::new(1.0, -1.0, -1.0),  // +X  u->y  v->z
    DVec3::new(-1.0, -1.0, -1.0), // -X  u->y  v->z
    DVec3::new(-1.0, 1.0, -1.0),  // +Y  u->x  v->z
    DVec3::new(-1.0, -1.0, -1.0), // -Y  u->x  v->z
    DVec3::new(-1.0, -1.0, 1.0),  // +Z  u->x  v->y
    DVec3::new(-1.0, -1.0, -1.0), // -Z  u->x  v->y
];

/// Direction `u` increases (magnitude 2: u in [0,1] -> axis coord [-1,1]).
pub const FACE_U: [DVec3; 6] = [
    DVec3::new(0.0, 2.0, 0.0),
    DVec3::new(0.0, 2.0, 0.0),
    DVec3::new(2.0, 0.0, 0.0),
    DVec3::new(2.0, 0.0, 0.0),
    DVec3::new(2.0, 0.0, 0.0),
    DVec3::new(2.0, 0.0, 0.0),
];

/// Direction `v` increases (magnitude 2).
pub const FACE_V: [DVec3; 6] = [
    DVec3::new(0.0, 0.0, 2.0),
    DVec3::new(0.0, 0.0, 2.0),
    DVec3::new(0.0, 0.0, 2.0),
    DVec3::new(0.0, 0.0, 2.0),
    DVec3::new(0.0, 2.0, 0.0),
    DVec3::new(0.0, 2.0, 0.0),
];

const FACE_U_AXIS: [usize; 6] = [1, 1, 0, 0, 0, 0];
const FACE_V_AXIS: [usize; 6] = [2, 2, 2, 2, 1, 1];

/// (axis index, sign) for a face: axis = face/2, +face if even else -face.
pub fn face_axis_sign(face: u8) -> (usize, f64) {
    let axis = (face / 2) as usize;
    let sign = if face.is_multiple_of(2) { 1.0 } else { -1.0 };
    (axis, sign)
}

#[inline]
fn axis_comp(v: DVec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Map face-local uv in [0,1] to a unit direction on the sphere. `u`/`v`
/// slightly outside [0,1] are allowed (used by neighbor stepping): the point
/// extends past the face plane and `normalize` projects it onto an adjacent
/// face.
pub fn uv_to_dir(face: u8, u: f64, v: f64) -> DVec3 {
    let i = face as usize;
    (FACE_CORNER[i] + FACE_U[i] * u + FACE_V[i] * v).normalize()
}

/// Inverse of `uv_to_dir`: pick the dominant axis -> face, then recover uv by
/// projecting onto that face's plane. Scale-invariant: works on any non-zero
/// `dir` (no normalization needed — the dominant axis and recovered uv depend
/// only on component ratios). Returns exact uv (may be marginally outside
/// [0,1] due to fp); `dir_to_cell` clamps for indexing.
pub fn dir_to_uv(dir: DVec3) -> (u8, f64, f64) {
    assert!(
        dir.is_finite() && dir.length_squared() > 0.0,
        "dir_to_uv requires a finite non-zero direction"
    );
    let d = dir;
    let ax = d.x.abs();
    let ay = d.y.abs();
    let az = d.z.abs();
    let (axis, val) = if ax >= ay && ax >= az {
        (0usize, d.x)
    } else if ay >= az {
        (1, d.y)
    } else {
        (2, d.z)
    };
    let sign = if val >= 0.0 { 1.0 } else { -1.0 };
    let s = if sign > 0.0 { 0usize } else { 1 };
    let face = (axis * 2 + s) as u8;
    let f = face as usize;
    // project onto the face plane: cube = d * (sign / d[axis])  -> cube[axis] = sign
    let t = sign / val;
    let cube = d * t;
    let u = (axis_comp(cube, FACE_U_AXIS[f]) + 1.0) / 2.0;
    let v = (axis_comp(cube, FACE_V_AXIS[f]) + 1.0) / 2.0;
    (face, u, v)
}

// ---------------------------------------------------------------------------
// Tangent frame at a face/uv point (right-handed orthonormal TBN)
// ---------------------------------------------------------------------------

/// Orthonormal (normal, tangent, bitangent) at `uv_to_dir(face,u,v)`.
/// `normal = dir`; `tangent` is the analytic d(dir)/du direction (the cube
/// u-axis projected onto the tangent plane); `bitangent = normal x tangent` for
/// a clean right-handed frame. This is exact and safe right up to a face edge
/// (no finite-difference that would cross face boundaries).
pub fn tangent_frame(face: u8, u: f64, v: f64) -> (DVec3, DVec3, DVec3) {
    let normal = uv_to_dir(face, u, v);
    let fu = FACE_U[face as usize];
    let tangent = (fu - normal * fu.dot(normal)).normalize();
    let bitangent = normal.cross(tangent);
    (normal, tangent, bitangent)
}

// ---------------------------------------------------------------------------
// Surface point
// ---------------------------------------------------------------------------

/// World position of a surface point: `dir * (radius + elevation)`. `dir` must
/// be unit (e.g. from `uv_to_dir`); the planet center is the caller's
/// responsibility (add it via `WorldPos` arithmetic / origin-shifting).
pub fn dir_to_surface(dir: DVec3, radius: f64, elevation: f64) -> WorldPos {
    WorldPos(dir * (radius + elevation))
}

// ---------------------------------------------------------------------------
// Floating origin (origin-shifting)
// ---------------------------------------------------------------------------

/// The f64 world-space point treated as the render origin. Render positions are
/// `world - origin` cast to f32, keeping the camera near 0 so f32 jitter at
/// planet scale is eliminated.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct OriginOffset(pub DVec3);

pub fn world_to_render(pos: WorldPos, origin: OriginOffset) -> RenderPos {
    pos.to_render(origin)
}

/// Snap the origin to the nearest `snap`-sized grid point, keeping the camera
/// inside the threshold sphere after a recenter instead of in a far corner of
/// a floor-quantized cube.
pub fn recenter(_origin: OriginOffset, camera: WorldPos, snap: f64) -> OriginOffset {
    let c = camera.0;
    let sx = (c.x / snap).round() * snap;
    let sy = (c.y / snap).round() * snap;
    let sz = (c.z / snap).round() * snap;
    OriginOffset(DVec3::new(sx, sy, sz))
}

/// True when the camera has drifted more than `threshold` from the origin and
/// `recenter` should be called.
pub fn needs_recenter(camera: WorldPos, origin: OriginOffset, threshold: f64) -> bool {
    (camera.0 - origin.0).length() > threshold
}

// ---------------------------------------------------------------------------
// Spherical quadtree cell key + neighbors
// ---------------------------------------------------------------------------

/// A leaf of a per-face quadtree. `(i,j)` is the cell within the face at `lod`;
/// `cells_per_edge(lod) = 2^lod` cells per edge, so lod 0 = the whole face.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct CellKey {
    pub face: u8,
    pub i: u32,
    pub j: u32,
    pub lod: u8,
}

pub fn cells_per_edge(lod: u8) -> u32 {
    1u32 << lod
}

/// Center uv of a cell: `((i+0.5)/N, (j+0.5)/N)`.
pub fn cell_uv_center(key: CellKey) -> (f64, f64) {
    let n = cells_per_edge(key.lod) as f64;
    ((key.i as f64 + 0.5) / n, (key.j as f64 + 0.5) / n)
}

/// Unit direction at a cell's center.
pub fn cell_to_dir(key: CellKey) -> DVec3 {
    let (u, v) = cell_uv_center(key);
    uv_to_dir(key.face, u, v)
}

/// World position of a cell center on a sphere of `radius` (elevation 0).
pub fn cell_center(key: CellKey, radius: f64) -> WorldPos {
    dir_to_surface(cell_to_dir(key), radius, 0.0)
}

/// Quantize a face/uv to the cell index at `lod` (clamped to valid range).
pub fn uv_to_cell(face: u8, u: f64, v: f64, lod: u8) -> CellKey {
    let n = cells_per_edge(lod) as f64;
    let max = (cells_per_edge(lod) - 1) as f64;
    let i = (u * n).floor().max(0.0).min(max) as u32;
    let j = (v * n).floor().max(0.0).min(max) as u32;
    CellKey { face, i, j, lod }
}

/// Quantize a direction to the cell it falls in at `lod`.
pub fn dir_to_cell(dir: DVec3, lod: u8) -> CellKey {
    let (face, u, v) = dir_to_uv(dir);
    uv_to_cell(face, u, v, lod)
}

/// Approximate world-space arc length of one cell along the sphere. A face
/// spans ~pi/2 edge-to-edge, so a cell is ~`radius * (pi/2) / 2^lod`. Cells are
/// not perfectly uniform on the spherified cube (denser near corners); this is
/// an average used for LOD budgeting, not an exact measure.
pub fn cell_size(lod: u8, radius: f64) -> f64 {
    radius * std::f64::consts::FRAC_PI_2 / cells_per_edge(lod) as f64
}

/// Which of the 4 cell edges a neighbor lies across.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum NeighborSide {
    NegU,
    PosU,
    NegV,
    PosV,
}

/// One neighbor of a cell. Works across face boundaries: stepping one cell in
/// uv extends past the face plane, `normalize` projects onto the adjacent face,
/// and `dir_to_uv`/`uv_to_cell` land in the correct neighbor cell there.
pub fn cell_neighbor(key: CellKey, side: NeighborSide) -> CellKey {
    let n = cells_per_edge(key.lod) as f64;
    let (uc, vc) = cell_uv_center(key);
    let step = 1.0 / n;
    let (nu, nv) = match side {
        NeighborSide::NegU => (uc - step, vc),
        NeighborSide::PosU => (uc + step, vc),
        NeighborSide::NegV => (uc, vc - step),
        NeighborSide::PosV => (uc, vc + step),
    };
    let dir = uv_to_dir(key.face, nu, nv);
    let (face, u, v) = dir_to_uv(dir);
    uv_to_cell(face, u, v, key.lod)
}

/// All 4 neighbors in order [NegU, PosU, NegV, PosV]. On a closed sphere every
/// cell always has exactly 4 neighbors (edge cells wrap to adjacent faces).
pub fn cell_neighbors(key: CellKey) -> [CellKey; 4] {
    [
        cell_neighbor(key, NeighborSide::NegU),
        cell_neighbor(key, NeighborSide::PosU),
        cell_neighbor(key, NeighborSide::NegV),
        cell_neighbor(key, NeighborSide::PosV),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::rng_from_seed;
    use rand::RngCore;

    fn rand_unit(rng: &mut impl RngCore) -> DVec3 {
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

    fn u2f(u: u64) -> f64 {
        (u as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    // 1.5 round-trip uv_to_dir o dir_to_uv within 1e-9 over 10k random dirs
    #[test]
    fn dir_uv_roundtrip() {
        let mut rng = rng_from_seed(0xC0FFEE);
        let mut max_err: f64 = 0.0;
        for _ in 0..10_000 {
            let d = rand_unit(&mut rng);
            let (f, u, v) = dir_to_uv(d);
            let d2 = uv_to_dir(f, u, v);
            let e = (d - d2).length();
            assert!(e < 1e-9, "round-trip err {e} too big at f={f} u={u} v={v}");
            max_err = max_err.max(e);
        }
        // also exercise exact face centers, edge midpoints and corners: the
        // *direction* must round-trip exactly. At edges/corners several faces
        // meet (ties), so dir_to_uv may return a different (valid) face — only
        // the face center is unambiguously owned by its own face.
        for face in 0..6 {
            for &(u, v) in &[
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.5, 0.5),
                (0.0, 0.5),
                (1.0, 0.5),
                (0.5, 0.0),
                (0.5, 1.0),
            ] {
                let d = uv_to_dir(face, u, v);
                let (f2, u2, v2) = dir_to_uv(d);
                assert!(f2 < 6);
                let d2 = uv_to_dir(f2, u2, v2);
                assert!(
                    (d - d2).length() < 1e-9,
                    "edge/corner dir round-trip failed at {face}/{u},{v}"
                );
                if u == 0.5 && v == 0.5 {
                    assert_eq!(f2, face, "face center not owned by its face");
                }
            }
        }
        let _ = max_err;
    }

    #[test]
    #[should_panic(expected = "finite non-zero direction")]
    fn dir_to_uv_rejects_zero_direction() {
        let _ = dir_to_uv(DVec3::ZERO);
    }

    // 1.6 tangent frame orthonormal within 1e-6 and matches finite-difference
    #[test]
    fn tangent_frame_orthonormal() {
        for face in 0..6u8 {
            for (u, v) in [(0.5, 0.5), (0.3, 0.7), (0.2, 0.2), (0.8, 0.6)] {
                let (n, t, b) = tangent_frame(face, u, v);
                assert!((n.length() - 1.0).abs() < 1e-6, "normal not unit");
                assert!((t.length() - 1.0).abs() < 1e-6, "tangent not unit");
                assert!((b.length() - 1.0).abs() < 1e-6, "bitangent not unit");
                assert!(n.dot(t).abs() < 1e-6, "n.t too big");
                assert!(n.dot(b).abs() < 1e-6, "n.b too big");
                assert!(t.dot(b).abs() < 1e-6, "t.b too big");
                // right-handed: cross(n,t) == b
                assert!((n.cross(t) - b).length() < 1e-6, "not right-handed");
                // finite-difference agreement (interior only)
                let eps = 1e-5;
                let du = (uv_to_dir(face, u + eps, v) - uv_to_dir(face, u - eps, v)) * (0.5 / eps);
                let du = (du - n * du.dot(n)).normalize();
                assert!(du.dot(t) > 1.0 - 1e-4, "tangent != finite-diff dir");
            }
        }
    }

    // 1.9 cell quantization round-trips
    #[test]
    fn cell_quantization_roundtrip() {
        let mut rng = rng_from_seed(0xBADF00D);
        for _ in 0..10_000 {
            let face = (rng.next_u64() % 6) as u8;
            let u = u2f(rng.next_u64()).abs(); // [0,1]-ish
            let v = u2f(rng.next_u64()).abs();
            let lod = (rng.next_u64() % 9) as u8;
            let k = uv_to_cell(face, u.clamp(0.0, 1.0), v.clamp(0.0, 1.0), lod);
            let d = cell_to_dir(k);
            let k2 = dir_to_cell(d, lod);
            assert_eq!(k, k2, "quant round-trip: {k:?} -> {k2:?}");
        }
    }

    // interior neighbors stay on the same face with adjacent indices
    #[test]
    fn interior_neighbors() {
        let lod = 3u8;
        let n = cells_per_edge(lod);
        for face in 0..6u8 {
            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    let k = CellKey { face, i, j, lod };
                    assert_eq!(
                        cell_neighbor(k, NeighborSide::NegU),
                        CellKey {
                            face,
                            i: i - 1,
                            j,
                            lod
                        }
                    );
                    assert_eq!(
                        cell_neighbor(k, NeighborSide::PosU),
                        CellKey {
                            face,
                            i: i + 1,
                            j,
                            lod
                        }
                    );
                    assert_eq!(
                        cell_neighbor(k, NeighborSide::NegV),
                        CellKey {
                            face,
                            i,
                            j: j - 1,
                            lod
                        }
                    );
                    assert_eq!(
                        cell_neighbor(k, NeighborSide::PosV),
                        CellKey {
                            face,
                            i,
                            j: j + 1,
                            lod
                        }
                    );
                }
            }
        }
    }

    // 1.10 edge neighbors wrap to the correct adjacent face across all 12 edges.
    // EXPECTED derived from the cube topology and cross-checked independently:
    // the shared edge is where |f_axis| == |adj_axis|.
    #[test]
    fn edge_neighbors_wrap() {
        const EXPECTED: [[u8; 4]; 6] = [
            [3, 2, 5, 4], // +X: NegU->-Y PosU->+Y NegV->-Z PosV->+Z
            [3, 2, 5, 4], // -X
            [1, 0, 5, 4], // +Y
            [1, 0, 5, 4], // -Y
            [1, 0, 3, 2], // +Z
            [1, 0, 3, 2], // -Z
        ];
        let lod = 3u8;
        let n = cells_per_edge(lod);
        let cell_ang = std::f64::consts::FRAC_PI_2 / n as f64;
        let sides = [
            NeighborSide::NegU,
            NeighborSide::PosU,
            NeighborSide::NegV,
            NeighborSide::PosV,
        ];
        for face in 0..6u8 {
            for (s, side) in sides.iter().enumerate() {
                let (i, j) = match s {
                    0 => (0u32, n / 2),
                    1 => (n - 1, n / 2),
                    2 => (n / 2, 0),
                    _ => (n / 2, n - 1),
                };
                let k = CellKey { face, i, j, lod };
                let nb = cell_neighbor(k, *side);
                assert_eq!(
                    nb.face, EXPECTED[face as usize][s],
                    "face {face} side {s:?} expected {} got {}",
                    EXPECTED[face as usize][s], nb.face
                );
                // independent geometric fact: shared edge center has |f_axis|==|adj_axis|
                let (eu, ev) = match s {
                    0 => (0.0, (j as f64 + 0.5) / n as f64),
                    1 => (1.0, (j as f64 + 0.5) / n as f64),
                    2 => ((i as f64 + 0.5) / n as f64, 0.0),
                    _ => ((i as f64 + 0.5) / n as f64, 1.0),
                };
                let edge_dir = uv_to_dir(face, eu, ev).normalize();
                let (faxis, _) = face_axis_sign(face);
                let adj_axis = if s == 0 || s == 1 {
                    FACE_U_AXIS[face as usize]
                } else {
                    FACE_V_AXIS[face as usize]
                };
                assert!(
                    (axis_comp(edge_dir, faxis).abs() - axis_comp(edge_dir, adj_axis).abs()).abs()
                        < 1e-9,
                    "edge not on shared boundary"
                );
                // neighbor center must sit within ~1 cell of the shared edge
                let ncenter = cell_to_dir(nb);
                let ang = ncenter.dot(edge_dir).clamp(-1.0, 1.0).acos();
                assert!(
                    ang < 1.0 * cell_ang,
                    "neighbor too far from edge: {ang} > {}",
                    cell_ang
                );
            }
        }
    }

    // cell_size shrinks by ~2x per lod level
    #[test]
    fn cell_size_scales() {
        let r = 12000.0;
        for lod in 0..10u8 {
            assert!(
                (cell_size(lod, r) - r * std::f64::consts::FRAC_PI_2 / cells_per_edge(lod) as f64)
                    .abs()
                    < 1e-9
            );
        }
        for lod in 0..9u8 {
            assert!((cell_size(lod, r) / cell_size(lod + 1, r) - 2.0).abs() < 1e-9);
        }
    }

    // origin-shifting: render pos = world - origin (f32); recenter snaps to grid
    #[test]
    fn origin_shifting() {
        let origin = OriginOffset(DVec3::new(100.0, 200.0, 300.0));
        let p = WorldPos::new(100.5, 200.5, 300.5);
        let r = world_to_render(p, origin);
        assert_eq!(r, RenderPos::new(0.5, 0.5, 0.5));
        assert!(!needs_recenter(p, origin, 1.0));
        assert!(needs_recenter(
            WorldPos::new(102.0, 200.0, 300.0),
            origin,
            1.0
        ));
        let o2 = recenter(origin, WorldPos::new(123.4, 0.0, 0.0), 10.0);
        assert_eq!(o2, OriginOffset(DVec3::new(120.0, 0.0, 0.0)));

        let earth_camera = WorldPos::new(3_181_969.6, 1_883_150.2, 5_189_924.1);
        let earth_origin = recenter(OriginOffset::default(), earth_camera, 1000.0);
        assert!(!needs_recenter(earth_camera, earth_origin, 1000.0));
        assert_eq!(recenter(earth_origin, earth_camera, 1000.0), earth_origin);
    }
}
