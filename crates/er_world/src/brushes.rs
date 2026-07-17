//! Deterministic brush-like landform displacement (Milestone 2.2).
//!
//! A fixed capped set of brushes (mountain, plateau, crater, canyon, ridge)
//! is generated deterministically from the elevation seed using a u32 integer
//! hash that is bit-identical between Rust and WGSL.  Brush displacement is
//! added to the macro (low-frequency) displacement so that mesh edges,
//! normals, and water classification all use the same composed field.
//!
//! CPU evaluation uses a 3-D voxel spatial index; WGSL loops the fixed cap.

use glam::Vec3;
use std::collections::HashMap;

// ---- Brush constants (mirrored in elevation.wgsl) ----

/// Fixed number of brushes generated from the seed.
pub const BRUSH_CAP: u32 = 64;
/// Minimum dot-product threshold (largest angular radius ~0.10 rad).
pub const BRUSH_DOT_THRESHOLD_MIN: f32 = 0.995;
/// Maximum dot-product threshold (smallest angular radius ~0.032 rad).
pub const BRUSH_DOT_THRESHOLD_MAX: f32 = 0.9995;
/// Minimum brush amplitude (field units).
pub const BRUSH_AMP_MIN: f32 = 0.3;
/// Maximum brush amplitude (field units).
pub const BRUSH_AMP_MAX: f32 = 1.5;
/// Clamp on total brush displacement to prevent extreme overlap.
pub const BRUSH_TOTAL_CAP: f32 = 3.0;
/// Minimum elongation ratio (along-axis / cross-axis) for Canyon & Ridge.
pub const BRUSH_ELONGATION_MIN: f32 = 2.0;
/// Maximum elongation ratio for Canyon & Ridge.
pub const BRUSH_ELONGATION_MAX: f32 = 4.0;
/// Spatial-index grid resolution per axis (cells span [-1,1]).
const BRUSH_GRID_N: i32 = 32;

// ---- Integer hash (identical in Rust u32 and WGSL u32) ----

/// Deterministic u32 hash.  Pure wrapping arithmetic — bit-identical in
/// Rust and WGSL.
pub fn brush_hash(seed: u32, index: u32) -> u32 {
    let mut h = seed.wrapping_add(index.wrapping_mul(0x9E3779B9));
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= h >> 16;
    h
}

/// Convert a u32 hash to [0, 1] in f32 using the top 24 bits (exact in f32).
fn to_unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16777215.0
}

// ---- Brush kinds ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BrushKind {
    Mountain = 0,
    Plateau = 1,
    Crater = 2,
    Canyon = 3,
    Ridge = 4,
}

impl BrushKind {
    pub fn from_u32(v: u32) -> Self {
        match v % 5 {
            0 => Self::Mountain,
            1 => Self::Plateau,
            2 => Self::Crater,
            3 => Self::Canyon,
            _ => Self::Ridge,
        }
    }
}

impl BrushKind {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---- Brush profile (f32 arithmetic, identical CPU/WGSL) ----

/// Brush displacement profile.  `t` is the normalized distance [0, 1] where
/// 0 = brush center, 1 = brush edge.  `amp` is the amplitude in field units.
///
/// All formulas use only `+ - * min max` so they are bit-identical in Rust
/// f32 and WGSL f32.
pub fn brush_profile(kind: BrushKind, t: f32, amp: f32) -> f32 {
    let falloff = 1.0 - t * t;
    match kind {
        BrushKind::Mountain => amp * falloff,
        BrushKind::Plateau => amp * (falloff * 3.0).min(1.0),
        BrushKind::Crater => amp * (t * t * 3.0 - 1.0) * falloff,
        BrushKind::Canyon => -amp * falloff * 0.5,
        BrushKind::Ridge => amp * falloff * falloff,
    }
}

// ---- Brush ----

#[derive(Clone, Copy, Debug)]
pub struct Brush {
    /// Unit direction of the brush center on the sphere (f32 for parity).
    pub center: Vec3,
    /// Dot-product threshold: brush is active when `dot(dir, center) > cos_radius`.
    pub cos_radius: f32,
    /// Brush kind (0–4).
    pub kind: BrushKind,
    /// Amplitude in field units.
    pub amplitude: f32,
    /// Elongation axis (unit, on tangent plane).  Only used by Canyon & Ridge.
    pub tangent: Vec3,
    /// Along-axis / cross-axis ratio (>= 1).  Only used by Canyon & Ridge.
    pub elongation: f32,
}

impl Brush {
    /// Compute normalized distance `t` in [0, 1] for a query direction, or
    /// `None` if the brush is inactive (outside the outer radial bound).
    ///
    /// Radial kinds use `t = (1-d)/(1-cos_r)`.
    /// Elongated kinds (Canyon, Ridge) use squared tangent-plane elliptical
    /// distance, normalized by `sin(radius)^2` so the profile reaches zero
    /// continuously at the outer spherical-cap boundary.
    pub fn eval_t(&self, dir: Vec3) -> Option<f32> {
        let d = dir.dot(self.center);
        if d <= self.cos_radius {
            return None;
        }
        let t = match self.kind {
            BrushKind::Mountain | BrushKind::Plateau | BrushKind::Crater => {
                (1.0 - d) / (1.0 - self.cos_radius)
            }
            BrushKind::Canyon | BrushKind::Ridge => {
                let delta = dir - self.center * d;
                let d_along = delta.dot(self.tangent);
                let bitangent = self.center.cross(self.tangent);
                let d_cross = delta.dot(bitangent);
                let r_sq = 1.0 - self.cos_radius * self.cos_radius;
                (d_along * d_along + d_cross * d_cross * self.elongation * self.elongation) / r_sq
            }
        };
        Some(t.clamp(0.0, 1.0))
    }
}

// ---- Brush set with spatial index ----

pub struct BrushSet {
    brushes: Vec<Brush>,
    /// Voxel grid: cell key → brush indices that overlap that cell.
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    grid_n: i32,
}

impl BrushSet {
    /// Generate the fixed capped brush set from a seed.
    pub fn from_seed(seed: u32) -> Self {
        let mut brushes = Vec::with_capacity(BRUSH_CAP as usize);
        for i in 0..BRUSH_CAP {
            let h0 = brush_hash(seed, i * 8);
            let h1 = brush_hash(seed, i * 8 + 1);
            let h2 = brush_hash(seed, i * 8 + 2);
            let h3 = brush_hash(seed, i * 8 + 3);
            let h4 = brush_hash(seed, i * 8 + 4);
            let h5 = brush_hash(seed, i * 8 + 5);
            let h6 = brush_hash(seed, i * 8 + 6);
            let h7 = brush_hash(seed, i * 8 + 7);

            let cx = to_unit(h0) * 2.0 - 1.0;
            let cy = to_unit(h1) * 2.0 - 1.0;
            let cz = to_unit(h2) * 2.0 - 1.0;
            let center = Vec3::new(cx, cy, cz).normalize();

            let kind = BrushKind::from_u32(h3 % 5);
            let cos_radius = BRUSH_DOT_THRESHOLD_MIN
                + to_unit(h3) * (BRUSH_DOT_THRESHOLD_MAX - BRUSH_DOT_THRESHOLD_MIN);
            let amplitude = BRUSH_AMP_MIN + to_unit(h4) * (BRUSH_AMP_MAX - BRUSH_AMP_MIN);

            // Tangent axis: random vector projected onto tangent plane at center.
            let tx = to_unit(h5) * 2.0 - 1.0;
            let ty = to_unit(h6) * 2.0 - 1.0;
            let tz = to_unit(h7) * 2.0 - 1.0;
            let raw_tangent = Vec3::new(tx, ty, tz);
            let tangent_projection = raw_tangent - center * raw_tangent.dot(center);
            let fallback_axis = if center.x.abs() < 0.9 {
                Vec3::X
            } else {
                Vec3::Y
            };
            let fallback_projection = fallback_axis - center * fallback_axis.dot(center);
            let tangent = if tangent_projection.length_squared() > 1e-12 {
                tangent_projection.normalize()
            } else {
                fallback_projection.normalize()
            };

            // Elongation from bottom 8 bits of h7 (independent of tangent z
            // which uses the top 24 bits via to_unit).
            let elongation = BRUSH_ELONGATION_MIN
                + ((h7 & 0xFF) as f32 / 255.0) * (BRUSH_ELONGATION_MAX - BRUSH_ELONGATION_MIN);

            brushes.push(Brush {
                center,
                cos_radius,
                kind,
                amplitude,
                tangent,
                elongation,
            });
        }

        let grid_n = BRUSH_GRID_N;
        let cell_size = 2.0 / grid_n as f32;
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        for (idx, brush) in brushes.iter().enumerate() {
            // Cartesian radius corresponding to the dot-product threshold:
            // |dir - center|^2 = 2 - 2*dot = 2*(1 - cos_radius)
            let cart_radius = (2.0 * (1.0 - brush.cos_radius)).sqrt();
            let cell_r = (cart_radius / cell_size).ceil() as i32;
            let (cx, cy, cz) = Self::cell_key(brush.center, grid_n);
            for dx in -cell_r..=cell_r {
                for dy in -cell_r..=cell_r {
                    for dz in -cell_r..=cell_r {
                        let x = cx + dx;
                        let y = cy + dy;
                        let z = cz + dz;
                        if x < 0 || x >= grid_n || y < 0 || y >= grid_n || z < 0 || z >= grid_n {
                            continue;
                        }
                        cells.entry((x, y, z)).or_default().push(idx);
                    }
                }
            }
        }

        Self {
            brushes,
            cells,
            grid_n,
        }
    }

    fn cell_key(dir: Vec3, grid_n: i32) -> (i32, i32, i32) {
        let r = grid_n as f32;
        let to_cell = |v: f32| -> i32 { ((v * 0.5 + 0.5) * r).floor() as i32 };
        (
            to_cell(dir.x).clamp(0, grid_n - 1),
            to_cell(dir.y).clamp(0, grid_n - 1),
            to_cell(dir.z).clamp(0, grid_n - 1),
        )
    }

    /// Evaluate brush displacement using the spatial index.
    pub fn displacement_indexed(&self, dir: Vec3) -> f32 {
        let key = Self::cell_key(dir, self.grid_n);
        let mut sum = 0.0_f32;
        if let Some(indices) = self.cells.get(&key) {
            for &idx in indices {
                let brush = &self.brushes[idx];
                if let Some(t) = brush.eval_t(dir) {
                    sum += brush_profile(brush.kind, t, brush.amplitude);
                }
            }
        }
        sum.clamp(-BRUSH_TOTAL_CAP, BRUSH_TOTAL_CAP)
    }

    /// Evaluate brush displacement by checking all brushes (exhaustive).
    pub fn displacement_exhaustive(&self, dir: Vec3) -> f32 {
        let mut sum = 0.0_f32;
        for brush in &self.brushes {
            if let Some(t) = brush.eval_t(dir) {
                sum += brush_profile(brush.kind, t, brush.amplitude);
            }
        }
        sum.clamp(-BRUSH_TOTAL_CAP, BRUSH_TOTAL_CAP)
    }

    /// Access the raw brush array (for tests).
    pub fn brushes(&self) -> &[Brush] {
        &self.brushes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::math::uv_to_dir;
    use er_core::rng::rng_from_seed;
    use glam::DVec3;
    use rand::RngCore;

    const TEST_SEED: u32 = 0x1818_0000;

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

    fn dir_to_f32(d: DVec3) -> Vec3 {
        Vec3::new(d.x as f32, d.y as f32, d.z as f32)
    }

    // ---- 1. Per-kind profile tests ----

    #[test]
    fn mountain_profile_positive_dome() {
        let amp = 1.0_f32;
        assert!((brush_profile(BrushKind::Mountain, 0.0, amp) - amp).abs() < 1e-6);
        assert!((brush_profile(BrushKind::Mountain, 1.0, amp) - 0.0).abs() < 1e-6);
        let mid = brush_profile(BrushKind::Mountain, 0.5, amp);
        assert!(
            mid > 0.0 && mid < amp,
            "mountain mid {mid} should be in (0, amp)"
        );
    }

    #[test]
    fn plateau_profile_flat_top() {
        let amp = 1.0_f32;
        assert!((brush_profile(BrushKind::Plateau, 0.0, amp) - amp).abs() < 1e-6);
        // Flat near center
        assert!((brush_profile(BrushKind::Plateau, 0.3, amp) - amp).abs() < 1e-6);
        assert!((brush_profile(BrushKind::Plateau, 1.0, amp) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn crater_profile_bowl_with_rim() {
        let amp = 1.0_f32;
        let center = brush_profile(BrushKind::Crater, 0.0, amp);
        assert!(center < 0.0, "crater center {center} must be negative");
        let edge = brush_profile(BrushKind::Crater, 1.0, amp);
        assert!(edge.abs() < 1e-6, "crater edge {edge} must be ~0");
        // Rim: there should be a positive value somewhere
        let mut max_rim = 0.0_f32;
        for i in 0..100 {
            let t = i as f32 / 100.0;
            max_rim = max_rim.max(brush_profile(BrushKind::Crater, t, amp));
        }
        assert!(
            max_rim > 0.0,
            "crater must have a positive rim, max={max_rim}"
        );
    }

    #[test]
    fn canyon_profile_negative_shallow() {
        let amp = 1.0_f32;
        let center = brush_profile(BrushKind::Canyon, 0.0, amp);
        assert!(center < 0.0, "canyon center {center} must be negative");
        assert!((center - (-amp * 0.5)).abs() < 1e-6);
        assert!((brush_profile(BrushKind::Canyon, 1.0, amp) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn ridge_profile_positive_steeper_than_mountain() {
        let amp = 1.0_f32;
        assert!((brush_profile(BrushKind::Ridge, 0.0, amp) - amp).abs() < 1e-6);
        assert!((brush_profile(BrushKind::Ridge, 1.0, amp) - 0.0).abs() < 1e-6);
        let mid_mtn = brush_profile(BrushKind::Mountain, 0.5, amp);
        let mid_ridge = brush_profile(BrushKind::Ridge, 0.5, amp);
        assert!(
            mid_ridge < mid_mtn,
            "ridge {mid_ridge} should be steeper (lower at mid) than mountain {mid_mtn}"
        );
    }

    #[test]
    fn elongated_profile_reaches_zero_at_spherical_cap_boundary() {
        let cos_radius = 0.995_f32;
        let radius = cos_radius.acos();
        let brush = Brush {
            center: Vec3::Z,
            cos_radius,
            kind: BrushKind::Canyon,
            amplitude: 1.0,
            tangent: Vec3::X,
            elongation: 3.0,
        };
        let inside_angle = radius - 1e-6;
        let inside = Vec3::new(inside_angle.sin(), 0.0, inside_angle.cos());
        let t = brush.eval_t(inside).expect("point just inside brush cap");
        let displacement = brush_profile(brush.kind, t, brush.amplitude);

        assert!(t > 0.999, "boundary distance {t} must approach one");
        assert!(
            displacement.abs() < 1e-3,
            "boundary displacement {displacement} must approach zero"
        );
    }

    // ---- 2. Determinism ----

    #[test]
    fn brush_set_deterministic() {
        let a = BrushSet::from_seed(TEST_SEED);
        let b = BrushSet::from_seed(TEST_SEED);
        assert_eq!(a.brushes().len(), b.brushes().len());
        for (ba, bb) in a.brushes().iter().zip(b.brushes().iter()) {
            assert_eq!(
                ba.center.x.to_bits(),
                bb.center.x.to_bits(),
                "center x mismatch"
            );
            assert_eq!(
                ba.center.y.to_bits(),
                bb.center.y.to_bits(),
                "center y mismatch"
            );
            assert_eq!(
                ba.center.z.to_bits(),
                bb.center.z.to_bits(),
                "center z mismatch"
            );
            assert_eq!(
                ba.cos_radius.to_bits(),
                bb.cos_radius.to_bits(),
                "cos_radius mismatch"
            );
            assert_eq!(ba.kind, bb.kind, "kind mismatch");
            assert_eq!(
                ba.amplitude.to_bits(),
                bb.amplitude.to_bits(),
                "amp mismatch"
            );
            assert_eq!(
                ba.tangent.x.to_bits(),
                bb.tangent.x.to_bits(),
                "tangent x mismatch"
            );
            assert_eq!(
                ba.tangent.y.to_bits(),
                bb.tangent.y.to_bits(),
                "tangent y mismatch"
            );
            assert_eq!(
                ba.tangent.z.to_bits(),
                bb.tangent.z.to_bits(),
                "tangent z mismatch"
            );
            assert_eq!(
                ba.elongation.to_bits(),
                bb.elongation.to_bits(),
                "elongation mismatch"
            );
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = BrushSet::from_seed(TEST_SEED);
        let b = BrushSet::from_seed(TEST_SEED ^ 0xDEAD);
        let mut any_diff = false;
        for (ba, bb) in a.brushes().iter().zip(b.brushes().iter()) {
            if ba.center.x.to_bits() != bb.center.x.to_bits() {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "different seeds produced identical brush centers");
    }

    // ---- 3. Boundedness ----

    #[test]
    fn displacement_bounded() {
        let set = BrushSet::from_seed(TEST_SEED);
        let dirs = rand_dirs(0xBEEF, 5000);
        for d in &dirs {
            let disp = set.displacement_indexed(dir_to_f32(*d));
            assert!(disp.is_finite(), "non-finite displacement");
            assert!(
                disp >= -BRUSH_TOTAL_CAP && disp <= BRUSH_TOTAL_CAP,
                "displacement {disp} outside [-{BRUSH_TOTAL_CAP}, {BRUSH_TOTAL_CAP}]"
            );
        }
    }

    #[test]
    fn all_five_kinds_present() {
        let set = BrushSet::from_seed(TEST_SEED);
        let mut seen = [false; 5];
        for b in set.brushes() {
            seen[b.kind.as_u32() as usize] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "brush kind {i} never generated");
        }
    }

    // ---- 4. Indexed == exhaustive ----

    #[test]
    fn indexed_matches_exhaustive() {
        let set = BrushSet::from_seed(TEST_SEED);
        let dirs = rand_dirs(0xCAFE, 10_000);
        let mut max_diff = 0.0_f32;
        for d in &dirs {
            let dir_f = dir_to_f32(*d);
            let indexed = set.displacement_indexed(dir_f);
            let exhaustive = set.displacement_exhaustive(dir_f);
            let diff = (indexed - exhaustive).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-6,
                "indexed vs exhaustive diff {diff} at dir {d:?}: {indexed} vs {exhaustive}"
            );
        }
        eprintln!("Indexed vs exhaustive max diff: {max_diff:.2e}");
    }

    // ---- 5. Cube-edge continuity ----

    #[test]
    fn brush_displacement_cube_edge_continuous() {
        let set = BrushSet::from_seed(0xACC_0000);
        for i in 0..32 {
            let t = (i as f64 + 0.5) / 32.0;
            let d_x = uv_to_dir(0, 1.0, t);
            let d_y = uv_to_dir(2, 1.0, t);
            let ex = set.displacement_indexed(dir_to_f32(d_x));
            let ey = set.displacement_indexed(dir_to_f32(d_y));
            assert!(
                (ex - ey).abs() < 1e-6,
                "brush edge discontinuity at i={i}: {ex} vs {ey}"
            );
        }
        for j in 0..32 {
            let t = (j as f64 + 0.5) / 32.0;
            let d_xn = uv_to_dir(0, t, 0.0);
            let d_zn = uv_to_dir(5, 1.0, t);
            let ex = set.displacement_indexed(dir_to_f32(d_xn));
            let ez = set.displacement_indexed(dir_to_f32(d_zn));
            assert!(
                (ex - ez).abs() < 1e-6,
                "brush edge discontinuity at j={j}: {ex} vs {ez}"
            );
        }
    }

    // ---- 6. Brush displacement is nonzero somewhere ----

    #[test]
    fn brush_displacement_is_active() {
        let set = BrushSet::from_seed(TEST_SEED);
        let dirs = rand_dirs(0xF00D, 5000);
        let mut nonzero = 0;
        for d in &dirs {
            let disp = set.displacement_indexed(dir_to_f32(*d));
            if disp.abs() > 0.01 {
                nonzero += 1;
            }
        }
        assert!(nonzero > 0, "brush displacement is never active");
        eprintln!(
            "Brush displacement active at {nonzero}/{} sample points",
            dirs.len()
        );
    }

    // ---- 7. Hash is pure u32 arithmetic (deterministic) ----

    #[test]
    fn hash_deterministic() {
        for i in 0..100u32 {
            let a = brush_hash(TEST_SEED, i);
            let b = brush_hash(TEST_SEED, i);
            assert_eq!(a, b, "hash not deterministic for index {i}");
        }
    }

    // ---- 8. Anisotropy: Canyon & Ridge are elongated ----

    /// Build a synthetic brush at the north pole with a known tangent axis
    /// and elongation for deterministic anisotropy testing.
    fn make_elongated_brush(kind: BrushKind, elongation: f32) -> Brush {
        Brush {
            center: Vec3::new(0.0, 0.0, 1.0),
            cos_radius: 0.995, // ~0.10 rad angular radius
            kind,
            amplitude: 1.0,
            tangent: Vec3::new(1.0, 0.0, 0.0), // along x-axis
            elongation,
        }
    }

    /// Find the angular extent (in radians) where displacement magnitude
    /// exceeds `threshold`, moving from center along a tangent direction.
    fn angular_extent(brush: &Brush, axis: Vec3, threshold: f32) -> f32 {
        let mut lo = 0.0_f32;
        let mut hi = 0.3_f32; // beyond max brush radius
        for _ in 0..50 {
            let mid = (lo + hi) * 0.5;
            let dir = (brush.center + axis * mid.tan()).normalize();
            let disp = match brush.eval_t(dir) {
                Some(t) => brush_profile(brush.kind, t, brush.amplitude),
                None => 0.0,
            };
            if disp.abs() > threshold {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    #[test]
    fn canyon_along_axis_longer_than_cross_axis() {
        let brush = make_elongated_brush(BrushKind::Canyon, 3.0);
        let bitangent = brush.center.cross(brush.tangent);
        let threshold = 0.01;
        let along = angular_extent(&brush, brush.tangent, threshold);
        let cross = angular_extent(&brush, bitangent, threshold);
        eprintln!(
            "Canyon along={along:.4} cross={cross:.4} ratio={:.2}",
            along / cross
        );
        assert!(
            along > cross * 1.5,
            "canyon along-axis ({along:.4}) must be materially > cross-axis ({cross:.4})"
        );
    }

    #[test]
    fn ridge_along_axis_longer_than_cross_axis() {
        let brush = make_elongated_brush(BrushKind::Ridge, 3.0);
        let bitangent = brush.center.cross(brush.tangent);
        let threshold = 0.01;
        let along = angular_extent(&brush, brush.tangent, threshold);
        let cross = angular_extent(&brush, bitangent, threshold);
        eprintln!(
            "Ridge along={along:.4} cross={cross:.4} ratio={:.2}",
            along / cross
        );
        assert!(
            along > cross * 1.5,
            "ridge along-axis ({along:.4}) must be materially > cross-axis ({cross:.4})"
        );
    }

    #[test]
    fn mountain_remains_radial() {
        // Mountain should have the same extent in all directions.
        let brush = make_elongated_brush(BrushKind::Mountain, 3.0);
        let bitangent = brush.center.cross(brush.tangent);
        let threshold = 0.01;
        let along = angular_extent(&brush, brush.tangent, threshold);
        let cross = angular_extent(&brush, bitangent, threshold);
        // Mountain ignores elongation, so extents should be equal.
        assert!(
            (along - cross).abs() < 0.001,
            "mountain should be radial: along={along:.4} cross={cross:.4}"
        );
    }

    #[test]
    fn elongated_brushes_in_generated_set() {
        let set = BrushSet::from_seed(TEST_SEED);
        let mut found_elongated = false;
        for b in set.brushes() {
            if b.kind == BrushKind::Canyon || b.kind == BrushKind::Ridge {
                assert!(
                    b.elongation >= BRUSH_ELONGATION_MIN && b.elongation <= BRUSH_ELONGATION_MAX,
                    "elongation {} outside [{}, {}]",
                    b.elongation,
                    BRUSH_ELONGATION_MIN,
                    BRUSH_ELONGATION_MAX
                );
                // Tangent must be unit and orthogonal to center.
                assert!((b.tangent.length() - 1.0).abs() < 1e-5, "tangent not unit");
                assert!(
                    b.tangent.dot(b.center).abs() < 1e-5,
                    "tangent not orthogonal to center"
                );
                found_elongated = true;
            }
        }
        assert!(found_elongated, "no canyon or ridge brushes generated");
    }
}
