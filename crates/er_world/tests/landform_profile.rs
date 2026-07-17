//! Milestone 2.2 landform profile, mask property, and brush-admission tests.
//!
//! These tests measure the composed metric elevation field with fixed seeds,
//! then assert broad-but-meaningful bounds that catch regressions without
//! encoding one exact machine result.

use er_core::math::uv_to_dir;
use er_core::rng::rng_from_seed;
use er_core::seed::PlanetSeed;
use er_world::brushes::{brush_profile, BrushKind, BrushSet, BRUSH_CAP, BRUSH_TOTAL_CAP};
use er_world::elevation::{
    elevation_at, elevation_metric_full_eval, elevation_params, ElevationNoise,
};
use er_world::params::planet_params;
use er_world::terrain_space::metric_surface_point;
use er_world::terrain_space::{
    METRIC_DRAINAGE_AMP, METRIC_RIDGE_DETAIL_AMP, METRIC_TALUS_AMP, METRIC_TECTONIC_AMP,
    METRIC_VALLEY_AMP,
};
use glam::DVec3;
use rand::RngCore;

const TEST_R: f64 = 6_371_000.0;
const ELEVATION_SCALE_M: f64 = 1000.0;
const PROFILE_SEED: PlanetSeed = PlanetSeed(0x1818);
const PROFILE_SAMPLE_COUNT: usize = 20_000;
const SLOPE_SAMPLE_COUNT: usize = 2_000;
const SLOPE_STEP_M: f64 = 100.0;

fn u2f(u: u64) -> f64 {
    (u as f64 / u64::MAX as f64) * 2.0 - 1.0
}

fn rand_dirs(seed: u64, count: usize) -> Vec<DVec3> {
    let mut rng = rng_from_seed(seed);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let x = u2f(rng.next_u64());
        let y = u2f(rng.next_u64());
        let z = u2f(rng.next_u64());
        let v = DVec3::new(x, y, z);
        let l2 = v.length_squared();
        if l2 > 1e-6 && l2 < 1.0 {
            out.push(v.normalize());
        }
    }
    out
}

fn metric_points(seed: u64, count: usize) -> Vec<DVec3> {
    rand_dirs(seed, count)
        .iter()
        .map(|d| metric_surface_point(*d, TEST_R))
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// =====================================================================
// 1. Height profile in meters with calibrated bounds
// =====================================================================

#[test]
fn elevation_profile_within_approved_bounds() {
    let params = elevation_params(PROFILE_SEED);
    let climate = planet_params(PROFILE_SEED);
    let noise = ElevationNoise::new_metric(&params);
    let points = metric_points(0x511, PROFILE_SAMPLE_COUNT);
    let sea_level_m = climate.sea_level * ELEVATION_SCALE_M;

    let mut elevs_m: Vec<f64> = points
        .iter()
        .map(|p| elevation_at(*p, &noise) * ELEVATION_SCALE_M)
        .collect();
    elevs_m.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let count = elevs_m.len();
    let min = elevs_m[0];
    let max = elevs_m[count - 1];
    let p1 = percentile(&elevs_m, 0.01);
    let p10 = percentile(&elevs_m, 0.10);
    let p50 = percentile(&elevs_m, 0.50);
    let p90 = percentile(&elevs_m, 0.90);
    let p99 = percentile(&elevs_m, 0.99);
    let ocean_fraction = elevs_m.iter().filter(|&&e| e < sea_level_m).count() as f64 / count as f64;

    eprintln!(
        "Height profile (m, sea_level={sea_level_m:.1}m, n={count}): \
         min={min:.0} p1={p1:.0} p10={p10:.0} p50={p50:.0} \
         p90={p90:.0} p99={p99:.0} max={max:.0} ocean_frac={ocean_fraction:.3}"
    );

    // Calibrated bounds in meters.  Theoretical range: continental ±7000 m,
    // mountain 0..+2500, tectonic 0..+1500, residual ±~2000 plus carving.
    // Practical observed: min≈−5900, max≈+7500, ocean≈46%.
    assert!(min >= -12_000.0, "min {min:.0}m below -12000m");
    assert!(max <= 12_000.0, "max {max:.0}m above 12000m");
    assert!(
        p1 >= -10_000.0 && p1 <= -1_000.0,
        "p1 {p1:.0}m outside [-10000, -1000]"
    );
    assert!(
        p50 >= -1_500.0 && p50 <= 1_500.0,
        "p50 {p50:.0}m outside [-1500, 1500]"
    );
    assert!(
        p99 >= 3_000.0 && p99 <= 10_000.0,
        "p99 {p99:.0}m outside [3000, 10000]"
    );
    assert!(
        ocean_fraction >= 0.30 && ocean_fraction <= 0.70,
        "ocean_fraction {ocean_fraction:.3} outside [0.30, 0.70]"
    );
    assert!(max > p99, "max must exceed p99");
    assert!(p1 < p50 && p50 < p99, "percentiles must be ordered");
}

// =====================================================================
// 2. Slope histogram in degrees with Earth-like bounds
// =====================================================================

#[test]
fn slope_histogram_within_approved_bounds() {
    let params = elevation_params(PROFILE_SEED);
    let noise = ElevationNoise::new_metric(&params);
    let dirs = rand_dirs(0x5109E, SLOPE_SAMPLE_COUNT);

    let step = SLOPE_STEP_M;
    let mut slopes_deg: Vec<f64> = Vec::with_capacity(dirs.len());

    for dir in &dirs {
        let pos = metric_surface_point(*dir, TEST_R);
        let n = pos.normalize();
        let t1 = if n.x.abs() < 0.9 {
            DVec3::new(1.0, 0.0, 0.0).cross(n)
        } else {
            DVec3::new(0.0, 1.0, 0.0).cross(n)
        }
        .normalize();
        let t2 = n.cross(t1);

        // Reproject offset samples onto the sphere so we measure true
        // surface slope, not off-sphere tangent-plane slope.
        let eval_at_offset = |offset: DVec3| -> f64 {
            let p = (pos + offset).normalize() * TEST_R;
            elevation_at(p, &noise) * ELEVATION_SCALE_M
        };

        let e_tp = eval_at_offset(t1 * step);
        let e_tm = eval_at_offset(-t1 * step);
        let e_sp = eval_at_offset(t2 * step);
        let e_sm = eval_at_offset(-t2 * step);

        // rise/run in meters: elevation difference over horizontal distance.
        let rise1 = (e_tp - e_tm) / (2.0 * step);
        let rise2 = (e_sp - e_sm) / (2.0 * step);
        let slope_ratio = (rise1 * rise1 + rise2 * rise2).sqrt();
        let slope_deg = slope_ratio.atan().to_degrees();
        slopes_deg.push(slope_deg);
    }

    slopes_deg.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let sp50 = percentile(&slopes_deg, 0.50);
    let sp95 = percentile(&slopes_deg, 0.95);
    let sp99 = percentile(&slopes_deg, 0.99);
    let smax = slopes_deg[slopes_deg.len() - 1];
    let frac_gt30 =
        slopes_deg.iter().filter(|&&s| s > 30.0).count() as f64 / slopes_deg.len() as f64;
    let frac_gt60 =
        slopes_deg.iter().filter(|&&s| s > 60.0).count() as f64 / slopes_deg.len() as f64;

    eprintln!(
        "Slope histogram (deg, step={step}m, n={}): \
         p50={sp50:.1}° p95={sp95:.1}° p99={sp99:.1}° max={smax:.1}° \
         frac>30°={frac_gt30:.3} frac>60°={frac_gt60:.4}",
        slopes_deg.len()
    );

    // Earth-like readable terrain: median slopes single digits to low
    // teens, p95 below ~45°, nonzero cliff fraction, very small >60°.
    assert!(
        sp50 > 1.0 && sp50 < 20.0,
        "median slope {sp50:.1}° outside [1, 20]"
    );
    assert!(sp95 < 50.0, "p95 slope {sp95:.1}° above 50°");
    assert!(smax < 85.0, "max slope {smax:.1}° above 85°");
    assert!(smax.is_finite(), "max slope not finite");
    assert!(frac_gt30 > 0.001, "no slopes >30° (terrain too flat)");
    assert!(frac_gt60 < 0.05, "too many slopes >60° ({frac_gt60:.4})");
    assert!(sp99 > sp50, "p99 must exceed p50");
}

// =====================================================================
// 3. Mask property tests: finite, bounded [0,1], non-constant,
//    deterministic, cube-edge continuous
// =====================================================================

fn sample_masks(pos: DVec3, noise: &ElevationNoise) -> [f32; 6] {
    let s = elevation_metric_full_eval(pos, noise);
    [
        s.tectonic_belt,
        s.drainage,
        s.erosion,
        s.ridge_mask,
        s.valley_mask,
        s.talus_raw,
    ]
}

const MASK_NAMES: [&str; 6] = [
    "tectonic_belt",
    "drainage",
    "erosion",
    "ridge_mask",
    "valley_mask",
    "talus_raw",
];

#[test]
fn masks_finite_bounded_nonconstant_deterministic() {
    let params = elevation_params(PROFILE_SEED);
    let noise = ElevationNoise::new_metric(&params);
    let points = metric_points(0xA45, 500);

    for (mi, &name) in MASK_NAMES.iter().enumerate() {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut all_same = true;
        let first = sample_masks(points[0], &noise)[mi];

        for p in &points {
            let pass1 = sample_masks(*p, &noise);
            let pass2 = sample_masks(*p, &noise);
            let v = pass1[mi];

            assert!(v.is_finite(), "{name}: non-finite value {v}");

            // All masks bounded in [0, 1] (talus_raw is signed noise,
            // but it's stored as the raw noise value in [-1,1] — check that
            // separately).
            if name == "talus_raw" {
                assert!(v >= -1.0 && v <= 1.0, "{name}: {v} outside [-1,1]");
            } else {
                assert!(v >= 0.0 && v <= 1.0, "{name}: {v} outside [0,1]");
            }

            assert_eq!(
                v.to_bits(),
                pass2[mi].to_bits(),
                "{name}: not deterministic"
            );

            if v.to_bits() != first.to_bits() {
                all_same = false;
            }
            min = min.min(v);
            max = max.max(v);
        }

        assert!(!all_same, "{name}: constant across all samples");
        assert!(min < max, "{name}: min == max ({min})");
    }
}

#[test]
fn masks_cube_edge_continuous() {
    let params = elevation_params(PlanetSeed(0xACC));
    let noise = ElevationNoise::new_metric(&params);

    for i in 0..32 {
        let t = (i as f64 + 0.5) / 32.0;
        let d_x = uv_to_dir(0, 1.0, t);
        let d_y = uv_to_dir(2, 1.0, t);
        let px = metric_surface_point(d_x, TEST_R);
        let py = metric_surface_point(d_y, TEST_R);
        let mx = sample_masks(px, &noise);
        let my = sample_masks(py, &noise);
        for (mi, &name) in MASK_NAMES.iter().enumerate() {
            assert!(
                (mx[mi] - my[mi]).abs() < 1e-9,
                "{name} edge discontinuity at i={i}: {} vs {}",
                mx[mi],
                my[mi]
            );
        }
    }

    for j in 0..32 {
        let t = (j as f64 + 0.5) / 32.0;
        let d_xn = uv_to_dir(0, t, 0.0);
        let d_zn = uv_to_dir(5, 1.0, t);
        let px = metric_surface_point(d_xn, TEST_R);
        let pz = metric_surface_point(d_zn, TEST_R);
        let mx = sample_masks(px, &noise);
        let mz = sample_masks(pz, &noise);
        for (mi, &name) in MASK_NAMES.iter().enumerate() {
            assert!(
                (mx[mi] - mz[mi]).abs() < 1e-9,
                "{name} edge discontinuity at j={j}: {} vs {}",
                mx[mi],
                mz[mi]
            );
        }
    }
}

// =====================================================================
// 4. Directional contribution tests: valley/drainage carve, ridge uplifts
// =====================================================================

#[test]
fn valley_and_drainage_contributions_are_active_carving() {
    let params = elevation_params(PROFILE_SEED);
    let noise = ElevationNoise::new_metric(&params);
    let points = metric_points(0xB405, 2_000);

    let mut active_valleys = 0;
    let mut active_drainage = 0;
    for p in &points {
        let s = elevation_metric_full_eval(*p, &noise);
        let valley_disp = -(s.valley_mask as f64) * METRIC_VALLEY_AMP;
        assert!(
            valley_disp <= 1e-12,
            "valley displacement {valley_disp} must be <= 0 (carving only)"
        );
        let drainage_disp = -(s.drainage as f64) * METRIC_DRAINAGE_AMP;
        assert!(
            drainage_disp <= 1e-12,
            "drainage displacement {drainage_disp} must be <= 0 (carving only)"
        );
        active_valleys += usize::from(valley_disp < -0.05);
        active_drainage += usize::from(drainage_disp < -0.05);
    }
    assert!(
        active_valleys > points.len() / 10,
        "valley carving is not active enough"
    );
    assert!(
        active_drainage > points.len() / 10,
        "drainage carving is not active enough"
    );
}

#[test]
fn ridge_and_tectonic_contributions_are_active_uplift() {
    let params = elevation_params(PROFILE_SEED);
    let noise = ElevationNoise::new_metric(&params);
    let points = metric_points(0xB405, 2_000);

    let mut active_ridges = 0;
    let mut active_tectonics = 0;
    for p in &points {
        let s = elevation_metric_full_eval(*p, &noise);
        let ridge_disp = s.ridge_mask as f64 * METRIC_RIDGE_DETAIL_AMP;
        assert!(
            ridge_disp >= -1e-12,
            "ridge displacement {ridge_disp} must be >= 0 (uplift only)"
        );
        let tectonic_disp = s.tectonic_belt as f64 * METRIC_TECTONIC_AMP;
        assert!(
            tectonic_disp >= -1e-12,
            "tectonic displacement {tectonic_disp} must be >= 0 (uplift only)"
        );
        active_ridges += usize::from(ridge_disp > 0.05);
        active_tectonics += usize::from(tectonic_disp > 0.1);
    }
    assert!(
        active_ridges > points.len() / 10,
        "ridge uplift is not active enough"
    );
    assert!(
        active_tectonics > points.len() / 10,
        "tectonic uplift is not active enough"
    );
}

#[test]
fn landform_layer_max_contributions_documented() {
    let params = elevation_params(PROFILE_SEED);
    let noise = ElevationNoise::new_metric(&params);
    let points = metric_points(0xB405, 10_000);

    let mut max_tectonic: f64 = 0.0;
    let mut max_ridge: f64 = 0.0;
    let mut max_valley: f64 = 0.0;
    let mut max_drainage: f64 = 0.0;
    let mut max_talus: f64 = 0.0;
    let mut max_macro: f64 = 0.0;
    let mut max_residual: f64 = 0.0;

    for p in &points {
        let s = elevation_metric_full_eval(*p, &noise);
        // Tectonic is positive (belt * amp).
        max_tectonic = max_tectonic.max(s.tectonic_belt as f64 * METRIC_TECTONIC_AMP);
        // Ridge is positive (mask * amp).
        max_ridge = max_ridge.max(s.ridge_mask as f64 * METRIC_RIDGE_DETAIL_AMP);
        // Valley carving is negative, measure absolute magnitude.
        max_valley = max_valley.max(s.valley_mask as f64 * METRIC_VALLEY_AMP);
        // Drainage carving is negative, measure absolute magnitude.
        max_drainage = max_drainage.max(s.drainage as f64 * METRIC_DRAINAGE_AMP);
        // Talus is signed.
        max_talus = max_talus.max((s.talus_raw as f64 * METRIC_TALUS_AMP).abs());
        max_macro = max_macro.max(s.macro_displacement.abs());
        max_residual = max_residual.max(s.residual_displacement.abs());
    }

    eprintln!(
        "Layer max contributions (field units): \
         tectonic={max_tectonic:.3} ridge={max_ridge:.3} valley={max_valley:.3} \
         drainage={max_drainage:.3} talus={max_talus:.3} \
         macro={max_macro:.3} residual={max_residual:.3}"
    );

    // Each layer must contribute meaningfully.
    assert!(max_tectonic > 0.1, "tectonic contribution too small");
    assert!(max_ridge > 0.05, "ridge contribution too small");
    assert!(max_valley > 0.05, "valley contribution too small");
    assert!(max_drainage > 0.05, "drainage contribution too small");
    assert!(max_talus > 0.02, "talus contribution too small");

    // Contributions must not exceed amplitude ceilings.
    assert!(max_tectonic <= 1.5 + 1e-6, "tectonic exceeds amp");
    assert!(max_ridge <= 0.3 + 1e-6, "ridge exceeds amp");
    assert!(max_valley <= 0.2 + 1e-6, "valley exceeds amp");
    assert!(max_drainage <= 0.2 + 1e-6, "drainage exceeds amp");
    assert!(max_talus <= 0.08 + 1e-6, "talus exceeds amp");

    // Macro must dominate residual.
    assert!(
        max_macro > max_residual,
        "macro ({max_macro}) must dominate residual ({max_residual})"
    );
}

// =====================================================================
// 5. Cube-edge seam effect and macro/residual consistency
// =====================================================================

#[test]
fn full_metric_field_cube_edge_seam_effect() {
    let params = elevation_params(PlanetSeed(0xACC));
    let noise = ElevationNoise::new_metric(&params);

    let mut max_seam: f64 = 0.0;

    for i in 0..64 {
        let t = (i as f64 + 0.5) / 64.0;
        let d_x = uv_to_dir(0, 1.0, t);
        let d_y = uv_to_dir(2, 1.0, t);
        let px = metric_surface_point(d_x, TEST_R);
        let py = metric_surface_point(d_y, TEST_R);
        let ex = elevation_at(px, &noise);
        let ey = elevation_at(py, &noise);
        let diff = (ex - ey).abs();
        max_seam = max_seam.max(diff);
        assert!(
            diff < 1e-9,
            "full metric seam at i={i}: {ex} vs {ey} (diff {diff})"
        );
    }

    for j in 0..64 {
        let t = (j as f64 + 0.5) / 64.0;
        let d_xn = uv_to_dir(0, t, 0.0);
        let d_zn = uv_to_dir(5, 1.0, t);
        let px = metric_surface_point(d_xn, TEST_R);
        let pz = metric_surface_point(d_zn, TEST_R);
        let ex = elevation_at(px, &noise);
        let ez = elevation_at(pz, &noise);
        let diff = (ex - ez).abs();
        max_seam = max_seam.max(diff);
        assert!(
            diff < 1e-9,
            "full metric seam at j={j}: {ex} vs {ez} (diff {diff})"
        );
    }

    eprintln!("Full metric field max seam effect: {max_seam:.12} (tolerance 1e-9)");
    assert!(max_seam < 1e-9, "seam effect {max_seam} exceeds 1e-9");
}

#[test]
fn metric_macro_residual_split_is_consistent() {
    let params = elevation_params(PROFILE_SEED);
    let noise = ElevationNoise::new_metric(&params);
    let points = metric_points(0x5F17, 500);

    for p in &points {
        let s = elevation_metric_full_eval(*p, &noise);
        let composed = s.macro_displacement + s.residual_displacement;
        let direct = elevation_at(*p, &noise);
        assert!(
            (composed - direct).abs() < 1e-10,
            "macro+residual != elevation_at: {composed} vs {direct}"
        );
    }
}

// =====================================================================
// 6. Brush landform tests (Milestone 2.2)
// =====================================================================

fn dir_to_f32(d: DVec3) -> glam::Vec3 {
    glam::Vec3::new(d.x as f32, d.y as f32, d.z as f32)
}

/// Per-kind profile shape tests: each kind has the correct sign and
/// boundary behaviour at t=0 (center) and t=1 (edge).
#[test]
fn brush_per_kind_profile_shapes() {
    let amp = 1.0_f32;

    // Mountain: positive dome, peak at center, zero at edge.
    assert!((brush_profile(BrushKind::Mountain, 0.0, amp) - amp).abs() < 1e-6);
    assert!(brush_profile(BrushKind::Mountain, 0.5, amp) > 0.0);
    assert!(brush_profile(BrushKind::Mountain, 1.0, amp).abs() < 1e-6);

    // Plateau: flat top near center, zero at edge.
    assert!((brush_profile(BrushKind::Plateau, 0.0, amp) - amp).abs() < 1e-6);
    assert!((brush_profile(BrushKind::Plateau, 0.3, amp) - amp).abs() < 1e-6);
    assert!(brush_profile(BrushKind::Plateau, 1.0, amp).abs() < 1e-6);

    // Crater: negative at center, zero at edge, positive rim somewhere.
    assert!(brush_profile(BrushKind::Crater, 0.0, amp) < 0.0);
    assert!(brush_profile(BrushKind::Crater, 1.0, amp).abs() < 1e-6);
    let mut max_rim = 0.0_f32;
    for i in 0..100 {
        let t = i as f32 / 100.0;
        max_rim = max_rim.max(brush_profile(BrushKind::Crater, t, amp));
    }
    assert!(max_rim > 0.0, "crater must have positive rim");

    // Canyon: negative, half amplitude at center, zero at edge.
    assert!((brush_profile(BrushKind::Canyon, 0.0, amp) - (-0.5)).abs() < 1e-6);
    assert!(brush_profile(BrushKind::Canyon, 1.0, amp).abs() < 1e-6);

    // Ridge: positive, steeper than mountain at mid-range.
    assert!((brush_profile(BrushKind::Ridge, 0.0, amp) - amp).abs() < 1e-6);
    assert!(brush_profile(BrushKind::Ridge, 1.0, amp).abs() < 1e-6);
    assert!(
        brush_profile(BrushKind::Ridge, 0.5, amp) < brush_profile(BrushKind::Mountain, 0.5, amp),
        "ridge should be steeper than mountain"
    );
}

/// Brush set is deterministic: same seed → identical brushes.
#[test]
fn brush_set_deterministic() {
    let seed = elevation_params(PROFILE_SEED).seed as u32;
    let a = BrushSet::from_seed(seed);
    let b = BrushSet::from_seed(seed);
    assert_eq!(a.brushes().len(), b.brushes().len());
    for (ba, bb) in a.brushes().iter().zip(b.brushes().iter()) {
        assert_eq!(ba.center.x.to_bits(), bb.center.x.to_bits());
        assert_eq!(ba.center.y.to_bits(), bb.center.y.to_bits());
        assert_eq!(ba.center.z.to_bits(), bb.center.z.to_bits());
        assert_eq!(ba.cos_radius.to_bits(), bb.cos_radius.to_bits());
        assert_eq!(ba.kind, bb.kind);
        assert_eq!(ba.amplitude.to_bits(), bb.amplitude.to_bits());
        assert_eq!(ba.tangent.x.to_bits(), bb.tangent.x.to_bits());
        assert_eq!(ba.tangent.y.to_bits(), bb.tangent.y.to_bits());
        assert_eq!(ba.tangent.z.to_bits(), bb.tangent.z.to_bits());
        assert_eq!(ba.elongation.to_bits(), bb.elongation.to_bits());
    }
}

/// All five brush kinds are represented in the generated set.
#[test]
fn brush_all_five_kinds_present() {
    let seed = elevation_params(PROFILE_SEED).seed as u32;
    let set = BrushSet::from_seed(seed);
    let mut seen = [false; 5];
    for b in set.brushes() {
        seen[b.kind as u8 as usize] = true;
    }
    for (i, &s) in seen.iter().enumerate() {
        assert!(s, "brush kind {i} never generated");
    }
}

/// Brush displacement is bounded within [-BRUSH_TOTAL_CAP, BRUSH_TOTAL_CAP].
#[test]
fn brush_displacement_bounded() {
    let seed = elevation_params(PROFILE_SEED).seed as u32;
    let set = BrushSet::from_seed(seed);
    let dirs = rand_dirs(0xBEEF, 5000);
    for d in &dirs {
        let disp = set.displacement_indexed(dir_to_f32(*d));
        assert!(disp.is_finite());
        assert!(
            disp >= -BRUSH_TOTAL_CAP && disp <= BRUSH_TOTAL_CAP,
            "displacement {disp} outside bounds"
        );
    }
}

/// Indexed evaluation matches exhaustive evaluation exactly.
#[test]
fn brush_indexed_matches_exhaustive() {
    let seed = elevation_params(PROFILE_SEED).seed as u32;
    let set = BrushSet::from_seed(seed);
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
            "indexed vs exhaustive: {indexed} vs {exhaustive}"
        );
    }
    eprintln!("Brush indexed vs exhaustive max diff: {max_diff:.2e}");
}

/// Brush displacement is continuous across cube-face edges.
#[test]
fn brush_displacement_cube_edge_continuous() {
    let seed = elevation_params(PlanetSeed(0xACC)).seed as u32;
    let set = BrushSet::from_seed(seed);
    for i in 0..32 {
        let t = (i as f64 + 0.5) / 32.0;
        let d_x = uv_to_dir(0, 1.0, t);
        let d_y = uv_to_dir(2, 1.0, t);
        let ex = set.displacement_indexed(dir_to_f32(d_x));
        let ey = set.displacement_indexed(dir_to_f32(d_y));
        assert!((ex - ey).abs() < 1e-6, "brush seam at i={i}: {ex} vs {ey}");
    }
    for j in 0..32 {
        let t = (j as f64 + 0.5) / 32.0;
        let d_xn = uv_to_dir(0, t, 0.0);
        let d_zn = uv_to_dir(5, 1.0, t);
        let ex = set.displacement_indexed(dir_to_f32(d_xn));
        let ez = set.displacement_indexed(dir_to_f32(d_zn));
        assert!((ex - ez).abs() < 1e-6, "brush seam at j={j}: {ex} vs {ez}");
    }
}

/// Brush displacement is included in macro_displacement (macro inclusion).
#[test]
fn brush_displacement_included_in_macro() {
    let params = elevation_params(PROFILE_SEED);
    let noise = ElevationNoise::new_metric(&params);
    let points = metric_points(0xF00D, 5000);

    let mut any_nonzero = false;
    for p in &points {
        let s = elevation_metric_full_eval(*p, &noise);
        // brush_displacement field must be populated.
        assert!(s.brush_displacement.is_finite());
        // full_elevation must equal macro + residual (brush is in macro).
        let composed = s.macro_displacement + s.residual_displacement;
        assert!(
            (composed - s.full_elevation).abs() < 1e-10,
            "macro+residual != full: {composed} vs {}",
            s.full_elevation
        );
        if s.brush_displacement.abs() > 0.01 {
            any_nonzero = true;
        }
    }
    assert!(any_nonzero, "brush displacement never active");
}

/// Brush displacement is active at a meaningful fraction of sample points.
#[test]
fn brush_displacement_active_fraction() {
    let seed = elevation_params(PROFILE_SEED).seed as u32;
    let set = BrushSet::from_seed(seed);
    let dirs = rand_dirs(0xF00D, 5000);
    let mut active = 0;
    for d in &dirs {
        if set.displacement_indexed(dir_to_f32(*d)).abs() > 0.01 {
            active += 1;
        }
    }
    let frac = active as f64 / dirs.len() as f64;
    eprintln!(
        "Brush displacement active at {active}/{} ({frac:.3})",
        dirs.len()
    );
    assert!(active > 0, "brush displacement never active");
}

/// Brush cap is the expected fixed value.
#[test]
fn brush_cap_is_fixed() {
    assert_eq!(BRUSH_CAP, 64);
}
