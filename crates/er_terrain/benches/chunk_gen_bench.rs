use criterion::{black_box, criterion_group, criterion_main, Criterion};
use er_core::config::PLANET_RADIUS_DEFAULT;
use er_core::math::CellKey;
use er_terrain::generate_chunk_mesh;
use er_terrain::lod::screen_error;
use glam::DVec3;

/// The `(face=0, i=0, j=0)` cell at the given quadtree `lod`.
fn key_at(lod: u8) -> CellKey {
    CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod,
    }
}

/// Cost of generating one terrain chunk mesh (positions + skirts + indices) at
/// a few quadtree depths. `generate_chunk_mesh` is LOD-independent in work per
/// chunk (fixed 17x17 grid + 4 skirts), but the benchmark establishes a stable
/// baseline across the depth range the quadtree actually selects at.
fn bench_generate_chunk_mesh(c: &mut Criterion) {
    let radius = PLANET_RADIUS_DEFAULT;

    for &lod in &[0u8, 6, 12] {
        let key = key_at(lod);
        c.bench_function(format!("generate_chunk_mesh depth={lod}").as_str(), |b| {
            b.iter(|| black_box(generate_chunk_mesh(black_box(key), black_box(radius))))
        });
    }
}

/// Cost of the pure LOD screen-error metric at a few depths. `screen_error` is
/// a trivially callable pure fn (no Bevy world / queries), so it is safe to
/// micro-benchmark here.
fn bench_screen_error(c: &mut Criterion) {
    let radius = PLANET_RADIUS_DEFAULT;
    // Camera just outside the +X face (face 0) of the planet.
    let camera = DVec3::new(radius * 2.0, 0.0, 0.0);

    for &lod in &[0u8, 6, 12] {
        let key = key_at(lod);
        c.bench_function(format!("screen_error depth={lod}").as_str(), |b| {
            b.iter(|| {
                black_box(screen_error(
                    black_box(key),
                    black_box(camera),
                    black_box(radius),
                ))
            })
        });
    }
}

criterion_group!(benches, bench_generate_chunk_mesh, bench_screen_error);
criterion_main!(benches);
