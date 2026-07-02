use criterion::{black_box, criterion_group, criterion_main, Criterion};
use er_core::config::PLANET_RADIUS_DEFAULT;
use er_core::math::CellKey;
use er_core::seed::PlanetSeed;
use er_terrain::generate_chunk_mesh;
use er_terrain::lod::screen_error;
use er_world::elevation::{elevation_params, ElevationNoise};
use er_world::params::{climate_noise, planet_params};
use glam::DVec3;

fn key_at(lod: u8) -> CellKey {
    CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod,
    }
}

fn bench_generate_chunk_mesh(c: &mut Criterion) {
    let radius = PLANET_RADIUS_DEFAULT;
    let seed = PlanetSeed(0xC0FFEE);
    let elev_params = elevation_params(seed);
    let noise = ElevationNoise::new(&elev_params);
    let pp = planet_params(seed);
    let cn = climate_noise(&pp);

    for &lod in &[0u8, 6, 12] {
        let key = key_at(lod);
        c.bench_function(format!("generate_chunk_mesh depth={lod}").as_str(), |b| {
            b.iter(|| {
                black_box(generate_chunk_mesh(
                    black_box(key),
                    black_box(radius),
                    black_box(&noise),
                    black_box(&elev_params),
                    black_box(&pp),
                    black_box(&cn),
                ))
            })
        });
    }
}

fn bench_screen_error(c: &mut Criterion) {
    let radius = PLANET_RADIUS_DEFAULT;
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
