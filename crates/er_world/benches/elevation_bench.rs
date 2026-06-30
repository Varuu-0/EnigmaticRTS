use criterion::{black_box, criterion_group, criterion_main, Criterion};
use er_core::seed::PlanetSeed;
use er_world::elevation::{elevation, elevation_params, ElevationNoise};
use glam::DVec3;

fn bench_single_eval(c: &mut Criterion) {
    let params = elevation_params(PlanetSeed(0xC0FFEE));
    let noise = ElevationNoise::new(&params);
    let dir = DVec3::new(0.5, 0.3, 0.8).normalize();

    c.bench_function("single_elevation", |b| {
        b.iter(|| {
            black_box(elevation(
                black_box(dir),
                black_box(&noise),
                black_box(&params),
            ))
        })
    });
}

fn bench_chunk_17x17(c: &mut Criterion) {
    let params = elevation_params(PlanetSeed(0xC0FFEE));
    let noise = ElevationNoise::new(&params);

    c.bench_function("chunk_17x17", |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..17 {
                for j in 0..17 {
                    let u = i as f64 / 16.0;
                    let v = j as f64 / 16.0;
                    let dir = DVec3::new(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.5).normalize();
                    sum += elevation(dir, &noise, &params);
                }
            }
            black_box(sum)
        })
    });
}

criterion_group!(benches, bench_single_eval, bench_chunk_17x17);
criterion_main!(benches);
