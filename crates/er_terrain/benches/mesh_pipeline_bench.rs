//! Headless Earth-scale terrain mesh throughput matrix.
//!
//! Run with:
//! `cargo bench -p er_terrain --bench mesh_pipeline_bench`

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use er_core::config::EARTH_RADIUS_M;
use er_core::math::{cells_per_edge, CellKey};
use er_core::seed::PlanetSeed;
use er_terrain::{generate_chunk_mesh, ATTRIBUTE_CURVATURE, ATTRIBUTE_NORMAL};
use er_world::cache::WorldCache;
use er_world::elevation::elevation_params;
use er_world::params::planet_params;
use er_world::terrain_field::{ProceduralTerrainField, TerrainField};

const LODS: &[u8] = &[0, 8, 12, 14, 15, 16, 17];
const CHUNK_COUNTS: &[usize] = &[1, 16, 64];
const TRIALS: usize = 3;
const FRAME_BUDGET: Duration = Duration::from_nanos(16_666_667);
const ELEVATION_SCALE: f32 = 1.0;

fn main() {
    let seed = PlanetSeed(0xC0FFEE);
    let pp = planet_params(seed);
    let field = ProceduralTerrainField::new_metric(elevation_params(seed), pp, EARTH_RADIUS_M);
    let sea_level = pp.sea_level;
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(16);

    println!("Earth terrain mesh pipeline benchmark");
    println!(
        "radius={EARTH_RADIUS_M:.0}m vertices/chunk={} workers={workers} trials={TRIALS}",
        sample_vertex_count(&field, sea_level)
    );
    println!(
        "{:>4} {:>6} {:>9} {:>10} {:>10} {:>10} {:>10} {:>16}",
        "lod", "chunks", "mode", "total_ms", "ms/chunk", "chunks/16ms", "speedup", "checksum"
    );

    for &lod in LODS {
        for &count in CHUNK_COUNTS {
            let keys = keys_for_batch(lod, count);
            let sequential = best_of(TRIALS, || generate_sequential(&keys, &field, sea_level));
            print_result(lod, count, "seq", sequential, 1.0);

            if count > 1 && workers > 1 {
                let parallel = best_of(TRIALS, || {
                    generate_parallel(&keys, &field, workers, sea_level)
                });
                print_result(
                    lod,
                    count,
                    "parallel",
                    parallel,
                    sequential.elapsed.as_secs_f64() / parallel.elapsed.as_secs_f64(),
                );
                assert_eq!(sequential.checksum, parallel.checksum);
            }
        }
    }

    let cached_field = ProceduralTerrainField::with_cache_metric(
        elevation_params(seed),
        pp,
        Arc::new(WorldCache::with_lod(1_048_576, 22)),
        EARTH_RADIUS_M,
    );
    let keys = keys_for_batch(12, 64);
    let cold = generate_sequential(&keys, &cached_field, sea_level);
    let warm = best_of(TRIALS, || {
        generate_sequential(&keys, &cached_field, sea_level)
    });
    print_result(12, keys.len(), "cache-cold", cold, 1.0);
    print_result(
        12,
        keys.len(),
        "cache-warm",
        warm,
        cold.elapsed.as_secs_f64() / warm.elapsed.as_secs_f64(),
    );
    assert_eq!(cold.checksum, warm.checksum);
}

#[derive(Clone, Copy)]
struct Measurement {
    elapsed: Duration,
    checksum: u64,
}

fn best_of(trials: usize, mut measure: impl FnMut() -> Measurement) -> Measurement {
    black_box(measure());
    (0..trials)
        .map(|_| measure())
        .min_by_key(|result| result.elapsed)
        .unwrap()
}

fn generate_sequential(keys: &[CellKey], field: &dyn TerrainField, sea_level: f64) -> Measurement {
    let start = Instant::now();
    let checksum = keys.iter().fold(0u64, |checksum, &key| {
        checksum.wrapping_add(mesh_checksum(&generate_chunk_mesh(
            key,
            EARTH_RADIUS_M,
            ELEVATION_SCALE,
            sea_level,
            field,
        )))
    });
    Measurement {
        elapsed: start.elapsed(),
        checksum: black_box(checksum),
    }
}

fn generate_parallel(
    keys: &[CellKey],
    field: &dyn TerrainField,
    workers: usize,
    sea_level: f64,
) -> Measurement {
    let start = Instant::now();
    let chunk_size = keys.len().div_ceil(workers);
    let checksum = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                keys.chunks(chunk_size)
                    .map(|batch| {
                        scope.spawn(move || generate_sequential(batch, field, sea_level).checksum)
                    })
                    .collect::<Vec<_>>()
            })
            .join()
            .unwrap()
            .into_iter()
            .fold(0u64, |sum, task| sum.wrapping_add(task.join().unwrap()))
    });
    Measurement {
        elapsed: start.elapsed(),
        checksum: black_box(checksum),
    }
}

fn print_result(lod: u8, count: usize, mode: &str, result: Measurement, speedup: f64) {
    let total_ms = result.elapsed.as_secs_f64() * 1000.0;
    let per_chunk_ms = total_ms / count as f64;
    let chunks_per_frame = FRAME_BUDGET.as_secs_f64() * count as f64 / result.elapsed.as_secs_f64();
    println!(
        "{lod:>4} {count:>6} {mode:>9} {total_ms:>10.3} {per_chunk_ms:>10.3} {chunks_per_frame:>10.1} {speedup:>10.2} {checksum:>16x}",
        checksum = result.checksum,
    );
}

fn keys_for_batch(lod: u8, count: usize) -> Vec<CellKey> {
    let edge = u64::from(cells_per_edge(lod));
    let cells_per_face = edge * edge;
    (0..count as u64)
        .map(|index| {
            let offset = ((index / 6) * 7) % cells_per_face;
            CellKey {
                face: (index % 6) as u8,
                i: (offset % edge) as u32,
                j: (offset / edge) as u32,
                lod,
            }
        })
        .collect()
}

fn sample_vertex_count(field: &dyn TerrainField, sea_level: f64) -> usize {
    generate_chunk_mesh(
        CellKey {
            face: 0,
            i: 0,
            j: 0,
            lod: 0,
        },
        EARTH_RADIUS_M,
        ELEVATION_SCALE,
        sea_level,
        field,
    )
    .count_vertices()
}

fn mesh_checksum(mesh: &Mesh) -> u64 {
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut checksum = 0xcbf29ce484222325u64;
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    {
        for position in positions {
            for value in position {
                checksum = checksum
                    .wrapping_mul(FNV_PRIME)
                    .wrapping_add(u64::from(value.to_bits()));
            }
        }
    }
    if let Some(indices) = mesh.indices() {
        match indices {
            Indices::U16(indices) => {
                for &index in indices {
                    checksum = checksum
                        .wrapping_mul(FNV_PRIME)
                        .wrapping_add(u64::from(index));
                }
            }
            Indices::U32(indices) => {
                for &index in indices {
                    checksum = checksum
                        .wrapping_mul(FNV_PRIME)
                        .wrapping_add(u64::from(index));
                }
            }
        }
    }
    if let Some(VertexAttributeValues::Float32x3(normals)) = mesh.attribute(ATTRIBUTE_NORMAL) {
        for normal in normals {
            for value in normal {
                checksum = checksum
                    .wrapping_mul(FNV_PRIME)
                    .wrapping_add(u64::from(value.to_bits()));
            }
        }
    }
    if let Some(VertexAttributeValues::Float32(curvatures)) = mesh.attribute(ATTRIBUTE_CURVATURE) {
        for value in curvatures {
            checksum = checksum
                .wrapping_mul(FNV_PRIME)
                .wrapping_add(u64::from(value.to_bits()));
        }
    }
    checksum
}
