//! Maintained deterministic regression test for the chart tile mapping and
//! the RAM/disk surface cache.
//!
//! Gates exercised:
//! - Earth-scale: all 652×652×6 provider tiles map to the originating face
//!   and a valid chart index, and `key_for_tile` agrees with
//!   `key_for_direction(tile_center_dir)` for a dense sample.
//! - RAM LRU: bulk insert N >> capacity, assert eviction order, re-insert and
//!   confirm checksum parity after eviction+reload.
//! - Disk cache: bulk writes with capacity enforcement, reload round-trip
//!   with checksum parity, and that capacity stays bounded.
//!
//! Reports rates (tiles/s, records/s) and timings via stdout so the harness
//! output is self-describing.

use std::sync::Arc;
use std::time::Instant;

use er_world::surface_cache::{
    ChartMacroField, CreationMetadata, SurfaceCache, SurfaceCacheKey, SurfaceDiskCache,
    SurfaceTileRecord,
};
use er_world::surface_charts::{
    ChartOwnership, SurfaceChartMetadata, SURFACE_CHART_PROJECTION_REVISION,
};
use er_world::terrain_field::MacroTerrainField;

const R: f64 = 6_371_000.0;
const EARTH_TILES_PER_FACE_EDGE: u32 = 652;
const EARTH_CHART_LEVEL: u8 = 10;
const EARTH_CHARTS_PER_EDGE: u32 = 1 << EARTH_CHART_LEVEL; // 1024
const CORE_RES: u32 = 4;
const HALO: u32 = 1;
const STORED: u32 = CORE_RES + HALO * 2; // 6
const STORED_N: usize = (STORED as usize) * (STORED as usize); // 36

fn earth_meta() -> SurfaceChartMetadata {
    SurfaceChartMetadata {
        seed: 0xC0FFEE,
        projection_revision: SURFACE_CHART_PROJECTION_REVISION,
        model_revision: "stress-earth-v1".to_owned(),
        conditioning_revision: 1,
        residual_revision: 1,
        sea_level_datum_m: 0,
        pixel_scale_m: 30,
        halo_samples: HALO,
        core_resolution: CORE_RES,
        ownership: ChartOwnership::LearnedReliefProceduralShoreline,
        planet_radius_m: R as u64,
        charts_per_face_edge: 652,
    }
}

fn earth_field() -> Arc<ChartMacroField> {
    earth_field_with_cap(256)
}

fn earth_field_with_cap(ram_capacity: usize) -> Arc<ChartMacroField> {
    Arc::new(ChartMacroField::new(
        ram_capacity,
        None,
        earth_meta(),
        EARTH_CHART_LEVEL,
        1000.0,
    ))
}

fn payload_for(elevation: i16) -> (Vec<i16>, Vec<f32>) {
    let elev: Vec<i16> = vec![elevation; STORED_N];
    let climate: Vec<f32> = vec![0.0; STORED_N * 4];
    (elev, climate)
}

fn record_for(key: SurfaceCacheKey, elevation: i16) -> SurfaceTileRecord {
    let (elev, climate) = payload_for(elevation);
    SurfaceTileRecord::from_payload(
        key,
        Arc::from(elev),
        Arc::from(climate),
        CreationMetadata::now("stress"),
    )
}

// ---------------------------------------------------------------------------
// 1. Earth-scale chart mapping: full 652x652x6 tile grid.
// ---------------------------------------------------------------------------

#[test]
fn earth_scale_full_tile_grid_maps_to_correct_face_and_chart() {
    let field = earth_field();
    let start = Instant::now();
    let total_tiles = (EARTH_TILES_PER_FACE_EDGE as u64) * (EARTH_TILES_PER_FACE_EDGE as u64) * 6;

    let mut wrong_face = 0u64;
    let mut chart_x_overflow = 0u64;
    let mut chart_y_overflow = 0u64;
    let mut key_mismatch = 0u64;

    for face in 0..6u8 {
        for x in 0..EARTH_TILES_PER_FACE_EDGE {
            for y in 0..EARTH_TILES_PER_FACE_EDGE {
                let key = field.key_for_tile(face, x, y, EARTH_TILES_PER_FACE_EDGE);
                if key.face != face {
                    wrong_face += 1;
                }
                if key.x >= EARTH_CHARTS_PER_EDGE {
                    chart_x_overflow += 1;
                }
                if key.y >= EARTH_CHARTS_PER_EDGE {
                    chart_y_overflow += 1;
                }
                // Dense consistency check on a 1-in-7 sample to keep the test
                // fast while still catching any systematic drift.
                if (x + y) % 7 == 0 {
                    let dir = field.tile_center_dir(face, x, y, EARTH_TILES_PER_FACE_EDGE);
                    let key_dir = field.key_for_direction(dir);
                    if key != key_dir {
                        key_mismatch += 1;
                    }
                }
            }
        }
    }
    let elapsed = start.elapsed();
    let tiles_per_sec = total_tiles as f64 / elapsed.as_secs_f64().max(1e-9);

    println!(
        "EARTH_MAPPING: total_tiles={total_tiles} elapsed_ms={} tiles/s={tiles_per_sec:.0} wrong_face={wrong_face} chart_x_overflow={chart_x_overflow} chart_y_overflow={chart_y_overflow} key_mismatch={key_mismatch}",
        elapsed.as_millis()
    );

    assert_eq!(wrong_face, 0, "some tiles mapped to the wrong face");
    assert_eq!(chart_x_overflow, 0, "chart x overflowed chart grid");
    assert_eq!(chart_y_overflow, 0, "chart y overflowed chart grid");
    assert_eq!(key_mismatch, 0, "key_for_tile != key_for_direction");
}

#[test]
fn earth_scale_boundary_and_corner_tiles_are_origin_face() {
    let field = earth_field();
    let mut bad = 0u64;
    let last = EARTH_TILES_PER_FACE_EDGE - 1;
    for face in 0..6u8 {
        for &(x, y) in &[
            (0u32, 0u32),
            (0, last),
            (last, 0),
            (last, last),
            (0, last / 2),
            (last, last / 2),
            (last / 2, 0),
            (last / 2, last),
        ] {
            let key = field.key_for_tile(face, x, y, EARTH_TILES_PER_FACE_EDGE);
            if key.face != face {
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "boundary/corner tile mapped to wrong face");
}

// ---------------------------------------------------------------------------
// 2. RAM LRU bulk insert/evict/reload with checksum parity.
// ---------------------------------------------------------------------------

#[test]
fn ram_lru_bulk_insert_evict_reload_checksum_parity() {
    let ram_capacity = 512usize;
    let total_records = ram_capacity * 4; // 2048 — well past capacity
    let field = earth_field_with_cap(ram_capacity);
    // Drive through the public ChartMacroField API to exercise the real
    // store path (which also bumps the revision counter).
    let start = Instant::now();
    let mut keys: Vec<SurfaceCacheKey> = Vec::with_capacity(total_records);
    for i in 0..total_records as u32 {
        // Spread across faces and tile indices to generate distinct keys.
        let face = (i % 6) as u8;
        let x = i % EARTH_TILES_PER_FACE_EDGE;
        let y = (i / 6) % EARTH_TILES_PER_FACE_EDGE;
        let key = field.key_for_tile(face, x, y, EARTH_TILES_PER_FACE_EDGE);
        let record = record_for(key.clone(), (i % 32000) as i16);
        field.cache().store(record).unwrap();
        field.bump_revision();
        keys.push(key);
    }
    let insert_elapsed = start.elapsed();
    let insert_rate = total_records as f64 / insert_elapsed.as_secs_f64().max(1e-9);
    let resident = field.cache().ram.len();
    let revision = field.revision();

    // Resident count must be bounded by capacity.
    assert!(
        resident <= ram_capacity,
        "RAM cache exceeded capacity: {resident} > {ram_capacity}"
    );

    // Evicted keys (the oldest 3/4) must be misses; the most recent
    // `ram_capacity` must be resident.
    let evicted_count = total_records - ram_capacity;
    let mut miss_among_evicted = 0u64;
    for key in keys.iter().take(evicted_count) {
        if field.cache().get_resident(key).is_none() {
            miss_among_evicted += 1;
        }
    }
    let mut hit_among_recent = 0u64;
    for key in keys.iter().skip(evicted_count) {
        if field.cache().get_resident(key).is_some() {
            hit_among_recent += 1;
        }
    }

    // Reload the first evicted key and confirm checksum parity.
    let reload_key = keys[0].clone();
    let original_checksum = {
        let (elev, climate) = payload_for(0);
        let r = SurfaceTileRecord::from_payload(
            reload_key.clone(),
            Arc::from(elev),
            Arc::from(climate),
            CreationMetadata::now("stress"),
        );
        r.payload_checksum
    };
    let reload_start = Instant::now();
    let reloaded = record_for(reload_key.clone(), 0);
    field.cache().store(reloaded).unwrap();
    field.bump_revision();
    let reload_elapsed = reload_start.elapsed();
    let resident_checksum = field
        .cache()
        .get_resident(&reload_key)
        .expect("reloaded key resident")
        .payload_checksum;

    println!(
        "RAM_LRU: total={total_records} capacity={ram_capacity} resident_after_insert={resident} insert_ms={} insert_rate={insert_rate:.0}/s evicted_misses={miss_among_evicted}/{evicted_count} recent_hits={hit_among_recent}/{ram_capacity} reload_ms={} checksum_match={} revision={revision}",
        insert_elapsed.as_millis(),
        reload_elapsed.as_millis(),
        resident_checksum == original_checksum
    );

    assert_eq!(miss_among_evicted as usize, evicted_count);
    assert_eq!(hit_among_recent as usize, ram_capacity);
    assert_eq!(resident_checksum, original_checksum);
}

// ---------------------------------------------------------------------------
// 3. Disk cache bulk insert/evict/reload with checksum parity + capacity.
// ---------------------------------------------------------------------------

#[test]
fn disk_cache_bulk_insert_evict_reload_checksum_parity() {
    let disk_capacity = 64usize;
    let total_records = disk_capacity * 4; // 256
    let dir = std::env::temp_dir().join(format!(
        "ersurf_m4_stress_disk_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let disk = SurfaceDiskCache::new(&dir, disk_capacity).unwrap();
    let cache = SurfaceCache::new(disk_capacity * 2, Some(disk));

    let start = Instant::now();
    let mut keys: Vec<(SurfaceCacheKey, [u8; 32])> = Vec::with_capacity(total_records);
    for i in 0..total_records as u32 {
        let face = (i % 6) as u8;
        let x = i % EARTH_TILES_PER_FACE_EDGE;
        let y = (i / 6) % EARTH_TILES_PER_FACE_EDGE;
        let field = earth_field();
        let key = field.key_for_tile(face, x, y, EARTH_TILES_PER_FACE_EDGE);
        let record = record_for(key.clone(), (i % 32000) as i16);
        let checksum = record.payload_checksum;
        cache.store(record).unwrap();
        keys.push((key, checksum));
    }
    let insert_elapsed = start.elapsed();
    let insert_rate = total_records as f64 / insert_elapsed.as_secs_f64().max(1e-9);

    // Disk count must respect capacity (after each store enforces it).
    let disk_count = cache.disk.as_ref().unwrap().count().unwrap();
    assert!(
        disk_count <= disk_capacity,
        "disk cache exceeded capacity: {disk_count} > {disk_capacity}"
    );

    // Reload the last `disk_capacity` records (the survivors) and confirm
    // checksum parity. Clear RAM first so load_from_disk is the real path.
    cache.ram.clear();
    let survivors: Vec<(SurfaceCacheKey, [u8; 32])> =
        keys.iter().rev().take(disk_capacity).cloned().collect();
    let mut parity_ok = 0u64;
    let mut parity_bad = 0u64;
    let reload_start = Instant::now();
    for (key, expected) in &survivors {
        let promoted = cache.load_from_disk(key).unwrap();
        if promoted {
            let resident = cache.get_resident(key).expect("resident after load");
            if resident.payload_checksum == *expected {
                parity_ok += 1;
            } else {
                parity_bad += 1;
            }
        } else {
            parity_bad += 1;
        }
    }
    let reload_elapsed = reload_start.elapsed();
    let reload_rate = survivors.len() as f64 / reload_elapsed.as_secs_f64().max(1e-9);

    println!(
        "DISK_CACHE: total={total_records} capacity={disk_capacity} disk_after_insert={disk_count} insert_ms={} insert_rate={insert_rate:.0}/s survivors_reloaded={} reload_ms={} reload_rate={reload_rate:.0}/s checksum_parity_ok={parity_ok} checksum_parity_bad={parity_bad}",
        insert_elapsed.as_millis(),
        reload_elapsed.as_millis(),
        survivors.len()
    );

    assert!(disk_count <= disk_capacity);
    assert_eq!(parity_bad, 0, "some reloaded records had checksum drift");
    assert_eq!(parity_ok as usize, survivors.len());

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// 4. Concurrent disk writes (capacity + concurrency safety under load).
// ---------------------------------------------------------------------------

#[test]
fn disk_cache_concurrent_bulk_writes_respect_capacity() {
    let disk_capacity = 16usize;
    let writers = 8usize;
    let per_writer = 32usize;
    let dir = std::env::temp_dir().join(format!(
        "ersurf_m4_stress_conc_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let disk = Arc::new(SurfaceDiskCache::new(&dir, disk_capacity).unwrap());

    let start = Instant::now();
    let mut handles = Vec::new();
    for w in 0..writers {
        let disk = Arc::clone(&disk);
        let field = earth_field();
        handles.push(std::thread::spawn(move || {
            for i in 0..per_writer {
                let idx = (w * per_writer + i) as u32;
                let face = (idx % 6) as u8;
                let x = idx % EARTH_TILES_PER_FACE_EDGE;
                let y = (idx / 6) % EARTH_TILES_PER_FACE_EDGE;
                let key = field.key_for_tile(face, x, y, EARTH_TILES_PER_FACE_EDGE);
                let record = record_for(key, (idx % 32000) as i16);
                disk.write(&record).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let elapsed = start.elapsed();
    let total = (writers * per_writer) as u64;
    let rate = total as f64 / elapsed.as_secs_f64().max(1e-9);
    let disk_count = disk.count().unwrap();

    println!(
        "DISK_CONCURRENT: writers={writers} per_writer={per_writer} total={total} capacity={disk_capacity} disk_after={disk_count} elapsed_ms={} writes/s={rate:.0}",
        elapsed.as_millis()
    );

    assert!(
        disk_count <= disk_capacity,
        "concurrent writes exceeded capacity"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
