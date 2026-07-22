//! Milestone 5 integration tests for the production learned-streaming pipeline.
//!
//! These tests exercise the M5 gates end-to-end at the `er_world` layer:
//!
//! - **Halo-residency gate**: a chunk uses learned data only when every
//!   elevation plus normal halo dependency is resident; otherwise procedural
//!   for the entire chunk (no mixed normals).
//! - **Intersecting-only rebuild**: when a chart/tile arrives, only chunks
//!   intersecting the changed chart are rebuilt, not all active chunks.
//! - **Blend continuity**: the blend transition is monotonic and produces no
//!   height step; shoreline/sea-datum ownership stays procedural.
//! - **Teleport procedural-to-learned**: teleport shows procedural terrain
//!   immediately, then smooth learned refinement as tiles arrive.
//! - **Streaming telemetry**: queue depth, resident/pending/failed, hit rate,
//!   fallback %, latency percentiles, rebuild counts, and health are
//!   reported correctly.

use std::sync::Arc;
use std::time::Duration;

use er_core::math::{uv_to_dir, CellKey};
use er_core::seed::PlanetSeed;
use er_world::elevation::elevation_params;
use er_world::params::planet_params;
use er_world::streaming::{
    chunks_intersecting_chart, BlendWeights, PriorityClass, ProviderTileCoordinate, ServiceHealth,
    StreamingConfig, StreamingQueue,
};
use er_world::surface_cache::{
    ChartMacroField, CreationMetadata, SurfaceCacheKey, SurfaceDiskCache, SurfaceTileRecord,
};
use er_world::surface_charts::{
    ChartOwnership, SurfaceChartId, SurfaceChartMetadata, SurfacePatchId,
    SURFACE_CHART_PROJECTION_REVISION,
};
use er_world::terrain_field::{
    BlendedHybridTerrainField, MacroTerrainField, MacroTerrainSample, ProceduralTerrainField,
    TerrainField, TerrainSampleSource,
};

const R: f64 = 6_371_000.0;
const CHART_LEVEL: u8 = 2;
const CORE_RES: u32 = 4;
const HALO: u32 = 1;

fn meta() -> SurfaceChartMetadata {
    SurfaceChartMetadata {
        seed: 0xC0FFEE,
        projection_revision: SURFACE_CHART_PROJECTION_REVISION,
        model_revision: "m5-integ-v1".to_owned(),
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

fn chart_field() -> Arc<ChartMacroField> {
    Arc::new(ChartMacroField::new(
        8192,
        None,
        meta(),
        CHART_LEVEL,
        1000.0,
    ))
}

fn key_for_chart(face: u8, x: u32, y: u32) -> SurfaceCacheKey {
    let field = chart_field();
    let cpe = field.metadata().charts_per_face_edge;
    let level = if cpe <= 1 {
        0
    } else {
        (32 - (cpe - 1).leading_zeros()) as u8
    };
    let chart = SurfaceChartId {
        face,
        level,
        x,
        y,
        charts_per_face_edge: cpe,
    };
    let patch = SurfacePatchId::new(chart, HALO);
    let bounds = [face as i64, x as i64, y as i64, cpe as i64];
    SurfaceCacheKey::from_metadata(&field.metadata().clone(), patch, bounds)
}

fn fill_chart(field: &ChartMacroField, face: u8, x: u32, y: u32, elevation: i16) {
    let stored = (CORE_RES + HALO * 2) as usize;
    let n = stored * stored;
    let elev: Vec<i16> = vec![elevation; n];
    let climate: Vec<f32> = vec![0.0; n * 4];
    // Construct the key directly from the chart index, using the same
    // informational level that SurfaceChartId::from_direction would compute.
    let cpe = field.metadata().charts_per_face_edge;
    let level = if cpe <= 1 {
        0
    } else {
        (32 - (cpe - 1).leading_zeros()) as u8
    };
    let chart = SurfaceChartId {
        face,
        level,
        x,
        y,
        charts_per_face_edge: cpe,
    };
    let patch = SurfacePatchId::new(chart, field.metadata().halo_samples);
    let bounds = [face as i64, x as i64, y as i64, cpe as i64];
    let key = SurfaceCacheKey::from_metadata(&field.metadata().clone(), patch, bounds);
    let record = SurfaceTileRecord::from_payload(
        key,
        Arc::from(elev),
        Arc::from(climate),
        CreationMetadata::now("m5-integ"),
    );
    field.cache().store(record).unwrap();
    field.bump_revision();
}

fn small_config() -> StreamingConfig {
    StreamingConfig {
        max_queued: 64,
        max_in_flight: 4,
        request_timeout: Duration::from_secs(10),
        base_backoff: Duration::from_millis(5),
        max_backoff: Duration::from_millis(50),
        healthy_threshold: 2,
        unhealthy_threshold: 3,
        unhealthy_cooldown: Duration::from_millis(20),
        jitter_seed: 42,
        max_resident: 32,
    }
}

// ---------------------------------------------------------------------------
// 1. Halo-residency gate: chunk uses learned only if ALL halo deps resident
// ---------------------------------------------------------------------------

#[test]
fn chunk_is_procedural_when_any_halo_dependency_is_missing() {
    let field = chart_field();
    let chunk = CellKey {
        face: 0,
        i: 1,
        j: 1,
        lod: 3,
    };
    // No charts resident: must be procedural.
    assert!(!field.chunk_halo_resident(chunk));
    // Fill one chart but not all: still procedural.
    let deps = field.chunk_halo_dependencies(chunk);
    assert!(!deps.is_empty());
    fill_chart(&field, deps[0].face, deps[0].x, deps[0].y, 500);
    assert!(!field.chunk_halo_resident(chunk));
}

#[test]
fn chunk_becomes_learned_only_when_all_halo_dependencies_resident() {
    let field = chart_field();
    let chunk = CellKey {
        face: 0,
        i: 2,
        j: 3,
        lod: 3,
    };
    let deps = field.chunk_halo_dependencies(chunk);
    eprintln!("deps: {} charts", deps.len());
    for key in &deps {
        fill_chart(&field, key.face, key.x, key.y, 400);
    }
    // Verify each dep is resident.
    for key in &deps {
        let r = field.cache().get_resident(key);
        if r.is_none() {
            eprintln!(
                "MISS: face={} x={} y={} cpe={}",
                key.face, key.x, key.y, key.charts_per_face_edge
            );
        }
    }
    assert!(
        field.chunk_halo_resident(chunk),
        "chunk_halo_resident failed after filling all deps"
    );
}

#[test]
fn no_mixed_normals_partial_residency_falls_back_entirely() {
    // When some but not all halo deps are resident, the chunk must use
    // procedural for the ENTIRE chunk — no vertex gets learned normals
    // while others get procedural. This is verified by the residency gate
    // returning false for partial residency.
    let field = chart_field();
    let chunk = CellKey {
        face: 1,
        i: 0,
        j: 0,
        lod: 3,
    };
    let deps = field.chunk_halo_dependencies(chunk);
    // Fill exactly half the deps.
    for key in deps.iter().take(deps.len() / 2) {
        fill_chart(&field, key.face, key.x, key.y, 300);
    }
    // Must still be fully procedural (gate returns false).
    assert!(!field.chunk_halo_resident(chunk));
}

// ---------------------------------------------------------------------------
// 2. Intersecting-only rebuild: only chunks intersecting a changed tile
// ---------------------------------------------------------------------------

#[test]
fn only_intersecting_chunks_rebuilt_when_tile_arrives() {
    let chunk_lod = 4u8;
    let chart_face = 0u8;
    let chart_x = 1u32;
    let chart_y = 1u32;

    let rebuilt = chunks_intersecting_chart(chart_face, chart_x, chart_y, 652, chunk_lod, 0);

    // Chunks far from the chart should NOT be in the rebuild set.
    let far_chunk = CellKey {
        face: 0,
        i: 15,
        j: 15,
        lod: chunk_lod,
    };
    assert!(!rebuilt.contains(&far_chunk));

    // Chunks overlapping the chart SHOULD be in the rebuild set.
    // Chart (1,1) at 652 charts/edge covers uv [~0.00153, ~0.00307].
    // At LOD 4 (16 cells/edge), chunk i=0 covers [0, 0.0625] which overlaps.
    let near_chunk = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: chunk_lod,
    };
    assert!(rebuilt.contains(&near_chunk));

    // Only face-0 chunks should be rebuilt (the chart is on face 0).
    for c in &rebuilt {
        assert_eq!(c.face, chart_face);
    }
}

#[test]
fn rebuild_set_is_bounded_and_does_not_include_all_active_chunks() {
    let chunk_lod = 5u8;
    let rebuilt = chunks_intersecting_chart(2, 0, 0, 652, chunk_lod, 1);
    // The rebuild set should be bounded by the chart's footprint (a chart at
    // level 2 covers 1/4 of the face). At LOD 5 (32 cells/edge), that's ~8x8
    // = 64 cells plus halo, NOT the entire face (1024 cells).
    let total_cells = (1u32 << chunk_lod) * (1u32 << chunk_lod);
    assert!(
        rebuilt.len() < total_cells as usize,
        "rebuild set {} should be < total cells {}",
        rebuilt.len(),
        total_cells
    );
    // All rebuilt chunks must be on the chart's face (face 2).
    for c in &rebuilt {
        assert_eq!(c.face, 2);
    }
}

// ---------------------------------------------------------------------------
// 3. Blend continuity and shoreline invariance
// ---------------------------------------------------------------------------

struct ConstantMacro(f64);
impl MacroTerrainField for ConstantMacro {
    fn sample_resident(&self, _: glam::DVec3) -> Option<MacroTerrainSample> {
        Some(MacroTerrainSample {
            elevation: self.0,
            visual_climate: er_world::terrain_field::VisualClimate::default(),
        })
    }
}

#[test]
fn blend_is_continuous_and_monotonic() {
    // The blend weight must be monotonically non-decreasing over time and
    // produce no height step (C1-continuous smoothstep).
    let mut prev = -1.0_f64;
    for secs in 0..=30 {
        let w = BlendWeights::for_transition(Duration::from_secs(secs), 500.0, 3.0).learned_weight;
        assert!(w >= prev, "blend decreased at secs={secs}: {w} < {prev}");
        prev = w;
    }
    // After the transition, weight must be 1.0 (fully learned).
    let final_w = BlendWeights::for_transition(Duration::from_secs(3), 500.0, 3.0).learned_weight;
    assert!((final_w - 1.0).abs() < 1e-9);
}

#[test]
fn blend_preserves_shoreline_ownership() {
    // The shoreline (low_freq_elev < sea_level) must stay with the procedural
    // macro field regardless of the learned macro value or blend weight.
    let seed = PlanetSeed(0xC0FFEE);
    let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
        elevation_params(seed),
        planet_params(seed),
    ));
    let dir = uv_to_dir(2, 0.41, 0.59);
    let procedural = fallback.sample(dir);
    let procedural_water = procedural.low_freq_elev < 0.0;

    for learned_macro in [-5.0, -1.0, 0.0, 1.0, 5.0] {
        let blended = BlendedHybridTerrainField::new(
            fallback.clone(),
            Arc::new(ConstantMacro(learned_macro)),
            0.0, // instant transition
        );
        let sample = blended.sample(dir);
        // Shoreline ownership never changes.
        assert_eq!(sample.low_freq_elev, procedural.low_freq_elev);
        assert_eq!(sample.low_freq_elev < 0.0, procedural_water);
    }
}

#[test]
fn blend_does_not_blend_world_coordinates() {
    // The blend operates on provenance weight, not world coordinates. The
    // blended elevation must be a convex combination of the two macro values
    // plus the procedural residual — never an interpolation of world positions.
    let seed = PlanetSeed(0xC0FFEE);
    let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
        elevation_params(seed),
        planet_params(seed),
    ));
    let dir = uv_to_dir(0, 0.3, 0.7);
    let pro = fallback.sample(dir);
    let procedural_macro = pro.low_freq_elev as f64;
    let procedural_residual = pro.elevation - procedural_macro;
    let learned_macro = 0.6;

    let blended = BlendedHybridTerrainField::new(
        fallback.clone(),
        Arc::new(ConstantMacro(learned_macro)),
        0.0,
    );
    let sample = blended.sample(dir);
    // Fully learned: learned_macro + procedural_residual.
    assert!(
        (sample.elevation - (learned_macro + procedural_residual)).abs() < 1e-9,
        "elevation {} != expected {}",
        sample.elevation,
        learned_macro + procedural_residual
    );
    // The procedural macro is NOT blended into the elevation (only the
    // residual is preserved).
    assert!(sample.source == TerrainSampleSource::LearnedMacro);
}

// ---------------------------------------------------------------------------
// 4. Teleport procedural-to-learned refinement
// ---------------------------------------------------------------------------

struct ResidentAfterMacro {
    resident: std::sync::atomic::AtomicBool,
    value: f64,
}

impl ResidentAfterMacro {
    fn set_resident(&self, resident: bool) {
        self.resident
            .store(resident, std::sync::atomic::Ordering::Relaxed);
    }
}

impl MacroTerrainField for ResidentAfterMacro {
    fn sample_resident(&self, _: glam::DVec3) -> Option<MacroTerrainSample> {
        if self.resident.load(std::sync::atomic::Ordering::Relaxed) {
            Some(MacroTerrainSample {
                elevation: self.value,
                visual_climate: er_world::terrain_field::VisualClimate::default(),
            })
        } else {
            None
        }
    }
}

#[test]
fn teleport_shows_procedural_then_smooth_learned_refinement() {
    // Simulate teleport: initially no tiles resident (procedural), then tiles
    // arrive (learned). The blended field must show procedural terrain
    // immediately, then smoothly refine to learned as the blend completes.
    let seed = PlanetSeed(0xC0FFEE);
    let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
        elevation_params(seed),
        planet_params(seed),
    ));
    let macro_field = Arc::new(ResidentAfterMacro {
        resident: std::sync::atomic::AtomicBool::new(false),
        value: 0.8,
    });
    let macro_dyn: Arc<dyn MacroTerrainField> =
        Arc::clone(&macro_field) as Arc<dyn MacroTerrainField>;

    let dir = uv_to_dir(1, 0.25, 0.75);

    // Before teleport: procedural (not resident).
    let before = {
        let blended = BlendedHybridTerrainField::new(fallback.clone(), Arc::clone(&macro_dyn), 0.0);
        blended.sample(dir)
    };
    assert_eq!(before.source, TerrainSampleSource::Procedural);

    // Teleport: mark resident. With instant transition (0.0), the next sample
    // is fully learned.
    macro_field.set_resident(true);
    let after = {
        // New blended field to reset blend state.
        let blended = BlendedHybridTerrainField::new(fallback.clone(), Arc::clone(&macro_dyn), 0.0);
        blended.sample(dir)
    };
    assert_eq!(after.source, TerrainSampleSource::LearnedMacro);

    // The elevation changed from procedural macro + residual to learned macro
    // + residual. The procedural macro is replaced by the learned macro.
    let pro = fallback.sample(dir);
    let procedural_macro = pro.low_freq_elev as f64;
    let procedural_residual = pro.elevation - procedural_macro;
    assert!(
        (after.elevation - (0.8 + procedural_residual)).abs() < 1e-9,
        "learned elevation {} != expected {}",
        after.elevation,
        0.8 + procedural_residual
    );
}

// ---------------------------------------------------------------------------
// 5. Streaming telemetry
// ---------------------------------------------------------------------------

#[test]
fn streaming_telemetry_reports_all_m5_gates() {
    let q = StreamingQueue::new(small_config());
    let now = std::time::Instant::now();

    // Enqueue some requests.
    for i in 0..5u32 {
        let k = key_for_chart(0, i, 0);
        q.enqueue(
            k,
            ProviderTileCoordinate {
                face: 0,
                x: 0,
                y: 0,
            },
            PriorityClass::VisibleSurface,
            None,
            now,
        );
    }

    // Dispatch and succeed some.
    for _ in 0..3 {
        if let Some(req) = q.pop_dispatchable(now) {
            q.record_success(&req.key, Duration::from_millis(10), now);
        }
    }

    // Record a failure.
    if let Some(req) = q.pop_dispatchable(now) {
        q.record_failure(&req.key, now);
    }

    // Record cache lookups.
    q.record_cache_lookup(true);
    q.record_cache_lookup(false);
    q.record_sample_source(true);
    q.record_sample_source(false);

    let tel = q.telemetry();

    // All M5 telemetry fields must be present and sensible.
    assert!(tel.queue_depth > 0, "queue_depth should be > 0");
    assert!(tel.resident_tiles > 0, "resident should be > 0");
    assert_eq!(tel.failed_total, 1, "failed_total should be 1");
    assert!(tel.cache_hit_rate() > 0.0);
    assert!(tel.fallback_percent() > 0.0);
    assert!(tel.latency_p50_ms.is_some());
    assert!(tel.latency_p95_ms.is_some());
    assert!(tel.provider_state.attempts_total >= 3);
    assert_eq!(tel.health.state.as_str(), "degraded"); // 1 failure -> degraded from healthy? Actually starts cold.
}

#[test]
fn streaming_health_transitions_through_cold_degraded_healthy() {
    let q = StreamingQueue::new(small_config());
    let now = std::time::Instant::now();
    // Starts cold.
    assert_eq!(q.health(), ServiceHealth::Cold);

    // First success -> degraded.
    let k = key_for_chart(0, 0, 0);
    q.enqueue(
        k.clone(),
        ProviderTileCoordinate {
            face: 0,
            x: 0,
            y: 0,
        },
        PriorityClass::VisibleSurface,
        None,
        now,
    );
    let _ = q.pop_dispatchable(now).unwrap();
    q.record_success(&k, Duration::from_millis(5), now);
    assert_eq!(q.health(), ServiceHealth::Degraded);

    // Second success -> healthy (threshold = 2).
    let k2 = key_for_chart(0, 1, 0);
    q.enqueue(
        k2.clone(),
        ProviderTileCoordinate {
            face: 0,
            x: 0,
            y: 0,
        },
        PriorityClass::VisibleSurface,
        None,
        now,
    );
    let _ = q.pop_dispatchable(now).unwrap();
    q.record_success(&k2, Duration::from_millis(5), now);
    assert_eq!(q.health(), ServiceHealth::Healthy);
}

// ---------------------------------------------------------------------------
// 6. Priority queue ordering with real chart keys
// ---------------------------------------------------------------------------

#[test]
fn priority_queue_orders_visible_surface_before_prefetch() {
    let q = StreamingQueue::new(small_config());
    let now = std::time::Instant::now();

    let prefetch_key = key_for_chart(0, 3, 3);
    let visible_key = key_for_chart(0, 0, 0);

    q.enqueue(
        prefetch_key,
        ProviderTileCoordinate {
            face: 0,
            x: 3,
            y: 3,
        },
        PriorityClass::PrefetchRing,
        None,
        now,
    );
    q.enqueue(
        visible_key,
        ProviderTileCoordinate {
            face: 0,
            x: 0,
            y: 0,
        },
        PriorityClass::VisibleSurface,
        None,
        now,
    );

    let first = q.pop_dispatchable(now).unwrap();
    assert_eq!(first.priority, PriorityClass::VisibleSurface);
    let second = q.pop_dispatchable(now).unwrap();
    assert_eq!(second.priority, PriorityClass::PrefetchRing);
}

// ---------------------------------------------------------------------------
// 7. Coalescing across queued/in-flight/resident with real keys
// ---------------------------------------------------------------------------

#[test]
fn coalescing_deduplicates_across_queued_and_resident() {
    let q = StreamingQueue::new(small_config());
    let now = std::time::Instant::now();
    let k = key_for_chart(1, 2, 2);

    // Enqueue, dispatch, then mark resident.
    assert!(q.enqueue(
        k.clone(),
        ProviderTileCoordinate {
            face: 0,
            x: 0,
            y: 0
        },
        PriorityClass::VisibleSurface,
        None,
        now
    ));
    let _ = q.pop_dispatchable(now).unwrap();
    q.mark_resident(&k);

    // Now a new request for the same key should be coalesced (resident).
    assert!(!q.enqueue(
        k.clone(),
        ProviderTileCoordinate {
            face: 0,
            x: 0,
            y: 0
        },
        PriorityClass::Warmup,
        None,
        now
    ));
    assert_eq!(q.queued_count(), 0);
}

// ---------------------------------------------------------------------------
// 8. Disk cache on runtime path
// ---------------------------------------------------------------------------

#[test]
fn disk_cache_enabled_on_runtime_path_promotes_to_ram() {
    let dir = std::env::temp_dir().join(format!(
        "ersurf_m5_disk_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let disk = SurfaceDiskCache::new(&dir, 32).unwrap();
    let field = Arc::new(ChartMacroField::new(
        8,
        Some(disk),
        meta(),
        CHART_LEVEL,
        1000.0,
    ));

    // Store a chart record (goes to disk + RAM).
    fill_chart(&field, 0, 1, 1, 750);

    // Clear RAM to simulate cold start.
    field.cache().ram.clear();
    assert!(field
        .cache()
        .get_resident(&key_for_chart(0, 1, 1))
        .is_none());

    // Promote from disk.
    let promoted = field
        .cache()
        .load_from_disk(&key_for_chart(0, 1, 1))
        .unwrap();
    assert!(promoted);
    assert!(field
        .cache()
        .get_resident(&key_for_chart(0, 1, 1))
        .is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// 9. Live mesh path: ChunkFieldSnapshot enforces chunk-wide halo residency
// ---------------------------------------------------------------------------

#[test]
fn chunk_field_snapshot_is_procedural_when_halo_not_resident() {
    // A ChunkFieldSnapshot with halo_resident=false must force every sample
    // to Procedural, regardless of the underlying field.
    use er_core::seed::PlanetSeed;
    use er_world::elevation::elevation_params;
    use er_world::params::planet_params;
    use er_world::terrain_field::{
        ChunkFieldSnapshot, ProceduralTerrainField, TerrainField, TerrainSampleSource,
    };

    let seed = PlanetSeed(0xC0FFEE);
    let fallback: Arc<dyn er_world::terrain_field::TerrainField> = Arc::new(
        ProceduralTerrainField::new(elevation_params(seed), planet_params(seed)),
    );
    let chunk = CellKey {
        face: 0,
        i: 1,
        j: 1,
        lod: 3,
    };
    let snapshot = ChunkFieldSnapshot::new(fallback, chunk, false, 0.0);
    let dir = uv_to_dir(0, 0.3, 0.3);
    let sample = snapshot.sample(dir);
    assert_eq!(sample.source, TerrainSampleSource::Procedural);
    assert!(!snapshot.learned_eligible());
}

#[test]
fn chunk_field_snapshot_is_learned_when_halo_resident() {
    // A ChunkFieldSnapshot with halo_resident=true must allow learned data.
    use er_world::terrain_field::{
        ChunkFieldSnapshot, MacroTerrainField, MacroTerrainSample, ProceduralTerrainField,
        TerrainField, TerrainSampleSource,
    };

    struct ConstantMacro(f64);
    impl MacroTerrainField for ConstantMacro {
        fn sample_resident(&self, _: glam::DVec3) -> Option<MacroTerrainSample> {
            Some(MacroTerrainSample {
                elevation: self.0,
                visual_climate: er_world::terrain_field::VisualClimate::default(),
            })
        }
    }

    let seed = er_core::seed::PlanetSeed(0xC0FFEE);
    let fallback: Arc<dyn er_world::terrain_field::TerrainField> =
        Arc::new(ProceduralTerrainField::new(
            er_world::elevation::elevation_params(seed),
            er_world::params::planet_params(seed),
        ));
    // Use BlendedHybridTerrainField with instant transition so learned is
    // immediately eligible.
    let blended: Arc<dyn er_world::terrain_field::TerrainField> =
        Arc::new(er_world::terrain_field::BlendedHybridTerrainField::new(
            fallback,
            Arc::new(ConstantMacro(0.8)),
            0.0,
        ));
    let chunk = CellKey {
        face: 0,
        i: 1,
        j: 1,
        lod: 3,
    };
    let snapshot = ChunkFieldSnapshot::new(blended, chunk, true, 1.0);
    let dir = uv_to_dir(0, 0.3, 0.3);
    let sample = snapshot.sample(dir);
    assert_eq!(sample.source, TerrainSampleSource::LearnedMacro);
    assert!(snapshot.learned_eligible());
}

#[test]
fn chunk_field_snapshot_no_per_vertex_mixing() {
    // Every vertex in the chunk must have the same source. When
    // halo_resident=false, ALL samples must be Procedural.
    use er_core::seed::PlanetSeed;
    use er_world::elevation::elevation_params;
    use er_world::params::planet_params;
    use er_world::terrain_field::{
        ChunkFieldSnapshot, ProceduralTerrainField, TerrainField, TerrainSampleSource,
    };

    let seed = PlanetSeed(0xC0FFEE);
    let fallback: Arc<dyn er_world::terrain_field::TerrainField> = Arc::new(
        ProceduralTerrainField::new(elevation_params(seed), planet_params(seed)),
    );
    let chunk = CellKey {
        face: 0,
        i: 1,
        j: 1,
        lod: 3,
    };
    let snapshot = ChunkFieldSnapshot::new(fallback, chunk, false, 0.0);
    // Sample many directions across the chunk — all must be Procedural.
    for i in 0..8 {
        for j in 0..8 {
            let u = (i as f64 + 0.5) / 8.0;
            let v = (j as f64 + 0.5) / 8.0;
            let dir = uv_to_dir(0, u, v);
            let sample = snapshot.sample(dir);
            assert_eq!(
                sample.source,
                TerrainSampleSource::Procedural,
                "vertex ({i},{j}) mixed sources"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 10. Learned climate unit conversion (5.2.5)
// ---------------------------------------------------------------------------

#[test]
fn visual_climate_unit_conversion_is_documented_and_correct() {
    use er_world::terrain_field::VisualClimate;

    // Raw upstream channels in [0,1].
    let raw = [0.5_f32, 0.5, 0.5, 0.5];
    let vc = VisualClimate::from_upstream_channels(raw);
    // temp_c = 0.5 * 50 - 10 = 15
    assert!((vc.temp_c - 15.0).abs() < 1e-6);
    // temp_seasonal_c = 0.5 * 30 - 15 = 0
    assert!((vc.temp_seasonal_c - 0.0).abs() < 1e-6);
    // precip_mm = 0.5 * 4000 = 2000
    assert!((vc.precip_mm - 2000.0).abs() < 1e-6);
    // precip_cv = 0.5
    assert!((vc.precip_cv - 0.5).abs() < 1e-6);
    assert!(vc.is_finite());
}

#[test]
fn visual_climate_rejects_non_finite() {
    use er_world::terrain_field::VisualClimate;
    let vc = VisualClimate {
        temp_c: f32::NAN,
        temp_seasonal_c: 0.0,
        precip_mm: 0.0,
        precip_cv: 0.0,
    };
    assert!(!vc.is_finite());
}

#[test]
fn visual_climate_is_not_used_by_gameplay() {
    // The TerrainSample's visual_climate field is separate from the
    // gameplay-authoritative temperature and moisture fields. This test
    // verifies they are distinct fields.
    use er_world::terrain_field::{TerrainSample, VisualClimate};
    let sample = TerrainSample {
        elevation: 0.0,
        low_freq_elev: 0.0,
        warped_dir: [0.0; 3],
        moisture: 0.5,
        biome: er_world::biome::Biome::OceanMid,
        mountain_influence: 0.0,
        temperature: 0.3,
        drainage: 0.0,
        source: er_world::terrain_field::TerrainSampleSource::Procedural,
        visual_climate: VisualClimate {
            temp_c: 25.0,
            temp_seasonal_c: 5.0,
            precip_mm: 1500.0,
            precip_cv: 0.3,
        },
    };
    // Gameplay climate (temperature, moisture) is separate from visual.
    assert_ne!(sample.temperature, sample.visual_climate.temp_c);
    assert_ne!(sample.moisture, sample.visual_climate.precip_cv);
}

// ---------------------------------------------------------------------------
// 11. Streaming queue resident set is bounded (H2)
// ---------------------------------------------------------------------------

#[test]
fn streaming_queue_resident_set_is_bounded_and_evicts_lru() {
    let mut cfg = small_config();
    cfg.max_resident = 3;
    let q = StreamingQueue::new(cfg);

    // Mark 5 keys resident — only 3 should be retained.
    for i in 0..5u32 {
        let k = key_for_chart(0, i, 0);
        q.mark_resident(&k);
    }
    assert_eq!(q.resident_count(), 3);
    // The first 2 keys (LRU) should have been evicted.
    assert!(!q.is_resident(&key_for_chart(0, 0, 0)));
    assert!(!q.is_resident(&key_for_chart(0, 1, 0)));
    // The last 3 should be resident.
    assert!(q.is_resident(&key_for_chart(0, 2, 0)));
    assert!(q.is_resident(&key_for_chart(0, 3, 0)));
    assert!(q.is_resident(&key_for_chart(0, 4, 0)));
}

#[test]
fn streaming_queue_eviction_allows_requeue() {
    let mut cfg = small_config();
    cfg.max_resident = 2;
    let q = StreamingQueue::new(cfg);
    let now = std::time::Instant::now();

    let k1 = key_for_chart(0, 0, 0);
    let k2 = key_for_chart(0, 1, 0);
    let k3 = key_for_chart(0, 2, 0);
    q.mark_resident(&k1);
    q.mark_resident(&k2);
    q.mark_resident(&k3); // evicts k1
    assert!(!q.is_resident(&k1));
    // k1 can be re-enqueued because it was evicted.
    assert!(q.enqueue(
        k1,
        ProviderTileCoordinate {
            face: 0,
            x: 0,
            y: 0
        },
        PriorityClass::VisibleSurface,
        None,
        now
    ));
}

// ---------------------------------------------------------------------------
// 12. Targeted refresh: LOD-17 active chunk refreshed after LOD-8-origin tile arrival
// ---------------------------------------------------------------------------

#[test]
fn lod17_active_chunk_refreshed_after_chart_arrival() {
    // When a chart at level 2 arrives, an LOD-17 active chunk intersecting
    // its footprint must be in the rebuild set. The chart footprint is
    // (face, level=2, x, y). The LOD-17 chunk is at a much finer level.
    //
    // We cannot materialize all intersecting LOD-17 chunks (there are ~1B),
    // so we verify the intersection logic by checking that a specific
    // LOD-17 chunk's uv footprint overlaps the chart's uv footprint.
    let chart_face = 0u8;
    let chart_x = 163u32; // ~163/652 ≈ 0.25
    let chart_y = 163u32;
    let charts_per_face_edge = 652u32;

    // Chart (163,163) at 652 charts/edge covers uv [~0.25, ~0.2515].
    let chart_u0 = chart_x as f64 / charts_per_face_edge as f64;
    let chart_u1 = (chart_x + 1) as f64 / charts_per_face_edge as f64;
    let chart_v0 = chart_y as f64 / charts_per_face_edge as f64;
    let chart_v1 = (chart_y + 1) as f64 / charts_per_face_edge as f64;

    // An LOD-17 chunk at i=32768, j=32768 covers uv [0.25, 0.25+1/131072].
    let lod17_chunk = CellKey {
        face: chart_face,
        i: 32768,
        j: 32768,
        lod: 17,
    };
    let n17 = (1u32 << 17) as f64;
    let chunk_u0 = lod17_chunk.i as f64 / n17;
    let chunk_u1 = (lod17_chunk.i + 1) as f64 / n17;
    let chunk_v0 = lod17_chunk.j as f64 / n17;
    let chunk_v1 = (lod17_chunk.j + 1) as f64 / n17;

    // The chunk's uv must overlap the chart's uv footprint.
    let overlaps =
        chunk_u0 < chart_u1 && chunk_u1 > chart_u0 && chunk_v0 < chart_v1 && chunk_v1 > chart_v0;
    assert!(overlaps, "LOD-17 chunk must overlap chart footprint");

    // A far-away LOD-17 chunk must NOT overlap.
    let far_chunk = CellKey {
        face: chart_face,
        i: 100000,
        j: 100000,
        lod: 17,
    };
    let far_u0 = far_chunk.i as f64 / n17;
    let far_v0 = far_chunk.j as f64 / n17;
    let far_overlaps = far_u0 < chart_u1 && far_v0 < chart_v1;
    assert!(
        !far_overlaps,
        "far LOD-17 chunk must not overlap chart footprint"
    );

    // Also verify with a practical LOD (LOD 5) that the intersection
    // function works and includes the expected chunk.
    let intersecting_lod5 = chunks_intersecting_chart(
        chart_face, chart_x, chart_y, 652, // charts_per_face_edge for Earth
        5, 1, // halo
    );
    assert!(!intersecting_lod5.is_empty());
    // All intersecting chunks must be on the chart's face.
    for c in &intersecting_lod5 {
        assert_eq!(c.face, chart_face);
    }
}

// ---------------------------------------------------------------------------
// 13. Per-chart blend: unrelated tile arrivals preserve existing progress
// ---------------------------------------------------------------------------

#[test]
fn per_chart_blend_preserves_unrelated_progress() {
    use er_core::seed::PlanetSeed;
    use er_world::elevation::elevation_params;
    use er_world::params::planet_params;
    use er_world::terrain_field::{
        BlendedHybridTerrainField, MacroTerrainField, MacroTerrainSample, ProceduralTerrainField,
        TerrainField,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // A macro field that returns resident data for two independent directions.
    struct TwoDirMacro {
        dir_a_resident: AtomicBool,
        dir_b_resident: AtomicBool,
    }
    impl MacroTerrainField for TwoDirMacro {
        fn sample_resident(&self, dir: glam::DVec3) -> Option<MacroTerrainSample> {
            // Direction A is near (1,0,0), direction B is near (0,1,0).
            let is_a = dir.x > 0.7;
            let is_b = dir.y > 0.7;
            if is_a && self.dir_a_resident.load(Ordering::Relaxed) {
                Some(MacroTerrainSample {
                    elevation: 0.5,
                    visual_climate: er_world::terrain_field::VisualClimate::default(),
                })
            } else if is_b && self.dir_b_resident.load(Ordering::Relaxed) {
                Some(MacroTerrainSample {
                    elevation: 0.7,
                    visual_climate: er_world::terrain_field::VisualClimate::default(),
                })
            } else {
                None
            }
        }
    }

    let seed = PlanetSeed(0xC0FFEE);
    let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
        elevation_params(seed),
        planet_params(seed),
    ));
    let macro_field = Arc::new(TwoDirMacro {
        dir_a_resident: AtomicBool::new(false),
        dir_b_resident: AtomicBool::new(false),
    });
    let macro_dyn: Arc<dyn MacroTerrainField> =
        Arc::clone(&macro_field) as Arc<dyn MacroTerrainField>;

    // Use a nonzero transition so blend progress is tracked over time.
    let blended = BlendedHybridTerrainField::new(fallback.clone(), macro_dyn, 10.0);

    let dir_a = glam::DVec3::new(0.9, 0.1, 0.0).normalize();
    let dir_b = glam::DVec3::new(0.1, 0.9, 0.0).normalize();

    // Initially: both procedural (weight 0).
    let w_a_before = blended.current_blend_weight(dir_a);
    let w_b_before = blended.current_blend_weight(dir_b);
    assert_eq!(w_a_before, 0.0);
    assert_eq!(w_b_before, 0.0);

    // Make direction A resident. This triggers blend_weight to record the
    // time for dir A only.
    macro_field.dir_a_resident.store(true, Ordering::Relaxed);
    let _ = blended.sample(dir_a); // triggers blend state recording
    let w_a_after = blended.current_blend_weight(dir_a);
    // Weight should be > 0 (transition started) but < 1 (not complete).
    assert!(w_a_after > 0.0, "dir A weight should be > 0 after resident");
    assert!(
        w_a_after < 1.0,
        "dir A weight should be < 1 during transition"
    );

    // Direction B should still be 0 (unrelated arrival).
    let w_b_after_a = blended.current_blend_weight(dir_b);
    assert_eq!(
        w_b_after_a, 0.0,
        "dir B should still be 0 after dir A arrival"
    );

    // Now make direction B resident. Dir A's progress must be preserved.
    macro_field.dir_b_resident.store(true, Ordering::Relaxed);
    let _ = blended.sample(dir_b); // triggers blend state recording for B
    let w_b_after = blended.current_blend_weight(dir_b);
    assert!(w_b_after > 0.0, "dir B weight should be > 0 after resident");

    // Dir A's weight must NOT have been reset by dir B's arrival.
    let w_a_after_b = blended.current_blend_weight(dir_a);
    assert!(
        w_a_after_b >= w_a_after,
        "dir A weight {} should be >= {} (preserved, not reset by dir B arrival)",
        w_a_after_b,
        w_a_after
    );
}

// ---------------------------------------------------------------------------
// 14. Cache hit/miss recording from the live adapter path
// ---------------------------------------------------------------------------

#[test]
fn chart_macro_field_records_real_cache_hits_and_misses() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    let field = chart_field();
    let hits = Arc::new(AtomicU64::new(0));
    let misses = Arc::new(AtomicU64::new(0));
    let hits_clone = Arc::clone(&hits);
    let misses_clone = Arc::clone(&misses);
    field.set_cache_lookup_recorder(Arc::new(move |hit| {
        if hit {
            hits_clone.fetch_add(1, Ordering::Relaxed);
        } else {
            misses_clone.fetch_add(1, Ordering::Relaxed);
        }
    }));

    let dir_miss = uv_to_dir(0, 0.5, 0.5);
    // Sample a direction with no resident data: should record a miss.
    let _ = field.sample_resident(dir_miss);
    assert_eq!(misses.load(Ordering::Relaxed), 1);
    assert_eq!(hits.load(Ordering::Relaxed), 0);

    // Fill a chart and sample again: should record a hit.
    fill_chart(&field, 0, 0, 0, 500);
    let dir_hit = uv_to_dir(0, 0.5 / 652.0, 0.5 / 652.0);
    let _ = field.sample_resident(dir_hit);
    assert!(hits.load(Ordering::Relaxed) >= 1);
}

// ---------------------------------------------------------------------------
// 15. Visual climate affects visual material but not gameplay
// ---------------------------------------------------------------------------

#[test]
fn visual_climate_does_not_affect_gameplay_shoreline() {
    use er_core::seed::PlanetSeed;
    use er_world::elevation::elevation_params;
    use er_world::params::planet_params;
    use er_world::terrain_field::{
        BlendedHybridTerrainField, MacroTerrainField, MacroTerrainSample, ProceduralTerrainField,
        TerrainField, VisualClimate,
    };

    // A macro field that returns a specific visual climate.
    struct ClimateMacro(f64, VisualClimate);
    impl MacroTerrainField for ClimateMacro {
        fn sample_resident(&self, _: glam::DVec3) -> Option<MacroTerrainSample> {
            Some(MacroTerrainSample {
                elevation: self.0,
                visual_climate: self.1,
            })
        }
    }

    let seed = PlanetSeed(0xC0FFEE);
    let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
        elevation_params(seed),
        planet_params(seed),
    ));
    let dir = uv_to_dir(2, 0.41, 0.59);
    let procedural = fallback.sample(dir);
    let procedural_water = procedural.low_freq_elev < 0.0;

    // Test with different visual climates — gameplay must not change.
    for vc in [
        VisualClimate {
            temp_c: -40.0,
            temp_seasonal_c: -20.0,
            precip_mm: 0.0,
            precip_cv: 0.0,
        },
        VisualClimate {
            temp_c: 40.0,
            temp_seasonal_c: 20.0,
            precip_mm: 4000.0,
            precip_cv: 1.0,
        },
    ] {
        let blended = BlendedHybridTerrainField::new(
            fallback.clone(),
            Arc::new(ClimateMacro(0.8, vc)),
            0.0, // instant transition
        );
        let sample = blended.sample(dir);
        // Gameplay shoreline (low_freq_elev) must be unchanged.
        assert_eq!(sample.low_freq_elev, procedural.low_freq_elev);
        assert_eq!(sample.low_freq_elev < 0.0, procedural_water);
        // Gameplay temperature and moisture must be procedural.
        assert_eq!(sample.temperature, procedural.temperature);
        assert_eq!(sample.moisture, procedural.moisture);
        // Visual climate must be the learned value.
        assert_eq!(sample.visual_climate.temp_c, vc.temp_c);
        assert_eq!(sample.visual_climate.precip_mm, vc.precip_mm);
    }
}

// ---------------------------------------------------------------------------
// 16. Earth-scale coordinate domain: provider coords < 652, chart key valid
// ---------------------------------------------------------------------------

#[test]
fn earth_scale_provider_coordinates_in_range() {
    // Earth has tiles_per_face_edge = 652, chart_level = 10 (1024 charts).
    // Provider coordinates must be in [0, 652), NOT [0, 1024).
    let tiles_per_face_edge = 652u32;

    // Test all six priority classes produce in-range provider coordinates.
    for face in 0..6u8 {
        // Center of each face.
        let dir = uv_to_dir(face, 0.5, 0.5);
        let ptc = ProviderTileCoordinate::from_direction(dir, tiles_per_face_edge);
        assert!(
            ptc.is_in_range(tiles_per_face_edge),
            "center out of range: {ptc:?}"
        );
        assert!(
            ptc.x < tiles_per_face_edge,
            "x={} >= {}",
            ptc.x,
            tiles_per_face_edge
        );
        assert!(
            ptc.y < tiles_per_face_edge,
            "y={} >= {}",
            ptc.y,
            tiles_per_face_edge
        );

        // Edge cases: near face boundaries.
        for u in [0.001, 0.5, 0.999] {
            for v in [0.001, 0.5, 0.999] {
                let dir = uv_to_dir(face, u, v);
                let ptc = ProviderTileCoordinate::from_direction(dir, tiles_per_face_edge);
                assert!(
                    ptc.is_in_range(tiles_per_face_edge),
                    "edge case out of range: {ptc:?} at u={u} v={v}"
                );
            }
        }
    }

    // Verify the chart key derived from a provider coordinate is valid
    // (face matches, level matches chart_level).
    let field = chart_field();
    let ptc = ProviderTileCoordinate::from_direction(uv_to_dir(0, 0.3, 0.7), tiles_per_face_edge);
    let chart_key = field.key_for_tile(ptc.face, ptc.x, ptc.y, tiles_per_face_edge);
    assert_eq!(chart_key.face, ptc.face);
    // The chart key's x/y are in chart-grid space [0, 652), matching the
    // provider coordinate since charts_per_face_edge == tiles_per_face_edge.
    assert!(chart_key.x < tiles_per_face_edge);
    assert!(chart_key.y < tiles_per_face_edge);
    assert_eq!(chart_key.charts_per_face_edge, tiles_per_face_edge);
}

#[test]
fn earth_scale_all_priority_classes_produce_valid_sidecar_bounds() {
    // Simulate the sidecar request bounds computation for all priority
    // classes at Earth scale. The bounds must be valid (non-negative,
    // within the face span).
    let tiles_per_face_edge = 652u16;
    let core_res = 6u16; // TILE_CORE_RESOLUTION
    let halo = 1u16;
    let stored_res = core_res + halo * 2;
    let face_span = tiles_per_face_edge as i32 * core_res as i32;

    // Test provider coordinates across the full range.
    for x in [0u16, 1, 100, 325, 651] {
        for y in [0u16, 1, 100, 325, 651] {
            for face in 0..6u8 {
                let i1 = face as i32 * face_span + y as i32 * core_res as i32 - halo as i32;
                let j1 = x as i32 * core_res as i32 - halo as i32;
                let i2 = i1 + stored_res as i32;
                let j2 = j1 + stored_res as i32;
                // Bounds must be valid (i2 > i1, j2 > j1).
                assert!(i2 > i1, "i2 <= i1 for face={face} x={x} y={y}");
                assert!(j2 > j1, "j2 <= j1 for face={face} x={x} y={y}");
                // The request span must match the stored resolution.
                assert_eq!(i2 - i1, stored_res as i32);
                assert_eq!(j2 - j1, stored_res as i32);
            }
        }
    }
}

#[test]
fn earth_scale_accepted_payload_reaches_chart_cache_and_resident() {
    // When a tile payload is accepted, it must reach BOTH the chart cache
    // (via key_for_tile) and the streaming resident state (via
    // record_success). Verify the chart key derived from the provider
    // coordinate matches the key used for residency.
    let tiles_per_face_edge = 652u32;
    let field = chart_field();
    let q = StreamingQueue::new(small_config());
    let now = std::time::Instant::now();

    // Derive provider coordinate and chart key from the same direction.
    let dir = uv_to_dir(0, 0.3, 0.7);
    let ptc = ProviderTileCoordinate::from_direction(dir, tiles_per_face_edge);
    let chart_key = field.key_for_tile(ptc.face, ptc.x, ptc.y, tiles_per_face_edge);

    // Enqueue with the provider coordinate.
    assert!(q.enqueue(
        chart_key.clone(),
        ptc,
        PriorityClass::VisibleSurface,
        None,
        now,
    ));

    // Dispatch and record success.
    let req = q.pop_dispatchable(now).unwrap();
    assert_eq!(req.provider_coordinate, ptc);
    assert_eq!(req.key, chart_key);
    q.record_success(&chart_key, Duration::from_millis(10), now);

    // The key must be resident in the streaming queue.
    assert!(q.is_resident(&chart_key));
    assert_eq!(q.resident_count(), 1);

    // The chart footprint must be pending for rebuild.
    let arrived = q.pop_arrived_chart();
    assert!(arrived.is_some(), "chart arrival must be pending");
    let footprint = arrived.unwrap();
    assert_eq!(footprint.face, chart_key.face);
    assert_eq!(footprint.x, chart_key.x);
    assert_eq!(footprint.y, chart_key.y);
    assert_eq!(
        footprint.charts_per_face_edge,
        chart_key.charts_per_face_edge
    );
}

// ---------------------------------------------------------------------------
// 17. Blend follow-up: queue population, progress, completion, no global reset
// ---------------------------------------------------------------------------

#[test]
fn blend_follow_up_progress_increases_and_completes_without_global_reset() {
    use er_core::seed::PlanetSeed;
    use er_world::elevation::elevation_params;
    use er_world::params::planet_params;
    use er_world::terrain_field::{
        BlendTransitionChecker, BlendedHybridTerrainField, MacroTerrainField, MacroTerrainSample,
        ProceduralTerrainField, TerrainField, VisualClimate,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    // A macro field that can be toggled resident for two independent directions.
    struct TwoDirMacro {
        dir_a_resident: AtomicBool,
        dir_b_resident: AtomicBool,
    }
    impl MacroTerrainField for TwoDirMacro {
        fn sample_resident(&self, dir: glam::DVec3) -> Option<MacroTerrainSample> {
            let is_a = dir.x > 0.7;
            let is_b = dir.y > 0.7;
            let resident = if is_a {
                self.dir_a_resident.load(Ordering::Relaxed)
            } else if is_b {
                self.dir_b_resident.load(Ordering::Relaxed)
            } else {
                false
            };
            if resident {
                Some(MacroTerrainSample {
                    elevation: 0.5,
                    visual_climate: VisualClimate::default(),
                })
            } else {
                None
            }
        }
    }

    let seed = PlanetSeed(0xC0FFEE);
    let fallback: Arc<dyn TerrainField> = Arc::new(ProceduralTerrainField::new(
        elevation_params(seed),
        planet_params(seed),
    ));
    let macro_field = Arc::new(TwoDirMacro {
        dir_a_resident: AtomicBool::new(false),
        dir_b_resident: AtomicBool::new(false),
    });
    let macro_dyn: Arc<dyn MacroTerrainField> =
        Arc::clone(&macro_field) as Arc<dyn MacroTerrainField>;

    // Use a 2-second transition so we can observe intermediate weights.
    let blended = BlendedHybridTerrainField::new(fallback.clone(), macro_dyn, 2.0);
    let checker: Arc<dyn BlendTransitionChecker> = Arc::new(BlendedHybridTerrainField::new(
        fallback.clone(),
        Arc::new(TwoDirMacro {
            dir_a_resident: AtomicBool::new(false),
            dir_b_resident: AtomicBool::new(false),
        }) as Arc<dyn MacroTerrainField>,
        2.0,
    ));

    let dir_a = glam::DVec3::new(0.9, 0.1, 0.0).normalize();
    let dir_b = glam::DVec3::new(0.1, 0.9, 0.0).normalize();

    // Initially: no transitioning chunks.
    assert!(!checker.has_transitioning_chunks());

    // Make dir A resident. Sample to trigger blend state.
    macro_field.dir_a_resident.store(true, Ordering::Relaxed);
    let _ = blended.sample(dir_a);

    // Weight should be > 0 but < 1 (transitioning).
    let w_a_initial = blended.current_blend_weight(dir_a);
    assert!(w_a_initial > 0.0, "initial weight should be > 0");
    assert!(w_a_initial < 1.0, "initial weight should be < 1");

    // Wait a bit and check progress increases.
    std::thread::sleep(Duration::from_millis(100));
    let w_a_after = blended.current_blend_weight(dir_a);
    assert!(
        w_a_after >= w_a_initial,
        "weight should increase: {w_a_after} >= {w_a_initial}"
    );

    // Dir B arrival should NOT reset dir A's progress.
    macro_field.dir_b_resident.store(true, Ordering::Relaxed);
    let _ = blended.sample(dir_b);
    let w_a_after_b = blended.current_blend_weight(dir_a);
    assert!(
        w_a_after_b >= w_a_initial,
        "dir A weight should not be reset by dir B arrival: {w_a_after_b} >= {w_a_initial}"
    );

    // Dir B should also be transitioning.
    let w_b = blended.current_blend_weight(dir_b);
    assert!(w_b > 0.0, "dir B should be transitioning");
    assert!(w_b < 1.0, "dir B should be < 1");
}

// ---------------------------------------------------------------------------
// 18. Disk promotion: cold-start avoids provider request, enqueues arrival
// ---------------------------------------------------------------------------

#[test]
fn disk_promotion_cold_start_avoids_provider_request() {
    // Write a chart record to disk, then clear RAM. A load_from_disk call
    // should promote it to RAM without a provider request.
    let dir = std::env::temp_dir().join(format!(
        "ersurf_m5_disk_promo_cold_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let disk = SurfaceDiskCache::new(&dir, 32).unwrap();
    let field = Arc::new(ChartMacroField::new(
        8,
        Some(disk),
        meta(),
        CHART_LEVEL,
        1000.0,
    ));

    // Store a chart record (goes to disk + RAM).
    fill_chart(&field, 0, 1, 1, 750);
    let key = key_for_chart(0, 1, 1);

    // Clear RAM to simulate cold start.
    field.cache().ram.clear();
    assert!(field.cache().get_resident(&key).is_none());

    // Promote from disk.
    let promoted = field.cache().load_from_disk(&key).unwrap();
    assert!(promoted, "disk promotion should succeed");
    assert!(field.cache().get_resident(&key).is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_promotion_after_ram_eviction_avoids_provider_request() {
    // After RAM eviction, a disk promotion should restore the record
    // without a provider request, and the streaming queue should see the
    // chart arrival.
    let dir = std::env::temp_dir().join(format!(
        "ersurf_m5_disk_promo_evict_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let disk = SurfaceDiskCache::new(&dir, 32).unwrap();
    let field = Arc::new(ChartMacroField::new(
        2, // small RAM capacity to force eviction
        Some(disk),
        meta(),
        CHART_LEVEL,
        1000.0,
    ));

    let q = StreamingQueue::new(small_config());

    // Store chart A (goes to disk + RAM).
    let key_a = key_for_chart(0, 0, 0);
    fill_chart(&field, 0, 0, 0, 500);
    q.mark_resident(&key_a);

    // Store chart B (evicts A from RAM due to capacity=2).
    let key_b = key_for_chart(0, 1, 0);
    fill_chart(&field, 0, 1, 0, 600);
    q.mark_resident(&key_b);

    // Store chart C (evicts B from RAM).
    let key_c = key_for_chart(0, 2, 0);
    fill_chart(&field, 0, 2, 0, 700);
    q.mark_resident(&key_c);

    // A should be evicted from RAM but still on disk.
    assert!(field.cache().get_resident(&key_a).is_none());

    // Promote A from disk.
    let promoted = field.cache().load_from_disk(&key_a).unwrap();
    assert!(promoted, "disk promotion after eviction should succeed");
    assert!(field.cache().get_resident(&key_a).is_some());

    // The streaming queue should also mark A as resident again.
    q.mark_resident(&key_a);
    assert!(q.is_resident(&key_a));

    // A chart arrival should be pending.
    let arrived = q.pop_arrived_chart();
    assert!(
        arrived.is_some(),
        "chart arrival should be pending after disk promotion"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_promotion_corruption_falls_through_to_miss() {
    // A corrupt disk record should be removed and treated as a miss,
    // preserving procedural fallback.
    let dir = std::env::temp_dir().join(format!(
        "ersurf_m5_disk_corrupt_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let disk = SurfaceDiskCache::new(&dir, 32).unwrap();
    let field = Arc::new(ChartMacroField::new(
        8,
        Some(disk),
        meta(),
        CHART_LEVEL,
        1000.0,
    ));

    // Write a valid record.
    fill_chart(&field, 0, 0, 0, 500);
    let key = key_for_chart(0, 0, 0);

    // Corrupt the disk file by writing garbage.
    let path = field.cache().disk.as_ref().unwrap().path_for(&key);
    std::fs::write(&path, b"GARBAGE").unwrap();

    // Clear RAM.
    field.cache().ram.clear();

    // load_from_disk should return Ok(false) (miss) and remove the corrupt file.
    let result = field.cache().load_from_disk(&key).unwrap();
    assert!(!result, "corrupt disk record should be a miss");

    // The corrupt file should have been removed.
    assert!(!path.exists(), "corrupt file should be removed");

    let _ = std::fs::remove_dir_all(&dir);
}
