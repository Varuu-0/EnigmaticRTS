//! Experimental loopback adapter for Terrain Diffusion's Flask API.
//!
//! This is intentionally feature-gated and opt-in. The upstream model is
//! planar, while the game is a cube-sphere, so the temporary face-atlas mapping
//! is a visual evaluation path rather than normal gameplay terrain.
//!
//! ## Native / upsampling model metadata
//!
//! `TerrainDiffusionMetadata` is the single source of truth for the source
//! model's native resolution, the API upsampling factor, the cube-face atlas
//! layout, and the horizontal projection parameters. The native model resolves
//! at `NATIVE_PIXEL_SCALE_M` (~30 m per pixel). When `api_scale > 1` the
//! sidecar interpolates — this adapter records that status in the diagnostic
//! resource rather than letting the upsampled effective resolution masquerade
//! as a learned sub-native resolution.

use bevy::prelude::*;
use bevy::tasks::{futures::check_ready, AsyncComputeTaskPool, Task};
use er_core::math::{dir_to_uv, uv_to_dir};
use er_core::seed::PlanetSeed;
use er_world::streaming::{PriorityClass, ProviderTileCoordinate, StreamingConfig, StreamingQueue};
use er_world::surface_cache::{
    ChartMacroField, CreationMetadata, SurfaceDiskCache, SurfaceTileRecord,
};
use er_world::surface_charts::{
    ChartOwnership, SurfaceChartMetadata, SURFACE_CHART_PROJECTION_REVISION,
};
use er_world::{
    LearnedTerrainTile, LearnedTileCache, LearnedTileGeneration, LearnedTileKey, TileCoordinate,
};
use glam::DVec3;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Maximum number of per-frame provider-timing samples retained in the ring
/// buffer. At 144 fps this covers ~7 minutes; enough for a 30-minute stress
/// run to hold a representative tail without unbounded growth.
const PROVIDER_TIMING_HISTORY_CAP: usize = 60_000;

pub const NATIVE_MODEL_RESOLUTION: u16 = 512;
pub const NATIVE_PIXEL_SCALE_M: u16 = 30;

const TILE_CORE_RESOLUTION: u16 = 512;

/// Compute the number of native-model tiles per cube-face edge so that each
/// 512×30 m tile maps at native scale across the face's quarter‑circumference
/// arc.  Minimum 1.
pub fn compute_tiles_per_face_edge(planet_radius_m: f64) -> u16 {
    let arc_quarter_m = std::f64::consts::PI * planet_radius_m / 2.0;
    let tile_width_m = NATIVE_MODEL_RESOLUTION as f64 * NATIVE_PIXEL_SCALE_M as f64;
    ((arc_quarter_m / tile_width_m).ceil() as u64)
        .max(1)
        .min(u16::MAX as u64) as u16
}

/// Compute the chart quadtree level for a given tiles-per-face-edge count.
/// The level is the smallest power-of-two `2^level` that is >= the tile
/// count, so every provider tile maps to a valid chart without wrapping.
///
/// For power-of-two tile counts (e.g. miniature = 4) this is exact
/// (level 2 = 4 charts). For non-power-of-two counts (e.g. Earth = 652)
/// the chart grid is the next power of two (1024 at level 10), and tiles
/// 0..N-1 map to charts 0..N-1 with the trailing charts never populated.
pub fn compute_chart_level(tiles_per_face_edge: u16) -> u8 {
    (tiles_per_face_edge as u32)
        .next_power_of_two()
        .trailing_zeros()
        .min(31) as u8
}
const TILE_HALO: u16 = 1;
const API_SCALE: u8 = 1;
const ELEVATION_SCALE_M: f64 = 1000.0;
const CAMERA_PREFETCH_RADIUS: i32 = 2;
const SIDECAR_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// RAM LRU capacity for resident surface tiles.
const STREAMING_RAM_CAPACITY: usize = 256;
/// Disk cache max entries for cold surface tiles.
const STREAMING_DISK_CAPACITY: usize = 2048;
/// Directory for the learned-surface disk cache.
const STREAMING_DISK_DIR: &str = "learned_surface_cache";

/// Single source of truth for the native model resolution, atlas layout,
/// API upsampling factor, and horizontal projection metadata.
///
/// `api_scale` is the factor the sidecar API multiplies native resolution by.
/// When `api_scale > 1` the sidecar interpolates; `is_upsampled()` returns
/// true so diagnostics can flag that the effective resolution is not a
/// learned sub-native resolution.
#[derive(Clone, Debug)]
pub struct TerrainDiffusionMetadata {
    pub native_resolution: u16,
    pub native_pixel_scale_m: u16,
    pub api_scale: u8,
    pub halo_samples: u16,
    pub tiles_per_face_edge: u16,
}

impl TerrainDiffusionMetadata {
    pub fn stored_resolution(&self) -> u16 {
        self.native_resolution + self.halo_samples * 2
    }

    pub fn is_upsampled(&self) -> bool {
        self.api_scale > 1
    }

    pub fn effective_pixel_scale_m(&self) -> u16 {
        let divisor = if self.api_scale == 0 {
            1_u8
        } else {
            self.api_scale
        };
        (self.native_pixel_scale_m as f64 / divisor as f64) as u16
    }
}

impl Default for TerrainDiffusionMetadata {
    fn default() -> Self {
        Self {
            native_resolution: NATIVE_MODEL_RESOLUTION,
            native_pixel_scale_m: NATIVE_PIXEL_SCALE_M,
            api_scale: API_SCALE,
            halo_samples: TILE_HALO,
            tiles_per_face_edge: 4,
        }
    }
}

#[derive(Clone)]
pub struct TerrainDiffusionConfig {
    pub endpoint: SocketAddr,
    pub seed: PlanetSeed,
    pub api_scale: u8,
    pub metadata: TerrainDiffusionMetadata,
}

/// Compact public runtime diagnostic resource.
///
/// Updated every frame by the plugin's `Update` systems. Consumers can read
/// its fields directly (they are clone-able plain data) without touching the
/// private `TerrainDiffusionRuntime`.
#[derive(Resource, Clone, Debug)]
pub struct TerrainDiffusionDiagnostic {
    pub metadata: TerrainDiffusionMetadata,
    pub tile_count: usize,
    pub queue_depth: usize,
    pub request_failures: u32,
    pub last_latency_ms: Option<f64>,
    pub fallback_active: bool,
    pub invalid_tiles_discarded: u32,
    pub in_flight: bool,
    /// M5 streaming telemetry: resident tiles, in-flight count, cache hit
    /// rate, fallback percentage, latency percentiles, rebuild counts, and
    /// service health.
    pub streaming: StreamingDiagnostic,
}

/// Structured M5 streaming diagnostic snapshot, suitable for machine-readable
/// stress reports.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct StreamingDiagnostic {
    pub resident_tiles: usize,
    pub pending_in_flight: usize,
    pub failed_total: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
    pub fallback_percent: f64,
    pub latency_p50_ms: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    pub rebuilds_queued: u64,
    pub rebuilds_completed: u64,
    pub health: String,
    pub provider_attempts: u64,
    pub provider_timeouts: u64,
    pub provider_cancellations: u64,
    pub provider_coalesced: u64,
    pub provider_stale_discarded: u64,
}

impl Default for TerrainDiffusionDiagnostic {
    fn default() -> Self {
        Self {
            metadata: TerrainDiffusionMetadata::default(),
            tile_count: 0,
            queue_depth: 0,
            request_failures: 0,
            last_latency_ms: None,
            fallback_active: true,
            invalid_tiles_discarded: 0,
            in_flight: false,
            streaming: StreamingDiagnostic::default(),
        }
    }
}

/// Bounded ring buffer of per-frame main-thread provider work (microseconds).
///
/// Only the Bevy `Update` systems owned by the Terrain Diffusion plugin are
/// timed: `queue_camera_tiles`, `poll_tile_request`, `start_tile_request`, and
/// `publish_diagnostic`. The HTTP/TCP request and model inference execute on
/// the async compute thread pool and are explicitly **not** counted — they are
/// sidecar/network/model work, not main-thread work.
///
/// This is the source of truth for the Milestone 3 "provider-attributable
/// main-thread hitch" gate. The stress runner reads `percentile_ms(95)` and
/// requires it to be <= 1 ms.
#[derive(Resource, Clone, Debug)]
pub struct ProviderTimingHistory {
    samples_us: VecDeque<u64>,
    current_frame_us: u64,
}

impl Default for ProviderTimingHistory {
    fn default() -> Self {
        Self {
            samples_us: VecDeque::with_capacity(4096),
            current_frame_us: 0,
        }
    }
}

impl ProviderTimingHistory {
    /// Add microseconds of main-thread work to the current frame's accumulator.
    #[inline]
    pub fn add_us(&mut self, us: u64) {
        self.current_frame_us = self.current_frame_us.saturating_add(us);
    }

    /// Finalize the current frame and push it into the ring buffer.
    pub fn finalize_frame(&mut self) {
        if self.samples_us.len() >= PROVIDER_TIMING_HISTORY_CAP {
            self.samples_us.pop_front();
        }
        self.samples_us.push_back(self.current_frame_us);
        self.current_frame_us = 0;
    }

    /// Number of recorded frames.
    pub fn len(&self) -> usize {
        self.samples_us.len()
    }

    /// True when no frames have been recorded.
    pub fn is_empty(&self) -> bool {
        self.samples_us.is_empty()
    }

    /// Compute a percentile in milliseconds from the recorded frame samples.
    /// Returns `None` when no samples exist.
    pub fn percentile_ms(&self, pct: f64) -> Option<f64> {
        if self.samples_us.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = self.samples_us.iter().copied().collect();
        sorted.sort_unstable();
        let idx = ((pct / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        let val = sorted[idx.min(sorted.len() - 1)];
        Some(val as f64 / 1000.0)
    }

    /// Maximum main-thread provider work observed, in milliseconds.
    pub fn max_ms(&self) -> Option<f64> {
        self.samples_us.iter().max().map(|v| *v as f64 / 1000.0)
    }

    /// Mean main-thread provider work in milliseconds.
    pub fn mean_ms(&self) -> Option<f64> {
        if self.samples_us.is_empty() {
            return None;
        }
        let sum: u128 = self.samples_us.iter().map(|&v| v as u128).sum();
        Some(sum as f64 / self.samples_us.len() as f64 / 1000.0)
    }
}

pub struct TerrainDiffusionStartup {
    /// Legacy diagnostic cache. Retained for backward compatibility with
    /// existing tests/diagnostics but the live hybrid field now sources
    /// from `chart_field`.
    pub cache: Arc<LearnedTileCache>,
    /// M4 sphere-native chart-backed macro field. This is the live runtime
    /// source that `HybridTerrainField` samples from.
    pub chart_field: Arc<ChartMacroField>,
    /// M5 production streaming queue: bounded priority queue with
    /// coalescing, backoff, health, and telemetry.
    pub streaming_queue: Arc<StreamingQueue>,
    pub config: TerrainDiffusionConfig,
}

/// Reads `--terrain-diffusion` and optional `--terrain-diffusion-port <port>`.
/// Model streaming is deliberately disabled under screenshot and benchmark
/// modes so their output remains deterministic.
pub fn startup_from_args(
    disabled: bool,
    planet_radius_m: f64,
    seed: PlanetSeed,
) -> Option<TerrainDiffusionStartup> {
    let args: Vec<String> = std::env::args().collect();
    // Stress and acceptance modes require the full learned streaming pipeline
    // even when the explicit visual-evaluation flag is omitted.
    if !args.iter().any(|arg| arg == "--terrain-diffusion")
        && !args.iter().any(|arg| arg == "--learned-stress")
        && !args.iter().any(|arg| arg == "--m5-test")
    {
        return None;
    }
    if disabled {
        warn!("Terrain diffusion disabled in benchmark and screenshot modes");
        return None;
    }

    let port = args
        .windows(2)
        .find(|args| args[0] == "--terrain-diffusion-port")
        .and_then(|args| args[1].parse::<u16>().ok())
        .unwrap_or(8000);
    let api_scale_arg = args
        .windows(2)
        .find(|args| args[0] == "--terrain-diffusion-scale")
        .and_then(|args| args[1].parse::<u8>().ok())
        .unwrap_or(API_SCALE);

    let final_api_scale = if api_scale_arg == 0 {
        warn!("Terrain Diffusion API scale was set to 0; falling back to native scale 1");
        1u8
    } else if api_scale_arg > 1 {
        warn!(
            api_scale = api_scale_arg,
            "Terrain Diffusion API upsampling detected — effective resolution is NOT native sub-30m; underlying model remains {NATIVE_PIXEL_SCALE_M}m"
        );
        api_scale_arg
    } else {
        api_scale_arg
    };

    let tiles_per_face_edge = compute_tiles_per_face_edge(planet_radius_m);
    let metadata = TerrainDiffusionMetadata {
        api_scale: final_api_scale,
        tiles_per_face_edge,
        ..TerrainDiffusionMetadata::default()
    };
    let model_revision = format!(
        "terrain-diffusion-native{NATIVE_PIXEL_SCALE_M}m-api-scale{}_{}",
        final_api_scale,
        if final_api_scale > 1 {
            "UPSAMPLED"
        } else {
            "NATIVE"
        }
    );
    let generation = LearnedTileGeneration {
        model_revision: model_revision.clone(),
        seed: seed.0,
        projection_revision: 2,
        pixel_scale_m: metadata.native_pixel_scale_m,
        sea_level_datum_m: 0,
    };
    let cache = Arc::new(LearnedTileCache::new(
        generation,
        tiles_per_face_edge,
        TILE_CORE_RESOLUTION,
        ELEVATION_SCALE_M,
    ));

    // Build the M4 sphere-native chart metadata and chart-backed macro field.
    // The chart level is the quadtree level whose 2^level charts per edge is
    // >= tiles_per_face_edge, so every provider tile maps to a valid chart
    // without wrapping. For power-of-two tile counts (e.g. miniature = 4)
    // this is exact (level 2 = 4 charts); for non-power-of-two counts (e.g.
    // Earth = 652) the chart grid is the next power of two (1024), and tiles
    // 0..N-1 map to charts 0..N-1 with the trailing charts simply never
    // populated. This never reintroduces a four-tile-per-face assumption:
    // the chart count derives from the planet radius, not a hard-coded 4.
    let chart_level = compute_chart_level(tiles_per_face_edge);
    let chart_metadata = SurfaceChartMetadata {
        seed: seed.0,
        projection_revision: SURFACE_CHART_PROJECTION_REVISION,
        model_revision,
        conditioning_revision: 1,
        residual_revision: 1,
        sea_level_datum_m: 0,
        pixel_scale_m: NATIVE_PIXEL_SCALE_M as u32,
        halo_samples: TILE_HALO as u32,
        core_resolution: TILE_CORE_RESOLUTION as u32,
        ownership: ChartOwnership::LearnedReliefProceduralShoreline,
        planet_radius_m: planet_radius_m as u64,
        charts_per_face_edge: tiles_per_face_edge as u32,
    };

    // Enable the M4 disk cache on the runtime path with bounded capacities.
    // Disk I/O happens only on the background store path, never on the
    // mesh-worker read path.
    let disk_cache_dir = PathBuf::from(STREAMING_DISK_DIR);
    let disk_cache = SurfaceDiskCache::new(&disk_cache_dir, STREAMING_DISK_CAPACITY)
        .map_err(|error| {
            warn!(?disk_cache_dir, %error, "Could not open learned-surface disk cache; falling back to RAM-only");
        })
        .ok();
    let chart_field = Arc::new(ChartMacroField::new(
        STREAMING_RAM_CAPACITY,
        disk_cache,
        chart_metadata,
        chart_level,
        ELEVATION_SCALE_M,
    ));

    // Build the M5 production streaming queue with deterministic priority
    // classes, bounded concurrency, backoff, and health state.
    let streaming_config = StreamingConfig {
        max_queued: 1024,
        // The locked Flask sidecar is serial (`threaded=False`). Multiple
        // sockets only create head-of-line blocking and stale priorities.
        max_in_flight: 1,
        request_timeout: SIDECAR_REQUEST_TIMEOUT,
        base_backoff: Duration::from_millis(200),
        max_backoff: Duration::from_secs(30),
        healthy_threshold: 4,
        unhealthy_threshold: 8,
        unhealthy_cooldown: Duration::from_secs(5),
        jitter_seed: seed.0,
        max_resident: STREAMING_RAM_CAPACITY,
    };
    let streaming_queue = Arc::new(StreamingQueue::new(streaming_config));

    // H2: Register the streaming queue's evict_resident as the RAM cache
    // eviction callback so the streaming queue's resident set stays in sync
    // with the SurfaceCache RAM LRU. When the RAM cache evicts a key, the
    // streaming queue also evicts it, allowing a future camera re-request
    // to re-enqueue.
    {
        let sq = Arc::clone(&streaming_queue);
        chart_field
            .cache()
            .ram
            .set_eviction_callback(Arc::new(move |key| {
                sq.evict_resident(key);
            }));
    }

    // B2/Fix3: Register the streaming queue's record_cache_lookup as the
    // chart field's cache-lookup recorder so real hit/miss telemetry flows
    // from the actual mesh-sampling path (ChartMacroField::sample_resident)
    // into the streaming telemetry without blocking workers.
    {
        let sq = Arc::clone(&streaming_queue);
        chart_field.set_cache_lookup_recorder(Arc::new(move |hit| {
            sq.record_cache_lookup(hit);
        }));
    }

    let endpoint = SocketAddr::from(([127, 0, 0, 1], port));
    info!(
        endpoint = %endpoint,
        api_scale = final_api_scale,
        native_resolution = metadata.native_resolution,
        native_pixel_scale_m = metadata.native_pixel_scale_m,
        tiles_per_face_edge,
        chart_level,
        ram_capacity = STREAMING_RAM_CAPACITY,
        disk_capacity = STREAMING_DISK_CAPACITY,
        "Terrain Diffusion streaming mode enabled; chart-backed macro field + priority queue are the live runtime sources"
    );
    Some(TerrainDiffusionStartup {
        cache,
        chart_field,
        streaming_queue,
        config: TerrainDiffusionConfig {
            endpoint,
            seed,
            api_scale: final_api_scale,
            metadata,
        },
    })
}

/// System set for the Terrain Diffusion plugin's Update systems. Used to
/// order consumers (like the learned stress report) after diagnostic
/// publication.
#[derive(SystemSet, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TerrainDiffusionUpdate;

pub struct TerrainDiffusionPlugin {
    cache: Arc<LearnedTileCache>,
    chart_field: Arc<ChartMacroField>,
    streaming_queue: Arc<StreamingQueue>,
    config: TerrainDiffusionConfig,
}

impl TerrainDiffusionPlugin {
    pub fn new(
        cache: Arc<LearnedTileCache>,
        chart_field: Arc<ChartMacroField>,
        streaming_queue: Arc<StreamingQueue>,
        config: TerrainDiffusionConfig,
    ) -> Self {
        Self {
            cache,
            chart_field,
            streaming_queue,
            config,
        }
    }
}

impl Plugin for TerrainDiffusionPlugin {
    fn build(&self, app: &mut App) {
        let diagnostic = TerrainDiffusionDiagnostic {
            metadata: self.config.metadata.clone(),
            fallback_active: true,
            ..Default::default()
        };
        info!(
            native_resolution = diagnostic.metadata.native_resolution,
            stored_resolution = diagnostic.metadata.stored_resolution(),
            native_pixel_scale_m = diagnostic.metadata.native_pixel_scale_m,
            effective_pixel_scale_m = diagnostic.metadata.effective_pixel_scale_m(),
            api_scale = diagnostic.metadata.api_scale,
            halo_samples = diagnostic.metadata.halo_samples,
            tiles_per_face_edge = diagnostic.metadata.tiles_per_face_edge,
            "Terrain Diffusion streaming metadata initialized"
        );
        app.insert_resource(diagnostic)
            .insert_resource(TerrainDiffusionRuntime {
                cache: Arc::clone(&self.cache),
                chart_field: Arc::clone(&self.chart_field),
                streaming_queue: Arc::clone(&self.streaming_queue),
                config: self.config.clone(),
                in_flight: Vec::new(),
                failures: 0,
                invalid_tiles: 0,
                last_latency: None,
            })
            .insert_resource(ProviderTimingHistory::default())
            .configure_sets(
                Update,
                TerrainDiffusionUpdate.after(er_terrain::TerrainUpdate),
            )
            .add_systems(
                Update,
                (
                    queue_camera_tiles,
                    dispatch_streaming_requests,
                    poll_tile_requests,
                    publish_diagnostic,
                )
                    .chain()
                    .in_set(TerrainDiffusionUpdate),
            );
    }
}

#[derive(Resource)]
pub struct TerrainDiffusionRuntime {
    pub cache: Arc<LearnedTileCache>,
    pub chart_field: Arc<ChartMacroField>,
    pub streaming_queue: Arc<StreamingQueue>,
    config: TerrainDiffusionConfig,
    /// Bounded concurrent in-flight tasks. The streaming queue controls how
    /// many are dispatched; this holds the actual async tasks with their
    /// dispatch time for timeout-based cleanup.
    in_flight: Vec<(SurfaceCacheKeyWrapper, Instant, Task<TileResponse>)>,
    failures: usize,
    invalid_tiles: usize,
    last_latency: Option<f64>,
}

/// Wrapper to make `SurfaceCacheKey` usable as a hashable in-flight token
/// without exposing the full key type in the runtime struct.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
struct SurfaceCacheKeyWrapper(er_world::surface_cache::SurfaceCacheKey);

struct TileResponse {
    key: er_world::surface_cache::SurfaceCacheKey,
    tile_result: Result<LearnedTerrainTile, TileRequestFailure>,
    latency_ms: f64,
    /// Whether this response came from disk promotion (true) or network
    /// fetch (false). Used for telemetry.
    from_disk: bool,
}

/// Decoded sidecar payload ready for the M4 chart cache record.
#[derive(Clone)]
struct ChartPayload {
    coordinate: TileCoordinate,
    /// Provider's actual tiles-per-face-edge. Used to convert the tile
    /// coordinate to a sphere direction. Must NOT be confused with the
    /// chart field's `charts_per_edge()` (which is 2^chart_level and may
    /// differ for non-power-of-two tile counts like Earth's 652).
    tiles_per_face_edge: u16,
    elevation_m: Vec<i16>,
    climate: Vec<f32>,
}

#[derive(Debug)]
struct TileRequestFailure {
    coordinate: TileCoordinate,
    message: String,
}

fn queue_camera_tiles(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    runtime: Res<TerrainDiffusionRuntime>,
    mut timing: ResMut<ProviderTimingHistory>,
) {
    let start = Instant::now();
    let Ok(camera_transform) = camera_query.single() else {
        timing.add_us(start.elapsed().as_micros() as u64);
        return;
    };
    let camera_position = camera_transform.translation().as_dvec3();
    if camera_position.length_squared() <= f64::EPSILON {
        timing.add_us(start.elapsed().as_micros() as u64);
        return;
    }
    let camera_dir = camera_position.normalize();
    let tiles_per_face_edge = runtime.config.metadata.tiles_per_face_edge as u32;
    let now = Instant::now();
    let camera_chunk = er_core::math::dir_to_cell(camera_dir, 8);

    // Helper: enqueue a tile for a direction, deriving BOTH the provider
    // coordinate (from tiles_per_face_edge) and the chart cache key (from
    // the chart field). This preserves both identities explicitly.
    let enqueue_tile =
        |dir: DVec3, priority: PriorityClass, origin: Option<er_core::math::CellKey>| {
            let provider_coord = ProviderTileCoordinate::from_direction(dir, tiles_per_face_edge);
            // Skip out-of-range coordinates (should not happen due to clamping,
            // but guard against floating-point edge cases).
            if !provider_coord.is_in_range(tiles_per_face_edge) {
                return;
            }
            let chart_key = runtime.chart_field.key_for_tile(
                provider_coord.face,
                provider_coord.x,
                provider_coord.y,
                tiles_per_face_edge,
            );
            runtime
                .streaming_queue
                .enqueue(chart_key, provider_coord, priority, origin, now);
        };

    // 1. Visible surface + normal halo + prefetch ring.
    // Derive tile directions from the camera direction by offsetting in
    // uv space using the actual tiles_per_face_edge.
    let (face, u, v) = dir_to_uv(camera_dir);
    let center_provider = ProviderTileCoordinate::from_direction(camera_dir, tiles_per_face_edge);
    let n = tiles_per_face_edge as f64;
    for y_offset in -CAMERA_PREFETCH_RADIUS..=CAMERA_PREFETCH_RADIUS {
        for x_offset in -CAMERA_PREFETCH_RADIUS..=CAMERA_PREFETCH_RADIUS {
            let pf_u = u + (center_provider.x as f64 + 0.5 - u * n + x_offset as f64) / n;
            let pf_v = v + (center_provider.y as f64 + 0.5 - v * n + y_offset as f64) / n;
            let dir = uv_to_dir(face, pf_u, pf_v);
            let priority = if x_offset == 0 && y_offset == 0 {
                PriorityClass::VisibleSurface
            } else if x_offset.abs() <= 1 && y_offset.abs() <= 1 {
                PriorityClass::NormalHalo
            } else {
                PriorityClass::PrefetchRing
            };
            enqueue_tile(dir, priority, Some(camera_chunk));
        }
    }

    // 2. Camera-forward corridor: tiles ahead of the camera direction.
    let forward = camera_transform.forward().as_dvec3();
    let forward_dir = (camera_dir + forward * 0.3).normalize();
    let (fwd_face, fwd_u, fwd_v) = dir_to_uv(forward_dir);
    let fwd_center = ProviderTileCoordinate::from_direction(forward_dir, tiles_per_face_edge);
    for y_off in -1i32..=1 {
        for x_off in -1i32..=1 {
            let pf_u = fwd_u + (fwd_center.x as f64 + 0.5 - fwd_u * n + x_off as f64) / n;
            let pf_v = fwd_v + (fwd_center.y as f64 + 0.5 - fwd_v * n + y_off as f64) / n;
            let dir = uv_to_dir(fwd_face, pf_u, pf_v);
            enqueue_tile(
                dir,
                PriorityClass::CameraForwardCorridor,
                Some(camera_chunk),
            );
        }
    }

    // 3. Far/root coverage: one tile per face root.
    for root_face in 0..6u8 {
        let dir = er_core::math::uv_to_dir(root_face, 0.5, 0.5);
        enqueue_tile(dir, PriorityClass::FarRootCoverage, None);
    }

    // 4. Warmup: background tiles around the camera's face.
    for y_off in -3i32..=3 {
        for x_off in -3i32..=3 {
            if x_off.abs() <= 2 && y_off.abs() <= 2 {
                continue;
            }
            let pf_u = u + (center_provider.x as f64 + 0.5 - u * n + x_off as f64) / n;
            let pf_v = v + (center_provider.y as f64 + 0.5 - v * n + y_off as f64) / n;
            let dir = uv_to_dir(face, pf_u, pf_v);
            enqueue_tile(dir, PriorityClass::Warmup, Some(camera_chunk));
        }
    }

    timing.add_us(start.elapsed().as_micros() as u64);
}

/// Dispatch dispatchable requests from the streaming queue onto the async
/// compute pool. Respects the queue's concurrency limit, health state,
/// backoff, and deadlines.
fn dispatch_streaming_requests(
    mut runtime: ResMut<TerrainDiffusionRuntime>,
    mut timing: ResMut<ProviderTimingHistory>,
) {
    let start = Instant::now();
    let now = Instant::now();
    let config = runtime.config.clone();
    let generation = runtime.cache.generation().clone();
    let tiles_per_face_edge = runtime.config.metadata.tiles_per_face_edge;
    let chart_field = Arc::clone(&runtime.chart_field);

    while let Some(request) = runtime.streaming_queue.pop_dispatchable(now) {
        let key = request.key.clone();
        let provider_coord = request.provider_coordinate;
        let config = config.clone();
        let generation = generation.clone();
        let chart_field = Arc::clone(&chart_field);
        let streaming_queue = Arc::clone(&runtime.streaming_queue);
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let req_start = std::time::Instant::now();

            // Fix 1: LIVE background disk promotion before network fetch.
            // Attempt to load the chart key from disk off-thread. If found
            // and checksum-verified, promote to RAM, mark the queue resident,
            // and skip the HTTP fetch entirely. This avoids redundant sidecar
            // requests on cold starts and after RAM eviction.
            let disk_key = chart_field.key_for_tile(
                provider_coord.face,
                provider_coord.x,
                provider_coord.y,
                tiles_per_face_edge as u32,
            );
            match chart_field.cache().load_from_disk(&disk_key) {
                Ok(true) => {
                    // Disk hit: promote to RAM and mark resident. Bump
                    // revision so the terrain system sees the new data.
                    chart_field.bump_revision();
                    streaming_queue.mark_resident(&disk_key);
                    let latency_ms = req_start.elapsed().as_secs_f64() * 1000.0;
                    let coordinate = TileCoordinate {
                        face: provider_coord.face,
                        x: provider_coord.x as u16,
                        y: provider_coord.y as u16,
                    };
                    // Build a synthetic tile from the resident record so the
                    // legacy cache also has it (for diagnostic compatibility).
                    let tile = LearnedTerrainTile {
                        key: LearnedTileKey {
                            generation: generation.clone(),
                            coordinate,
                        },
                        core_resolution: TILE_CORE_RESOLUTION,
                        halo: TILE_HALO,
                        elevation_m: Arc::from([]),
                        climate: None,
                    };
                    return TileResponse {
                        key: key.clone(),
                        tile_result: Ok(tile),
                        latency_ms,
                        from_disk: true,
                    };
                }
                Ok(false) => {
                    // Disk miss: fall through to network fetch.
                }
                Err(error) => {
                    // Disk error: log and fall through to network fetch.
                    // Procedural fallback is preserved.
                    warn!(%error, ?disk_key, "Disk promotion failed; falling through to provider fetch");
                }
            }

            // Network fetch path.
            let coordinate = TileCoordinate {
                face: provider_coord.face,
                x: provider_coord.x as u16,
                y: provider_coord.y as u16,
            };
            let result = request_tile(config, generation, coordinate, tiles_per_face_edge);
            let latency_ms = req_start.elapsed().as_secs_f64() * 1000.0;
            match result {
                Ok((tile_result, chart_payload)) => {
                    // H1: Store the decoded payload into the M4 chart cache
                    // (disk write + checksum verification + capacity scan)
                    // OFF the main thread, here in the async task. This
                    // keeps all disk I/O off the Bevy Update path.
                    store_chart_payload_offthread(&chart_field, chart_payload);
                    TileResponse {
                        key: key.clone(),
                        tile_result: Ok(tile_result),
                        latency_ms,
                        from_disk: false,
                    }
                }
                Err(failure) => TileResponse {
                    key: key.clone(),
                    tile_result: Err(failure),
                    latency_ms,
                    from_disk: false,
                },
            }
        });
        runtime
            .in_flight
            .push((SurfaceCacheKeyWrapper(request.key), now, task));
        // Stop if we've hit the in-flight cap (the queue's pop_dispatchable
        // already enforces this, but the in_flight Vec grows separately).
        if runtime.in_flight.len() >= runtime.streaming_queue.config().max_in_flight {
            break;
        }
    }
    timing.add_us(start.elapsed().as_micros() as u64);
}

fn poll_tile_requests(
    mut runtime: ResMut<TerrainDiffusionRuntime>,
    mut diag: ResMut<TerrainDiffusionDiagnostic>,
    mut timing: ResMut<ProviderTimingHistory>,
) {
    let start = Instant::now();
    let now = Instant::now();
    let timeout = runtime.streaming_queue.config().request_timeout;

    // Fix 7: Sweep timed-out in-flight tasks. If a task has been in-flight
    // longer than the request timeout, cancel it and record a timeout. This
    // handles async task panics/drops where check_ready never returns Some.
    let mut timed_out: Vec<usize> = Vec::new();
    for (i, (_key_wrapper, dispatched_at, _task)) in runtime.in_flight.iter().enumerate() {
        if now.duration_since(*dispatched_at) > timeout {
            timed_out.push(i);
        }
    }
    for &i in timed_out.iter().rev() {
        let (key_wrapper, _, _) = runtime.in_flight.remove(i);
        let key = key_wrapper.0;
        runtime.streaming_queue.record_timeout(&key, now);
        runtime.failures += 1;
        warn!(
            ?key,
            "In-flight task timed out; procedural fallback remains active"
        );
    }

    // Phase 1: check which tasks are ready, collecting the responses.
    let mut ready: Vec<(usize, TileResponse)> = Vec::new();
    for (i, (_key_wrapper, _dispatched_at, task)) in runtime.in_flight.iter_mut().enumerate() {
        if let Some(response) = check_ready(task) {
            ready.push((i, response));
        }
    }

    if ready.is_empty() {
        diag.in_flight = !runtime.in_flight.is_empty();
        timing.add_us(start.elapsed().as_micros() as u64);
        return;
    }

    // Phase 2: process each ready response. We clone the Arcs out of runtime
    // first to avoid aliased borrows.
    let cache = Arc::clone(&runtime.cache);
    let streaming_queue = Arc::clone(&runtime.streaming_queue);

    let mut invalid_tiles = runtime.invalid_tiles;
    let mut failures = runtime.failures;
    let mut last_latency = runtime.last_latency;

    for (_i, response) in &ready {
        let key = response.key.clone();
        match &response.tile_result {
            Ok(tile) => {
                let coordinate = tile.key.coordinate;
                if response.from_disk {
                    // Disk-promoted tile: mark_resident was already called
                    // in the async task. Just record success for health/
                    // latency tracking (it will coalesce the resident key).
                    last_latency = Some(response.latency_ms);
                    streaming_queue.record_success(
                        &key,
                        Duration::from_secs_f64(response.latency_ms / 1000.0),
                        now,
                    );
                    info!(
                        ?coordinate,
                        chart_resident = streaming_queue.resident_count(),
                        latency_ms = response.latency_ms,
                        "Terrain Diffusion tile promoted from disk"
                    );
                } else {
                    // Network-fetched tile: insert into legacy cache and
                    // record success.
                    let tile_clone = LearnedTerrainTile {
                        key: tile.key.clone(),
                        core_resolution: tile.core_resolution,
                        halo: tile.halo,
                        elevation_m: Arc::clone(&tile.elevation_m),
                        climate: tile.climate.clone(),
                    };
                    if let Err(error) = cache.insert(tile_clone) {
                        invalid_tiles += 1;
                        streaming_queue.record_failure(&key, now);
                        warn!(%error, ?coordinate, "Discarded invalid Terrain Diffusion tile");
                    } else {
                        // The chart payload was already stored off-thread in
                        // the async task (H1). Here we just record success,
                        // which marks the key resident and transitions health.
                        // M1/M3: record_success is the single source of truth
                        // for "key became resident" — no separate mark_resident
                        // call needed.
                        last_latency = Some(response.latency_ms);
                        streaming_queue.record_success(
                            &key,
                            Duration::from_secs_f64(response.latency_ms / 1000.0),
                            now,
                        );
                        info!(
                            ?coordinate,
                            resident_tiles = cache.len(),
                            chart_resident = streaming_queue.resident_count(),
                            latency_ms = response.latency_ms,
                            "Terrain Diffusion tile ready"
                        );
                    }
                }
            }
            Err(error) => {
                failures += 1;
                streaming_queue.record_failure(&key, now);
                warn!(
                    ?error.coordinate,
                    error = %error.message,
                    "Terrain Diffusion request failed; procedural terrain remains active"
                );
            }
        }
    }

    runtime.invalid_tiles = invalid_tiles;
    runtime.failures = failures;
    runtime.last_latency = last_latency;

    // Phase 3: remove completed tasks (reverse order to preserve indices).
    // Dropping the Task cancels it if it was still running, which is the
    // desired behavior for stale/superseded requests.
    for &(i, _) in ready.iter().rev() {
        #[allow(unused_must_use)]
        let _ = runtime.in_flight.remove(i);
    }

    diag.in_flight = !runtime.in_flight.is_empty();
    timing.add_us(start.elapsed().as_micros() as u64);
}

/// Store a decoded sidecar payload into the M4 chart cache OFF the main
/// thread. This performs the disk write, checksum verification, and capacity
/// scan on the async compute thread pool, keeping all disk I/O off the Bevy
/// Update path (H1).
///
/// The tile coordinate is converted to a sphere direction using the
/// provider's actual `tiles_per_face_edge` (NOT the chart field's
/// `charts_per_edge()`). This is critical: for non-power-of-two tile counts
/// (e.g. Earth's 652), `charts_per_edge` (2^level) differs from
/// `tiles_per_face_edge`, and using the wrong denominator compresses the uv,
/// causing tiles to map to wrong-face directions and wrong chart keys.
fn store_chart_payload_offthread(chart_field: &ChartMacroField, payload: ChartPayload) {
    let coordinate = payload.coordinate;
    let key = chart_field.key_for_tile(
        coordinate.face,
        coordinate.x as u32,
        coordinate.y as u32,
        payload.tiles_per_face_edge as u32,
    );
    let record = SurfaceTileRecord::from_payload(
        key,
        Arc::from(payload.elevation_m),
        Arc::from(payload.climate),
        CreationMetadata::now("terrain-diffusion-sidecar"),
    );
    match chart_field.cache().store(record) {
        Ok(()) => {
            chart_field.bump_revision();
        }
        Err(error) => {
            warn!(%error, ?coordinate, "Failed to store chart cache record off-thread; procedural fallback remains active");
        }
    }
}

fn request_tile(
    config: TerrainDiffusionConfig,
    generation: LearnedTileGeneration,
    coordinate: TileCoordinate,
    tiles_per_face_edge: u16,
) -> Result<(LearnedTerrainTile, ChartPayload), TileRequestFailure> {
    let fail = |message: String| TileRequestFailure {
        coordinate,
        message,
    };
    let stored_resolution = TILE_CORE_RESOLUTION + TILE_HALO * 2;
    let face_span = tiles_per_face_edge as i32 * TILE_CORE_RESOLUTION as i32;
    let i1 = coordinate.face as i32 * face_span + coordinate.y as i32 * TILE_CORE_RESOLUTION as i32
        - TILE_HALO as i32;
    let j1 = coordinate.x as i32 * TILE_CORE_RESOLUTION as i32 - TILE_HALO as i32;
    let i2 = i1 + stored_resolution as i32;
    let j2 = j1 + stored_resolution as i32;
    let path = format!(
        "/terrain?i1={i1}&j1={j1}&i2={i2}&j2={j2}&scale={}&seed={}",
        config.api_scale, config.seed.0
    );
    let response = get_loopback(config.endpoint, &path).map_err(fail)?;
    let (height, width, elevation_m, climate) =
        parse_terrain_response_with_climate(&response).map_err(fail)?;
    if height != stored_resolution || width != stored_resolution {
        return Err(fail(format!(
            "sidecar returned {height}x{width}; expected {stored_resolution}x{stored_resolution}"
        )));
    }

    Ok((
        LearnedTerrainTile {
            key: LearnedTileKey {
                generation,
                coordinate,
            },
            core_resolution: TILE_CORE_RESOLUTION,
            halo: TILE_HALO,
            elevation_m: Arc::from(elevation_m.clone()),
            climate: Some(Arc::from(climate.clone())),
        },
        ChartPayload {
            coordinate,
            tiles_per_face_edge,
            elevation_m,
            climate,
        },
    ))
}

fn get_loopback(endpoint: SocketAddr, path: &str) -> Result<Vec<u8>, String> {
    let mut stream = TcpStream::connect_timeout(&endpoint, SIDECAR_REQUEST_TIMEOUT)
        .map_err(|error| format!("could not connect to {endpoint}: {error}"))?;
    stream
        .set_read_timeout(Some(SIDECAR_REQUEST_TIMEOUT))
        .map_err(|error| format!("could not set read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(SIDECAR_REQUEST_TIMEOUT))
        .map_err(|error| format!("could not set write timeout: {error}"))?;
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {endpoint}\r\nConnection: close\r\nAccept: application/octet-stream\r\n\r\n"
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("could not write request: {error}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| format!("could not read response: {error}"))?;
    Ok(response)
}

#[allow(dead_code)]
fn parse_terrain_response(response: &[u8]) -> Result<(u16, u16, Vec<i16>), String> {
    let Some(headers_end) = response.windows(4).position(|window| window == b"\r\n\r\n") else {
        return Err("response is missing an HTTP header terminator".to_owned());
    };
    let headers = std::str::from_utf8(&response[..headers_end])
        .map_err(|error| format!("response headers are not UTF-8: {error}"))?;
    let mut lines = headers.lines();
    let status = lines.next().unwrap_or_default();
    if !status.contains(" 200 ") {
        return Err(format!("sidecar returned {status}"));
    }
    let mut height = None;
    let mut width = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("x-height") {
            height = value.trim().parse::<u16>().ok();
        } else if name.eq_ignore_ascii_case("x-width") {
            width = value.trim().parse::<u16>().ok();
        }
    }
    let height = height.ok_or_else(|| "response is missing X-Height".to_owned())?;
    let width = width.ok_or_else(|| "response is missing X-Width".to_owned())?;
    let samples = height as usize * width as usize;
    let elevation_len = samples * std::mem::size_of::<i16>();
    let climate_len = samples * 4 * std::mem::size_of::<f32>();
    let body = &response[headers_end + 4..];
    if body.len() != elevation_len + climate_len {
        return Err(format!(
            "response payload is {} bytes; expected {}",
            body.len(),
            elevation_len + climate_len
        ));
    }
    let elevation_m = body[..elevation_len]
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    for bytes in body[elevation_len..].chunks_exact(4) {
        let climate = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if !climate.is_finite() {
            return Err("response contains a non-finite climate value".to_owned());
        }
    }
    Ok((height, width, elevation_m))
}

/// Decode a Terrain Diffusion `/terrain` response, retaining the four climate
/// channels (interleaved per sample: `[s0c0, s0c1, s0c2, s0c3, s1c0, ...]`).
///
/// This is the M4 path that retains climate for the versioned cache record.
/// The elevation-only [`parse_terrain_response`] remains for backward
/// compatibility with existing diagnostic tests.
fn parse_terrain_response_with_climate(
    response: &[u8],
) -> Result<(u16, u16, Vec<i16>, Vec<f32>), String> {
    let Some(headers_end) = response.windows(4).position(|window| window == b"\r\n\r\n") else {
        return Err("response is missing an HTTP header terminator".to_owned());
    };
    let headers = std::str::from_utf8(&response[..headers_end])
        .map_err(|error| format!("response headers are not UTF-8: {error}"))?;
    let mut lines = headers.lines();
    let status = lines.next().unwrap_or_default();
    if !status.contains(" 200 ") {
        return Err(format!("sidecar returned {status}"));
    }
    let mut height = None;
    let mut width = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("x-height") {
            height = value.trim().parse::<u16>().ok();
        } else if name.eq_ignore_ascii_case("x-width") {
            width = value.trim().parse::<u16>().ok();
        }
    }
    let height = height.ok_or_else(|| "response is missing X-Height".to_owned())?;
    let width = width.ok_or_else(|| "response is missing X-Width".to_owned())?;
    let samples = height as usize * width as usize;
    let elevation_len = samples * std::mem::size_of::<i16>();
    let climate_len = samples * 4 * std::mem::size_of::<f32>();
    let body = &response[headers_end + 4..];
    if body.len() != elevation_len + climate_len {
        return Err(format!(
            "response payload is {} bytes; expected {}",
            body.len(),
            elevation_len + climate_len
        ));
    }
    let elevation_m: Vec<i16> = body[..elevation_len]
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    let mut climate = Vec::with_capacity(samples * 4);
    for bytes in body[elevation_len..].chunks_exact(4) {
        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if !value.is_finite() {
            return Err("response contains a non-finite climate value".to_owned());
        }
        climate.push(value);
    }
    Ok((height, width, elevation_m, climate))
}

fn publish_diagnostic(
    runtime: Res<TerrainDiffusionRuntime>,
    mut diag: ResMut<TerrainDiffusionDiagnostic>,
    mut timing: ResMut<ProviderTimingHistory>,
) {
    let start = Instant::now();
    let stream_tel = runtime.streaming_queue.telemetry();
    // M2: Use the streaming queue's resident count as the single source of
    // truth for tile_count and fallback_active, eliminating the divergence
    // between the legacy LearnedTileCache and the streaming queue.
    diag.tile_count = stream_tel.resident_tiles;
    diag.queue_depth = stream_tel.queue_depth;
    diag.request_failures = runtime.failures as u32;
    diag.invalid_tiles_discarded = runtime.invalid_tiles as u32;
    diag.last_latency_ms = runtime.last_latency;
    diag.fallback_active = stream_tel.resident_tiles == 0;
    diag.in_flight = stream_tel.pending_in_flight > 0;
    diag.streaming = StreamingDiagnostic {
        resident_tiles: stream_tel.resident_tiles,
        pending_in_flight: stream_tel.pending_in_flight,
        failed_total: stream_tel.failed_total,
        cache_hits: stream_tel.cache_hits,
        cache_misses: stream_tel.cache_misses,
        cache_hit_rate: stream_tel.cache_hit_rate(),
        fallback_percent: stream_tel.fallback_percent(),
        latency_p50_ms: stream_tel.latency_p50_ms,
        latency_p95_ms: stream_tel.latency_p95_ms,
        rebuilds_queued: stream_tel.rebuilds_queued,
        rebuilds_completed: stream_tel.rebuilds_completed,
        health: stream_tel.health.state.as_str().to_owned(),
        provider_attempts: stream_tel.provider_state.attempts_total,
        provider_timeouts: stream_tel.provider_state.timeouts_total,
        provider_cancellations: stream_tel.provider_state.cancellations_total,
        provider_coalesced: stream_tel.provider_state.coalesced_duplicates_total,
        provider_stale_discarded: stream_tel.provider_state.stale_discarded_total,
    };
    timing.add_us(start.elapsed().as_micros() as u64);
    timing.finalize_frame();
}

/// Adapter that implements `er_terrain::RebuildChunkSource` by delegating to
/// the `StreamingQueue`'s `pop_rebuild_chunk`. This wires the streaming
/// queue's targeted rebuild tracking into the terrain systems so only chunks
/// intersecting an arrived chart/tile are rebuilt, not all active chunks.
pub struct StreamingRebuildSource {
    queue: Arc<StreamingQueue>,
}

impl StreamingRebuildSource {
    pub fn new(queue: Arc<StreamingQueue>) -> Self {
        Self { queue }
    }
}

impl er_terrain::RebuildChunkSource for StreamingRebuildSource {
    fn pop_arrived_chart(&self) -> Option<er_world::streaming::ChartFootprint> {
        self.queue.pop_arrived_chart()
    }

    fn rebuilds_completed(&self) -> u64 {
        self.queue.telemetry().rebuilds_completed
    }

    fn rebuilds_queued(&self) -> u64 {
        self.queue.telemetry().rebuilds_queued
    }
}

/// Adapter that implements `er_terrain::SampleSourceRecorder` by delegating
/// to the `StreamingQueue`'s `record_sample_source` and `record_cache_lookup`.
/// This feeds real fallback/learned percentages from the mesh-generation path
/// into the streaming telemetry (B2).
pub struct StreamingSampleRecorder {
    queue: Arc<StreamingQueue>,
}

impl StreamingSampleRecorder {
    pub fn new(queue: Arc<StreamingQueue>) -> Self {
        Self { queue }
    }
}

impl er_terrain::SampleSourceRecorder for StreamingSampleRecorder {
    fn record_sample_source(&self, learned: bool) {
        self.queue.record_sample_source(learned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_per_edge_miniature_yields_4() {
        assert_eq!(compute_tiles_per_face_edge(36_000.0), 4);
    }

    #[test]
    fn tiles_per_edge_earth_yields_652() {
        assert_eq!(compute_tiles_per_face_edge(6_371_000.0), 652);
    }

    #[test]
    fn tiles_per_edge_floor_1_for_tiny_radius() {
        assert_eq!(compute_tiles_per_face_edge(0.0), 1);
        assert_eq!(compute_tiles_per_face_edge(-1.0), 1);
        assert_eq!(compute_tiles_per_face_edge(1.0), 1);
    }

    #[test]
    fn tiles_per_edge_monotonic() {
        let huge = compute_tiles_per_face_edge(2.0 * 6_371_000.0);
        let earth = compute_tiles_per_face_edge(6_371_000.0);
        assert!(huge > earth);
    }

    #[test]
    fn chart_level_exact_power_of_two() {
        // Miniature planet: tiles_per_face_edge = 4 = 2^2, so chart_level = 2.
        assert_eq!(compute_chart_level(4), 2);
        assert_eq!(compute_chart_level(1), 0);
        assert_eq!(compute_chart_level(2), 1);
        assert_eq!(compute_chart_level(8), 3);
        assert_eq!(compute_chart_level(16), 4);
    }

    #[test]
    fn chart_level_non_power_of_two_rounds_up() {
        // Earth: tiles_per_face_edge = 652. The next power of two >= 652 is
        // 1024 = 2^10, so chart_level = 10 and charts_per_edge = 1024 >= 652.
        assert_eq!(compute_chart_level(652), 10);
        assert_eq!(1u32 << compute_chart_level(652), 1024);
        // Every chart count must be >= the tile count.
        for n in [1u16, 3, 5, 7, 100, 511, 512, 513, 651, 652, 653, 1023, 1024] {
            let level = compute_chart_level(n);
            let charts = 1u32 << level;
            assert!(
                charts >= n as u32,
                "tiles_per_face_edge={n}: charts_per_edge={charts} < {n}"
            );
        }
    }

    #[test]
    fn chart_level_for_actual_planet_radii() {
        // Miniature (36 km radius) -> 4 tiles -> level 2.
        let miniature = compute_tiles_per_face_edge(36_000.0);
        assert_eq!(compute_chart_level(miniature), 2);
        // Earth (6371 km radius) -> 652 tiles -> level 10.
        let earth = compute_tiles_per_face_edge(6_371_000.0);
        assert_eq!(compute_chart_level(earth), 10);
    }

    #[test]
    fn prefetch_boundary_with_high_tile_count_does_not_panic() {
        let gen = LearnedTileGeneration {
            model_revision: "prefetch-fixture".to_owned(),
            seed: 0xC0FFEE,
            projection_revision: 2,
            pixel_scale_m: NATIVE_PIXEL_SCALE_M,
            sea_level_datum_m: 0,
        };
        let cache = LearnedTileCache::new(gen.clone(), 652, 512, ELEVATION_SCALE_M);
        for face in 0..6u8 {
            for corner in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)] {
                let dir = er_core::math::uv_to_dir(face, corner.0, corner.1);
                let key = cache.tile_key_for_direction(dir);
                assert!(
                    key.coordinate.x < 652,
                    "face={} uv=({},{}) -> x={}",
                    face,
                    corner.0,
                    corner.1,
                    key.coordinate.x
                );
                assert!(
                    key.coordinate.y < 652,
                    "face={} uv=({},{}) -> y={}",
                    face,
                    corner.0,
                    corner.1,
                    key.coordinate.y
                );
            }
        }
    }

    #[test]
    fn metadata_default_unupsampled() {
        let meta = TerrainDiffusionMetadata::default();
        assert!(!meta.is_upsampled());
        assert_eq!(meta.effective_pixel_scale_m(), NATIVE_PIXEL_SCALE_M);
        assert_eq!(meta.stored_resolution(), 514);
    }

    #[test]
    fn metadata_upsampled_flag() {
        let meta = TerrainDiffusionMetadata {
            api_scale: 4,
            ..TerrainDiffusionMetadata::default()
        };
        assert!(meta.is_upsampled());
        assert_eq!(meta.effective_pixel_scale_m(), 7);
    }

    #[test]
    fn metadata_scale_zero_divisor_guarded() {
        let meta = TerrainDiffusionMetadata {
            api_scale: 0,
            ..TerrainDiffusionMetadata::default()
        };
        assert!(!meta.is_upsampled());
        assert_eq!(meta.effective_pixel_scale_m(), NATIVE_PIXEL_SCALE_M);
    }

    #[test]
    fn default_diagnostic_starts_with_procedural_fallback() {
        let diagnostic = TerrainDiffusionDiagnostic::default();
        assert!(diagnostic.fallback_active);
        assert_eq!(diagnostic.tile_count, 0);
        assert_eq!(diagnostic.queue_depth, 0);
        assert_eq!(diagnostic.request_failures, 0);
        assert_eq!(diagnostic.invalid_tiles_discarded, 0);
        assert!(!diagnostic.in_flight);
        assert_eq!(diagnostic.last_latency_ms, None);
    }

    #[test]
    fn parses_strict_little_endian_terrain_payload() {
        let mut body = Vec::new();
        for elevation in [100_i16, -200, 300, -400] {
            body.extend_from_slice(&elevation.to_le_bytes());
        }
        for climate in [1.0_f32; 16] {
            body.extend_from_slice(&climate.to_le_bytes());
        }
        let response = "HTTP/1.1 200 OK\r\nX-Height: 2\r\nX-Width: 2\r\n\r\n"
            .to_string()
            .into_bytes()
            .into_iter()
            .chain(body)
            .collect::<Vec<_>>();

        let (height, width, elevation) = parse_terrain_response(&response).unwrap();
        assert_eq!((height, width), (2, 2));
        assert_eq!(elevation, vec![100, -200, 300, -400]);
    }

    #[test]
    fn rejects_payload_without_all_climate_channels() {
        let response = b"HTTP/1.1 200 OK\r\nX-Height: 1\r\nX-Width: 1\r\n\r\n\x00\x00";
        assert!(parse_terrain_response(response).is_err());
    }

    #[test]
    fn climate_retaining_parser_returns_four_channels() {
        let mut body = Vec::new();
        let elevations = [100_i16, -200, 300, -400];
        for e in &elevations {
            body.extend_from_slice(&e.to_le_bytes());
        }
        // 4 samples * 4 channels = 16 f32 values.
        let climate_values: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        for c in &climate_values {
            body.extend_from_slice(&c.to_le_bytes());
        }
        let response = "HTTP/1.1 200 OK\r\nX-Height: 2\r\nX-Width: 2\r\n\r\n"
            .to_string()
            .into_bytes()
            .into_iter()
            .chain(body)
            .collect::<Vec<_>>();

        let (height, width, elevation, climate) =
            parse_terrain_response_with_climate(&response).unwrap();
        assert_eq!((height, width), (2, 2));
        assert_eq!(elevation, vec![100, -200, 300, -400]);
        assert_eq!(climate.len(), 16);
        assert_eq!(climate, climate_values);
    }

    #[test]
    fn climate_retaining_parser_rejects_non_finite() {
        let mut body = Vec::new();
        body.extend_from_slice(&0_i16.to_le_bytes());
        // NaN in the first climate channel.
        body.extend_from_slice(&f32::NAN.to_le_bytes());
        body.extend_from_slice(&[0_u8; 12]); // remaining 3 channels
        let response = "HTTP/1.1 200 OK\r\nX-Height: 1\r\nX-Width: 1\r\n\r\n"
            .to_string()
            .into_bytes()
            .into_iter()
            .chain(body)
            .collect::<Vec<_>>();
        assert!(parse_terrain_response_with_climate(&response).is_err());
    }

    #[test]
    fn provider_timing_empty_returns_none() {
        let timing = ProviderTimingHistory::default();
        assert!(timing.is_empty());
        assert_eq!(timing.len(), 0);
        assert_eq!(timing.percentile_ms(50.0), None);
        assert_eq!(timing.percentile_ms(95.0), None);
        assert_eq!(timing.max_ms(), None);
        assert_eq!(timing.mean_ms(), None);
    }

    #[test]
    fn provider_timing_accumulates_within_frame() {
        let mut timing = ProviderTimingHistory::default();
        timing.add_us(100);
        timing.add_us(200);
        // Not finalized yet — no samples recorded.
        assert!(timing.is_empty());
        timing.finalize_frame();
        assert_eq!(timing.len(), 1);
        assert_eq!(timing.percentile_ms(50.0), Some(0.3));
        assert_eq!(timing.max_ms(), Some(0.3));
    }

    #[test]
    fn provider_timing_percentiles_correct() {
        let mut timing = ProviderTimingHistory::default();
        // 10 frames: 0.1ms through 1.0ms
        for i in 1..=10 {
            timing.add_us(i * 100);
            timing.finalize_frame();
        }
        assert_eq!(timing.len(), 10);
        let p50 = timing.percentile_ms(50.0).unwrap();
        let p95 = timing.percentile_ms(95.0).unwrap();
        let max = timing.max_ms().unwrap();
        // Values are 0.1..1.0ms. P50 index = round(0.5*9)=5 → 0.6ms (round half away from zero).
        assert!((p50 - 0.6).abs() < 0.01, "p50 was {p50}");
        // P95: index = round(0.95*9) = round(8.55) = 9 → 1.0ms
        assert!((p95 - 1.0).abs() < 0.01, "p95 was {p95}");
        assert!((max - 1.0).abs() < 0.001);
    }

    #[test]
    fn provider_timing_ring_buffer_evicts_oldest() {
        let mut timing = ProviderTimingHistory::default();
        // Fill beyond capacity to verify eviction.
        for i in 0..(PROVIDER_TIMING_HISTORY_CAP + 10) {
            timing.add_us((i % 1000) as u64 + 1);
            timing.finalize_frame();
        }
        assert_eq!(timing.len(), PROVIDER_TIMING_HISTORY_CAP);
    }

    #[test]
    fn provider_timing_saturating_add() {
        let mut timing = ProviderTimingHistory::default();
        timing.add_us(u64::MAX);
        timing.add_us(1);
        timing.finalize_frame();
        assert_eq!(timing.percentile_ms(50.0), Some(u64::MAX as f64 / 1000.0));
    }
}
