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
use er_world::{
    LearnedTerrainTile, LearnedTileCache, LearnedTileGeneration, LearnedTileKey, TileCoordinate,
};
use std::collections::{HashSet, VecDeque};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::Duration;

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
const TILE_HALO: u16 = 1;
const API_SCALE: u8 = 1;
const ELEVATION_SCALE_M: f64 = 1000.0;
const CAMERA_PREFETCH_RADIUS: i32 = 1;
const SIDECAR_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

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
        }
    }
}

pub struct TerrainDiffusionStartup {
    pub cache: Arc<LearnedTileCache>,
    pub config: TerrainDiffusionConfig,
}

/// Reads `--terrain-diffusion` and optional `--terrain-diffusion-port <port>`.
/// Model streaming is deliberately disabled under screenshot and benchmark
/// modes so their output remains deterministic.
pub fn startup_from_args(disabled: bool, planet_radius_m: f64) -> Option<TerrainDiffusionStartup> {
    let args: Vec<String> = std::env::args().collect();
    if !args.iter().any(|arg| arg == "--terrain-diffusion") {
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
    let seed = PlanetSeed(0xC0FFEE);
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
    let generation = LearnedTileGeneration {
        model_revision: format!(
            "terrain-diffusion-native{NATIVE_PIXEL_SCALE_M}m-api-scale{}_{}",
            final_api_scale,
            if final_api_scale > 1 {
                "UPSAMPLED"
            } else {
                "NATIVE"
            }
        ),
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
    let endpoint = SocketAddr::from(([127, 0, 0, 1], port));
    info!(
        endpoint = %endpoint,
        api_scale = final_api_scale,
        native_resolution = metadata.native_resolution,
        native_pixel_scale_m = metadata.native_pixel_scale_m,
        tiles_per_face_edge,
        "Terrain Diffusion diagnostic mode enabled; use the loopback sidecar only for visual evaluation"
    );
    Some(TerrainDiffusionStartup {
        cache,
        config: TerrainDiffusionConfig {
            endpoint,
            seed,
            api_scale: final_api_scale,
            metadata,
        },
    })
}

pub struct TerrainDiffusionPlugin {
    cache: Arc<LearnedTileCache>,
    config: TerrainDiffusionConfig,
}

impl TerrainDiffusionPlugin {
    pub fn new(cache: Arc<LearnedTileCache>, config: TerrainDiffusionConfig) -> Self {
        Self { cache, config }
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
            native_pixel_scale_m = diagnostic.metadata.native_pixel_scale_m,
            api_scale = diagnostic.metadata.api_scale,
            halo_samples = diagnostic.metadata.halo_samples,
            tiles_per_face_edge = diagnostic.metadata.tiles_per_face_edge,
            "Terrain Diffusion diagnostic metadata initialized"
        );
        app.insert_resource(diagnostic)
            .insert_resource(TerrainDiffusionRuntime {
                cache: Arc::clone(&self.cache),
                config: self.config.clone(),
                queued: VecDeque::new(),
                requested: HashSet::new(),
                in_flight: None,
                failures: 0,
                invalid_tiles: 0,
                last_latency: None,
            })
            .add_systems(
                Update,
                (
                    queue_camera_tiles,
                    poll_tile_request,
                    start_tile_request,
                    publish_diagnostic,
                )
                    .chain(),
            );
    }
}

#[derive(Resource)]
struct TerrainDiffusionRuntime {
    cache: Arc<LearnedTileCache>,
    config: TerrainDiffusionConfig,
    queued: VecDeque<TileCoordinate>,
    requested: HashSet<TileCoordinate>,
    in_flight: Option<Task<TileResponse>>,
    failures: usize,
    invalid_tiles: usize,
    last_latency: Option<f64>,
}

struct TileResponse {
    tile_result: Result<LearnedTerrainTile, TileRequestFailure>,
    latency_ms: f64,
}

#[derive(Debug)]
struct TileRequestFailure {
    coordinate: TileCoordinate,
    message: String,
}

fn queue_camera_tiles(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut runtime: ResMut<TerrainDiffusionRuntime>,
) {
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };
    let camera_position = camera_transform.translation().as_dvec3();
    if camera_position.length_squared() <= f64::EPSILON {
        return;
    }
    let (face, u, v) = dir_to_uv(camera_position.normalize());
    let center = runtime
        .cache
        .tile_key_for_direction(camera_position.normalize());
    let tiles_per_edge = runtime.config.metadata.tiles_per_face_edge as f64;

    for y_offset in -CAMERA_PREFETCH_RADIUS..=CAMERA_PREFETCH_RADIUS {
        for x_offset in -CAMERA_PREFETCH_RADIUS..=CAMERA_PREFETCH_RADIUS {
            let prefetch_u = u
                + (center.coordinate.x as f64 + 0.5 - u * tiles_per_edge + x_offset as f64)
                    / tiles_per_edge;
            let prefetch_v = v
                + (center.coordinate.y as f64 + 0.5 - v * tiles_per_edge + y_offset as f64)
                    / tiles_per_edge;
            let coordinate = runtime
                .cache
                .tile_key_for_direction(uv_to_dir(face, prefetch_u, prefetch_v))
                .coordinate;
            if !runtime.cache.contains(coordinate) && runtime.requested.insert(coordinate) {
                runtime.queued.push_back(coordinate);
            }
        }
    }
}

fn poll_tile_request(
    mut runtime: ResMut<TerrainDiffusionRuntime>,
    mut diag: ResMut<TerrainDiffusionDiagnostic>,
) {
    let Some(task) = runtime.in_flight.as_mut() else {
        diag.in_flight = false;
        return;
    };
    let Some(response) = check_ready(task) else {
        diag.in_flight = true;
        return;
    };
    runtime.in_flight = None;
    runtime.last_latency = Some(response.latency_ms);
    diag.in_flight = false;

    match response.tile_result {
        Ok(tile) => {
            let coordinate = tile.key.coordinate;
            if let Err(error) = runtime.cache.insert(tile) {
                runtime.requested.remove(&coordinate);
                runtime.invalid_tiles += 1;
                warn!(%error, ?coordinate, "Discarded invalid Terrain Diffusion tile");
            } else {
                info!(
                    ?coordinate,
                    resident_tiles = runtime.cache.len(),
                    latency_ms = response.latency_ms,
                    "Terrain Diffusion tile ready"
                );
            }
        }
        Err(error) => {
            runtime.requested.remove(&error.coordinate);
            runtime.failures += 1;
            warn!(
                ?error.coordinate,
                error = %error.message,
                "Terrain Diffusion request failed; procedural terrain remains active"
            );
        }
    }
}

fn start_tile_request(mut runtime: ResMut<TerrainDiffusionRuntime>) {
    if runtime.in_flight.is_some() {
        return;
    }
    let Some(coordinate) = runtime.queued.pop_front() else {
        return;
    };
    let config = runtime.config.clone();
    let generation = runtime.cache.generation().clone();
    runtime.in_flight = Some(AsyncComputeTaskPool::get().spawn(async move {
        let start = std::time::Instant::now();
        let tile_result = request_tile(config, generation, coordinate);
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        TileResponse {
            tile_result,
            latency_ms,
        }
    }));
}

fn request_tile(
    config: TerrainDiffusionConfig,
    generation: LearnedTileGeneration,
    coordinate: TileCoordinate,
) -> Result<LearnedTerrainTile, TileRequestFailure> {
    let fail = |message: String| TileRequestFailure {
        coordinate,
        message,
    };
    let stored_resolution = TILE_CORE_RESOLUTION + TILE_HALO * 2;
    let face_span = config.metadata.tiles_per_face_edge as i32 * TILE_CORE_RESOLUTION as i32;
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
    let (height, width, elevation_m) = parse_terrain_response(&response).map_err(fail)?;
    if height != stored_resolution || width != stored_resolution {
        return Err(fail(format!(
            "sidecar returned {height}x{width}; expected {stored_resolution}x{stored_resolution}"
        )));
    }

    Ok(LearnedTerrainTile {
        key: LearnedTileKey {
            generation,
            coordinate,
        },
        core_resolution: TILE_CORE_RESOLUTION,
        halo: TILE_HALO,
        elevation_m: Arc::from(elevation_m),
    })
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

fn publish_diagnostic(
    runtime: Res<TerrainDiffusionRuntime>,
    mut diag: ResMut<TerrainDiffusionDiagnostic>,
) {
    diag.tile_count = runtime.cache.len();
    diag.queue_depth = runtime.queued.len();
    diag.request_failures = runtime.failures as u32;
    diag.invalid_tiles_discarded = runtime.invalid_tiles as u32;
    diag.last_latency_ms = runtime.last_latency;
    diag.fallback_active = diag.tile_count == 0;
    diag.in_flight = runtime.in_flight.is_some();
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
        let response = format!("HTTP/1.1 200 OK\r\nX-Height: 2\r\nX-Width: 2\r\n\r\n")
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
}
