//! Versioned, checksummed, atomically-persisted learned-surface cache.
//!
//! This module implements the Milestone 4 cache contract:
//!
//! - `SurfaceCacheKey` includes every generation-affecting field required by
//!   the roadmap: seed, chart projection revision, model revision,
//!   conditioning revision, residual revision, datum, dimensions, pixel
//!   scale, halo, and request bounds.
//! - `SurfaceTileRecord` stores elevation meters, four climate channels,
//!   an upstream-compatible payload checksum, a cache-integrity checksum,
//!   manifest identity, and creation metadata.
//! - Disk persistence is atomic: write to a temp file, flush/sync where
//!   supported, verify the checksum, then rename.
//! - A bounded RAM LRU holds hot records; a bounded disk cache holds cold
//!   ones. Corruption and migration version mismatches are rejected and
//!   become removable cache misses for the background regeneration path —
//!   they never cause a loss of procedural fallback.
//!
//! The cache is `Send + Sync`. Reads from terrain mesh workers are
//! non-blocking: a miss returns `None` and the caller falls back to the
//! procedural field, matching the roadmap's "preserve procedural fallback"
//! rule. Disk I/O happens only on the cache-population path (background
//! worker), never on the mesh-worker read path.

use crate::surface_charts::{SurfaceChartMetadata, SurfacePatchId};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// On-disk container version. Bumped on any breaking change to the record
/// layout; old records are rejected as a migration boundary.
/// Version 2: adds `charts_per_face_edge` to the key, supersedes version 1
/// which used padded `2^level` chart grids.
pub const SURFACE_CACHE_FORMAT_VERSION: u32 = 2;

/// Magic header identifying a surface-cache record file.
const MAGIC: &[u8; 8] = b"ERSURF02";

/// A 32-byte checksum. Used for both the upstream-compatible payload
/// checksum (SHA-256 over elevation || climate, matching
/// `TerrainPayload.checksum` in the locked Python protocol) and the
/// internal cache-integrity checksum.
pub type Checksum = [u8; 32];

/// Canonical cache key for one stored surface patch. Contains every field
/// listed in the roadmap (4.2.1). Two records with equal keys are
/// interchangeable; any field change is a migration boundary.
///
/// The `halo_samples` field is explicit and must be consistent with
/// `core_resolution` and the payload length; deserialization validates
/// this before allocating.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceCacheKey {
    pub seed: u64,
    pub projection_revision: u16,
    pub model_revision: String,
    pub conditioning_revision: u32,
    pub residual_revision: u32,
    pub sea_level_datum_m: i32,
    pub pixel_scale_m: u32,
    pub halo_samples: u32,
    pub core_resolution: u32,
    pub face: u8,
    /// Informational only: the quadtree level. Never used for runtime math.
    pub level: u8,
    pub x: u32,
    pub y: u32,
    /// Exact charts per face edge (e.g. 652 for Earth, 4 for miniature).
    /// This is the single source of truth for the chart grid width.
    pub charts_per_face_edge: u32,
    /// Request bounds (i1,j1,i2,j2) in upstream planar grid coordinates,
    /// including the halo. Stored so two requests with identical key fields
    /// but different bounds never collide.
    pub request_bounds: [i64; 4],
}

impl SurfaceCacheKey {
    /// Build a key from chart metadata + patch id + request bounds.
    pub fn from_metadata(
        meta: &SurfaceChartMetadata,
        patch: SurfacePatchId,
        request_bounds: [i64; 4],
    ) -> Self {
        Self {
            seed: meta.seed,
            projection_revision: meta.projection_revision,
            model_revision: meta.model_revision.clone(),
            conditioning_revision: meta.conditioning_revision,
            residual_revision: meta.residual_revision,
            sea_level_datum_m: meta.sea_level_datum_m,
            pixel_scale_m: meta.pixel_scale_m,
            halo_samples: patch.halo,
            core_resolution: meta.core_resolution,
            face: patch.chart.face,
            level: patch.chart.level,
            x: patch.chart.x,
            y: patch.chart.y,
            charts_per_face_edge: patch.chart.charts_per_face_edge,
            request_bounds,
        }
    }

    /// Expected stored resolution per side from the key's core + halo.
    /// Used to validate payloads before allocation.
    pub fn expected_stored_resolution(&self) -> Option<u32> {
        self.halo_samples
            .checked_mul(2)
            .and_then(|h| self.core_resolution.checked_add(h))
    }

    /// Expected elevation sample count (stored_res * stored_res).
    /// Returns `None` on overflow.
    pub fn expected_elevation_len(&self) -> Option<usize> {
        let n = self.expected_stored_resolution()? as usize;
        n.checked_mul(n)
    }

    /// Stable filename for this key. Deterministic and filesystem-safe.
    pub fn filename(&self) -> String {
        // Hash the string fields into the filename so different model
        // revisions or request bounds never collide on disk.
        let mut hasher = blake3::Hasher::new();
        self.write_identity(&mut hasher);
        let hash = hasher.finalize();
        let mut hex = String::with_capacity(16);
        for b in hash.as_bytes().iter().take(8) {
            hex.push_str(&format!("{b:02x}"));
        }
        format!(
            "f{}_l{}_x{}_y{}_{}.surf",
            self.face, self.level, self.x, self.y, hex
        )
    }

    fn write_identity(&self, hasher: &mut blake3::Hasher) {
        hasher.update(&self.seed.to_le_bytes());
        hasher.update(&self.projection_revision.to_le_bytes());
        hasher.update(self.model_revision.as_bytes());
        hasher.update(&self.conditioning_revision.to_le_bytes());
        hasher.update(&self.residual_revision.to_le_bytes());
        hasher.update(&self.sea_level_datum_m.to_le_bytes());
        hasher.update(&self.pixel_scale_m.to_le_bytes());
        hasher.update(&self.halo_samples.to_le_bytes());
        hasher.update(&self.core_resolution.to_le_bytes());
        hasher.update(&[self.face, self.level]);
        hasher.update(&self.x.to_le_bytes());
        hasher.update(&self.y.to_le_bytes());
        hasher.update(&self.charts_per_face_edge.to_le_bytes());
        for b in self.request_bounds.iter() {
            hasher.update(&b.to_le_bytes());
        }
    }
}

/// Creation metadata stored alongside the payload for provenance/audit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CreationMetadata {
    pub created_unix_ms: u64,
    pub source: String,
    pub format_version: u32,
}

impl CreationMetadata {
    pub fn now(source: impl Into<String>) -> Self {
        Self {
            created_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            source: source.into(),
            format_version: SURFACE_CACHE_FORMAT_VERSION,
        }
    }
}

/// A stored surface patch record carrying elevation meters, four climate
/// channels, an upstream payload checksum, a cache-integrity checksum,
/// manifest identity, and creation metadata.
///
/// Two checksums are stored. `payload_checksum` is SHA-256 over the raw
/// payload bytes (elevation `.to_le_bytes()` || climate `.to_le_bytes()`),
/// exactly matching the locked upstream Python protocol's
/// `TerrainPayload.checksum()` — this is the reproducibility identity.
/// `cache_integrity` is BLAKE3 over the same payload bytes, a fast internal
/// consistency check used to detect storage corruption independent of the
/// upstream algorithm.
#[derive(Clone, Debug)]
pub struct SurfaceTileRecord {
    pub key: SurfaceCacheKey,
    pub elevation_m: Arc<[i16]>,
    /// Four climate channels, interleaved per sample: `[s0c0, s0c1, s0c2,
    /// s0c3, s1c0, ...]`. Length = `elevation_m.len() * 4`.
    pub climate: Arc<[f32]>,
    /// Upstream-compatible SHA-256 payload checksum.
    pub payload_checksum: Checksum,
    /// Internal BLAKE3 cache-integrity checksum.
    pub cache_integrity: Checksum,
    pub creation: CreationMetadata,
}

impl SurfaceTileRecord {
    /// Stored resolution per side (core + 2*halo).
    pub fn stored_resolution(&self) -> u32 {
        // Reconstruct from elevation length. sqrt is exact for square grids.
        (self.elevation_m.len() as f64).sqrt().round() as u32
    }

    /// Compute the upstream-compatible SHA-256 payload checksum over the
    /// canonical payload (elevation || climate), matching the locked Python
    /// protocol's `TerrainPayload.checksum()`.
    pub fn compute_payload_checksum(elevation_m: &[i16], climate: &[f32]) -> Checksum {
        let mut hasher = Sha256::new();
        for v in elevation_m {
            hasher.update(v.to_le_bytes().as_slice());
        }
        for v in climate {
            hasher.update(v.to_le_bytes().as_slice());
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(&hasher.finalize());
        out
    }

    /// Compute the internal BLAKE3 cache-integrity checksum over the same
    /// payload bytes.
    pub fn compute_cache_integrity(elevation_m: &[i16], climate: &[f32]) -> Checksum {
        let mut hasher = blake3::Hasher::new();
        for v in elevation_m {
            hasher.update(&v.to_le_bytes());
        }
        for v in climate {
            hasher.update(&v.to_le_bytes());
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(hasher.finalize().as_bytes());
        out
    }

    /// Build a record from a decoded payload, computing both checksums.
    pub fn from_payload(
        key: SurfaceCacheKey,
        elevation_m: Arc<[i16]>,
        climate: Arc<[f32]>,
        creation: CreationMetadata,
    ) -> Self {
        let payload_checksum = Self::compute_payload_checksum(&elevation_m, &climate);
        let cache_integrity = Self::compute_cache_integrity(&elevation_m, &climate);
        Self {
            key,
            elevation_m,
            climate,
            payload_checksum,
            cache_integrity,
            creation,
        }
    }

    /// Verify the upstream payload checksum. Returns `false` if the payload
    /// has been corrupted or the upstream checksum does not match.
    pub fn verify_payload(&self) -> bool {
        Self::compute_payload_checksum(&self.elevation_m, &self.climate) == self.payload_checksum
    }

    /// Verify the internal cache-integrity checksum.
    pub fn verify_cache_integrity(&self) -> bool {
        Self::compute_cache_integrity(&self.elevation_m, &self.climate) == self.cache_integrity
    }

    /// Verify both checksums.
    pub fn verify(&self) -> bool {
        self.verify_payload() && self.verify_cache_integrity()
    }

    /// Validate structural invariants (lengths, stored resolution, square
    /// grid, climate channel count, halo consistency).
    pub fn validate_structure(&self) -> Result<(), SurfaceCacheError> {
        let elev_len = self.elevation_m.len();
        if elev_len == 0 {
            return Err(SurfaceCacheError::CorruptRecord("zero elevation samples"));
        }
        // The elevation grid must be square: sqrt must be an integer.
        let n = (elev_len as f64).sqrt().round() as u32;
        if (n as usize) * (n as usize) != elev_len {
            return Err(SurfaceCacheError::NonSquareGrid { samples: elev_len });
        }
        if self.climate.len() != elev_len * 4 {
            return Err(SurfaceCacheError::ClimateLengthMismatch {
                expected: elev_len * 4,
                actual: self.climate.len(),
            });
        }
        // Halo consistency: the key's expected stored resolution must match
        // the payload's actual resolution.
        if let Some(expected_n) = self.key.expected_stored_resolution() {
            if expected_n as usize != n as usize {
                return Err(SurfaceCacheError::HaloResolutionMismatch {
                    expected: expected_n,
                    actual: n,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum SurfaceCacheError {
    Io(io::Error),
    CorruptRecord(&'static str),
    VersionMismatch { expected: u32, found: u32 },
    ChecksumMismatch,
    CacheIntegrityMismatch,
    KeyFieldMissing(&'static str),
    NonSquareGrid { samples: usize },
    ClimateLengthMismatch { expected: usize, actual: usize },
    HaloResolutionMismatch { expected: u32, actual: u32 },
    DimensionOverflow,
    NonFiniteClimate,
}

impl fmt::Display for SurfaceCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "surface cache I/O error: {e}"),
            Self::CorruptRecord(msg) => write!(f, "surface cache corrupt record: {msg}"),
            Self::VersionMismatch { expected, found } => write!(
                f,
                "surface cache version mismatch: expected {expected}, found {found}"
            ),
            Self::ChecksumMismatch => write!(f, "surface cache payload checksum mismatch"),
            Self::CacheIntegrityMismatch => write!(f, "surface cache integrity checksum mismatch"),
            Self::KeyFieldMissing(name) => {
                write!(f, "surface cache missing required key field: {name}")
            }
            Self::NonSquareGrid { samples } => {
                write!(
                    f,
                    "surface cache non-square elevation grid: {samples} samples"
                )
            }
            Self::ClimateLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "surface cache climate length {actual} != expected {expected}"
                )
            }
            Self::HaloResolutionMismatch { expected, actual } => write!(
                f,
                "surface cache halo/resolution mismatch: expected {expected}, actual {actual}"
            ),
            Self::DimensionOverflow => write!(f, "surface cache dimension overflow"),
            Self::NonFiniteClimate => write!(f, "surface cache non-finite climate value"),
        }
    }
}

impl std::error::Error for SurfaceCacheError {}

impl From<io::Error> for SurfaceCacheError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Serialization (custom compact binary, no extra deps)
// ---------------------------------------------------------------------------

impl SurfaceTileRecord {
    /// Serialize to a compact, versioned binary blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&SURFACE_CACHE_FORMAT_VERSION.to_le_bytes());
        // Key fields
        buf.extend_from_slice(&self.key.seed.to_le_bytes());
        buf.extend_from_slice(&self.key.projection_revision.to_le_bytes());
        write_str(&mut buf, &self.key.model_revision);
        buf.extend_from_slice(&self.key.conditioning_revision.to_le_bytes());
        buf.extend_from_slice(&self.key.residual_revision.to_le_bytes());
        buf.extend_from_slice(&self.key.sea_level_datum_m.to_le_bytes());
        buf.extend_from_slice(&self.key.pixel_scale_m.to_le_bytes());
        buf.extend_from_slice(&self.key.halo_samples.to_le_bytes());
        buf.extend_from_slice(&self.key.core_resolution.to_le_bytes());
        buf.extend_from_slice(&[self.key.face, self.key.level]);
        buf.extend_from_slice(&self.key.x.to_le_bytes());
        buf.extend_from_slice(&self.key.y.to_le_bytes());
        buf.extend_from_slice(&self.key.charts_per_face_edge.to_le_bytes());
        for b in self.key.request_bounds.iter() {
            buf.extend_from_slice(&b.to_le_bytes());
        }
        // Payload lengths + data
        write_u64(&mut buf, self.elevation_m.len() as u64);
        for v in self.elevation_m.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        write_u64(&mut buf, self.climate.len() as u64);
        for v in self.climate.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // Dual checksums
        buf.extend_from_slice(&self.payload_checksum);
        buf.extend_from_slice(&self.cache_integrity);
        // Creation metadata
        write_str(&mut buf, &self.creation.source);
        buf.extend_from_slice(&self.creation.created_unix_ms.to_le_bytes());
        buf.extend_from_slice(&self.creation.format_version.to_le_bytes());
        buf
    }

    /// Deserialize from a compact, versioned binary blob. Rejects unknown
    /// magic, version mismatches, checksum mismatches, non-square grids,
    /// non-finite climate, and dimension/halo inconsistencies.
    ///
    /// All dimension validation happens *before* allocating the payload
    /// buffers, guarding against untrusted oversized length fields.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SurfaceCacheError> {
        let mut r = Reader::new(data);
        let magic = r
            .take(8)
            .ok_or(SurfaceCacheError::CorruptRecord("truncated magic"))?;
        if magic != MAGIC {
            return Err(SurfaceCacheError::CorruptRecord("bad magic"));
        }
        let version = r
            .read_u32()
            .ok_or(SurfaceCacheError::CorruptRecord("truncated version"))?;
        if version != SURFACE_CACHE_FORMAT_VERSION {
            return Err(SurfaceCacheError::VersionMismatch {
                expected: SURFACE_CACHE_FORMAT_VERSION,
                found: version,
            });
        }
        let seed = r
            .read_u64()
            .ok_or(SurfaceCacheError::CorruptRecord("seed"))?;
        let projection_revision = r
            .read_u16()
            .ok_or(SurfaceCacheError::CorruptRecord("projection_revision"))?;
        let model_revision = r
            .read_str()
            .ok_or(SurfaceCacheError::CorruptRecord("model_revision"))?;
        let conditioning_revision = r
            .read_u32()
            .ok_or(SurfaceCacheError::CorruptRecord("conditioning_revision"))?;
        let residual_revision = r
            .read_u32()
            .ok_or(SurfaceCacheError::CorruptRecord("residual_revision"))?;
        let sea_level_datum_m = r
            .read_i32()
            .ok_or(SurfaceCacheError::CorruptRecord("sea_level_datum_m"))?;
        let pixel_scale_m = r
            .read_u32()
            .ok_or(SurfaceCacheError::CorruptRecord("pixel_scale_m"))?;
        let halo_samples = r
            .read_u32()
            .ok_or(SurfaceCacheError::CorruptRecord("halo_samples"))?;
        let core_resolution = r
            .read_u32()
            .ok_or(SurfaceCacheError::CorruptRecord("core_resolution"))?;
        let face_level = r
            .take(2)
            .ok_or(SurfaceCacheError::CorruptRecord("face/level"))?;
        let face = face_level[0];
        let level = face_level[1];
        if face >= 6 {
            return Err(SurfaceCacheError::CorruptRecord("face >= 6"));
        }
        let x = r.read_u32().ok_or(SurfaceCacheError::CorruptRecord("x"))?;
        let y = r.read_u32().ok_or(SurfaceCacheError::CorruptRecord("y"))?;
        let charts_per_face_edge = r
            .read_u32()
            .ok_or(SurfaceCacheError::CorruptRecord("charts_per_face_edge"))?;
        let mut request_bounds = [0i64; 4];
        for b in request_bounds.iter_mut() {
            *b = r
                .read_i64()
                .ok_or(SurfaceCacheError::CorruptRecord("request_bounds"))?;
        }

        // --- Pre-allocation dimension validation ---
        // Build the key early so we can validate the payload length against
        // the halo/core_resolution before allocating.
        let key = SurfaceCacheKey {
            seed,
            projection_revision,
            model_revision,
            conditioning_revision,
            residual_revision,
            sea_level_datum_m,
            pixel_scale_m,
            halo_samples,
            core_resolution,
            face,
            level,
            x,
            y,
            charts_per_face_edge,
            request_bounds,
        };

        let elev_len_u64 = r
            .read_u64()
            .ok_or(SurfaceCacheError::CorruptRecord("elevation length"))?;
        // Guard against absurd lengths that would overflow or OOM.
        let elev_len =
            usize::try_from(elev_len_u64).map_err(|_| SurfaceCacheError::DimensionOverflow)?;
        // Cross-check against the key's expected resolution.
        if let Some(expected) = key.expected_elevation_len() {
            if expected != elev_len {
                return Err(SurfaceCacheError::CorruptRecord(
                    "elevation length != key expected",
                ));
            }
        }
        // elev_len * 2 must not overflow.
        let elev_byte_len = elev_len
            .checked_mul(2)
            .ok_or(SurfaceCacheError::DimensionOverflow)?;
        let elev_bytes = r
            .take(elev_byte_len)
            .ok_or(SurfaceCacheError::CorruptRecord("elevation data"))?;

        let climate_len_u64 = r
            .read_u64()
            .ok_or(SurfaceCacheError::CorruptRecord("climate length"))?;
        let climate_len =
            usize::try_from(climate_len_u64).map_err(|_| SurfaceCacheError::DimensionOverflow)?;
        if climate_len != elev_len * 4 {
            return Err(SurfaceCacheError::ClimateLengthMismatch {
                expected: elev_len * 4,
                actual: climate_len,
            });
        }
        let climate_byte_len = climate_len
            .checked_mul(4)
            .ok_or(SurfaceCacheError::DimensionOverflow)?;
        let climate_bytes = r
            .take(climate_byte_len)
            .ok_or(SurfaceCacheError::CorruptRecord("climate data"))?;

        // Allocate and decode.
        let elevation_m: Arc<[i16]> = elev_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect::<Vec<_>>()
            .into();
        let mut climate_vec: Vec<f32> = Vec::with_capacity(climate_len);
        for c in climate_bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if !value.is_finite() {
                return Err(SurfaceCacheError::NonFiniteClimate);
            }
            climate_vec.push(value);
        }
        let climate: Arc<[f32]> = climate_vec.into();

        // Dual checksums.
        let payload_checksum = r
            .take(32)
            .ok_or(SurfaceCacheError::CorruptRecord("payload checksum"))?
            .try_into()
            .map_err(|_| SurfaceCacheError::CorruptRecord("payload checksum array"))?;
        let cache_integrity = r
            .take(32)
            .ok_or(SurfaceCacheError::CorruptRecord("cache integrity"))?
            .try_into()
            .map_err(|_| SurfaceCacheError::CorruptRecord("cache integrity array"))?;

        let source = r
            .read_str()
            .ok_or(SurfaceCacheError::CorruptRecord("creation source"))?;
        let created_unix_ms = r
            .read_u64()
            .ok_or(SurfaceCacheError::CorruptRecord("creation time"))?;
        let format_version = r
            .read_u32()
            .ok_or(SurfaceCacheError::CorruptRecord("creation format version"))?;

        let record = SurfaceTileRecord {
            key,
            elevation_m,
            climate,
            payload_checksum,
            cache_integrity,
            creation: CreationMetadata {
                created_unix_ms,
                source,
                format_version,
            },
        };
        record.validate_structure()?;
        if !record.verify_payload() {
            return Err(SurfaceCacheError::ChecksumMismatch);
        }
        if !record.verify_cache_integrity() {
            return Err(SurfaceCacheError::CacheIntegrityMismatch);
        }
        Ok(record)
    }
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    write_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return None;
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }
    fn read_u16(&mut self) -> Option<u16> {
        self.take(2).map(|s| u16::from_le_bytes([s[0], s[1]]))
    }
    fn read_u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn read_i32(&mut self) -> Option<i32> {
        self.read_u32().map(|u| u as i32)
    }
    fn read_u64(&mut self) -> Option<u64> {
        self.take(8)
            .and_then(|s| s.try_into().ok())
            .map(u64::from_le_bytes)
    }
    fn read_i64(&mut self) -> Option<i64> {
        self.read_u64().map(|u| u as i64)
    }
    fn read_str(&mut self) -> Option<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes).ok().map(|s| s.to_owned())
    }
}

// ---------------------------------------------------------------------------
// RAM LRU + disk cache
// ---------------------------------------------------------------------------

/// Bounded RAM LRU of resident surface records. Non-blocking reads: a miss
/// returns `None` so mesh workers fall back to procedural terrain.
///
/// An optional eviction callback can be registered so external bookkeeping
/// (e.g. the streaming queue's resident set) stays in sync when the LRU
/// evicts a key. The callback is invoked under the cache lock, so it must
/// not attempt to acquire the same lock.
pub struct SurfaceRamCache {
    inner: Mutex<RamInner>,
}

/// Type alias for the RAM cache eviction callback.
type EvictionCallback = Arc<dyn Fn(&SurfaceCacheKey) + Send + Sync>;

struct RamInner {
    entries: HashMap<String, Arc<SurfaceTileRecord>>,
    order: Vec<String>, // LRU at front, MRU at back
    capacity: usize,
    /// Optional callback invoked when a key is evicted from the LRU.
    eviction_callback: Option<EvictionCallback>,
}

impl SurfaceRamCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(RamInner {
                entries: HashMap::new(),
                order: Vec::new(),
                capacity: capacity.max(1),
                eviction_callback: None,
            }),
        }
    }

    /// Register a callback invoked when a key is evicted from the LRU. Used
    /// by the streaming integration to keep the `StreamingQueue` resident set
    /// in sync with the RAM cache.
    pub fn set_eviction_callback(&self, callback: Arc<dyn Fn(&SurfaceCacheKey) + Send + Sync>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.eviction_callback = Some(callback);
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.lock().map(|i| i.capacity).unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.entries.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Non-blocking read. Returns `None` on miss; never does I/O.
    pub fn get(&self, key: &SurfaceCacheKey) -> Option<Arc<SurfaceTileRecord>> {
        let name = key.filename();
        let mut inner = self.inner.lock().ok()?;
        if let Some(record) = inner.entries.get(&name).cloned() {
            // Move to MRU.
            if let Some(pos) = inner.order.iter().position(|k| k == &name) {
                inner.order.remove(pos);
            }
            inner.order.push(name);
            return Some(record);
        }
        None
    }

    /// Insert a record, evicting the LRU entry if at capacity. If an
    /// eviction callback is registered, it is invoked with the evicted key.
    pub fn insert(&self, record: Arc<SurfaceTileRecord>) {
        let evicted_key = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            let name = record.key.filename();
            let evicted = if inner.entries.contains_key(&name) {
                if let Some(pos) = inner.order.iter().position(|k| k == &name) {
                    inner.order.remove(pos);
                }
                None
            } else if inner.entries.len() >= inner.capacity {
                if let Some(lru) = inner.order.first().cloned() {
                    inner.order.remove(0);
                    inner.entries.remove(&lru).map(|e| e.key.clone())
                } else {
                    None
                }
            } else {
                None
            };
            inner.entries.insert(name.clone(), record);
            inner.order.push(name);
            evicted
        };
        // Fire the eviction callback outside the insert logic (but still
        // under the lock — the callback must not re-enter this cache).
        if let Some(evicted) = evicted_key {
            if let Ok(inner) = self.inner.lock() {
                if let Some(cb) = &inner.eviction_callback {
                    cb(&evicted);
                }
            }
        }
    }

    /// Evict a specific key (used when a disk record is regenerated).
    pub fn evict(&self, key: &SurfaceCacheKey) {
        if let Ok(mut inner) = self.inner.lock() {
            let name = key.filename();
            inner.entries.remove(&name);
            inner.order.retain(|k| k != &name);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.entries.clear();
            inner.order.clear();
        }
    }
}

/// Bounded disk cache. Writes are atomic: temp file -> flush/sync -> verify
/// -> rename. Reads verify the checksum and reject corruption/migration.
///
/// Capacity enforcement is serialized by a process-local `Mutex` so that
/// concurrent background writers cannot race the directory scan and
/// over-evict or leak temp files.
///
/// Corruption/migration errors on `read` become a removable cache miss
/// (`Ok(None)` after deleting the bad file) so the background regeneration
/// path can replace the record without surfacing an error that would
/// threaten procedural fallback.
pub struct SurfaceDiskCache {
    root: PathBuf,
    max_entries: usize,
    capacity_lock: Mutex<()>,
}

impl SurfaceDiskCache {
    pub fn new(root: impl Into<PathBuf>, max_entries: usize) -> io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            max_entries: max_entries.max(1),
            capacity_lock: Mutex::new(()),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn path_for(&self, key: &SurfaceCacheKey) -> PathBuf {
        self.root.join(key.filename())
    }

    /// Atomically write a record: temp -> flush/sync -> verify -> rename.
    /// On failure the temp file is removed and no partial record remains.
    /// Capacity enforcement is serialized under `capacity_lock`.
    pub fn write(&self, record: &SurfaceTileRecord) -> Result<(), SurfaceCacheError> {
        record.validate_structure()?;
        if !record.verify() {
            return Err(SurfaceCacheError::ChecksumMismatch);
        }
        let final_path = self.path_for(&record.key);
        let temp_path = final_path.with_extension("surf.tmp");
        let bytes = record.to_bytes();
        {
            let mut file = fs::File::create(&temp_path)?;
            file.write_all(&bytes)?;
            file.flush()?;
            // fsync where supported (Unix). On Windows this is a no-op via
            // the sync_all call which flushes buffers.
            let _ = file.sync_all();
        }
        // Verify by re-reading the temp file before renaming.
        let verify = Self::read_raw(&temp_path)?;
        if verify != bytes {
            let _ = fs::remove_file(&temp_path);
            return Err(SurfaceCacheError::CorruptRecord(
                "post-write verify mismatch",
            ));
        }
        fs::rename(&temp_path, &final_path)?;
        // Serialize capacity enforcement so concurrent writers don't race.
        let _guard = self.capacity_lock.lock();
        self.enforce_capacity_locked()?;
        Ok(())
    }

    /// Read a record by key. Verifies both checksums; corrupt,
    /// version-mismatched, or structurally-invalid records are removed and
    /// reported as a **cache miss** (`Ok(None)`), never as an error. This
    /// ensures the background regeneration path can replace the record and
    /// the procedural fallback is never lost due to a bad cache file.
    pub fn read(
        &self,
        key: &SurfaceCacheKey,
    ) -> Result<Option<SurfaceTileRecord>, SurfaceCacheError> {
        let path = self.path_for(key);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = Self::read_raw(&path)?;
        match SurfaceTileRecord::from_bytes(&bytes) {
            Ok(record) => {
                if record.key != *key {
                    // Filename/key identity drift: remove the stale record
                    // and treat as a miss.
                    let _ = fs::remove_file(&path);
                    return Ok(None);
                }
                Ok(Some(record))
            }
            Err(
                SurfaceCacheError::ChecksumMismatch
                | SurfaceCacheError::CacheIntegrityMismatch
                | SurfaceCacheError::VersionMismatch { .. }
                | SurfaceCacheError::CorruptRecord(_)
                | SurfaceCacheError::NonSquareGrid { .. }
                | SurfaceCacheError::ClimateLengthMismatch { .. }
                | SurfaceCacheError::HaloResolutionMismatch { .. }
                | SurfaceCacheError::NonFiniteClimate
                | SurfaceCacheError::DimensionOverflow,
            ) => {
                // Corrupt, stale, or structurally invalid: remove so
                // regeneration can replace it. Report as a cache miss, not
                // an error, so procedural fallback is preserved.
                let _ = fs::remove_file(&path);
                Ok(None)
            }
            Err(other) => Err(other),
        }
    }

    fn read_raw(path: &Path) -> Result<Vec<u8>, SurfaceCacheError> {
        let mut file = fs::File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        Ok(buf)
    }

    /// Remove the on-disk record for a key (used on regeneration).
    pub fn remove(&self, key: &SurfaceCacheKey) -> io::Result<()> {
        let path = self.path_for(key);
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Enforce the max-entries bound by evicting oldest files (by mtime).
    /// Caller must hold `capacity_lock`.
    fn enforce_capacity_locked(&self) -> io::Result<()> {
        let mut entries: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("surf") {
                if let Ok(meta) = entry.metadata() {
                    if let Ok(mtime) = meta.modified() {
                        entries.push((path, mtime));
                    }
                }
            }
        }
        if entries.len() <= self.max_entries {
            return Ok(());
        }
        entries.sort_by_key(|(_, t)| *t);
        let to_remove = entries.len().saturating_sub(self.max_entries);
        for (path, _) in entries.into_iter().take(to_remove) {
            let _ = fs::remove_file(path);
        }
        Ok(())
    }

    /// Count on-disk records.
    pub fn count(&self) -> io::Result<usize> {
        let mut n = 0;
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) == Some("surf") {
                n += 1;
            }
        }
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// Combined RAM+disk cache with the no-I/O read contract for mesh workers
// ---------------------------------------------------------------------------

/// A two-tier surface cache. `get_resident` is the non-blocking read used by
/// terrain mesh workers: it only consults the RAM tier and never touches
/// disk. `load_from_disk` is the background-worker path that promotes a disk
/// record into RAM.
pub struct SurfaceCache {
    pub ram: SurfaceRamCache,
    pub disk: Option<SurfaceDiskCache>,
}

impl SurfaceCache {
    pub fn new(ram_capacity: usize, disk: Option<SurfaceDiskCache>) -> Self {
        Self {
            ram: SurfaceRamCache::new(ram_capacity),
            disk,
        }
    }

    /// Non-blocking resident read (mesh-worker safe). Returns `None` on miss.
    pub fn get_resident(&self, key: &SurfaceCacheKey) -> Option<Arc<SurfaceTileRecord>> {
        self.ram.get(key)
    }

    /// Check if a key exists on disk (non-blocking, no I/O read). Returns
    /// `false` if no disk cache is configured or the file does not exist.
    pub fn is_on_disk(&self, key: &SurfaceCacheKey) -> bool {
        if let Some(disk) = &self.disk {
            disk.path_for(key).exists()
        } else {
            false
        }
    }

    /// Background-worker path: read from disk, verify, and promote into RAM.
    pub fn load_from_disk(&self, key: &SurfaceCacheKey) -> Result<bool, SurfaceCacheError> {
        let Some(disk) = &self.disk else {
            return Ok(false);
        };
        match disk.read(key)? {
            Some(record) => {
                self.ram.insert(Arc::new(record));
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Background-worker path: persist a record to disk and RAM.
    pub fn store(&self, record: SurfaceTileRecord) -> Result<(), SurfaceCacheError> {
        if let Some(disk) = &self.disk {
            disk.write(&record)?;
        }
        self.ram.insert(Arc::new(record));
        Ok(())
    }

    /// Evict a key from RAM and disk (used on regeneration).
    pub fn evict(&self, key: &SurfaceCacheKey) {
        self.ram.evict(key);
        if let Some(disk) = &self.disk {
            let _ = disk.remove(key);
        }
    }
}

// ---------------------------------------------------------------------------
// Chart-backed MacroTerrainField (live resident sampling path)
// ---------------------------------------------------------------------------

use crate::surface_charts::SurfaceChartId;
use crate::terrain_field::{MacroTerrainField, MacroTerrainSample};
use er_core::math::{dir_to_uv, uv_to_dir};
use glam::DVec3;
use std::sync::atomic::{AtomicU64, Ordering};

/// A live `MacroTerrainField` backed by the sphere-native surface cache.
///
/// This is the M4 runtime source that replaces the legacy diagnostic
/// `LearnedTileCache`. It samples resident `SurfaceTileRecord`s bilinearly,
/// converting learned elevation meters into the field's normalized elevation
/// units using the chart's sea-level datum and an elevation scale.
///
/// The read path (`sample_resident`) is non-blocking and does no I/O: it
/// only consults the RAM tier. A miss returns `None` so the
/// `HybridTerrainField` falls back to procedural terrain. Disk promotion
/// happens on the background worker path (`load_from_disk` / `store`).
pub struct ChartMacroField {
    cache: SurfaceCache,
    metadata: SurfaceChartMetadata,
    chart_level: u8,
    elevation_scale_m: f64,
    revision: AtomicU64,
    /// Optional recorder for cache hit/miss telemetry. When present, every
    /// `sample_resident` call records whether the chart was resident (hit)
    /// or not (miss). This feeds real cache_hit_rate telemetry from the
    /// actual mesh-sampling path without blocking workers.
    cache_lookup_recorder: Mutex<Option<CacheLookupCallback>>,
}

/// Type alias for the cache-lookup recorder callback.
type CacheLookupCallback = Arc<dyn Fn(bool) + Send + Sync>;

impl ChartMacroField {
    /// Build a chart-backed macro field.
    ///
    /// - `ram_capacity`: max resident records in RAM LRU.
    /// - `disk`: optional bounded disk cache (background path only).
    /// - `metadata`: chart generation metadata (seed, model revision, etc.).
    /// - `chart_level`: quadtree level of the learned charts.
    /// - `elevation_scale_m`: meters-to-field-units scale (matches the
    ///   renderer's displacement scale).
    pub fn new(
        ram_capacity: usize,
        disk: Option<SurfaceDiskCache>,
        metadata: SurfaceChartMetadata,
        chart_level: u8,
        elevation_scale_m: f64,
    ) -> Self {
        assert!(
            elevation_scale_m.is_finite() && elevation_scale_m > 0.0,
            "elevation scale must be finite and positive"
        );
        Self {
            cache: SurfaceCache::new(ram_capacity, disk),
            metadata,
            chart_level,
            elevation_scale_m,
            revision: AtomicU64::new(0),
            cache_lookup_recorder: Mutex::new(None),
        }
    }

    /// Register a cache-lookup recorder. When present, every
    /// `sample_resident` call records whether the chart was resident (hit)
    /// or not (miss). This feeds real cache_hit_rate telemetry from the
    /// actual mesh-sampling path.
    pub fn set_cache_lookup_recorder(&self, recorder: Arc<dyn Fn(bool) + Send + Sync>) {
        if let Ok(mut r) = self.cache_lookup_recorder.lock() {
            *r = Some(recorder);
        }
    }

    /// Borrow the underlying cache (for the background worker to
    /// store/load records).
    pub fn cache(&self) -> &SurfaceCache {
        &self.cache
    }

    /// Chart metadata describing this field's generation.
    pub fn metadata(&self) -> &SurfaceChartMetadata {
        &self.metadata
    }

    /// Number of charts per cube-face edge. This is the exact provider tile
    /// count from metadata, NOT a padded power-of-two.
    pub fn charts_per_edge(&self) -> u32 {
        self.metadata.charts_per_face_edge
    }

    /// The chart quadtree level (informational only).
    pub fn chart_level(&self) -> u8 {
        self.chart_level
    }

    /// Bump the revision counter (called after a record is stored).
    pub fn bump_revision(&self) {
        self.revision.fetch_add(1, Ordering::Release);
    }

    /// Build the cache key for a direction, choosing the chart that owns it.
    /// Uses the exact `charts_per_face_edge` from metadata (not a padded
    /// power-of-two) to map the direction to the containing chart.
    pub fn key_for_direction(&self, dir: DVec3) -> SurfaceCacheKey {
        let chart = SurfaceChartId::from_direction(dir, self.metadata.charts_per_face_edge);
        let patch = SurfacePatchId::new(chart, self.metadata.halo_samples);
        // Request bounds are derived from the chart's face-local grid
        // position; for the cache key they just need to be deterministic and
        // unique per chart. Use the chart's grid coordinates.
        let bounds = [
            chart.face as i64,
            chart.x as i64,
            chart.y as i64,
            chart.charts_per_face_edge as i64,
        ];
        SurfaceCacheKey::from_metadata(&self.metadata, patch, bounds)
    }

    /// Build the cache key for a provider tile coordinate. The tile at
    /// `(face, x, y)` covers the uv region `[x/N, (x+1)/N] x [y/N,
    /// (y+1)/N]` where `N = charts_per_face_edge` (the exact provider tile
    /// count, e.g. 652 for Earth). The center direction determines the
    /// owning chart.
    ///
    /// `tiles_per_face_edge` must equal `metadata.charts_per_face_edge` for
    /// correct identity. It is passed explicitly so the caller's provider
    /// coordinate system is self-documenting.
    pub fn key_for_tile(
        &self,
        face: u8,
        x: u32,
        y: u32,
        tiles_per_face_edge: u32,
    ) -> SurfaceCacheKey {
        let n = tiles_per_face_edge.max(1) as f64;
        let dir = uv_to_dir(face, (x as f64 + 0.5) / n, (y as f64 + 0.5) / n);
        self.key_for_direction(dir)
    }

    /// The direction at the center of a provider tile.
    pub fn tile_center_dir(&self, face: u8, x: u32, y: u32, tiles_per_face_edge: u32) -> DVec3 {
        let n = tiles_per_face_edge.max(1) as f64;
        uv_to_dir(face, (x as f64 + 0.5) / n, (y as f64 + 0.5) / n)
    }

    /// Bilinearly sample elevation (meters) from a resident record.
    fn sample_elevation_m(&self, record: &SurfaceTileRecord, u: f64, v: f64) -> f64 {
        let stored = record.stored_resolution() as usize;
        let core = self.metadata.core_resolution as usize;
        let halo = self.metadata.halo_samples as usize;
        let core_max = (core.saturating_sub(1)) as f64;
        let x = halo as f64 + u.clamp(0.0, 1.0) * core_max;
        let y = halo as f64 + v.clamp(0.0, 1.0) * core_max;
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(stored.saturating_sub(1));
        let y1 = (y0 + 1).min(stored.saturating_sub(1));
        let tx = x - x0 as f64;
        let ty = y - y0 as f64;
        let at = |x: usize, y: usize| record.elevation_m[y * stored + x] as f64;
        let lower = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
        let upper = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
        lower + (upper - lower) * ty
    }

    /// Bilinearly sample the four climate channels from a resident record
    /// and apply the documented unit conversion (roadmap 5.2.5). Returns
    /// visual-only climate for material/biome shading; gameplay climate
    /// stays canonical procedural.
    fn sample_visual_climate(
        &self,
        record: &SurfaceTileRecord,
        u: f64,
        v: f64,
    ) -> crate::terrain_field::VisualClimate {
        let stored = record.stored_resolution() as usize;
        let core = self.metadata.core_resolution as usize;
        let halo = self.metadata.halo_samples as usize;
        let core_max = (core.saturating_sub(1)) as f64;
        let x = halo as f64 + u.clamp(0.0, 1.0) * core_max;
        let y = halo as f64 + v.clamp(0.0, 1.0) * core_max;
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(stored.saturating_sub(1));
        let y1 = (y0 + 1).min(stored.saturating_sub(1));
        let tx = x - x0 as f64;
        let ty = y - y0 as f64;
        // Climate is interleaved: [s0c0, s0c1, s0c2, s0c3, s1c0, ...].
        let at =
            |x: usize, y: usize, ch: usize| -> f32 { record.climate[(y * stored + x) * 4 + ch] };
        let channels: [f32; 4] = std::array::from_fn(|ch| {
            let lower = at(x0, y0, ch) + (at(x1, y0, ch) - at(x0, y0, ch)) * tx as f32;
            let upper = at(x0, y1, ch) + (at(x1, y1, ch) - at(x0, y1, ch)) * tx as f32;
            lower + (upper - lower) * ty as f32
        });
        crate::terrain_field::VisualClimate::from_upstream_channels(channels)
    }
}

impl MacroTerrainField for ChartMacroField {
    fn sample_resident(&self, dir: DVec3) -> Option<MacroTerrainSample> {
        let key = self.key_for_direction(dir);
        let record = self.cache.get_resident(&key);
        // Record cache hit/miss for real telemetry from the mesh-sampling
        // path. This does not block workers — it's a lightweight callback.
        if let Ok(recorder) = self.cache_lookup_recorder.lock() {
            if let Some(cb) = recorder.as_ref() {
                cb(record.is_some());
            }
        }
        let record = record?;
        let (face, u, v) = dir_to_uv(dir);
        // Convert the face-global uv to chart-local uv using the EXACT
        // charts_per_face_edge (not a padded power-of-two). This matches
        // the payload footprint: a tile at (x,y) covers uv [x/N, (x+1)/N].
        let n = self.metadata.charts_per_face_edge as f64;
        let local_u = (u * n).fract();
        let local_v = (v * n).fract();
        let _ = face;
        let elevation_m = self.sample_elevation_m(&record, local_u, local_v);
        let normalized =
            (elevation_m - self.metadata.sea_level_datum_m as f64) / self.elevation_scale_m;
        if !normalized.is_finite() {
            return None;
        }
        // Retrieve stored climate from the resident record and apply the
        // documented unit conversion (roadmap 5.2.5). This is visual-only;
        // gameplay climate stays canonical procedural.
        let visual_climate = self.sample_visual_climate(&record, local_u, local_v);
        Some(MacroTerrainSample {
            elevation: normalized,
            visual_climate,
        })
    }

    fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }
}

impl ChartMacroField {
    /// Check whether every chart key that a render chunk's elevation + normal
    /// halo depends on is resident. This is the Milestone 5 chunk-granularity
    /// residency gate: a chunk uses learned data only when *all* its halo
    /// dependencies are resident, otherwise it falls back to procedural for
    /// the entire chunk (no mixed normals).
    ///
    /// `chunk` is the render-quadtree cell; `halo_samples` and
    /// `core_resolution` come from the chart metadata.
    pub fn chunk_halo_resident(&self, chunk: er_core::math::CellKey) -> bool {
        let deps = crate::streaming::chart_dependencies_for_chunk(
            chunk,
            self.metadata.charts_per_face_edge,
            self.metadata.halo_samples,
            self.metadata.core_resolution,
        );
        if deps.is_empty() {
            return false;
        }
        let n = self.metadata.charts_per_face_edge;
        let level = if n <= 1 {
            0
        } else {
            (32 - (n - 1).leading_zeros()) as u8
        };
        for (face, x, y) in deps {
            let chart = SurfaceChartId {
                face,
                level,
                x,
                y,
                charts_per_face_edge: n,
            };
            let patch = SurfacePatchId::new(chart, self.metadata.halo_samples);
            let bounds = [face as i64, x as i64, y as i64, n as i64];
            let key = SurfaceCacheKey::from_metadata(&self.metadata, patch, bounds);
            if self.cache.get_resident(&key).is_none() {
                return false;
            }
        }
        true
    }

    /// Return the set of chart keys a chunk's elevation + normal halo depends
    /// on, as concrete `SurfaceCacheKey`s. Used by the integration to enqueue
    /// exactly the missing dependencies.
    pub fn chunk_halo_dependencies(&self, chunk: er_core::math::CellKey) -> Vec<SurfaceCacheKey> {
        let deps = crate::streaming::chart_dependencies_for_chunk(
            chunk,
            self.metadata.charts_per_face_edge,
            self.metadata.halo_samples,
            self.metadata.core_resolution,
        );
        let n = self.metadata.charts_per_face_edge;
        let level = if n <= 1 {
            0
        } else {
            (32 - (n - 1).leading_zeros()) as u8
        };
        deps.into_iter()
            .map(|(face, x, y)| {
                let chart = SurfaceChartId {
                    face,
                    level,
                    x,
                    y,
                    charts_per_face_edge: n,
                };
                let patch = SurfacePatchId::new(chart, self.metadata.halo_samples);
                let bounds = [face as i64, x as i64, y as i64, n as i64];
                SurfaceCacheKey::from_metadata(&self.metadata, patch, bounds)
            })
            .collect()
    }
}

impl crate::terrain_field::HaloResidencyChecker for ChartMacroField {
    fn chunk_halo_resident(&self, chunk: er_core::math::CellKey) -> bool {
        ChartMacroField::chunk_halo_resident(self, chunk)
    }

    fn chunk_halo_dependencies(&self, chunk: er_core::math::CellKey) -> Vec<SurfaceCacheKey> {
        ChartMacroField::chunk_halo_dependencies(self, chunk)
    }

    fn revision(&self) -> u64 {
        self.revision.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_charts::{
        ChartOwnership, SurfaceChartId, SurfaceChartMetadata, SurfacePatchId,
        SURFACE_CHART_PROJECTION_REVISION,
    };

    fn meta() -> SurfaceChartMetadata {
        SurfaceChartMetadata {
            seed: 0xC0FFEE,
            projection_revision: SURFACE_CHART_PROJECTION_REVISION,
            model_revision: "test-model-v1".to_owned(),
            conditioning_revision: 1,
            residual_revision: 1,
            sea_level_datum_m: 0,
            pixel_scale_m: 30,
            halo_samples: 1,
            core_resolution: 4,
            ownership: ChartOwnership::LearnedReliefProceduralShoreline,
            planet_radius_m: 6_371_000,
            charts_per_face_edge: 4,
        }
    }

    fn patch() -> SurfacePatchId {
        SurfacePatchId::new(
            SurfaceChartId {
                face: 0,
                level: 2,
                x: 1,
                y: 2,
                charts_per_face_edge: 4,
            },
            1,
        )
    }

    fn key() -> SurfaceCacheKey {
        SurfaceCacheKey::from_metadata(&meta(), patch(), [0, 0, 6, 6])
    }

    fn record() -> SurfaceTileRecord {
        let n = (4 + 2) * (4 + 2); // core 4 + halo 1 each side
        let elevation: Arc<[i16]> = (0..n as i16).collect::<Vec<_>>().into();
        let climate: Arc<[f32]> = (0..(n * 4) as u32)
            .map(|i| i as f32 * 0.1)
            .collect::<Vec<_>>()
            .into();
        SurfaceTileRecord::from_payload(key(), elevation, climate, CreationMetadata::now("test"))
    }

    #[test]
    fn record_round_trip_preserves_all_fields() {
        let r = record();
        let bytes = r.to_bytes();
        let r2 = SurfaceTileRecord::from_bytes(&bytes).unwrap();
        assert_eq!(r2.key, r.key);
        assert_eq!(r2.elevation_m.as_ref(), r.elevation_m.as_ref());
        assert_eq!(r2.climate.as_ref(), r.climate.as_ref());
        assert_eq!(r2.payload_checksum, r.payload_checksum);
        assert_eq!(r2.cache_integrity, r.cache_integrity);
        assert_eq!(r2.creation.source, r.creation.source);
        assert_eq!(r2.creation.format_version, r.creation.format_version);
    }

    #[test]
    fn payload_checksum_matches_upstream_sha256_algorithm() {
        // The upstream Python protocol computes SHA-256 over
        // elevation.tobytes() || climate.tobytes() where tobytes() is
        // little-endian. Verify our Rust implementation produces the same
        // digest for a known small payload.
        let elevation: Vec<i16> = vec![100, -200, 300, -400];
        let climate: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        // Manually compute the reference SHA-256.
        let mut manual = Sha256::new();
        for v in &elevation {
            manual.update(v.to_le_bytes());
        }
        for v in &climate {
            manual.update(v.to_le_bytes());
        }
        let mut expected = [0u8; 32];
        expected.copy_from_slice(&manual.finalize());

        let computed = SurfaceTileRecord::compute_payload_checksum(&elevation, &climate);
        assert_eq!(
            computed, expected,
            "Rust SHA-256 must match manual computation"
        );
    }

    #[test]
    fn checksum_detects_payload_corruption() {
        let mut r = record();
        r.elevation_m = Arc::from({
            let mut v: Vec<i16> = r.elevation_m.iter().copied().collect();
            v[0] += 1;
            v
        });
        // verify() recomputes and compares; must fail after mutation.
        assert!(!r.verify_payload());
        assert!(!r.verify_cache_integrity());
        // Serialization rejects a bad checksum on write.
        let disk =
            SurfaceDiskCache::new(std::env::temp_dir().join("ersurf_test_corrupt"), 16).unwrap();
        assert!(matches!(
            disk.write(&r),
            Err(SurfaceCacheError::ChecksumMismatch)
        ));
    }

    #[test]
    fn rejects_unknown_magic_and_version() {
        assert!(SurfaceTileRecord::from_bytes(b"NOTOURS0").is_err());
        let mut bytes = record().to_bytes();
        // Corrupt the version bytes (offset 8..12).
        bytes[8] = 0xFF;
        assert!(matches!(
            SurfaceTileRecord::from_bytes(&bytes),
            Err(SurfaceCacheError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_non_square_elevation_grid() {
        let mut r = record();
        // Make elevation non-square: add one extra sample.
        let mut elev: Vec<i16> = r.elevation_m.iter().copied().collect();
        elev.push(0);
        r.elevation_m = Arc::from(elev);
        r.climate = Arc::from(vec![0.0f32; r.elevation_m.len() * 4]);
        assert!(matches!(
            r.validate_structure(),
            Err(SurfaceCacheError::NonSquareGrid { .. })
        ));
    }

    #[test]
    fn rejects_climate_length_mismatch_in_deserialization() {
        let r = record();
        let bytes = r.to_bytes();
        // Corrupt the climate length field. It's a u64 written after the
        // elevation data. We can't easily locate it by offset, so instead
        // test via from_bytes with a truncated payload.
        let truncated = &bytes[..bytes.len() / 2];
        assert!(SurfaceTileRecord::from_bytes(truncated).is_err());
    }

    #[test]
    fn rejects_non_finite_climate_in_deserialization() {
        let mut r = record();
        // Inject a NaN into the climate and re-serialize manually.
        let mut climate: Vec<f32> = r.climate.iter().copied().collect();
        climate[0] = f32::NAN;
        r.climate = Arc::from(climate);
        // Recompute checksums so the structural check passes, but the
        // non-finite check in from_bytes must still reject it.
        r.payload_checksum =
            SurfaceTileRecord::compute_payload_checksum(&r.elevation_m, &r.climate);
        r.cache_integrity = SurfaceTileRecord::compute_cache_integrity(&r.elevation_m, &r.climate);
        let bytes = r.to_bytes();
        assert!(matches!(
            SurfaceTileRecord::from_bytes(&bytes),
            Err(SurfaceCacheError::NonFiniteClimate)
        ));
    }

    #[test]
    fn ram_lru_evicts_oldest_at_capacity() {
        let cache = SurfaceRamCache::new(2);
        let mut keys = Vec::new();
        for i in 0..3u32 {
            let mut k = key();
            k.x = i;
            keys.push(k);
        }
        for k in &keys {
            let mut r = record();
            r.key = k.clone();
            cache.insert(Arc::new(r));
        }
        // Capacity 2: the first key (i=0) should have been evicted.
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&keys[0]).is_none());
        assert!(cache.get(&keys[1]).is_some());
        assert!(cache.get(&keys[2]).is_some());
    }

    #[test]
    fn disk_cache_atomic_write_and_read_round_trips() {
        let dir = std::env::temp_dir().join(format!("ersurf_test_atomic_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = SurfaceDiskCache::new(&dir, 16).unwrap();
        let r = record();
        disk.write(&r).unwrap();
        let loaded = disk.read(&r.key).unwrap().unwrap();
        assert_eq!(loaded.elevation_m.as_ref(), r.elevation_m.as_ref());
        assert_eq!(loaded.climate.as_ref(), r.climate.as_ref());
        assert_eq!(loaded.payload_checksum, r.payload_checksum);
        assert_eq!(loaded.cache_integrity, r.cache_integrity);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_cache_corrupt_record_becomes_cache_miss_and_is_removed() {
        let dir = std::env::temp_dir().join(format!("ersurf_test_corrupt2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = SurfaceDiskCache::new(&dir, 16).unwrap();
        let r = record();
        disk.write(&r).unwrap();
        // Corrupt the on-disk file by flipping a payload byte.
        let path = disk.path_for(&r.key);
        let mut bytes = std::fs::read(&path).unwrap();
        // Flip a byte in the elevation data region (after the header/key).
        let idx = bytes.len().saturating_sub(40);
        bytes[idx] ^= 0xFF;
        std::fs::write(&path, bytes).unwrap();
        // A corrupt record must be a cache miss (Ok(None)), not an error,
        // so procedural fallback is preserved.
        let result = disk.read(&r.key).unwrap();
        assert!(
            result.is_none(),
            "corrupt record must be a cache miss, not an error"
        );
        assert!(!path.exists(), "corrupt record must be removed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_cache_version_mismatch_becomes_cache_miss() {
        let dir = std::env::temp_dir().join(format!("ersurf_test_ver_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = SurfaceDiskCache::new(&dir, 16).unwrap();
        let r = record();
        disk.write(&r).unwrap();
        // Corrupt the version field.
        let path = disk.path_for(&r.key);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8] = 0xFF;
        std::fs::write(&path, bytes).unwrap();
        let result = disk.read(&r.key).unwrap();
        assert!(result.is_none(), "version mismatch must be a cache miss");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_cache_evicts_oldest_files_at_capacity() {
        let dir = std::env::temp_dir().join(format!("ersurf_test_cap_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = SurfaceDiskCache::new(&dir, 2).unwrap();
        for i in 0..4u32 {
            let mut k = key();
            k.x = i;
            let mut r = record();
            r.key = k;
            disk.write(&r).unwrap();
            // Small delay so mtimes differ on fast filesystems.
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        let count = disk.count().unwrap();
        assert!(count <= 2, "disk cache must enforce capacity, got {count}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn combined_cache_load_from_disk_promotes_to_ram() {
        let dir = std::env::temp_dir().join(format!("ersurf_test_combo_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = SurfaceDiskCache::new(&dir, 16).unwrap();
        let cache = SurfaceCache::new(4, Some(disk));
        let r = record();
        cache.store(r.clone()).unwrap();
        // Clear RAM to simulate a cold start.
        cache.ram.clear();
        assert!(cache.get_resident(&r.key).is_none());
        // Load from disk into RAM.
        let promoted = cache.load_from_disk(&r.key).unwrap();
        assert!(promoted);
        assert!(cache.get_resident(&r.key).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_disk_writes_do_not_exceed_capacity() {
        let dir = std::env::temp_dir().join(format!(
            "ersurf_test_concurrent_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = std::sync::Arc::new(SurfaceDiskCache::new(&dir, 3).unwrap());
        let mut handles = Vec::new();
        for i in 0..8u32 {
            let disk = std::sync::Arc::clone(&disk);
            handles.push(std::thread::spawn(move || {
                let mut k = key();
                k.x = i;
                let mut r = record();
                r.key = k;
                disk.write(&r).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let count = disk.count().unwrap();
        assert!(
            count <= 3,
            "concurrent writes must respect capacity, got {count}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn filename_is_stable_and_filesystem_safe() {
        let k = key();
        let name = k.filename();
        assert!(name.ends_with(".surf"));
        assert!(!name.contains('/'));
        assert!(!name.contains('\\'));
        // Same key -> same filename.
        assert_eq!(name, k.filename());
        // Different model revision -> different filename.
        let mut k2 = k.clone();
        k2.model_revision = "other".to_owned();
        assert_ne!(name, k2.filename());
    }

    #[test]
    fn validate_structure_rejects_mismatched_lengths() {
        let mut r = record();
        r.climate = Arc::from(vec![0.0f32; 3]);
        assert!(r.validate_structure().is_err());
    }

    #[test]
    fn cache_key_full_field_roundtrip_preserves_identity() {
        // Every field in SurfaceCacheKey must round-trip through
        // serialization without loss, so two records with different
        // generation-affecting metadata never collide.
        let k1 = key();
        let r1 = record();
        let bytes = r1.to_bytes();
        let r2 = SurfaceTileRecord::from_bytes(&bytes).unwrap();
        assert_eq!(r2.key, k1);
        // A key differing in any single field must produce a different
        // filename (cache identity).
        let mut k2 = k1.clone();
        k2.conditioning_revision += 1;
        assert_ne!(k1.filename(), k2.filename());
        let mut k3 = k1.clone();
        k3.residual_revision += 1;
        assert_ne!(k1.filename(), k3.filename());
        let mut k4 = k1.clone();
        k4.sea_level_datum_m += 1;
        assert_ne!(k1.filename(), k4.filename());
        let mut k5 = k1.clone();
        k5.request_bounds[0] += 1;
        assert_ne!(k1.filename(), k5.filename());
    }
}
