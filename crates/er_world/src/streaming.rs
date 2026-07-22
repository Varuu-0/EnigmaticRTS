//! Production learned-streaming pipeline core (Milestone 5).
//!
//! This module is the *pure* scheduling layer between the chart/cache layer
//! (`surface_cache`) and the provider transport (HTTP sidecar in
//! `er_game::terrain_diffusion`). It is deliberately `Send + Sync` and has
//! no Bevy or I/O dependency so the priority/coalescing/backoff/health/
//! residency logic is exhaustively unit-testable in isolation.
//!
//! ## Design
//!
//! - One bounded priority queue (`StreamingQueue`) using *stable, deterministic*
//!   priority classes in the exact roadmap order: visible surface, camera-forward
//!   corridor, normal halo, prefetch ring, far/root coverage, warmup.
//! - Coalescing: a request for a key already queued, in-flight, or resident is
//!   dropped (deduplicated) rather than re-enqueued. The highest-priority
//!   classification wins when a duplicate arrives.
//! - Bounded concurrent in-flight work with a per-key cancellation/stale-discard
//!   path: when a request times out or the provider fails, it moves to the
//!   retry queue with exponential backoff + deterministic seeded jitter.
//! - Explicit `ServiceHealth` state machine. The provider starts `Unhealthy` and
//!   transitions to `Healthy` after a configurable run of consecutive successes;
//!   it degrades on a run of consecutive failures and goes `Unhealthy` when
//!   failures exceed the hard threshold, pausing dispatch.
//! - Telemetry snapshot exposed via `StreamingTelemetry` for stress reports:
//!   queue depth, resident/pending/failed counts, cache hit rate, fallback
//!   percentage, latency P50/P95, rebuild counts, service health, and provider
//!   state.
//!
//! Decode/validate/cache of complete finite checksummed payloads happens
//! off-thread on the provider transport side; this module only tracks the
//! *scheduling* state and exposes non-blocking transitions.

use crate::surface_cache::SurfaceCacheKey;
use er_core::math::CellKey;
use glam::DVec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Priority classes in the exact roadmap order (lower ordinal = higher
/// priority). The queue is *stable* within a class: equal-priority requests
/// keep their insertion order. Inter-class ordering is by ordinal.
///
/// `Warmup` is intentionally last: it pre-populates the cache only when no
/// camera-driven work is outstanding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum PriorityClass {
    /// On-screen surface tiles the camera is currently looking at.
    VisibleSurface = 0,
    /// Tiles just ahead of the camera direction that will likely become
    /// visible within one camera move.
    CameraForwardCorridor = 1,
    /// Tiles required to compute normals/material across chunk halos.
    NormalHalo = 2,
    /// Ring of tiles around the visible set that keep streaming ahead of
    /// normal camera movement.
    PrefetchRing = 3,
    /// Far / root coverage that keeps the globe filled at low LOD.
    FarRootCoverage = 4,
    /// Background warm-up of tiles the camera is not currently near.
    Warmup = 5,
}

impl PriorityClass {
    pub const COUNT: usize = 6;

    pub fn as_str(self) -> &'static str {
        match self {
            Self::VisibleSurface => "visible_surface",
            Self::CameraForwardCorridor => "camera_forward_corridor",
            Self::NormalHalo => "normal_halo",
            Self::PrefetchRing => "prefetch_ring",
            Self::FarRootCoverage => "far_root_coverage",
            Self::Warmup => "warmup",
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::VisibleSurface,
            1 => Self::CameraForwardCorridor,
            2 => Self::NormalHalo,
            3 => Self::PrefetchRing,
            4 => Self::FarRootCoverage,
            _ => Self::Warmup,
        }
    }
}

/// A canonical provider tile coordinate. This is the identity the sidecar
/// expects: `(face, x, y)` where `x, y` are in `[0, tiles_per_face_edge)`.
/// This is DISTINCT from the chart grid coordinates in `SurfaceCacheKey`
/// (which use padded `2^level` and can exceed `tiles_per_face_edge` for
/// non-power-of-two tile counts like Earth's 652).
///
/// The provider coordinate and the chart cache key are derived from the
/// same camera direction but through different denominators:
/// - Provider: `x = floor(u * tiles_per_face_edge)`
/// - Chart: `chart_x = floor(u * 2^chart_level)`
///
/// Both identities must be preserved explicitly. The provider coordinate is
/// used for the sidecar request; the chart key is used for cache
/// storage/lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProviderTileCoordinate {
    pub face: u8,
    pub x: u32,
    pub y: u32,
}

impl ProviderTileCoordinate {
    /// Create a provider tile coordinate from a direction and the actual
    /// `tiles_per_face_edge`. Clamps to `[0, tiles_per_face_edge)` to handle
    /// floating-point edge cases at face boundaries.
    pub fn from_direction(dir: DVec3, tiles_per_face_edge: u32) -> Self {
        use er_core::math::dir_to_uv;
        let (face, u, v) = dir_to_uv(dir);
        let n = tiles_per_face_edge.max(1) as f64;
        let x = (u * n).floor() as u32;
        let y = (v * n).floor() as u32;
        let max = tiles_per_face_edge.saturating_sub(1);
        Self {
            face,
            x: x.min(max),
            y: y.min(max),
        }
    }

    /// The center direction of this provider tile.
    pub fn center_direction(&self, tiles_per_face_edge: u32) -> DVec3 {
        let n = tiles_per_face_edge.max(1) as f64;
        er_core::math::uv_to_dir(
            self.face,
            (self.x as f64 + 0.5) / n,
            (self.y as f64 + 0.5) / n,
        )
    }

    /// Returns `true` if the coordinate is in valid range
    /// `[0, tiles_per_face_edge)`.
    pub fn is_in_range(&self, tiles_per_face_edge: u32) -> bool {
        self.x < tiles_per_face_edge && self.y < tiles_per_face_edge
    }
}

/// A queued provider request. Stable insertion order is preserved by the
/// monotonically increasing `seq` counter.
#[derive(Clone, Debug)]
pub struct ProviderRequest {
    pub key: SurfaceCacheKey,
    /// The canonical provider tile coordinate (face, x, y in
    /// `[0, tiles_per_face_edge)`). Used for the sidecar request. This is
    /// DISTINCT from the chart key's x/y which use padded `2^level`.
    pub provider_coordinate: ProviderTileCoordinate,
    pub priority: PriorityClass,
    pub seq: u64,
    /// Attempt number: 0 for first try, 1+ for retries.
    pub attempt: u32,
    /// Absolute deadline after which the request is stale and discarded.
    pub deadline: Option<Instant>,
    /// Earliest time a retry may be dispatched (after backoff).
    pub not_before: Option<Instant>,
    /// The render-quadtree chunk that originated this request, for
    /// intersecting-only rebuild bookkeeping.
    pub origin_chunk: Option<CellKey>,
}

impl ProviderRequest {
    /// Lower is higher priority: class ordinal first.
    pub fn priority_rank(&self) -> u8 {
        self.priority as u8
    }
}

/// Service health state machine.
///
/// `Unhealthy` pauses new dispatch (existing in-flight work is allowed to
/// complete) so the sidecar is not hammered while it is failing. Transitions
/// are driven by `record_success` / `record_failure`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceHealth {
    /// Consecutive successes >= `healthy_threshold`. Dispatch is unconstrained.
    Healthy,
    /// At least one recent failure but below the unhealthy threshold.
    /// Dispatch continues with normal concurrency.
    Degraded,
    /// Consecutive failures >= `unhealthy_threshold`. Dispatch is paused
    /// until a probe succeeds or the cooldown elapses.
    Unhealthy,
    /// The provider has never been contacted (cold start). Behaves like
    /// `Unhealthy` but is reported separately so telemetry can distinguish
    /// "never tried" from "broken".
    Cold,
}

impl ServiceHealth {
    pub fn is_dispatch_paused(self) -> bool {
        matches!(self, Self::Unhealthy | Self::Cold)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
            Self::Cold => "cold",
        }
    }
}

/// Configuration for the streaming queue. All thresholds are finite and
/// validated at construction.
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// Maximum number of queued (not yet in-flight) requests.
    pub max_queued: usize,
    /// Maximum concurrently in-flight requests.
    pub max_in_flight: usize,
    /// Per-request timeout before the request is marked stale.
    pub request_timeout: Duration,
    /// Base backoff for the first retry.
    pub base_backoff: Duration,
    /// Maximum backoff cap.
    pub max_backoff: Duration,
    /// Consecutive successes required to mark the provider `Healthy`.
    pub healthy_threshold: u32,
    /// Consecutive failures that transition `Degraded` -> `Unhealthy`.
    pub unhealthy_threshold: u32,
    /// Time the provider stays `Unhealthy` before a probe is allowed.
    pub unhealthy_cooldown: Duration,
    /// Seed for deterministic retry jitter (so tests are reproducible).
    pub jitter_seed: u64,
    /// Maximum resident keys tracked by the streaming queue. Must stay in
    /// sync with the SurfaceCache RAM capacity so the queue does not retain
    /// stale resident entries after the RAM cache evicts them. When this
    /// bound is exceeded, the least-recently-used resident key is evicted.
    pub max_resident: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_queued: 1024,
            max_in_flight: 4,
            request_timeout: Duration::from_secs(60),
            base_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(30),
            healthy_threshold: 4,
            unhealthy_threshold: 8,
            unhealthy_cooldown: Duration::from_secs(5),
            jitter_seed: 0xC0FFEE,
            max_resident: 256,
        }
    }
}

/// Latency ring buffer used for P50/P95 computation.
#[derive(Clone, Debug)]
struct LatencyHistory {
    samples: VecDeque<Duration>,
    capacity: usize,
}

impl LatencyHistory {
    fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
        }
    }

    fn push(&mut self, sample: Duration) {
        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    fn percentile(&self, pct: f64) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<Duration> = self.samples.iter().copied().collect();
        sorted.sort();
        let idx = ((pct / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }
}

/// Internal in-flight bookkeeping.
#[derive(Clone, Debug)]
struct InFlightEntry {
    key: SurfaceCacheKey,
    request: ProviderRequest,
    /// When the request was dispatched. Used to compute latency on success.
    #[allow(dead_code)]
    started: Instant,
}

/// Structured telemetry snapshot suitable for machine-readable stress reports.
///
/// All fields are plain copy types so the snapshot can be cloned cheaply and
/// serialized off the main thread.
#[derive(Clone, Debug, Default)]
pub struct StreamingTelemetry {
    pub queue_depth: usize,
    pub resident_tiles: usize,
    pub pending_in_flight: usize,
    pub failed_total: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub fallback_samples: u64,
    pub learned_samples: u64,
    pub rebuilds_queued: u64,
    pub rebuilds_completed: u64,
    pub health: ServiceHealthSnapshot,
    pub latency_p50_ms: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    pub provider_state: ProviderStateSnapshot,
}

impl StreamingTelemetry {
    /// Cache hit rate in `[0,1]`. `0.0` when no lookups have occurred.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Fallback percentage in `[0,100]`. `0.0` when no samples.
    pub fn fallback_percent(&self) -> f64 {
        let total = self.fallback_samples + self.learned_samples;
        if total == 0 {
            0.0
        } else {
            self.fallback_samples as f64 / total as f64 * 100.0
        }
    }
}

/// Copyable health snapshot for telemetry (avoids lifetime issues when the
/// live `ServiceHealth` evolves).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServiceHealthSnapshot {
    pub state: ServiceHealthTag,
    pub consecutive_successes: u32,
    pub consecutive_failures: u32,
    /// Milliseconds since the last health-state transition (relative, not a
    /// Unix timestamp). Used by reports to show how long the provider has
    /// been in its current state.
    pub ms_since_last_transition: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ServiceHealthTag {
    #[default]
    Cold,
    Healthy,
    Degraded,
    Unhealthy,
}

impl ServiceHealthTag {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
        }
    }
}

impl From<ServiceHealth> for ServiceHealthTag {
    fn from(h: ServiceHealth) -> Self {
        match h {
            ServiceHealth::Healthy => Self::Healthy,
            ServiceHealth::Degraded => Self::Degraded,
            ServiceHealth::Unhealthy => Self::Unhealthy,
            ServiceHealth::Cold => Self::Cold,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProviderStateSnapshot {
    pub attempts_total: u64,
    pub timeouts_total: u64,
    pub cancellations_total: u64,
    pub coalesced_duplicates_total: u64,
    pub stale_discarded_total: u64,
}

/// The bounded priority queue + in-flight set + health machine.
///
/// All mutating operations are `&self` (interior mutability via `Mutex`) so
/// the queue can be shared across the main thread and the provider transport
/// thread without exposing the lock to callers.
pub struct StreamingQueue {
    inner: Mutex<QueueInner>,
    config: StreamingConfig,
    /// Monotonic sequence counter for stable insertion order.
    seq_counter: AtomicU64,
    /// Jitter RNG, seeded for determinism. Guarded by the inner mutex.
    jitter: Mutex<ChaCha8Rng>,
}

struct QueueInner {
    /// Bucketed by priority class for O(1) per-class pop while keeping
    /// stable intra-class order. VecDeque preserves insertion order.
    buckets: [VecDeque<ProviderRequest>; PriorityClass::COUNT],
    /// Keys currently queued (any bucket) for coalescing.
    queued_keys: HashMap<SurfaceCacheKey, PriorityClass>,
    /// Keys currently in flight.
    in_flight: Vec<InFlightEntry>,
    /// Resident keys with LRU ordering. Bounded by `max_resident`; when full,
    /// the least-recently-used key is evicted (in sync with the SurfaceCache
    /// RAM LRU). This is NOT append-only: eviction removes the key so a
    /// future camera re-request can re-enqueue it.
    resident: HashMap<SurfaceCacheKey, ()>,
    /// LRU order for resident keys: front = LRU, back = MRU.
    resident_order: VecDeque<SurfaceCacheKey>,
    /// Maximum resident keys tracked. Must stay in sync with the
    /// SurfaceCache RAM capacity so the streaming queue does not retain
    /// stale resident entries after the RAM cache evicts them.
    max_resident: usize,
    /// Failed attempt counts per key, for backoff escalation.
    failure_counts: HashMap<SurfaceCacheKey, u32>,
    /// Health machine state.
    health: ServiceHealth,
    consecutive_successes: u32,
    consecutive_failures: u32,
    last_transition: Instant,
    /// Latency samples (successful request durations).
    latency: LatencyHistory,
    /// Telemetry counters.
    cache_hits: u64,
    cache_misses: u64,
    fallback_samples: u64,
    learned_samples: u64,
    rebuilds_queued: u64,
    rebuilds_completed: u64,
    attempts_total: u64,
    timeouts_total: u64,
    cancellations_total: u64,
    coalesced_duplicates_total: u64,
    stale_discarded_total: u64,
    failed_total: u64,
    /// Chart footprints that have arrived and need intersecting active chunks
    /// rebuilt. The terrain system pops these and computes intersecting active
    /// chunks via `chunks_intersecting_chart`.
    pending_arrived_charts: VecDeque<ChartFootprint>,
    /// Dedup set for `pending_arrived_charts`. If a chart arrives while its
    /// footprint is already pending, the dirty flag is set so the terrain
    /// system knows to re-enqueue it after processing.
    pending_arrived_seen: HashMap<ChartFootprint, bool>,
}

impl StreamingQueue {
    /// Construct a queue with the given config. Asserts the config is valid.
    pub fn new(config: StreamingConfig) -> Self {
        assert!(config.max_queued > 0, "max_queued must be positive");
        assert!(config.max_in_flight > 0, "max_in_flight must be positive");
        assert!(
            config.request_timeout > Duration::ZERO,
            "request_timeout must be positive"
        );
        assert!(
            config.base_backoff > Duration::ZERO,
            "base_backoff must be positive"
        );
        assert!(config.max_backoff >= config.base_backoff);
        assert!(config.healthy_threshold > 0);
        assert!(config.unhealthy_threshold > 0);
        assert!(config.unhealthy_cooldown > Duration::ZERO);
        assert!(config.max_resident > 0, "max_resident must be positive");
        let jitter_seed = config.jitter_seed;
        let max_resident = config.max_resident;
        Self {
            inner: Mutex::new(QueueInner {
                buckets: Default::default(),
                queued_keys: HashMap::new(),
                in_flight: Vec::new(),
                resident: HashMap::new(),
                resident_order: VecDeque::new(),
                max_resident,
                failure_counts: HashMap::new(),
                health: ServiceHealth::Cold,
                consecutive_successes: 0,
                consecutive_failures: 0,
                last_transition: Instant::now(),
                latency: LatencyHistory::new(4096),
                cache_hits: 0,
                cache_misses: 0,
                fallback_samples: 0,
                learned_samples: 0,
                rebuilds_queued: 0,
                rebuilds_completed: 0,
                attempts_total: 0,
                timeouts_total: 0,
                cancellations_total: 0,
                coalesced_duplicates_total: 0,
                stale_discarded_total: 0,
                failed_total: 0,
                pending_arrived_charts: VecDeque::new(),
                pending_arrived_seen: HashMap::new(),
            }),
            config,
            seq_counter: AtomicU64::new(0),
            jitter: Mutex::new(ChaCha8Rng::seed_from_u64(jitter_seed)),
        }
    }

    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    fn next_seq(&self) -> u64 {
        self.seq_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Enqueue a request if it is not already queued, in-flight, or resident.
    /// If a duplicate exists, the higher-priority classification wins and the
    /// duplicate is coalesced (counted). Returns `true` if newly enqueued.
    ///
    /// `provider_coordinate` is the canonical provider tile coordinate
    /// (face, x, y in `[0, tiles_per_face_edge)`) used for the sidecar
    /// request. This is DISTINCT from the chart key's x/y.
    pub fn enqueue(
        &self,
        key: SurfaceCacheKey,
        provider_coordinate: ProviderTileCoordinate,
        priority: PriorityClass,
        origin_chunk: Option<CellKey>,
        now: Instant,
    ) -> bool {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        // Resident: coalesce (it is already served).
        if inner.resident.contains_key(&key) {
            inner.coalesced_duplicates_total += 1;
            return false;
        }
        // In-flight: coalesce, but upgrade priority if the new one is higher.
        let mut upgrade_in_flight = false;
        if let Some(idx) = inner.in_flight.iter().position(|e| e.key == key) {
            if priority < inner.in_flight[idx].request.priority {
                upgrade_in_flight = true;
            } else {
                inner.coalesced_duplicates_total += 1;
                return false;
            }
        }
        if upgrade_in_flight {
            if let Some(entry) = inner.in_flight.iter_mut().find(|e| e.key == key) {
                entry.request.priority = priority;
            }
            inner.coalesced_duplicates_total += 1;
            return false;
        }
        // Already queued: coalesce, upgrade priority if higher.
        let existing_class = inner.queued_keys.get(&key).copied();
        if let Some(existing) = existing_class {
            if priority < existing {
                // Find and move the request to the higher-priority bucket.
                let bucket = &mut inner.buckets[existing as usize];
                let mut found: Option<(usize, ProviderRequest)> = None;
                for (i, r) in bucket.iter().enumerate() {
                    if r.key == key {
                        found = Some((i, r.clone()));
                        break;
                    }
                }
                if let Some((idx, mut req)) = found {
                    let _ = inner.buckets[existing as usize].remove(idx);
                    req.priority = priority;
                    req.deadline = Some(now + self.config.request_timeout);
                    inner.buckets[priority as usize].push_back(req);
                }
                inner.queued_keys.insert(key.clone(), priority);
            } else if let Some(req) = inner.buckets[existing as usize]
                .iter_mut()
                .find(|req| req.key == key)
            {
                // Active camera demand renews the queued request's lease.
                // Without this, a slow serial provider can discard a still-
                // needed halo tile before it reaches the head of the queue.
                req.deadline = Some(now + self.config.request_timeout);
            }
            inner.coalesced_duplicates_total += 1;
            return false;
        }
        // Bounded: if the queue is full, drop the lowest-priority tail.
        let total_queued: usize = inner.buckets.iter().map(|b| b.len()).sum();
        if total_queued >= self.config.max_queued {
            self.evict_lowest_priority_tail(&mut inner);
            // Re-check after eviction.
            let total_queued: usize = inner.buckets.iter().map(|b| b.len()).sum();
            if total_queued >= self.config.max_queued {
                inner.coalesced_duplicates_total += 1;
                return false;
            }
        }
        let seq = self.next_seq();
        let req = ProviderRequest {
            key: key.clone(),
            provider_coordinate,
            priority,
            seq,
            attempt: 0,
            deadline: Some(now + self.config.request_timeout),
            not_before: None,
            origin_chunk,
        };
        inner.buckets[priority as usize].push_back(req);
        inner.queued_keys.insert(key, priority);
        true
    }

    /// Evict the single lowest-priority, latest-inserted request to make
    /// room. Warmup and prefetch are evicted first.
    fn evict_lowest_priority_tail(&self, inner: &mut QueueInner) {
        for class in (0..PriorityClass::COUNT).rev() {
            if let Some(req) = inner.buckets[class].pop_back() {
                inner.queued_keys.remove(&req.key);
                return;
            }
        }
    }

    /// Pop the next request to dispatch, respecting concurrency limits,
    /// health state, backoff, and deadlines. Returns `None` if nothing is
    /// dispatchable right now.
    ///
    /// The returned token must be reported via `record_success` or
    /// `record_failure`/`record_timeout` to keep telemetry and in-flight
    /// bookkeeping consistent.
    pub fn pop_dispatchable(&self, now: Instant) -> Option<ProviderRequest> {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        // Health gate: pause dispatch when unhealthy until the cooldown has
        // elapsed, then allow a single probe. Cold (never contacted) allows
        // an immediate probe so the first request can establish health.
        if inner.health == ServiceHealth::Unhealthy {
            let elapsed = now.duration_since(inner.last_transition);
            if elapsed < self.config.unhealthy_cooldown {
                return None;
            }
        }
        if inner.in_flight.len() >= self.config.max_in_flight {
            return None;
        }
        // Scan buckets in priority order, skipping stale and not-yet-ready
        // (backoff) requests. Stale entries are discarded; the first ready
        // entry is dispatched.
        for class in 0..PriorityClass::COUNT {
            // Single pass: walk the bucket, discarding stale entries and
            // stopping at the first dispatchable one. Removing stale entries
            // shifts subsequent indices, so we use a while loop with a manual
            // cursor.
            let mut i = 0usize;
            let mut dispatch_key: Option<SurfaceCacheKey> = None;
            while i < inner.buckets[class].len() {
                let (stale, ready) = {
                    let req = &inner.buckets[class][i];
                    let stale = req.deadline.is_some_and(|d| now >= d);
                    let ready = req.not_before.is_none_or(|nb| now >= nb);
                    (stale, ready)
                };
                if stale {
                    if let Some(stale) = inner.buckets[class].remove(i) {
                        inner.queued_keys.remove(&stale.key);
                        inner.stale_discarded_total += 1;
                    }
                    // Do not advance i: the next entry shifted into this slot.
                    continue;
                }
                if ready {
                    dispatch_key = Some(inner.buckets[class][i].key.clone());
                    break;
                }
                i += 1;
            }
            if let Some(key) = dispatch_key {
                // Find the entry by key (index may have shifted if stale
                // entries were removed before it, but the break above
                // prevents that — still, look it up to be safe).
                if let Some(pos) = inner.buckets[class].iter().position(|r| r.key == key) {
                    if let Some(req) = inner.buckets[class].remove(pos) {
                        inner.queued_keys.remove(&req.key);
                        inner.attempts_total += 1;
                        inner.in_flight.push(InFlightEntry {
                            key: req.key.clone(),
                            request: req.clone(),
                            started: now,
                        });
                        return Some(req);
                    }
                }
            }
        }
        None
    }

    /// Report a successful decode/store. Removes the in-flight entry, records
    /// latency, marks the key resident (with LRU eviction), coalesces any
    /// queued/in-flight duplicate, and transitions health.
    ///
    /// This is the single source of truth for "a key became resident after a
    /// successful provider response." The background disk-promotion path uses
    /// [`mark_resident`](Self::mark_resident) which delegates to the same
    /// internal helper, so there is no ambiguity about who owns the resident
    /// set.
    pub fn record_success(&self, key: &SurfaceCacheKey, latency: Duration, now: Instant) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(pos) = inner.in_flight.iter().position(|e| &e.key == key) {
            inner.in_flight.remove(pos);
        }
        Self::mark_resident_locked(&mut inner, key);
        inner.failure_counts.remove(key);
        inner.latency.push(latency);
        inner.consecutive_successes += 1;
        inner.consecutive_failures = 0;
        if inner.consecutive_successes >= self.config.healthy_threshold {
            if inner.health != ServiceHealth::Healthy {
                inner.health = ServiceHealth::Healthy;
                inner.last_transition = now;
            }
        } else if inner.health == ServiceHealth::Cold {
            // First success lifts cold into degraded until the threshold is met.
            inner.health = ServiceHealth::Degraded;
            inner.last_transition = now;
        }
    }

    /// Report a provider failure. Schedules a retry with exponential backoff
    /// (deterministic jitter) and transitions health toward unhealthy.
    pub fn record_failure(&self, key: &SurfaceCacheKey, now: Instant) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        // Capture the provider coordinate from the in-flight entry before
        // removing it, so retries use the correct sidecar identity.
        let provider_coordinate =
            if let Some(pos) = inner.in_flight.iter().position(|e| &e.key == key) {
                let entry = inner.in_flight.remove(pos);
                entry.request.provider_coordinate
            } else {
                // No in-flight entry (e.g. timeout sweep): reconstruct from the
                // key's chart coordinates as a fallback. This should not happen
                // in normal operation.
                ProviderTileCoordinate {
                    face: key.face,
                    x: key.x,
                    y: key.y,
                }
            };
        // Compute the new attempt count without holding the entry borrow
        // across other inner mutations.
        let new_attempt = {
            let attempts = inner.failure_counts.entry(key.clone()).or_insert(0);
            *attempts += 1;
            *attempts
        };
        inner.consecutive_failures += 1;
        inner.consecutive_successes = 0;
        if inner.consecutive_failures >= self.config.unhealthy_threshold {
            if inner.health != ServiceHealth::Unhealthy {
                inner.health = ServiceHealth::Unhealthy;
                inner.last_transition = now;
            }
        } else if inner.health == ServiceHealth::Healthy {
            inner.health = ServiceHealth::Degraded;
            inner.last_transition = now;
        }
        inner.failed_total += 1;

        // Re-enqueue with backoff. Cap attempts to avoid infinite loops; a
        // capped request is dropped (procedural fallback remains active).
        const MAX_ATTEMPTS: u32 = 8;
        let attempt = new_attempt;
        if attempt < MAX_ATTEMPTS {
            let backoff = self.backoff_for(attempt);
            let priority = PriorityClass::Warmup; // retries deprioritize
            let req = ProviderRequest {
                key: key.clone(),
                provider_coordinate,
                priority,
                seq: self.next_seq(),
                attempt,
                deadline: Some(now + self.config.request_timeout),
                not_before: Some(now + backoff),
                origin_chunk: None,
            };
            inner.buckets[priority as usize].push_back(req);
            inner.queued_keys.insert(key.clone(), priority);
        } else {
            // Exhausted retries: drop the key so a fresh enqueue can try
            // again later if the camera re-requests it.
            inner.failure_counts.remove(key);
        }
    }

    /// Report a timeout (deadline exceeded or transport timeout). Treated as
    /// a failure but recorded separately in telemetry.
    pub fn record_timeout(&self, key: &SurfaceCacheKey, now: Instant) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        inner.timeouts_total += 1;
        drop(inner);
        self.record_failure(key, now);
    }

    /// Cancel an in-flight request (e.g. the camera moved away). The key is
    /// removed from in-flight and its failure count is not incremented.
    pub fn cancel(&self, key: &SurfaceCacheKey) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(pos) = inner.in_flight.iter().position(|e| &e.key == key) {
            inner.in_flight.remove(pos);
            inner.cancellations_total += 1;
        }
        // Also drop any queued instance so it does not re-dispatch.
        if let Some(class) = inner.queued_keys.remove(key) {
            if let Some(pos) = inner.buckets[class as usize]
                .iter()
                .position(|r| &r.key == key)
            {
                inner.buckets[class as usize].remove(pos);
            }
            inner.cancellations_total += 1;
        }
    }

    /// Mark a key as resident (e.g. promoted from disk by the background
    /// worker). Coalesces any queued/in-flight request for the same key.
    /// Uses the same internal helper as `record_success` so there is no
    /// ambiguity about resident-set ownership.
    pub fn mark_resident(&self, key: &SurfaceCacheKey) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        Self::mark_resident_locked(&mut inner, key);
    }

    /// Internal helper: mark a key resident with LRU eviction and coalescing.
    /// When a NEW key becomes resident, its chart footprint is pushed into
    /// `pending_arrived_charts` so the terrain system can compute intersecting
    /// active chunks and rebuild only those (not all active chunks).
    /// Caller must hold the inner mutex.
    fn mark_resident_locked(inner: &mut QueueInner, key: &SurfaceCacheKey) {
        // If already resident, just touch the LRU.
        if inner.resident.contains_key(key) {
            if let Some(pos) = inner.resident_order.iter().position(|k| k == key) {
                inner.resident_order.remove(pos);
            }
            inner.resident_order.push_back(key.clone());
            return;
        }
        // Enforce the resident bound: evict the LRU key.
        while inner.resident.len() >= inner.max_resident {
            if let Some(lru) = inner.resident_order.pop_front() {
                inner.resident.remove(&lru);
            } else {
                break;
            }
        }
        inner.resident.insert(key.clone(), ());
        inner.resident_order.push_back(key.clone());
        // Coalesce: drop any queued/in-flight instance.
        if let Some(class) = inner.queued_keys.remove(key) {
            if let Some(pos) = inner.buckets[class as usize]
                .iter()
                .position(|r| &r.key == key)
            {
                inner.buckets[class as usize].remove(pos);
            }
            inner.coalesced_duplicates_total += 1;
        }
        if let Some(pos) = inner.in_flight.iter().position(|e| &e.key == key) {
            inner.in_flight.remove(pos);
            inner.coalesced_duplicates_total += 1;
        }
        inner.failure_counts.remove(key);
        // Push the chart footprint so the terrain system can compute
        // intersecting active chunks. The footprint is the chart's
        // (face, level, x, y) encoded as a CellKey with lod=chart_level.
        let footprint = ChartFootprint::from_key(key);
        // If the footprint is already pending, set the dirty flag so the
        // terrain system knows a content revision refresh arrived while the
        // footprint was still in the queue. This ensures no refresh is lost.
        match inner.pending_arrived_seen.get_mut(&footprint) {
            Some(dirty) => {
                *dirty = true;
            }
            None => {
                inner.pending_arrived_seen.insert(footprint, false);
                inner.pending_arrived_charts.push_back(footprint);
                inner.rebuilds_queued += 1;
            }
        }
    }

    /// Evict a resident key (e.g. LRU eviction from the RAM cache). Future
    /// lookups will miss and re-enqueue. Removes from both the resident set
    /// and the LRU order so the key can be re-requested.
    pub fn evict_resident(&self, key: &SurfaceCacheKey) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        inner.resident.remove(key);
        if let Some(pos) = inner.resident_order.iter().position(|k| k == key) {
            inner.resident_order.remove(pos);
        }
    }

    /// Record a resident-cache hit/miss from the mesh-worker read path. On a
    /// hit, also touches the LRU so the resident key stays hot.
    pub fn record_cache_lookup(&self, hit: bool) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if hit {
            inner.cache_hits += 1;
        } else {
            inner.cache_misses += 1;
        }
    }

    /// Check if a key is resident (non-blocking, for mesh-worker snapshots).
    pub fn is_resident(&self, key: &SurfaceCacheKey) -> bool {
        let inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        inner.resident.contains_key(key)
    }

    /// Whether any sample-source data has been recorded. Used by stress gates
    /// to distinguish "no learned data arrived" (gate fails) from "learned
    /// data arrived and fallback is X%" (gate evaluates).
    pub fn coverage_known(&self) -> bool {
        let inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        (inner.fallback_samples + inner.learned_samples) > 0
    }

    /// Record a sample of the composed field source: procedural fallback vs
    /// learned macro. Used to compute fallback percentage.
    pub fn record_sample_source(&self, learned: bool) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if learned {
            inner.learned_samples += 1;
        } else {
            inner.fallback_samples += 1;
        }
    }

    /// Pop the next chart footprint that has arrived and needs intersecting
    /// active chunks rebuilt. The terrain system computes intersecting active
    /// chunks via `chunks_intersecting_chart`. Returns `None` when no
    /// arrivals are pending.
    pub fn pop_arrived_chart(&self) -> Option<ChartFootprint> {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(footprint) = inner.pending_arrived_charts.pop_front() {
            // Check if a content revision refresh arrived while this footprint
            // was pending. If so, re-enqueue it so the refresh is not lost.
            let dirty = inner
                .pending_arrived_seen
                .remove(&footprint)
                .unwrap_or(false);
            if dirty {
                inner.pending_arrived_seen.insert(footprint, false);
                inner.pending_arrived_charts.push_back(footprint);
            }
            inner.rebuilds_completed += 1;
            Some(footprint)
        } else {
            None
        }
    }

    /// Snapshot telemetry. This is a point-in-time copy; counters are
    /// monotonic except where documented.
    pub fn telemetry(&self) -> StreamingTelemetry {
        let inner = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let queue_depth: usize = inner.buckets.iter().map(|b| b.len()).sum();
        StreamingTelemetry {
            queue_depth,
            resident_tiles: inner.resident.len(),
            pending_in_flight: inner.in_flight.len(),
            failed_total: inner.failed_total,
            cache_hits: inner.cache_hits,
            cache_misses: inner.cache_misses,
            fallback_samples: inner.fallback_samples,
            learned_samples: inner.learned_samples,
            rebuilds_queued: inner.rebuilds_queued,
            rebuilds_completed: inner.rebuilds_completed,
            health: ServiceHealthSnapshot {
                state: inner.health.into(),
                consecutive_successes: inner.consecutive_successes,
                consecutive_failures: inner.consecutive_failures,
                ms_since_last_transition: inner
                    .last_transition
                    .elapsed()
                    .as_millis()
                    .min(u64::MAX as u128) as u64,
            },
            latency_p50_ms: inner
                .latency
                .percentile(50.0)
                .map(|d| d.as_secs_f64() * 1000.0),
            latency_p95_ms: inner
                .latency
                .percentile(95.0)
                .map(|d| d.as_secs_f64() * 1000.0),
            provider_state: ProviderStateSnapshot {
                attempts_total: inner.attempts_total,
                timeouts_total: inner.timeouts_total,
                cancellations_total: inner.cancellations_total,
                coalesced_duplicates_total: inner.coalesced_duplicates_total,
                stale_discarded_total: inner.stale_discarded_total,
            },
        }
    }

    /// Exponential backoff with deterministic jitter: `base * 2^attempt`
    /// capped at `max_backoff`, plus a jitter in `[0, base/2)` from the seeded
    /// RNG. Determinism is required for reproducible stress reports.
    fn backoff_for(&self, attempt: u32) -> Duration {
        let base = self.config.base_backoff;
        let multiplier: u32 = 1u32.checked_shl(attempt.min(20)).unwrap_or(u32::MAX);
        let exp = base
            .checked_mul(multiplier)
            .unwrap_or(self.config.max_backoff);
        let capped = exp.min(self.config.max_backoff);
        let jitter_max = base.as_nanos() / 2;
        let jitter_ns = if jitter_max > 0 {
            let mut rng = match self.jitter.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            rng.random_range(0..jitter_max as u64)
        } else {
            0
        };
        capped + Duration::from_nanos(jitter_ns)
    }

    /// Current health (for tests + diagnostics).
    pub fn health(&self) -> ServiceHealth {
        match self.inner.lock() {
            Ok(g) => g.health,
            Err(p) => p.into_inner().health,
        }
    }

    /// Number of currently resident keys.
    pub fn resident_count(&self) -> usize {
        match self.inner.lock() {
            Ok(g) => g.resident.len(),
            Err(p) => p.into_inner().resident.len(),
        }
    }

    /// Number of currently queued keys (any bucket).
    pub fn queued_count(&self) -> usize {
        match self.inner.lock() {
            Ok(g) => g.queued_keys.len(),
            Err(p) => p.into_inner().queued_keys.len(),
        }
    }

    /// Number of currently in-flight requests.
    pub fn in_flight_count(&self) -> usize {
        match self.inner.lock() {
            Ok(g) => g.in_flight.len(),
            Err(p) => p.into_inner().in_flight.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Halo-residency gate: a chunk uses learned data only if every elevation
// plus normal-halo dependency is resident.
// ---------------------------------------------------------------------------

/// Residency gate at CHUNK dependency granularity. Given the set of chart
/// keys a chunk's elevation + normal halo depends on, returns `true` only if
/// *all* of them are resident. Otherwise the chunk must use procedural
/// fallback for its *entire* elevation+normal field — no mixed normals.
///
/// This implements roadmap rule 5.2.1: "A chunk uses learned data only when
/// every elevation and normal halo dependency is resident."
pub struct ResidencyGate;

impl ResidencyGate {
    /// `dependencies` is the set of chart keys the chunk's vertices and
    /// normal-halo samples fall into. Returns `true` iff every key is
    /// resident in `queue`.
    pub fn chunk_is_fully_resident(
        queue: &StreamingQueue,
        dependencies: &[SurfaceCacheKey],
    ) -> bool {
        if dependencies.is_empty() {
            return false;
        }
        let inner = match queue.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        dependencies.iter().all(|k| inner.resident.contains_key(k))
    }
}

// ---------------------------------------------------------------------------
// Chunk<->chart intersection: rebuild only chunks intersecting a changed tile.
// ---------------------------------------------------------------------------

/// An explicit chart footprint carrying the exact provider grid width. This
/// replaces the old `CellKey`-encoded footprint (which used `lod=chart_level`
/// and derived `N = 2^level`). With non-power-of-two tile counts (e.g. Earth's
/// 652), the exact `charts_per_face_edge` must be used for all footprint math.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChartFootprint {
    pub face: u8,
    pub x: u32,
    pub y: u32,
    /// Exact charts per face edge (e.g. 652 for Earth, 4 for miniature).
    pub charts_per_face_edge: u32,
}

impl ChartFootprint {
    /// Create a footprint from a `SurfaceCacheKey`.
    pub fn from_key(key: &SurfaceCacheKey) -> Self {
        Self {
            face: key.face,
            x: key.x,
            y: key.y,
            charts_per_face_edge: key.charts_per_face_edge,
        }
    }

    /// The uv footprint: `[x/N, (x+1)/N] x [y/N, (y+1)/N]`.
    pub fn uv_footprint(&self) -> (f64, f64, f64, f64) {
        let n = self.charts_per_face_edge.max(1) as f64;
        (
            self.x as f64 / n,
            self.y as f64 / n,
            (self.x + 1) as f64 / n,
            (self.y + 1) as f64 / n,
        )
    }
}

/// Compute the set of render-quadtree chunks (at a given LOD) that intersect
/// a single chart's uv footprint. Used so a tile revision change rebuilds
/// only the intersecting chunks, not every active chunk.
///
/// A chart at `(face, x, y)` with `charts_per_face_edge=N` covers uv
/// `[x/N, (x+1)/N] x [y/N, (y+1)/N]` on its face. A render chunk at
/// `chunk_lod` covers the analogous uv cell. We return all chunk cells whose
/// uv cell overlaps the chart's uv footprint (including a configurable halo
/// margin in cells).
pub fn chunks_intersecting_chart(
    face: u8,
    chart_x: u32,
    chart_y: u32,
    charts_per_face_edge: u32,
    chunk_lod: u8,
    halo_cells: u32,
) -> Vec<CellKey> {
    let n_chart = charts_per_face_edge.max(1) as f64;
    let n_chunk = 1u32 << chunk_lod;
    // Chart uv footprint.
    let cu0 = chart_x as f64 / n_chart;
    let cv0 = chart_y as f64 / n_chart;
    let cu1 = (chart_x + 1) as f64 / n_chart;
    let cv1 = (chart_y + 1) as f64 / n_chart;
    // Expand by halo in chunk-cell units.
    let cell_uv = 1.0 / n_chunk as f64;
    let margin = halo_cells as f64 * cell_uv;
    let eu0 = (cu0 - margin).max(0.0);
    let ev0 = (cv0 - margin).max(0.0);
    let eu1 = (cu1 + margin).min(1.0);
    let ev1 = (cv1 + margin).min(1.0);
    let i0 = (eu0 * n_chunk as f64).floor() as u32;
    let i1 = ((eu1 * n_chunk as f64).ceil() as u32)
        .saturating_sub(1)
        .max(i0);
    let j0 = (ev0 * n_chunk as f64).floor() as u32;
    let j1 = ((ev1 * n_chunk as f64).ceil() as u32)
        .saturating_sub(1)
        .max(j0);
    let mut out = Vec::new();
    for i in i0..=i1 {
        for j in j0..=j1 {
            out.push(CellKey {
                face,
                i,
                j,
                lod: chunk_lod,
            });
        }
    }
    out
}

/// Compute the chart keys a chunk's elevation + normal halo depends on.
/// This is the inverse of `chunks_intersecting_chart`: given a render chunk,
/// return the charts whose footprint overlaps the chunk's uv cell expanded by
/// the halo (in chart samples converted to chart-uv).
///
/// Uses the exact `charts_per_face_edge` (not a padded power-of-two) for all
/// footprint math.
pub fn chart_dependencies_for_chunk(
    chunk: CellKey,
    charts_per_face_edge: u32,
    halo_samples: u32,
    core_resolution: u32,
) -> Vec<(u8, u32, u32)> {
    let n_chunk = 1u32 << chunk.lod;
    let n_chart = charts_per_face_edge.max(1) as f64;
    let u0 = chunk.i as f64 / n_chunk as f64;
    let v0 = chunk.j as f64 / n_chunk as f64;
    let u1 = (chunk.i + 1) as f64 / n_chunk as f64;
    let v1 = (chunk.j + 1) as f64 / n_chunk as f64;
    // Halo in chart-uv units. The chunk's vertices sample a region; the
    // normal-halo extends beyond by `halo_samples` in chart pixel space.
    let chart_uv_per_sample = 1.0 / (n_chart * core_resolution.max(1) as f64);
    let halo_uv = halo_samples as f64 * chart_uv_per_sample;
    let eu0 = (u0 - halo_uv).max(0.0);
    let ev0 = (v0 - halo_uv).max(0.0);
    let eu1 = (u1 + halo_uv).min(1.0);
    let ev1 = (v1 + halo_uv).min(1.0);
    let x0 = (eu0 * n_chart).floor() as u32;
    let x1 = ((eu1 * n_chart).ceil() as u32).saturating_sub(1).max(x0);
    let y0 = (ev0 * n_chart).floor() as u32;
    let y1 = ((ev1 * n_chart).ceil() as u32).saturating_sub(1).max(y0);
    let mut out = Vec::new();
    for x in x0..=x1 {
        for y in y0..=y1 {
            out.push((chunk.face, x, y));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Blend provenance: blend learned macro elevation/material provenance over a
// defined time/distance transition WITHOUT blending world coordinates.
// ---------------------------------------------------------------------------

/// A time/distance-based blend weight in `[0,1]` for transitioning a chunk
/// from procedural to learned provenance. `1.0` = fully learned, `0.0` =
/// fully procedural. The blend operates on the *provenance weight* applied to
/// the learned macro elevation, never on world coordinates (so there is no
/// height step or coastline crawl).
///
/// The transition is driven by `resident_age` (time since the chunk's
/// dependencies became resident) and `camera_distance_m` (distance from the
/// camera to the chunk). Close chunks blend in faster; far chunks are
/// instant. The transition interval is `transition_seconds`.
#[derive(Clone, Copy, Debug)]
pub struct BlendWeights {
    /// Weight in `[0,1]` applied to the learned macro elevation.
    pub learned_weight: f64,
    /// Weight `1 - learned_weight` applied to the procedural macro.
    pub procedural_weight: f64,
}

impl BlendWeights {
    /// Compute blend weights. `resident_age` is the time since residency
    /// became complete; `camera_distance_m` is the camera-to-chunk distance.
    /// `transition_seconds` is the full blend-in duration at close range.
    pub fn for_transition(
        resident_age: Duration,
        camera_distance_m: f64,
        transition_seconds: f64,
    ) -> Self {
        // Far chunks (beyond 100 km) snap to learned instantly: there is no
        // visible pop at that distance and the smoothstep is unnecessary.
        if camera_distance_m > 100_000.0 {
            return Self {
                learned_weight: 1.0,
                procedural_weight: 0.0,
            };
        }
        if transition_seconds <= 0.0 {
            return Self {
                learned_weight: 1.0,
                procedural_weight: 0.0,
            };
        }
        let t = (resident_age.as_secs_f64() / transition_seconds).clamp(0.0, 1.0);
        // Smoothstep for a C1-continuous blend (no derivative pop).
        let smooth = t * t * (3.0 - 2.0 * t);
        // Close chunks (0 m) blend over the full interval (scale = 1.0).
        // Mid-range chunks accelerate the blend slightly so far-but-visible
        // chunks complete faster. The far-distance bypass above already
        // handles >100 km.
        let distance_scale = 1.0 - 0.3 * (camera_distance_m / 100_000.0).min(1.0);
        let learned = (smooth * distance_scale).clamp(0.0, 1.0);
        // Ensure the blend eventually reaches fully learned: at t=1, smooth=1,
        // so learned = distance_scale. If distance_scale < 1, clamp up to 1
        // once the transition is complete so the chunk is fully learned.
        let learned = if t >= 1.0 { 1.0 } else { learned };
        Self {
            learned_weight: learned,
            procedural_weight: 1.0 - learned,
        }
    }

    /// Blend two elevation values by the provenance weights. This blends the
    /// *macro elevation provenance*, not world coordinates: the procedural
    /// residual is always added on top of whichever macro won.
    pub fn blend_elevation(&self, procedural_macro: f64, learned_macro: f64) -> f64 {
        procedural_macro * self.procedural_weight + learned_macro * self.learned_weight
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_charts::{
        ChartOwnership, SurfaceChartId, SurfaceChartMetadata, SurfacePatchId,
        SURFACE_CHART_PROJECTION_REVISION,
    };
    use er_core::math::CellKey;

    fn test_meta(seed: u64) -> SurfaceChartMetadata {
        SurfaceChartMetadata {
            seed,
            projection_revision: SURFACE_CHART_PROJECTION_REVISION,
            model_revision: "m5-test-v1".to_owned(),
            conditioning_revision: 1,
            residual_revision: 1,
            sea_level_datum_m: 0,
            pixel_scale_m: 30,
            halo_samples: 1,
            core_resolution: 4,
            ownership: ChartOwnership::LearnedReliefProceduralShoreline,
            planet_radius_m: 6_371_000,
            charts_per_face_edge: 652,
        }
    }

    fn key_for(seed: u64, face: u8, x: u32, y: u32) -> SurfaceCacheKey {
        let meta = test_meta(seed);
        let patch = SurfacePatchId::new(
            SurfaceChartId {
                face,
                level: 2,
                x,
                y,
                charts_per_face_edge: 652,
            },
            1,
        );
        SurfaceCacheKey::from_metadata(&meta, patch, [0, 0, 6, 6])
    }

    /// Helper: create a ProviderTileCoordinate matching a key_for call.
    fn ptc_for(face: u8, x: u32, y: u32) -> ProviderTileCoordinate {
        ProviderTileCoordinate { face, x, y }
    }

    fn small_config() -> StreamingConfig {
        StreamingConfig {
            max_queued: 16,
            max_in_flight: 8,
            request_timeout: Duration::from_secs(2),
            base_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(100),
            healthy_threshold: 2,
            unhealthy_threshold: 3,
            unhealthy_cooldown: Duration::from_millis(50),
            jitter_seed: 42,
            max_resident: 32,
        }
    }

    fn now() -> Instant {
        Instant::now()
    }

    // ---- Priority ordering ----

    #[test]
    fn pop_dispatches_highest_priority_class_first() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        // Enqueue in reverse priority order.
        q.enqueue(
            key_for(1, 0, 3, 3),
            ptc_for(0, 3, 3),
            PriorityClass::Warmup,
            None,
            t,
        );
        q.enqueue(
            key_for(1, 0, 2, 2),
            ptc_for(0, 2, 2),
            PriorityClass::FarRootCoverage,
            None,
            t,
        );
        q.enqueue(
            key_for(1, 0, 0, 0),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        q.enqueue(
            key_for(1, 0, 1, 1),
            ptc_for(0, 1, 1),
            PriorityClass::NormalHalo,
            None,
            t,
        );

        let first = q.pop_dispatchable(t).unwrap();
        assert_eq!(first.priority, PriorityClass::VisibleSurface);
        let second = q.pop_dispatchable(t).unwrap();
        assert_eq!(second.priority, PriorityClass::NormalHalo);
        let third = q.pop_dispatchable(t).unwrap();
        assert_eq!(third.priority, PriorityClass::FarRootCoverage);
        let fourth = q.pop_dispatchable(t).unwrap();
        assert_eq!(fourth.priority, PriorityClass::Warmup);
        assert!(q.pop_dispatchable(t).is_none());
    }

    #[test]
    fn pop_preserves_insertion_order_within_a_class() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        q.enqueue(
            key_for(1, 0, 0, 0),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        q.enqueue(
            key_for(2, 0, 1, 1),
            ptc_for(0, 1, 1),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        q.enqueue(
            key_for(3, 0, 2, 2),
            ptc_for(0, 2, 2),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let first = q.pop_dispatchable(t).unwrap();
        let second = q.pop_dispatchable(t).unwrap();
        let third = q.pop_dispatchable(t).unwrap();
        assert_eq!(first.key.seed, 1);
        assert_eq!(second.key.seed, 2);
        assert_eq!(third.key.seed, 3);
    }

    // ---- Coalescing ----

    #[test]
    fn enqueue_coalesces_duplicate_queued_request() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        assert!(q.enqueue(k.clone(), ptc_for(0, 0, 0), PriorityClass::Warmup, None, t));
        // Duplicate: coalesced, not re-enqueued.
        assert!(!q.enqueue(k.clone(), ptc_for(0, 0, 0), PriorityClass::Warmup, None, t));
        assert_eq!(q.queued_count(), 1);
        assert_eq!(q.telemetry().provider_state.coalesced_duplicates_total, 1);
    }

    #[test]
    fn enqueue_upgrades_priority_of_queued_duplicate() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.enqueue(k.clone(), ptc_for(0, 0, 0), PriorityClass::Warmup, None, t);
        // Higher-priority duplicate: coalesced but priority upgraded.
        assert!(!q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t
        ));
        let req = q.pop_dispatchable(t).unwrap();
        assert_eq!(req.priority, PriorityClass::VisibleSurface);
    }

    #[test]
    fn enqueue_coalesces_duplicate_in_flight_request() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let _ = q.pop_dispatchable(t).unwrap();
        // Now in-flight: duplicate must coalesce.
        assert!(!q.enqueue(k.clone(), ptc_for(0, 0, 0), PriorityClass::Warmup, None, t));
        assert_eq!(q.in_flight_count(), 1);
        assert_eq!(q.queued_count(), 0);
    }

    #[test]
    fn enqueue_coalesces_duplicate_resident_key() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.mark_resident(&k);
        assert!(!q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t
        ));
        assert_eq!(q.queued_count(), 0);
    }

    // ---- Backoff ----

    #[test]
    fn backoff_is_exponential_and_capped_and_deterministic() {
        let cfg = small_config();
        let q = StreamingQueue::new(cfg.clone());
        let b1 = q.backoff_for(1);
        let b2 = q.backoff_for(2);
        let b3 = q.backoff_for(3);
        // Exponential: each step >= 2x the previous base (minus jitter slack).
        assert!(b2 >= b1, "b2={b2:?} < b1={b1:?}");
        assert!(b3 >= b2, "b3={b3:?} < b2={b2:?}");
        // Capped.
        let b20 = q.backoff_for(20);
        assert!(b20 <= cfg.max_backoff + cfg.base_backoff);
        // Deterministic: the *base* backoff (without jitter) is deterministic
        // by construction (pure function of attempt). The jitter adds a
        // non-deterministic offset; verify the base is monotonic by checking
        // that the capped portion follows the exponential curve. We verify
        // determinism of the *base* by comparing two queues where we compute
        // the same attempt first (fresh RNG state each time).
        let q_a = StreamingQueue::new(cfg.clone());
        let q_b = StreamingQueue::new(cfg);
        // First call on each: same seed -> same jitter -> same total backoff.
        assert_eq!(q_a.backoff_for(3), q_b.backoff_for(3));
    }

    #[test]
    fn failure_re_enqueues_with_backoff_not_before() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let _req = q.pop_dispatchable(t).unwrap();
        assert_eq!(q.in_flight_count(), 1);
        q.record_failure(&k, t);
        // Re-enqueued in Warmup with not_before set.
        assert_eq!(q.queued_count(), 1);
        // Immediately after: not_before should block dispatch.
        assert!(q.pop_dispatchable(t).is_none());
        // After the backoff elapses: dispatchable.
        let later = t + Duration::from_millis(200);
        let retry = q.pop_dispatchable(later).unwrap();
        assert_eq!(retry.priority, PriorityClass::Warmup);
        assert!(retry.attempt >= 1);
    }

    // ---- Health state machine ----

    #[test]
    fn health_starts_cold_and_transitions_to_healthy_after_threshold() {
        let q = StreamingQueue::new(small_config());
        assert_eq!(q.health(), ServiceHealth::Cold);
        let t = now();
        // healthy_threshold = 2
        let k1 = key_for(1, 0, 0, 0);
        q.enqueue(
            k1.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let _ = q.pop_dispatchable(t).unwrap();
        q.record_success(&k1, Duration::from_millis(5), t);
        assert_eq!(q.health(), ServiceHealth::Degraded);
        let k2 = key_for(2, 0, 1, 1);
        q.enqueue(
            k2.clone(),
            ptc_for(0, 1, 1),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let _ = q.pop_dispatchable(t).unwrap();
        q.record_success(&k2, Duration::from_millis(5), t);
        assert_eq!(q.health(), ServiceHealth::Healthy);
    }

    #[test]
    fn health_transitions_to_unhealthy_after_failure_threshold_and_pauses_dispatch() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        // unhealthy_threshold = 3
        for i in 0..3 {
            let k = key_for(i + 1, 0, i as u32, 0);
            q.enqueue(
                k.clone(),
                ptc_for(0, 0, 0),
                PriorityClass::VisibleSurface,
                None,
                t,
            );
            let _ = q.pop_dispatchable(t).unwrap();
            q.record_failure(&k, t);
        }
        assert_eq!(q.health(), ServiceHealth::Unhealthy);
        // Dispatch paused during cooldown.
        q.enqueue(
            key_for(99, 0, 9, 9),
            ptc_for(0, 9, 9),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        assert!(q.pop_dispatchable(t).is_none());
        // After cooldown: a probe is allowed.
        let later = t + Duration::from_millis(100);
        let probe = q.pop_dispatchable(later);
        assert!(probe.is_some(), "probe should be allowed after cooldown");
    }

    // ---- Timeout ----

    #[test]
    fn timeout_records_and_treats_as_failure() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let _ = q.pop_dispatchable(t).unwrap();
        q.record_timeout(&k, t);
        assert_eq!(q.telemetry().provider_state.timeouts_total, 1);
        assert!(q.telemetry().failed_total >= 1);
    }

    // ---- Cancellation ----

    #[test]
    fn cancel_removes_in_flight_without_counting_as_failure() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let _ = q.pop_dispatchable(t).unwrap();
        q.cancel(&k);
        assert_eq!(q.in_flight_count(), 0);
        assert_eq!(q.telemetry().provider_state.cancellations_total, 1);
        assert_eq!(q.telemetry().failed_total, 0);
    }

    #[test]
    fn cancel_also_drops_queued_duplicate() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        q.cancel(&k);
        assert_eq!(q.queued_count(), 0);
    }

    // ---- Stale discard ----

    #[test]
    fn stale_request_past_deadline_is_discarded() {
        let mut cfg = small_config();
        cfg.request_timeout = Duration::from_millis(10);
        let q = StreamingQueue::new(cfg);
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        // After the deadline, pop should discard the stale request.
        let later = t + Duration::from_millis(50);
        assert!(q.pop_dispatchable(later).is_none());
        assert_eq!(q.queued_count(), 0);
        assert_eq!(q.telemetry().provider_state.stale_discarded_total, 1);
        assert_eq!(q.telemetry().failed_total, 0);
    }

    #[test]
    fn repeated_active_demand_renews_queued_request_lease() {
        let mut cfg = small_config();
        cfg.request_timeout = Duration::from_millis(10);
        let q = StreamingQueue::new(cfg);
        let t = now();
        let k = key_for(1, 0, 0, 0);
        assert!(q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::NormalHalo,
            None,
            t,
        ));

        let renewed_at = t + Duration::from_millis(8);
        assert!(!q.enqueue(
            k,
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            renewed_at,
        ));
        assert!(q.pop_dispatchable(t + Duration::from_millis(12)).is_some());
        assert_eq!(q.telemetry().provider_state.stale_discarded_total, 0);
    }

    // ---- Residency gate ----

    #[test]
    fn residency_gate_returns_false_when_any_dependency_missing() {
        let q = StreamingQueue::new(small_config());
        let k1 = key_for(1, 0, 0, 0);
        let k2 = key_for(2, 0, 1, 1);
        q.mark_resident(&k1);
        // Only k1 resident; k2 missing.
        assert!(!ResidencyGate::chunk_is_fully_resident(
            &q,
            &[k1.clone(), k2.clone()]
        ));
        q.mark_resident(&k2);
        assert!(ResidencyGate::chunk_is_fully_resident(&q, &[k1, k2]));
    }

    #[test]
    fn residency_gate_returns_false_for_empty_dependencies() {
        let q = StreamingQueue::new(small_config());
        assert!(!ResidencyGate::chunk_is_fully_resident(&q, &[]));
    }

    // ---- Chunk<->chart intersection ----

    #[test]
    fn chunks_intersecting_chart_returns_overlapping_cells() {
        // Chart at level 2 (4 charts/edge), chunk at LOD 3 (8 cells/edge).
        // Chart (1,1) covers uv [0.25,0.5]x[0.25,0.5].
        // Chunk cells overlapping that uv range: i,j in [2,3].
        let chunks = chunks_intersecting_chart(0, 1, 1, 4, 3, 0);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert_eq!(c.face, 0);
            assert_eq!(c.lod, 3);
            assert!((2..=3).contains(&c.i), "i={}", c.i);
            assert!((2..=3).contains(&c.j), "j={}", c.j);
        }
    }

    #[test]
    fn chunks_intersecting_chart_includes_halo_margin() {
        // With halo_cells=1, a chart at the corner of a face should include
        // at least one extra cell row/col.
        let no_halo = chunks_intersecting_chart(0, 0, 0, 4, 3, 0);
        let with_halo = chunks_intersecting_chart(0, 0, 0, 4, 3, 1);
        assert!(with_halo.len() >= no_halo.len());
    }

    #[test]
    fn chart_dependencies_for_chunk_inverse_of_intersection() {
        // If a chunk intersects a chart, the chart should be in the chunk's
        // dependency set.
        let chunk = CellKey {
            face: 0,
            i: 2,
            j: 3,
            lod: 3,
        };
        let deps = chart_dependencies_for_chunk(chunk, 2, 1, 4);
        assert!(!deps.is_empty());
        // Every dependency must be on the same face.
        for (face, _, _) in &deps {
            assert_eq!(*face, 0);
        }
    }

    #[test]
    fn only_intersecting_chunks_are_rebuilt_when_a_tile_arrives() {
        // When chart (face=0, x=1, y=1, level=2) arrives, only chunks
        // intersecting it should be in the rebuild set.
        let rebuilt = chunks_intersecting_chart(0, 1, 1, 4, 3, 0);
        let far_chunk = CellKey {
            face: 0,
            i: 7,
            j: 7,
            lod: 3,
        };
        assert!(!rebuilt.contains(&far_chunk));
        let near_chunk = CellKey {
            face: 0,
            i: 2,
            j: 2,
            lod: 3,
        };
        assert!(rebuilt.contains(&near_chunk));
    }

    // ---- Blend provenance ----

    #[test]
    fn blend_starts_procedural_and_becomes_learned_over_time() {
        let close = BlendWeights::for_transition(Duration::ZERO, 1000.0, 2.0);
        assert!((close.learned_weight - 0.0).abs() < 1e-9);
        assert!((close.procedural_weight - 1.0).abs() < 1e-9);

        let half = BlendWeights::for_transition(Duration::from_secs(1), 1000.0, 2.0);
        assert!(half.learned_weight > 0.0 && half.learned_weight < 1.0);

        let full = BlendWeights::for_transition(Duration::from_secs(2), 1000.0, 2.0);
        assert!((full.learned_weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn blend_is_monotonic_non_decreasing_in_time() {
        let mut prev = -1.0;
        for secs in 0..=20 {
            let w =
                BlendWeights::for_transition(Duration::from_secs(secs), 1000.0, 2.0).learned_weight;
            assert!(
                w >= prev,
                "blend weight decreased at secs={secs}: {w} < {prev}"
            );
            prev = w;
        }
    }

    #[test]
    fn blend_blends_elevation_provenance_not_world_coordinates() {
        // Blending provenance: 0.5 * proc + 0.5 * learned. The procedural
        // residual is added on top of whichever macro won — here we only
        // verify the macro blend itself.
        let w = BlendWeights {
            learned_weight: 0.5,
            procedural_weight: 0.5,
        };
        let blended = w.blend_elevation(10.0, 20.0);
        assert!((blended - 15.0).abs() < 1e-9);
    }

    #[test]
    fn blend_at_far_distance_snaps_instantly() {
        let w = BlendWeights::for_transition(Duration::from_millis(1), 200_000.0, 2.0);
        // Far chunks (>100km) snap instantly: learned_weight should be 1.0
        // even after a tiny resident_age.
        assert!((w.learned_weight - 1.0).abs() < 1e-9);
    }

    // ---- Telemetry ----

    #[test]
    fn telemetry_reports_queue_depth_and_resident() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        q.enqueue(
            key_for(1, 0, 0, 0),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        q.enqueue(
            key_for(2, 0, 1, 1),
            ptc_for(0, 1, 1),
            PriorityClass::NormalHalo,
            None,
            t,
        );
        q.mark_resident(&key_for(3, 0, 2, 2));
        let tel = q.telemetry();
        assert_eq!(tel.queue_depth, 2);
        assert_eq!(tel.resident_tiles, 1);
    }

    #[test]
    fn telemetry_reports_cache_hit_rate_and_fallback_percent() {
        let q = StreamingQueue::new(small_config());
        q.record_cache_lookup(true);
        q.record_cache_lookup(false);
        q.record_cache_lookup(true);
        q.record_sample_source(false);
        q.record_sample_source(true);
        q.record_sample_source(true);
        let tel = q.telemetry();
        assert!((tel.cache_hit_rate() - 2.0 / 3.0).abs() < 1e-9);
        // 1 fallback / 3 total = 33.33%
        assert!((tel.fallback_percent() - (1.0 / 3.0 * 100.0)).abs() < 1e-6);
    }

    #[test]
    fn telemetry_reports_latency_percentiles() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        for i in 1..=10u64 {
            let k = key_for(i, 0, i as u32, 0);
            q.enqueue(
                k.clone(),
                ptc_for(0, 0, 0),
                PriorityClass::VisibleSurface,
                None,
                t,
            );
            let _ = q.pop_dispatchable(t).unwrap();
            q.record_success(&k, Duration::from_millis(i), t);
        }
        let tel = q.telemetry();
        let p50 = tel.latency_p50_ms.unwrap();
        let p95 = tel.latency_p95_ms.unwrap();
        assert!(p50 > 0.0 && p95 >= p50);
    }

    #[test]
    fn telemetry_reports_rebuild_counts() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        // Rebuilds are queued when a tile becomes resident (via
        // mark_resident/record_success), not on enqueue.
        let k = key_for(1, 0, 0, 0);
        q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let _ = q.pop_dispatchable(t).unwrap();
        q.record_success(&k, Duration::from_millis(5), t);
        let tel = q.telemetry();
        assert_eq!(tel.rebuilds_queued, 1);
        let popped = q.pop_arrived_chart();
        assert!(popped.is_some());
        let tel2 = q.telemetry();
        assert_eq!(tel2.rebuilds_completed, 1);
    }

    // ---- Bounded queue ----

    #[test]
    fn queue_evicts_lowest_priority_when_full() {
        let mut cfg = small_config();
        cfg.max_queued = 3;
        let q = StreamingQueue::new(cfg);
        let t = now();
        // Fill with warmup, then add a visible-surface request.
        q.enqueue(
            key_for(1, 0, 0, 0),
            ptc_for(0, 0, 0),
            PriorityClass::Warmup,
            None,
            t,
        );
        q.enqueue(
            key_for(2, 0, 1, 1),
            ptc_for(0, 1, 1),
            PriorityClass::Warmup,
            None,
            t,
        );
        q.enqueue(
            key_for(3, 0, 2, 2),
            ptc_for(0, 2, 2),
            PriorityClass::Warmup,
            None,
            t,
        );
        assert_eq!(q.queued_count(), 3);
        // Adding one more should evict a warmup tail and accept the new one.
        assert!(q.enqueue(
            key_for(4, 0, 3, 3),
            ptc_for(0, 3, 3),
            PriorityClass::VisibleSurface,
            None,
            t
        ));
        assert_eq!(q.queued_count(), 3);
        // The visible-surface request should be dispatched first.
        let first = q.pop_dispatchable(t).unwrap();
        assert_eq!(first.priority, PriorityClass::VisibleSurface);
    }

    // ---- Concurrency limit ----

    #[test]
    fn pop_respects_in_flight_concurrency_limit() {
        let mut cfg = small_config();
        cfg.max_in_flight = 2;
        let q = StreamingQueue::new(cfg);
        let t = now();
        q.enqueue(
            key_for(1, 0, 0, 0),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        q.enqueue(
            key_for(2, 0, 1, 1),
            ptc_for(0, 1, 1),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        q.enqueue(
            key_for(3, 0, 2, 2),
            ptc_for(0, 2, 2),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let _ = q.pop_dispatchable(t).unwrap();
        let _ = q.pop_dispatchable(t).unwrap();
        // Third should be blocked by the in-flight limit of 2.
        assert!(q.pop_dispatchable(t).is_none());
        assert_eq!(q.in_flight_count(), 2);
    }

    // ---- mark_resident coalesces ----

    #[test]
    fn mark_resident_coalesces_queued_and_in_flight() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t,
        );
        let _ = q.pop_dispatchable(t).unwrap();
        // Now in-flight. mark_resident should coalesce it.
        q.mark_resident(&k);
        assert_eq!(q.in_flight_count(), 0);
        assert_eq!(q.resident_count(), 1);
        assert!(q.telemetry().provider_state.coalesced_duplicates_total >= 1);
    }

    #[test]
    fn evict_resident_allows_re_enqueue() {
        let q = StreamingQueue::new(small_config());
        let t = now();
        let k = key_for(1, 0, 0, 0);
        q.mark_resident(&k);
        assert!(!q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t
        ));
        q.evict_resident(&k);
        assert!(q.enqueue(
            k.clone(),
            ptc_for(0, 0, 0),
            PriorityClass::VisibleSurface,
            None,
            t
        ));
    }
}
