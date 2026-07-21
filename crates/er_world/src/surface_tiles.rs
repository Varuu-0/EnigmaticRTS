//! Stable cube-face storage for resident learned macro-terrain tiles.
//!
//! This module intentionally stores only validated, already-generated data.
//! Sidecar I/O and inference scheduling must populate it from a background
//! worker, never from terrain mesh generation.

use crate::terrain_field::{MacroTerrainField, MacroTerrainSample};
use er_core::math::dir_to_uv;
use glam::DVec3;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Metadata that changes the meaning or reproducibility of a learned tile set.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LearnedTileGeneration {
    pub model_revision: String,
    pub seed: u64,
    pub projection_revision: u16,
    pub pixel_scale_m: u16,
    pub sea_level_datum_m: i16,
}

/// A coordinate in one of the six spherified-cube faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileCoordinate {
    pub face: u8,
    pub x: u16,
    pub y: u16,
}

/// Full persistent identity of a learned tile. The cache keeps a single
/// generation at a time, but records retain all generation-affecting metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LearnedTileKey {
    pub generation: LearnedTileGeneration,
    pub coordinate: TileCoordinate,
}

/// Validated elevation payload for one face-local learned tile. `elevation_m`
/// is row-major signed meters and includes a pre-generated halo around its
/// `core_resolution` by `halo` samples per side.
#[derive(Clone, Debug)]
pub struct LearnedTerrainTile {
    pub key: LearnedTileKey,
    pub core_resolution: u16,
    pub halo: u16,
    pub elevation_m: Arc<[i16]>,
}

impl LearnedTerrainTile {
    pub fn stored_resolution(&self) -> u16 {
        self.core_resolution
            .saturating_add(self.halo.saturating_mul(2))
    }

    fn validate(&self) -> Result<(), TileInsertError> {
        if self.key.coordinate.face >= 6 {
            return Err(TileInsertError::InvalidFace(self.key.coordinate.face));
        }
        if self.core_resolution < 2 {
            return Err(TileInsertError::CoreResolutionTooSmall(
                self.core_resolution,
            ));
        }
        let Some(halo_width) = self.halo.checked_mul(2) else {
            return Err(TileInsertError::StoredResolutionOverflow);
        };
        if self.core_resolution.checked_add(halo_width).is_none() {
            return Err(TileInsertError::StoredResolutionOverflow);
        }
        let stored = self.stored_resolution() as usize;
        let expected = stored * stored;
        if self.elevation_m.len() != expected {
            return Err(TileInsertError::PayloadLength {
                expected,
                actual: self.elevation_m.len(),
            });
        }
        Ok(())
    }

    fn sample_elevation_m(&self, u: f64, v: f64) -> f64 {
        let stored = self.stored_resolution() as usize;
        let core_max = (self.core_resolution - 1) as f64;
        let x = self.halo as f64 + u.clamp(0.0, 1.0) * core_max;
        let y = self.halo as f64 + v.clamp(0.0, 1.0) * core_max;
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(stored - 1);
        let y1 = (y0 + 1).min(stored - 1);
        let tx = x - x0 as f64;
        let ty = y - y0 as f64;
        let at = |x: usize, y: usize| self.elevation_m[y * stored + x] as f64;
        let lower = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
        let upper = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
        lower + (upper - lower) * ty
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileInsertError {
    GenerationMismatch,
    InvalidFace(u8),
    CoordinateOutOfRange(TileCoordinate),
    CoreResolutionMismatch { expected: u16, actual: u16 },
    CoreResolutionTooSmall(u16),
    StoredResolutionOverflow,
    PayloadLength { expected: usize, actual: usize },
}

impl fmt::Display for TileInsertError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GenerationMismatch => formatter.write_str("tile generation does not match cache"),
            Self::InvalidFace(face) => write!(formatter, "invalid cube face {face}"),
            Self::CoordinateOutOfRange(coordinate) => write!(
                formatter,
                "tile coordinate {}/{},{} is outside the configured face atlas",
                coordinate.face, coordinate.x, coordinate.y
            ),
            Self::CoreResolutionMismatch { expected, actual } => write!(
                formatter,
                "tile core resolution {actual} does not match cache resolution {expected}"
            ),
            Self::CoreResolutionTooSmall(resolution) => write!(
                formatter,
                "tile core resolution {resolution} must contain at least two samples"
            ),
            Self::StoredResolutionOverflow => {
                formatter.write_str("tile core resolution and halo exceed u16 storage")
            }
            Self::PayloadLength { expected, actual } => write!(
                formatter,
                "tile payload has {actual} samples; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for TileInsertError {}

/// Thread-safe resident cache for exactly one learned tile generation. The
/// reference model's meters are converted once here into the terrain field's
/// normalized elevation units using the renderer's global displacement scale.
pub struct LearnedTileCache {
    generation: LearnedTileGeneration,
    tiles_per_face_edge: u16,
    core_resolution: u16,
    elevation_scale_m: f64,
    tiles: RwLock<HashMap<TileCoordinate, Arc<LearnedTerrainTile>>>,
    revision: AtomicU64,
}

impl LearnedTileCache {
    pub fn new(
        generation: LearnedTileGeneration,
        tiles_per_face_edge: u16,
        core_resolution: u16,
        elevation_scale_m: f64,
    ) -> Self {
        assert!(
            tiles_per_face_edge > 0,
            "learned tile atlas must not be empty"
        );
        assert!(
            core_resolution >= 2,
            "learned tiles need at least two samples"
        );
        assert!(
            elevation_scale_m.is_finite() && elevation_scale_m > 0.0,
            "elevation scale must be finite and positive"
        );
        Self {
            generation,
            tiles_per_face_edge,
            core_resolution,
            elevation_scale_m,
            tiles: RwLock::new(HashMap::new()),
            revision: AtomicU64::new(0),
        }
    }

    pub fn generation(&self) -> &LearnedTileGeneration {
        &self.generation
    }

    pub fn len(&self) -> usize {
        self.tiles.read().map(|tiles| tiles.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn tile_key_for_direction(&self, dir: DVec3) -> LearnedTileKey {
        let (face, u, v) = dir_to_uv(dir);
        LearnedTileKey {
            generation: self.generation.clone(),
            coordinate: self.coordinate_for_uv(face, u, v),
        }
    }

    pub fn insert(&self, tile: LearnedTerrainTile) -> Result<(), TileInsertError> {
        tile.validate()?;
        if tile.key.generation != self.generation {
            return Err(TileInsertError::GenerationMismatch);
        }
        if tile.core_resolution != self.core_resolution {
            return Err(TileInsertError::CoreResolutionMismatch {
                expected: self.core_resolution,
                actual: tile.core_resolution,
            });
        }
        let coordinate = tile.key.coordinate;
        if coordinate.x >= self.tiles_per_face_edge || coordinate.y >= self.tiles_per_face_edge {
            return Err(TileInsertError::CoordinateOutOfRange(coordinate));
        }
        if let Ok(mut tiles) = self.tiles.write() {
            tiles.insert(coordinate, Arc::new(tile));
            self.revision.fetch_add(1, Ordering::Release);
        }
        Ok(())
    }

    pub fn contains(&self, coordinate: TileCoordinate) -> bool {
        self.tiles
            .read()
            .map(|tiles| tiles.contains_key(&coordinate))
            .unwrap_or(false)
    }

    fn coordinate_for_uv(&self, face: u8, u: f64, v: f64) -> TileCoordinate {
        TileCoordinate {
            face,
            x: face_tile_index(u, self.tiles_per_face_edge),
            y: face_tile_index(v, self.tiles_per_face_edge),
        }
    }
}

impl MacroTerrainField for LearnedTileCache {
    fn sample_resident(&self, dir: DVec3) -> Option<MacroTerrainSample> {
        let (face, u, v) = dir_to_uv(dir);
        let coordinate = self.coordinate_for_uv(face, u, v);
        let tiles = self.tiles.read().ok()?;
        let tile = tiles.get(&coordinate)?;
        let local_scale = self.tiles_per_face_edge as f64;
        let local_u = local_tile_coordinate(u, local_scale);
        let local_v = local_tile_coordinate(v, local_scale);
        let elevation_m = tile.sample_elevation_m(local_u, local_v);
        Some(MacroTerrainSample {
            elevation: (elevation_m - self.generation.sea_level_datum_m as f64)
                / self.elevation_scale_m,
        })
    }

    fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }
}

fn face_tile_index(value: f64, count: u16) -> u16 {
    let count_f64 = count as f64;
    (value.clamp(0.0, 1.0) * count_f64)
        .floor()
        .min(count_f64 - 1.0) as u16
}

fn local_tile_coordinate(value: f64, count: f64) -> f64 {
    let scaled = value.clamp(0.0, 1.0) * count;
    if scaled >= count {
        1.0
    } else {
        scaled - scaled.floor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::math::uv_to_dir;

    fn generation() -> LearnedTileGeneration {
        LearnedTileGeneration {
            model_revision: "fixture-v1".to_owned(),
            seed: 0xC0FFEE,
            projection_revision: 1,
            pixel_scale_m: 30,
            sea_level_datum_m: 0,
        }
    }

    fn tile(
        generation: LearnedTileGeneration,
        coordinate: TileCoordinate,
        resolution: u16,
        value: impl Fn(usize, usize) -> i16,
    ) -> LearnedTerrainTile {
        let mut elevation_m = Vec::with_capacity(resolution as usize * resolution as usize);
        for y in 0..resolution as usize {
            for x in 0..resolution as usize {
                elevation_m.push(value(x, y));
            }
        }
        LearnedTerrainTile {
            key: LearnedTileKey {
                generation,
                coordinate,
            },
            core_resolution: resolution,
            halo: 0,
            elevation_m: Arc::from(elevation_m),
        }
    }

    #[test]
    fn resident_tile_bilinearly_samples_elevation() {
        let cache = LearnedTileCache::new(generation(), 1, 3, 1000.0);
        let coordinate = TileCoordinate {
            face: 0,
            x: 0,
            y: 0,
        };
        cache
            .insert(tile(generation(), coordinate, 3, |x, y| {
                (x as i16 + y as i16 * 10) * 100
            }))
            .unwrap();

        let sample = cache.sample_resident(uv_to_dir(0, 0.25, 0.75)).unwrap();
        assert!((sample.elevation - 1.55).abs() < 1e-9);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.revision(), 1);
    }

    #[test]
    fn missing_tile_returns_none_without_blocking() {
        let cache = LearnedTileCache::new(generation(), 2, 3, 1000.0);
        assert!(cache.sample_resident(uv_to_dir(4, 0.2, 0.8)).is_none());
    }

    #[test]
    fn cube_face_boundary_uses_a_single_canonical_tile_key() {
        let cache = LearnedTileCache::new(generation(), 1, 3, 1000.0);
        let dir = uv_to_dir(0, 1.0, 0.5);
        let key = cache.tile_key_for_direction(dir);
        let repeated = cache.tile_key_for_direction(dir);

        assert_eq!(key, repeated);
        assert!(key.coordinate.face < 6);
    }

    #[test]
    fn rejects_tiles_with_wrong_generation_or_shape() {
        let cache = LearnedTileCache::new(generation(), 1, 3, 1000.0);
        let coordinate = TileCoordinate {
            face: 0,
            x: 0,
            y: 0,
        };
        let mut wrong_generation = generation();
        wrong_generation.seed += 1;
        assert_eq!(
            cache.insert(tile(wrong_generation, coordinate, 3, |_, _| 0)),
            Err(TileInsertError::GenerationMismatch)
        );

        let invalid = LearnedTerrainTile {
            key: LearnedTileKey {
                generation: generation(),
                coordinate,
            },
            core_resolution: 3,
            halo: 0,
            elevation_m: Arc::from(vec![0_i16; 8]),
        };
        assert_eq!(
            cache.insert(invalid),
            Err(TileInsertError::PayloadLength {
                expected: 9,
                actual: 8,
            })
        );
    }
}
