use crate::biome::Biome;
use er_core::math::{dir_to_cell, CellKey};
use glam::DVec3;
use std::collections::{BTreeSet, HashMap};
use std::sync::RwLock;

const CACHE_SHARDS: usize = 32;

/// Default cache LOD — ~0.86 m cells on the 36 km miniature planet. Used when
/// no preset supplies an explicit LOD.
pub const DEFAULT_CACHE_LOD: u8 = 16;

#[derive(Clone, Copy, Debug)]
pub struct CachedWorldData {
    pub elevation: f64,
    pub low_freq_elev: f32,
    pub warped_dir: [f32; 3],
    pub moisture: f32,
    pub biome: Biome,
    pub mountain_influence: f32,
    pub temperature: f32,
    pub drainage: f32,
}

pub struct WorldCache {
    shards: [CacheShard; CACHE_SHARDS],
    shard_count: usize,
    capacity_per_shard: usize,
    cache_lod: u8,
}

#[derive(Default)]
struct CacheShard {
    entries: RwLock<CellCache>,
    elevations: RwLock<ElevationCache>,
}

#[derive(Default)]
struct CellCache {
    values: HashMap<CellKey, CachedWorldData>,
    ordered_keys: BTreeSet<(u8, u8, u32, u32)>,
}

#[derive(Default)]
struct ElevationCache {
    values: HashMap<[u64; 3], f64>,
    ordered_keys: BTreeSet<[u64; 3]>,
}

impl WorldCache {
    pub fn new(capacity: usize) -> Self {
        Self::with_lod(capacity, DEFAULT_CACHE_LOD)
    }

    pub fn with_lod(capacity: usize, cache_lod: u8) -> Self {
        let shard_count = capacity.clamp(1, CACHE_SHARDS);
        let capacity_per_shard = capacity.div_ceil(shard_count).max(1);
        Self {
            shards: std::array::from_fn(|_| CacheShard {
                entries: RwLock::new(CellCache {
                    values: HashMap::with_capacity(capacity_per_shard),
                    ordered_keys: BTreeSet::new(),
                }),
                elevations: RwLock::new(ElevationCache {
                    values: HashMap::with_capacity(capacity_per_shard),
                    ordered_keys: BTreeSet::new(),
                }),
            }),
            shard_count,
            capacity_per_shard,
            cache_lod,
        }
    }

    pub fn get_or_insert(
        &self,
        dir: DVec3,
        compute: impl FnOnce() -> CachedWorldData,
    ) -> CachedWorldData {
        let key = dir_to_cell(dir, self.cache_lod);
        let shard = &self.shards[cell_shard(key, self.shard_count)];

        if let Ok(entries) = shard.entries.read() {
            if let Some(data) = entries.values.get(&key) {
                return *data;
            }
        }

        let data = compute();
        if let Ok(mut entries) = shard.entries.write() {
            if let Some(existing) = entries.values.get(&key) {
                return *existing;
            }

            if entries.values.len() >= self.capacity_per_shard {
                // Preserve stable eviction without scanning every hash-map key
                // while holding the shard's write lock.
                if let Some(ordered_key) = entries.ordered_keys.pop_first() {
                    entries.values.remove(&cell_key_from_ordered(ordered_key));
                }
            }

            entries.ordered_keys.insert(ordered_cell_key(key));
            entries.values.insert(key, data);
        }
        data
    }

    pub fn get_or_insert_elevation(&self, dir: DVec3, compute: impl FnOnce() -> f64) -> f64 {
        let key = [dir.x.to_bits(), dir.y.to_bits(), dir.z.to_bits()];
        let shard = &self.shards[elevation_shard(key, self.shard_count)];

        if let Ok(elevations) = shard.elevations.read() {
            if let Some(elevation) = elevations.values.get(&key) {
                return *elevation;
            }
        }

        let elevation = compute();
        if let Ok(mut elevations) = shard.elevations.write() {
            if let Some(existing) = elevations.values.get(&key) {
                return *existing;
            }
            if elevations.values.len() >= self.capacity_per_shard {
                if let Some(evicted_key) = elevations.ordered_keys.pop_first() {
                    elevations.values.remove(&evicted_key);
                }
            }
            elevations.ordered_keys.insert(key);
            elevations.values.insert(key, elevation);
        }
        elevation
    }

    pub fn contains(&self, dir: DVec3) -> bool {
        let key = dir_to_cell(dir, self.cache_lod);
        self.shards[cell_shard(key, self.shard_count)]
            .entries
            .read()
            .map(|e| e.values.contains_key(&key))
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.entries.read().map(|e| e.values.len()).unwrap_or(0))
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut entries) = shard.entries.write() {
                entries.values.clear();
                entries.ordered_keys.clear();
            }
            if let Ok(mut elevations) = shard.elevations.write() {
                elevations.values.clear();
                elevations.ordered_keys.clear();
            }
        }
    }
}

fn ordered_cell_key(key: CellKey) -> (u8, u8, u32, u32) {
    (key.face, key.lod, key.i, key.j)
}

fn cell_key_from_ordered((face, lod, i, j): (u8, u8, u32, u32)) -> CellKey {
    CellKey { face, i, j, lod }
}

fn cell_shard(key: CellKey, shard_count: usize) -> usize {
    let mixed = u64::from(key.i).wrapping_mul(0x9E37_79B1)
        ^ u64::from(key.j).wrapping_mul(0x85EB_CA77)
        ^ (u64::from(key.face) << 8)
        ^ u64::from(key.lod);
    mixed as usize % shard_count
}

fn elevation_shard(key: [u64; 3], shard_count: usize) -> usize {
    let mixed = key[0].wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ key[1].rotate_left(21)
        ^ key[2].rotate_left(42);
    mixed as usize % shard_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::math::uv_to_dir;

    #[test]
    fn cache_insert_and_retrieve() {
        let cache = WorldCache::new(100);
        let dir = uv_to_dir(0, 0.5, 0.5);
        assert!(!cache.contains(dir));
        let data = CachedWorldData {
            elevation: 0.42,
            low_freq_elev: 0.3,
            warped_dir: [1.0, 0.0, 0.0],
            moisture: 0.5,
            biome: Biome::Grassland,
            mountain_influence: 0.1,
            temperature: 0.5,
            drainage: 0.25,
        };
        let result = cache.get_or_insert(dir, || data);
        assert_eq!(result.elevation, 0.42);
        assert!(cache.contains(dir));
        let result2 = cache.get_or_insert(dir, || panic!("should be cached"));
        assert_eq!(result2.elevation, 0.42);
    }

    #[test]
    fn cache_eviction() {
        let cache = WorldCache::new(10);
        for i in 0..20u32 {
            let face = (i % 6) as u8;
            let u = ((i as f64 / 20.0) * 0.9 + 0.05).min(0.95);
            let v = (((i as f64 / 20.0) * 0.9 + 0.05) * 0.5 + 0.25).min(0.95);
            let dir = uv_to_dir(face, u, v);
            cache.get_or_insert(dir, || CachedWorldData {
                elevation: i as f64,
                low_freq_elev: 0.0,
                warped_dir: [0.0, 0.0, 0.0],
                moisture: 0.0,
                biome: Biome::OceanShallow,
                mountain_influence: 0.0,
                temperature: 0.0,
                drainage: 0.0,
            });
        }
        assert!(cache.len() <= 12);
    }

    #[test]
    fn exact_elevation_cache_distinguishes_nearby_directions() {
        let cache = WorldCache::new(100);
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(1.0, f64::EPSILON, 0.0).normalize();
        assert_eq!(cache.get_or_insert_elevation(a, || 1.0), 1.0);
        assert_eq!(cache.get_or_insert_elevation(b, || 2.0), 2.0);
        assert_eq!(cache.get_or_insert_elevation(a, || panic!("cached")), 1.0);
    }

    #[test]
    fn eviction_keeps_ordered_indexes_in_sync() {
        let cache = WorldCache::with_lod(1, 8);
        for i in 0..100u32 {
            let dir = uv_to_dir(0, (f64::from(i) + 0.5) / 100.0, 0.5);
            cache.get_or_insert(dir, || CachedWorldData {
                elevation: f64::from(i),
                low_freq_elev: 0.0,
                warped_dir: [0.0; 3],
                moisture: 0.0,
                biome: Biome::OceanShallow,
                mountain_influence: 0.0,
                temperature: 0.0,
                drainage: 0.0,
            });
            cache.get_or_insert_elevation(dir, || f64::from(i));
        }

        let entries = cache.shards[0].entries.read().unwrap();
        let elevations = cache.shards[0].elevations.read().unwrap();
        assert_eq!(entries.values.len(), 1);
        assert_eq!(entries.ordered_keys.len(), entries.values.len());
        assert_eq!(elevations.values.len(), 1);
        assert_eq!(elevations.ordered_keys.len(), elevations.values.len());
    }
}
