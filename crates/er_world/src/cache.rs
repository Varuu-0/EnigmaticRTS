use crate::biome::Biome;
use er_core::math::{dir_to_cell, CellKey};
use glam::DVec3;
use std::collections::HashMap;
use std::sync::RwLock;

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
}

pub struct WorldCache {
    entries: RwLock<HashMap<CellKey, CachedWorldData>>,
    capacity: usize,
    cache_lod: u8,
}

impl WorldCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            cache_lod: DEFAULT_CACHE_LOD,
        }
    }

    pub fn with_lod(capacity: usize, cache_lod: u8) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            cache_lod,
        }
    }

    pub fn get_or_insert(
        &self,
        dir: DVec3,
        compute: impl FnOnce() -> CachedWorldData,
    ) -> CachedWorldData {
        let key = dir_to_cell(dir, self.cache_lod);

        if let Ok(entries) = self.entries.read() {
            if let Some(data) = entries.get(&key) {
                return *data;
            }
        }

        let data = compute();
        if let Ok(mut entries) = self.entries.write() {
            if let Some(existing) = entries.get(&key) {
                return *existing;
            }

            if entries.len() >= self.capacity {
                let to_remove = entries.len() - self.capacity + self.capacity / 10;
                let keys: Vec<CellKey> = entries.keys().take(to_remove).copied().collect();
                for k in keys {
                    entries.remove(&k);
                }
            }

            entries.insert(key, data);
        }
        data
    }

    pub fn contains(&self, dir: DVec3) -> bool {
        let key = dir_to_cell(dir, self.cache_lod);
        self.entries
            .read()
            .map(|e| e.contains_key(&key))
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
        }
    }
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
            });
        }
        assert!(cache.len() <= 12);
    }
}
