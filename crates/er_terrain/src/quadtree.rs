use bevy::ecs::entity::Entity;
use bevy::ecs::resource::Resource;
use er_core::math::CellKey;
use std::collections::HashMap;

#[derive(Resource)]
pub struct ActiveChunks {
    pub chunks: HashMap<CellKey, Entity>,
    pub pending_splits: Vec<CellKey>,
    pub pending_merges: Vec<CellKey>,
}

impl ActiveChunks {
    pub fn insert(&mut self, key: CellKey, entity: Entity) {
        self.chunks.insert(key, entity);
    }

    pub fn remove(&mut self, key: &CellKey) -> Option<Entity> {
        self.chunks.remove(key)
    }

    pub fn contains(&self, key: &CellKey) -> bool {
        self.chunks.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &CellKey> {
        self.chunks.keys()
    }

    pub fn clear_pending(&mut self) {
        self.pending_splits.clear();
        self.pending_merges.clear();
    }
}

impl Default for ActiveChunks {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            pending_splits: Vec::new(),
            pending_merges: Vec::new(),
        }
    }
}

pub fn children_of(key: CellKey) -> [CellKey; 4] {
    let lod = key.lod + 1;
    let i = key.i * 2;
    let j = key.j * 2;
    [
        CellKey { face: key.face, i, j, lod },
        CellKey { face: key.face, i: i + 1, j, lod },
        CellKey { face: key.face, i, j: j + 1, lod },
        CellKey { face: key.face, i: i + 1, j: j + 1, lod },
    ]
}

pub fn parent_of(key: CellKey) -> Option<CellKey> {
    if key.lod == 0 {
        return None;
    }
    Some(CellKey {
        face: key.face,
        i: key.i / 2,
        j: key.j / 2,
        lod: key.lod - 1,
    })
}

pub fn root_chunks() -> Vec<CellKey> {
    (0..6)
        .map(|face| CellKey { face, i: 0, j: 0, lod: 0 })
        .collect()
}
