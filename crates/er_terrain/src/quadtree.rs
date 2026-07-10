use bevy::ecs::entity::Entity;
use bevy::ecs::resource::Resource;
use er_core::math::{cell_neighbor, CellKey, NeighborSide};
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

/// Find the LOD of the active chunk adjacent across `side` of `key`. Walks up
/// the quadtree from `key` toward coarser ancestors: at each level it checks
/// the same-level neighbor; the first active match is the coarsest neighbor
/// across that edge (the one the stitch must collapse to). If no coarser
/// neighbor is found, returns `key.lod` — either the neighbor is same/finer
/// (handled from the other side) or there is none.
pub fn neighbor_lod_across_edge(
    key: CellKey,
    side: NeighborSide,
    active_chunks: &ActiveChunks,
) -> u8 {
    let mut current = key;
    loop {
        let nb = cell_neighbor(current, side);
        if active_chunks.contains(&nb) {
            return nb.lod;
        }
        // Only ascend when current is on the queried edge of its parent.
        // If current is an inner child, the neighbor across this edge is a
        // sibling (same/finer), not a coarser chunk — no stitch needed.
        let on_parent_edge = match side {
            NeighborSide::NegU => current.i % 2 == 0,
            NeighborSide::PosU => current.i % 2 == 1,
            NeighborSide::NegV => current.j % 2 == 0,
            NeighborSide::PosV => current.j % 2 == 1,
        };
        if !on_parent_edge {
            return key.lod;
        }
        match parent_of(current) {
            Some(parent) => current = parent,
            None => return key.lod,
        }
    }
}

pub fn root_chunks() -> Vec<CellKey> {
    (0..6)
        .map(|face| CellKey { face, i: 0, j: 0, lod: 0 })
        .collect()
}

/// A parent chunk kept alive as a visible fallback while its four children's
/// meshes generate. The parent is removed from `ActiveChunks` (so the LOD
/// controller stops re-evaluating it) but stays rendered until every child has
/// a mesh, at which point `finalize_retirements` despawns it and reveals the
/// children atomically.
pub struct RetainedSplit {
    pub parent_entity: Entity,
    pub children: [Entity; 4],
}

#[derive(Resource, Default)]
pub struct RetainedSplits {
    pub map: HashMap<CellKey, RetainedSplit>,
}

/// The four children of a parent that is being merged back to a coarser LOD.
/// The children stay rendered as the visible fallback until the new parent's
/// mesh is ready; then `finalize_retained_merges` reveals the parent and despawns
/// the children atomically. This avoids the 1–2 frame black gap that a plain
/// non-retained merge creates.
pub struct RetainedMerge {
    pub parent_key: CellKey,
    pub parent_entity: Entity,
    pub children: [Entity; 4],
}

#[derive(Resource, Default)]
pub struct RetainedMerges {
    pub map: HashMap<CellKey, RetainedMerge>,
}
