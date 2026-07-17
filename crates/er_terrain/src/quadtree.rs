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
        CellKey {
            face: key.face,
            i,
            j,
            lod,
        },
        CellKey {
            face: key.face,
            i: i + 1,
            j,
            lod,
        },
        CellKey {
            face: key.face,
            i,
            j: j + 1,
            lod,
        },
        CellKey {
            face: key.face,
            i: i + 1,
            j: j + 1,
            lod,
        },
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
    coarser_neighbor_across_edge(key, side, active_chunks).map_or(key.lod, |neighbor| neighbor.lod)
}

/// Returns the active coarser leaf whose boundary touches `key` across `side`.
/// Same-LOD and finer neighbors require no stitching from this chunk.
pub fn coarser_neighbor_across_edge(
    key: CellKey,
    side: NeighborSide,
    active_chunks: &ActiveChunks,
) -> Option<CellKey> {
    let mut current = key;
    loop {
        let nb = cell_neighbor(current, side);
        if active_chunks.contains(&nb) {
            return (nb.lod < key.lod).then_some(nb);
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
            return None;
        }
        match parent_of(current) {
            Some(parent) => current = parent,
            None => return None,
        }
    }
}

pub fn root_chunks() -> Vec<CellKey> {
    (0..6)
        .map(|face| CellKey {
            face,
            i: 0,
            j: 0,
            lod: 0,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_cover_each_cube_face_once() {
        let roots = root_chunks();
        assert_eq!(roots.len(), 6);
        for (face, root) in roots.iter().enumerate() {
            assert_eq!(root.face, face as u8);
            assert_eq!(root.lod, 0);
        }
    }

    #[test]
    fn children_and_parent_form_a_split_merge_roundtrip() {
        let parent = CellKey {
            face: 3,
            i: 4,
            j: 7,
            lod: 5,
        };
        let children = children_of(parent);
        for child in children {
            assert_eq!(parent_of(child), Some(parent));
            assert_eq!(child.lod, parent.lod + 1);
        }
        assert_eq!(parent_of(root_chunks()[0]), None);
    }
}

#[cfg(test)]
mod deterministic_tests {
    use super::*;
    use er_core::math::{cell_neighbor, NeighborSide};
    use std::collections::HashSet;

    fn key(i: u32, j: u32, lod: u8) -> CellKey {
        CellKey { face: 0, i, j, lod }
    }

    #[test]
    fn children_round_trip_to_their_parent_and_are_unique() {
        let parent = key(3, 5, 4);
        let children = children_of(parent);
        assert_eq!(children.iter().copied().collect::<HashSet<_>>().len(), 4);
        for child in children {
            assert_eq!(parent_of(child), Some(parent));
            assert_eq!(child.lod, parent.lod + 1);
        }
        assert_eq!(parent_of(key(0, 0, 0)), None);
    }

    #[test]
    fn roots_cover_each_face_exactly_once() {
        let roots = root_chunks();
        assert_eq!(roots.len(), 6);
        assert_eq!(
            roots
                .iter()
                .map(|key| key.face)
                .collect::<HashSet<_>>()
                .len(),
            6
        );
        assert!(roots
            .iter()
            .all(|key| key.i == 0 && key.j == 0 && key.lod == 0));
    }

    #[test]
    fn finds_a_coarser_neighbor_across_a_parent_boundary() {
        let child = key(0, 2, 3);
        let parent = parent_of(child).unwrap();
        let neighbor = cell_neighbor(parent, NeighborSide::NegU);
        let mut active = ActiveChunks::default();
        active.chunks.insert(neighbor, Entity::PLACEHOLDER);

        assert_eq!(
            neighbor_lod_across_edge(child, NeighborSide::NegU, &active),
            parent.lod
        );
    }

    #[test]
    fn inner_child_does_not_search_for_a_coarser_neighbor() {
        let child = key(1, 2, 3);
        let parent = parent_of(child).unwrap();
        let mut active = ActiveChunks::default();
        active.chunks.insert(
            cell_neighbor(parent, NeighborSide::NegU),
            Entity::PLACEHOLDER,
        );

        assert_eq!(
            neighbor_lod_across_edge(child, NeighborSide::NegU, &active),
            child.lod
        );
    }
}
