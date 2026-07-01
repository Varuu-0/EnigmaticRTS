use bevy::ecs::component::Component;
use er_core::math::{cell_neighbors, CellKey};

#[derive(Component, Clone, Debug)]
pub struct ChunkComponent {
    pub key: CellKey,
    pub neighbors: [CellKey; 4],
    pub neighbor_depth: [u8; 4],
}

impl ChunkComponent {
    pub fn new(key: CellKey) -> Self {
        Self {
            neighbors: cell_neighbors(key),
            key,
            neighbor_depth: [key.lod; 4],
        }
    }
}
