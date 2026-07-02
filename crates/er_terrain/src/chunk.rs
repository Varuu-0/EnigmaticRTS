use bevy::ecs::component::Component;
use er_core::math::CellKey;

#[derive(Component, Clone, Debug)]
pub struct ChunkComponent {
    pub key: CellKey,
    pub neighbor_depth: [u8; 4],
}

impl ChunkComponent {
    pub fn new(key: CellKey) -> Self {
        Self {
            key,
            neighbor_depth: [key.lod; 4],
        }
    }
}
