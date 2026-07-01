use bevy::ecs::component::Component;
use er_core::math::CellKey;

#[derive(Component, Clone, Debug)]
pub struct ChunkComponent {
    pub key: CellKey,
}

impl ChunkComponent {
    pub fn new(key: CellKey) -> Self {
        Self { key }
    }
}
