use bevy::ecs::resource::Resource;

#[derive(Resource, Default)]
pub struct TerrainDebugInfo {
    pub active_chunks: usize,
    pub max_depth: u8,
    pub pending_splits: usize,
    pub pending_merges: usize,
    pub visible_chunks: usize,
}
