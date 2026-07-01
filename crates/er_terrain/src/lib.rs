//! er_terrain: heightfield quadtree LOD, chunk mesh gen, skirts/edge-stitch,
//! culling, and the aggressive LOD controller. (Phase 3)

pub mod chunk;
pub mod culling;
pub mod debug;
pub mod lod;
pub mod material;
pub mod mesh_gen;
pub mod quadtree;
pub mod systems;

pub use chunk::ChunkComponent;
pub use debug::TerrainDebugInfo;
pub use material::{TerrainMaterial, TerrainMaterialUniform};
pub use mesh_gen::{generate_chunk_mesh, ATTRIBUTE_MORPH};
pub use quadtree::{children_of, parent_of, root_chunks, ActiveChunks};
pub use systems::{spawn_chunk_at, TerrainPlugin, TerrainState};

pub fn version() -> &'static str {
    "0"
}
