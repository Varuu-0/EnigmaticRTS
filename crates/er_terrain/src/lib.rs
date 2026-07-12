//! er_terrain: heightfield quadtree LOD, chunk mesh gen, skirts/edge-stitch,
//! culling, and the aggressive LOD controller. (Phase 3)

pub mod chunk;
pub mod culling;
pub mod debug;
pub mod lod;
pub mod material;
pub mod mesh_gen;
pub mod ocean;
pub mod profiler;
pub mod quadtree;
pub mod systems;

pub use chunk::{ChunkComponent, HoldForMerge, HoldHidden};
pub use debug::TerrainDebugInfo;
pub use material::{TerrainMaterial, TerrainMaterialUniform};
pub use mesh_gen::{
    generate_chunk_mesh, ATTRIBUTE_ELEVATION, ATTRIBUTE_GRID, ATTRIBUTE_LOW_FREQ_ELEV,
    ATTRIBUTE_MOISTURE_LOW, ATTRIBUTE_MORPH, ATTRIBUTE_NORMAL, ATTRIBUTE_TEMPERATURE,
    ATTRIBUTE_WARPED_DIR,
};
pub use profiler::FrameProfiler;
pub use quadtree::{
    children_of, parent_of, root_chunks, ActiveChunks, RetainedMerge, RetainedMerges,
    RetainedSplit, RetainedSplits,
};
pub use systems::{PendingChunkMeshes, SunDirection, TerrainPlugin, TerrainState, TerrainUpdate};

pub fn version() -> &'static str {
    "0"
}
