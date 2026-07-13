use bevy::ecs::resource::Resource;
use er_core::config::{CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES};

/// Bytes of vertex attribute data per terrain vertex:
///   Position(F32x3=12) + Morph(F32=4) + Grid(U32x2=8) + LowFreqElev(F32=4)
///   + WarpedDir(F32x3=12) + MoistureLow(F32=4) + Elevation(F32=4)
///   + Normal(F32x3=12) + Temperature(F32=4) = 64
const BYTES_PER_VERTEX: usize = 64;

const VERTS_PER_CHUNK: usize =
    (CHUNK_VERT_RES as usize) * (CHUNK_VERT_RES as usize) + 4 * (CHUNK_VERT_RES as usize);

const INDICES_PER_CHUNK: usize =
    (CHUNK_QUADS_PER_EDGE as usize) * (CHUNK_QUADS_PER_EDGE as usize) * 6
        + 4 * (CHUNK_QUADS_PER_EDGE as usize) * 6;

/// Estimated GPU geometry bytes for a single terrain chunk mesh: vertex
/// attribute data plus 32-bit indices.  Derived from `CHUNK_VERT_RES`,
/// `CHUNK_QUADS_PER_EDGE`, and the fixed set of vertex attributes inserted by
/// `generate_chunk_mesh`.
pub const ESTIMATED_BYTES_PER_CHUNK_MESH: usize =
    VERTS_PER_CHUNK * BYTES_PER_VERTEX + INDICES_PER_CHUNK * 4;

#[derive(Resource, Default)]
pub struct TerrainDebugInfo {
    pub active_chunks: usize,
    pub max_depth: u8,
    pub pending_splits: usize,
    pub pending_merges: usize,
    pub pending_meshes: usize,
    pub visible_chunks: usize,
    pub frame_time_ms: f32,
    /// Chunk meshes that completed async generation and were applied to
    /// entities this frame.
    pub meshes_built: usize,
    /// Estimated total geometry bytes for all terrain chunk meshes currently
    /// alive (meshed chunk count x per-mesh estimate).
    pub estimated_mesh_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimated_bytes_match_vertex_and_index_counts() {
        let n = CHUNK_VERT_RES as usize;
        let quads = CHUNK_QUADS_PER_EDGE as usize;
        let expected_verts = n * n + 4 * n;
        let expected_indices = quads * quads * 6 + 4 * quads * 6;
        assert_eq!(VERTS_PER_CHUNK, expected_verts);
        assert_eq!(INDICES_PER_CHUNK, expected_indices);
        assert_eq!(
            ESTIMATED_BYTES_PER_CHUNK_MESH,
            expected_verts * BYTES_PER_VERTEX + expected_indices * 4
        );
    }

    #[test]
    fn default_new_fields_are_zero() {
        let info = TerrainDebugInfo::default();
        assert_eq!(info.meshes_built, 0);
        assert_eq!(info.estimated_mesh_bytes, 0);
    }

    #[test]
    fn estimated_bytes_nonzero_and_scales() {
        assert!(ESTIMATED_BYTES_PER_CHUNK_MESH > 0);
        let bytes_for_100 = 100 * ESTIMATED_BYTES_PER_CHUNK_MESH;
        assert_eq!(bytes_for_100 / ESTIMATED_BYTES_PER_CHUNK_MESH, 100);
    }
}
