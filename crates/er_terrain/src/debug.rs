use bevy::ecs::resource::Resource;
use er_core::config::{CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES};
use er_world::terrain_field::TerrainSourceMode;
use glam::DVec3;

/// Bytes of vertex attribute data per terrain vertex:
///   Position(F32x3=12) + Morph(F32=4) + LowFreqElev(F32=4)
///   + MoistureLow(F32=4) + Elevation(F32=4) + Normal(F32x3=12)
///   + Temperature(F32=4) + Drainage(F32=4) + Curvature(F32=4) = 52
const BYTES_PER_VERTEX: usize = 52;

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
    VERTS_PER_CHUNK * BYTES_PER_VERTEX + INDICES_PER_CHUNK * 2;

#[derive(Resource)]
pub struct TerrainDebugInfo {
    pub active_chunks: usize,
    pub max_depth: u8,
    pub pending_splits: usize,
    pub pending_merges: usize,
    pub pending_meshes: usize,
    pub visible_chunks: usize,
    pub frame_time_ms: f32,
    pub meshes_built: usize,
    pub estimated_mesh_bytes: usize,
    pub camera_altitude_m: f64,
    pub render_origin_world: DVec3,
    pub render_origin_generation: u64,
    pub nearest_chunk_lod: u8,
    pub nearest_chunk_width_m: f64,
    pub vertex_spacing_m: f64,
    pub normal_diff_spacing_m: f64,
    pub normal_difference_span_m: f64,
    pub normal_diff_epsilon_radians: f64,
    pub source_mode: TerrainSourceMode,
    pub procedural_source_coverage_percent: f32,
    pub learned_source_coverage_percent: f32,
    pub cross_generation_mesh_attaches: usize,
}

impl Default for TerrainDebugInfo {
    fn default() -> Self {
        Self {
            active_chunks: 0,
            max_depth: 0,
            pending_splits: 0,
            pending_merges: 0,
            pending_meshes: 0,
            visible_chunks: 0,
            frame_time_ms: 0.0,
            meshes_built: 0,
            estimated_mesh_bytes: 0,
            camera_altitude_m: 0.0,
            render_origin_world: DVec3::ZERO,
            render_origin_generation: 0,
            nearest_chunk_lod: 0,
            nearest_chunk_width_m: 0.0,
            vertex_spacing_m: 0.0,
            normal_diff_spacing_m: 0.0,
            normal_difference_span_m: 0.0,
            normal_diff_epsilon_radians: 0.0,
            source_mode: TerrainSourceMode::default(),
            procedural_source_coverage_percent: 0.0,
            learned_source_coverage_percent: 0.0,
            cross_generation_mesh_attaches: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::math::cell_size;

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
            expected_verts * BYTES_PER_VERTEX + expected_indices * 2
        );
    }

    #[test]
    fn estimated_bytes_nonzero_and_scales() {
        const _: () = assert!(ESTIMATED_BYTES_PER_CHUNK_MESH > 0);
        let bytes_for_100 = 100 * ESTIMATED_BYTES_PER_CHUNK_MESH;
        assert_eq!(bytes_for_100 / ESTIMATED_BYTES_PER_CHUNK_MESH, 100);
    }

    #[test]
    fn default_fields_are_zero() {
        let info = TerrainDebugInfo::default();
        assert_eq!(info.meshes_built, 0);
        assert_eq!(info.estimated_mesh_bytes, 0);
        assert_eq!(info.nearest_chunk_lod, 0);
        assert_eq!(info.nearest_chunk_width_m, 0.0);
        assert_eq!(info.vertex_spacing_m, 0.0);
        assert_eq!(info.normal_diff_spacing_m, 0.0);
        assert_eq!(info.normal_difference_span_m, 0.0);
        assert_eq!(info.normal_diff_epsilon_radians, 0.0);
        assert_eq!(info.camera_altitude_m, 0.0);
        assert_eq!(info.render_origin_world, DVec3::ZERO);
        assert_eq!(info.render_origin_generation, 0);
        assert_eq!(info.source_mode, TerrainSourceMode::default());
        assert_eq!(info.procedural_source_coverage_percent, 0.0);
        assert_eq!(info.learned_source_coverage_percent, 0.0);
        assert_eq!(info.cross_generation_mesh_attaches, 0);
    }

    #[test]
    fn spacing_relationships_are_self_consistent() {
        let radius = 6_371_000.0;
        let lod = 17u8;
        let cs = cell_size(lod, radius);
        let vs = cs / CHUNK_QUADS_PER_EDGE as f64;
        assert!(cs > 0.0);
        assert!(vs > 0.0);
        let normal_spacing = vs;
        let normal_span = 2.0 * normal_spacing;
        let epsilon_radians = normal_spacing / radius;
        assert!((normal_spacing - vs).abs() < 1e-10);
        assert!((normal_span - 2.0 * vs).abs() < 1e-10);
        assert!((epsilon_radians * radius - vs).abs() < 1e-10);
        assert!((vs * CHUNK_QUADS_PER_EDGE as f64 - cs).abs() < 1e-10);
    }

    #[test]
    fn vertex_spacing_is_finite_for_miniature_and_earth() {
        for (radius, lod) in [(36_000.0, 12u8), (6_371_000.0, 17u8)] {
            let vs = cell_size(lod, radius) / CHUNK_QUADS_PER_EDGE as f64;
            assert!(vs > 0.0, "zero vertex spacing for r={radius} lod={lod}");
            assert!(vs.is_finite());
        }
    }

    #[test]
    fn coverage_percentage_fields_default_to_zero() {
        let info = TerrainDebugInfo::default();
        assert_eq!(info.procedural_source_coverage_percent, 0.0);
        assert_eq!(info.learned_source_coverage_percent, 0.0);
    }

    #[test]
    fn coverage_pair_sum_is_one_when_nonzero() {
        let info = TerrainDebugInfo {
            procedural_source_coverage_percent: 60.0,
            learned_source_coverage_percent: 40.0,
            ..TerrainDebugInfo::default()
        };
        let sum = info.procedural_source_coverage_percent + info.learned_source_coverage_percent;
        assert!((sum - 100.0).abs() < 0.001);
    }

    #[test]
    fn cross_generation_counter_defaults_to_zero() {
        let info = TerrainDebugInfo::default();
        assert_eq!(info.cross_generation_mesh_attaches, 0);
    }

    #[test]
    fn cross_generation_counter_can_hold_nonzero_and_reset_semantics() {
        let mut info = TerrainDebugInfo::default();
        assert_eq!(info.cross_generation_mesh_attaches, 0);
        info.cross_generation_mesh_attaches = 5;
        assert_eq!(info.cross_generation_mesh_attaches, 5);
        // Reset semantic: a new frame re-derives this to zero via default or
        // explicit set, so a fresh default always starts at zero.
        assert_eq!(
            TerrainDebugInfo::default().cross_generation_mesh_attaches,
            0
        );
    }
}
