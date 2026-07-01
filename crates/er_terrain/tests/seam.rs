use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use er_core::config::{CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES};
use er_core::math::{cells_per_edge, uv_to_dir, CellKey};
use er_terrain::{generate_chunk_mesh, ATTRIBUTE_MORPH};

#[test]
fn chunk_mesh_vertex_and_index_counts() {
    let key = CellKey { face: 0, i: 0, j: 0, lod: 0 };
    let mesh = generate_chunk_mesh(key, 12000.0);

    let n = CHUNK_VERT_RES as usize;
    let quads = CHUNK_QUADS_PER_EDGE as usize;
    let expected_verts = n * n + 4 * n;
    let expected_indices = quads * quads * 6 + 4 * quads * 6;

    let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("position attr");
    match positions {
        VertexAttributeValues::Float32x3(vec) => {
            assert_eq!(vec.len(), expected_verts, "vertex count mismatch");
        }
        _ => panic!("expected Float32x3 positions"),
    }

    match mesh.indices() {
        Some(Indices::U32(indices)) => {
            assert_eq!(indices.len(), expected_indices, "index count mismatch");
        }
        _ => panic!("expected U32 indices"),
    }
}

#[test]
fn chunk_mesh_morph_values() {
    let key = CellKey { face: 0, i: 0, j: 0, lod: 0 };
    let mesh = generate_chunk_mesh(key, 12000.0);

    let n = CHUNK_VERT_RES as usize;
    let surface_count = n * n;

    let morphs = mesh.attribute(ATTRIBUTE_MORPH).expect("morph attr");
    match morphs {
        VertexAttributeValues::Float32(vec) => {
            assert_eq!(vec.len(), n * n + 4 * n, "morph count mismatch");
            for i in 0..surface_count {
                assert_eq!(vec[i], 1.0, "surface vertex {i} morph should be 1.0");
            }
            for i in surface_count..(surface_count + 4 * n) {
                assert_eq!(vec[i], 0.0, "skirt vertex {i} morph should be 0.0");
            }
        }
        _ => panic!("expected Float32 morphs"),
    }
}

#[test]
fn adjacent_chunks_share_edge_vertices() {
    let radius = 12000.0;
    let lod = 2u8;
    let key1 = CellKey { face: 0, i: 0, j: 0, lod };
    let key2 = CellKey { face: 0, i: 1, j: 0, lod };

    let mesh1 = generate_chunk_mesh(key1, radius);
    let mesh2 = generate_chunk_mesh(key2, radius);

    let n = CHUNK_VERT_RES as usize;

    let pos1 = match mesh1.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos2 = match mesh2.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };

    for j in 0..n {
        let idx1 = j * n + (n - 1);
        let idx2 = j * n;
        let p1 = pos1[idx1];
        let p2 = pos2[idx2];
        let diff = (p1[0] - p2[0]).abs() + (p1[1] - p2[1]).abs() + (p1[2] - p2[2]).abs();
        assert!(
            diff < 0.001,
            "edge vertex mismatch at j={j}: {p1:?} vs {p2:?} (diff={diff})"
        );
    }
}

#[test]
fn skirt_vertices_below_surface() {
    let radius = 12000.0;
    let key = CellKey { face: 0, i: 0, j: 0, lod: 3 };
    let mesh = generate_chunk_mesh(key, radius);

    let n = CHUNK_VERT_RES as usize;
    let surface_count = n * n;

    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };

    for i in surface_count..positions.len() {
        let pos = positions[i];
        let dist = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
        assert!(
            dist < radius as f32,
            "skirt vertex {i} at dist {dist} should be below surface radius {radius}"
        );
    }
}

#[test]
fn face_edge_chunks_adjacent_across_boundary() {
    let radius = 12000.0;
    let lod = 2u8;
    let cells = cells_per_edge(lod);
    let n = CHUNK_VERT_RES as usize;

    let key1 = CellKey { face: 0, i: cells - 1, j: cells / 2, lod };
    let mesh1 = generate_chunk_mesh(key1, radius);

    let key2 = CellKey { face: 2, i: 0, j: cells / 2, lod };
    let mesh2 = generate_chunk_mesh(key2, radius);

    let pos1 = match mesh1.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos2 = match mesh2.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };

    let mut max_diff = 0.0f32;
    for j in 0..n {
        let idx1 = j * n + (n - 1);
        let idx2 = j * n;
        let p1 = pos1[idx1];
        let p2 = pos2[idx2];
        let diff = (p1[0] - p2[0]).abs() + (p1[1] - p2[1]).abs() + (p1[2] - p2[2]).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(
        max_diff < 0.01,
        "cross-face edge mismatch: max_diff={max_diff}"
    );
}
