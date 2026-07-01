use bevy::render::mesh::{Indices, Mesh, MeshVertexAttribute};
use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::{PrimitiveTopology, VertexFormat};
use er_core::config::{CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES};
use er_core::math::{cell_size, cells_per_edge, uv_to_dir, CellKey};

pub const ATTRIBUTE_MORPH: MeshVertexAttribute =
    MeshVertexAttribute::new("Morph", 988540918, VertexFormat::Float32);

pub fn generate_chunk_mesh(key: CellKey, radius: f64) -> Mesh {
    let n = CHUNK_VERT_RES as usize;
    let quads = CHUNK_QUADS_PER_EDGE as usize;
    let cells = cells_per_edge(key.lod) as f64;

    let u_min = key.i as f64 / cells;
    let u_max = (key.i + 1) as f64 / cells;
    let v_min = key.j as f64 / cells;
    let v_max = (key.j + 1) as f64 / cells;

    let skirt_depth = cell_size(key.lod, radius) * 0.2;
    let skirt_radius = radius - skirt_depth;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * n + 4 * n);
    let mut morphs: Vec<f32> = Vec::with_capacity(n * n + 4 * n);

    for gj in 0..n {
        for gi in 0..n {
            let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
            let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
            let dir = uv_to_dir(key.face, u, v);
            let pos = dir * radius;
            positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
            morphs.push(1.0);
        }
    }

    let surface_count = n * n;

    let top_skirt = surface_count;
    for gi in 0..n {
        let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u, v_min);
        let pos = dir * skirt_radius;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
    }

    let bot_skirt = top_skirt + n;
    for gi in 0..n {
        let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u, v_max);
        let pos = dir * skirt_radius;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
    }

    let left_skirt = bot_skirt + n;
    for gj in 0..n {
        let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u_min, v);
        let pos = dir * skirt_radius;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
    }

    let right_skirt = left_skirt + n;
    for gj in 0..n {
        let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u_max, v);
        let pos = dir * skirt_radius;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
    }

    let mut indices: Vec<u32> = Vec::with_capacity(quads * quads * 6 + 4 * quads * 6);

    for gj in 0..quads {
        for gi in 0..quads {
            let v00 = (gj * n + gi) as u32;
            let v10 = (gj * n + gi + 1) as u32;
            let v01 = ((gj + 1) * n + gi) as u32;
            let v11 = ((gj + 1) * n + gi + 1) as u32;
            indices.extend_from_slice(&[v00, v01, v11, v00, v11, v10]);
        }
    }

    for gi in 0..quads {
        let g0 = gi as u32;
        let g1 = (gi + 1) as u32;
        let s0 = (top_skirt + gi) as u32;
        let s1 = (top_skirt + gi + 1) as u32;
        indices.extend_from_slice(&[g0, s0, s1, g0, s1, g1]);
    }

    for gi in 0..quads {
        let g0 = ((n - 1) * n + gi) as u32;
        let g1 = ((n - 1) * n + gi + 1) as u32;
        let s0 = (bot_skirt + gi) as u32;
        let s1 = (bot_skirt + gi + 1) as u32;
        indices.extend_from_slice(&[g0, s1, s0, g0, g1, s1]);
    }

    for gj in 0..quads {
        let g0 = (gj * n) as u32;
        let g1 = ((gj + 1) * n) as u32;
        let s0 = (left_skirt + gj) as u32;
        let s1 = (left_skirt + gj + 1) as u32;
        indices.extend_from_slice(&[g0, s1, s0, g0, g1, s1]);
    }

    for gj in 0..quads {
        let g0 = (gj * n + (n - 1)) as u32;
        let g1 = ((gj + 1) * n + (n - 1)) as u32;
        let s0 = (right_skirt + gj) as u32;
        let s1 = (right_skirt + gj + 1) as u32;
        indices.extend_from_slice(&[g0, s0, s1, g0, s1, g1]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(ATTRIBUTE_MORPH, morphs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
