use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use er_core::config::{CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES};
use er_core::math::{cells_per_edge, CellKey};
use er_core::seed::PlanetSeed;
use er_terrain::{
    generate_chunk_mesh as generate_chunk_mesh_with_field, ChunkComponent, ATTRIBUTE_GRID,
    ATTRIBUTE_MORPH, ATTRIBUTE_NORMAL,
};
use er_world::cache::WorldCache;
use er_world::elevation::{elevation_params, ElevationNoise, ElevationParams};
use er_world::params::{climate_noise, planet_params, ClimateNoise, PlanetParams};
use er_world::terrain_field::ProceduralTerrainField;
use glam::Vec3;

const ELEVATION_SCALE: f32 = 1000.0;

fn test_elevation() -> (ElevationNoise, ElevationParams, PlanetParams, ClimateNoise) {
    let seed = PlanetSeed(0xC0FFEE);
    let elev_params = elevation_params(seed);
    let noise = ElevationNoise::new(&elev_params);
    let pp = planet_params(seed);
    let cn = climate_noise(&pp);
    (noise, elev_params, pp, cn)
}

// Keep the historical seam fixtures while exercising the same field boundary
// used by asynchronous terrain mesh workers.
#[allow(clippy::too_many_arguments)]
fn generate_chunk_mesh(
    key: CellKey,
    radius: f64,
    elevation_scale: f32,
    _noise: &ElevationNoise,
    elevation_params: &ElevationParams,
    planet_params: &PlanetParams,
    _climate_noise: &ClimateNoise,
    _cache: Option<&WorldCache>,
) -> Mesh {
    let field = ProceduralTerrainField::new(*elevation_params, *planet_params);
    generate_chunk_mesh_with_field(key, radius, elevation_scale, &field)
}

#[test]
fn chunk_mesh_vertex_and_index_counts() {
    let (noise, elev_params, pp, cn) = test_elevation();
    let key = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 0,
    };
    let mesh = generate_chunk_mesh(
        key,
        12000.0,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    );

    let n = CHUNK_VERT_RES as usize;
    let quads = CHUNK_QUADS_PER_EDGE as usize;
    let expected_verts = n * n + 4 * n;
    let expected_indices = quads * quads * 6 + 4 * quads * 6;

    let positions = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .expect("position attr");
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
    let (noise, elev_params, pp, cn) = test_elevation();
    let key = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 0,
    };
    let mesh = generate_chunk_mesh(
        key,
        12000.0,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    );

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
    let (noise, elev_params, pp, cn) = test_elevation();
    let radius = 12000.0;
    let lod = 2u8;
    let key1 = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod,
    };
    let key2 = CellKey {
        face: 0,
        i: 1,
        j: 0,
        lod,
    };

    let mesh1 = generate_chunk_mesh(
        key1,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    );
    let mesh2 = generate_chunk_mesh(
        key2,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    );

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
    let (noise, elev_params, pp, cn) = test_elevation();
    let radius = 12000.0;
    let key = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 3,
    };
    let mesh = generate_chunk_mesh(
        key,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    );

    let n = CHUNK_VERT_RES as usize;
    let surface_count = n * n;

    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };

    for skirt_index in surface_count..positions.len() {
        let strip = (skirt_index - surface_count) / n;
        let surface_index = paired_surface_index(strip, (skirt_index - surface_count) % n, n);
        let skirt = Vec3::from_array(positions[skirt_index]);
        let surface = Vec3::from_array(positions[surface_index]);
        assert!(
            skirt.length() < surface.length(),
            "skirt vertex {skirt_index} should be below paired surface vertex {surface_index}"
        );
    }
}

#[test]
fn skirt_attributes_match_paired_surface_vertices() {
    let (noise, elev_params, pp, cn) = test_elevation();
    let n = CHUNK_VERT_RES as usize;
    let surface_count = n * n;

    for face in 0..6 {
        let key = CellKey {
            face,
            i: 0,
            j: 0,
            lod: 1,
        };
        let mesh = generate_chunk_mesh(
            key,
            12000.0,
            ELEVATION_SCALE,
            &noise,
            &elev_params,
            &pp,
            &cn,
            None,
        );
        let positions = positions(&mesh);
        let normals = match mesh.attribute(ATTRIBUTE_NORMAL).unwrap() {
            VertexAttributeValues::Float32x3(values) => values,
            _ => panic!("expected Float32x3 normals"),
        };

        for strip in 0..4 {
            let strip_start = surface_count + strip * n;
            for skirt_index in strip_start..strip_start + n {
                let surface_index = paired_surface_index(strip, skirt_index - strip_start, n);
                assert_eq!(
                    normals[skirt_index], normals[surface_index],
                    "face {face}: skirt normal {skirt_index} must reuse surface normal {surface_index}"
                );

                let skirt = Vec3::from_array(positions[skirt_index]);
                let surface = Vec3::from_array(positions[surface_index]);
                assert!(
                    skirt.length() < surface.length(),
                    "face {face}: skirt vertex {skirt_index} must be below paired surface vertex {surface_index}"
                );
                assert!(
                    surface.length() - skirt.length() <= ELEVATION_SCALE + 0.1,
                    "face {face}: skirt vertex {skirt_index} exceeds the elevation-scale depth cap"
                );
                assert!(
                    skirt.normalize().distance(surface.normalize()) < 1e-5,
                    "face {face}: skirt vertex {skirt_index} must remain radially aligned with surface vertex {surface_index}"
                );
            }
        }
    }
}

#[test]
fn skirt_triangles_face_outward_on_every_face() {
    let (noise, elev_params, pp, cn) = test_elevation();
    let n = CHUNK_VERT_RES as usize;
    let quads = CHUNK_QUADS_PER_EDGE as usize;
    let surface_index_count = quads * quads * 6;
    let strip_index_count = quads * 6;

    for face in 0..6 {
        let key = CellKey {
            face,
            i: 0,
            j: 0,
            lod: 0,
        };
        let mesh = generate_chunk_mesh(
            key,
            12000.0,
            ELEVATION_SCALE,
            &noise,
            &elev_params,
            &pp,
            &cn,
            None,
        );
        let positions = positions(&mesh);
        let indices = match mesh.indices().unwrap() {
            Indices::U32(values) => values,
            _ => panic!("expected U32 indices"),
        };

        for (strip, edge_indices) in indices[surface_index_count..]
            .chunks_exact(strip_index_count)
            .enumerate()
        {
            let outward = skirt_outward_direction(&positions, n, strip);
            for triangle in edge_indices.chunks_exact(3) {
                let a = Vec3::from_array(positions[triangle[0] as usize]);
                let b = Vec3::from_array(positions[triangle[1] as usize]);
                let c = Vec3::from_array(positions[triangle[2] as usize]);
                let normal = (b - a).cross(c - a);
                assert!(
                    normal.dot(outward) > 0.0,
                    "face {face}, skirt strip {strip}: triangle {triangle:?} faces inward"
                );
            }
        }
    }
}

#[test]
fn face_edge_chunks_adjacent_across_boundary() {
    let (noise, elev_params, pp, cn) = test_elevation();
    let radius = 12000.0;
    let lod = 2u8;
    let cells = cells_per_edge(lod);
    let n = CHUNK_VERT_RES as usize;

    let key1 = CellKey {
        face: 0,
        i: cells - 1,
        j: cells / 2,
        lod,
    };
    let mesh1 = generate_chunk_mesh(
        key1,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    );

    let key2 = CellKey {
        face: 2,
        i: cells - 1,
        j: cells / 2,
        lod,
    };
    let mesh2 = generate_chunk_mesh(
        key2,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    );

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
        let idx2 = j * n + (n - 1);
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

#[test]
fn chunk_component_neighbor_depth_default() {
    let key = CellKey {
        face: 0,
        i: 1,
        j: 2,
        lod: 3,
    };
    let chunk = ChunkComponent::new(key);
    assert_eq!(chunk.neighbor_depth, [3, 3, 3, 3]);
}

#[test]
fn grid_attribute_values() {
    let (noise, elev_params, pp, cn) = test_elevation();
    let key = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 0,
    };
    let mesh = generate_chunk_mesh(
        key,
        12000.0,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    );
    let n = CHUNK_VERT_RES as usize;
    let n1 = (n - 1) as u32;

    let grid = mesh.attribute(ATTRIBUTE_GRID).expect("grid attr");
    match grid {
        VertexAttributeValues::Uint32x2(v) => {
            assert_eq!(v.len(), n * n + 4 * n, "grid count mismatch");
            assert_eq!(v[0], [0, 0], "first surface grid");
            assert_eq!(v[n * n - 1], [n1, n1], "last surface grid");
            assert_eq!(v[n * n], [0, 0], "top skirt first");
            assert_eq!(v[n * n + n], [0, n1], "bot skirt first");
        }
        _ => panic!("expected Uint32x2 grid"),
    }
}

fn positions(mesh: &Mesh) -> Vec<[f32; 3]> {
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v.clone(),
        _ => panic!("expected Float32x3 positions"),
    }
}

fn manhattan(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn paired_surface_index(strip: usize, k: usize, n: usize) -> usize {
    match strip {
        0 => k,
        1 => (n - 1) * n + k,
        2 => k * n,
        3 => k * n + (n - 1),
        _ => panic!("invalid skirt strip {strip}"),
    }
}

fn skirt_outward_direction(positions: &[[f32; 3]], n: usize, strip: usize) -> Vec3 {
    let center = Vec3::from_array(positions[(n / 2) * n + n / 2]).normalize();
    let edge_surface_index = match strip {
        0 => n / 2,
        1 => (n - 1) * n + n / 2,
        2 => (n / 2) * n,
        3 => (n / 2) * n + (n - 1),
        _ => panic!("invalid skirt strip {strip}"),
    };
    (Vec3::from_array(positions[edge_surface_index]).normalize() - center).normalize()
}

#[test]
fn finer_even_edge_vertices_coincide_with_coarser() {
    let (noise, elev_params, pp, cn) = test_elevation();
    let radius = 12000.0;
    let n = CHUNK_VERT_RES as usize;
    let fine = CellKey {
        face: 0,
        i: 1,
        j: 0,
        lod: 2,
    };
    let coarse = CellKey {
        face: 0,
        i: 1,
        j: 0,
        lod: 1,
    };

    let pf = positions(&generate_chunk_mesh(
        fine,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    ));
    let pc = positions(&generate_chunk_mesh(
        coarse,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    ));

    for k in 0..=(n / 2) {
        let gj_fine = 2 * k;
        let fine_idx = gj_fine * n + (n - 1);
        let coarse_idx = k * n;
        let diff = manhattan(pf[fine_idx], pc[coarse_idx]);
        assert!(
            diff < 0.01,
            "fine even edge vert (gj={gj_fine}) != coarse edge vert (gj={k}): diff={diff}"
        );
    }
}

#[test]
fn edge_stitch_snaps_inbetween_to_coarse_edge() {
    let (noise, elev_params, pp, cn) = test_elevation();
    let radius = 12000.0;
    let n = CHUNK_VERT_RES as usize;
    let fine = CellKey {
        face: 0,
        i: 1,
        j: 0,
        lod: 2,
    };
    let coarse = CellKey {
        face: 0,
        i: 1,
        j: 0,
        lod: 1,
    };

    let pf = positions(&generate_chunk_mesh(
        fine,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    ));
    let pc = positions(&generate_chunk_mesh(
        coarse,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    ));

    for k in 0..(n / 2) {
        let gj_lo = 2 * k;
        let gj_hi = 2 * k + 2;
        let fine_lo = gj_lo * n + (n - 1);
        let fine_mid = (gj_lo + 1) * n + (n - 1);
        let fine_hi = gj_hi * n + (n - 1);
        let stitched = lerp3(pf[fine_lo], pf[fine_hi], 0.5);
        let coarse_mid = lerp3(pc[k * n], pc[(k + 1) * n], 0.5);
        let diff = manhattan(stitched, coarse_mid);
        assert!(
            diff < 0.01,
            "stitched in-between vert (gj={}) != coarse edge midpoint: diff={diff}",
            gj_lo + 1
        );
        let raw_diff = manhattan(pf[fine_mid], coarse_mid);
        assert!(
            raw_diff > 1e-3,
            "in-between vert already on coarse edge (raw_diff={raw_diff})"
        );
    }
}
