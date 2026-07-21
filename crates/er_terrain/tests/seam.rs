use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use er_core::config::{CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES};
use er_core::math::{cell_to_dir, cells_per_edge, CellKey};
use er_core::seed::PlanetSeed;
use er_terrain::{
    generate_chunk_mesh as generate_chunk_mesh_with_field, generate_chunk_mesh_stitched,
    ChunkComponent, ATTRIBUTE_CURVATURE, ATTRIBUTE_DRAINAGE, ATTRIBUTE_ELEVATION, ATTRIBUTE_MORPH,
    ATTRIBUTE_NORMAL,
};
use er_world::cache::WorldCache;
use er_world::elevation::{elevation_params, ElevationNoise, ElevationParams};
use er_world::params::{climate_noise, planet_params, ClimateNoise, PlanetParams};
use er_world::terrain_field::{
    ProceduralTerrainField, TerrainField, TerrainSample, TerrainSampleSource,
};
use er_world::Biome;
use glam::DVec3;
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
    generate_chunk_mesh_with_field(
        key,
        radius,
        elevation_scale,
        planet_params.sea_level,
        &field,
    )
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
        Some(Indices::U16(indices)) => {
            assert_eq!(indices.len(), expected_indices, "index count mismatch");
        }
        _ => panic!("expected U16 indices"),
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
            for (i, &v) in vec.iter().enumerate().take(surface_count) {
                assert_eq!(v, 1.0, "surface vertex {i} morph should be 1.0");
            }
            for (i, &v) in vec.iter().enumerate().skip(surface_count).take(4 * n) {
                assert_eq!(v, 0.0, "skirt vertex {i} morph should be 0.0");
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

    let pos1_raw = match mesh1.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos2_raw = match mesh2.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos1 = world_positions(key1, radius, pos1_raw);
    let pos2 = world_positions(key2, radius, pos2_raw);

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

    let positions_raw = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let positions = world_positions(key, radius, positions_raw);

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
    let radius = 12000.0;

    for face in 0..6 {
        let key = CellKey {
            face,
            i: 0,
            j: 0,
            lod: 1,
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
        let positions_raw = positions(&mesh);
        let positions = world_positions(key, radius, &positions_raw);
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
    let radius = 12000.0;

    for face in 0..6 {
        let key = CellKey {
            face,
            i: 0,
            j: 0,
            lod: 0,
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
        let positions_raw = positions(&mesh);
        let positions = world_positions(key, radius, &positions_raw);
        let indices: Vec<u32> = match mesh.indices().unwrap() {
            Indices::U16(values) => values.iter().map(|value| u32::from(*value)).collect(),
            Indices::U32(values) => values.clone(),
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

    let pos1_raw = match mesh1.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos2_raw = match mesh2.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos1 = world_positions(key1, radius, pos1_raw);
    let pos2 = world_positions(key2, radius, pos2_raw);

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

fn positions(mesh: &Mesh) -> Vec<[f32; 3]> {
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v.clone(),
        _ => panic!("expected Float32x3 positions"),
    }
}

fn world_positions(key: CellKey, radius: f64, positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let anchor = cell_to_dir(key) * radius;
    positions
        .iter()
        .map(|&p| {
            let wp = DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64) + anchor;
            [wp.x as f32, wp.y as f32, wp.z as f32]
        })
        .collect()
}

fn world_positions_f64(key: CellKey, radius: f64, positions: &[[f32; 3]]) -> Vec<DVec3> {
    let anchor = cell_to_dir(key) * radius;
    positions
        .iter()
        .map(|&p| DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64) + anchor)
        .collect()
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

    let pf_raw = positions(&generate_chunk_mesh(
        fine,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    ));
    let pc_raw = positions(&generate_chunk_mesh(
        coarse,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    ));
    let pf = world_positions(fine, radius, &pf_raw);
    let pc = world_positions(coarse, radius, &pc_raw);

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

    let pf_raw2 = positions(&generate_chunk_mesh(
        fine,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    ));
    let pc_raw2 = positions(&generate_chunk_mesh(
        coarse,
        radius,
        ELEVATION_SCALE,
        &noise,
        &elev_params,
        &pp,
        &cn,
        None,
    ));
    let pf2 = world_positions(fine, radius, &pf_raw2);
    let pc2 = world_positions(coarse, radius, &pc_raw2);

    for k in 0..(n / 2) {
        let gj_lo = 2 * k;
        let gj_hi = 2 * k + 2;
        let fine_lo = gj_lo * n + (n - 1);
        let fine_mid = (gj_lo + 1) * n + (n - 1);
        let fine_hi = gj_hi * n + (n - 1);
        let stitched = lerp3(pf2[fine_lo], pf2[fine_hi], 0.5);
        let coarse_mid = lerp3(pc2[k * n], pc2[(k + 1) * n], 0.5);
        let diff = manhattan(stitched, coarse_mid);
        assert!(
            diff < 0.01,
            "stitched in-between vert (gj={}) != coarse edge midpoint: diff={diff}",
            gj_lo + 1
        );
        let raw_diff = manhattan(pf2[fine_mid], coarse_mid);
        assert!(
            raw_diff > 1e-3,
            "in-between vert already on coarse edge (raw_diff={raw_diff})"
        );
    }
}

#[test]
fn generated_mixed_lod_edge_matches_coarse_mesh_segments() {
    let (_, elev_params, pp, _) = test_elevation();
    let field = ProceduralTerrainField::new(elev_params, pp);
    let radius = 12000.0;
    let n = CHUNK_VERT_RES as usize;
    let fine = CellKey {
        face: 0,
        i: 2,
        j: 0,
        lod: 3,
    };
    let coarse = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 2,
    };

    let fine_mesh = generate_chunk_mesh_stitched(
        fine,
        radius,
        ELEVATION_SCALE,
        pp.sea_level,
        &field,
        [Some(coarse), None, None, None],
    );
    let coarse_mesh =
        generate_chunk_mesh_with_field(coarse, radius, ELEVATION_SCALE, pp.sea_level, &field);
    let fine_world = world_positions(fine, radius, &positions(&fine_mesh));
    let coarse_world = world_positions(coarse, radius, &positions(&coarse_mesh));

    for k in 0..n {
        let coarse_grid = k as f32 * 0.5;
        let segment = coarse_grid.floor().min((CHUNK_QUADS_PER_EDGE - 1) as f32) as usize;
        let expected = lerp3(
            coarse_world[segment * n + (n - 1)],
            coarse_world[(segment + 1) * n + (n - 1)],
            coarse_grid - segment as f32,
        );
        let actual = fine_world[k * n];
        let diff = manhattan(actual, expected);
        assert!(
            diff < 0.02,
            "stitched fine edge vertex {k} misses coarse segment by {diff}: {actual:?} vs {expected:?}"
        );
    }
}

// ---- Metric Earth-scale seam test (Milestone 2.2) ----
//
// Proves that adjacent/stitched chunk edges remain continuous when the
// terrain field uses the full metric landform composition (continental +
// mountain + tectonic + ridge + valley + drainage + erosion + talus).

fn test_metric_field() -> (ProceduralTerrainField, PlanetParams) {
    let seed = PlanetSeed(0xC0FFEE);
    let elev_params = elevation_params(seed);
    let pp = planet_params(seed);
    (
        ProceduralTerrainField::new_metric(elev_params, pp, 6_371_000.0),
        pp,
    )
}

#[test]
fn metric_earth_scale_adjacent_chunks_share_edge_vertices() {
    let (field, pp) = test_metric_field();
    let radius = 6_371_000.0;
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

    let mesh1 = generate_chunk_mesh_with_field(key1, radius, ELEVATION_SCALE, pp.sea_level, &field);
    let mesh2 = generate_chunk_mesh_with_field(key2, radius, ELEVATION_SCALE, pp.sea_level, &field);

    let n = CHUNK_VERT_RES as usize;
    let pos1_raw = match mesh1.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos2_raw = match mesh2.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos1 = world_positions_f64(key1, radius, pos1_raw);
    let pos2 = world_positions_f64(key2, radius, pos2_raw);

    // Local f32 offsets retain sub-meter precision; reconstruct in f64 so the
    // test does not add absolute Earth-radius f32 quantization.
    let max_error_m = 0.5;
    for j in 0..n {
        let idx1 = j * n + (n - 1);
        let idx2 = j * n;
        let p1 = pos1[idx1];
        let p2 = pos2[idx2];
        let diff = (p1 - p2).abs().element_sum();
        assert!(
            diff < max_error_m,
            "metric edge vertex mismatch at j={j}: {p1:?} vs {p2:?} (diff={diff})"
        );
    }
}

#[test]
fn sparse_metric_sampling_keeps_deep_lod_edge_attributes_identical() {
    let (field, pp) = test_metric_field();
    let radius = 6_371_000.0;
    let lod = 17u8;
    let key1 = CellKey {
        face: 0,
        i: 65_535,
        j: 65_536,
        lod,
    };
    let key2 = CellKey { i: 65_536, ..key1 };
    let mesh1 = generate_chunk_mesh_with_field(key1, radius, ELEVATION_SCALE, pp.sea_level, &field);
    let mesh2 = generate_chunk_mesh_with_field(key2, radius, ELEVATION_SCALE, pp.sea_level, &field);
    let n = CHUNK_VERT_RES as usize;

    for attribute in [ATTRIBUTE_NORMAL, ATTRIBUTE_DRAINAGE, ATTRIBUTE_CURVATURE] {
        match (mesh1.attribute(attribute), mesh2.attribute(attribute)) {
            (
                Some(VertexAttributeValues::Float32x3(left)),
                Some(VertexAttributeValues::Float32x3(right)),
            ) => {
                for j in 0..n {
                    assert_eq!(left[j * n + n - 1], right[j * n]);
                }
            }
            (
                Some(VertexAttributeValues::Float32(left)),
                Some(VertexAttributeValues::Float32(right)),
            ) => {
                for j in 0..n {
                    assert_eq!(left[j * n + n - 1], right[j * n]);
                }
            }
            _ => panic!("unexpected attribute format"),
        }
    }
}

#[test]
fn metric_earth_scale_face_edge_adjacent_across_boundary() {
    let (field, pp) = test_metric_field();
    let radius = 6_371_000.0;
    let lod = 2u8;
    let cells = cells_per_edge(lod);
    let n = CHUNK_VERT_RES as usize;

    let key1 = CellKey {
        face: 0,
        i: cells - 1,
        j: cells / 2,
        lod,
    };
    let mesh1 = generate_chunk_mesh_with_field(key1, radius, ELEVATION_SCALE, pp.sea_level, &field);

    let key2 = CellKey {
        face: 2,
        i: cells - 1,
        j: cells / 2,
        lod,
    };
    let mesh2 = generate_chunk_mesh_with_field(key2, radius, ELEVATION_SCALE, pp.sea_level, &field);

    let pos1_raw = match mesh1.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos2_raw = match mesh2.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
        VertexAttributeValues::Float32x3(v) => v,
        _ => panic!(),
    };
    let pos1 = world_positions_f64(key1, radius, pos1_raw);
    let pos2 = world_positions_f64(key2, radius, pos2_raw);

    let mut max_diff = 0.0f64;
    for j in 0..n {
        let idx1 = j * n + (n - 1);
        let idx2 = j * n + (n - 1);
        let p1 = pos1[idx1];
        let p2 = pos2[idx2];
        let diff = (p1 - p2).abs().element_sum();
        max_diff = max_diff.max(diff);
    }
    assert!(
        max_diff < 0.5,
        "metric cross-face edge mismatch: max_diff={max_diff}"
    );
}

#[test]
fn metric_earth_scale_edge_stitch_snaps_to_coarse() {
    let (field, pp) = test_metric_field();
    let radius = 6_371_000.0;
    let n = CHUNK_VERT_RES as usize;
    // Fine chunk's left edge must be adjacent to the coarse chunk's
    // right edge (same pattern as the miniature-scale test above).
    let fine = CellKey {
        face: 0,
        i: 2,
        j: 0,
        lod: 3,
    };
    let coarse = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 2,
    };

    let fine_mesh = generate_chunk_mesh_stitched(
        fine,
        radius,
        ELEVATION_SCALE,
        pp.sea_level,
        &field,
        [Some(coarse), None, None, None],
    );
    let coarse_mesh =
        generate_chunk_mesh_with_field(coarse, radius, ELEVATION_SCALE, pp.sea_level, &field);
    let fine_world = world_positions_f64(fine, radius, &positions(&fine_mesh));
    let coarse_world = world_positions_f64(coarse, radius, &positions(&coarse_mesh));

    let max_error_m = 0.5;
    for k in 0..n {
        let coarse_grid = k as f64 * 0.5;
        let segment = coarse_grid.floor().min((CHUNK_QUADS_PER_EDGE - 1) as f64) as usize;
        let expected = coarse_world[segment * n + (n - 1)].lerp(
            coarse_world[(segment + 1) * n + (n - 1)],
            coarse_grid - segment as f64,
        );
        let actual = fine_world[k * n];
        let diff = (actual - expected).abs().element_sum();
        assert!(
            diff < max_error_m,
            "metric stitched fine edge vertex {k} misses coarse segment by {diff}"
        );
    }
}

#[test]
fn metric_material_inputs_are_finite_bounded_and_nonconstant() {
    let (field, pp) = test_metric_field();
    let key = CellKey {
        face: 0,
        i: 512,
        j: 512,
        lod: 10,
    };
    let mesh =
        generate_chunk_mesh_with_field(key, 6_371_000.0, ELEVATION_SCALE, pp.sea_level, &field);
    let surface_count = (CHUNK_VERT_RES as usize).pow(2);

    for (name, attribute) in [
        ("drainage", ATTRIBUTE_DRAINAGE),
        ("curvature", ATTRIBUTE_CURVATURE),
    ] {
        let values = match mesh.attribute(attribute).expect("material attribute") {
            VertexAttributeValues::Float32(values) => values,
            _ => panic!("{name} must use Float32"),
        };
        assert_eq!(values.len(), surface_count + 4 * CHUNK_VERT_RES as usize);
        let surface = &values[..surface_count];
        assert!(surface.iter().all(|value| value.is_finite()));
        let min = surface.iter().copied().fold(f32::INFINITY, f32::min);
        let max = surface.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("metric material {name} range: [{min:.4}, {max:.4}]");
        assert!(min >= -1.0 && max <= 1.0, "{name} range [{min}, {max}]");
        assert!(max - min > 1e-6, "{name} is constant at {min}");
    }
}

// ---- Macro-water sea-level clamp regression ----
//
// Terrain-owned macro-water vertices must be clamped exactly to the shared
// sea-level radius even when the full elevation differs from sea level. This
// guards the terrain-owned water surface decision (mesh_gen.rs:411-415 and the
// stitched-edge `surface_position` path) against regressions that would let
// macro-water vertices follow the full elevation and break the flat,
// continuous, seam-safe ocean surface.

/// Deterministic field where every sample is classified as macro-water
/// (`low_freq_elev < sea_level`) but carries a full `elevation` that differs
/// from sea level by a large margin. This isolates the clamp: if the clamp is
/// removed, vertices would land at `radius + elevation * scale` instead of
/// `radius + sea_level * scale`.
struct MacroWaterField {
    low_freq_elev: f32,
    elevation: f64,
}

impl TerrainField for MacroWaterField {
    fn sample(&self, _dir: DVec3) -> TerrainSample {
        TerrainSample {
            elevation: self.elevation,
            low_freq_elev: self.low_freq_elev,
            warped_dir: [0.0, 0.0, 0.0],
            moisture: 0.0,
            biome: Biome::OceanMid,
            mountain_influence: 0.0,
            temperature: 0.0,
            drainage: 0.0,
            source: TerrainSampleSource::Procedural,
        }
    }
}

#[test]
fn macro_water_surface_vertices_clamped_to_sea_level() {
    // low_freq below sea level => macro-water; full elevation far above sea
    // level so a missing clamp would be immediately visible.
    let sea_level = 0.0;
    let field = MacroWaterField {
        low_freq_elev: -0.5,
        elevation: 0.8,
    };
    let radius = 12000.0;
    let key = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 0,
    };
    let mesh = generate_chunk_mesh_with_field(key, radius, ELEVATION_SCALE, sea_level, &field);

    let n = CHUNK_VERT_RES as usize;
    let surface_count = n * n;
    let positions_raw = positions(&mesh);
    let world = world_positions_f64(key, radius, &positions_raw);

    let expected_radius = radius + sea_level * ELEVATION_SCALE as f64;
    let wrong_radius = radius + field.elevation * ELEVATION_SCALE as f64;
    assert!(
        (wrong_radius - expected_radius).abs() > 1.0,
        "test setup must have a meaningful elevation/sea-level gap"
    );

    for (i, p) in world.iter().enumerate().take(surface_count) {
        let r = p.length();
        assert!(
            (r - expected_radius).abs() < 1e-3,
            "surface vertex {i} radius {r} != sea-level radius {expected_radius}"
        );
    }
}

#[test]
fn macro_water_skirt_vertices_clamped_to_sea_level() {
    let sea_level = 0.0;
    let field = MacroWaterField {
        low_freq_elev: -0.5,
        elevation: 0.8,
    };
    let radius = 12000.0;
    let key = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 0,
    };
    let mesh = generate_chunk_mesh_with_field(key, radius, ELEVATION_SCALE, sea_level, &field);

    let n = CHUNK_VERT_RES as usize;
    let surface_count = n * n;
    let positions_raw = positions(&mesh);
    let world = world_positions_f64(key, radius, &positions_raw);

    let expected_radius = radius + sea_level * ELEVATION_SCALE as f64;

    // Skirt vertices are pushed below their paired surface vertex, so they must
    // be strictly below the sea-level radius while remaining radially aligned
    // with the (clamped) surface vertex.
    for skirt_index in surface_count..world.len() {
        let strip = (skirt_index - surface_count) / n;
        let surface_index = paired_surface_index(strip, (skirt_index - surface_count) % n, n);
        let skirt = world[skirt_index];
        let surface = world[surface_index];
        assert!(
            skirt.length() < surface.length(),
            "skirt vertex {skirt_index} must be below paired surface vertex {surface_index}"
        );
        assert!(
            skirt.normalize().distance(surface.normalize()) < 1e-5,
            "skirt vertex {skirt_index} must be radially aligned with surface vertex {surface_index}"
        );
        assert!(
            (surface.length() - expected_radius).abs() < 1e-3,
            "paired surface vertex {surface_index} must be at sea-level radius"
        );
    }
}

#[test]
fn macro_water_stitched_edge_clamped_to_sea_level() {
    let sea_level = 0.0;
    let field = MacroWaterField {
        low_freq_elev: -0.5,
        elevation: 0.8,
    };
    let radius = 12000.0;
    let n = CHUNK_VERT_RES as usize;
    let fine = CellKey {
        face: 0,
        i: 2,
        j: 0,
        lod: 3,
    };
    let coarse = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 2,
    };

    let fine_mesh = generate_chunk_mesh_stitched(
        fine,
        radius,
        ELEVATION_SCALE,
        sea_level,
        &field,
        [Some(coarse), None, None, None],
    );
    let fine_world = world_positions_f64(fine, radius, &positions(&fine_mesh));

    let expected_radius = radius + sea_level * ELEVATION_SCALE as f64;
    let wrong_radius = radius + field.elevation * ELEVATION_SCALE as f64;
    assert!(
        (wrong_radius - expected_radius).abs() > 1.0,
        "test setup must have a meaningful elevation/sea-level gap"
    );

    // The stitched edge snaps to coarse-mesh segment endpoints (both clamped
    // to sea level via `surface_position`) and linearly interpolates the chord
    // between them. The chord sags below the sea-level sphere by a tiny amount
    // (<1 m here), while the full-elevation sphere is ~800 m away. So every
    // stitched vertex must sit far closer to the sea-level radius than to the
    // full-elevation radius — proving the clamp held through the stitch path.
    for k in 0..n {
        let idx = k * n;
        let r = fine_world[idx].length();
        let sea_err = (r - expected_radius).abs();
        let elev_err = (r - wrong_radius).abs();
        assert!(
            sea_err < elev_err,
            "stitched edge vertex {k} radius {r} is closer to full-elevation radius {wrong_radius} ({elev_err}) than to sea-level radius {expected_radius} ({sea_err}); clamp missing"
        );
    }
}

#[test]
fn macro_water_elevation_attribute_preserves_full_elevation() {
    // The clamp affects only the vertex *position*; the per-vertex ELEVATION
    // attribute must still report the full (unclamped) elevation so material
    // shading and depth bands remain correct.
    let sea_level = 0.0;
    let field = MacroWaterField {
        low_freq_elev: -0.5,
        elevation: 0.8,
    };
    let radius = 12000.0;
    let key = CellKey {
        face: 0,
        i: 0,
        j: 0,
        lod: 0,
    };
    let mesh = generate_chunk_mesh_with_field(key, radius, ELEVATION_SCALE, sea_level, &field);

    let elevations = match mesh.attribute(ATTRIBUTE_ELEVATION).expect("elevation attr") {
        VertexAttributeValues::Float32(values) => values,
        _ => panic!("expected Float32 elevations"),
    };

    for (i, &e) in elevations.iter().enumerate() {
        assert!(
            (e as f64 - field.elevation).abs() < 1e-5,
            "vertex {i} ELEVATION attribute {e} != full elevation {}",
            field.elevation
        );
    }
}
