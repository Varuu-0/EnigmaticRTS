use bevy::asset::RenderAssetUsages;
use bevy::render::mesh::{Indices, Mesh, MeshVertexAttribute};
use bevy::render::render_resource::{PrimitiveTopology, VertexFormat};
use er_core::config::{CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES};
use er_core::math::{
    cell_size, cell_to_dir, cells_per_edge, uv_to_dir, CellKey, NeighborSide, FACE_CORNER, FACE_U,
    FACE_V,
};
use er_world::terrain_field::{TerrainField, TerrainSample};
use glam::DVec3;

pub const ATTRIBUTE_MORPH: MeshVertexAttribute =
    MeshVertexAttribute::new("Morph", 988540918, VertexFormat::Float32);

pub const ATTRIBUTE_GRID: MeshVertexAttribute =
    MeshVertexAttribute::new("Grid", 988540919, VertexFormat::Uint32x2);

pub const ATTRIBUTE_LOW_FREQ_ELEV: MeshVertexAttribute =
    MeshVertexAttribute::new("LowFreqElev", 988540920, VertexFormat::Float32);

pub const ATTRIBUTE_MOISTURE_LOW: MeshVertexAttribute =
    MeshVertexAttribute::new("MoistureLow", 988540922, VertexFormat::Float32);

pub const ATTRIBUTE_ELEVATION: MeshVertexAttribute =
    MeshVertexAttribute::new("Elevation", 988540923, VertexFormat::Float32);

pub const ATTRIBUTE_NORMAL: MeshVertexAttribute =
    MeshVertexAttribute::new("Normal", 988540924, VertexFormat::Float32x3);

pub const ATTRIBUTE_TEMPERATURE: MeshVertexAttribute =
    MeshVertexAttribute::new("Temperature", 988540925, VertexFormat::Float32);

pub const ATTRIBUTE_DRAINAGE: MeshVertexAttribute =
    MeshVertexAttribute::new("Drainage", 988540926, VertexFormat::Float32);

pub const ATTRIBUTE_CURVATURE: MeshVertexAttribute =
    MeshVertexAttribute::new("Curvature", 988540927, VertexFormat::Float32);

pub const ATTRIBUTE_DIRECTION: MeshVertexAttribute =
    MeshVertexAttribute::new("Direction", 988540928, VertexFormat::Float32x3);

// Skirts only conceal sub-pixel LOD cracks. Scaling them directly with a root
// chunk makes their radial walls visible beyond the planet silhouette.
const SKIRT_CELL_DEPTH_FRACTION: f64 = 0.02;
const EARTH_MATERIAL_SAMPLE_SPACING_M: f64 = 100.0;
pub type StitchNeighbors = [Option<CellKey>; 4];

const NEIGHBOR_SIDES: [NeighborSide; 4] = [
    NeighborSide::NegU,
    NeighborSide::PosU,
    NeighborSide::NegV,
    NeighborSide::PosV,
];

pub(crate) fn normal_sample_spacing_m(chunk_lod: u8, radius: f64) -> f64 {
    let vertex_spacing = cell_size(chunk_lod, radius) / CHUNK_QUADS_PER_EDGE as f64;
    if radius >= 1_000_000.0 {
        // Earth LODs overlap in screen space. A stable physical footprint
        // prevents normal and curvature masks from revealing chunk borders.
        EARTH_MATERIAL_SAMPLE_SPACING_M
    } else {
        vertex_spacing
    }
}

fn central_diff_eps_radians(chunk_lod: u8, radius: f64) -> f64 {
    (normal_sample_spacing_m(chunk_lod, radius) / radius).clamp(1e-8, 0.25)
}

fn append_tri(indices: &mut Vec<u32>, a: u32, b: u32, c: u32, flip: bool) {
    if flip {
        indices.extend_from_slice(&[a, c, b]);
    } else {
        indices.extend_from_slice(&[a, b, c]);
    }
}

struct VertexData {
    low_freq: f32,
    moisture: f32,
    elevation: f64,
    temperature: f32,
    drainage: f32,
}

impl From<TerrainSample> for VertexData {
    fn from(value: TerrainSample) -> Self {
        Self {
            low_freq: value.low_freq_elev,
            moisture: value.moisture,
            elevation: value.elevation,
            temperature: value.temperature,
            drainage: value.drainage,
        }
    }
}

fn compute_surface_shape(
    dir: DVec3,
    center_elevation: f64,
    radius: f64,
    elevation_scale: f32,
    field: &dyn TerrainField,
    chunk_lod: u8,
) -> (DVec3, f32) {
    let eps = central_diff_eps_radians(chunk_lod, radius);
    let ref_vec = if dir.y.abs() > 0.99 {
        DVec3::new(1.0, 0.0, 0.0)
    } else {
        DVec3::new(0.0, 1.0, 0.0)
    };
    let t1 = ref_vec.cross(dir).normalize();
    let t2 = dir.cross(t1);

    let d1_plus = (dir + t1 * eps).normalize();
    let d1_minus = (dir - t1 * eps).normalize();
    let d2_plus = (dir + t2 * eps).normalize();
    let d2_minus = (dir - t2 * eps).normalize();

    let e1_plus = field.sample(d1_plus).elevation;
    let e1_minus = field.sample(d1_minus).elevation;
    let e2_plus = field.sample(d2_plus).elevation;
    let e2_minus = field.sample(d2_minus).elevation;

    let sr1_plus = radius + e1_plus * elevation_scale as f64;
    let sr1_minus = radius + e1_minus * elevation_scale as f64;
    let sr2_plus = radius + e2_plus * elevation_scale as f64;
    let sr2_minus = radius + e2_minus * elevation_scale as f64;

    let p1 = d1_plus * sr1_plus - d1_minus * sr1_minus;
    let p2 = d2_plus * sr2_plus - d2_minus * sr2_minus;

    let normal = p1.cross(p2);
    let normal = if normal.length_squared() > f64::EPSILON {
        normal.normalize()
    } else {
        dir
    };
    let neighbor_sum = e1_plus + e1_minus + e2_plus + e2_minus;
    let laplacian_m = (neighbor_sum - center_elevation * 4.0) * elevation_scale as f64;
    let sample_step_m = eps * radius;
    // Convert the discrete Laplacian to inverse meters, then normalize it to
    // the 200 m material-detail scale so the mask remains comparable by LOD.
    let curvature = (laplacian_m / sample_step_m.powi(2) * 200.0).clamp(-1.0, 1.0) as f32;
    (normal, curvature)
}

pub fn generate_chunk_mesh(
    key: CellKey,
    radius: f64,
    elevation_scale: f32,
    field: &dyn TerrainField,
) -> Mesh {
    generate_chunk_mesh_stitched(key, radius, elevation_scale, field, [None; 4])
}

pub fn generate_chunk_mesh_stitched(
    key: CellKey,
    radius: f64,
    elevation_scale: f32,
    field: &dyn TerrainField,
    stitch_neighbors: StitchNeighbors,
) -> Mesh {
    let n = CHUNK_VERT_RES as usize;
    let quads = CHUNK_QUADS_PER_EDGE as usize;
    let cells = cells_per_edge(key.lod) as f64;
    let n1 = (n - 1) as u32;

    let u_min = key.i as f64 / cells;
    let u_max = (key.i + 1) as f64 / cells;
    let v_min = key.j as f64 / cells;
    let v_max = (key.j + 1) as f64 / cells;

    let anchor_dir = cell_to_dir(key);
    let anchor = anchor_dir * radius;

    let skirt_depth = (cell_size(key.lod, radius) * SKIRT_CELL_DEPTH_FRACTION)
        .min(elevation_scale.abs().max(1.0) as f64);

    let total = n * n + 4 * n;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(total);
    let mut morphs: Vec<f32> = Vec::with_capacity(total);
    let mut grids: Vec<[u32; 2]> = Vec::with_capacity(total);
    let mut low_freq_elevs: Vec<f32> = Vec::with_capacity(total);
    let mut moisture_lows: Vec<f32> = Vec::with_capacity(total);
    let mut elevations: Vec<f32> = Vec::with_capacity(total);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(total);
    let mut temperatures: Vec<f32> = Vec::with_capacity(total);
    let mut drainages: Vec<f32> = Vec::with_capacity(total);
    let mut curvatures: Vec<f32> = Vec::with_capacity(total);
    let mut directions: Vec<[f32; 3]> = Vec::with_capacity(total);

    let mut surf_low_freq: Vec<f32> = vec![0.0; n * n];
    let mut surf_moisture: Vec<f32> = vec![0.0; n * n];
    let mut surf_elev: Vec<f64> = vec![0.0; n * n];
    let mut surf_temp: Vec<f32> = vec![0.0; n * n];
    let mut surf_normals: Vec<[f32; 3]> = vec![[0.0; 3]; n * n];
    let mut surf_drainage: Vec<f32> = vec![0.0; n * n];
    let mut surf_curvature: Vec<f32> = vec![0.0; n * n];
    let mut surf_directions: Vec<[f32; 3]> = vec![[0.0; 3]; n * n];

    for gj in 0..n {
        for gi in 0..n {
            let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
            let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
            let dir = uv_to_dir(key.face, u, v);
            let vd = VertexData::from(field.sample(dir));

            let surf_idx = gj * n + gi;
            let lf = vd.low_freq;
            surf_low_freq[surf_idx] = lf;
            surf_moisture[surf_idx] = vd.moisture;
            surf_elev[surf_idx] = vd.elevation;
            surf_temp[surf_idx] = vd.temperature;
            surf_drainage[surf_idx] = vd.drainage;

            let surface_radius = radius + vd.elevation * elevation_scale as f64;
            let pos = dir * surface_radius;
            let (normal, curvature) =
                compute_surface_shape(dir, vd.elevation, radius, elevation_scale, field, key.lod);
            let normal = [normal.x as f32, normal.y as f32, normal.z as f32];
            surf_normals[surf_idx] = normal;
            surf_curvature[surf_idx] = curvature;
            let direction = dir.as_vec3().to_array();
            surf_directions[surf_idx] = direction;

            let local = pos - anchor;
            positions.push([local.x as f32, local.y as f32, local.z as f32]);
            morphs.push(1.0);
            grids.push([gi as u32, gj as u32]);
            low_freq_elevs.push(lf);
            moisture_lows.push(vd.moisture);
            elevations.push(vd.elevation as f32);
            normals.push(normal);
            temperatures.push(vd.temperature);
            drainages.push(vd.drainage);
            curvatures.push(curvature);
            directions.push(direction);
        }
    }

    for (side, coarse_neighbor) in NEIGHBOR_SIDES.into_iter().zip(stitch_neighbors) {
        if let Some(coarse_neighbor) = coarse_neighbor {
            assert_eq!(
                coarse_neighbor.lod + 1,
                key.lod,
                "stitching requires a 2:1 balanced quadtree"
            );
            snap_edge_to_coarse_mesh(
                &mut positions,
                key,
                side,
                coarse_neighbor,
                radius,
                elevation_scale,
                field,
            );
        }
    }

    let surface_count = n * n;

    let top_skirt = surface_count;
    for gi in 0..n {
        let surf_idx = gi;
        let se = surf_elev[surf_idx];
        let surface = anchor + DVec3::from_array(positions[surf_idx].map(f64::from));
        let pos = surface - surface.normalize() * skirt_depth;
        let local = pos - anchor;
        positions.push([local.x as f32, local.y as f32, local.z as f32]);
        morphs.push(0.0);
        grids.push([gi as u32, 0]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
        elevations.push(se as f32);
        normals.push(surf_normals[surf_idx]);
        temperatures.push(surf_temp[surf_idx]);
        drainages.push(surf_drainage[surf_idx]);
        curvatures.push(surf_curvature[surf_idx]);
        directions.push(surf_directions[surf_idx]);
    }

    let bot_skirt = top_skirt + n;
    for gi in 0..n {
        let surf_idx = (n - 1) * n + gi;
        let se = surf_elev[surf_idx];
        let surface = anchor + DVec3::from_array(positions[surf_idx].map(f64::from));
        let pos = surface - surface.normalize() * skirt_depth;
        let local = pos - anchor;
        positions.push([local.x as f32, local.y as f32, local.z as f32]);
        morphs.push(0.0);
        grids.push([gi as u32, n1]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
        elevations.push(se as f32);
        normals.push(surf_normals[surf_idx]);
        temperatures.push(surf_temp[surf_idx]);
        drainages.push(surf_drainage[surf_idx]);
        curvatures.push(surf_curvature[surf_idx]);
        directions.push(surf_directions[surf_idx]);
    }

    let left_skirt = bot_skirt + n;
    for gj in 0..n {
        let surf_idx = gj * n;
        let se = surf_elev[surf_idx];
        let surface = anchor + DVec3::from_array(positions[surf_idx].map(f64::from));
        let pos = surface - surface.normalize() * skirt_depth;
        let local = pos - anchor;
        positions.push([local.x as f32, local.y as f32, local.z as f32]);
        morphs.push(0.0);
        grids.push([0, gj as u32]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
        elevations.push(se as f32);
        normals.push(surf_normals[surf_idx]);
        temperatures.push(surf_temp[surf_idx]);
        drainages.push(surf_drainage[surf_idx]);
        curvatures.push(surf_curvature[surf_idx]);
        directions.push(surf_directions[surf_idx]);
    }

    let right_skirt = left_skirt + n;
    for gj in 0..n {
        let surf_idx = gj * n + (n - 1);
        let se = surf_elev[surf_idx];
        let surface = anchor + DVec3::from_array(positions[surf_idx].map(f64::from));
        let pos = surface - surface.normalize() * skirt_depth;
        let local = pos - anchor;
        positions.push([local.x as f32, local.y as f32, local.z as f32]);
        morphs.push(0.0);
        grids.push([n1, gj as u32]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
        elevations.push(se as f32);
        normals.push(surf_normals[surf_idx]);
        temperatures.push(surf_temp[surf_idx]);
        drainages.push(surf_drainage[surf_idx]);
        curvatures.push(surf_curvature[surf_idx]);
        directions.push(surf_directions[surf_idx]);
    }

    let mut indices: Vec<u32> = Vec::with_capacity(quads * quads * 6 + 4 * quads * 6);

    // Some cube faces have a left-handed (u,v) parameterization; without a
    // winding flip their triangles face inward and back-face culling reveals the
    // planet interior as a dark wireframe grid.
    let face_normal = uv_to_dir(key.face, 0.5, 0.5);
    let flip_winding = FACE_U[key.face as usize]
        .cross(FACE_V[key.face as usize])
        .dot(face_normal)
        < 0.0;

    for gj in 0..quads {
        for gi in 0..quads {
            let v00 = (gj * n + gi) as u32;
            let v10 = (gj * n + gi + 1) as u32;
            let v01 = ((gj + 1) * n + gi) as u32;
            let v11 = ((gj + 1) * n + gi + 1) as u32;
            append_tri(&mut indices, v00, v10, v11, flip_winding);
            append_tri(&mut indices, v00, v11, v01, flip_winding);
        }
    }

    for gi in 0..quads {
        let g0 = gi as u32;
        let g1 = (gi + 1) as u32;
        let s0 = (top_skirt + gi) as u32;
        let s1 = (top_skirt + gi + 1) as u32;
        append_tri(&mut indices, g0, s0, s1, flip_winding);
        append_tri(&mut indices, g0, s1, g1, flip_winding);
    }

    for gi in 0..quads {
        let g0 = ((n - 1) * n + gi) as u32;
        let g1 = ((n - 1) * n + gi + 1) as u32;
        let s0 = (bot_skirt + gi) as u32;
        let s1 = (bot_skirt + gi + 1) as u32;
        append_tri(&mut indices, g0, s1, s0, flip_winding);
        append_tri(&mut indices, g0, g1, s1, flip_winding);
    }

    for gj in 0..quads {
        let g0 = (gj * n) as u32;
        let g1 = ((gj + 1) * n) as u32;
        let s0 = (left_skirt + gj) as u32;
        let s1 = (left_skirt + gj + 1) as u32;
        append_tri(&mut indices, g0, s1, s0, flip_winding);
        append_tri(&mut indices, g0, g1, s1, flip_winding);
    }

    for gj in 0..quads {
        let g0 = (gj * n + (n - 1)) as u32;
        let g1 = ((gj + 1) * n + (n - 1)) as u32;
        let s0 = (right_skirt + gj) as u32;
        let s1 = (right_skirt + gj + 1) as u32;
        append_tri(&mut indices, g0, s0, s1, flip_winding);
        append_tri(&mut indices, g0, s1, g1, flip_winding);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(ATTRIBUTE_MORPH, morphs);
    mesh.insert_attribute(ATTRIBUTE_GRID, grids);
    mesh.insert_attribute(ATTRIBUTE_LOW_FREQ_ELEV, low_freq_elevs);
    mesh.insert_attribute(ATTRIBUTE_MOISTURE_LOW, moisture_lows);
    mesh.insert_attribute(ATTRIBUTE_ELEVATION, elevations);
    mesh.insert_attribute(ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(ATTRIBUTE_TEMPERATURE, temperatures);
    mesh.insert_attribute(ATTRIBUTE_DRAINAGE, drainages);
    mesh.insert_attribute(ATTRIBUTE_CURVATURE, curvatures);
    mesh.insert_attribute(ATTRIBUTE_DIRECTION, directions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn snap_edge_to_coarse_mesh(
    positions: &mut [[f32; 3]],
    key: CellKey,
    side: NeighborSide,
    coarse: CellKey,
    radius: f64,
    elevation_scale: f32,
    field: &dyn TerrainField,
) {
    debug_assert!(coarse.lod < key.lod);

    let n = CHUNK_VERT_RES as usize;
    let fine_cells = cells_per_edge(key.lod) as f64;
    let fine_u_min = key.i as f64 / fine_cells;
    let fine_u_max = (key.i + 1) as f64 / fine_cells;
    let fine_v_min = key.j as f64 / fine_cells;
    let fine_v_max = (key.j + 1) as f64 / fine_cells;
    let fine_anchor = cell_to_dir(key) * radius;

    let coarse_cells = cells_per_edge(coarse.lod) as f64;
    let coarse_u_min = coarse.i as f64 / coarse_cells;
    let coarse_u_max = (coarse.i + 1) as f64 / coarse_cells;
    let coarse_v_min = coarse.j as f64 / coarse_cells;
    let coarse_v_max = (coarse.j + 1) as f64 / coarse_cells;

    let midpoint_dir = edge_direction(
        key.face, side, fine_u_min, fine_u_max, fine_v_min, fine_v_max, 0.5,
    );
    let (mid_u, mid_v) = direction_on_face(coarse.face, midpoint_dir);
    let coarse_edge = closest_edge(
        mid_u,
        mid_v,
        coarse_u_min,
        coarse_u_max,
        coarse_v_min,
        coarse_v_max,
    );

    for k in 0..n {
        let t = k as f64 / (n - 1) as f64;
        let fine_dir = edge_direction(
            key.face, side, fine_u_min, fine_u_max, fine_v_min, fine_v_max, t,
        );
        let (coarse_u, coarse_v) = direction_on_face(coarse.face, fine_dir);
        let edge_t = match coarse_edge {
            NeighborSide::NegU | NeighborSide::PosU => {
                (coarse_v - coarse_v_min) / (coarse_v_max - coarse_v_min)
            }
            NeighborSide::NegV | NeighborSide::PosV => {
                (coarse_u - coarse_u_min) / (coarse_u_max - coarse_u_min)
            }
        }
        .clamp(0.0, 1.0);

        let coarse_grid = edge_t * CHUNK_QUADS_PER_EDGE as f64;
        let segment = coarse_grid.floor().min((CHUNK_QUADS_PER_EDGE - 1) as f64) as usize;
        let segment_t = coarse_grid - segment as f64;
        let segment_start = segment as f64 / CHUNK_QUADS_PER_EDGE as f64;
        let segment_end = (segment + 1) as f64 / CHUNK_QUADS_PER_EDGE as f64;
        let start_dir = edge_direction(
            coarse.face,
            coarse_edge,
            coarse_u_min,
            coarse_u_max,
            coarse_v_min,
            coarse_v_max,
            segment_start,
        );
        let end_dir = edge_direction(
            coarse.face,
            coarse_edge,
            coarse_u_min,
            coarse_u_max,
            coarse_v_min,
            coarse_v_max,
            segment_end,
        );
        let start = surface_position(start_dir, radius, elevation_scale, field);
        let end = surface_position(end_dir, radius, elevation_scale, field);
        let snapped = start.lerp(end, segment_t) - fine_anchor;
        positions[edge_vertex_index(side, k, n)] = snapped.as_vec3().to_array();
    }
}

fn surface_position(
    dir: DVec3,
    radius: f64,
    elevation_scale: f32,
    field: &dyn TerrainField,
) -> DVec3 {
    let elevation = field.sample(dir).elevation * elevation_scale as f64;
    dir * (radius + elevation)
}

fn edge_direction(
    face: u8,
    side: NeighborSide,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
    t: f64,
) -> DVec3 {
    match side {
        NeighborSide::NegU => uv_to_dir(face, u_min, v_min + (v_max - v_min) * t),
        NeighborSide::PosU => uv_to_dir(face, u_max, v_min + (v_max - v_min) * t),
        NeighborSide::NegV => uv_to_dir(face, u_min + (u_max - u_min) * t, v_min),
        NeighborSide::PosV => uv_to_dir(face, u_min + (u_max - u_min) * t, v_max),
    }
}

fn direction_on_face(face: u8, dir: DVec3) -> (f64, f64) {
    let axis = (face / 2) as usize;
    let sign = if face % 2 == 0 { 1.0 } else { -1.0 };
    let axis_value = match axis {
        0 => dir.x,
        1 => dir.y,
        _ => dir.z,
    };
    let cube = dir * (sign / axis_value);
    let from_corner = cube - FACE_CORNER[face as usize];
    (
        from_corner.dot(FACE_U[face as usize]) / FACE_U[face as usize].length_squared(),
        from_corner.dot(FACE_V[face as usize]) / FACE_V[face as usize].length_squared(),
    )
}

fn closest_edge(u: f64, v: f64, u_min: f64, u_max: f64, v_min: f64, v_max: f64) -> NeighborSide {
    let candidates = [
        (NeighborSide::NegU, (u - u_min).abs()),
        (NeighborSide::PosU, (u - u_max).abs()),
        (NeighborSide::NegV, (v - v_min).abs()),
        (NeighborSide::PosV, (v - v_max).abs()),
    ];
    candidates
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|candidate| candidate.0)
        .unwrap()
}

fn edge_vertex_index(side: NeighborSide, k: usize, n: usize) -> usize {
    match side {
        NeighborSide::NegU => k * n,
        NeighborSide::PosU => k * n + n - 1,
        NeighborSide::NegV => k,
        NeighborSide::PosV => (n - 1) * n + k,
    }
}
