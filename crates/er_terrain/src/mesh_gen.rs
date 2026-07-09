use bevy::render::mesh::{Indices, Mesh, MeshVertexAttribute};
use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::{PrimitiveTopology, VertexFormat};
use er_core::config::{CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES};
use er_core::math::{cell_size, cells_per_edge, uv_to_dir, CellKey, FACE_U, FACE_V};
use er_world::biome::{biome, elevation_low_freq, moisture};
use er_world::cache::{CachedWorldData, WorldCache};
use er_world::elevation::{elevation, ElevationNoise, ElevationParams};
use er_world::params::{ClimateNoise, PlanetParams};
use glam::DVec3;

pub const ATTRIBUTE_MORPH: MeshVertexAttribute =
    MeshVertexAttribute::new("Morph", 988540918, VertexFormat::Float32);

pub const ATTRIBUTE_GRID: MeshVertexAttribute =
    MeshVertexAttribute::new("Grid", 988540919, VertexFormat::Uint32x2);

pub const ATTRIBUTE_LOW_FREQ_ELEV: MeshVertexAttribute =
    MeshVertexAttribute::new("LowFreqElev", 988540920, VertexFormat::Float32);

pub const ATTRIBUTE_WARPED_DIR: MeshVertexAttribute =
    MeshVertexAttribute::new("WarpedDir", 988540921, VertexFormat::Float32x3);

pub const ATTRIBUTE_MOISTURE_LOW: MeshVertexAttribute =
    MeshVertexAttribute::new("MoistureLow", 988540922, VertexFormat::Float32);

fn append_tri(indices: &mut Vec<u32>, a: u32, b: u32, c: u32, flip: bool) {
    if flip {
        indices.extend_from_slice(&[a, c, b]);
    } else {
        indices.extend_from_slice(&[a, b, c]);
    }
}

fn compute_cached_vertex(
    dir: DVec3,
    noise: &ElevationNoise,
    elev_params: &ElevationParams,
    planet_params: &PlanetParams,
    climate_noise: &ClimateNoise,
) -> CachedWorldData {
    let split = elevation_low_freq(dir, noise, elev_params);
    let moist = moisture(dir, split.mountain_influence, planet_params, climate_noise);
    let elev = elevation(dir, noise, elev_params);
    let b = biome(
        dir,
        elev,
        split.low_freq_elev,
        split.mountain_influence,
        planet_params,
        climate_noise,
    );
    CachedWorldData {
        elevation: elev,
        low_freq_elev: split.low_freq_elev as f32,
        warped_dir: [split.warped_dir.x as f32, split.warped_dir.y as f32, split.warped_dir.z as f32],
        moisture: moist as f32,
        biome: b,
        mountain_influence: split.mountain_influence as f32,
    }
}

struct VertexData {
    low_freq: f32,
    warped_dir: [f32; 3],
    moisture: f32,
}

fn vertex_data(
    dir: DVec3,
    noise: &ElevationNoise,
    elev_params: &ElevationParams,
    planet_params: &PlanetParams,
    climate_noise: &ClimateNoise,
    cache: Option<&WorldCache>,
) -> VertexData {
    if let Some(cache) = cache {
        let c = cache.get_or_insert(dir, || {
            compute_cached_vertex(dir, noise, elev_params, planet_params, climate_noise)
        });
        VertexData {
            low_freq: c.low_freq_elev,
            warped_dir: c.warped_dir,
            moisture: c.moisture,
        }
    } else {
        let split = elevation_low_freq(dir, noise, elev_params);
        let moist = moisture(dir, split.mountain_influence, planet_params, climate_noise);
        VertexData {
            low_freq: split.low_freq_elev as f32,
            warped_dir: [split.warped_dir.x as f32, split.warped_dir.y as f32, split.warped_dir.z as f32],
            moisture: moist as f32,
        }
    }
}

pub fn generate_chunk_mesh(
    key: CellKey,
    radius: f64,
    noise: &ElevationNoise,
    elev_params: &ElevationParams,
    planet_params: &PlanetParams,
    climate_noise: &ClimateNoise,
    cache: Option<&WorldCache>,
) -> Mesh {
    let n = CHUNK_VERT_RES as usize;
    let quads = CHUNK_QUADS_PER_EDGE as usize;
    let cells = cells_per_edge(key.lod) as f64;
    let n1 = (n - 1) as u32;

    let u_min = key.i as f64 / cells;
    let u_max = (key.i + 1) as f64 / cells;
    let v_min = key.j as f64 / cells;
    let v_max = (key.j + 1) as f64 / cells;

    let skirt_depth = cell_size(key.lod, radius) * 0.2;
    let skirt_radius = radius - skirt_depth;

    let total = n * n + 4 * n;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(total);
    let mut morphs: Vec<f32> = Vec::with_capacity(total);
    let mut grids: Vec<[u32; 2]> = Vec::with_capacity(total);
    let mut low_freq_elevs: Vec<f32> = Vec::with_capacity(total);
    let mut warped_dirs: Vec<[f32; 3]> = Vec::with_capacity(total);
    let mut moisture_lows: Vec<f32> = Vec::with_capacity(total);

    let mut surf_low_freq: Vec<f32> = vec![0.0; n * n];
    let mut surf_warped_dir: Vec<[f32; 3]> = vec![[0.0; 3]; n * n];
    let mut surf_moisture: Vec<f32> = vec![0.0; n * n];

    for gj in 0..n {
        for gi in 0..n {
            let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
            let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
            let dir = uv_to_dir(key.face, u, v);
            let pos = dir * radius;
            let vd = vertex_data(dir, noise, elev_params, planet_params, climate_noise, cache);

            let surf_idx = gj * n + gi;
            let lf = vd.low_freq;
            let wd = vd.warped_dir;
            surf_low_freq[surf_idx] = lf;
            surf_warped_dir[surf_idx] = wd;
            surf_moisture[surf_idx] = vd.moisture;

            positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
            morphs.push(1.0);
            grids.push([gi as u32, gj as u32]);
            low_freq_elevs.push(lf);
            warped_dirs.push(wd);
            moisture_lows.push(vd.moisture);
        }
    }

    let surface_count = n * n;

    let top_skirt = surface_count;
    for gi in 0..n {
        let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u, v_min);
        let pos = dir * skirt_radius;
        let surf_idx = gi;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
        grids.push([gi as u32, 0]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        warped_dirs.push(surf_warped_dir[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
    }

    let bot_skirt = top_skirt + n;
    for gi in 0..n {
        let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u, v_max);
        let pos = dir * skirt_radius;
        let surf_idx = (n - 1) * n + gi;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
        grids.push([gi as u32, n1]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        warped_dirs.push(surf_warped_dir[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
    }

    let left_skirt = bot_skirt + n;
    for gj in 0..n {
        let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u_min, v);
        let pos = dir * skirt_radius;
        let surf_idx = gj * n;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
        grids.push([0, gj as u32]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        warped_dirs.push(surf_warped_dir[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
    }

    let right_skirt = left_skirt + n;
    for gj in 0..n {
        let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u_max, v);
        let pos = dir * skirt_radius;
        let surf_idx = gj * n + (n - 1);
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
        grids.push([n1, gj as u32]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        warped_dirs.push(surf_warped_dir[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
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
        append_tri(&mut indices, g0, s0, s1, flip_winding);
        append_tri(&mut indices, g0, s1, g1, flip_winding);
    }

    for gj in 0..quads {
        let g0 = (gj * n) as u32;
        let g1 = ((gj + 1) * n) as u32;
        let s0 = (left_skirt + gj) as u32;
        let s1 = (left_skirt + gj + 1) as u32;
        append_tri(&mut indices, g0, s0, s1, flip_winding);
        append_tri(&mut indices, g0, s1, g1, flip_winding);
    }

    for gj in 0..quads {
        let g0 = (gj * n + (n - 1)) as u32;
        let g1 = ((gj + 1) * n + (n - 1)) as u32;
        let s0 = (right_skirt + gj) as u32;
        let s1 = (right_skirt + gj + 1) as u32;
        append_tri(&mut indices, g0, s1, s0, flip_winding);
        append_tri(&mut indices, g0, g1, s1, flip_winding);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(ATTRIBUTE_MORPH, morphs);
    mesh.insert_attribute(ATTRIBUTE_GRID, grids);
    mesh.insert_attribute(ATTRIBUTE_LOW_FREQ_ELEV, low_freq_elevs);
    mesh.insert_attribute(ATTRIBUTE_WARPED_DIR, warped_dirs);
    mesh.insert_attribute(ATTRIBUTE_MOISTURE_LOW, moisture_lows);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
