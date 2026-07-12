use bevy::asset::RenderAssetUsages;
use bevy::render::mesh::{Indices, Mesh, MeshVertexAttribute};
use bevy::render::render_resource::{PrimitiveTopology, VertexFormat};
use er_core::config::{CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES};
use er_core::math::{cell_size, cells_per_edge, uv_to_dir, CellKey, FACE_U, FACE_V};
use er_world::biome::{biome, elevation_low_freq, moisture, temperature};
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

pub const ATTRIBUTE_ELEVATION: MeshVertexAttribute =
    MeshVertexAttribute::new("Elevation", 988540923, VertexFormat::Float32);

pub const ATTRIBUTE_NORMAL: MeshVertexAttribute =
    MeshVertexAttribute::new("Normal", 988540924, VertexFormat::Float32x3);

pub const ATTRIBUTE_TEMPERATURE: MeshVertexAttribute =
    MeshVertexAttribute::new("Temperature", 988540925, VertexFormat::Float32);

// Skirts only conceal sub-pixel LOD cracks. Scaling them directly with a root
// chunk makes their radial walls visible beyond the planet silhouette.
const SKIRT_CELL_DEPTH_FRACTION: f64 = 0.02;

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
    let temp = temperature(dir, elev, planet_params, climate_noise);
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
        warped_dir: [
            split.warped_dir.x as f32,
            split.warped_dir.y as f32,
            split.warped_dir.z as f32,
        ],
        moisture: moist as f32,
        biome: b,
        mountain_influence: split.mountain_influence as f32,
        temperature: temp as f32,
    }
}

struct VertexData {
    low_freq: f32,
    warped_dir: [f32; 3],
    moisture: f32,
    elevation: f64,
    temperature: f32,
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
            elevation: c.elevation,
            temperature: c.temperature,
        }
    } else {
        let split = elevation_low_freq(dir, noise, elev_params);
        let moist = moisture(dir, split.mountain_influence, planet_params, climate_noise);
        let elev = elevation(dir, noise, elev_params);
        let temp = temperature(dir, elev, planet_params, climate_noise);
        VertexData {
            low_freq: split.low_freq_elev as f32,
            warped_dir: [
                split.warped_dir.x as f32,
                split.warped_dir.y as f32,
                split.warped_dir.z as f32,
            ],
            moisture: moist as f32,
            elevation: elev,
            temperature: temp as f32,
        }
    }
}

fn cached_elevation(
    dir: DVec3,
    noise: &ElevationNoise,
    elev_params: &ElevationParams,
    planet_params: &PlanetParams,
    climate_noise: &ClimateNoise,
    cache: Option<&WorldCache>,
) -> f64 {
    if let Some(cache) = cache {
        cache
            .get_or_insert(dir, || {
                compute_cached_vertex(dir, noise, elev_params, planet_params, climate_noise)
            })
            .elevation
    } else {
        elevation(dir, noise, elev_params)
    }
}

fn compute_surface_normal(
    dir: DVec3,
    elev: f64,
    radius: f64,
    elevation_scale: f32,
    noise: &ElevationNoise,
    elev_params: &ElevationParams,
    planet_params: &PlanetParams,
    climate_noise: &ClimateNoise,
    cache: Option<&WorldCache>,
) -> DVec3 {
    let eps = 0.0008;
    let ref_vec = if dir.y.abs() > 0.99 {
        DVec3::new(1.0, 0.0, 0.0)
    } else {
        DVec3::new(0.0, 1.0, 0.0)
    };
    let t1 = ref_vec.cross(dir).normalize();
    let t2 = dir.cross(t1);

    let d1 = (dir + t1 * eps).normalize();
    let d2 = (dir + t2 * eps).normalize();

    let e1 = cached_elevation(d1, noise, elev_params, planet_params, climate_noise, cache);
    let e2 = cached_elevation(d2, noise, elev_params, planet_params, climate_noise, cache);

    let sr0 = radius + elev * elevation_scale as f64;
    let sr1 = radius + e1 * elevation_scale as f64;
    let sr2 = radius + e2 * elevation_scale as f64;

    let p0 = dir * sr0;
    let p1 = d1 * sr1;
    let p2 = d2 * sr2;

    (p1 - p0).cross(p2 - p0).normalize()
}

pub fn generate_chunk_mesh(
    key: CellKey,
    radius: f64,
    elevation_scale: f32,
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

    let skirt_depth = (cell_size(key.lod, radius) * SKIRT_CELL_DEPTH_FRACTION)
        .min(elevation_scale.abs().max(1.0) as f64);

    let total = n * n + 4 * n;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(total);
    let mut morphs: Vec<f32> = Vec::with_capacity(total);
    let mut grids: Vec<[u32; 2]> = Vec::with_capacity(total);
    let mut low_freq_elevs: Vec<f32> = Vec::with_capacity(total);
    let mut warped_dirs: Vec<[f32; 3]> = Vec::with_capacity(total);
    let mut moisture_lows: Vec<f32> = Vec::with_capacity(total);
    let mut elevations: Vec<f32> = Vec::with_capacity(total);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(total);
    let mut temperatures: Vec<f32> = Vec::with_capacity(total);

    let mut surf_low_freq: Vec<f32> = vec![0.0; n * n];
    let mut surf_warped_dir: Vec<[f32; 3]> = vec![[0.0; 3]; n * n];
    let mut surf_moisture: Vec<f32> = vec![0.0; n * n];
    let mut surf_elev: Vec<f64> = vec![0.0; n * n];
    let mut surf_temp: Vec<f32> = vec![0.0; n * n];
    let mut surf_normals: Vec<[f32; 3]> = vec![[0.0; 3]; n * n];

    for gj in 0..n {
        for gi in 0..n {
            let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
            let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
            let dir = uv_to_dir(key.face, u, v);
            let vd = vertex_data(dir, noise, elev_params, planet_params, climate_noise, cache);

            let surf_idx = gj * n + gi;
            let lf = vd.low_freq;
            let wd = vd.warped_dir;
            surf_low_freq[surf_idx] = lf;
            surf_warped_dir[surf_idx] = wd;
            surf_moisture[surf_idx] = vd.moisture;
            surf_elev[surf_idx] = vd.elevation;
            surf_temp[surf_idx] = vd.temperature;

            let surface_radius = radius + vd.elevation * elevation_scale as f64;
            let pos = dir * surface_radius;
            let normal = compute_surface_normal(
                dir,
                vd.elevation,
                radius,
                elevation_scale,
                noise,
                elev_params,
                planet_params,
                climate_noise,
                cache,
            );
            let normal = [normal.x as f32, normal.y as f32, normal.z as f32];
            surf_normals[surf_idx] = normal;

            positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
            morphs.push(1.0);
            grids.push([gi as u32, gj as u32]);
            low_freq_elevs.push(lf);
            warped_dirs.push(wd);
            moisture_lows.push(vd.moisture);
            elevations.push(vd.elevation as f32);
            normals.push(normal);
            temperatures.push(vd.temperature);
        }
    }

    let surface_count = n * n;

    let top_skirt = surface_count;
    for gi in 0..n {
        let surf_idx = gi;
        let se = surf_elev[surf_idx];
        let surface_radius = radius + se * elevation_scale as f64;
        let skirt_radius = surface_radius - skirt_depth;
        let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u, v_min);
        let pos = dir * skirt_radius;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
        grids.push([gi as u32, 0]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        warped_dirs.push(surf_warped_dir[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
        elevations.push(se as f32);
        normals.push(surf_normals[surf_idx]);
        temperatures.push(surf_temp[surf_idx]);
    }

    let bot_skirt = top_skirt + n;
    for gi in 0..n {
        let surf_idx = (n - 1) * n + gi;
        let se = surf_elev[surf_idx];
        let surface_radius = radius + se * elevation_scale as f64;
        let skirt_radius = surface_radius - skirt_depth;
        let u = u_min + (u_max - u_min) * (gi as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u, v_max);
        let pos = dir * skirt_radius;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
        grids.push([gi as u32, n1]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        warped_dirs.push(surf_warped_dir[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
        elevations.push(se as f32);
        normals.push(surf_normals[surf_idx]);
        temperatures.push(surf_temp[surf_idx]);
    }

    let left_skirt = bot_skirt + n;
    for gj in 0..n {
        let surf_idx = gj * n;
        let se = surf_elev[surf_idx];
        let surface_radius = radius + se * elevation_scale as f64;
        let skirt_radius = surface_radius - skirt_depth;
        let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u_min, v);
        let pos = dir * skirt_radius;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
        grids.push([0, gj as u32]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        warped_dirs.push(surf_warped_dir[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
        elevations.push(se as f32);
        normals.push(surf_normals[surf_idx]);
        temperatures.push(surf_temp[surf_idx]);
    }

    let right_skirt = left_skirt + n;
    for gj in 0..n {
        let surf_idx = gj * n + (n - 1);
        let se = surf_elev[surf_idx];
        let surface_radius = radius + se * elevation_scale as f64;
        let skirt_radius = surface_radius - skirt_depth;
        let v = v_min + (v_max - v_min) * (gj as f64 / (n - 1) as f64);
        let dir = uv_to_dir(key.face, u_max, v);
        let pos = dir * skirt_radius;
        positions.push([pos.x as f32, pos.y as f32, pos.z as f32]);
        morphs.push(0.0);
        grids.push([n1, gj as u32]);
        low_freq_elevs.push(surf_low_freq[surf_idx]);
        warped_dirs.push(surf_warped_dir[surf_idx]);
        moisture_lows.push(surf_moisture[surf_idx]);
        elevations.push(se as f32);
        normals.push(surf_normals[surf_idx]);
        temperatures.push(surf_temp[surf_idx]);
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
    mesh.insert_attribute(ATTRIBUTE_WARPED_DIR, warped_dirs);
    mesh.insert_attribute(ATTRIBUTE_MOISTURE_LOW, moisture_lows);
    mesh.insert_attribute(ATTRIBUTE_ELEVATION, elevations);
    mesh.insert_attribute(ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(ATTRIBUTE_TEMPERATURE, temperatures);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
