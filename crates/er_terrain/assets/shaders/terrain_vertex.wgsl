#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_position_local_to_clip}

struct TerrainMaterialUniform {
    seed: i32,
    sea_level: f32,
    continental_freq: f32,
    continental_amp: f32,
    continental_octaves: i32,
    mountain_freq: f32,
    mountain_amp: f32,
    mountain_octaves: i32,
    hill_freq: f32,
    hill_amp: f32,
    hill_octaves: i32,
    detail_freq: f32,
    detail_amp: f32,
    detail_octaves: i32,
    warp_freq: f32,
    warp_amp: f32,
    lacunarity: f32,
    gain: f32,
    planet_radius: f32,
    elevation_scale: f32,
    face: i32,
    u_min: f32,
    u_max: f32,
    v_min: f32,
    v_max: f32,
    chunk_depth: i32,
    neighbor_depth_0: f32,
    neighbor_depth_1: f32,
    neighbor_depth_2: f32,
    neighbor_depth_3: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: TerrainMaterialUniform;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) morph: f32,
    @location(2) grid: vec2<u32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) elevation: f32,
};

// FACE_CORNER/FACE_U/FACE_V + uv_to_dir live in spherify.wgsl (prepended before
// this file); kept in parity with er_core::math by tests/shader_parity.rs.

fn make_elev_params(m: TerrainMaterialUniform) -> ElevationParams {
    var p: ElevationParams;
    p.seed = m.seed;
    p.sea_level = m.sea_level;
    p.continental_freq = m.continental_freq;
    p.continental_amp = m.continental_amp;
    p.continental_octaves = m.continental_octaves;
    p.mountain_freq = m.mountain_freq;
    p.mountain_amp = m.mountain_amp;
    p.mountain_octaves = m.mountain_octaves;
    p.hill_freq = m.hill_freq;
    p.hill_amp = m.hill_amp;
    p.hill_octaves = m.hill_octaves;
    p.detail_freq = m.detail_freq;
    p.detail_amp = m.detail_amp;
    p.detail_octaves = m.detail_octaves;
    p.warp_freq = m.warp_freq;
    p.warp_amp = m.warp_amp;
    p.lacunarity = m.lacunarity;
    p.gain = m.gain;
    p._pad0 = 0.0;
    p._pad1 = 0.0;
    return p;
}

struct GridResult {
    pos: vec3<f32>,
    elev: f32,
}

fn grid(gi: u32, gj: u32, m: TerrainMaterialUniform, ep: ElevationParams) -> GridResult {
    let u = m.u_min + (m.u_max - m.u_min) * (f32(gi) / 16.0);
    let v = m.v_min + (m.v_max - m.v_min) * (f32(gj) / 16.0);
    let d = uv_to_dir(m.face, u, v);
    let e = compute_elevation(d, ep);
    var r: GridResult;
    r.pos = d * (m.planet_radius + e * m.elevation_scale);
    r.elev = e;
    return r;
}

struct StitchResult {
    pos: vec3<f32>,
    elev: f32,
}

// Edge stitch: when the neighbor across an edge is coarser, collapse this chunk's
// in-between edge vertices onto the coarser grid so no T-junction / crack remains.
// Both position AND elevation are interpolated so the fragment color matches the
// coarse neighbor exactly at the shared edge. Only surface verts (morph ~ 1) call
// this; skirt verts (morph = 0) are unaffected.
fn stitch(gi: u32, gj: u32, base_pos: vec3<f32>, base_elev: f32, m: TerrainMaterialUniform, ep: ElevationParams) -> StitchResult {
    let cd = m.chunk_depth;

    // NegU edge (gi == 0), along-edge index = gj, neighbor_depth_0.
    if (gi == 0u) {
        let step = 1u << u32(clamp(cd - i32(m.neighbor_depth_0), i32(0), i32(4)));
        if (step > 1u && (gj % step) != 0u) {
            let k_lo = (gj / step) * step;
            let k_hi = min(k_lo + step, 16u);
            let t = f32(gj - k_lo) / f32(step);
            let a = grid(0u, k_lo, m, ep);
            let b = grid(0u, k_hi, m, ep);
            var r: StitchResult;
            r.pos = mix(a.pos, b.pos, t);
            r.elev = mix(a.elev, b.elev, t);
            return r;
        }
    }
    // PosU edge (gi == 16), along-edge index = gj, neighbor_depth_1.
    if (gi == 16u) {
        let step = 1u << u32(clamp(cd - i32(m.neighbor_depth_1), i32(0), i32(4)));
        if (step > 1u && (gj % step) != 0u) {
            let k_lo = (gj / step) * step;
            let k_hi = min(k_lo + step, 16u);
            let t = f32(gj - k_lo) / f32(step);
            let a = grid(16u, k_lo, m, ep);
            let b = grid(16u, k_hi, m, ep);
            var r: StitchResult;
            r.pos = mix(a.pos, b.pos, t);
            r.elev = mix(a.elev, b.elev, t);
            return r;
        }
    }
    // NegV edge (gj == 0), along-edge index = gi, neighbor_depth_2.
    if (gj == 0u) {
        let step = 1u << u32(clamp(cd - i32(m.neighbor_depth_2), i32(0), i32(4)));
        if (step > 1u && (gi % step) != 0u) {
            let k_lo = (gi / step) * step;
            let k_hi = min(k_lo + step, 16u);
            let t = f32(gi - k_lo) / f32(step);
            let a = grid(k_lo, 0u, m, ep);
            let b = grid(k_hi, 0u, m, ep);
            var r: StitchResult;
            r.pos = mix(a.pos, b.pos, t);
            r.elev = mix(a.elev, b.elev, t);
            return r;
        }
    }
    // PosV edge (gj == 16), along-edge index = gi, neighbor_depth_3.
    if (gj == 16u) {
        let step = 1u << u32(clamp(cd - i32(m.neighbor_depth_3), i32(0), i32(4)));
        if (step > 1u && (gi % step) != 0u) {
            let k_lo = (gi / step) * step;
            let k_hi = min(k_lo + step, 16u);
            let t = f32(gi - k_lo) / f32(step);
            let a = grid(k_lo, 16u, m, ep);
            let b = grid(k_hi, 16u, m, ep);
            var r: StitchResult;
            r.pos = mix(a.pos, b.pos, t);
            r.elev = mix(a.elev, b.elev, t);
            return r;
        }
    }
    var r: StitchResult;
    r.pos = base_pos;
    r.elev = base_elev;
    return r;
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;

    let model = get_world_from_local(in.instance_index);

    let dir = normalize(in.position);

    let elev_params = make_elev_params(material);
    let elev = compute_elevation(dir, elev_params);

    let surface_radius = material.planet_radius + elev * material.elevation_scale;
    var displaced = dir * surface_radius;
    var final_elev = elev;

    if (in.morph < 0.5) {
        let cells_per_edge = f32(1u << u32(material.chunk_depth));
        let chunk_size = material.planet_radius * 1.5707963 / cells_per_edge;
        displaced = dir * (surface_radius - chunk_size * 0.2);
    }

    if (in.morph > 0.5) {
        let s = stitch(in.grid.x, in.grid.y, displaced, elev, material, elev_params);
        displaced = s.pos;
        final_elev = s.elev;
    }

    let world_pos = mesh_position_local_to_world(model, vec4<f32>(displaced, 1.0));
    out.world_position = world_pos.xyz;
    out.clip_position = mesh_position_local_to_clip(model, vec4<f32>(displaced, 1.0));
    out.elevation = final_elev;

    return out;
}
