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
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: TerrainMaterialUniform;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) morph: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) elevation: f32,
};

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

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;

    let model = get_world_from_local(in.instance_index);

    let dir = normalize(in.position);

    let elev_params = make_elev_params(material);
    let elev = compute_elevation(dir, elev_params);

    let displaced = dir * (material.planet_radius + elev * material.elevation_scale * in.morph);

    let world_pos = mesh_position_local_to_world(model, vec4<f32>(displaced, 1.0));
    out.world_position = world_pos.xyz;
    out.clip_position = mesh_position_local_to_clip(model, vec4<f32>(displaced, 1.0));
    out.elevation = elev;

    return out;
}
