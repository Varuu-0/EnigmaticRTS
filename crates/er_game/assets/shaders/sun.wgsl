#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_position_local_to_clip}

struct SunUniform {
    color_r: f32,
    color_g: f32,
    color_b: f32,
    intensity: f32,
    time: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sun_mat: SunUniform;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) local_pos: vec3<f32>,
};

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let model = get_world_from_local(in.instance_index);
    let wp = mesh_position_local_to_world(model, vec4<f32>(in.position, 1.0));
    out.world_position = wp.xyz;
    out.local_pos = in.position;
    out.clip_position = mesh_position_local_to_clip(model, vec4<f32>(in.position, 1.0));
    return out;
}

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) local_pos: vec3<f32>,
};

fn smoothstep_f(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let n = normalize(input.local_pos);
    let r = length(n.xy);

    // Core brightness (center of the disk)
    let core = 1.0 - smoothstep_f(0.0, 0.8, r);

    // Corona glow (soft halo)
    let corona = pow(max(1.0 - r, 0.0), 2.0) * 0.5;

    // Subtle pulsing
    let pulse = 0.5 + 0.5 * sin(sun_mat.time * 2.0);
    let pulse_factor = 1.0 + pulse * 0.05;

    // Radial lens flare streaks
    let angle = atan2(n.y, n.x);
    let streaks = sin(angle * 6.0 + sun_mat.time) * 0.5 + 0.5;
    let flare = pow(max(1.0 - r, 0.0), 4.0) * streaks * 0.1;

    let color = vec3<f32>(sun_mat.color_r, sun_mat.color_g, sun_mat.color_b)
        * sun_mat.intensity
        * (core + corona + flare)
        * pulse_factor;

    return vec4<f32>(color, 1.0);
}
