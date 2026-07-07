#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_position_local_to_clip}

struct SunUniform {
    color_r: f32,
    color_g: f32,
    color_b: f32,
    intensity: f32,
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

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let n = normalize(input.local_pos);
    let glow = pow(max(1.0 - length(n.xy), 0.0), 2.0) * 0.5;
    let color = vec3<f32>(sun_mat.color_r, sun_mat.color_g, sun_mat.color_b) * sun_mat.intensity * (1.0 + glow);
    return vec4<f32>(color, 1.0);
}
