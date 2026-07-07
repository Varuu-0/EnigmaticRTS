#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_position_local_to_clip}

struct AtmosphereUniform {
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    sun_x: f32,
    sun_y: f32,
    sun_z: f32,
    planet_radius: f32,
    atmosphere_radius: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> atmosphere: AtmosphereUniform;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
};

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let model = get_world_from_local(in.instance_index);
    let wp = mesh_position_local_to_world(model, vec4<f32>(in.position, 1.0));
    out.world_position = wp.xyz;
    out.clip_position = mesh_position_local_to_clip(model, vec4<f32>(in.position, 1.0));
    return out;
}

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
};

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let dir = normalize(input.world_position);
    let view_dir = normalize(vec3<f32>(atmosphere.camera_x, atmosphere.camera_y, atmosphere.camera_z) - input.world_position);
    let sun_dir = normalize(vec3<f32>(atmosphere.sun_x, atmosphere.sun_y, atmosphere.sun_z));

    let cos_zenith = dot(dir, view_dir);
    let optical_depth = sqrt(max(1.0 - cos_zenith * cos_zenith, 0.0));

    let rayleigh = vec3<f32>(0.3, 0.6, 1.0) * optical_depth * 0.6;

    let sun_dot = dot(dir, sun_dir);
    let sunset = pow(max(1.0 - abs(sun_dot), 0.0), 4.0)
        * vec3<f32>(1.0, 0.5, 0.2) * optical_depth * 0.5;

    let mie = pow(max(dot(view_dir, sun_dir), 0.0), 32.0)
        * vec3<f32>(1.0, 0.9, 0.7) * 0.3;

    let color = rayleigh + sunset + mie;
    return vec4<f32>(color, optical_depth * 0.7);
}
