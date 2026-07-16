#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_position_local_to_clip}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) morph: f32,
    @location(2) grid: vec2<u32>,
    @location(3) low_freq_elev: f32,
    @location(4) warped_dir: vec3<f32>,
    @location(5) moisture_low: f32,
    @location(6) elevation: f32,
    @location(7) normal: vec3<f32>,
    @location(8) temperature: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) elevation: f32,
    @location(2) dir: vec3<f32>,
    @location(3) moisture: f32,
    @location(4) low_freq_elev: f32,
    @location(5) temperature: f32,
    @location(6) normal: vec3<f32>,
    @location(7) morph: f32,
};

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let model = get_world_from_local(in.instance_index);
    let displaced = in.position;
    let final_elev = in.elevation;

    let world_pos = mesh_position_local_to_world(model, vec4<f32>(displaced, 1.0));
    out.world_position = world_pos.xyz;
    out.clip_position = mesh_position_local_to_clip(model, vec4<f32>(displaced, 1.0));
    out.elevation = final_elev;
    out.dir = normalize(displaced);
    out.moisture = in.moisture_low;
    out.low_freq_elev = in.low_freq_elev;
    out.temperature = in.temperature;
    out.normal = in.normal;
    out.morph = in.morph;
    return out;
}
