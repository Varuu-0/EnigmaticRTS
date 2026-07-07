#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_position_local_to_clip}

struct StarfieldUniform {
    seed: f32,
    brightness: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> starfield: StarfieldUniform;

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

fn hash13(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn star_at(dir: vec3<f32>, seed: f32) -> f32 {
    let scale = 600.0;
    let cell = floor(dir * scale);
    let h = hash13(cell + vec3<f32>(seed));
    if (h > 0.985) {
        let h2 = hash13(cell + vec3<f32>(seed + 17.0));
        return pow(h2, 6.0);
    }
    return 0.0;
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let dir = normalize(input.world_position);

    var brightness = 0.0;
    brightness = brightness + star_at(dir, starfield.seed);
    brightness = brightness + star_at(dir * 1.3, starfield.seed + 31.0) * 0.7;
    brightness = brightness + star_at(dir * 1.7, starfield.seed + 57.0) * 0.5;

    brightness = brightness * starfield.brightness;

    let star_color = vec3<f32>(0.9, 0.95, 1.0) * brightness;
    return vec4<f32>(star_color, 1.0);
}
