#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_position_local_to_clip}

struct CloudUniform {
    sun_dir_x: f32,
    sun_dir_y: f32,
    sun_dir_z: f32,
    time: f32,
    camera_pos_x: f32,
    camera_pos_y: f32,
    camera_pos_z: f32,
    planet_radius: f32,
    render_origin_x: f32,
    render_origin_y: f32,
    render_origin_z: f32,
    _pad: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> cloud: CloudUniform;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput
    let model = get_world_from_local(in.instance_index)
    let wp = mesh_position_local_to_world(model, vec4<f32>(in.position, 1.0))
    out.world_position = wp.xyz
    out.clip_position = mesh_position_local_to_clip(model, vec4<f32>(in.position, 1.0))
    return out
}

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
}

fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453)
}

fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p)
    let f = fract(p)
    let u = f * f * (3.0 - 2.0 * f)
    let n000 = hash3(i + vec3<f32>(0.0, 0.0, 0.0))
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0))
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0))
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0))
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0))
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0))
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0))
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0))
    let nx00 = mix(n000, n100, u.x)
    let nx10 = mix(n010, n110, u.x)
    let nx01 = mix(n001, n101, u.x)
    let nx11 = mix(n011, n111, u.x)
    let nxy0 = mix(nx00, nx10, u.y)
    let nxy1 = mix(nx01, nx11, u.y)
    return mix(nxy0, nxy1, u.z)
}

fn fbm(p: vec3<f32>) -> f32 {
    var sum = 0.0
    var amp = 0.5
    var freq = 1.0
    for (var i = 0; i < 6; i = i + 1) {
        sum = sum + vnoise(p * freq) * amp
        freq = freq * 2.1
        amp = amp * 0.5
    }
    return sum
}

fn smoothstep_f(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let render_origin = vec3<f32>(cloud.render_origin_x, cloud.render_origin_y, cloud.render_origin_z);
    let planet_dir = normalize(input.world_position + render_origin);
    let sun_dir = normalize(vec3<f32>(cloud.sun_dir_x, cloud.sun_dir_y, cloud.sun_dir_z));
    let cam_pos = vec3<f32>(cloud.camera_pos_x, cloud.camera_pos_y, cloud.camera_pos_z);
    let view_dir = normalize(cam_pos - input.world_position);

    let t = cloud.time * 0.003;

    // Large-scale cloud formations
    let cp = planet_dir * 3.0 + vec3<f32>(t, 0.0, t * 0.5);
    var n = fbm(cp);
    // Fine detail
    let n2 = fbm(planet_dir * 8.0 + vec3<f32>(t * 1.3, 0.0, t * 0.7));
    n = n * 0.7 + n2 * 0.3;

    let density = smoothstep_f(0.42, 0.6, n);

    if (density < 0.01) {
        discard;
    }

    // Sun lighting on clouds
    let sun_facing = max(dot(planet_dir, sun_dir), 0.0);
    let sun_up = max(sun_dir.y, 0.0);
    let cloud_light = mix(0.25, 1.0, sun_facing);

    let cloud_color = mix(
        vec3<f32>(0.35, 0.38, 0.45),
        vec3<f32>(0.9, 0.92, 0.95),
        cloud_light * (0.5 + sun_up * 0.5),
    );

    // Edge fade at planet limb
    let cos_view = max(dot(planet_dir, view_dir), 0.0);
    let edge_fade = smoothstep_f(0.0, 0.12, cos_view);

    let alpha = density * edge_fade * 0.88;

    return vec4<f32>(cloud_color, alpha);
}
