#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_position_local_to_clip}

struct CloudUniform {
    sun_dir_x: f32,
    sun_dir_y: f32,
    sun_dir_z: f32,
    time: f32,
    camera_pos_x: f32,
    camera_pos_y: f32,
    camera_pos_z: f32,
    opacity: f32,
    storminess: f32,
    formation: f32,
    role: f32,
    noise_seed: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> cloud: CloudUniform;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) local_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let model = get_world_from_local(in.instance_index);
    let wp = mesh_position_local_to_world(model, vec4<f32>(in.position, 1.0));
    out.world_position = wp.xyz;
    out.local_position = in.position;
    let axis_x = model[0].xyz;
    let axis_y = model[1].xyz;
    let axis_z = model[2].xyz;
    out.world_normal = normalize(
        axis_x * (in.position.x / max(dot(axis_x, axis_x), 1e-8))
        + axis_y * (in.position.y / max(dot(axis_y, axis_y), 1e-8))
        + axis_z * (in.position.z / max(dot(axis_z, axis_z), 1e-8))
    );
    out.clip_position = mesh_position_local_to_clip(model, vec4<f32>(in.position, 1.0));
    return out;
}

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) local_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
}

fn mod289_3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec4<f32>) -> vec4<f32> {
    return mod289_4(((x * 34.0) + 1.0) * x);
}

fn taylor_inverse_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

// MIT-licensed 3D simplex noise by Ian McEwan, Ashima Arts.
fn simplex_noise(v: vec3<f32>) -> f32 {
    let c = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let d = vec4<f32>(0.0, 0.5, 1.0, 2.0);
    var i = floor(v + dot(v, c.yyy));
    let x0 = v - i + dot(i, c.xxx);
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);
    let x1 = x0 - i1 + c.xxx;
    let x2 = x0 - i2 + c.yyy;
    let x3 = x0 - d.yyy;
    i = mod289_3(i);
    let p = permute(
        permute(
            permute(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
                + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0),
        ) + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0),
    );
    let n = 1.0 / 7.0;
    let ns = n * d.wyz - d.xzx;
    let j = p - 49.0 * floor(p * ns.z * ns.z);
    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);
    let x = x_ * ns.x + ns.yyyy;
    let y = y_ * ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);
    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);
    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));
    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;
    var p0 = vec3<f32>(a0.xy, h.x);
    var p1 = vec3<f32>(a0.zw, h.y);
    var p2 = vec3<f32>(a1.xy, h.z);
    var p3 = vec3<f32>(a1.zw, h.w);
    let norm = taylor_inverse_sqrt(vec4<f32>(
        dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3),
    ));
    p0 = p0 * norm.x;
    p1 = p1 * norm.y;
    p2 = p2 * norm.z;
    p3 = p3 * norm.w;
    var m = max(0.6 - vec4<f32>(
        dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3),
    ), vec4<f32>(0.0));
    m = m * m;
    return 42.0 * dot(m * m, vec4<f32>(
        dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3),
    ));
}

fn fbm(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var sample_position = p;
    let octave_rotation = mat3x3<f32>(
        vec3<f32>(0.00, 0.80, 0.60),
        vec3<f32>(-0.80, 0.36, -0.48),
        vec3<f32>(-0.60, -0.48, 0.64),
    );
    for (var i = 0; i < 6; i = i + 1) {
        sum = sum + (simplex_noise(sample_position) * 0.5 + 0.5) * amp;
        sample_position = octave_rotation * sample_position * 2.1
            + vec3<f32>(13.7, 7.1, 19.3);
        amp = amp * 0.5;
    }
    return sum;
}

fn fbm_lite(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var sample_position = p;
    let octave_rotation = mat3x3<f32>(
        vec3<f32>(0.00, 0.80, 0.60),
        vec3<f32>(-0.80, 0.36, -0.48),
        vec3<f32>(-0.60, -0.48, 0.64),
    );
    for (var i = 0; i < 3; i = i + 1) {
        sum = sum + (simplex_noise(sample_position) * 0.5 + 0.5) * amp;
        sample_position = octave_rotation * sample_position * 2.1
            + vec3<f32>(13.7, 7.1, 19.3);
        amp = amp * 0.5;
    }
    return sum;
}

fn smoothstep_f(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let sun_dir = normalize(vec3<f32>(cloud.sun_dir_x, cloud.sun_dir_y, cloud.sun_dir_z));
    let cam_pos = vec3<f32>(cloud.camera_pos_x, cloud.camera_pos_y, cloud.camera_pos_z);
    let view_dir = normalize(cam_pos - input.world_position);
    let normal = normalize(input.world_normal);
    let facing = max(dot(normal, view_dir), 0.0);
    if (facing <= 0.0 || cloud.opacity <= 0.001) {
        discard;
    }

    let seed = vec3<f32>(cloud.noise_seed, cloud.noise_seed * 1.37, cloud.noise_seed * 2.11);
    let motion = vec3<f32>(cloud.time * 0.025, cloud.time * 0.012, -cloud.time * 0.018);
    // Warp local space before sampling so every puff has an irregular, soft
    // silhouette instead of a perfectly spherical one.
    let warp = fbm_lite(input.local_position * 0.65 + seed + motion * 0.5);
    let warped = input.local_position * (1.0 + (warp - 0.5) * 0.30);
    let shape = fbm(warped * 1.55 + seed + motion);
    let detail = fbm(warped * 4.2 + seed * 0.73 - motion * 1.4);
    let density_floor = mix(0.22, 0.12, cloud.storminess);
    let base_density = smoothstep_f(density_floor, 0.58, shape * 0.78 + detail * 0.22);
    let density = max(0.0, base_density - detail * (1.0 - base_density) * 0.45);

    // Approximate sun travel through the puff analytically to create a cool,
    // shaded underside without a costly ray march.
    let sun_facing = dot(normal, sun_dir);
    let optical_depth = density * clamp(1.0 - sun_facing, 0.0, 2.0) * 1.9;
    let shadow = 1.0 - exp(-optical_depth);
    let cloud_base = mix(
        vec3<f32>(0.63, 0.65, 0.68),
        vec3<f32>(0.38, 0.42, 0.48),
        cloud.storminess,
    );
    let cloud_shadow = mix(
        vec3<f32>(0.34, 0.37, 0.43),
        vec3<f32>(0.12, 0.15, 0.21),
        cloud.storminess,
    );
    let cloud_highlight = vec3<f32>(0.94, 0.95, 0.97);
    let silver_lining = pow(facing, 3.0) * pow(max(sun_facing, 0.0), 2.0)
        * mix(0.22, 0.34, cloud.storminess);
    let cloud_color = mix(
        mix(cloud_base, cloud_shadow, shadow),
        cloud_highlight,
        silver_lining,
    );

    // Detail breaks the otherwise circular rim while preserving a dense core.
    let rim = 1.0 - facing;
    let edge_fade = pow(facing, 1.15 + detail * rim * 1.8);
    let role_density = mix(0.88, 1.12, cloud.role);
    let alpha = cloud.opacity * mix(0.46, 0.92, density) * edge_fade * role_density;

    return vec4<f32>(cloud_color, alpha);
}
