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

fn smoothstep_f(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let dir = normalize(input.world_position);
    let cam_pos = vec3<f32>(atmosphere.camera_x, atmosphere.camera_y, atmosphere.camera_z);
    let view_dir = normalize(cam_pos - input.world_position);
    let sun_dir = normalize(vec3<f32>(atmosphere.sun_x, atmosphere.sun_y, atmosphere.sun_z));

    // Grazing angle: 0 when facing camera, 1 at the limb (planet edge)
    let cos_zenith = dot(dir, view_dir);
    let grazing = sqrt(max(1.0 - cos_zenith * cos_zenith, 0.0));

    // Sun elevation at this atmosphere point (1 = sun overhead, -1 = below horizon)
    let sun_angle = dot(dir, sun_dir);

    // Day/night transition factor
    let day_factor = smoothstep_f(-0.1, 0.25, sun_angle);

    // Limb factor: atmosphere only visible right at the planet edge
    let limb_factor = smoothstep_f(0.8, 0.98, grazing);

    // Rayleigh scattering (blue sky, only at the limb)
    let rayleigh_color = vec3<f32>(0.3, 0.6, 1.0);
    let rayleigh = rayleigh_color * limb_factor * 0.5 * day_factor;

    // Sunset/sunrise glow near the terminator
    let terminator = smoothstep_f(-0.05, 0.1, sun_angle) * (1.0 - smoothstep_f(0.15, 0.4, sun_angle));
    let sunset = vec3<f32>(1.0, 0.4, 0.1) * terminator * limb_factor * 1.5;

    // Mie forward scattering (bright sun disk glow)
    let sun_dot_view = dot(view_dir, sun_dir);
    let mie = pow(max(sun_dot_view, 0.0), 32.0)
        * vec3<f32>(1.0, 0.85, 0.6) * 0.6;

    // Limb glow color
    let limb_color = mix(vec3<f32>(0.1, 0.15, 0.3), vec3<f32>(0.3, 0.6, 1.0), day_factor);

    // Subtle night-side glow
    let night = vec3<f32>(0.01, 0.02, 0.04) * limb_factor * (1.0 - day_factor);

    let color = rayleigh + sunset + mie + limb_factor * 0.5 * limb_color + night;
    let alpha = clamp(limb_factor * 0.7 + mie * 0.1, 0.0, 0.85);

    return vec4<f32>(color, alpha);
}
