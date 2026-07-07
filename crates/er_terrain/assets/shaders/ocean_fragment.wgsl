struct FragmentInput {
    @location(0) world_position: vec3<f32>,
};

fn make_elev_params(m: OceanMaterialUniform) -> ElevationParams {
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

fn ocean_depth_color(depth: f32) -> vec3<f32> {
    if (depth < 0.3) { return vec3<f32>(0.1, 0.4, 0.6); }
    if (depth < 0.6) { return vec3<f32>(0.06, 0.25, 0.45); }
    if (depth < 1.0) { return vec3<f32>(0.03, 0.12, 0.25); }
    return vec3<f32>(0.01, 0.05, 0.12);
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let dir = normalize(input.world_position);
    let ep = make_elev_params(ocean_material);
    let elev = compute_elevation(dir, ep);

    if (elev >= ocean_material.sea_level) {
        discard;
    }

    let depth = ocean_material.sea_level - elev;
    let base_color = ocean_depth_color(depth);

    let sun_dir = normalize(vec3<f32>(
        ocean_material.sun_dir_x,
        ocean_material.sun_dir_y,
        ocean_material.sun_dir_z,
    ));
    let sun_dot = max(dot(dir, sun_dir), 0.0);
    let specular = pow(sun_dot, 64.0) * 0.3;

    let ripple = sin(ocean_material.time * 2.0 + dir.x * 30.0 + dir.z * 25.0) * 0.02;

    let color = base_color + vec3<f32>(specular) + vec3<f32>(ripple);
    return vec4<f32>(color, 1.0);
}
