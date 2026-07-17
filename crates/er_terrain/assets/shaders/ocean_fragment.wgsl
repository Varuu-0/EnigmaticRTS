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
    let shallow = vec3<f32>(0.15, 0.5, 0.65);
    let mid = vec3<f32>(0.08, 0.3, 0.5);
    let deep = vec3<f32>(0.04, 0.18, 0.35);
    let abyss = vec3<f32>(0.02, 0.10, 0.20);
    var col = mix(shallow, mid, smoothstep(0.15, 0.45, depth));
    col = mix(col, deep, smoothstep(0.45, 0.75, depth));
    col = mix(col, abyss, smoothstep(0.75, 1.25, depth));
    return col;
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let render_origin = vec3<f32>(ocean_material.render_origin_x, ocean_material.render_origin_y, ocean_material.render_origin_z);
    let planet_dir = normalize(input.world_position + render_origin);
    let ep = make_elev_params(ocean_material);
    let elev = select(
        compute_low_freq_elevation(planet_dir, ep),
        compute_low_freq_elevation_metric(planet_dir, ep, ocean_material.planet_radius),
        ocean_material.planet_radius >= 1000000.0,
    );

    // Don't discard over land - return transparent but still write depth
    // This ensures the ocean sphere occludes far-side terrain
    if (elev >= ocean_material.sea_level) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let depth = ocean_material.sea_level - elev;
    var color = ocean_depth_color(depth);

    let sun_dir = normalize(vec3<f32>(
        ocean_material.sun_dir_x,
        ocean_material.sun_dir_y,
        ocean_material.sun_dir_z,
    ));
    let sun_dot = max(dot(planet_dir, sun_dir), 0.0);

    // Fresnel reflection — sky tint at grazing angles
    let camera_pos = vec3<f32>(
        ocean_material.camera_pos_x,
        ocean_material.camera_pos_y,
        ocean_material.camera_pos_z,
    );
    let view_dir = normalize(camera_pos - input.world_position);
    let fresnel = pow(1.0 - max(dot(view_dir, planet_dir), 0.0), 3.0);
    let sky_color = vec3<f32>(0.4, 0.6, 0.9);
    color = mix(color, sky_color, fresnel * 0.5);

    // Subsurface scattering near shore
    let shallow_glow = (1.0 - smoothstep(0.0, 0.3, depth)) * 0.3;
    color = color + vec3<f32>(0.1, 0.2, 0.15) * shallow_glow;

    // Multi-frequency ripples
    let ripple1 = sin(ocean_material.time * 1.5 + planet_dir.x * 20.0 + planet_dir.z * 15.0) * 0.015;
    let ripple2 = sin(ocean_material.time * 3.0 + planet_dir.y * 35.0 - planet_dir.x * 10.0) * 0.008;
    let ripple3 = sin(ocean_material.time * 0.7 + planet_dir.z * 12.0 + planet_dir.y * 18.0) * 0.005;
    let ripple = ripple1 + ripple2 + ripple3;
    color = color + vec3<f32>(ripple);

    // Better specular glint — tight sun reflection + broader glow
    let spec = pow(sun_dot, 128.0) * 0.5 + pow(sun_dot, 16.0) * 0.1;
    color = color + vec3<f32>(spec);

    return vec4<f32>(color, 1.0);
}
