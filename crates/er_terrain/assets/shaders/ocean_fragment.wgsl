struct FragmentInput {
    @location(0) world_position: vec3<f32>,
};

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
    // The sea is one geometric datum. The terrain depth buffer hides this
    // surface over land and reveals it where the terrain lies below sea level.
    var color = ocean_depth_color(0.65);

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
