struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) elevation: f32,
};

fn elevation_color(elev: f32) -> vec3<f32> {
    if (elev < 0.0) {
        let t = clamp(-elev * 0.5, 0.0, 1.0);
        return mix(vec3<f32>(0.1, 0.4, 0.6), vec3<f32>(0.02, 0.05, 0.15), t);
    }
    if (elev < 0.1) {
        return vec3<f32>(0.76, 0.70, 0.50);
    }
    if (elev < 0.5) {
        let t = (elev - 0.1) / 0.4;
        return mix(vec3<f32>(0.76, 0.70, 0.50), vec3<f32>(0.2, 0.5, 0.15), t);
    }
    if (elev < 1.0) {
        let t = (elev - 0.5) / 0.5;
        return mix(vec3<f32>(0.2, 0.5, 0.15), vec3<f32>(0.15, 0.35, 0.08), t);
    }
    if (elev < 1.5) {
        let t = (elev - 1.0) / 0.5;
        return mix(vec3<f32>(0.15, 0.35, 0.08), vec3<f32>(0.45, 0.38, 0.30), t);
    }
    let t = clamp((elev - 1.5) / 0.5, 0.0, 1.0);
    return mix(vec3<f32>(0.45, 0.38, 0.30), vec3<f32>(0.9, 0.9, 0.95), t);
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let dp1 = dpdx(input.world_position);
    let dp2 = dpdy(input.world_position);
    let normal = normalize(cross(dp1, dp2));

    let sun_dir = normalize(vec3<f32>(0.5, 0.8, 0.3));
    let diffuse = max(dot(normal, sun_dir), 0.0);
    let ambient = 0.15;

    let base_color = elevation_color(input.elevation);
    let color = base_color * (ambient + diffuse * 0.85);

    return vec4<f32>(color, 1.0);
}
