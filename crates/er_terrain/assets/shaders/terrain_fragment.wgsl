struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) elevation: f32,
    @location(2) dir: vec3<f32>,
    @location(3) moisture: f32,
    @location(4) low_freq_elev: f32,
    @location(5) temperature: f32,
};

fn biome_color(biome: u32) -> vec3<f32> {
    switch (biome) {
        case 0u: { return vec3<f32>(0.1, 0.4, 0.6); }
        case 1u: { return vec3<f32>(0.06, 0.25, 0.45); }
        case 2u: { return vec3<f32>(0.03, 0.12, 0.25); }
        case 3u: { return vec3<f32>(0.01, 0.05, 0.12); }
        case 4u: { return vec3<f32>(0.76, 0.70, 0.50); }
        case 5u: { return vec3<f32>(0.35, 0.55, 0.20); }
        case 6u: { return vec3<f32>(0.15, 0.40, 0.12); }
        case 7u: { return vec3<f32>(0.12, 0.50, 0.08); }
        case 8u: { return vec3<f32>(0.85, 0.75, 0.45); }
        case 9u: { return vec3<f32>(0.50, 0.48, 0.42); }
        case 10u: { return vec3<f32>(0.92, 0.94, 0.96); }
        case 11u: { return vec3<f32>(0.40, 0.36, 0.32); }
        case 12u: { return vec3<f32>(0.35, 0.25, 0.40); }
        case 13u: { return vec3<f32>(0.25, 0.08, 0.05); }
        default: { return vec3<f32>(1.0, 0.0, 1.0); }
    }
}

fn classify_biome(elevation: f32, temperature: f32, moisture: f32, low_freq_elev: f32, m: TerrainMaterialUniform) -> u32 {
    if (elevation < m.sea_level_climate) {
        let depth = m.sea_level_climate - elevation;
        if (depth < 0.3) { return 0u; }
        if (depth < 0.6) { return 1u; }
        if (depth < 1.0) { return 2u; }
        return 3u;
    }
    if (abs(elevation - m.sea_level_climate) < m.beach_threshold) { return 4u; }
    if (low_freq_elev > m.volcanic_threshold) { return 13u; }
    if (elevation > m.high_alt_threshold) { return 11u; }
    if (temperature > m.toxic_temp_threshold && moisture > m.toxic_moisture_threshold) { return 12u; }
    if (temperature < 0.25) {
        if (moisture < 0.7) { return 9u; }
        return 10u;
    } else if (temperature < 0.6) {
        if (moisture < 0.7) { return 5u; }
        return 6u;
    } else {
        if (moisture < 0.35) { return 8u; }
        if (moisture < 0.7) { return 5u; }
        return 7u;
    }
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let dp1 = dpdx(input.world_position);
    let dp2 = dpdy(input.world_position);
    let normal = normalize(cross(dp1, dp2));

    let sun_dir = normalize(vec3<f32>(0.5, 0.8, 0.3));
    let diffuse = max(dot(normal, sun_dir), 0.0);
    let ambient = 0.15;

    let biome_idx = classify_biome(input.elevation, input.temperature, input.moisture, input.low_freq_elev, material);
    let base_color = biome_color(biome_idx);
    let color = base_color * (ambient + diffuse * 0.85);

    return vec4<f32>(color, 1.0);
}
