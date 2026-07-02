struct BiomeParams {
    sea_level: f32,
    lapse_rate: f32,
    temp_gradient: f32,
    temp_noise_freq: f32,
    temp_noise_amp: f32,
    moisture_noise_freq: f32,
    moisture_noise_amp: f32,
    rain_shadow_strength: f32,
    high_alt_threshold: f32,
    beach_threshold: f32,
    volcanic_threshold: f32,
    toxic_moisture_threshold: f32,
    toxic_temp_threshold: f32,
    temp_noise_seed: i32,
    moisture_noise_seed: i32,
    lacunarity: f32,
    gain: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(3) var<uniform> biome_params: BiomeParams;
@group(0) @binding(4) var<storage, read> biome_dirs: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> biomes: array<u32>;

fn compute_temperature(dir: vec3<f32>, elevation: f32, bp: BiomeParams) -> f32 {
    let temp_noise = fnl_fbm_opensimplex2_3d(
        bp.temp_noise_seed, dir, bp.temp_noise_freq, 3, bp.lacunarity, bp.gain
    );
    let temp = 1.0 - abs(dir.y) * bp.temp_gradient
        - elevation * bp.lapse_rate
        + temp_noise * bp.temp_noise_amp;
    return clamp(temp, 0.0, 1.0);
}

fn compute_moisture(dir: vec3<f32>, mountain_influence: f32, bp: BiomeParams) -> f32 {
    let m_noise = fnl_fbm_opensimplex2_3d(
        bp.moisture_noise_seed, dir, bp.moisture_noise_freq, 3, bp.lacunarity, bp.gain
    );
    let m = m_noise * bp.moisture_noise_amp * 0.5 + 0.5
        - mountain_influence * bp.rain_shadow_strength;
    return clamp(m, 0.0, 1.0);
}

fn classify_biome(elevation: f32, temperature: f32, moisture: f32, low_freq_elev: f32, bp: BiomeParams) -> u32 {
    if (elevation < bp.sea_level) {
        let depth = bp.sea_level - elevation;
        if (depth < 0.3) { return 0u; }
        if (depth < 0.6) { return 1u; }
        if (depth < 1.0) { return 2u; }
        return 3u;
    }
    if (abs(elevation - bp.sea_level) < bp.beach_threshold) { return 4u; }
    if (low_freq_elev > bp.volcanic_threshold) { return 13u; }
    if (elevation > bp.high_alt_threshold) { return 11u; }
    if (temperature > bp.toxic_temp_threshold && moisture > bp.toxic_moisture_threshold) { return 12u; }
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

@compute @workgroup_size(64)
fn biome_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&biome_dirs)) {
        return;
    }
    let dir = biome_dirs[i].xyz;

    let warped = fnl_domain_warp_3d(dir, params.seed, params.warp_freq, params.warp_amp);

    let continental = fnl_fbm_opensimplex2_3d(
        params.seed, warped, params.continental_freq,
        params.continental_octaves, params.lacunarity, params.gain
    );
    let mountain_raw = fnl_ridged_opensimplex2_3d(
        params.seed, warped, params.mountain_freq,
        params.mountain_octaves, params.lacunarity, params.gain
    );
    let mountain_mask = max(0.0, continental);
    let mountains = mountain_raw * mountain_mask;
    let low_freq_elev = continental * params.continental_amp + mountains * params.mountain_amp;
    let mountain_influence = max(0.0, mountain_raw) * mountain_mask;

    let hills = fnl_fbm_opensimplex2_3d(
        params.seed, warped, params.hill_freq,
        params.hill_octaves, params.lacunarity, params.gain
    );
    let detail = fnl_fbm_value_3d(
        params.seed, warped, params.detail_freq,
        params.detail_octaves, params.lacunarity, params.gain
    );
    let elev = low_freq_elev + hills * params.hill_amp + detail * params.detail_amp;

    let temp = compute_temperature(dir, elev, biome_params);
    let moist = compute_moisture(dir, mountain_influence, biome_params);
    biomes[i] = classify_biome(elev, temp, moist, low_freq_elev, biome_params);
}
