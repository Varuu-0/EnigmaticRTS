struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) elevation: f32,
    @location(2) dir: vec3<f32>,
    @location(3) moisture: f32,
    @location(4) low_freq_elev: f32,
    @location(5) temperature: f32,
};

// --- Simple value noise for detail/palette variation ---

fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

fn fbm_detail(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 3; i = i + 1) {
        sum = sum + vnoise(p * freq) * amp;
        freq = freq * 2.1;
        amp = amp * 0.5;
    }
    return sum;
}

// --- Triplanar detail sampling ---
fn triplanar(dir: vec3<f32>, normal: vec3<f32>, freq: f32) -> f32 {
    let w = abs(normal);
    let wt = w.x + w.y + w.z;
    let nxy = fbm_detail(vec3<f32>(dir.xy * freq, 0.0));
    let nyz = fbm_detail(vec3<f32>(0.0, dir.yz * freq));
    let nxz = fbm_detail(vec3<f32>(dir.x * freq, 0.0, dir.z * freq));
    return (nxy * w.x + nyz * w.y + nxz * w.z) / wt;
}

// --- Blended biome coloring (smoothstep transitions, replaces discrete classify) ---

fn biome_color_blended(
    elev: f32,
    temp: f32,
    moist: f32,
    lfe: f32,
    m: TerrainMaterialUniform,
    detail: f32,
) -> vec3<f32> {
    let sl = m.sea_level_climate;

    // Ocean depth gradient (smooth between bands)
    let c_shallow = vec3<f32>(0.1, 0.4, 0.6);
    let c_ocean_mid = vec3<f32>(0.06, 0.25, 0.45);
    let c_ocean_deep = vec3<f32>(0.03, 0.12, 0.25);
    let c_abyss = vec3<f32>(0.01, 0.05, 0.12);

    let depth = max(sl - elev, 0.0);
    let t1 = smoothstep(0.15, 0.45, depth);
    let t2 = smoothstep(0.45, 0.75, depth);
    let t3 = smoothstep(0.75, 1.25, depth);
    var ocean_col = mix(c_shallow, c_ocean_mid, t1);
    ocean_col = mix(ocean_col, c_ocean_deep, t2);
    ocean_col = mix(ocean_col, c_abyss, t3);

    // Land biome colors
    let c_beach = vec3<f32>(0.76, 0.70, 0.50);
    let c_grass = vec3<f32>(0.35, 0.55, 0.20);
    let c_forest = vec3<f32>(0.15, 0.40, 0.12);
    let c_jungle = vec3<f32>(0.12, 0.50, 0.08);
    let c_desert = vec3<f32>(0.85, 0.75, 0.45);
    let c_tundra = vec3<f32>(0.50, 0.48, 0.42);
    let c_snow = vec3<f32>(0.92, 0.94, 0.96);
    let c_mtn = vec3<f32>(0.40, 0.36, 0.32);
    let c_volcanic = vec3<f32>(0.25, 0.08, 0.05);
    let c_toxic = vec3<f32>(0.35, 0.25, 0.40);

    // Whittaker temperature/moisture zones (smooth)
    let cold_w = 1.0 - smoothstep(0.15, 0.30, temp);
    let hot_w = smoothstep(0.50, 0.65, temp);
    let temp_w = 1.0 - cold_w - hot_w;

    let dry_w = 1.0 - smoothstep(0.20, 0.40, moist);
    let wet_w = smoothstep(0.55, 0.75, moist);
    let mid_w = 1.0 - dry_w - wet_w;

    let cold_col = mix(c_tundra, c_snow, wet_w);
    let hot_col = mix(c_desert, mix(c_grass, c_jungle, wet_w), mid_w + wet_w);
    let temp_col = mix(c_grass, mix(c_forest, c_jungle, wet_w * 0.5), mid_w);

    var land_col = cold_col * cold_w + temp_col * temp_w + hot_col * hot_w;

    // Shoreline blend (beach ↔ ocean ↔ land)
    let shore_t = smoothstep(sl - m.beach_threshold, sl + m.beach_threshold, elev);
    land_col = mix(c_beach, land_col, smoothstep(0.0, m.beach_threshold * 2.0, elev - sl));

    var color = mix(ocean_col, land_col, shore_t);

    // Volcanic override (smooth)
    let vol_b = smoothstep(m.volcanic_threshold - 0.08, m.volcanic_threshold + 0.08, lfe);
    color = mix(color, c_volcanic, vol_b);

    // Mountain override (smooth, partial)
    let mtn_b = smoothstep(m.high_alt_threshold - 0.04, m.high_alt_threshold + 0.04, elev);
    color = mix(color, c_mtn, mtn_b * 0.65);

    // Toxic override (smooth, partial)
    let tox_b = smoothstep(m.toxic_temp_threshold - 0.04, m.toxic_temp_threshold + 0.04, temp)
             * smoothstep(m.toxic_moisture_threshold - 0.04, m.toxic_moisture_threshold + 0.04, moist);
    color = mix(color, c_toxic, tox_b * 0.4);

    // Palette variation: modulate brightness/hue with detail noise
    let variation = 0.85 + detail * 0.30;
    color = color * variation;

    return color;
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    // Surface normal from screen-space derivatives
    let dp1 = dpdx(input.world_position);
    let dp2 = dpdy(input.world_position);
    let normal = normalize(cross(dp1, dp2));

    // Sun direction from uniform (6.14)
    let sun_dir = normalize(vec3<f32>(material.sun_dir_x, material.sun_dir_y, material.sun_dir_z));

    // Detail noise: triplanar sampling (6.13)
    let detail = triplanar(input.dir, normal, 80.0);

    // Blended biome color (6.10, 6.11)
    let base_color = biome_color_blended(
        input.elevation, input.temperature, input.moisture,
        input.low_freq_elev, material, detail,
    );

    // Slope-based rock overlay (6.12)
    let up = input.dir;
    let slope = 1.0 - abs(dot(normal, up));
    let rock_col = vec3<f32>(0.38, 0.34, 0.30) * (0.9 + detail * 0.2);
    let rock_blend = smoothstep(0.35, 0.55, slope);
    var color = mix(base_color, rock_col, rock_blend);

    // Snow on high-altitude flat surfaces
    let snow_blend = smoothstep(0.75, 0.85, input.elevation) * (1.0 - rock_blend);
    color = mix(color, vec3<f32>(0.92, 0.94, 0.96), snow_blend * 0.8);

    // Lighting: diffuse + ambient + elevation AO (6.15)
    let diffuse = max(dot(normal, sun_dir), 0.0);
    let ambient = 0.18;
    let ao = mix(0.7, 1.0, smoothstep(-0.5, 0.3, input.low_freq_elev));
    let light = ambient + diffuse * 0.82;

    // Specular on wet/ocean surfaces
    let is_wet = step(input.elevation, material.sea_level_climate);
    let spec = pow(max(dot(normal, sun_dir), 0.0), 32.0) * 0.15 * is_wet;

    color = color * light * ao + vec3<f32>(spec);

    return vec4<f32>(color, 1.0);
}
