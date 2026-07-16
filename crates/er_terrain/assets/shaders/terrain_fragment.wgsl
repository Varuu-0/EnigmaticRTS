struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) elevation: f32,
    @location(2) dir: vec3<f32>,
    @location(3) moisture: f32,
    @location(4) low_freq_elev: f32,
    @location(5) temperature: f32,
    @location(6) normal: vec3<f32>,
    @location(7) morph: f32,
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

// --- Blended biome coloring (smoothstep transitions, richer palette) ---

fn biome_color_blended(
    elev: f32,
    temp: f32,
    moist: f32,
    lfe: f32,
    m: TerrainMaterialUniform,
    detail: f32,
    dir: vec3<f32>,
) -> vec3<f32> {
    let sl = m.sea_level_climate;
    // Shorelines follow the macro elevation rather than the fine residual used
    // for mesh displacement. This prevents small residual hills and valleys
    // from alternating between land and water at adjacent vertices.
    let water_elev = lfe;

    // Ocean depth gradient — brighter, more varied
    let c_shallow = vec3<f32>(0.15, 0.5, 0.65);
    let c_ocean_mid = vec3<f32>(0.08, 0.3, 0.5);
    let c_ocean_deep = vec3<f32>(0.04, 0.18, 0.35);
    let c_abyss = vec3<f32>(0.02, 0.10, 0.20);

    let depth = max(sl - water_elev, 0.0);
    let t1 = smoothstep(0.15, 0.45, depth);
    let t2 = smoothstep(0.45, 0.75, depth);
    let t3 = smoothstep(0.75, 1.25, depth);
    var ocean_col = mix(c_shallow, c_ocean_mid, t1);
    ocean_col = mix(ocean_col, c_ocean_deep, t2);
    ocean_col = mix(ocean_col, c_abyss, t3);

    // Land biome colors — richer, more natural tones
    let c_beach = vec3<f32>(0.76, 0.70, 0.50);
    let c_grass = vec3<f32>(0.30, 0.55, 0.20);
    let c_forest = vec3<f32>(0.12, 0.38, 0.10);
    let c_jungle = vec3<f32>(0.10, 0.45, 0.06);
    let c_desert = vec3<f32>(0.85, 0.72, 0.42);
    let c_tundra = vec3<f32>(0.55, 0.52, 0.46);
    let c_snow = vec3<f32>(0.92, 0.94, 0.96);
    let c_mtn = vec3<f32>(0.42, 0.38, 0.34);
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

    // Shoreline blend (beach <-> ocean <-> land)
    let shore_t = smoothstep(sl - m.beach_threshold, sl + m.beach_threshold, water_elev);
    land_col = mix(
        c_beach,
        land_col,
        smoothstep(0.0, m.beach_threshold * 2.0, water_elev - sl),
    );

    var color = mix(ocean_col, land_col, shore_t);

    let is_earth = m.planet_radius >= 1000000.0;

    // Volcanic override (smooth)
    let vol_lo = select(m.volcanic_threshold - 0.08, 5.0, is_earth);
    let vol_hi = select(m.volcanic_threshold + 0.08, 7.0, is_earth);
    let vol_b = smoothstep(vol_lo, vol_hi, lfe);
    color = mix(color, c_volcanic, vol_b);

    // Mountain override (smooth, partial)
    let mtn_lo = select(m.high_alt_threshold - 0.04, 3.5, is_earth);
    let mtn_hi = select(m.high_alt_threshold + 0.04, 5.5, is_earth);
    let mtn_b = smoothstep(mtn_lo, mtn_hi, elev);
    color = mix(color, c_mtn, mtn_b * 0.65);

    // Toxic override (smooth, partial)
    let tox_b = smoothstep(m.toxic_temp_threshold - 0.04, m.toxic_temp_threshold + 0.04, temp)
             * smoothstep(m.toxic_moisture_threshold - 0.04, m.toxic_moisture_threshold + 0.04, moist);
    color = mix(color, c_toxic, tox_b * 0.4);

    // One smooth sample preserves palette breakup without the former 14
    // value-noise evaluations per shaded pixel.
    let hue = detail - 0.5;
    color = color * (1.0 + hue * 0.16);
    color.r = clamp(color.r + hue * 0.02, 0.0, 1.0);
    color.b = clamp(color.b - hue * 0.02, 0.0, 1.0);

    return color;
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    if (material.debug_skirt_highlight > 0.5 && input.morph < 0.5) {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }

    let sun_dir = normalize(vec3<f32>(material.sun_dir_x, material.sun_dir_y, material.sun_dir_z));
    let camera_pos = vec3<f32>(material.camera_pos_x, material.camera_pos_y, material.camera_pos_z);
    let view_dir = normalize(camera_pos - input.world_position);

    let up = input.dir;
    let normal = normalize(input.normal);
    let slope = 1.0 - abs(dot(normal, up));

    let detail = vnoise(input.dir * 55.0);

    var color = biome_color_blended(
        input.elevation, input.temperature, input.moisture,
        input.low_freq_elev, material, detail, input.dir,
    );

    let is_earth = material.planet_radius >= 1000000.0;

    let rock_col = vec3<f32>(0.42, 0.38, 0.34) * (0.9 + detail * 0.2);
    // The metric field retains fine procedural residuals. At globe scale those
    // slopes should not read as exposed rock; reserve rock for steep relief.
    let rock_lo = select(0.35, 0.65, is_earth);
    let rock_hi = select(0.55, 0.85, is_earth);
    let rock_blend = smoothstep(rock_lo, rock_hi, slope);
    color = mix(color, rock_col, rock_blend);

    let snow_lo = select(0.75, 5.5, is_earth);
    let snow_hi = select(0.85, 7.5, is_earth);
    let snow_blend = smoothstep(snow_lo, snow_hi, input.elevation) * (1.0 - rock_blend);
    color = mix(color, vec3<f32>(0.92, 0.94, 0.96), snow_blend * 0.8);

    let ao_lo = select(-0.5, -8.0, is_earth);
    let ao_hi = select(0.5, 8.0, is_earth);
    let elev_ao = mix(0.55, 1.0, smoothstep(ao_lo, ao_hi, input.low_freq_elev));
    let slope_ao = mix(1.0, 0.75, slope);
    let ao = elev_ao * slope_ao;

    let sky_color = vec3<f32>(0.30, 0.38, 0.50);
    let ground_color = vec3<f32>(0.15, 0.12, 0.10);
    let hemi = mix(ground_color, sky_color, max(dot(normal, up), 0.0) * 0.5 + 0.5);
    let ambient = hemi * ao * 0.35;

    let diffuse = max(dot(normal, sun_dir), 0.0);

    let is_wet = 1.0 - smoothstep(
        material.sea_level_climate - material.beach_threshold,
        material.sea_level_climate + material.beach_threshold,
        input.low_freq_elev,
    );
    let half_vec = normalize(sun_dir + view_dir);
    let spec_power = mix(6.0, 48.0, is_wet);
    let spec_strength = mix(0.04, 0.25, is_wet);
    let spec = pow(max(dot(normal, half_vec), 0.0), spec_power) * spec_strength;

    let sun_color = vec3<f32>(1.0, 0.96, 0.88);
    color = color * (ambient + sun_color * diffuse) + vec3<f32>(spec);

    let fresnel = pow(1.0 - max(dot(view_dir, normal), 0.0), 3.0);
    let atmosphere_color = vec3<f32>(0.3, 0.6, 1.0);
    color = mix(color, atmosphere_color, fresnel * 0.4);

    return vec4<f32>(color, 1.0);
}
