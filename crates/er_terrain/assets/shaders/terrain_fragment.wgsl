struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) elevation: f32,
    @location(2) moisture: f32,
    @location(3) low_freq_elev: f32,
    @location(4) temperature: f32,
    @location(5) normal: vec3<f32>,
    @location(6) morph: f32,
    @location(7) drainage: f32,
    @location(8) curvature: f32,
    @location(9) direction: vec3<f32>,
};

// --- Simple value noise for detail/palette variation ---

fn hash3(p: vec3<f32>) -> f32 {
    let cell = vec3<i32>(p);
    var h = bitcast<u32>(cell.x) * 0x8da6b343u;
    h = h ^ (bitcast<u32>(cell.y) * 0xd8163841u);
    h = h ^ (bitcast<u32>(cell.z) * 0xcb1ab31fu);
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return f32(h >> 8u) / 16777215.0;
}

const MATERIAL_DETAIL_PERIOD_CELLS: f32 = 128.0;

fn wrap_detail_cell(p: vec3<f32>) -> vec3<f32> {
    return p - floor(p / MATERIAL_DETAIL_PERIOD_CELLS) * MATERIAL_DETAIL_PERIOD_CELLS;
}

fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3(wrap_detail_cell(i + vec3<f32>(0.0, 0.0, 0.0)));
    let n100 = hash3(wrap_detail_cell(i + vec3<f32>(1.0, 0.0, 0.0)));
    let n010 = hash3(wrap_detail_cell(i + vec3<f32>(0.0, 1.0, 0.0)));
    let n110 = hash3(wrap_detail_cell(i + vec3<f32>(1.0, 1.0, 0.0)));
    let n001 = hash3(wrap_detail_cell(i + vec3<f32>(0.0, 0.0, 1.0)));
    let n101 = hash3(wrap_detail_cell(i + vec3<f32>(1.0, 0.0, 1.0)));
    let n011 = hash3(wrap_detail_cell(i + vec3<f32>(0.0, 1.0, 1.0)));
    let n111 = hash3(wrap_detail_cell(i + vec3<f32>(1.0, 1.0, 1.0)));
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

const MATERIAL_DETAIL_WAVELENGTH_M: f32 = 8.0;

fn triplanar_noise(metric_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    let p = metric_pos / MATERIAL_DETAIL_WAVELENGTH_M;
    var weights = pow(abs(normal), vec3<f32>(4.0));
    weights = weights / max(weights.x + weights.y + weights.z, 0.0001);
    let x_projection = vnoise(vec3<f32>(p.y, p.z, 17.0));
    let y_projection = vnoise(vec3<f32>(p.x, p.z, 37.0));
    let z_projection = vnoise(vec3<f32>(p.x, p.y, 53.0));
    return dot(vec3<f32>(x_projection, y_projection, z_projection), weights);
}

fn detail_normal(
    metric_pos: vec3<f32>,
    geometric_normal: vec3<f32>,
    detail: f32,
) -> vec3<f32> {
    let reference = select(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        abs(geometric_normal.y) < 0.99,
    );
    let tangent = normalize(cross(reference, geometric_normal));
    let bitangent = normalize(cross(geometric_normal, tangent));
    let sample_step_m = 2.0;
    let detail_height_m = 0.5;
    let tangent_sample = triplanar_noise(
        metric_pos + tangent * sample_step_m,
        geometric_normal,
    );
    let bitangent_sample = triplanar_noise(
        metric_pos + bitangent * sample_step_m,
        geometric_normal,
    );
    let tangent_gradient = (tangent_sample - detail) * detail_height_m / sample_step_m;
    let bitangent_gradient = (bitangent_sample - detail) * detail_height_m / sample_step_m;
    return normalize(
        geometric_normal
            - tangent * tangent_gradient
            - bitangent * bitangent_gradient,
    );
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

    // Reuse the material detail sample for palette breakup.
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

    let planet_dir = normalize(input.direction);
    let render_origin = vec3<f32>(material.render_origin_x, material.render_origin_y, material.render_origin_z);
    let detail_period_m = MATERIAL_DETAIL_WAVELENGTH_M * MATERIAL_DETAIL_PERIOD_CELLS;
    let origin_mod = render_origin - floor(render_origin / detail_period_m) * detail_period_m;
    let detail_metric_pos = input.world_position + origin_mod;
    let sun_dir = normalize(vec3<f32>(material.sun_dir_x, material.sun_dir_y, material.sun_dir_z));
    let camera_pos = vec3<f32>(material.camera_pos_x, material.camera_pos_y, material.camera_pos_z);
    let view_dir = normalize(camera_pos - input.world_position);

    let up = planet_dir;
    let pixel_footprint_m = max(length(dpdx(input.world_position)), length(dpdy(input.world_position)));
    // Full-resolution field normals and masks alias when sparse vertices
    // undersample their 100 m footprint. Fade them toward the radial normal
    // once that footprint is sub-pixel.
    let terrain_shape_weight = 1.0 - smoothstep(50.0, 150.0, pixel_footprint_m);
    let geometric_normal = normalize(mix(up, normalize(input.normal), terrain_shape_weight));
    let sampled_detail = triplanar_noise(detail_metric_pos, up);
    let detail_weight = 1.0 - smoothstep(
        0.25,
        1.0,
        pixel_footprint_m,
    );
    let detail = mix(0.5, sampled_detail, detail_weight);
    let detailed_normal = detail_normal(detail_metric_pos, geometric_normal, sampled_detail);
    let normal = normalize(mix(geometric_normal, detailed_normal, detail_weight));
    let slope = 1.0 - abs(dot(normal, up));

    let world_north = vec3<f32>(0.0, 1.0, 0.0);
    let north_raw = world_north - up * dot(world_north, up);
    let north_length = length(north_raw);
    let north = select(
        normalize(cross(vec3<f32>(1.0, 0.0, 0.0), up)),
        north_raw / max(north_length, 0.001),
        north_length > 0.001,
    );
    let downslope_raw = up - normal * dot(up, normal);
    let downslope_length = length(downslope_raw);
    let downslope = select(
        north,
        downslope_raw / max(downslope_length, 0.001),
        downslope_length > 0.001,
    );
    let aspect_north = dot(downslope, north) * 0.5 + 0.5;

    var color = biome_color_blended(
        input.elevation, input.temperature, input.moisture,
        input.low_freq_elev, material, detail, planet_dir,
    );

    let is_earth = material.planet_radius >= 1000000.0;

    let rock_col = vec3<f32>(0.42, 0.38, 0.34) * (0.9 + detail * 0.2);
    // The metric field retains fine procedural residuals. At globe scale those
    // slopes should not read as exposed rock; reserve rock for steep relief.
    let rock_lo = select(0.35, 0.65, is_earth);
    let rock_hi = select(0.55, 0.85, is_earth);
    let filtered_curvature = input.curvature * terrain_shape_weight;
    let filtered_drainage = input.drainage * terrain_shape_weight;
    let convexity = smoothstep(0.02, 0.35, -filtered_curvature);
    let aspect_exposure = mix(0.85, 1.15, aspect_north);
    let rock_blend = clamp(
        smoothstep(rock_lo, rock_hi, slope) * aspect_exposure + convexity * 0.08,
        0.0,
        1.0,
    );
    color = mix(color, rock_col, rock_blend);

    let channel_wetness = filtered_drainage
        * (1.0 - smoothstep(0.45, 0.85, slope))
        * smoothstep(material.sea_level_climate, material.sea_level_climate + 4.0, input.low_freq_elev);
    let concavity = smoothstep(0.02, 0.35, filtered_curvature);
    let sediment = clamp(channel_wetness * 0.25 + concavity * 0.08, 0.0, 1.0);
    color = mix(color, color * vec3<f32>(0.62, 0.72, 0.62), sediment);

    let snow_lo = select(0.75, 5.5, is_earth);
    let snow_hi = select(0.85, 7.5, is_earth);
    let cold_enough = 1.0 - smoothstep(0.32, 0.48, input.temperature);
    let snow_blend = smoothstep(snow_lo, snow_hi, input.elevation)
        * (1.0 - rock_blend)
        * mix(0.75, 1.0, aspect_north)
        * max(cold_enough, 0.2);
    color = mix(color, vec3<f32>(0.92, 0.94, 0.96), snow_blend * 0.8);

    let ao_lo = select(-0.5, -8.0, is_earth);
    let ao_hi = select(0.5, 8.0, is_earth);
    let elev_ao = mix(0.55, 1.0, smoothstep(ao_lo, ao_hi, input.low_freq_elev));
    let slope_ao = mix(1.0, 0.75, slope);
    let ao = elev_ao * slope_ao;

    let sky_color = vec3<f32>(0.30, 0.38, 0.50);
    let ground_color = vec3<f32>(0.15, 0.12, 0.10);
    let hemi = mix(ground_color, sky_color, max(dot(normal, up), 0.0) * 0.5 + 0.5);
    let ambient = hemi * ao * 0.8;

    let diffuse = max(dot(normal, sun_dir), 0.0);

    let coast_wet = 1.0 - smoothstep(
        material.sea_level_climate,
        material.sea_level_climate + material.beach_threshold * 3.0,
        input.low_freq_elev,
    );
    let is_wet = clamp(max(coast_wet, channel_wetness), 0.0, 1.0);
    let half_vec = normalize(sun_dir + view_dir);
    let spec_power = mix(6.0, 48.0, is_wet);
    let spec_strength = mix(0.04, 0.25, is_wet);
    let spec = pow(max(dot(normal, half_vec), 0.0), spec_power) * spec_strength;

    let sun_color = vec3<f32>(1.0, 0.96, 0.88);
    color = color * (ambient + sun_color * diffuse * 0.75) + vec3<f32>(spec);

    let fresnel = pow(1.0 - max(dot(view_dir, normal), 0.0), 3.0);
    let atmosphere_color = vec3<f32>(0.3, 0.6, 1.0);
    color = mix(color, atmosphere_color, fresnel * 0.4);

    return vec4<f32>(color, 1.0);
}
