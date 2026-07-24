// Debug view mode discriminants — must stay in sync with
// `DebugViewMode` in `crates/er_terrain/src/debug_view.rs`.

const DEBUG_VIEW_OFF: u32 = 0u;
const DEBUG_VIEW_ANNUAL_TEMPERATURE: u32 = 1u;
const DEBUG_VIEW_SEASONAL_TEMPERATURE: u32 = 2u;
const DEBUG_VIEW_PRECIPITATION: u32 = 3u;
const DEBUG_VIEW_PET_ARIDITY: u32 = 4u;
const DEBUG_VIEW_PREVAILING_WIND: u32 = 5u;
const DEBUG_VIEW_CONTINENTALITY: u32 = 6u;
const DEBUG_VIEW_ELEVATION: u32 = 7u;
const DEBUG_VIEW_SLOPE: u32 = 8u;
const DEBUG_VIEW_ASPECT: u32 = 9u;
const DEBUG_VIEW_RELIEF: u32 = 10u;
const DEBUG_VIEW_FLOW_ACCUMULATION: u32 = 11u;
const DEBUG_VIEW_WETNESS: u32 = 12u;
const DEBUG_VIEW_SUBSTRATE: u32 = 13u;
const DEBUG_VIEW_SOIL_PROPERTIES: u32 = 14u;
const DEBUG_VIEW_DISTURBANCE: u32 = 15u;
const DEBUG_VIEW_TOP_BIOME_MEMBERSHIPS: u32 = 16u;
const DEBUG_VIEW_ECOTONE_STRENGTH: u32 = 17u;
const DEBUG_VIEW_FINAL_MATERIAL_MASKS: u32 = 18u;

/// Unmistakable magenta/black checkerboard used for debug modes whose
/// underlying scientific field is not yet available in the shader. This must
/// never produce a color that could be mistaken for real environmental data.
fn debug_unavailable_checker(world_position: vec3<f32>) -> vec3<f32> {
    let cell = floor(world_position.xz / 2.0);
    let checker = (cell.x + cell.y) - 2.0 * floor((cell.x + cell.y) / 2.0);
    let is_light = checker > 0.5;
    return select(
        vec3<f32>(0.8, 0.0, 0.8),
        vec3<f32>(0.0, 0.0, 0.0),
        is_light,
    );
}

/// Convert a scalar `value` in `[0, 1]` to a simple heat-map ramp.
fn debug_heat_ramp(value: f32) -> vec3<f32> {
    let t = clamp(value, 0.0, 1.0);
    let r = smoothstep(0.0, 0.5, t);
    let g = select(0.0, smoothstep(0.25, 0.75, t), t > 0.25);
    let b = smoothstep(0.5, 1.0, t);
    return vec3<f32>(r, g, b);
}

/// Separate bathymetry from land elevation around the shared sea datum.
fn debug_elevation_ramp(elevation: f32, sea_level: f32) -> vec3<f32> {
    if (elevation < sea_level) {
        let depth = clamp((sea_level - elevation) / 8.0, 0.0, 1.0);
        return mix(vec3<f32>(0.1, 0.55, 0.9), vec3<f32>(0.01, 0.02, 0.18), depth);
    }
    let height = clamp((elevation - sea_level) / 8.0, 0.0, 1.0);
    let low_to_rock = mix(
        vec3<f32>(0.08, 0.42, 0.12),
        vec3<f32>(0.62, 0.48, 0.22),
        smoothstep(0.0, 0.55, height),
    );
    return mix(low_to_rock, vec3<f32>(0.96, 0.97, 1.0), smoothstep(0.55, 1.0, height));
}

/// Cyclic hue wheel for a full azimuth. North is red, east green, south cyan,
/// and west violet; north wraps continuously back to red.
fn debug_aspect_ramp(north_component: f32, east_component: f32) -> vec3<f32> {
    let phase = atan2(east_component, north_component);
    return vec3<f32>(
        0.5 + 0.5 * cos(phase),
        0.5 + 0.5 * cos(phase - 2.0943952),
        0.5 + 0.5 * cos(phase + 2.0943952),
    );
}

// --- M6.2 climate debug ramps ---
// These assume a packed climate transport of two vec4 interpolants:
//   climate_a = (annual_temp_c, temp_range_c, annual_precip_mm, moisture_index)
//   climate_b = (pet_mm, aridity_index, wind_direction_rad, continentality_c)
// All ranges are fixed, physically meaningful normalization windows intended
// for Earth-like planets. Values outside the window clamp to the ramp ends.

/// Annual mean temperature: blue (-30 C) -> white (+2.5 C) -> red (+35 C).
fn debug_temperature_ramp(annual_temp_c: f32) -> vec3<f32> {
    let t = clamp((annual_temp_c + 30.0) / 65.0, 0.0, 1.0);
    let cold = vec3<f32>(0.0, 0.2, 0.7);
    let mild = vec3<f32>(0.9, 0.95, 0.85);
    let hot = vec3<f32>(0.75, 0.05, 0.05);
    if (t < 0.5) {
        return mix(cold, mild, smoothstep(0.0, 0.5, t));
    }
    return mix(mild, hot, smoothstep(0.5, 1.0, t));
}

/// Seasonal temperature range: blue (0 C) -> green -> red (50 C).
fn debug_temperature_range_ramp(temp_range_c: f32) -> vec3<f32> {
    return debug_heat_ramp(clamp(temp_range_c / 50.0, 0.0, 1.0));
}

/// Annual precipitation: tan/white (0 mm) -> dark blue (4000 mm).
fn debug_precipitation_ramp(annual_precip_mm: f32) -> vec3<f32> {
    let t = clamp(annual_precip_mm / 4000.0, 0.0, 1.0);
    let dry = vec3<f32>(0.95, 0.85, 0.55);
    let wet = vec3<f32>(0.05, 0.25, 0.65);
    return mix(dry, wet, smoothstep(0.0, 1.0, t));
}

/// PET / aridity index: teal (humid, 0) -> yellow -> red (arid, 2.0).
fn debug_aridity_ramp(aridity_index: f32) -> vec3<f32> {
    let t = clamp(aridity_index / 2.0, 0.0, 1.0);
    let humid = vec3<f32>(0.05, 0.35, 0.55);
    let arid = vec3<f32>(0.85, 0.1, 0.05);
    return mix(humid, arid, smoothstep(0.0, 1.0, t));
}

/// Prevailing wind direction: cyclic hue wheel from 0 to 2*pi radians.
fn debug_wind_direction_ramp(wind_dir_rad: f32) -> vec3<f32> {
    let phase = wind_dir_rad;
    return vec3<f32>(
        0.5 + 0.5 * cos(phase),
        0.5 + 0.5 * cos(phase - 2.0943952),
        0.5 + 0.5 * cos(phase + 2.0943952),
    );
}

/// Continentality: blue (maritime, 0 C annual range) -> red (continental, 100 C).
fn debug_continentality_ramp(continentality_c: f32) -> vec3<f32> {
    let t = clamp(continentality_c / 100.0, 0.0, 1.0);
    let maritime = vec3<f32>(0.05, 0.25, 0.65);
    let continental = vec3<f32>(0.75, 0.1, 0.05);
    return mix(maritime, continental, smoothstep(0.0, 1.0, t));
}

// --- M6.3 landform, substrate, and soil debug ramps ---

/// Local terrain relief: dark green (flat) -> ochre -> white (>= 2 km).
fn debug_relief_ramp(relief_m: f32) -> vec3<f32> {
    let t = clamp(relief_m / 2000.0, 0.0, 1.0);
    let low = vec3<f32>(0.04, 0.24, 0.10);
    let middle = vec3<f32>(0.62, 0.40, 0.12);
    let high = vec3<f32>(0.95, 0.95, 0.95);
    if (t < 0.55) {
        return mix(low, middle, smoothstep(0.0, 0.55, t));
    }
    return mix(middle, high, smoothstep(0.55, 1.0, t));
}

/// Stable categorical substrate palette matching `SubstrateClass` discriminants.
fn debug_substrate_ramp(substrate_class: f32) -> vec3<f32> {
    switch i32(round(substrate_class)) {
        case 0: { return vec3<f32>(0.10, 0.22, 0.36); } // oceanic sediment
        case 1: { return vec3<f32>(0.47, 0.31, 0.18); } // continental sediment
        case 2: { return vec3<f32>(0.69, 0.55, 0.30); } // alluvium
        case 3: { return vec3<f32>(0.58, 0.62, 0.56); } // glacial till
        case 4: { return vec3<f32>(0.12, 0.12, 0.13); } // basalt
        case 5: { return vec3<f32>(0.30, 0.27, 0.25); } // andesite
        case 6: { return vec3<f32>(0.66, 0.42, 0.38); } // rhyolite
        case 7: { return vec3<f32>(0.74, 0.68, 0.61); } // granite
        case 8: { return vec3<f32>(0.39, 0.43, 0.36); } // schist
        case 9: { return vec3<f32>(0.74, 0.52, 0.28); } // sandstone
        case 10: { return vec3<f32>(0.28, 0.30, 0.27); } // shale
        case 11: { return vec3<f32>(0.78, 0.75, 0.58); } // carbonate
        case 12: { return vec3<f32>(0.65, 0.72, 0.54); } // karst
        case 13: { return vec3<f32>(0.38, 0.25, 0.16); } // regolith
        case 14: { return vec3<f32>(0.69, 0.12, 0.06); } // laterite
        default: { return vec3<f32>(0.8, 0.0, 0.8); }
    }
}

/// Soil diagnostic: depth/fertility establish the base, organics add green,
/// salt washes toward white, and permafrost toward blue.
fn debug_soil_ramp(
    depth_m: f32,
    fertility: f32,
    organic_fraction: f32,
    salinity_ds_m: f32,
    permafrost_fraction: f32,
) -> vec3<f32> {
    let depth = clamp(depth_m / 4.0, 0.0, 1.0);
    var color = mix(vec3<f32>(0.45, 0.36, 0.29), vec3<f32>(0.14, 0.08, 0.03), depth);
    color = mix(color, vec3<f32>(0.16, 0.42, 0.10), clamp(organic_fraction, 0.0, 1.0) * 0.65);
    color = mix(color, vec3<f32>(0.72, 0.53, 0.20), clamp(fertility, 0.0, 1.0) * 0.35);
    color = mix(color, vec3<f32>(0.92, 0.90, 0.80), clamp(salinity_ds_m / 30.0, 0.0, 1.0));
    return mix(color, vec3<f32>(0.46, 0.72, 0.88), clamp(permafrost_fraction, 0.0, 1.0));
}
