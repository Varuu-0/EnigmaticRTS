//! Debug visualization modes for M6.1 (geometric / hydrologic / substrate)
//! and M6.2 (climate) environmental drivers.
//!
//! Each variant maps 1:1 to a `debug_view_mode` uniform value consumed by the
//! terrain fragment shader. The WGSL constants `DEBUG_VIEW_*` must stay in sync
//! with the discriminants here.
//!
//! Modes backed by fields that are not yet available in the shader render an
//! unmistakable magenta/black checkerboard placeholder — never fabricated
//! scientific values.

/// Number of variants in [`DebugViewMode`], including `Off`.
pub const DEBUG_VIEW_MODE_COUNT: usize = 19;

/// Debug visualization modes covering every M6.1 environmental driver.
///
/// Discriminants are stable: they are written into the uniform buffer and
/// consumed by WGSL, so they must never be renumbered or reordered.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DebugViewMode {
    /// No debug overlay — normal material pipeline runs unchanged.
    #[default]
    Off = 0,
    // --- Climate drivers (roadmap 6.1) ---
    AnnualTemperature = 1,
    SeasonalTemperature = 2,
    Precipitation = 3,
    PetAridity = 4,
    PrevailingWind = 5,
    Continentality = 6,
    // --- Landform drivers (roadmap 6.1) ---
    Elevation = 7,
    Slope = 8,
    Aspect = 9,
    Relief = 10,
    // --- Hydrology drivers (roadmap 6.1) ---
    FlowAccumulation = 11,
    Wetness = 12,
    // --- Substrate / soil drivers (roadmap 6.1) ---
    Substrate = 13,
    SoilProperties = 14,
    // --- Disturbance driver (roadmap 6.1) ---
    Disturbance = 15,
    // --- Ecosystem drivers (roadmap 6.1) ---
    TopBiomeMemberships = 16,
    EcotoneStrength = 17,
    // --- Final output (roadmap 6.1) ---
    FinalMaterialMasks = 18,
}

impl DebugViewMode {
    /// All variants in discriminant order.
    pub const ALL: [DebugViewMode; DEBUG_VIEW_MODE_COUNT] = [
        DebugViewMode::Off,
        DebugViewMode::AnnualTemperature,
        DebugViewMode::SeasonalTemperature,
        DebugViewMode::Precipitation,
        DebugViewMode::PetAridity,
        DebugViewMode::PrevailingWind,
        DebugViewMode::Continentality,
        DebugViewMode::Elevation,
        DebugViewMode::Slope,
        DebugViewMode::Aspect,
        DebugViewMode::Relief,
        DebugViewMode::FlowAccumulation,
        DebugViewMode::Wetness,
        DebugViewMode::Substrate,
        DebugViewMode::SoilProperties,
        DebugViewMode::Disturbance,
        DebugViewMode::TopBiomeMemberships,
        DebugViewMode::EcotoneStrength,
        DebugViewMode::FinalMaterialMasks,
    ];

    /// Stable human-readable name for UI / logging.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::AnnualTemperature => "annual_temperature",
            Self::SeasonalTemperature => "seasonal_temperature",
            Self::Precipitation => "precipitation",
            Self::PetAridity => "pet_aridity",
            Self::PrevailingWind => "prevailing_wind",
            Self::Continentality => "continentality",
            Self::Elevation => "elevation",
            Self::Slope => "slope",
            Self::Aspect => "aspect",
            Self::Relief => "relief",
            Self::FlowAccumulation => "flow_accumulation",
            Self::Wetness => "wetness",
            Self::Substrate => "substrate",
            Self::SoilProperties => "soil_properties",
            Self::Disturbance => "disturbance",
            Self::TopBiomeMemberships => "top_biome_memberships",
            Self::EcotoneStrength => "ecotone_strength",
            Self::FinalMaterialMasks => "final_material_masks",
        }
    }

    /// Cycles to the next mode, wrapping `FinalMaterialMasks` back to `Off`.
    pub const fn next(self) -> Self {
        let v = self as u32;
        if v as usize >= DEBUG_VIEW_MODE_COUNT - 1 {
            Self::Off
        } else {
            Self::from_u32_const(v + 1)
        }
    }

    /// Parse a raw uniform value back into a mode.
    pub fn from_u32(value: u32) -> Option<Self> {
        if (value as usize) < DEBUG_VIEW_MODE_COUNT {
            Some(Self::from_u32_const(value))
        } else {
            None
        }
    }

    const fn from_u32_const(value: u32) -> Self {
        match value {
            0 => Self::Off,
            1 => Self::AnnualTemperature,
            2 => Self::SeasonalTemperature,
            3 => Self::Precipitation,
            4 => Self::PetAridity,
            5 => Self::PrevailingWind,
            6 => Self::Continentality,
            7 => Self::Elevation,
            8 => Self::Slope,
            9 => Self::Aspect,
            10 => Self::Relief,
            11 => Self::FlowAccumulation,
            12 => Self::Wetness,
            13 => Self::Substrate,
            14 => Self::SoilProperties,
            15 => Self::Disturbance,
            16 => Self::TopBiomeMemberships,
            17 => Self::EcotoneStrength,
            18 => Self::FinalMaterialMasks,
            _ => Self::Off,
        }
    }

    /// Returns `true` if the fragment shader has the interpolant data needed
    /// to render an honest visualization of this driver right now.
    ///
    /// Available views are:
    ///   * M6.2 climate transport (`climate_a`, `climate_b`) — annual and
    ///     seasonal temperature, precipitation, PET/aridity, prevailing wind
    ///     direction, and continentality.
    ///   * M6.1 geometric views derived from the vertex normal and elevation.
    ///   * M6.3 landform, substrate, and soil transport (`environment_a`,
    ///     `environment_b`) — relief, substrate, and soil properties.
    ///
    /// Unavailable views (hydrology, disturbance, ecosystem, and final material
    /// masks) do not yet exist as shader inputs; the fragment shader renders a
    /// checkerboard placeholder for them.
    pub const fn available_now(self) -> bool {
        matches!(
            self,
            Self::Off
                | Self::AnnualTemperature
                | Self::SeasonalTemperature
                | Self::Precipitation
                | Self::PetAridity
                | Self::PrevailingWind
                | Self::Continentality
                | Self::Elevation
                | Self::Slope
                | Self::Aspect
                | Self::Relief
                | Self::Substrate
                | Self::SoilProperties
        )
    }

    /// Returns `true` if this mode is `Off`.
    pub const fn is_off(self) -> bool {
        matches!(self, Self::Off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_matches_variant_array() {
        assert_eq!(DEBUG_VIEW_MODE_COUNT, DebugViewMode::ALL.len());
    }

    #[test]
    fn off_is_default_and_zero_discriminant() {
        assert_eq!(DebugViewMode::default(), DebugViewMode::Off);
        assert_eq!(DebugViewMode::Off as u32, 0);
    }

    #[test]
    fn discriminants_are_contiguous_from_zero() {
        for (i, &mode) in DebugViewMode::ALL.iter().enumerate() {
            assert_eq!(mode as u32, i as u32, "variant {:?} at index {i}", mode);
        }
    }

    #[test]
    fn wgsl_constants_match_rust_discriminants() {
        let wgsl = include_str!("../assets/shaders/debug_view.wgsl");
        for &mode in &DebugViewMode::ALL {
            let constant = format!(
                "const DEBUG_VIEW_{}: u32 = {}u;",
                mode.name().to_ascii_uppercase(),
                mode as u32
            );
            assert!(
                wgsl.contains(&constant),
                "missing or mismatched WGSL constant: {constant}"
            );
        }
    }

    #[test]
    fn from_u32_round_trips_every_variant() {
        for &mode in &DebugViewMode::ALL {
            assert_eq!(DebugViewMode::from_u32(mode as u32), Some(mode));
        }
    }

    #[test]
    fn from_u32_rejects_out_of_range() {
        assert_eq!(DebugViewMode::from_u32(DEBUG_VIEW_MODE_COUNT as u32), None);
        assert_eq!(DebugViewMode::from_u32(u32::MAX), None);
    }

    #[test]
    fn next_cycles_back_to_off() {
        let last = DebugViewMode::ALL[DEBUG_VIEW_MODE_COUNT - 1];
        assert_eq!(last.next(), DebugViewMode::Off);
        assert_eq!(DebugViewMode::Off.next(), DebugViewMode::ALL[1]);
    }

    #[test]
    fn next_visits_every_variant_exactly_once_per_cycle() {
        let mut visited = Vec::new();
        let mut mode = DebugViewMode::Off;
        for _ in 0..DEBUG_VIEW_MODE_COUNT {
            visited.push(mode);
            mode = mode.next();
        }
        assert_eq!(mode, DebugViewMode::Off);
        assert_eq!(visited.len(), DEBUG_VIEW_MODE_COUNT);
        let mut sorted = visited.clone();
        sorted.sort_by_key(|m| *m as u32);
        assert_eq!(visited, sorted, "cycle must be in discriminant order");
    }

    #[test]
    fn names_are_unique_and_nonempty() {
        let mut seen = std::collections::HashSet::new();
        for &mode in &DebugViewMode::ALL {
            let name = mode.name();
            assert!(!name.is_empty(), "empty name for {:?}", mode);
            assert!(seen.insert(name), "duplicate name '{name}' for {:?}", mode);
        }
    }

    #[test]
    fn available_now_covers_expected_views() {
        // Off is trivially available.
        assert!(DebugViewMode::Off.available_now());
        // M6.2 climate views are backed by the packed climate_a / climate_b
        // vertex interpolants.
        assert!(DebugViewMode::AnnualTemperature.available_now());
        assert!(DebugViewMode::SeasonalTemperature.available_now());
        assert!(DebugViewMode::Precipitation.available_now());
        assert!(DebugViewMode::PetAridity.available_now());
        assert!(DebugViewMode::PrevailingWind.available_now());
        assert!(DebugViewMode::Continentality.available_now());
        // Geometric views derive from vertex normal + elevation attributes.
        assert!(DebugViewMode::Elevation.available_now());
        assert!(DebugViewMode::Slope.available_now());
        assert!(DebugViewMode::Aspect.available_now());
        // M6.3 fields are backed by packed canonical landform/substrate/soil data.
        assert!(DebugViewMode::Relief.available_now());
        assert!(DebugViewMode::Substrate.available_now());
        assert!(DebugViewMode::SoilProperties.available_now());
        // Remaining future fields are not yet available as shader inputs.
        let unavailable = [
            DebugViewMode::FlowAccumulation,
            DebugViewMode::Wetness,
            DebugViewMode::Disturbance,
            DebugViewMode::TopBiomeMemberships,
            DebugViewMode::EcotoneStrength,
            DebugViewMode::FinalMaterialMasks,
        ];
        for mode in unavailable {
            assert!(
                !mode.available_now(),
                "{:?} should not be available yet",
                mode
            );
        }
    }

    #[test]
    fn is_off_only_for_off() {
        assert!(DebugViewMode::Off.is_off());
        for &mode in &DebugViewMode::ALL[1..] {
            assert!(!mode.is_off(), "{:?} should not be off", mode);
        }
    }
}
