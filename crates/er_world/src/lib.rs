//! er_world: procedural generation — elevation, biomes, water, planet params,
//! solar-system generation, Kepler orbits, and the query cache.
//! Planet-creation pass.

pub mod biome;
pub mod brushes;
pub mod cache;
pub mod elevation;
pub mod params;
pub mod surface_tiles;
pub mod terrain_field;
pub mod terrain_space;
pub mod water;

pub use biome::{
    biome, classify_biome, elevation_low_freq, elevation_split, moisture, temperature, Biome,
    BiomeData, BiomeRegistry, ElevationSplit, LowFreqElevation,
};
pub use cache::{CachedWorldData, WorldCache, DEFAULT_CACHE_LOD as CACHE_LOD};
pub use elevation::{
    elevation_at, elevation_metric_full_eval, metric_landform_sample, ElevationNoise,
    ElevationParams, MetricLandformSample,
};
pub use params::{climate_noise, planet_params, ClimateNoise, PlanetParams};
pub use surface_tiles::{
    LearnedTerrainTile, LearnedTileCache, LearnedTileGeneration, LearnedTileKey, TileCoordinate,
    TileInsertError,
};
pub use terrain_field::{
    GlobalMacroField, HybridTerrainField, MacroSample, MacroTerrainField, MacroTerrainSample,
    ProceduralResidualField, ProceduralTerrainField, ResidualSample, TerrainField, TerrainSample,
    TerrainSampleSource, TerrainSourceMode,
};
pub use terrain_space::{
    metric_surface_point, vertex_spacing_m, CONTINENTAL_WAVELENGTH_M, DRAINAGE_WAVELENGTH_M,
    EROSION_WAVELENGTH_M, FOOTHILL_WAVELENGTH_M, METRIC_CONTINENTAL_AMP, METRIC_DETAIL_AMP,
    METRIC_DRAINAGE_AMP, METRIC_HILL_AMP, METRIC_MOUNTAIN_AMP, METRIC_RIDGE_DETAIL_AMP,
    METRIC_TALUS_AMP, METRIC_TECTONIC_AMP, METRIC_VALLEY_AMP, MICRO_DETAIL_WAVELENGTH_M,
    MOUNTAIN_WAVELENGTH_M, RIDGE_DETAIL_WAVELENGTH_M, RIDGE_WAVELENGTH_M, TALUS_WAVELENGTH_M,
    TECTONIC_BELT_WAVELENGTH_M, VALLEY_WAVELENGTH_M, WARP_WAVELENGTH_M,
};
pub use water::{
    is_ocean, is_water, lake_surface_elev, ocean_depth_band, water_depth, water_surface_elev,
};

pub fn version() -> &'static str {
    "0"
}
