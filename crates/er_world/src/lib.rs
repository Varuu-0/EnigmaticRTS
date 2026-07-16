//! er_world: procedural generation — elevation, biomes, water, planet params,
//! solar-system generation, Kepler orbits, and the query cache.
//! Planet-creation pass.

pub mod biome;
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
pub use params::{climate_noise, planet_params, ClimateNoise, PlanetParams};
pub use surface_tiles::{
    LearnedTerrainTile, LearnedTileCache, LearnedTileGeneration, LearnedTileKey, TileCoordinate,
    TileInsertError,
};
pub use terrain_field::{
    HybridTerrainField, MacroTerrainField, MacroTerrainSample, ProceduralTerrainField,
    TerrainField, TerrainSample, TerrainSampleSource, TerrainSourceMode,
};
pub use terrain_space::{
    metric_surface_point, vertex_spacing_m, CONTINENTAL_WAVELENGTH_M, FOOTHILL_WAVELENGTH_M,
    MOUNTAIN_WAVELENGTH_M, RIDGE_WAVELENGTH_M, WARP_WAVELENGTH_M,
};
pub use water::{
    is_ocean, is_water, lake_surface_elev, ocean_depth_band, water_depth, water_surface_elev,
};

pub fn version() -> &'static str {
    "0"
}
