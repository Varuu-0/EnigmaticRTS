//! er_world: procedural generation — elevation, biomes, water, planet params,
//! solar-system generation, Kepler orbits, and the query cache.
//! Planet-creation pass.

pub mod biome;
pub mod cache;
pub mod elevation;
pub mod params;
pub mod water;

pub use biome::{
    biome, classify_biome, elevation_low_freq, elevation_split, moisture, temperature, Biome,
    BiomeData, BiomeRegistry, ElevationSplit, LowFreqElevation,
};
pub use cache::{CachedWorldData, WorldCache, CACHE_LOD};
pub use params::{climate_noise, planet_params, ClimateNoise, PlanetParams};
pub use water::{
    is_ocean, is_water, lake_surface_elev, ocean_depth_band, water_depth, water_surface_elev,
};

pub fn version() -> &'static str {
    "0"
}
