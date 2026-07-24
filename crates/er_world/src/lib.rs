//! er_world: procedural generation — elevation, biomes, water, planet params,
//! solar-system generation, Kepler orbits, and the query cache.
//! Planet-creation pass.

pub mod biome;
pub mod brushes;
pub mod cache;
pub mod climate;
pub mod elevation;
pub mod environment;
pub mod geology;
pub mod landform;
pub mod params;
pub mod soil;
pub mod streaming;
pub mod surface_cache;
pub mod surface_charts;
pub mod surface_tiles;
pub mod terrain_field;
pub mod terrain_space;
pub mod water;

pub use climate::{
    ClimateAtlas, ClimateAtlasBuildStats, ClimateAtlasIdentity, ClimateSample,
    ClimateSampleValidity, CLIMATE_ATLAS_LEN, CLIMATE_ATLAS_LOD, CLIMATE_CELLS_PER_EDGE,
};
pub use environment::{
    collect_legacy_environment_baseline, field_metadata, planet_params_with_preset, BiomeMixture,
    ClimateNormals, DisturbanceSample, EnvironmentModelVersion, EnvironmentPreset,
    EnvironmentSample, FieldMetadata, HydrologySample, LandformSample, LegacyEnvironmentBaseline,
    SoilSample, WorldEnvironmentIdentity, WorldEnvironmentSnapshot, ENVIRONMENT_MODEL_VERSION,
    HYBRID_CLIMATE_MODEL, LEGACY_ENVIRONMENT_PIPELINE_GAPS,
};
pub use geology::{
    GeologyAtlas, GeologyAtlasBuildStats, GeologyAtlasIdentity, SubstrateClass, SubstrateSample,
    SubstrateSampleValidity, GEOLOGY_ATLAS_LEN, GEOLOGY_ATLAS_LOD, GEOLOGY_CELLS_PER_EDGE,
};
pub use landform::{
    LandformAtlas, LandformAtlasBuildStats, LandformAtlasIdentity, LandformClass, LandformSampleV2,
    LANDFORM_ATLAS_LEN, LANDFORM_ATLAS_LOD, LANDFORM_ATLAS_VERSION,
};
pub use soil::{
    classify_drainage, sample_soil, SoilDrainageClass, SoilFunctionalSample, SoilFunctionalValidity,
};

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
pub use surface_cache::{
    ChartMacroField, CreationMetadata, SurfaceCache, SurfaceCacheError, SurfaceCacheKey,
    SurfaceDiskCache, SurfaceRamCache, SurfaceTileRecord, SURFACE_CACHE_FORMAT_VERSION,
};
pub use surface_charts::{
    all_cube_corners, all_cube_edges, edge_uv, faces_at_corner, field_normal_deg, ChartOwnership,
    ChartSamplePoint, SurfaceChartField, SurfaceChartId, SurfaceChartMetadata, SurfacePatchId,
    SurfaceRegion, SyntheticGradientField, SURFACE_CHART_PROJECTION_REVISION,
};
pub use surface_tiles::{
    LearnedTerrainTile, LearnedTileCache, LearnedTileGeneration, LearnedTileKey, TileCoordinate,
    TileInsertError,
};
pub use terrain_field::{
    BlendTransitionChecker, BlendedHybridTerrainField, ChunkFieldSnapshot, GlobalMacroField,
    HaloResidencyChecker, HybridTerrainField, MacroSample, MacroTerrainField, MacroTerrainSample,
    ProceduralResidualField, ProceduralTerrainField, ResidualSample, TerrainField, TerrainSample,
    TerrainSampleSource, TerrainSourceMode, VisualClimate,
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
