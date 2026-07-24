//! Versioned, machine-readable provenance for screenshots and benchmarks.

use bevy::prelude::*;
use er_core::config::{
    CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES, EARTH_RADIUS_M, LOD_SPLIT_BUDGET_PER_FRAME,
    MAX_INFLIGHT_TERRAIN_MESHES, MINIMUM_TERRAIN_COVERAGE_LOD,
};
use er_terrain::TerrainState;
use serde::Serialize;
use std::path::PathBuf;

use crate::settings::GraphicsSettings;

/// M6.1 performance budgets for the default (procedural Earth) path. These are
/// regression targets, not hard limits: the manifest records them so future
/// runs can detect drift.
#[derive(Clone, Serialize)]
pub struct PerformanceBudgets {
    pub max_frame_p95_regression_percent: f32,
    pub max_chunk_generation_regression_percent: f32,
    pub max_disabled_debug_overhead_ms: f32,
    pub max_enabled_debug_overhead_ms: f32,
    pub max_inflight_meshes: usize,
    pub lod_split_budget_per_frame: usize,
}

impl Default for PerformanceBudgets {
    fn default() -> Self {
        Self {
            max_frame_p95_regression_percent: 5.0,
            max_chunk_generation_regression_percent: 10.0,
            max_disabled_debug_overhead_ms: 0.05,
            max_enabled_debug_overhead_ms: 0.30,
            max_inflight_meshes: MAX_INFLIGHT_TERRAIN_MESHES,
            lod_split_budget_per_frame: LOD_SPLIT_BUDGET_PER_FRAME,
        }
    }
}

/// Summary of a single environment field's metadata (units, range, scale,
/// provenance, update frequency, authority).
#[derive(Clone, Serialize)]
pub struct FieldMetadataSummary {
    pub name: String,
    pub units: String,
    pub range_min: f32,
    pub range_max: f32,
    pub spatial_scale: String,
    pub provenance: String,
    pub update_frequency: String,
    pub authority: String,
    pub availability: String,
}

/// Legacy fixed-seed baseline statistics sampled from the current procedural
/// biome/environment pipeline. Used to lock pre-M6.2 behavior for regression.
#[derive(Clone, Serialize, Default)]
pub struct LegacyBaselineStats {
    pub sample_count: usize,
    pub land_fraction: f32,
    pub ocean_fraction: f32,
    pub mean_elevation_m: f32,
    pub min_elevation_m: f32,
    pub max_elevation_m: f32,
    pub mean_temperature: f32,
    pub min_temperature: f32,
    pub max_temperature: f32,
    pub mean_moisture: f32,
    pub min_moisture: f32,
    pub max_moisture: f32,
    pub biome_counts: Vec<(String, usize)>,
    pub biome_fractions: Vec<(String, f32)>,
}

/// M6.1 environment model identity and metadata for the manifest.
#[derive(Clone, Serialize)]
pub struct EnvironmentManifest {
    pub model_version: er_world::environment::EnvironmentModelVersion,
    pub preset: er_world::environment::EnvironmentPreset,
    pub legacy_pipeline_gaps: Vec<String>,
    pub climate_model: er_world::environment::HybridClimateModelContract,
    pub preset_expectations: er_world::environment::EnvironmentPresetExpectations,
    pub field_metadata: Vec<FieldMetadataSummary>,
    pub legacy_baseline: LegacyBaselineStats,
    pub legacy_baseline_matches_golden: bool,
    pub performance_budgets: PerformanceBudgets,
}

impl EnvironmentManifest {
    /// Collect the environment manifest from the er_world environment API.
    /// This performs fixed-seed baseline sampling (startup/opt-in only).
    pub fn collect(
        model: er_world::environment::EnvironmentModelVersion,
        preset: er_world::environment::EnvironmentPreset,
        baseline: &er_world::environment::LegacyEnvironmentBaseline,
        fields: &[er_world::environment::FieldMetadata],
    ) -> Self {
        let stats = LegacyBaselineStats {
            sample_count: baseline.sample_count,
            land_fraction: baseline.land_fraction,
            ocean_fraction: baseline.ocean_fraction,
            mean_elevation_m: baseline.mean_elevation_m,
            min_elevation_m: baseline.min_elevation_m,
            max_elevation_m: baseline.max_elevation_m,
            mean_temperature: baseline.mean_temperature,
            min_temperature: baseline.min_temperature,
            max_temperature: baseline.max_temperature,
            mean_moisture: baseline.mean_moisture,
            min_moisture: baseline.min_moisture,
            max_moisture: baseline.max_moisture,
            biome_counts: baseline.biome_counts.clone(),
            biome_fractions: baseline.biome_fractions.clone(),
        };
        let field_metadata = fields
            .iter()
            .map(|f| FieldMetadataSummary {
                name: f.name.to_owned(),
                units: f.units.to_owned(),
                range_min: f.range_min,
                range_max: f.range_max,
                spatial_scale: f.spatial_scale.to_owned(),
                provenance: f.provenance.as_str().to_owned(),
                update_frequency: f.update_frequency.as_str().to_owned(),
                authority: f.authority.as_str().to_owned(),
                availability: f.availability.as_str().to_owned(),
            })
            .collect();
        let preset_expectations = *er_world::environment::preset_expectations()
            .iter()
            .find(|expectations| expectations.preset == preset)
            .expect("every environment preset must have locked expectations");
        Self {
            model_version: model,
            preset,
            legacy_pipeline_gaps: er_world::environment::LEGACY_ENVIRONMENT_PIPELINE_GAPS
                .iter()
                .map(|gap| (*gap).to_owned())
                .collect(),
            climate_model: er_world::environment::HYBRID_CLIMATE_MODEL,
            preset_expectations,
            field_metadata,
            legacy_baseline: stats,
            legacy_baseline_matches_golden: baseline.matches_golden(),
            performance_budgets: PerformanceBudgets::default(),
        }
    }
}

#[derive(Clone)]
pub struct BaselineManifestPlugin {
    output_path: Option<PathBuf>,
    exit_after_write: bool,
    terrain_diffusion: Option<TerrainDiffusionManifest>,
    hardware_profile: Option<String>,
    seed: u64,
}

#[derive(Clone, Serialize)]
pub struct TerrainDiffusionManifest {
    pub endpoint: String,
    pub native_resolution: u16,
    pub native_pixel_scale_m: u16,
    pub api_scale: u8,
    pub halo_samples: u16,
    pub tiles_per_face_edge: u16,
    pub is_upsampled: bool,
}

impl BaselineManifestPlugin {
    pub fn new(
        output_path: Option<PathBuf>,
        exit_after_write: bool,
        terrain_diffusion: Option<TerrainDiffusionManifest>,
        hardware_profile: Option<String>,
        seed: u64,
    ) -> Self {
        Self {
            output_path,
            exit_after_write,
            terrain_diffusion,
            hardware_profile,
            seed,
        }
    }
}

impl Plugin for BaselineManifestPlugin {
    fn build(&self, app: &mut App) {
        let Some(output_path) = self.output_path.clone() else {
            return;
        };
        app.insert_resource(BaselineManifestRequest {
            output_path,
            exit_after_write: self.exit_after_write,
            terrain_diffusion: self.terrain_diffusion.clone(),
            hardware_profile: self.hardware_profile.clone(),
            seed: self.seed,
        })
        .add_systems(Startup, write_manifest);
    }
}

#[derive(Resource)]
struct BaselineManifestRequest {
    output_path: PathBuf,
    exit_after_write: bool,
    terrain_diffusion: Option<TerrainDiffusionManifest>,
    hardware_profile: Option<String>,
    seed: u64,
}

#[derive(Serialize)]
struct BaselineManifest {
    format: &'static str,
    build: BuildManifest,
    terrain: TerrainManifest,
    presentation: PresentationManifest,
    gpu: GpuManifest,
    terrain_diffusion: Option<TerrainDiffusionManifest>,
    hardware_profile: Option<String>,
    environment: EnvironmentManifest,
}

#[derive(Serialize)]
struct BuildManifest {
    app_version: &'static str,
    rustc_version: &'static str,
    target: &'static str,
    profile: &'static str,
    bevy_version: &'static str,
    terrain_diffusion_feature: bool,
}

#[derive(Serialize)]
struct TerrainManifest {
    preset: String,
    source_mode: String,
    radius_m: f64,
    elevation_scale_m: f32,
    max_quadtree_depth: u8,
    screen_error_threshold: f32,
    merge_hysteresis: f32,
    lod_split_budget_per_frame: usize,
    max_render_distance_m: f64,
    seed: u64,
    chunk_vertex_resolution: usize,
    chunk_quads_per_edge: usize,
    minimum_coverage_lod: u8,
}

#[derive(Serialize)]
struct PresentationManifest {
    present_mode: String,
    vsync: bool,
    fullscreen: bool,
    msaa_samples: u32,
    desired_maximum_frame_latency: u32,
}

#[derive(Serialize)]
struct GpuManifest {
    telemetry_available: bool,
    status: String,
    adapter: Option<String>,
    vendor_id: Option<u32>,
    device_id: Option<u32>,
    dedicated_video_memory_bytes: Option<u64>,
    vram_budget_bytes: Option<u64>,
    vram_usage_bytes: Option<u64>,
}

fn write_manifest(
    request: Res<BaselineManifestRequest>,
    terrain: Res<TerrainState>,
    settings: Res<GraphicsSettings>,
    mut exit: MessageWriter<AppExit>,
) {
    let baseline = er_world::environment::collect_legacy_environment_baseline(
        request.seed,
        terrain.environment_preset,
        terrain.planet_radius,
        terrain.elevation_scale,
        er_world::environment::DEFAULT_BASELINE_SAMPLE_COUNT,
    );
    let environment = EnvironmentManifest::collect(
        er_world::environment::EnvironmentModelVersion::current(),
        terrain.environment_preset,
        &baseline,
        er_world::environment::field_metadata(),
    );
    let manifest = collect_manifest(
        &terrain,
        &settings,
        request.terrain_diffusion.clone(),
        request.hardware_profile.clone(),
        request.seed,
        environment,
    );
    let Ok(json) = serde_json::to_vec_pretty(&manifest) else {
        error!("Could not serialize baseline manifest");
        if request.exit_after_write {
            exit.write(AppExit::error());
        }
        return;
    };
    if let Some(parent) = request.output_path.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            error!(path = ?request.output_path, %error, "Could not create baseline manifest directory");
            if request.exit_after_write {
                exit.write(AppExit::error());
            }
            return;
        }
    }
    if let Err(error) = std::fs::write(&request.output_path, json) {
        error!(path = ?request.output_path, %error, "Could not write baseline manifest");
        if request.exit_after_write {
            exit.write(AppExit::error());
        }
        return;
    }
    info!(path = ?request.output_path, "Wrote baseline manifest");
    if request.exit_after_write {
        exit.write(AppExit::Success);
    }
}

fn collect_manifest(
    terrain: &TerrainState,
    settings: &GraphicsSettings,
    terrain_diffusion: Option<TerrainDiffusionManifest>,
    hardware_profile: Option<String>,
    seed: u64,
    environment: EnvironmentManifest,
) -> BaselineManifest {
    let sample = er_game::gpu_telemetry::sample();
    let available = sample.is_available();
    let status = format!("{:?}", sample.status);
    BaselineManifest {
        format: "enigmatic-rts-baseline-manifest/v2",
        build: BuildManifest {
            app_version: env!("CARGO_PKG_VERSION"),
            rustc_version: env!("ER_BUILD_RUSTC"),
            target: env!("ER_BUILD_TARGET"),
            profile: env!("ER_BUILD_PROFILE"),
            bevy_version: "0.19",
            terrain_diffusion_feature: cfg!(feature = "terrain_diffusion"),
        },
        terrain: TerrainManifest {
            preset: if terrain.planet_radius == EARTH_RADIUS_M {
                "EarthScale"
            } else {
                "MiniatureDebug"
            }
            .to_owned(),
            source_mode: format!("{:?}", terrain.source_mode),
            radius_m: terrain.planet_radius,
            elevation_scale_m: terrain.elevation_scale,
            max_quadtree_depth: terrain.max_quadtree_depth,
            screen_error_threshold: terrain.screen_error_threshold,
            merge_hysteresis: terrain.merge_hysteresis,
            lod_split_budget_per_frame: terrain.lod_split_budget_per_frame,
            max_render_distance_m: terrain.max_render_distance,
            seed,
            chunk_vertex_resolution: CHUNK_VERT_RES as usize,
            chunk_quads_per_edge: CHUNK_QUADS_PER_EDGE as usize,
            minimum_coverage_lod: MINIMUM_TERRAIN_COVERAGE_LOD,
        },
        presentation: PresentationManifest {
            present_mode: format!("{:?}", settings.present_mode()),
            vsync: settings.vsync,
            fullscreen: settings.fullscreen,
            msaa_samples: settings.msaa,
            desired_maximum_frame_latency: 3,
        },
        gpu: GpuManifest {
            telemetry_available: available,
            status,
            adapter: available.then_some(sample.description),
            vendor_id: available.then_some(sample.vendor_id),
            device_id: available.then_some(sample.device_id),
            dedicated_video_memory_bytes: available.then_some(sample.dedicated_video_memory_bytes),
            vram_budget_bytes: available.then_some(sample.vram_budget_bytes),
            vram_usage_bytes: available.then_some(sample.vram_usage_bytes),
        },
        terrain_diffusion,
        hardware_profile,
        environment,
    }
}

pub fn parse_dump_path() -> Option<PathBuf> {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|pair| pair[0] == "--dump-baseline-manifest")
        .map(|pair| PathBuf::from(&pair[1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::config::PlanetPreset;
    use er_terrain::TerrainState;

    fn empty_terrain_state() -> TerrainState {
        TerrainState::for_preset(
            PlanetPreset::EarthScale,
            1000.0,
            er_core::seed::PlanetSeed(0xC0FFEE),
        )
    }

    fn empty_settings() -> GraphicsSettings {
        GraphicsSettings::default()
    }

    fn sample_environment_manifest(preset: &str) -> EnvironmentManifest {
        EnvironmentManifest {
            model_version: er_world::environment::EnvironmentModelVersion::current(),
            preset: match preset {
                "HotArid" => er_world::environment::EnvironmentPreset::HotArid,
                "ColdIcehouse" => er_world::environment::EnvironmentPreset::ColdIcehouse,
                _ => er_world::environment::EnvironmentPreset::EarthLike,
            },
            legacy_pipeline_gaps: er_world::environment::LEGACY_ENVIRONMENT_PIPELINE_GAPS
                .iter()
                .map(|gap| (*gap).to_owned())
                .collect(),
            climate_model: er_world::environment::HYBRID_CLIMATE_MODEL,
            preset_expectations: *er_world::environment::preset_expectations()
                .iter()
                .find(|expected| {
                    expected.preset
                        == match preset {
                            "HotArid" => er_world::environment::EnvironmentPreset::HotArid,
                            "ColdIcehouse" => {
                                er_world::environment::EnvironmentPreset::ColdIcehouse
                            }
                            _ => er_world::environment::EnvironmentPreset::EarthLike,
                        }
                })
                .unwrap(),
            field_metadata: vec![FieldMetadataSummary {
                name: "temperature".to_owned(),
                units: "degrees_celsius".to_owned(),
                range_min: -90.0,
                range_max: 60.0,
                spatial_scale: "global_atlas".to_owned(),
                provenance: "procedural".to_owned(),
                update_frequency: "static".to_owned(),
                authority: "canonical".to_owned(),
                availability: "planned_m6_2".to_owned(),
            }],
            legacy_baseline: LegacyBaselineStats {
                sample_count: 1000,
                land_fraction: 0.71,
                ocean_fraction: 0.29,
                mean_elevation_m: 250.0,
                min_elevation_m: -8000.0,
                max_elevation_m: 6000.0,
                mean_temperature: 0.5,
                min_temperature: 0.0,
                max_temperature: 1.0,
                mean_moisture: 0.5,
                min_moisture: 0.0,
                max_moisture: 1.0,
                biome_counts: vec![("Ocean".to_owned(), 290), ("Plains".to_owned(), 400)],
                biome_fractions: vec![("Ocean".to_owned(), 0.29), ("Plains".to_owned(), 0.40)],
            },
            legacy_baseline_matches_golden: false,
            performance_budgets: PerformanceBudgets::default(),
        }
    }

    #[test]
    fn real_environment_manifest_collection_covers_every_preset() {
        for preset in [
            er_world::environment::EnvironmentPreset::EarthLike,
            er_world::environment::EnvironmentPreset::HotArid,
            er_world::environment::EnvironmentPreset::ColdIcehouse,
        ] {
            let baseline = er_world::environment::collect_legacy_environment_baseline(
                er_world::environment::LEGACY_BASELINE_SEED,
                preset,
                6_371_000.0,
                1000.0,
                er_world::environment::DEFAULT_BASELINE_SAMPLE_COUNT,
            );
            let manifest = EnvironmentManifest::collect(
                er_world::environment::EnvironmentModelVersion::current(),
                preset,
                &baseline,
                er_world::environment::field_metadata(),
            );
            assert_eq!(manifest.preset, preset);
            assert_eq!(manifest.field_metadata.len(), 61);
            assert!(manifest.legacy_baseline_matches_golden);
        }
    }

    #[test]
    fn terrain_diffusion_manifest_marks_upsampling() {
        let manifest = TerrainDiffusionManifest {
            endpoint: "127.0.0.1:8000".to_owned(),
            native_resolution: 512,
            native_pixel_scale_m: 30,
            api_scale: 1,
            halo_samples: 1,
            tiles_per_face_edge: 652,
            is_upsampled: false,
        };
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("native_pixel_scale_m"));
        assert!(json.contains("\"is_upsampled\":false"));
    }

    #[test]
    fn manifest_includes_all_milestone_zero_fields() {
        let terrain = empty_terrain_state();
        let settings = empty_settings();
        let diffusion = Some(TerrainDiffusionManifest {
            endpoint: "127.0.0.1:8000".to_owned(),
            native_resolution: 512,
            native_pixel_scale_m: 30,
            api_scale: 1,
            halo_samples: 1,
            tiles_per_face_edge: 652,
            is_upsampled: true,
        });
        let manifest = collect_manifest(
            &terrain,
            &settings,
            diffusion,
            Some("rtx3060_optimus".to_owned()),
            0xC0FFEE,
            sample_environment_manifest("EarthLike"),
        );
        let json = serde_json::to_string(&manifest).unwrap();
        // Roadmap 0.1.1: rust version, bevy version, GPU adapter, present mode,
        // terrain preset, LOD config, and Terrain Diffusion metadata.
        assert!(json.contains("\"rustc_version\""));
        assert!(json.contains("\"bevy_version\""));
        assert!(json.contains("\"adapter\""));
        assert!(json.contains("\"present_mode\""));
        assert!(json.contains("\"preset\""));
        assert!(json.contains("\"max_quadtree_depth\""));
        assert!(json.contains("\"terrain_diffusion\""));
        assert!(json.contains("\"native_pixel_scale_m\""));
        // Hardware profile reference (0.1.5).
        assert!(json.contains("\"hardware_profile\""));
        assert!(json.contains("rtx3060_optimus"));
    }

    #[test]
    fn manifest_without_diffusion_omits_metadata_cleanly() {
        let terrain = empty_terrain_state();
        let settings = empty_settings();
        let manifest = collect_manifest(
            &terrain,
            &settings,
            None,
            None,
            0xC0FFEE,
            sample_environment_manifest("EarthLike"),
        );
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"terrain_diffusion\":null"));
        assert!(json.contains("\"hardware_profile\":null"));
    }

    #[test]
    fn manifest_format_version_is_v2() {
        let terrain = empty_terrain_state();
        let settings = empty_settings();
        let manifest = collect_manifest(
            &terrain,
            &settings,
            None,
            None,
            0xC0FFEE,
            sample_environment_manifest("EarthLike"),
        );
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("enigmatic-rts-baseline-manifest/v2"));
    }

    #[test]
    fn manifest_includes_environment_model_identity() {
        let terrain = empty_terrain_state();
        let settings = empty_settings();
        let manifest = collect_manifest(
            &terrain,
            &settings,
            None,
            None,
            0xC0FFEE,
            sample_environment_manifest("EarthLike"),
        );
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"environment\""));
        assert!(json.contains("\"model_version\""));
        assert!(json.contains("\"model\":2"));
        assert!(json.contains("\"legacy_pipeline_gaps\""));
        assert!(json.contains("\"climate_model\""));
        assert!(json.contains("diagnostic_and_regression_label"));
    }

    #[test]
    fn manifest_includes_environment_preset_label() {
        let terrain = empty_terrain_state();
        let settings = empty_settings();
        let manifest = collect_manifest(
            &terrain,
            &settings,
            None,
            None,
            0xC0FFEE,
            sample_environment_manifest("HotArid"),
        );
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"preset\":\"hot_arid\""));
    }

    #[test]
    fn manifest_includes_cold_icehouse_preset_label() {
        let terrain = empty_terrain_state();
        let settings = empty_settings();
        let manifest = collect_manifest(
            &terrain,
            &settings,
            None,
            None,
            0xC0FFEE,
            sample_environment_manifest("ColdIcehouse"),
        );
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"preset\":\"cold_icehouse\""));
    }

    #[test]
    fn manifest_includes_field_metadata_summary() {
        let terrain = empty_terrain_state();
        let settings = empty_settings();
        let manifest = collect_manifest(
            &terrain,
            &settings,
            None,
            None,
            0xC0FFEE,
            sample_environment_manifest("EarthLike"),
        );
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"field_metadata\""));
        assert!(json.contains("temperature"));
        assert!(json.contains("degrees_celsius"));
        assert!(json.contains("\"provenance\""));
        assert!(json.contains("\"authority\""));
    }

    #[test]
    fn manifest_includes_legacy_baseline_statistics() {
        let terrain = empty_terrain_state();
        let settings = empty_settings();
        let manifest = collect_manifest(
            &terrain,
            &settings,
            None,
            None,
            0xC0FFEE,
            sample_environment_manifest("EarthLike"),
        );
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"legacy_baseline\""));
        assert!(json.contains("\"sample_count\""));
        assert!(json.contains("\"land_fraction\""));
        assert!(json.contains("\"ocean_fraction\""));
        assert!(json.contains("\"biome_counts\""));
        assert!(json.contains("\"legacy_baseline_matches_golden\""));
    }

    #[test]
    fn manifest_includes_performance_budgets() {
        let terrain = empty_terrain_state();
        let settings = empty_settings();
        let manifest = collect_manifest(
            &terrain,
            &settings,
            None,
            None,
            0xC0FFEE,
            sample_environment_manifest("EarthLike"),
        );
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"performance_budgets\""));
        assert!(json.contains("\"max_frame_p95_regression_percent\""));
        assert!(json.contains("\"max_disabled_debug_overhead_ms\""));
        assert!(json.contains("\"max_inflight_meshes\""));
        assert!(json.contains("\"lod_split_budget_per_frame\""));
    }

    #[test]
    fn performance_budgets_defaults_match_config() {
        let budgets = PerformanceBudgets::default();
        assert_eq!(budgets.max_inflight_meshes, MAX_INFLIGHT_TERRAIN_MESHES);
        assert_eq!(
            budgets.lod_split_budget_per_frame,
            LOD_SPLIT_BUDGET_PER_FRAME
        );
    }
}
