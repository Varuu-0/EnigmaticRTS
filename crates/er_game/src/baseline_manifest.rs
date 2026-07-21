//! Versioned, machine-readable provenance for screenshots and benchmarks.

use bevy::prelude::*;
use er_core::config::{
    CHUNK_QUADS_PER_EDGE, CHUNK_VERT_RES, EARTH_RADIUS_M, MINIMUM_TERRAIN_COVERAGE_LOD,
};
use er_terrain::TerrainState;
use serde::Serialize;
use std::path::PathBuf;

use crate::settings::GraphicsSettings;

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
    let manifest = collect_manifest(
        &terrain,
        &settings,
        request.terrain_diffusion.clone(),
        request.hardware_profile.clone(),
        request.seed,
    );
    let Ok(json) = serde_json::to_vec_pretty(&manifest) else {
        error!("Could not serialize baseline manifest");
        return;
    };
    if let Some(parent) = request.output_path.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            error!(path = ?request.output_path, %error, "Could not create baseline manifest directory");
            return;
        }
    }
    if let Err(error) = std::fs::write(&request.output_path, json) {
        error!(path = ?request.output_path, %error, "Could not write baseline manifest");
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
) -> BaselineManifest {
    let sample = er_game::gpu_telemetry::sample();
    let available = sample.is_available();
    let status = format!("{:?}", sample.status);
    BaselineManifest {
        format: "enigmatic-rts-baseline-manifest/v1",
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
        assert!(json.contains("false"));
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
        let manifest = collect_manifest(&terrain, &settings, None, None, 0xC0FFEE);
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"terrain_diffusion\":null"));
        assert!(json.contains("\"hardware_profile\":null"));
    }
}
