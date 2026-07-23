use bevy::prelude::*;
use er_core::config::{DEFAULT_DAY_LENGTH_SEC, MAX_QUADTREE_DEPTH};
use er_core::math::{cells_per_edge, dir_to_surface, uv_to_dir, world_to_render, OriginOffset};
use er_terrain::{ChunkComponent, FrameProfiler, RenderOrigin, TerrainDebugInfo, TerrainState};

use crate::diagnostics::{PerformanceSnapshot, PerformanceSnapshotUpdate};
use crate::frame_timing::MainWorldFrameTimings;
use crate::space::{SimTime, TimeScale};
use er_terrain::SunDirection;

const FRAME_GRAPH_WIDTH: usize = 40;

#[derive(Component)]
struct DebugText;

#[derive(Resource, Default)]
pub struct LodDebugDraw(pub bool);

pub struct DebugOverlayPlugin;

impl Plugin for DebugOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(LodDebugDraw::default())
            .add_systems(Startup, setup_debug_text)
            .add_systems(
                PostUpdate,
                update_debug_text.after(PerformanceSnapshotUpdate),
            )
            .add_systems(
                Update,
                (toggle_lod_debug, draw_lod_gizmos).after(er_terrain::TerrainUpdate),
            );
    }
}

fn setup_debug_text(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(5.0),
            left: Val::Px(5.0),
            ..default()
        },
        DebugText,
    ));
}

#[allow(clippy::too_many_arguments)]
fn update_debug_text(
    debug: Res<TerrainDebugInfo>,
    profiler: Res<FrameProfiler>,
    main_world_timings: Res<MainWorldFrameTimings>,
    sun_direction: Res<SunDirection>,
    sim_time: Res<SimTime>,
    time_scale: Res<TimeScale>,
    performance: Res<PerformanceSnapshot>,
    mut query: Query<&mut Text, With<DebugText>>,
    mut displayed_revision: Local<u64>,
) {
    if performance.sample_revision == 0 || performance.sample_revision == *displayed_revision {
        return;
    }
    *displayed_revision = performance.sample_revision;

    if let Ok(mut text) = query.single_mut() {
        let mut sorted: Vec<(&'static str, std::time::Duration)> = profiler.timings.clone();
        sorted.sort_by_key(|&(_, d)| std::cmp::Reverse(d));

        let total_profiled: std::time::Duration = sorted.iter().map(|(_, d)| *d).sum();
        let frame_duration = main_world_timings
            .frame_duration
            .unwrap_or_else(|| std::time::Duration::from_secs_f32(performance.frame_ms / 1000.0));
        let attribution_ms = frame_duration.as_secs_f32() * 1000.0;
        let total_main_world: std::time::Duration = main_world_timings
            .stages
            .iter()
            .map(|(_, duration)| *duration)
            .sum();
        // Render submission, GPU synchronization, present, and frame pacing
        // happen outside the main-world schedules. Render CPU spans below
        // overlap the main world, so they are deliberately not subtracted here.
        let render_present_wait = frame_duration.saturating_sub(total_main_world);
        let render_present_stage = ("render/present/frame pacing", render_present_wait);

        let sun = sun_direction.0;
        let day_length = DEFAULT_DAY_LENGTH_SEC as f32;
        let day_percent = ((sim_time.0.rem_euclid(day_length)) / day_length * 100.0).round();
        let speed_str = if time_scale.current == 0.0 {
            "PAUSED".to_string()
        } else {
            format!("{:.1}x", time_scale.current)
        };

        let mut lines = String::new();
        lines.push_str(&format!(
            "FPS: {:.0} | Frame: {:.1}ms | P95/P99: {:.1}/{:.1}ms | 1%: {:.0}\n",
            performance.rolling_fps,
            performance.frame_ms,
            performance.frame_p95_ms,
            performance.frame_p99_ms,
            performance.one_percent_low_fps,
        ));
        lines.push_str(&format!(
            "Chunks: {} | LOD: {} | Split/Merge/Mesh: {}/{}/{} | Terrain mesh: {:.1} MiB | Built: {} | Draw work: {}\n",
            debug.active_chunks,
            debug.max_depth,
            debug.pending_splits,
            debug.pending_merges,
            debug.pending_meshes,
            debug.estimated_mesh_bytes as f64 / (1024.0 * 1024.0),
            debug.meshes_built,
            performance.visible_mesh_draw_estimate,
        ));
        lines.push_str(&format!(
            "Alt: {:.2} km | Origin: ({:.1},{:.1},{:.1}) gen {} | View-LOD {} width {:.0}m | Vtx/Nsp {:.2}/{:.2}m span {:.2}m eps {:.3e}rad | {:?} p:{:.0}% l:{:.0}%\n",
            debug.camera_altitude_m / 1000.0,
            debug.render_origin_world.x, debug.render_origin_world.y, debug.render_origin_world.z,
            debug.render_origin_generation,
            debug.nearest_chunk_lod,
            debug.nearest_chunk_width_m,
            debug.vertex_spacing_m,
            debug.normal_diff_spacing_m,
            debug.normal_difference_span_m,
            debug.normal_diff_epsilon_radians,
            debug.source_mode,
            debug.procedural_source_coverage_percent,
            debug.learned_source_coverage_percent,
        ));

        let process_memory = performance
            .process_memory_gib
            .map(|memory| format!("{memory:.2} GiB"))
            .unwrap_or_else(|| "waiting".to_owned());
        let process_cpu = performance
            .process_cpu_percent
            .map(|cpu| format!("{cpu:.0}%"))
            .unwrap_or_else(|| "waiting".to_owned());
        let gpu_vram = match (
            performance.gpu_vram_usage_bytes,
            performance.gpu_vram_budget_bytes,
        ) {
            (Some(usage), Some(budget)) => format!(
                "{:.2}/{:.2} GiB",
                usage as f64 / (1024.0 * 1024.0 * 1024.0),
                budget as f64 / (1024.0 * 1024.0 * 1024.0),
            ),
            _ => "unavailable".to_owned(),
        };
        let mesh_allocator = performance
            .mesh_allocator_bytes
            .map(|bytes| format!("{:.1} MiB", bytes / (1024.0 * 1024.0)))
            .unwrap_or_else(|| "waiting".to_owned());
        lines.push_str(&format!(
            "CPU: {} | RAM: {} | VRAM: {} | Mesh slabs: {} | Hitches 16/33/50: {}/{}/{}\n",
            process_cpu,
            process_memory,
            gpu_vram,
            mesh_allocator,
            performance.hitch_16ms_count,
            performance.hitch_33ms_count,
            performance.hitch_50ms_count,
        ));
        let opaque_gpu = performance
            .opaque_render_gpu_ms
            .map(|value| format!("{value:.2}ms"))
            .unwrap_or_else(|| "enable --gpu-diagnostics".to_owned());
        let opaque_cpu = performance
            .opaque_render_cpu_ms
            .map(|value| format!("{value:.2}ms"))
            .unwrap_or_else(|| "waiting".to_owned());
        lines.push_str(&format!(
            "Opaque pass CPU/GPU: {}/{}\n",
            opaque_cpu, opaque_gpu,
        ));
        let input_to_cpu = performance
            .input_to_cpu_frame_end_ms
            .map(|value| format!("{value:.2}ms"))
            .unwrap_or_else(|| "press a key or mouse button".to_owned());
        lines.push_str(&format!("Input to CPU frame end: {}\n", input_to_cpu));
        lines.push_str(&format!(
            "Sun: ({:.2}, {:.2}, {:.2}) | Day: {:.0}% | Speed: {}\n",
            sun.x, sun.y, sun.z, day_percent, speed_str
        ));
        lines.push_str(&format!(
            "Completed-frame attribution: {:.2}ms main world + {:.2}ms render/present = {:.1}ms\n",
            total_main_world.as_secs_f32() * 1000.0,
            render_present_wait.as_secs_f32() * 1000.0,
            attribution_ms,
        ));

        for (name, duration) in main_world_timings
            .stages
            .iter()
            .chain(std::iter::once(&render_present_stage))
        {
            let ms = duration.as_secs_f32() * 1000.0;
            let fraction = if attribution_ms > 0.0 {
                ms / attribution_ms
            } else {
                0.0
            };
            let bar_len = (fraction * FRAME_GRAPH_WIDTH as f32).round() as usize;
            let bar = "#".repeat(bar_len);
            lines.push_str(&format!(
                "  {:<22} {:>6.2}ms {:>5.1}% |{:<width$}|\n",
                name,
                ms,
                fraction * 100.0,
                bar,
                width = FRAME_GRAPH_WIDTH,
            ));
        }

        lines.push_str(&format!(
            "Instrumented Update detail: {:.2}ms (included in Update above)\n",
            total_profiled.as_secs_f32() * 1000.0,
        ));
        for (name, duration) in &sorted {
            lines.push_str(&format!(
                "  {:<22} {:>6.2}ms\n",
                name,
                duration.as_secs_f32() * 1000.0,
            ));
        }

        if !performance.render_cpu_spans.is_empty() {
            lines.push_str("Render CPU spans (parallel; not additive):\n");
            for (path, ms) in &performance.render_cpu_spans {
                let name = path
                    .trim_start_matches("render/")
                    .trim_end_matches("/elapsed_cpu");
                lines.push_str(&format!("  {:<22} {:>6.2}ms\n", name, ms));
            }
        }

        *text = Text::new(lines);
    }
}

fn toggle_lod_debug(keys: Res<ButtonInput<KeyCode>>, mut draw: ResMut<LodDebugDraw>) {
    if keys.just_pressed(KeyCode::F3) {
        draw.0 = !draw.0;
        info!("LOD debug draw: {}", draw.0);
    }
}

fn draw_lod_gizmos(
    draw: Res<LodDebugDraw>,
    terrain_state: Res<TerrainState>,
    render_origin: Res<RenderOrigin>,
    chunks: Query<&ChunkComponent>,
    mut gizmos: Gizmos,
) {
    if !draw.0 {
        return;
    }

    let radius = terrain_state.planet_radius;
    let origin = OriginOffset(render_origin.world);

    for chunk in &chunks {
        let key = chunk.key;
        let n = cells_per_edge(key.lod) as f64;
        let u0 = key.i as f64 / n;
        let u1 = (key.i as f64 + 1.0) / n;
        let v0 = key.j as f64 / n;
        let v1 = (key.j as f64 + 1.0) / n;

        let c0 = uv_to_dir(key.face, u0, v0);
        let c1 = uv_to_dir(key.face, u1, v0);
        let c2 = uv_to_dir(key.face, u1, v1);
        let c3 = uv_to_dir(key.face, u0, v1);

        let to_render = |dir| world_to_render(dir_to_surface(dir, radius, 0.0), origin).0;
        let pts = [
            to_render(c0),
            to_render(c1),
            to_render(c2),
            to_render(c3),
            to_render(c0),
        ];

        let hue = (key.lod as f32 / MAX_QUADTREE_DEPTH as f32) * 360.0;
        let color = Color::hsl(hue, 0.9, 0.5);
        gizmos.linestrip(pts, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use er_core::config::CHUNK_QUADS_PER_EDGE;
    use er_core::math::cell_size;
    use er_terrain::RenderOrigin;

    #[test]
    fn altitude_from_radius_and_field_zero() {
        let info = TerrainDebugInfo::default();
        assert_eq!(info.camera_altitude_m, 0.0);
    }

    #[test]
    fn render_origin_default_holds_zero_world() {
        let info = TerrainDebugInfo::default();
        assert_eq!(info.render_origin_world, glam::DVec3::ZERO);
        assert_eq!(info.render_origin_generation, 0);
    }

    #[test]
    fn debug_source_check_initially_procedural() {
        let info = TerrainDebugInfo::default();
        assert_eq!(
            info.source_mode,
            er_world::terrain_field::TerrainSourceMode::Procedural
        );
    }

    #[test]
    fn vertex_spacing_consistent_with_cell_size_for_miniature() {
        let lod = 12u8;
        let radius = 36_000.0;
        let cs = cell_size(lod, radius);
        let vs = cs / CHUNK_QUADS_PER_EDGE as f64;
        assert!(vs > 0.0);
        assert!(
            (vs * CHUNK_QUADS_PER_EDGE as f64 - cs).abs() < 1e-10,
            "vertex spacing * quads_per_edge should equal cell_size"
        );
    }

    #[test]
    fn normal_diff_spacing_is_finite_for_miniature_and_earth() {
        for (radius, lod) in [(36_000.0, 12u8), (6_371_000.0, 17u8)] {
            let vs = cell_size(lod, radius) / CHUNK_QUADS_PER_EDGE as f64;
            let nd = (vs / radius).clamp(1e-8, 0.25);
            assert!(nd > 0.0 && nd.is_finite());
        }
    }

    #[test]
    fn gizmo_origin_matches_parameter() {
        let origin_render = RenderOrigin {
            world: glam::DVec3::new(1_000.0, 2_000.0, 3_000.0),
            generation: 42,
            cell_size_m: 1000.0,
        };
        let origin_offset = OriginOffset(origin_render.world);
        assert_eq!(origin_offset.0, glam::DVec3::new(1_000.0, 2_000.0, 3_000.0));
    }
}
