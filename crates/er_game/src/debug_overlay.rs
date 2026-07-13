use bevy::prelude::*;
use er_core::config::{DEFAULT_DAY_LENGTH_SEC, MAX_QUADTREE_DEPTH};
use er_core::math::{cells_per_edge, dir_to_surface, uv_to_dir, world_to_render, OriginOffset};
use er_terrain::{ChunkComponent, FrameProfiler, TerrainDebugInfo, TerrainState};

use crate::chunk_cap_controller::DynamicChunkCapController;
use crate::diagnostics::PerformanceSnapshot;
use crate::space::{SimTime, TimeScale};
use er_terrain::SunDirection;

#[derive(Component)]
struct DebugText;

#[derive(Resource, Default)]
pub struct LodDebugDraw(pub bool);

pub struct DebugOverlayPlugin;

impl Plugin for DebugOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(LodDebugDraw::default())
            .add_systems(Startup, setup_debug_text)
            .add_systems(PostUpdate, update_debug_text)
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

fn update_debug_text(
    debug: Res<TerrainDebugInfo>,
    profiler: Res<FrameProfiler>,
    time: Res<Time>,
    sun_direction: Res<SunDirection>,
    sim_time: Res<SimTime>,
    time_scale: Res<TimeScale>,
    performance: Res<PerformanceSnapshot>,
    chunk_cap: Res<DynamicChunkCapController>,
    mut query: Query<&mut Text, With<DebugText>>,
) {
    if let Ok(mut text) = query.single_mut() {
        let fps = 1.0 / time.delta_secs().max(0.0001);
        let frame_ms = time.delta_secs() * 1000.0;

        let mut sorted: Vec<(&'static str, std::time::Duration)> = profiler.timings.clone();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let total_profiled: std::time::Duration = sorted.iter().map(|(_, d)| *d).sum();

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
            fps,
            frame_ms,
            performance.frame_p95_ms,
            performance.frame_p99_ms,
            performance.one_percent_low_fps,
        ));
        lines.push_str(&format!(
            "Chunks: {}/{} | LOD: {} | Cap p95: {:.1}ms | S/M: {}/{} | Terrain mesh: {:.1} MiB | Built: {} | Draw work: {}\n",
            debug.active_chunks,
            chunk_cap.current_cap,
            debug.max_depth,
            chunk_cap.last_p95_ms,
            debug.pending_splits,
            debug.pending_merges,
            debug.estimated_mesh_bytes as f64 / (1024.0 * 1024.0),
            debug.meshes_built,
            performance.visible_mesh_draw_estimate,
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
            "Profiled: {:.2}ms / {:.1}ms\n",
            total_profiled.as_secs_f32() * 1000.0,
            frame_ms
        ));

        let max_ms = sorted
            .first()
            .map(|(_, d)| d.as_secs_f32() * 1000.0)
            .unwrap_or(1.0)
            .max(0.1);

        for (name, duration) in &sorted {
            let ms = duration.as_secs_f32() * 1000.0;
            let bar_len = ((ms / max_ms) * 30.0) as usize;
            let bar = "#".repeat(bar_len);
            lines.push_str(&format!("  {:<16} {:>6.2}ms {}\n", name, ms, bar));
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
    chunks: Query<&ChunkComponent>,
    mut gizmos: Gizmos,
) {
    if !draw.0 {
        return;
    }

    let radius = terrain_state.planet_radius;
    let origin = OriginOffset::default();

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
