use bevy::prelude::*;
use er_core::config::MAX_QUADTREE_DEPTH;
use er_core::math::{cells_per_edge, dir_to_surface, uv_to_dir, world_to_render, OriginOffset};
use er_terrain::{ChunkComponent, TerrainDebugInfo, TerrainState};

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
                Update,
                (update_debug_text, toggle_lod_debug, draw_lod_gizmos),
            );
    }
}

fn setup_debug_text(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(16.0),
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
    mut query: Query<&mut Text, With<DebugText>>,
) {
    if let Ok(mut text) = query.single_mut() {
        *text = Text::new(format!(
            "Chunks: {} | Max LOD: {} | Splits: {} | Merges: {}",
            debug.active_chunks, debug.max_depth, debug.pending_splits, debug.pending_merges
        ));
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
        let pts = [to_render(c0), to_render(c1), to_render(c2), to_render(c3), to_render(c0)];

        let hue = (key.lod as f32 / MAX_QUADTREE_DEPTH as f32) * 360.0;
        let color = Color::hsl(hue, 0.9, 0.5);
        gizmos.linestrip(pts, color);
    }
}
