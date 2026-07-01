use bevy::prelude::*;
use er_terrain::TerrainDebugInfo;

#[derive(Component)]
struct DebugText;

pub struct DebugOverlayPlugin;

impl Plugin for DebugOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_debug_text)
            .add_systems(Update, update_debug_text);
    }
}

fn setup_debug_text(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 16.0,
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
