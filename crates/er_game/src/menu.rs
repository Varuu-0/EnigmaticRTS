//! ESC settings menu: VSync toggle, MSAA selection, quit.

use bevy::{ecs::message::MessageWriter, prelude::*, ui::Val};

use crate::settings::{apply_graphics_settings, save_settings, GraphicsSettings};

#[derive(Resource, Default)]
pub struct MenuOpen(pub bool);

#[derive(Component)]
struct MenuRoot;
#[derive(Component)]
struct VSyncButton;
#[derive(Component)]
struct VSyncStateLabel;
#[derive(Component)]
struct FullscreenButton;
#[derive(Component)]
struct FullscreenStateLabel;
#[derive(Component)]
struct MsaaButton(pub u32);
#[derive(Component)]
struct QuitButton;

const OVERLAY: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);
const PANEL: Color = Color::srgb(0.08, 0.09, 0.12);
const PANEL_BORDER: Color = Color::srgb(0.22, 0.24, 0.30);
const BTN: Color = Color::srgb(0.16, 0.18, 0.24);
const BTN_HOVER: Color = Color::srgb(0.28, 0.30, 0.38);
const BTN_ACTIVE: Color = Color::srgb(0.30, 0.55, 0.85);
const QUIT: Color = Color::srgb(0.50, 0.18, 0.18);
const QUIT_HOVER: Color = Color::srgb(0.65, 0.24, 0.24);
const TXT: Color = Color::srgb(0.92, 0.93, 0.96);
const TXT_DIM: Color = Color::srgb(0.60, 0.62, 0.70);

pub struct SettingsMenuPlugin;

impl Plugin for SettingsMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenuOpen>()
            .add_systems(Startup, build_menu)
            .add_systems(
                Update,
                (
                    toggle_menu,
                    sync_menu_visibility,
                    handle_button_actions,
                    update_vsync_label,
                    update_fullscreen_label,
                    update_button_colors,
                    apply_graphics_settings
                        .run_if(|open: Res<MenuOpen>| !open.0),
                )
                    .chain(),
            );
    }
}

fn text_bundle(text: &str, size: f32, color: Color) -> impl Bundle {
    (
        Text::new(text),
        TextFont {
            font_size: FontSize::Px(size),
            ..default()
        },
        TextColor(color),
    )
}

fn build_menu(mut commands: Commands) {
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(OVERLAY),
            MenuRoot,
            Visibility::Hidden,
        ))
        .with_children(|p| {
            p.spawn((
                Node {
                    width: Val::Px(440.0),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(24.0)),
                    border: UiRect::all(Val::Px(2.0)),
                    row_gap: Val::Px(8.0),
                    ..default()
                },
                BackgroundColor(PANEL),
                BorderColor::all(PANEL_BORDER),
            ))
            .with_children(|panel| {
                panel.spawn((
                    text_bundle("Settings", 30.0, TXT),
                    Node {
                        margin: UiRect::bottom(Val::Px(8.0)),
                        ..default()
                    },
                ));

                panel
                    .spawn((
                        Node {
                            padding: UiRect::all(Val::Px(10.0)),
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        Button,
                        VSyncButton,
                        BackgroundColor(BTN),
                    ))
                    .with_children(|b| {
                        b.spawn((text_bundle("VSync: OFF", 20.0, TXT), VSyncStateLabel));
                    });

                panel
                    .spawn((
                        Node {
                            padding: UiRect::all(Val::Px(10.0)),
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        Button,
                        FullscreenButton,
                        BackgroundColor(BTN),
                    ))
                    .with_children(|b| {
                        b.spawn((text_bundle("Fullscreen: OFF", 20.0, TXT), FullscreenStateLabel));
                    });

                panel.spawn((
                    text_bundle("Anti-Aliasing", 18.0, TXT_DIM),
                    Node {
                        margin: UiRect::top(Val::Px(8.0)),
                        ..default()
                    },
                ));

                panel
                    .spawn(Node {
                        flex_direction: FlexDirection::Row,
                        column_gap: Val::Px(8.0),
                        ..default()
                    })
                    .with_children(|row| {
                        for &(label, n) in &[("Off", 1u32), ("2x", 2), ("4x", 4), ("8x", 8)] {
                            row.spawn((
                                Node {
                                    flex_grow: 1.0,
                                    padding: UiRect::all(Val::Px(10.0)),
                                    justify_content: JustifyContent::Center,
                                    ..default()
                                },
                                Button,
                                MsaaButton(n),
                                BackgroundColor(BTN),
                            ))
                            .with_children(|b| {
                                b.spawn(text_bundle(label, 18.0, TXT));
                            });
                        }
                    });

                panel
                    .spawn((
                        Node {
                            padding: UiRect::all(Val::Px(12.0)),
                            justify_content: JustifyContent::Center,
                            margin: UiRect::top(Val::Px(8.0)),
                            ..default()
                        },
                        Button,
                        QuitButton,
                        BackgroundColor(QUIT),
                    ))
                    .with_children(|b| {
                        b.spawn(text_bundle("Quit", 20.0, TXT));
                    });
            });
        });
}

fn toggle_menu(keys: Res<ButtonInput<KeyCode>>, mut open: ResMut<MenuOpen>) {
    if keys.just_pressed(KeyCode::Escape) {
        open.0 = !open.0;
        info!("Settings menu {}", if open.0 { "opened" } else { "closed" });
    }
}

fn sync_menu_visibility(open: Res<MenuOpen>, mut q: Query<&mut Visibility, With<MenuRoot>>) {
    if !open.is_changed() {
        return;
    }
    for mut v in &mut q {
        *v = if open.0 {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

fn handle_button_actions(
    open: Res<MenuOpen>,
    mut settings: ResMut<GraphicsSettings>,
    vsync: Query<&Interaction, (With<VSyncButton>, Changed<Interaction>)>,
    fullscreen: Query<&Interaction, (With<FullscreenButton>, Changed<Interaction>)>,
    msaa: Query<(&Interaction, &MsaaButton), Changed<Interaction>>,
    quit: Query<&Interaction, (With<QuitButton>, Changed<Interaction>)>,
    mut exit: MessageWriter<AppExit>,
) {
    if !open.0 {
        return;
    }
    for interaction in &vsync {
        if *interaction == Interaction::Pressed {
            settings.vsync = !settings.vsync;
            save_settings(&settings);
            info!(
                "VSync set to {} (restart to apply)",
                if settings.vsync { "ON" } else { "OFF" }
            );
        }
    }
    for interaction in &fullscreen {
        if *interaction == Interaction::Pressed {
            settings.fullscreen = !settings.fullscreen;
            save_settings(&settings);
            info!(
                "Fullscreen set to {} (restart to apply)",
                if settings.fullscreen { "ON" } else { "OFF" }
            );
        }
    }
    for (interaction, mb) in &msaa {
        if *interaction == Interaction::Pressed {
            info!("MSAA set to {}x (live)", mb.0);
            settings.msaa = mb.0;
            save_settings(&settings);
        }
    }
    for interaction in &quit {
        if *interaction == Interaction::Pressed {
            info!("Quit requested");
            exit.write(AppExit::Success);
        }
    }
}

fn update_vsync_label(settings: Res<GraphicsSettings>, mut q: Query<&mut Text, With<VSyncStateLabel>>) {
    if !settings.is_changed() {
        return;
    }
    for mut t in &mut q {
        t.0 = format!("VSync: {}", if settings.vsync { "ON" } else { "OFF" });
    }
}

fn update_fullscreen_label(
    settings: Res<GraphicsSettings>,
    mut q: Query<&mut Text, With<FullscreenStateLabel>>,
) {
    if !settings.is_changed() {
        return;
    }
    for mut t in &mut q {
        t.0 = format!("Fullscreen: {}", if settings.fullscreen { "ON" } else { "OFF" });
    }
}

fn update_button_colors(
    open: Res<MenuOpen>,
    settings: Res<GraphicsSettings>,
    mut normal: Query<
        (&Interaction, &mut BackgroundColor, Option<&MsaaButton>),
        (With<Button>, Without<QuitButton>),
    >,
    mut quit: Query<(&Interaction, &mut BackgroundColor), With<QuitButton>>,
) {
    if !open.0 {
        return;
    }
    for (interaction, mut bg, msaa) in &mut normal {
        let color = if matches!(msaa, Some(mb) if mb.0 == settings.msaa) {
            BTN_ACTIVE
        } else if *interaction == Interaction::Hovered {
            BTN_HOVER
        } else {
            BTN
        };
        if bg.0 != color {
            *bg = BackgroundColor(color);
        }
    }
    for (interaction, mut bg) in &mut quit {
        let color = if *interaction == Interaction::Hovered {
            QUIT_HOVER
        } else {
            QUIT
        };
        if bg.0 != color {
            *bg = BackgroundColor(color);
        }
    }
}
