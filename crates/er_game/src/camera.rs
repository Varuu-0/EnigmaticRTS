use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

#[derive(Component)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub min_distance: f32,
    pub max_distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.3,
            distance: 30000.0,
            target: Vec3::ZERO,
            min_distance: 12050.0,
            max_distance: 200000.0,
        }
    }
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (orbit_camera_input, orbit_camera_update).chain());
    }
}

fn orbit_camera_update(mut query: Query<(&OrbitCamera, &mut Transform)>) {
    for (orbit, mut transform) in &mut query {
        let cp = orbit.pitch.cos();
        let direction = Vec3::new(
            cp * orbit.yaw.sin(),
            orbit.pitch.sin(),
            cp * orbit.yaw.cos(),
        )
        .normalize();

        transform.translation = orbit.target + direction * orbit.distance;
        transform.look_at(orbit.target, Vec3::Y);
    }
}

fn orbit_camera_input(
    mut motion_events: EventReader<MouseMotion>,
    mut wheel_events: EventReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut OrbitCamera>,
    time: Res<Time>,
) {
    let mut orbit = match query.single_mut() {
        Ok(o) => o,
        Err(_) => return,
    };

    let dt = time.delta_secs();
    let orbit_speed = 2.5;
    let zoom_speed = 20000.0;

    if keys.pressed(KeyCode::KeyA) {
        orbit.yaw += orbit_speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        orbit.yaw -= orbit_speed * dt;
    }
    if keys.pressed(KeyCode::KeyW) {
        orbit.pitch = (orbit.pitch + orbit_speed * dt).min(1.5);
    }
    if keys.pressed(KeyCode::KeyS) {
        orbit.pitch = (orbit.pitch - orbit_speed * dt).max(-1.5);
    }

    if buttons.pressed(MouseButton::Left) {
        for event in motion_events.read() {
            orbit.yaw -= event.delta.x * 0.005;
            orbit.pitch = (orbit.pitch + event.delta.y * 0.005).clamp(-1.5, 1.5);
        }
    }

    for event in wheel_events.read() {
        orbit.distance *= 1.0 - event.y * 0.1;
    }

    if keys.pressed(KeyCode::ArrowUp) {
        orbit.distance -= zoom_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        orbit.distance += zoom_speed * dt;
    }

    orbit.distance = orbit
        .distance
        .clamp(orbit.min_distance, orbit.max_distance);
}
