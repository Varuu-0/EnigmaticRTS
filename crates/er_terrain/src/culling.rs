use er_core::math::{cell_size, cell_to_dir, CellKey};
use glam::{DVec3, Vec3};

pub fn chunk_half_angle(key: CellKey, planet_radius: f64) -> f64 {
    let size = cell_size(key.lod, planet_radius);
    (size / planet_radius).min(1.0).asin() * 0.5
}

pub fn is_below_horizon(
    key: CellKey,
    camera_pos: DVec3,
    planet_radius: f64,
) -> bool {
    let d = camera_pos.length();
    if d < 1.0 {
        return false;
    }
    let cam_dir = camera_pos / d;
    let chunk_dir = cell_to_dir(key);
    let dot = chunk_dir.dot(cam_dir);
    let half_angle = chunk_half_angle(key, planet_radius);

    let horizon_cos = if d > planet_radius * 1.5 {
        let angle = (planet_radius / d).clamp(0.0, 1.0).acos() + half_angle;
        angle.cos()
    } else {
        (std::f64::consts::FRAC_PI_2 + half_angle).cos()
    };

    dot < horizon_cos
}

pub fn is_outside_render_distance(
    key: CellKey,
    camera_pos: DVec3,
    planet_radius: f64,
    max_distance: f64,
) -> bool {
    let chunk_center = cell_to_dir(key) * planet_radius;
    (chunk_center - camera_pos).length() > max_distance
}

pub fn frustum_cull_sphere(
    sphere_center: Vec3,
    sphere_radius: f32,
    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_right: Vec3,
    camera_up: Vec3,
    fov_cos: f32,
    aspect: f32,
) -> bool {
    let to_center = sphere_center - camera_pos;
    let dist = to_center.length();
    if dist < sphere_radius {
        return false;
    }
    let dir = to_center / dist;

    let forward_dot = dir.dot(camera_forward);
    if forward_dot < 0.0 && dist > sphere_radius {
        return true;
    }

    let effective_angle = (sphere_radius / dist).atan();
    let cos_eff = (fov_cos.acos() + effective_angle).cos();

    if forward_dot < cos_eff {
        return true;
    }

    let horiz_dot = dir.dot(camera_right).abs();
    let vert_dot = dir.dot(camera_up).abs();
    let horiz_limit = (aspect * cos_eff.acos().tan()).atan2(1.0).cos();
    let vert_limit = cos_eff;

    if horiz_dot > horiz_limit + (sphere_radius / dist).min(1.0) {
        return true;
    }
    if vert_dot > vert_limit + (sphere_radius / dist).min(1.0) {
        return true;
    }

    false
}
