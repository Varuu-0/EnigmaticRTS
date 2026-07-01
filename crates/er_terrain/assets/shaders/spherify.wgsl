// Spherified-cube projection (WGSL). Single source of truth for the GPU side;
// kept in parity with er_core::math (FACE_CORNER/FACE_U/FACE_V/uv_to_dir) by
// tests/shader_parity.rs. No Bevy bindings or #import directives here so it can
// be compiled raw by the parity test.

const FACE_CORNER: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(1.0, -1.0, -1.0),
    vec3<f32>(-1.0, -1.0, -1.0),
    vec3<f32>(-1.0, 1.0, -1.0),
    vec3<f32>(-1.0, -1.0, -1.0),
    vec3<f32>(-1.0, -1.0, 1.0),
    vec3<f32>(-1.0, -1.0, -1.0),
);
const FACE_U: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 2.0, 0.0),
    vec3<f32>(0.0, 2.0, 0.0),
    vec3<f32>(2.0, 0.0, 0.0),
    vec3<f32>(2.0, 0.0, 0.0),
    vec3<f32>(2.0, 0.0, 0.0),
    vec3<f32>(2.0, 0.0, 0.0),
);
const FACE_V: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 0.0, 2.0),
    vec3<f32>(0.0, 0.0, 2.0),
    vec3<f32>(0.0, 0.0, 2.0),
    vec3<f32>(0.0, 0.0, 2.0),
    vec3<f32>(0.0, 2.0, 0.0),
    vec3<f32>(0.0, 2.0, 0.0),
);

fn uv_to_dir(face: i32, u: f32, v: f32) -> vec3<f32> {
    let f = u32(face);
    return normalize(FACE_CORNER[f] + FACE_U[f] * u + FACE_V[f] * v);
}
