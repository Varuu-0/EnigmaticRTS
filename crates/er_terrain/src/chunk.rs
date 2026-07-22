use bevy::ecs::component::Component;
use er_core::math::CellKey;

#[derive(Component, Clone, Debug)]
pub struct ChunkComponent {
    pub key: CellKey,
    pub neighbor_depth: [u8; 4],
}

impl ChunkComponent {
    pub fn new(key: CellKey) -> Self {
        Self {
            key,
            neighbor_depth: [key.lod; 4],
        }
    }
}

/// Marker for a freshly-spawned split child whose mesh is still generating.
/// `cull_chunks` skips these (leaves them `Hidden`) so children are revealed
/// atomically — all four at once — by `finalize_retirements` only after every
/// child has a mesh. This avoids both gaps (parent stays visible as the
/// fallback) and z-fighting (parent is despawned the instant children appear).
#[derive(Component, Clone, Debug, Default)]
pub struct HoldHidden;

/// Marker for the four children of a parent that is being merged to a coarser
/// LOD. The children stay rendered as the visible fallback until the new
/// parent's mesh is ready and `finalize_retained_merges` performs the atomic
/// reveal/despawn. This prevents the 1–2 frame black gap of a plain merge.
#[derive(Component, Clone, Debug, Default)]
pub struct HoldForMerge;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LodTransitionRole {
    Incoming,
    Outgoing,
}

/// Short-lived visual handoff between otherwise atomic LOD replacements.
/// The progress is encoded in Bevy's per-instance `MeshTag`, avoiding mesh
/// uploads and per-chunk material instances during the transition.
#[derive(Component, Clone, Copy, Debug)]
pub struct LodTransition {
    pub role: LodTransitionRole,
    pub elapsed_seconds: f32,
}

impl LodTransition {
    pub fn new(role: LodTransitionRole) -> Self {
        Self {
            role,
            elapsed_seconds: 0.0,
        }
    }
}
