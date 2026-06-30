//! Deterministic RNG (ChaCha8) helpers.
//!
//! ChaCha8 is fast, statistically good, and fully reproducible from a u64 seed.
//! All procedural generation derives its randomness through these helpers so a
//! given seed always yields an identical world.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub type Rng = ChaCha8Rng;

/// Create a deterministic RNG from a u64 seed.
pub fn rng_from_seed(seed: u64) -> Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Derive an independent child RNG stream from a parent, mixing in `mix`.
/// The parent advances (consumes 32 bytes) and the child is seeded from the
/// result XORed with `mix`, giving stable, non-overlapping sub-streams.
pub fn rng_child(parent: &mut Rng, mix: u64) -> Rng {
    let mut bytes = [0u8; 32];
    parent.fill_bytes(&mut bytes);
    for (i, b) in bytes.iter_mut().enumerate() {
        *b ^= ((mix >> ((i % 8) * 8)) & 0xff) as u8;
    }
    ChaCha8Rng::from_seed(bytes)
}
