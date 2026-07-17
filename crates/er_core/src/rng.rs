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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(mut rng: Rng) -> [u64; 4] {
        [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ]
    }

    #[test]
    fn seed_replays_an_identical_stream() {
        assert_eq!(
            sample(rng_from_seed(0xC0FFEE)),
            sample(rng_from_seed(0xC0FFEE))
        );
        assert_ne!(sample(rng_from_seed(1)), sample(rng_from_seed(2)));
    }

    #[test]
    fn child_stream_replays_and_consumes_exactly_its_seed_material() {
        let mut first_parent = rng_from_seed(0xBAD5EED);
        let mut second_parent = rng_from_seed(0xBAD5EED);

        let first_child = rng_child(&mut first_parent, 17);
        let second_child = rng_child(&mut second_parent, 17);
        assert_eq!(sample(first_child), sample(second_child));
        assert_eq!(sample(first_parent), sample(second_parent));
    }

    #[test]
    fn child_mix_selects_an_independent_stream() {
        let mut first_parent = rng_from_seed(42);
        let mut second_parent = rng_from_seed(42);

        assert_ne!(
            sample(rng_child(&mut first_parent, 1)),
            sample(rng_child(&mut second_parent, 2)),
        );
    }
}
