//! Deterministic seed types.
//!
//! `SystemSeed` (an entire solar system) derives per-planet `PlanetSeed`s via
//! ChaCha8, so each body's procgen is stable and independent.

use crate::rng::{rng_child, rng_from_seed};
use rand::RngCore;

/// A raw u64 seed.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Seed(pub u64);

/// Seed for a single planet's procgen (elevation/biome/water/params).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PlanetSeed(pub u64);

/// Seed for an entire solar system (star + planet roster + orbits).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SystemSeed(pub u64);

impl SystemSeed {
    /// Derive a deterministic per-planet seed from the system seed + planet index.
    pub fn planet_seed(self, planet_index: u32) -> PlanetSeed {
        let mut rng = rng_from_seed(self.0);
        let mut child = rng_child(&mut rng, planet_index as u64);
        let mut buf = [0u8; 8];
        child.fill_bytes(&mut buf);
        PlanetSeed(u64::from_le_bytes(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planet_seed_is_repeatable_and_domain_separated() {
        let system = SystemSeed(0xC0FFEE);
        assert_eq!(system.planet_seed(3), system.planet_seed(3));
        assert_ne!(system.planet_seed(0), system.planet_seed(1));
        assert_ne!(system.planet_seed(0), SystemSeed(0xC0FFEF).planet_seed(0));
    }

    #[test]
    fn planet_seed_remains_unique_over_a_deterministic_roster() {
        let system = SystemSeed(0xD15EA5E);
        let seeds: std::collections::HashSet<_> =
            (0..64).map(|index| system.planet_seed(index)).collect();
        assert_eq!(seeds.len(), 64);
    }
}
