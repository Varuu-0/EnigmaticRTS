//! er_core: shared types, math, seed/RNG (ChaCha8), and tunable config.
//! Planet-creation pass; RTS crates are deferred.

pub mod config;
pub mod math;
pub mod rng;
pub mod seed;

pub fn version() -> &'static str {
    "0"
}
