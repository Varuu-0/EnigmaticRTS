//! Library target for er_game, exposing optional subsystems not yet wired into
//! the binary entry point.
//!
//! The binary (`main.rs`) remains the primary build target. This crate root
//! exists so standalone modules can be compiled and tested independently.

pub mod gpu_telemetry;
