# Hardware Evidence Profiles

Each TOML fixture defines a repeatable screenshot and benchmark target. The runtime manifest records the actual adapter, present mode, terrain configuration, build identity, and Terrain Diffusion metadata; the profile records the intended target.

`rtx3060_optimus.toml` is the current primary profile. It intentionally uses `AutoNoVsync`; do not substitute Immediate on Optimus systems.

`desktop_reference.toml` is a placeholder until a specific desktop GPU is selected. Update its adapter identifiers before promoting desktop goldens.

Use `tools/run_screenshot_evidence.ps1` to generate a fixed-seed evidence run. Promote reviewed PNGs manually to `screenshots_retained/<profile>/goldens/`, then use `tools/compare_goldens.ps1` to compare a later run byte-for-byte on the same hardware and software baseline.
