# Golden Screenshot Policy

Only reviewed, representative PNGs belong here. Keep full runs, telemetry JSON, and ad-hoc captures under ignored `screenshots/`.

Goldens are hardware-profile-specific because drivers, adapters, and presentation paths can affect pixels. Store them as `screenshots_retained/<profile>/goldens/<scenario>.png` and retain the matching `baseline_manifest.json` plus `*.telemetry.json` in a reviewed evidence record when promoting a baseline.

Promotion is manual by design. Run `tools/run_screenshot_evidence.ps1`, inspect the output and manifest, then copy only selected scenarios to the profile's `goldens` directory. `tools/compare_goldens.ps1` uses exact hashes and is intended only for identical profile/build conditions; changed rendering intentionally requires review and explicit golden promotion.
