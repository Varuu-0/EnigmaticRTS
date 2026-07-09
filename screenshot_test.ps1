# Screenshot Test Script for EnigmaticRTS
# Usage: .\screenshot_test.ps1 [-OutputDir <path>] [-Preset <name>] [-Frames <count>]
#
# Note: -Frames is the minimum number of frames to wait after each camera move.
# The harness actually waits until the terrain LOD system reports no pending
# chunk meshes for several consecutive frames, so each scenario may take longer
# than the minimum. This prevents the black boxes caused by taking a screenshot
# while chunk meshes are still generating asynchronously.

param(
    [string]$OutputDir = "screenshots",
    [string]$Preset = "default",
    [int]$Frames = 5
)

$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# Define presets
$presets = @{
    "default" = @(
        "orbit_far_0deg,0.0,0.3,150000.0"
        "orbit_far_90deg,1.5708,0.3,150000.0"
        "orbit_far_180deg,3.14159,0.3,150000.0"
        "orbit_far_270deg,4.7124,0.3,150000.0"
        "orbit_mid_0deg,0.0,0.3,70000.0"
        "orbit_mid_90deg,1.5708,0.3,70000.0"
        "orbit_close_0deg,0.0,0.3,45000.0"
        "orbit_close_45deg,0.7854,0.3,45000.0"
        "orbit_very_close_0deg,0.0,0.3,38000.0"
        "top_down,0.0,1.5,100000.0"
        "equator_view,0.0,0.0,60000.0"
    )
    "angles" = @(
        "angle_0,0.0,0.3,90000.0"
        "angle_45,0.7854,0.3,90000.0"
        "angle_90,1.5708,0.3,90000.0"
        "angle_135,2.3562,0.3,90000.0"
        "angle_180,3.14159,0.3,90000.0"
        "angle_225,3.9270,0.3,90000.0"
        "angle_270,4.7124,0.3,90000.0"
        "angle_315,5.4978,0.3,90000.0"
    )
    "zoom_levels" = @(
        "zoom_very_far,0.0,0.3,200000.0"
        "zoom_far,0.0,0.3,150000.0"
        "zoom_medium,0.0,0.3,90000.0"
        "zoom_close,0.0,0.3,50000.0"
        "zoom_very_close,0.0,0.3,40000.0"
        "zoom_surface,0.0,0.3,37000.0"
    )
    "lod_test" = @(
        "lod_far,0.0,0.3,120000.0"
        "lod_mid,0.0,0.3,60000.0"
        "lod_close,0.0,0.3,42000.0"
        "lod_very_close,0.0,0.3,38500.0"
    )
    "quick" = @(
        "quick_1,0.0,0.3,90000.0"
        "quick_2,1.5708,0.3,60000.0"
        "quick_3,3.14159,0.3,45000.0"
    )
}

if (-not $presets.ContainsKey($Preset)) {
    Write-Host "Available presets: $($presets.Keys -join ', ')"
    Write-Host "Custom preset not found, using 'default'"
    $Preset = "default"
}

$scenarios = $presets[$Preset]

Write-Host "=== Screenshot Test ===" -ForegroundColor Cyan
Write-Host "Output directory: $OutputDir"
Write-Host "Preset: $Preset"
Write-Host "Frames per scenario: $Frames"
Write-Host "Scenarios: $($scenarios.Count)"
Write-Host ""

# Build command arguments
$args = @("run", "-p", "er_game", "--")
$args += "--screenshot-test"
$args += $OutputDir
$args += "--frames"
$args += $Frames.ToString()

foreach ($scenario in $scenarios) {
    $args += "--scenario"
    $args += $scenario
}

Write-Host "Running: cargo $($args -join ' ')" -ForegroundColor Yellow
Write-Host ""

# Run the test
cargo @args

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Screenshots saved to: $OutputDir ===" -ForegroundColor Green
    Get-ChildItem -Path $OutputDir -Filter "*.png" | ForEach-Object {
        Write-Host "  - $($_.Name)"
    }
} else {
    Write-Host ""
    Write-Host "=== Test failed ===" -ForegroundColor Red
}
