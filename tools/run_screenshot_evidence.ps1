param(
    [string]$Profile = "rtx3060_optimus",
    [string]$OutputDir = "screenshots/evidence",
    [switch]$TerrainDiffusion
)

$profilePath = Join-Path $PSScriptRoot "../hardware_profiles/$Profile.toml"
if (-not (Test-Path -LiteralPath $profilePath)) {
    throw "Unknown hardware profile: $profilePath"
}

$seedLine = Select-String -LiteralPath $profilePath -Pattern '^seed\s*=\s*"(.+)"' | Select-Object -First 1
$seed = if ($seedLine) { $seedLine.Matches[0].Groups[1].Value } else { "0xC0FFEE" }
$args = @(
    "run", "-p", "er_game", "--",
    "--earth-scale",
    "--screenshot-test", $OutputDir,
    "--fixed-seed-coverage",
    "--seed", $seed,
    "--sim-time", "0",
    "--gpu-diagnostics"
)
if ($TerrainDiffusion) {
    $args += "--terrain-diffusion"
    $args = @("run", "-p", "er_game", "--features", "terrain_diffusion", "--") + $args[4..($args.Length - 1)]
}

& cargo @args
exit $LASTEXITCODE
