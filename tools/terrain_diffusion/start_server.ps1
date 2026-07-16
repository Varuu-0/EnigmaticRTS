param(
    [string] $Model = "xandergos/terrain-diffusion-30m",
    [int] $Port = 8000,
    [ValidateSet("cuda", "cpu")] [string] $Device = "cuda"
)

$toolsRoot = $PSScriptRoot
$repository = Join-Path $toolsRoot "upstream"
$python = Join-Path $toolsRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Terrain Diffusion venv is missing. Run .\tools\terrain_diffusion\setup_venv.ps1 first."
}
if (-not (Test-Path -LiteralPath $repository)) {
    throw "Terrain Diffusion source is missing. Run .\tools\terrain_diffusion\setup_venv.ps1 first."
}

$env:TERRAIN_DEVICE = $Device
# Keep this invocation aligned with Terrain Diffusion's public CLI. The
# sidecar owns model download/cache initialization before it starts listening.
Push-Location -LiteralPath $repository
try {
    & $python -m terrain_diffusion api $Model --host 127.0.0.1 --port $Port
} finally {
    Pop-Location
}
