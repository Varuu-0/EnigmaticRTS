param(
    [string] $Model = "xandergos/terrain-diffusion-30m",
    [int] $Port = 8000,
    [ValidateSet("cuda", "cpu")] [string] $Device = "cuda",
    [ValidateSet("fp32", "bf16", "fp16")] [string] $Dtype = "fp32",
    [ValidateSet("info", "verbose")] [string] $LogMode = "verbose",
    [switch] $NoCompile,
    [string] $CacheStrategy = "direct",
    [string] $CacheSize = "100M",
    [string] $BatchSize = "1,4",
    [int] $T = 2,
    [string] $ExtraKwargs = ""
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

$args = @(
    "-m", "terrain_diffusion", "api", $Model,
    "--host", "127.0.0.1",
    "--port", $Port,
    "--device", $Device,
    "--dtype", $Dtype,
    "--log-mode", $LogMode,
    "--caching-strategy", $CacheStrategy,
    "--cache-size", $CacheSize,
    "--batch-size", $BatchSize,
    "--kwarg", "T=$T"
)
if ($NoCompile) {
    $args += "--no-compile"
} else {
    $args += "--compile"
}
if ($ExtraKwargs) {
    foreach ($pair in $ExtraKwargs -split ",") {
        $trimmed = $pair.Trim()
        if ($trimmed) {
            $args += @("--kwarg", $trimmed)
        }
    }
}

Write-Host "Starting Terrain Diffusion: dtype=$Dtype device=$Device T=$T compile=$(-not $NoCompile)"
Push-Location -LiteralPath $repository
try {
    & $python @args
} finally {
    Pop-Location
}
