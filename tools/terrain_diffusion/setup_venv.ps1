param(
    [string] $PythonVersion = "3.11"
)

$toolsRoot = $PSScriptRoot
$repository = Join-Path $toolsRoot "upstream"
$venv = Join-Path $toolsRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$globalData = Join-Path $repository "data\global"
$worldClimZip = Join-Path $globalData "wc2.1_10m_bio.zip"
$worldClimUrl = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
$worldClimFiles = @(1, 4, 12, 15 | ForEach-Object {
    Join-Path $globalData "wc2.1_10m_bio_$_.tif"
})
$etopoNc = Join-Path $globalData "earth-topography-10arcmin.nc"
$etopoTiff = Join-Path $globalData "etopo_10m.tif"
$etopoUrl = "https://github.com/fatiando-data/earth-topography-10arcmin/releases/download/v1/earth-topography-10arcmin.nc"

if (-not (Test-Path -LiteralPath $repository)) {
    git clone https://github.com/xandergos/terrain-diffusion.git $repository
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone Terrain Diffusion."
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    & py "-$PythonVersion" -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Python $PythonVersion virtual environment."
    }
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

# Install an NVIDIA CUDA wheel before project requirements so inference uses
# the RTX GPU instead of silently falling back to CPU-only PyTorch.
& $python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install CUDA PyTorch."
}

& $python -m pip install -r (Join-Path $repository "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Terrain Diffusion requirements."
}

New-Item -ItemType Directory -Force -Path $globalData | Out-Null
if (@($worldClimFiles | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -gt 0) {
    Invoke-WebRequest -Uri $worldClimUrl -OutFile $worldClimZip
    Expand-Archive -LiteralPath $worldClimZip -DestinationPath $globalData -Force
    Remove-Item -LiteralPath $worldClimZip -Force
}

if (-not (Test-Path -LiteralPath $etopoTiff)) {
    if (-not (Test-Path -LiteralPath $etopoNc)) {
        Invoke-WebRequest -Uri $etopoUrl -OutFile $etopoNc
    }
    & $python -c "from rasterio.shutil import copy; import sys; copy(sys.argv[1], sys.argv[2], driver='GTiff', compress='deflate')" $etopoNc $etopoTiff
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to convert ETOPO data to GeoTIFF."
    }
    Remove-Item -LiteralPath $etopoNc -Force
}

Push-Location -LiteralPath $repository
try {
    # The upstream project is a source-tree package, so validate it from its
    # repository root rather than requiring packaging metadata it does not ship.
    & $python -c "import terrain_diffusion; print('Terrain Diffusion environment is ready')"
    if ($LASTEXITCODE -ne 0) {
        throw "Terrain Diffusion import check failed."
    }
} finally {
    Pop-Location
}
