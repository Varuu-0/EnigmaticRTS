param(
    [Parameter(Mandatory = $true)][string]$Profile,
    [Parameter(Mandatory = $true)][string]$ActualDir
)

$goldenDir = Join-Path $PSScriptRoot "../screenshots_retained/$Profile/goldens"
if (-not (Test-Path -LiteralPath $goldenDir)) {
    throw "Golden directory does not exist: $goldenDir"
}
if (-not (Test-Path -LiteralPath $ActualDir)) {
    throw "Evidence output does not exist: $ActualDir"
}

$failures = @()
Get-ChildItem -LiteralPath $goldenDir -Filter '*.png' -File | ForEach-Object {
    $actual = Join-Path $ActualDir $_.Name
    if (-not (Test-Path -LiteralPath $actual)) {
        $failures += "Missing capture: $($_.Name)"
        return
    }
    $expectedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $actual).Hash
    if ($expectedHash -ne $actualHash) {
        $failures += "Pixel mismatch: $($_.Name)"
    }
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}
Write-Host "All goldens match exactly for profile $Profile."
