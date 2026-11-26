$ErrorActionPreference = "Stop"
$Version = if ($env:TRISYNAPSE_MEMORY_VERSION) { $env:TRISYNAPSE_MEMORY_VERSION } else { "0.3.0" }
$Package = "trisynapse-memory[all]==$Version"
$StateRoot = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME ".local\state" }
$MetadataDir = Join-Path $StateRoot "trisynapse-memory"
$Architecture = $env:PROCESSOR_ARCHITECTURE
if ($Architecture -notin @("AMD64", "ARM64", "x86")) {
    throw "Unsupported CPU architecture: $Architecture"
}
Write-Host "Detected Windows / $Architecture."

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
}

$Uv = (Get-Command uv -ErrorAction Stop).Source
Write-Host "Installing or upgrading $Package..."
$Installed = $false
for ($Attempt = 1; $Attempt -le 5; $Attempt++) {
    & $Uv tool install --upgrade $Package
    if ($LASTEXITCODE -eq 0) {
        $Installed = $true
        break
    }
    if ($Attempt -lt 5) {
        Write-Host "Package is not available yet; retrying in 6 seconds..."
        Start-Sleep -Seconds 6
    }
}
if (-not $Installed) {
    throw "Package did not become available after 5 attempts: $Package"
}
New-Item -ItemType Directory -Force -Path $MetadataDir | Out-Null
@(
    "installed_at=$([DateTime]::UtcNow.ToString('o'))"
    "installer=github-release"
    "version=$Version"
    "os=Windows"
    "architecture=$Architecture"
    "uv=$Uv"
) | Set-Content -Encoding UTF8 (Join-Path $MetadataDir "install.env")

$BinDir = (& $Uv tool dir --bin).Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not determine the uv tool bin directory." }
$Command = Join-Path $BinDir "trisynapse-memory.exe"
if (-not (Test-Path -LiteralPath $Command -PathType Leaf)) {
    throw "Installed command was not found at $Command"
}
if ($BinDir -notin ($env:Path -split ";")) {
    Write-Host "Add $BinDir to your user PATH, then open a new terminal."
}
$InstalledVersion = (& $Command --version).Trim()
if ($LASTEXITCODE -ne 0 -or $InstalledVersion -ne $Version) {
    throw "Expected version $Version, got $InstalledVersion"
}
& $Command --json check
if ($LASTEXITCODE -ne 0) { throw "The installed command failed its health check." }
Write-Host "Installed. Run: trisynapse-memory"
