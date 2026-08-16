$ErrorActionPreference = "Stop"
$Version = if ($env:TRISYNAPSE_MEMORY_VERSION) { $env:TRISYNAPSE_MEMORY_VERSION } else { "0.1.0" }
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
& $Uv tool install --upgrade $Package
New-Item -ItemType Directory -Force -Path $MetadataDir | Out-Null
@(
    "installed_at=$([DateTime]::UtcNow.ToString('o'))"
    "installer=github-release"
    "version=$Version"
    "os=Windows"
    "architecture=$Architecture"
    "uv=$Uv"
) | Set-Content -Encoding UTF8 (Join-Path $MetadataDir "install.env")

$Command = Get-Command trisynapse-memory -ErrorAction SilentlyContinue
if (-not $Command) {
    $BinDir = (& $Uv tool dir --bin).Trim()
    $env:Path = "$BinDir;$env:Path"
    $Command = Get-Command trisynapse-memory -ErrorAction Stop
    Write-Host "Add $BinDir to your user PATH, then open a new terminal."
}
& $Command.Source --json check
Write-Host "Installed. Run: trisynapse-memory"
