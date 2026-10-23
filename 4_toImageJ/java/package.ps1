# One-click build: bundle a .npy LUT into Seg-LUT.jar for Fiji (ImageJ 1.x).
# Usage:
#   .\package.ps1
#   .\package.ps1 -LutPath "..\..\3_Test_using_LUT\Seg2_LUT_HV_2_4.npy"
#   .\package.ps1 -LutPath "D:\my_new_lut.npy" -SamplingInterval 4

param(
    [string]$LutPath = (Join-Path $PSScriptRoot "..\..\3_Test_using_LUT\Seg2_LUT_HV_2_4.npy"),
    [int]$SamplingInterval = 4
)

$ErrorActionPreference = "Stop"

# Maven needs a JDK (javac), not only a JRE. Fix common mis-set JAVA_HOME (...\bin).
if ($env:JAVA_HOME -and $env:JAVA_HOME -match '\\bin$') {
    $env:JAVA_HOME = Split-Path $env:JAVA_HOME -Parent
}
if (-not $env:JAVA_HOME) {
    $candidates = @(
        "C:\Program Files\Java\jdk-21",
        "C:\Program Files\Java\jdk-17",
        "C:\Program Files\Eclipse Adoptium\jdk-21*"
    )
    foreach ($c in $candidates) {
        $resolved = Get-Item $c -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($resolved -and (Test-Path (Join-Path $resolved.FullName "bin\javac.exe"))) {
            $env:JAVA_HOME = $resolved.FullName
            $env:Path = "$env:JAVA_HOME\bin;" + $env:Path
            break
        }
    }
}
if (-not (Get-Command javac -ErrorAction SilentlyContinue)) {
    throw "JDK not found (javac missing). Install JDK 8+ and set JAVA_HOME to the JDK root (not ...\bin)."
}

$LutPath = (Resolve-Path $LutPath).Path

$resourceDir = Join-Path $PSScriptRoot "src\main\resources\lut"
$destNpy = Join-Path $resourceDir "Seg2_LUT_HV_2_4.npy"
New-Item -ItemType Directory -Force -Path $resourceDir | Out-Null
Copy-Item -Force $LutPath $destNpy
Write-Host "Bundled LUT:" $destNpy

Push-Location $PSScriptRoot
try {
    $mvnCmd = $null
    if (Get-Command mvn -ErrorAction SilentlyContinue) {
        $mvnCmd = "mvn"
    } elseif (Test-Path (Join-Path $PSScriptRoot "mvnw.cmd")) {
        $mvnCmd = Join-Path $PSScriptRoot "mvnw.cmd"
    } else {
        throw "Maven not found. Install Maven or use included mvnw.cmd (requires JDK)."
    }
    if ($PSScriptRoot -match '^\\\\') {
        $mvnLine = if ($mvnCmd -eq "mvn") { "mvn -q clean package" } else { "mvnw.cmd -q clean package" }
        cmd /c "pushd `"$PSScriptRoot`" && $mvnLine && popd"
    } else {
        & $mvnCmd -q clean package
    }
    if ($LASTEXITCODE -ne 0) { throw "Maven build failed." }

    $jar = Join-Path $PSScriptRoot "target\Seg-LUT.jar"
    $dist = Join-Path $PSScriptRoot "..\dist"
    New-Item -ItemType Directory -Force -Path $dist | Out-Null
    Copy-Item -Force $jar (Join-Path $dist "Seg-LUT.jar")
    Write-Host ""
    Write-Host "Built:" (Join-Path $dist "Seg-LUT.jar")
    Write-Host "Install: copy Seg-LUT.jar to Fiji.app\plugins\ (or jars\) and restart Fiji."
    Write-Host "Menu: Plugins -> Segmentation -> SEG-LUT"
    if ($PSScriptRoot -match '^\\\\') {
        Write-Host "Note: project is on a network path; if build fails, map a drive letter or clone to a local disk."
    }
} finally {
    Pop-Location
}
