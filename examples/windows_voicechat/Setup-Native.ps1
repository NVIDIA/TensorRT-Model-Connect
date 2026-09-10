# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$SourceDirectory,
    [string]$DependencyDirectory,
    [string]$OutputDirectory,
    [int]$Jobs = 8,
    [switch]$UsePortableWindowsSdk,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
if (-not $SourceDirectory) { $SourceDirectory = Join-Path $PSScriptRoot '..\..' }
$SourceDirectory = [IO.Path]::GetFullPath($SourceDirectory)
$workspaceDirectory = Split-Path $SourceDirectory -Parent
if (-not $DependencyDirectory) { $DependencyDirectory = Join-Path $workspaceDirectory 'dependencies' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $workspaceDirectory 'runtime' }
$DependencyDirectory = [IO.Path]::GetFullPath($DependencyDirectory)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if ($Jobs -lt 1) { throw 'Jobs must be at least 1.' }
New-Item -ItemType Directory -Force -Path $DependencyDirectory, $OutputDirectory | Out-Null

function Get-VerifiedArchive([string]$Uri, [string]$Path, [string]$Sha256) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "Downloading $([IO.Path]::GetFileName($Path))"
        & curl.exe --fail --location --retry 3 --silent --show-error $Uri --output "$Path.partial"
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $Uri" }
        if ($Sha256 -and (Get-FileHash -LiteralPath "$Path.partial" -Algorithm SHA256).Hash -ne $Sha256) {
            throw "SHA256 mismatch for $Path.partial"
        }
        Move-Item -LiteralPath "$Path.partial" -Destination $Path -Force
    }
    if ($Sha256 -and (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Sha256) {
        throw "SHA256 mismatch for $Path. Move the invalid file aside and rerun setup."
    }
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

# Use an existing C++ toolchain when available; install Microsoft's signed
# Build Tools distribution if the machine has never been used for C++ builds.
$vsRoot = Join-Path $DependencyDirectory 'VSBuildTools'
$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vcvars) -and (Test-Path -LiteralPath $vswhere)) {
    $existing = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($existing) {
        $vsRoot = $existing.Trim()
        $vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
    }
}
if (-not (Test-Path -LiteralPath $vcvars)) {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'The first setup installs Microsoft C++ Build Tools. Run this script from an Administrator PowerShell window.'
    }
    $bootstrapper = Join-Path $DependencyDirectory 'vs_buildtools.exe'
    Get-VerifiedArchive 'https://aka.ms/vs/17/release/vs_buildtools.exe' $bootstrapper ''
    $signature = Get-AuthenticodeSignature -LiteralPath $bootstrapper
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw 'The Microsoft Build Tools installer signature could not be verified.'
    }
    Write-Host 'Installing Microsoft C++ Build Tools and Windows SDK. This can take several minutes.'
    $installer = Start-Process -FilePath $bootstrapper -ArgumentList @(
        '--quiet', '--wait', '--norestart', '--nocache', '--installPath', ('"' + $vsRoot + '"'),
        '--add', 'Microsoft.VisualStudio.Workload.VCTools', '--includeRecommended'
    ) -WindowStyle Hidden -PassThru -Wait
    if ($installer.ExitCode -notin @(0, 3010)) { throw "Microsoft Build Tools installation failed: $($installer.ExitCode)" }
}

# Microsoft's portable SDK is useful when a C++ compiler is already installed
# and the full Visual Studio Windows SDK installation is still in progress.
if ($UsePortableWindowsSdk) {
    $sdkArchive = Join-Path $DependencyDirectory 'windows-sdk-cpp-10.0.26100.9169.zip'
    $sdkX64Archive = Join-Path $DependencyDirectory 'windows-sdk-cpp-x64-10.0.26100.9169.zip'
    Get-VerifiedArchive 'https://api.nuget.org/v3-flatcontainer/microsoft.windows.sdk.cpp/10.0.26100.9169/microsoft.windows.sdk.cpp.10.0.26100.9169.nupkg' $sdkArchive '475269434dcd808a67853773272f972c3229c0e10c3ddc821290e70cc0f6904d'
    Get-VerifiedArchive 'https://api.nuget.org/v3-flatcontainer/microsoft.windows.sdk.cpp.x64/10.0.26100.9169/microsoft.windows.sdk.cpp.x64.10.0.26100.9169.nupkg' $sdkX64Archive 'df6226a051e320942abfbd57848b43d18772996ecd66beadad240f2a56ed2f7b'
    $sdkContainer = Join-Path $DependencyDirectory 'windows-sdk-portable'
    $sdkX64Container = Join-Path $DependencyDirectory 'windows-sdk-x64-portable'
    if (-not (Test-Path -LiteralPath "$sdkContainer\c\bin\10.0.26100.0\x64\rc.exe")) {
        Expand-Archive -LiteralPath $sdkArchive -DestinationPath $sdkContainer -Force
    }
    if (-not (Test-Path -LiteralPath "$sdkX64Container\c\um\x64\kernel32.Lib")) {
        Expand-Archive -LiteralPath $sdkX64Archive -DestinationPath $sdkX64Container -Force
    }
    $compilerDirectory = Get-ChildItem -LiteralPath (Join-Path $vsRoot 'VC\Tools\MSVC') -Directory |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    $msvcRoot = $compilerDirectory.FullName
    $sdkRoot = Join-Path $sdkContainer 'c'
    $sdkLibRoot = Join-Path $sdkX64Container 'c'
    $sdkInclude = Join-Path $sdkRoot 'Include\10.0.26100.0'
    $env:INCLUDE = "$msvcRoot\include;$sdkInclude\ucrt;$sdkInclude\shared;$sdkInclude\um;$sdkInclude\winrt;$sdkInclude\cppwinrt"
    $env:LIB = "$msvcRoot\lib\x64;$sdkLibRoot\ucrt\x64;$sdkLibRoot\um\x64"
    $env:PATH = "$msvcRoot\bin\Hostx64\x64;$sdkRoot\bin\10.0.26100.0\x64;" + $env:PATH
    $env:VSCMD_ARG_TGT_ARCH = 'x64'
    $env:VCToolsInstallDir = "$msvcRoot\"
    $env:WindowsSdkDir = "$sdkRoot\"
    $env:WindowsSDKVersion = '10.0.26100.0\'
    $env:UniversalCRTSdkDir = "$sdkRoot\"
    $env:UCRTVersion = '10.0.26100.0'
} else {
    # Import process environment without changing machine-wide PATH.
    $compilerEnvironment = & $env:ComSpec /d /s /c "`"`"$vcvars`" >nul && set`""
    if ($LASTEXITCODE -ne 0) { throw 'The Microsoft x64 compiler environment could not be initialized.' }
    foreach ($line in $compilerEnvironment) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

$cmakeCandidates = @(
    (Join-Path $DependencyDirectory 'python\Scripts\cmake.exe'),
    (Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe')
)
$ninjaCandidates = @(
    (Join-Path $DependencyDirectory 'python\Scripts\ninja.exe'),
    (Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe')
)
$cmake = $cmakeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$ninja = $ninjaCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $cmake) { $cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue).Source }
if (-not $ninja) { $ninja = (Get-Command ninja.exe -ErrorAction SilentlyContinue).Source }
if (-not $cmake -or -not $ninja) { throw 'CMake and Ninja are required. Add the C++ CMake component to Microsoft Build Tools.' }
$ctest = Join-Path (Split-Path $cmake -Parent) 'ctest.exe'

$cudaRoot = Join-Path $DependencyDirectory 'cuda-13.4'
$cudaManifestPath = Join-Path $DependencyDirectory 'cuda-redistrib-13.4.1.json'
Get-VerifiedArchive 'https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.1.json' $cudaManifestPath '291178934b2139727407c76f697bb0dfbac4014753a8094d7c290b0f4124a302'
$cudaManifest = Get-Content -LiteralPath $cudaManifestPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $cudaRoot | Out-Null
foreach ($component in @('cuda_cudart', 'cuda_crt', 'cuda_nvcc', 'libnvvm', 'cccl')) {
    $entry = $cudaManifest.$component.'windows-x86_64'
    if (-not $entry) { throw "The NVIDIA manifest is missing $component for Windows x64." }
    $archive = Join-Path $DependencyDirectory ([IO.Path]::GetFileName($entry.relative_path))
    Get-VerifiedArchive ('https://developer.download.nvidia.com/compute/cuda/redist/' + $entry.relative_path) $archive $entry.sha256
    $unpack = Join-Path $DependencyDirectory ('unpack-' + $component)
    Expand-Archive -LiteralPath $archive -DestinationPath $unpack -Force
    $archiveRoot = Get-ChildItem -LiteralPath $unpack -Directory | Select-Object -First 1
    Get-ChildItem -LiteralPath $archiveRoot.FullName | Copy-Item -Destination $cudaRoot -Recurse -Force
}

$rtxArchive = Join-Path $DependencyDirectory 'TensorRT-RTX-1.6.1.120-Windows-amd64-cuda-13.4.zip'
Get-VerifiedArchive 'https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.6/TensorRT-RTX-1.6.1.120-Windows-amd64-cuda-13.4-Release-external.zip' $rtxArchive '32612cd50842d2e4773071dfe8b947522d12a598ec8a01320ccbbda8272963e2'
$rtxContainer = Join-Path $DependencyDirectory 'tensorrt-rtx'
$rtxRoot = Join-Path $rtxContainer 'TensorRT-RTX-1.6.1.120'
if (-not (Test-Path -LiteralPath (Join-Path $rtxRoot 'include\NvInfer.h'))) {
    Expand-Archive -LiteralPath $rtxArchive -DestinationPath $rtxContainer -Force
}

$jsonArchive = Join-Path $DependencyDirectory 'nlohmann-json-3.12.0.zip'
Get-VerifiedArchive 'https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.zip' $jsonArchive '34660b5e9a407195d55e8da705ed26cc6d175ce5a6b1fb957e701fb4d5b04022'
$jsonRoot = Join-Path $DependencyDirectory 'nlohmann-json'
if (-not (Test-Path -LiteralPath (Join-Path $jsonRoot 'json-3.12.0\CMakeLists.txt'))) {
    Expand-Archive -LiteralPath $jsonArchive -DestinationPath $jsonRoot -Force
}
Invoke-Checked $cmake @('-S', "$jsonRoot\json-3.12.0", '-B', "$jsonRoot\build-ninja", '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja", '-DJSON_BuildTests=OFF', "-DCMAKE_INSTALL_PREFIX=$jsonRoot\install")
Invoke-Checked $cmake @('--install', "$jsonRoot\build-ninja")

$buildRoot = Join-Path $SourceDirectory 'build-windows-rtx'
$stageRoot = Join-Path $buildRoot 'install'
$env:CUDA_PATH = $cudaRoot
$env:PATH = "$cudaRoot\bin;$cudaRoot\bin\x64;$rtxRoot\bin;$buildRoot;" + $env:PATH
$tests = if ($SkipTests) { 'OFF' } else { 'ON' }
Invoke-Checked $cmake @(
    '-S', $SourceDirectory, '-B', $buildRoot, '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja",
    '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_CUDA_COMPILER=$cudaRoot\bin\nvcc.exe",
    "-DCUDAToolkit_ROOT=$cudaRoot", "-DCMAKE_PREFIX_PATH=$jsonRoot\install",
    '-DTRTMC_BUILD_BACKEND_TRT=OFF', '-DTRTMC_BUILD_BACKEND_RTX=ON',
    "-DTRTMC_RTX_INCLUDE_DIR=$rtxRoot\include", "-DTRTMC_RTX_LIBRARY_DIR=$rtxRoot\lib",
    '-DTRTMC_ENABLE_BYOK=OFF', '-DTRTMC_FAMILIES=nemotron_voicechat',
    '-DTRTMC_BUILD_CLI=OFF', '-DTRTMC_BUILD_EXAMPLES=OFF', "-DTRTMC_BUILD_TESTS=$tests",
    '-DTRTMC_BUILD_WINDOWS_VOICECHAT=ON'
)
Invoke-Checked $cmake @('--build', $buildRoot, '--parallel', "$Jobs")
if (-not $SkipTests) { Invoke-Checked $ctest @('--test-dir', $buildRoot, '--output-on-failure') }
Invoke-Checked $cmake @('--install', $buildRoot, '--prefix', $stageRoot)
Get-ChildItem -LiteralPath "$stageRoot\bin" -File | Copy-Item -Destination $OutputDirectory -Force
Get-ChildItem -LiteralPath "$cudaRoot\bin" -Filter '*.dll' -Recurse | Copy-Item -Destination $OutputDirectory -Force
Get-ChildItem -LiteralPath "$rtxRoot\bin" -Filter '*.dll' | Copy-Item -Destination $OutputDirectory -Force
$crtRoot = Get-ChildItem -LiteralPath (Join-Path $vsRoot 'VC\Redist\MSVC') -Directory |
    Where-Object { $_.Name -match '^\d' } | Sort-Object Name -Descending | Select-Object -First 1
if ($crtRoot) {
    Get-ChildItem -LiteralPath (Join-Path $crtRoot.FullName 'x64\Microsoft.VC143.CRT') -Filter '*.dll' |
        Copy-Item -Destination $OutputDirectory -Force
}
$licenseRoot = Join-Path $OutputDirectory 'licenses'
New-Item -ItemType Directory -Force -Path $licenseRoot | Out-Null
$cudaRuntimeArchive = Get-ChildItem -LiteralPath (Join-Path $DependencyDirectory 'unpack-cuda_cudart') -Directory | Select-Object -First 1
Copy-Item -LiteralPath (Join-Path $cudaRuntimeArchive.FullName 'LICENSE') -Destination (Join-Path $licenseRoot 'CUDA-runtime-LICENSE.txt') -Force
Copy-Item -LiteralPath (Join-Path $rtxRoot 'doc\Acknowledgements.txt') -Destination (Join-Path $licenseRoot 'TensorRT-RTX-Acknowledgements.txt') -Force
Copy-Item -LiteralPath (Join-Path $rtxRoot 'doc\README.txt') -Destination (Join-Path $licenseRoot 'TensorRT-RTX-README.txt') -Force
Copy-Item -LiteralPath (Join-Path $jsonRoot 'json-3.12.0\LICENSE.MIT') -Destination (Join-Path $licenseRoot 'nlohmann-json-LICENSE.txt') -Force
$rtxLicense = Join-Path $DependencyDirectory 'TensorRT-RTX-license.html'
Get-VerifiedArchive 'https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html' $rtxLicense ''
Copy-Item -LiteralPath $rtxLicense -Destination $licenseRoot -Force
Write-Host "Built TensorRT-RTX VoiceChat runtime: $OutputDirectory\trtmc_voicechat_bridge.exe"
