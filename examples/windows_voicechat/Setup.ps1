# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#Requires -Version 5.1

<#
.SYNOPSIS
Downloads dependencies and builds the Windows TensorRT-RTX voice example locally.
.DESCRIPTION
No third-party software or model weights are included in this source example.
Downloads, the virtual environment, runtime and app live under WorkspaceRoot,
outside the source checkout. The native CMake build directory is ignored by Git.
Run -Plan to see the stages without downloading, installing or building anything.
.EXAMPLE
.\Setup.ps1 -Plan
.EXAMPLE
.\Setup.ps1 -WorkspaceRoot D:\VoiceChat
.EXAMPLE
.\Setup.ps1 -Stage App
#>
[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [ValidateSet('Python', 'Dependencies', 'Native', 'Model', 'App')]
    [string[]]$Stage = @('Python', 'Dependencies', 'Native', 'Model', 'App'),
    [string]$Python,
    [ValidateRange(1, 128)][int]$Jobs = 8,
    [switch]$UsePortableWindowsSdk,
    [switch]$SkipTests,
    [switch]$RebuildBundle,
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (-not $WorkspaceRoot) { $WorkspaceRoot = Split-Path $sourceRoot -Parent }
$WorkspaceRoot = [IO.Path]::GetFullPath($WorkspaceRoot)
$sourcePrefix = $sourceRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if ($WorkspaceRoot.Equals($sourceRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $WorkspaceRoot.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'WorkspaceRoot must be outside the source checkout so downloaded dependencies and models cannot enter the example.'
}
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitProcess) {
    throw 'Use a 64-bit Windows PowerShell process on Windows x64.'
}

$dependencyRoot = Join-Path $WorkspaceRoot 'dependencies'
$venvPython = Join-Path $dependencyRoot 'python\Scripts\python.exe'
$modelPath = Join-Path $WorkspaceRoot 'models\Nemotron-VoiceChat-11B'
$bundlePath = Join-Path $WorkspaceRoot 'models\nemotron-voicechat-rtx.bundle'
$orderedStages = @('Python', 'Dependencies', 'Native', 'Model', 'App') | Where-Object { $_ -in $Stage }

Write-Host "Source: $sourceRoot"
Write-Host "Local downloads and outputs: $WorkspaceRoot"
Write-Host ('Stages: ' + ($orderedStages -join ', '))
if ('Model' -in $orderedStages) {
    Write-Host 'The first full build downloads the 44.4 GB checkpoint and creates an approximately 18 GB bundle.'
    Write-Host 'Allow at least 120 GB free disk space for downloads/builds and substantial system RAM (128 GB was tested).'
}
if ('Native' -in $orderedStages) {
    Write-Host 'The native stage needs Microsoft C++ Build Tools; installation on a new PC requires Administrator PowerShell.'
}
if ($Plan) {
    $descriptions = @{
        Python = 'Download checksum-verified CPython 3.12.14 from Astral and create a local virtual environment (or reuse -Python).'
        Dependencies = 'Install the pinned Python requirements from PyPI into the local virtual environment.'
        Native = 'Download CUDA, TensorRT-RTX, nlohmann-json and optional SDK; build and test the native RTX bridge.'
        Model = 'Download the pinned Hugging Face checkpoint/tokenizers and build a TensorRT-RTX W8A8 bundle; reuse an existing bundle unless -RebuildBundle.'
        App = 'Download checksum-verified Electron and assemble the app locally with a relative launcher.'
    }
    foreach ($step in $orderedStages) { Write-Host ("{0}: {1}" -f $step, $descriptions[$step]) }
    Write-Host 'Plan only: no files were created and no commands were run.'
    return
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

function Assert-Python([string]$Executable) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Python is missing: $Executable. Include the Python stage or pass -Python with a CPython 3.12 x64 executable."
    }
    Invoke-Checked $Executable @('-c', "import struct, sys; assert sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8, 'CPython 3.12 x64 is required'")
}

New-Item -ItemType Directory -Force -Path $WorkspaceRoot, $dependencyRoot | Out-Null
if ('Python' -in $orderedStages) {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if (-not $Python) {
            $pythonRoot = Join-Path $dependencyRoot 'cpython-3.12.14'
            $Python = Join-Path $pythonRoot 'python\python.exe'
            if (-not (Test-Path -LiteralPath $Python)) {
                if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) {
                    throw 'Windows tar.exe is required to unpack Python. Install current Windows updates or pass -Python.'
                }
                $archive = Join-Path $dependencyRoot 'cpython-3.12.14-20260901-windows-x64.tar.gz'
                $uri = 'https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%2B20260901-x86_64-pc-windows-msvc-install_only.tar.gz'
                $expectedHash = 'e90c1b6419da3bd812dd73bb3de40287a21abf153438147639ec5e20375ea93f'
                if (-not (Test-Path -LiteralPath $archive)) {
                    Invoke-Checked 'curl.exe' @('--fail', '--location', '--retry', '3', '--silent', '--show-error', $uri, '--output', "$archive.partial")
                    if ((Get-FileHash -LiteralPath "$archive.partial" -Algorithm SHA256).Hash -ne $expectedHash) {
                        throw 'Python download SHA256 mismatch; the archive has not been extracted.'
                    }
                    Move-Item -LiteralPath "$archive.partial" -Destination $archive -Force
                }
                if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
                    throw 'Python archive SHA256 mismatch. Move the invalid archive aside and rerun setup.'
                }
                New-Item -ItemType Directory -Force -Path $pythonRoot | Out-Null
                Invoke-Checked 'tar.exe' @('-xzf', $archive, '-C', $pythonRoot)
            }
        }
        $Python = [IO.Path]::GetFullPath($Python)
        Assert-Python $Python
        Invoke-Checked $Python @('-m', 'venv', (Join-Path $dependencyRoot 'python'))
    }
    Assert-Python $venvPython
    Write-Host "Python virtual environment: $venvPython"
}

if ('Dependencies' -in $orderedStages) {
    Assert-Python $venvPython
    $previousPipCache = $env:PIP_CACHE_DIR
    try {
        $env:PIP_CACHE_DIR = Join-Path $dependencyRoot 'pip-cache'
        Invoke-Checked $venvPython @('-m', 'pip', '--disable-pip-version-check', 'install', '--index-url', 'https://pypi.org/simple', '-r', (Join-Path $PSScriptRoot 'requirements-windows.txt'))
        Invoke-Checked $venvPython @('-m', 'pip', 'check')
    } finally {
        $env:PIP_CACHE_DIR = $previousPipCache
    }
}

if ('Native' -in $orderedStages) {
    & (Join-Path $PSScriptRoot 'Setup-Native.ps1') -SourceDirectory $sourceRoot -DependencyDirectory $dependencyRoot -OutputDirectory (Join-Path $WorkspaceRoot 'runtime') -Jobs $Jobs -UsePortableWindowsSdk:$UsePortableWindowsSdk -SkipTests:$SkipTests
}

if ('Model' -in $orderedStages) {
    if ((Test-Path -LiteralPath $bundlePath -PathType Leaf) -and -not $RebuildBundle) {
        Write-Host "Reusing $bundlePath. Use -RebuildBundle after changing the model builder or TensorRT-RTX version."
    } else {
        Assert-Python $venvPython
        Invoke-Checked $venvPython @((Join-Path $PSScriptRoot 'download_model.py'), '--workspace', $WorkspaceRoot)
        & (Join-Path $PSScriptRoot 'Build-Bundle.ps1') -WorkspaceRoot $WorkspaceRoot -Python $venvPython -ModelPath $modelPath -OutputPath $bundlePath
    }
}

if ('App' -in $orderedStages) {
    & (Join-Path $PSScriptRoot 'Setup-App.ps1') -WorkspaceRoot $WorkspaceRoot
}
Write-Host 'Selected setup stages completed. Use Run.ps1 with the same WorkspaceRoot to open the app.'
