# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
Builds the native MiniMax H3 video runtime for Windows.

.DESCRIPTION
Run this script from an x64 Visual Studio developer PowerShell. The caller
supplies compatible local CUDA and TensorRT-RTX SDK roots. The script does not
download SDKs, model checkpoints, or Python packages.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CudaRoot,

    [Parameter(Mandatory = $true)]
    [string]$TensorRtRtxRoot,

    [string]$BuildDirectory = "build-windows-h3",

    [switch]$BuildTests,

    [switch]$BuildBenchmarks
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-SdkRoot {
    param(
        [string]$Path,
        [string]$Label,
        [string]$Marker
    )

    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath (Join-Path $resolved $Marker) -PathType Leaf)) {
        throw "$Label does not contain $Marker"
    }
    return $resolved
}

function Resolve-SdkDirectory {
    param(
        [string]$Root,
        [string[]]$Candidates,
        [string]$Label
    )

    foreach ($candidate in $Candidates) {
        $path = Join-Path $Root $candidate
        if (Test-Path -LiteralPath $path -PathType Container) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    throw "$Label was not found under the selected SDK root"
}

foreach ($tool in @("cmake", "ninja", "cl.exe", "git")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required; run from an x64 Visual Studio developer PowerShell"
    }
}

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$CudaSdk = Resolve-SdkRoot $CudaRoot "CUDA 12.9 Toolkit" "bin\nvcc.exe"
$RtxSdk = Resolve-SdkRoot $TensorRtRtxRoot "TensorRT-RTX SDK" "include\NvInfer.h"
$RtxLibraryDirectory = Resolve-SdkDirectory $RtxSdk @("lib", "lib\x64") `
    "TensorRT-RTX library directory"
$RtxRuntimeDirectory = Resolve-SdkDirectory $RtxSdk @("bin", "lib") "TensorRT-RTX runtime directory"
$CudartLibrary = Join-Path $CudaSdk "lib\x64\cudart_static.lib"
if (-not (Test-Path -LiteralPath $CudartLibrary -PathType Leaf)) {
    throw "CUDA 12.9 Toolkit does not contain lib\x64\cudart_static.lib"
}

$NvccVersion = (& (Join-Path $CudaSdk "bin\nvcc.exe") --version 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $NvccVersion -notmatch "release 12\.9") {
    throw "The native MiniMax H3 Windows path requires CUDA 12.9"
}

$BuildPath = if ([IO.Path]::IsPathRooted($BuildDirectory)) {
    [IO.Path]::GetFullPath($BuildDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $BuildDirectory))
}
$BuildTestsValue = if ($BuildTests) { "ON" } else { "OFF" }
$BuildBenchmarksValue = if ($BuildBenchmarks) { "ON" } else { "OFF" }
$SourceRevision = (& git -C $RepositoryRoot rev-parse HEAD 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceRevision -notmatch "^[0-9a-f]{40}$") {
    throw "The Windows H3 build requires a Git checkout with a valid HEAD revision"
}
if ((& git -C $RepositoryRoot status --porcelain | Out-String).Trim()) {
    throw "The reproducible Windows H3 build requires a clean source checkout"
}

$env:CUDA_PATH = $CudaSdk
$env:TRTMC_MINIMAX_H3_SOURCE_REVISION = $SourceRevision
$ConfigureArguments = @(
    "-S", $RepositoryRoot,
    "-B", $BuildPath,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CXX_COMPILER=cl.exe",
    "-DCMAKE_CUDA_COMPILER=$(Join-Path $CudaSdk 'bin\nvcc.exe')",
    "-DCMAKE_CUDA_HOST_COMPILER=cl.exe",
    "-DCMAKE_CUDA_ARCHITECTURES=120-real",
    "-DCMAKE_CUDA_RUNTIME_LIBRARY=Static",
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
    "-DCUDAToolkit_ROOT=$CudaSdk",
    "-DTRTMC_CUDA_INCLUDE_DIR=$(Join-Path $CudaSdk 'include')",
    "-DTRTMC_CUDART_LIBRARY=$CudartLibrary",
    "-DTRTMC_RUNTIME_MODELS=minimax_h3",
    "-DTRTMC_ENABLE_TRT=OFF",
    "-DTRTMC_BUILD_BACKEND_TRT=OFF",
    "-DTRTMC_BUILD_BACKEND_RTX=ON",
    "-DTRTMC_RTX_ROOT=$RtxSdk",
    "-DTRTMC_RTX_INCLUDE_DIR=$(Join-Path $RtxSdk 'include')",
    "-DTRTMC_RTX_LIBRARY_DIR=$RtxLibraryDirectory",
    "-DTRTMC_RTX_RUNTIME_DIR=$RtxRuntimeDirectory",
    "-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF",
    "-DTRTMC_ENABLE_TVM_FFI=OFF",
    "-DTRTMC_BUILD_TESTS=$BuildTestsValue",
    "-DTRTMC_BUILD_BENCHMARKS=$BuildBenchmarksValue",
    "-DTRTMC_SOURCE_REVISION=$SourceRevision",
    "-DTRTMC_DISTRIBUTABLE_BUILD=ON",
    "-DTRTMC_RUNTIME_ONLY_CLI=ON",
    "-DTRTMC_LOCKED_H3_RUNTIME=ON"
)

& cmake @ConfigureArguments
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE"
}

$BuildTargets = @(
    "trtmc",
    "trtmc_core",
    "trtmc_backend_rtx",
    "trtmc_model_minimax_h3",
    "minimax_h3_setup"
)
if ($BuildBenchmarks) {
    $BuildTargets += "trtmc_benchmark_worker"
}
& cmake --build $BuildPath --parallel --target @BuildTargets
if ($LASTEXITCODE -ne 0) {
    throw "Native Windows MiniMax H3 build failed with exit code $LASTEXITCODE"
}

if ($BuildTests) {
    $TestTargets = @(
        "test_dynamic_library",
        "test_backend_loader",
        "test_bundle_format",
        "test_bundle_sha256",
        "test_cli_args",
        "test_cli_args_locked_h3_runtime",
        "test_cli_args_runtime_only",
        "test_config_cli_support",
        "test_trt_version",
        "test_windows_process_lockdown",
        "test_windows_utf8_argv",
        "test_windows_h3_installer",
        "test_windows_media",
        "test_model_plugin_loader",
        "test_pipeline_api",
        "test_trtmc_io",
        "test_minimax_h3_config_schema",
        "test_minimax_h3_math",
        "test_minimax_h3_denoiser_abi",
        "test_minimax_h3_conditioning",
        "test_minimax_h3_fl2va_runtime",
        "test_minimax_h3_ref2va_runtime",
        "test_minimax_h3_vsa_layout",
        "test_minimax_h3_vsa_cuda",
        "test_minimax_h3_cuda_rng"
    )
    & cmake --build $BuildPath --parallel --target @TestTargets
    if ($LASTEXITCODE -ne 0) {
        throw "Native Windows test build failed with exit code $LASTEXITCODE"
    }
    & ctest --test-dir $BuildPath --output-on-failure `
        -R "^(test_dynamic_library|test_backend_loader|test_bundle_format|test_bundle_sha256|test_cli_args|test_cli_args_locked_h3_runtime|test_cli_args_runtime_only|test_config_cli_support|test_trt_version|test_windows_process_lockdown|test_windows_utf8_argv|test_windows_h3_installer|test_windows_media|test_model_plugin_loader|test_pipeline_api|test_trtmc_io|test_minimax_h3_config_schema|test_minimax_h3_math|test_minimax_h3_denoiser_abi|test_minimax_h3_conditioning|test_minimax_h3_fl2va_runtime|test_minimax_h3_ref2va_runtime|test_minimax_h3_vsa_layout|test_minimax_h3_vsa_cuda|test_minimax_h3_cuda_rng)$"
    if ($LASTEXITCODE -ne 0) {
        throw "Native Windows tests failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Native Windows MiniMax H3 targets built successfully."
