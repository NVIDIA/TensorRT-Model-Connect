# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
Installs the pinned FastVideo MiniMax H3 VSA reproduction environment on Windows.

.DESCRIPTION
Creates a per-user Python environment, fetches an exact public FastVideo source
revision, applies the repository-owned Windows patch, and installs the pinned
CUDA 13 PyTorch and Triton-Windows cohort. It does not download model weights,
accept model licenses, or read or write authentication tokens.
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$InstallRoot = "",

    [string]$PythonLauncher = "py",

    [string[]]$PythonLauncherArguments = @("-3.13")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-TextSha256 {
    param([string]$Path)
    $stream = [IO.File]::OpenRead($Path)
    try {
        $algorithm = [Security.Cryptography.SHA256]::Create()
        try {
            $bytes = $algorithm.ComputeHash($stream)
            return ([BitConverter]::ToString($bytes)).Replace("-", "").ToLowerInvariant()
        } finally {
            $algorithm.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Get-NativeOsArchitecture {
    $machineArchitecture = [Environment]::GetEnvironmentVariable(
        "PROCESSOR_ARCHITECTURE",
        "Machine"
    )
    if ([string]::IsNullOrWhiteSpace($machineArchitecture)) {
        return "Unknown"
    }
    switch ($machineArchitecture.ToUpperInvariant()) {
        "ARM64" { return "Arm64" }
        "AMD64" { return "X64" }
        default { return $machineArchitecture }
    }
}

function Test-PathWithin {
    param([string]$Candidate, [string]$Parent)
    $candidateFull = [IO.Path]::GetFullPath($Candidate).TrimEnd("\", "/")
    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd("\", "/")
    if ($candidateFull.Equals($parentFull, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    return $candidateFull.StartsWith(
        $parentFull + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Invoke-CheckedProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Label,
        [string]$WorkingDirectory = ""
    )
    Write-Host $Label
    if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    } else {
        Push-Location -LiteralPath $WorkingDirectory
        try {
            & $FilePath @Arguments
            $exitCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }
    }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
}

function Invoke-PinnedDownload {
    param(
        [string]$Uri,
        [string]$Destination,
        [int64]$ExpectedSize,
        [string]$ExpectedSha256,
        [string]$Label
    )
    Write-Host $Label
    Invoke-WebRequest -Uri $Uri -OutFile $Destination -UseBasicParsing
    $item = Get-Item -LiteralPath $Destination
    if ($item.Length -ne $ExpectedSize -or
        (Get-TextSha256 $Destination) -ne $ExpectedSha256) {
        throw "$Label did not match the pinned size and SHA-256"
    }
}

if ($env:OS -ne "Windows_NT") {
    throw "The FastVideo VSA reproduction installer supports Windows only"
}

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$ProfilePath = Join-Path $RepositoryRoot `
    "tests\e2e\models\minimax_h3\fastvideo_windows_vsa_profile.json"
$Profile = Get-Content -Raw -LiteralPath $ProfilePath | ConvertFrom-Json
$ProfileSha256 = Get-TextSha256 $ProfilePath
$OsArchitecture = Get-NativeOsArchitecture
$OsBuild = [Environment]::OSVersion.Version.Build
$ProcessArchitecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
if ($OsArchitecture -ne $Profile.hardware.os_architecture -or
    $ProcessArchitecture -ne $Profile.hardware.process_architecture -or
    $OsBuild -lt $Profile.hardware.minimum_os_build) {
    throw "The validated cohort requires Arm64 Windows with an x64 PowerShell process"
}
$requirementsRelative = [string]$Profile.python.requirements_path
if ([IO.Path]::IsPathRooted($requirementsRelative)) {
    throw "The Python requirements path in the profile must be repository-relative"
}
$RequirementsPath = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $requirementsRelative))
if (-not (Test-PathWithin $RequirementsPath $RepositoryRoot) -or
    -not (Test-Path -LiteralPath $RequirementsPath -PathType Leaf)) {
    throw "The pinned Windows FastVideo requirements file is missing or outside the checkout"
}

if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $localData = [Environment]::GetFolderPath("LocalApplicationData")
    if ([string]::IsNullOrWhiteSpace($localData)) {
        throw "Windows did not report a LocalApplicationData directory"
    }
    $InstallRoot = Join-Path $localData `
        "TensorRT-Model-Connect\minimax-h3-fastvideo-vsa-sm121-v1"
}
$InstallPath = [IO.Path]::GetFullPath($InstallRoot)
if (Test-PathWithin $InstallPath $RepositoryRoot) {
    throw "InstallRoot must be outside the TensorRT-Model-Connect source checkout"
}

$patchRelative = [string]$Profile.fastvideo.patch_path
if ([IO.Path]::IsPathRooted($patchRelative)) {
    throw "The FastVideo patch path in the profile must be repository-relative"
}
$PatchPath = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $patchRelative))
if (-not (Test-PathWithin $PatchPath $RepositoryRoot)) {
    throw "The FastVideo patch path escapes the repository checkout"
}
$ExpectedPatchSha256 = [string]$Profile.fastvideo.patch_sha256
if ($ExpectedPatchSha256 -notmatch "^[0-9a-f]{64}$") {
    throw "The FastVideo patch SHA-256 placeholder must be replaced before installation"
}
if (-not (Test-Path -LiteralPath $PatchPath -PathType Leaf)) {
    throw "The checked-in FastVideo Windows patch is missing"
}
$ActualPatchSha256 = Get-TextSha256 $PatchPath
if ($ActualPatchSha256 -ne $ExpectedPatchSha256) {
    throw "The checked-in FastVideo Windows patch does not match the pinned SHA-256"
}

foreach ($tool in @("git", $PythonLauncher)) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required"
    }
}

$PythonVersionOutput = (& $PythonLauncher @PythonLauncherArguments --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to query the selected Python launcher"
}
$ExpectedPythonVersion = "Python $($Profile.python.validated_version)"
if ($PythonVersionOutput -ne $ExpectedPythonVersion) {
    throw "The validated profile requires exact Python $($Profile.python.validated_version)"
}
$PythonArchitectureOutput = (& $PythonLauncher @PythonLauncherArguments -c `
    'import platform, struct; print(platform.machine() + chr(58) + str(struct.calcsize(chr(80)) * 8))' `
    2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to query the selected Python architecture"
}
$ExpectedPythonArchitecture = `
    "$($Profile.hardware.python_architecture):$($Profile.hardware.python_bits)"
if ($PythonArchitectureOutput -ne $ExpectedPythonArchitecture) {
    throw "The validated cohort requires 64-bit x64 Python running under Windows emulation"
}

$ReceiptPath = Join-Path $InstallPath "install-receipt.json"
$FastVideoRoot = Join-Path $InstallPath "FastVideo"
$KernelPythonRoot = Join-Path $FastVideoRoot ([string]$Profile.fastvideo.kernel_python_path)
$VenvRoot = Join-Path $InstallPath ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$CudaArchiveRoot = Join-Path $InstallPath ([string]$Profile.cuda_toolchain.archive_root)
$PtxasPath = Join-Path $CudaArchiveRoot ([string]$Profile.cuda_toolchain.ptxas_relative_path)
if (Test-Path -LiteralPath $InstallPath) {
    if (-not (Test-Path -LiteralPath $ReceiptPath -PathType Leaf)) {
        throw "InstallRoot already exists without a completed install receipt; select a new InstallRoot"
    }
    $receipt = Get-Content -Raw -LiteralPath $ReceiptPath | ConvertFrom-Json
    if ($receipt.profile_sha256 -ne $ProfileSha256 -or
        $receipt.fastvideo_revision -ne $Profile.fastvideo.revision -or
        $receipt.patch_sha256 -ne $ExpectedPatchSha256 -or
        $receipt.cuda_nvcc_version -ne $Profile.cuda_toolchain.nvcc_version -or
        $receipt.PSObject.Properties.Name -notcontains "ptxas_sha256" -or
        $receipt.ptxas_sha256 -ne $Profile.cuda_toolchain.ptxas_sha256 -or
        -not (Test-Path -LiteralPath $PtxasPath -PathType Leaf) -or
        (Get-TextSha256 $PtxasPath) -ne $Profile.cuda_toolchain.ptxas_sha256 -or
        -not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        throw "The existing installation does not match this profile; select a new InstallRoot"
    }
    Write-Host "The pinned FastVideo VSA environment is already installed."
    return
}

if (-not $PSCmdlet.ShouldProcess($InstallPath, "Install the pinned FastVideo VSA environment")) {
    return
}

New-Item -ItemType Directory -Path $InstallPath | Out-Null
New-Item -ItemType Directory -Path (Join-Path $InstallPath "cache") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $InstallPath "downloads") | Out-Null
New-Item -ItemType Directory -Path $FastVideoRoot | Out-Null

$CudaArchivePath = Join-Path $InstallPath "downloads\cuda-nvcc.zip"
Invoke-PinnedDownload -Uri ([string]$Profile.cuda_toolchain.archive_url) `
    -Destination $CudaArchivePath `
    -ExpectedSize ([int64]$Profile.cuda_toolchain.archive_size_bytes) `
    -ExpectedSha256 ([string]$Profile.cuda_toolchain.archive_sha256) `
    -Label "Downloading the pinned NVIDIA CUDA NVCC redistributable"
Expand-Archive -LiteralPath $CudaArchivePath -DestinationPath $InstallPath
if (-not (Test-Path -LiteralPath $PtxasPath -PathType Leaf)) {
    throw "The pinned NVIDIA CUDA NVCC archive did not contain ptxas.exe"
}
$PtxasItem = Get-Item -LiteralPath $PtxasPath
if ($PtxasItem.Length -ne [int64]$Profile.cuda_toolchain.ptxas_size_bytes -or
    (Get-TextSha256 $PtxasPath) -ne $Profile.cuda_toolchain.ptxas_sha256) {
    throw "The extracted ptxas.exe does not match the pinned size and SHA-256"
}
$PtxasVersion = (& $PtxasPath --version 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or
    $PtxasVersion -notmatch ([regex]::Escape("V$($Profile.cuda_toolchain.nvcc_version)"))) {
    throw "The installed ptxas version does not match the reproduction profile"
}

Invoke-CheckedProcess -FilePath "git" -Arguments @("init", $FastVideoRoot) `
    -Label "Initializing the isolated FastVideo checkout"
Invoke-CheckedProcess -FilePath "git" -Arguments @(
    "-C", $FastVideoRoot, "config", "core.autocrlf", "false"
) -Label "Configuring deterministic FastVideo line endings"
Invoke-CheckedProcess -FilePath "git" -Arguments @(
    "-C", $FastVideoRoot, "remote", "add", "origin", [string]$Profile.fastvideo.repository
) -Label "Binding the public FastVideo repository"
Invoke-CheckedProcess -FilePath "git" -Arguments @(
    "-C", $FastVideoRoot, "fetch", "--depth=1", "origin", [string]$Profile.fastvideo.revision
) -Label "Fetching the pinned FastVideo revision"
Invoke-CheckedProcess -FilePath "git" -Arguments @(
    "-C", $FastVideoRoot, "checkout", "--detach", "FETCH_HEAD"
) -Label "Checking out the pinned FastVideo revision"

$SourceRevision = (& git -C $FastVideoRoot rev-parse HEAD 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceRevision -ne $Profile.fastvideo.revision) {
    throw "The FastVideo checkout did not resolve to the pinned revision"
}

Invoke-CheckedProcess -FilePath "git" -Arguments @(
    "-C", $FastVideoRoot, "apply", "--check", "--index", $PatchPath
) -Label "Checking the repository-owned FastVideo Windows patch"
Invoke-CheckedProcess -FilePath "git" -Arguments @(
    "-C", $FastVideoRoot, "apply", "--index", $PatchPath
) -Label "Applying the repository-owned FastVideo Windows patch"
Invoke-CheckedProcess -FilePath "git" -Arguments @(
    "-C", $FastVideoRoot, "diff", "--cached", "--check"
) -Label "Validating the patched FastVideo source"

$ActualChangedPaths = @(
    (& git -C $FastVideoRoot diff --cached --name-only --no-ext-diff) |
        ForEach-Object { ([string]$_).Trim().Replace("\", "/") } |
        Where-Object { $_ }
)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect the patched FastVideo paths"
}
$ExpectedChangedPaths = @($Profile.fastvideo.patch_expected_paths | ForEach-Object { [string]$_ })
$UnexpectedPaths = @(Compare-Object $ExpectedChangedPaths $ActualChangedPaths)
if ($UnexpectedPaths.Count -ne 0) {
    throw "The FastVideo patch changed a path outside the pinned patch contract"
}

$VenvArguments = @($PythonLauncherArguments) + @("-m", "venv", $VenvRoot)
Invoke-CheckedProcess -FilePath $PythonLauncher -Arguments $VenvArguments `
    -Label "Creating the isolated Python environment"
Invoke-CheckedProcess -FilePath $VenvPython -Arguments @(
    "-m", "pip", "install", "--upgrade", "pip==$($Profile.python.pip_version)"
) -Label "Installing the pinned pip version"
Invoke-CheckedProcess -FilePath $VenvPython -Arguments @(
    "-m", "pip", "install",
    "--index-url", [string]$Profile.python.torch_index_url,
    "torch==$($Profile.python.packages.torch)",
    "torchvision==$($Profile.python.packages.torchvision)"
) -Label "Installing the pinned CUDA 13 PyTorch cohort"
Invoke-CheckedProcess -FilePath $VenvPython -Arguments @(
    "-m", "pip", "install", "triton-windows==$($Profile.python.packages.'triton-windows')"
) -Label "Installing the pinned Triton-Windows package"
Invoke-CheckedProcess -FilePath $VenvPython -Arguments @(
    "-m", "pip", "install", "--requirement", $RequirementsPath
) -Label "Installing the pinned FastVideo inference dependencies"
Invoke-CheckedProcess -FilePath $VenvPython -Arguments @(
    "-m", "pip", "install", "--no-build-isolation", "--no-deps", "--editable", $FastVideoRoot
) -Label "Installing the patched FastVideo source"

$HealthTritonCache = Join-Path $InstallPath "cache\triton"
if (-not (Test-Path -LiteralPath $KernelPythonRoot -PathType Container)) {
    throw "The pinned FastVideo checkout is missing its pure-Python kernel package"
}
New-Item -ItemType Directory -Force -Path $HealthTritonCache | Out-Null
$PreviousHealthPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
$PreviousHealthTritonCache = [Environment]::GetEnvironmentVariable("TRITON_CACHE_DIR", "Process")
$PreviousHealthPtxas = [Environment]::GetEnvironmentVariable("TRITON_PTXAS.EXE_PATH", "Process")
[Environment]::SetEnvironmentVariable("PYTHONPATH", $KernelPythonRoot, "Process")
[Environment]::SetEnvironmentVariable("TRITON_CACHE_DIR", $HealthTritonCache, "Process")
[Environment]::SetEnvironmentVariable("TRITON_PTXAS.EXE_PATH", $PtxasPath, "Process")
$HealthCheck = @'
import importlib.metadata
import os
import torch
import fastvideo
from triton import knobs
from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton
from fastvideo_kernel.triton_kernels.index import map_to_index
assert torch.cuda.is_available(), "CUDA is unavailable"
assert callable(block_sparse_attn_triton), "Triton VSA kernel is unavailable"
assert callable(map_to_index), "VSA index kernel is unavailable"
assert knobs.nvidia.ptxas.version == "12.9", "Triton did not select CUDA 12.9 ptxas"
assert os.path.normcase(os.path.abspath(knobs.nvidia.ptxas.path)) == os.path.normcase(
    os.path.abspath(os.environ["TRITON_PTXAS.EXE_PATH"])
), "Triton did not select the pinned ptxas path"
print(torch.__version__)
print(importlib.metadata.version("triton-windows"))
'@
try {
    Invoke-CheckedProcess -FilePath $VenvPython -Arguments @("-c", $HealthCheck) `
        -Label "Checking the installed FastVideo CUDA and Triton VSA environment"
} finally {
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $PreviousHealthPythonPath, "Process")
    [Environment]::SetEnvironmentVariable("TRITON_CACHE_DIR", $PreviousHealthTritonCache, "Process")
    [Environment]::SetEnvironmentVariable("TRITON_PTXAS.EXE_PATH", $PreviousHealthPtxas, "Process")
}

$Receipt = [ordered]@{
    schema_version = 1
    profile_id = [string]$Profile.profile_id
    profile_sha256 = $ProfileSha256
    fastvideo_repository = [string]$Profile.fastvideo.repository
    fastvideo_revision = [string]$Profile.fastvideo.revision
    patch_sha256 = $ExpectedPatchSha256
    cuda_nvcc_version = [string]$Profile.cuda_toolchain.nvcc_version
    cuda_nvcc_archive_sha256 = [string]$Profile.cuda_toolchain.archive_sha256
    ptxas_sha256 = [string]$Profile.cuda_toolchain.ptxas_sha256
    os_architecture = $OsArchitecture
    os_build = $OsBuild
    process_architecture = $ProcessArchitecture
    python_architecture = [string]$Profile.hardware.python_architecture
    python_bits = [int]$Profile.hardware.python_bits
    python_required_major_minor = [string]$Profile.python.required_major_minor
    python_version = [string]$Profile.python.validated_version
    torch_version = [string]$Profile.python.packages.torch
    torchvision_version = [string]$Profile.python.packages.torchvision
    triton_windows_version = [string]$Profile.python.packages.'triton-windows'
    installed_at_utc = [DateTime]::UtcNow.ToString("o")
}
$encoding = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText(
    $ReceiptPath,
    (($Receipt | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
    $encoding
)

Write-Host "Installed the pinned FastVideo VSA environment successfully."
Write-Host "Model and adapter assets were not downloaded."
