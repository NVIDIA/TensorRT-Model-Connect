# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
Runs one pinned MiniMax H3 FastVideo VSA video generation on Windows.

.DESCRIPTION
Downloads the exact public base-model snapshot and VSA adapter through the
authenticated Hugging Face client, verifies the adapter hash, and invokes one
FastVideo generation. The fixed five-point scheduler grid performs four DiT
forwards. No warmup or repeated request is issued.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [string]$InstallRoot = "",

    [string]$Prompt = "",

    [string]$PromptFile = "",

    [int]$NumFrames = 124,

    [double]$DurationSeconds = 0.0,

    [int]$GpuIndex = 0,

    [string]$HuggingFaceCache = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-FileSha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
    param([string]$Value)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
        return [BitConverter]::ToString($sha256.ComputeHash($bytes)).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha256.Dispose()
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

function Invoke-HuggingFaceDownload {
    param(
        [string]$HfExecutable,
        [string]$Repository,
        [string]$Revision,
        [string]$Filename = "",
        [string]$Include = "",
        [string]$CacheDirectory = ""
    )
    if (-not [string]::IsNullOrWhiteSpace($Filename) -and
        -not [string]::IsNullOrWhiteSpace($Include)) {
        throw "Hugging Face download cannot combine Filename and Include"
    }
    $arguments = @("download", $Repository)
    if (-not [string]::IsNullOrWhiteSpace($Filename)) {
        $arguments += $Filename
    }
    if (-not [string]::IsNullOrWhiteSpace($Include)) {
        $arguments += @("--include", $Include)
    }
    $arguments += @("--revision", $Revision, "--quiet")
    if (-not [string]::IsNullOrWhiteSpace($CacheDirectory)) {
        $arguments += @("--cache-dir", $CacheDirectory)
    }
    $output = @(& $HfExecutable @arguments 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Hugging Face download failed; confirm license access and run hf auth login"
    }
    $resolved = @($output | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ })
    if ($resolved.Count -eq 0) {
        throw "Hugging Face did not return a downloaded path"
    }
    return $resolved[-1]
}

function Set-ProcessEnvironment {
    param([string]$Name, [AllowNull()][string]$Value)
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

if ($env:OS -ne "Windows_NT") {
    throw "The FastVideo VSA reproduction runner supports Windows only"
}
if ($GpuIndex -lt 0) {
    throw "GpuIndex must be nonnegative"
}

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$ProfilePath = Join-Path $RepositoryRoot `
    "tests\e2e\models\minimax_h3\fastvideo_windows_vsa_profile.json"
$Profile = Get-Content -Raw -LiteralPath $ProfilePath | ConvertFrom-Json
$ProfileSha256 = Get-FileSha256 $ProfilePath
$OsArchitecture = Get-NativeOsArchitecture
$OsBuild = [Environment]::OSVersion.Version.Build
$ProcessArchitecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
if ($OsArchitecture -ne $Profile.hardware.os_architecture -or
    $ProcessArchitecture -ne $Profile.hardware.process_architecture -or
    $OsBuild -lt $Profile.hardware.minimum_os_build) {
    throw "The validated cohort requires Arm64 Windows with an x64 PowerShell process"
}
$Execution = $Profile.workload.execution
if ($Execution.profile -ne "all" -or
    $Execution.num_gpus -ne 1 -or
    $Execution.execution_backend -ne "mp" -or
    -not $Execution.regional_torch_compile -or
    $Execution.whole_model_torch_compile -or
    -not $Execution.compile_vae -or
    $Execution.parallel_vae -or
    -not $Execution.replicated_dit -or
    -not $Execution.h3_sequential_load -or
    -not $Execution.lazy_module_load -or
    -not $Execution.pin_cpu_memory) {
    throw "The checked-in execution contract is not supported by this runner"
}
$LengthContract = $Profile.workload.video_length
$Fps = [int]$Profile.workload.fps
$DurationWasRequested = $PSBoundParameters.ContainsKey("DurationSeconds")
$FramesWereRequested = $PSBoundParameters.ContainsKey("NumFrames")
if ($DurationWasRequested -and $FramesWereRequested) {
    throw "Specify either DurationSeconds or NumFrames, not both"
}
if ($DurationWasRequested) {
    if ($DurationSeconds -lt [double]$LengthContract.minimum_requested_seconds -or
        $DurationSeconds -gt [double]$LengthContract.maximum_requested_seconds) {
        throw "DurationSeconds must be within the checked-in 5-to-15-second profile"
    }
    $requestedFrames = [int][Math]::Ceiling($DurationSeconds * $Fps)
    $alignment = [int]$LengthContract.frame_alignment
    $offset = [int]$LengthContract.frame_offset
    $alignedSteps = [int][Math]::Ceiling(($requestedFrames - $offset) / [double]$alignment)
    $NumFrames = $offset + ($alignment * $alignedSteps)
    if ($NumFrames -gt [int]$LengthContract.maximum_num_frames) {
        $NumFrames = [int]$LengthContract.maximum_num_frames
    }
}
$FrameMinimum = [int]$LengthContract.minimum_num_frames
$FrameMaximum = [int]$LengthContract.maximum_num_frames
$FrameAlignment = [int]$LengthContract.frame_alignment
$FrameOffset = [int]$LengthContract.frame_offset
if ($NumFrames -lt $FrameMinimum -or
    $NumFrames -gt $FrameMaximum -or
    (($NumFrames - $FrameOffset) % $FrameAlignment) -ne 0) {
    throw "NumFrames must be a native 17n+5 value from 124 through 345"
}
$AlignedDurationSeconds = $NumFrames / [double]$Fps
if ($DurationWasRequested) {
    Write-Host ("Requested {0:N3} seconds resolves to {1} frames ({2:N3} seconds)." -f `
        $DurationSeconds, $NumFrames, $AlignedDurationSeconds)
    if ($AlignedDurationSeconds -lt $DurationSeconds) {
        Write-Warning "The pinned FastVideo 15-second boundary caps this request at 345 frames (14.375 seconds)."
    }
}
$PromptWasRequested = $PSBoundParameters.ContainsKey("Prompt")
$PromptFileWasRequested = $PSBoundParameters.ContainsKey("PromptFile")
if ($PromptWasRequested -and $PromptFileWasRequested) {
    throw "Prompt and PromptFile are mutually exclusive"
}
if ($PromptWasRequested) {
    $PromptText = $Prompt
} else {
    if (-not $PromptFileWasRequested) {
        $PromptFile = Join-Path $RepositoryRoot `
            "tests\e2e\models\minimax_h3\prompts\t2va-example-1.json"
    }
    if ([string]::IsNullOrWhiteSpace($PromptFile)) {
        throw "PromptFile must not be empty when specified"
    }
    $PromptPath = (Resolve-Path -LiteralPath $PromptFile -ErrorAction Stop).Path
    $PromptSpec = Get-Content -Raw -Encoding UTF8 -LiteralPath $PromptPath | ConvertFrom-Json
    if ($PromptSpec.PSObject.Properties.Name -notcontains "prompt" -or
        $PromptSpec.prompt -isnot [string]) {
        throw "PromptFile must contain a string prompt field"
    }
    $PromptText = $PromptSpec.prompt
}
if ([string]::IsNullOrWhiteSpace($PromptText)) {
    throw "The selected prompt must be non-empty"
}
$PromptSha256 = Get-TextSha256 $PromptText
$PatchPath = [IO.Path]::GetFullPath(
    (Join-Path $RepositoryRoot ([string]$Profile.fastvideo.patch_path))
)
if (-not (Test-PathWithin $PatchPath $RepositoryRoot) -or
    -not (Test-Path -LiteralPath $PatchPath -PathType Leaf) -or
    (Get-FileSha256 $PatchPath) -ne $Profile.fastvideo.patch_sha256) {
    throw "The checked-in FastVideo patch does not match the reproduction profile"
}

if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $localData = [Environment]::GetFolderPath("LocalApplicationData")
    $InstallRoot = Join-Path $localData `
        "TensorRT-Model-Connect\minimax-h3-fastvideo-vsa-sm121-v1"
}
$InstallPath = [IO.Path]::GetFullPath($InstallRoot)
$ReceiptPath = Join-Path $InstallPath "install-receipt.json"
$FastVideoRoot = Join-Path $InstallPath "FastVideo"
$KernelPythonRoot = Join-Path $FastVideoRoot ([string]$Profile.fastvideo.kernel_python_path)
$CudaArchiveRoot = Join-Path $InstallPath ([string]$Profile.cuda_toolchain.archive_root)
$PtxasPath = Join-Path $CudaArchiveRoot ([string]$Profile.cuda_toolchain.ptxas_relative_path)
$Python = Join-Path $InstallPath ".venv\Scripts\python.exe"
$Hf = Join-Path $InstallPath ".venv\Scripts\hf.exe"
foreach ($required in @($ReceiptPath, $Python, $Hf, $PtxasPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "The pinned FastVideo VSA environment is not installed; run the installer first"
    }
}
if (-not (Test-Path -LiteralPath $FastVideoRoot -PathType Container)) {
    throw "The installed FastVideo source checkout is missing"
}
if (-not (Test-Path -LiteralPath $KernelPythonRoot -PathType Container)) {
    throw "The installed FastVideo pure-Python kernel package is missing"
}

$Receipt = Get-Content -Raw -LiteralPath $ReceiptPath | ConvertFrom-Json
if ($Receipt.profile_sha256 -ne $ProfileSha256 -or
    $Receipt.fastvideo_revision -ne $Profile.fastvideo.revision -or
    $Receipt.patch_sha256 -ne $Profile.fastvideo.patch_sha256 -or
    $Receipt.cuda_nvcc_version -ne $Profile.cuda_toolchain.nvcc_version -or
    $Receipt.PSObject.Properties.Name -notcontains "ptxas_sha256" -or
    $Receipt.ptxas_sha256 -ne $Profile.cuda_toolchain.ptxas_sha256 -or
    $Receipt.cuda_nvcc_archive_sha256 -ne $Profile.cuda_toolchain.archive_sha256) {
    throw "The installed environment does not match the checked-in reproduction profile"
}
$PtxasItem = Get-Item -LiteralPath $PtxasPath
$PtxasVersion = (& $PtxasPath --version 2>&1 | Out-String)
$PtxasExitCode = $LASTEXITCODE
if ($PtxasItem.Length -ne [int64]$Profile.cuda_toolchain.ptxas_size_bytes -or
    (Get-FileSha256 $PtxasPath) -ne $Profile.cuda_toolchain.ptxas_sha256 -or
    $PtxasExitCode -ne 0 -or
    $PtxasVersion -notmatch ([regex]::Escape("V$($Profile.cuda_toolchain.nvcc_version)"))) {
    throw "The installed ptxas version does not match the reproduction profile"
}
$SourceRevision = (& git -C $FastVideoRoot rev-parse HEAD 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceRevision -ne $Profile.fastvideo.revision) {
    throw "The installed FastVideo source revision does not match the profile"
}
$SourceRepository = (& git -C $FastVideoRoot remote get-url origin 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceRepository -ne $Profile.fastvideo.repository) {
    throw "The installed FastVideo source repository does not match the profile"
}
& git -C $FastVideoRoot diff --quiet --no-ext-diff
if ($LASTEXITCODE -ne 0) {
    throw "The installed FastVideo worktree has unverified changes"
}
$UntrackedPaths = @(& git -C $FastVideoRoot ls-files --others --exclude-standard)
if ($LASTEXITCODE -ne 0 -or $UntrackedPaths.Count -ne 0) {
    throw "The installed FastVideo worktree contains unverified untracked files"
}
& git -C $FastVideoRoot diff --cached --check
if ($LASTEXITCODE -ne 0) {
    throw "The installed FastVideo patch no longer passes git diff --check"
}
& git -C $FastVideoRoot apply --reverse --check --index $PatchPath
if ($LASTEXITCODE -ne 0) {
    throw "The installed FastVideo source does not exactly carry the checked-in patch"
}
$ChangedPaths = @(
    (& git -C $FastVideoRoot diff --cached --name-only --no-ext-diff) |
        ForEach-Object { ([string]$_).Trim().Replace("\", "/") } |
        Where-Object { $_ }
)
$ExpectedChangedPaths = @($Profile.fastvideo.patch_expected_paths | ForEach-Object { [string]$_ })
if (@(Compare-Object $ExpectedChangedPaths $ChangedPaths).Count -ne 0) {
    throw "The installed FastVideo patch path set does not match the profile"
}

$OutputPath = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-PathWithin $OutputPath $RepositoryRoot) {
    throw "OutputDirectory must be outside the source checkout"
}
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

$PromptPayloadPath = Join-Path (
    [IO.Path]::GetTempPath()
) ("trtmc-h3-prompt-{0}.txt" -f [Guid]::NewGuid().ToString("N"))
try {
    $PromptEncoding = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($PromptPayloadPath, $PromptText, $PromptEncoding)

if (-not [string]::IsNullOrWhiteSpace($HuggingFaceCache)) {
    $HuggingFaceCache = [IO.Path]::GetFullPath($HuggingFaceCache)
    New-Item -ItemType Directory -Force -Path $HuggingFaceCache | Out-Null
}
Write-Host "Resolving the authorized pinned MiniMax H3 tokenizer."
$TokenizerSnapshotPath = Invoke-HuggingFaceDownload -HfExecutable $Hf `
    -Repository ([string]$Profile.base_model.repository) `
    -Revision ([string]$Profile.base_model.revision) `
    -Include "processor/*" `
    -CacheDirectory $HuggingFaceCache
$TokenizerPath = Join-Path $TokenizerSnapshotPath "processor"
if (-not (Test-Path -LiteralPath $TokenizerPath -PathType Container)) {
    throw "The pinned MiniMax H3 tokenizer download did not produce the processor directory"
}

$PromptInspection = @'
import json
import pathlib
import sys
from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained(
    sys.argv[1],
    local_files_only=True,
)
prompt = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
print(json.dumps({"prompt_tokens": len(input_ids)}))
'@
$PromptInspectionOutput = (& $Python -c $PromptInspection $TokenizerPath $PromptPayloadPath 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to tokenize the selected prompt with the pinned MiniMax H3 tokenizer"
}
$PromptInfo = $PromptInspectionOutput | ConvertFrom-Json
$PromptTokenCount = [int]$PromptInfo.prompt_tokens
$PromptContract = $Profile.workload.prompt
if ($PromptTokenCount -lt [int]$PromptContract.minimum_tokens -or
    $PromptTokenCount -gt [int]$PromptContract.maximum_tokens) {
    throw "The FastVideo H3 profile supports 1 to 1024 prompt tokens without truncation; got $PromptTokenCount"
}

Write-Host "Resolving the authorized pinned MiniMax H3 snapshot."
$BaseModelPath = Invoke-HuggingFaceDownload -HfExecutable $Hf `
    -Repository ([string]$Profile.base_model.repository) `
    -Revision ([string]$Profile.base_model.revision) `
    -CacheDirectory $HuggingFaceCache
if (-not (Test-Path -LiteralPath $BaseModelPath -PathType Container)) {
    throw "The pinned MiniMax H3 snapshot download did not produce a directory"
}

Write-Host "Resolving the authorized pinned FastH3 VSA adapter."
$AdapterPath = Invoke-HuggingFaceDownload -HfExecutable $Hf `
    -Repository ([string]$Profile.adapter.repository) `
    -Revision ([string]$Profile.adapter.revision) `
    -Filename ([string]$Profile.adapter.file) `
    -CacheDirectory $HuggingFaceCache
if (-not (Test-Path -LiteralPath $AdapterPath -PathType Leaf)) {
    throw "The pinned FastH3 adapter download did not produce a file"
}
$AdapterItem = Get-Item -LiteralPath $AdapterPath
if ($AdapterItem.Length -ne [int64]$Profile.adapter.size_bytes -or
    (Get-FileSha256 $AdapterPath) -ne $Profile.adapter.sha256) {
    throw "The downloaded FastH3 adapter does not match the public size and SHA-256"
}

$EnvironmentNames = @(
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "FASTVIDEO_VSA_TRITON",
    "PYTHONPATH",
    "PYTHONUTF8",
    "TRITON_PTXAS.EXE_PATH",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR"
)
$PreviousEnvironment = @{}
foreach ($name in $EnvironmentNames) {
    $PreviousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$TritonCache = Join-Path $InstallPath "cache\triton"
$InductorCache = Join-Path $InstallPath "cache\torchinductor"
New-Item -ItemType Directory -Force -Path $TritonCache, $InductorCache | Out-Null
Set-ProcessEnvironment "CUDA_DEVICE_ORDER" "PCI_BUS_ID"
Set-ProcessEnvironment "CUDA_VISIBLE_DEVICES" ([string]$GpuIndex)
Set-ProcessEnvironment "FASTVIDEO_VSA_TRITON" "1"
Set-ProcessEnvironment "PYTHONPATH" $KernelPythonRoot
Set-ProcessEnvironment "PYTHONUTF8" "1"
Set-ProcessEnvironment "TRITON_PTXAS.EXE_PATH" $PtxasPath
Set-ProcessEnvironment "TRITON_CACHE_DIR" $TritonCache
Set-ProcessEnvironment "TORCHINDUCTOR_CACHE_DIR" $InductorCache

$Preflight = @'
import importlib.metadata
import json
import platform
import struct
import torch
from triton import knobs
from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton
from fastvideo_kernel.triton_kernels.index import map_to_index
payload = {
    "cuda": torch.cuda.is_available(),
    "count": torch.cuda.device_count(),
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
    "torch": torch.__version__,
    "triton_windows": importlib.metadata.version("triton-windows"),
    "ptxas_path": knobs.nvidia.ptxas.path,
    "ptxas_version": knobs.nvidia.ptxas.version,
    "python_architecture": platform.machine(),
    "python_bits": struct.calcsize("P") * 8,
    "python_version": platform.python_version(),
    "vsa_triton": callable(block_sparse_attn_triton) and callable(map_to_index),
}
print(json.dumps(payload))
'@

$GenerationScript = "examples\inference\basic\basic_fasth3_lora_preview.py"
$Arguments = @(
    "--model-path", $BaseModelPath,
    "--lora-path", $AdapterPath,
    "--lora-strength", "1.0",
    "--output", $OutputPath,
    "--profile", [string]$Execution.profile,
    "--height", [string]$Profile.workload.height,
    "--width", [string]$Profile.workload.width,
    "--num-frames", [string]$NumFrames,
    "--steps", [string]$Profile.workload.scheduler_grid_points,
    "--seed", [string]$Profile.workload.seed,
    "--num-gpus", [string]$Execution.num_gpus,
    "--execution-backend", [string]$Execution.execution_backend,
    "--vsa",
    "--vsa-sparsity", [string]$Profile.workload.vsa_sparsity,
    "--vsa-tile-size", [string]$Profile.workload.vsa_tile_size,
    "--vsa-kernel", "triton",
    "--no-fa4",
    "--video-decode-backend", "h3-vae",
    "--h3-sequential-load",
    "--lazy-module-load",
    "--pin-cpu-memory",
    "--compile-vae",
    "--no-parallel-vae",
    "--inference-torch-compile",
    "--no-torch-compile",
    "--replicated-dit",
    "--ulysses-a2a", "off",
    "--no-warmup",
    "--repeats", "1"
)
$GenerationLauncher = @'
import pathlib
import runpy
import sys

script = pathlib.Path(sys.argv[1]).resolve()
prompt = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
forwarded = sys.argv[3:]
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *forwarded, f"--prompt={prompt}"]
runpy.run_path(str(script), run_name="__main__")
'@

$GenerationExitCode = $null
$ElapsedSeconds = $null
try {
    $PreflightOutput = (& $Python -c $Preflight 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to query the FastVideo CUDA environment"
    }
    $Hardware = $PreflightOutput | ConvertFrom-Json
    if (-not $Hardware.cuda -or -not $Hardware.vsa_triton -or $Hardware.count -lt 1 -or
        $Hardware.capability[0] -ne 12 -or $Hardware.capability[1] -ne 1) {
        throw "This reproduction profile requires one visible compute-capability 12.1 GPU"
    }
    if ($Hardware.torch -ne $Profile.python.packages.torch -or
        $Hardware.triton_windows -ne $Profile.python.packages.'triton-windows' -or
        $Hardware.ptxas_version -ne "12.9" -or
        ([IO.Path]::GetFullPath([string]$Hardware.ptxas_path)) -ne $PtxasPath -or
        $Hardware.python_architecture -ne $Profile.hardware.python_architecture -or
        $Hardware.python_bits -ne $Profile.hardware.python_bits -or
        $Hardware.python_version -ne $Profile.python.validated_version) {
        throw "The installed Python, PyTorch, Triton-Windows, or ptxas version does not match the profile"
    }

    Write-Host "Prompt tokens: $PromptTokenCount (supported range: 1-1024; no truncation)."
    Write-Host ("Video length: {0} frames at {1} fps = {2:N3} seconds." -f `
        $NumFrames, $Fps, $AlignedDurationSeconds)
    Write-Host "Running one FastVideo request: 5 scheduler grid points = 4 DiT forwards."
    Write-Host "The full MiniMax H3 video VAE is enabled."
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    Push-Location -LiteralPath $FastVideoRoot
    try {
        & $Python -c $GenerationLauncher $GenerationScript $PromptPayloadPath @Arguments
        $GenerationExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
        $stopwatch.Stop()
        $ElapsedSeconds = $stopwatch.Elapsed.TotalSeconds
    }
} finally {
    foreach ($name in $EnvironmentNames) {
        Set-ProcessEnvironment $name $PreviousEnvironment[$name]
    }
}

$Summary = [ordered]@{
    schema_version = 1
    profile_id = [string]$Profile.profile_id
    status = if ($GenerationExitCode -eq 0) { "completed" } else { "failed" }
    process_exit_code = $GenerationExitCode
    launcher_wall_seconds = $ElapsedSeconds
    fastvideo_revision = [string]$Profile.fastvideo.revision
    fastvideo_patch_sha256 = [string]$Profile.fastvideo.patch_sha256
    base_model_revision = [string]$Profile.base_model.revision
    adapter_revision = [string]$Profile.adapter.revision
    adapter_sha256 = [string]$Profile.adapter.sha256
    height = [int]$Profile.workload.height
    width = [int]$Profile.workload.width
    num_frames = $NumFrames
    fps = $Fps
    requested_duration_seconds = if ($DurationWasRequested) { $DurationSeconds } else { $null }
    aligned_duration_seconds = $AlignedDurationSeconds
    prompt_tokens = $PromptTokenCount
    prompt_sha256 = $PromptSha256
    scheduler_grid_points = [int]$Profile.workload.scheduler_grid_points
    transformer_forwards = [int]$Profile.workload.transformer_forwards
    measured_requests = 1
    warmup_requests = 0
    video_decode_backend = "h3-vae"
    cuda_nvcc_version = [string]$Profile.cuda_toolchain.nvcc_version
    ptxas_sha256 = [string]$Profile.cuda_toolchain.ptxas_sha256
    os_architecture = $OsArchitecture
    os_build = $OsBuild
    process_architecture = $ProcessArchitecture
    python_architecture = [string]$Hardware.python_architecture
    python_bits = [int]$Hardware.python_bits
    long_sequence_memory_policy = [string]$Profile.workload.long_sequence_memory_policy
    cuda_allocator = [string]$Profile.workload.cuda_allocator
    execution = $Profile.workload.execution
}
$SummaryPath = Join-Path $OutputPath "fastvideo-vsa-summary.json"
$encoding = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText(
    $SummaryPath,
    (($Summary | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
    $encoding
)

if ($GenerationExitCode -ne 0) {
    throw "FastVideo generation failed with exit code $GenerationExitCode"
}
Write-Host "FastVideo VSA generation completed successfully."
Write-Host "A sanitized timing summary was written beside the generated video."
} finally {
    if ([IO.File]::Exists($PromptPayloadPath)) {
        [IO.File]::Delete($PromptPayloadPath)
    }
}
