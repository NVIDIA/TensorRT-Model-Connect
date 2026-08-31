# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
Runs the same-process MiniMax-H3 hot video benchmark on Windows.

.DESCRIPTION
The worker performs one untimed warmup by default, followed by two measured
public pipeline calls. Model loading, the warmup, result validation, telemetry,
and optional bundle hashing are outside the reported call times.

The script does not download checkpoints, model bundles, CUDA, TensorRT-RTX, or
other licensed components. Supply locally authorized inputs built from the same
clean source revision.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Bundle,

    [Parameter(Mandatory = $true)]
    [string]$CudaRoot,

    [Parameter(Mandatory = $true)]
    [string]$TensorRtRtxRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [string]$BuildDirectory = "",

    [string]$PromptFile = "",

    [int]$Warmup = 1,

    [int]$Iterations = 2,

    [double]$FirstBlockCacheThreshold = 0.30,

    [int]$TailWeightBudgetGiB = 24,

    [int]$GpuIndex = 0,

    [switch]$UseFastExit,

    [switch]$DisableTelemetry,

    [switch]$SkipBundleHash,

    [switch]$AllowUnverifiedBundleProvenance
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RequiredFile {
    param([string]$Path, [string]$Label)
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "$Label is not a file: $resolved"
    }
    return $resolved
}

function Resolve-RequiredDirectory {
    param([string]$Path, [string]$Label)
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath $resolved -PathType Container)) {
        throw "$Label is not a directory: $resolved"
    }
    return $resolved
}

function Find-RuntimeDirectory {
    param([string]$Root, [string]$Pattern, [string]$Label)
    foreach ($candidate in @((Join-Path $Root "bin"), (Join-Path $Root "lib"), $Root)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
            continue
        }
        if (Get-ChildItem -LiteralPath $candidate -Filter $Pattern -File -ErrorAction SilentlyContinue) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "$Label runtime matching $Pattern was not found under $Root"
}

function Resolve-SingleRuntimeFile {
    param([string]$Directory, [string]$Pattern, [string]$Label)
    $matches = @(Get-ChildItem -LiteralPath $Directory -Filter $Pattern -File)
    if ($matches.Count -ne 1) {
        throw "$Label must resolve to exactly one file matching $Pattern under $Directory"
    }
    return $matches[0].FullName
}

function Get-TextSha256 {
    param([string]$Value)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
        return -join @($algorithm.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") })
    } finally {
        $algorithm.Dispose()
    }
}

function Write-JsonFile {
    param([object]$Value, [string]$Path, [int]$Depth = 8)
    $encoding = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText(
        $Path,
        (($Value | ConvertTo-Json -Depth $Depth) + [Environment]::NewLine),
        $encoding
    )
}

function ConvertTo-NativeProcessArgument {
    param([string]$Value)
    if ($Value.Contains('"')) {
        throw "Native process arguments containing quotes are not supported"
    }
    $trailing = [regex]::Match($Value, "\\+$")
    if ($trailing.Success) {
        $Value = $Value.Substring(0, $Value.Length - $trailing.Length) +
            $trailing.Value + $trailing.Value
    }
    return '"' + $Value + '"'
}

function Invoke-NativeProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = @(
        $Arguments | ForEach-Object { ConvertTo-NativeProcessArgument ([string]$_) }
    ) -join " "
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object Diagnostics.Process
    $process.StartInfo = $startInfo
    $started = $false
    try {
        if (-not $process.Start()) {
            throw "Failed to start native process: $FilePath"
        }
        $started = $true
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        return [pscustomobject]@{
            exit_code = $process.ExitCode
            stdout = $stdoutTask.GetAwaiter().GetResult()
            stderr = $stderrTask.GetAwaiter().GetResult()
        }
    } finally {
        if ($started) {
            try {
                if (-not $process.HasExited) {
                    $process.Kill()
                    [void]$process.WaitForExit(10000)
                }
            } catch {
                Write-Warning "Unable to terminate interrupted process $FilePath"
            }
        }
        $process.Dispose()
    }
}

function Test-FinitePositive {
    param([double]$Value)
    return -not [double]::IsNaN($Value) -and
        -not [double]::IsInfinity($Value) -and
        $Value -gt 0.0
}

function Get-NativeWindowsArchitecture {
    $source = "HKLM system environment"
    $value = $null
    $registry = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey(
        "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    )
    if ($null -ne $registry) {
        try {
            $value = [string]$registry.GetValue("PROCESSOR_ARCHITECTURE", $null)
        } finally {
            $registry.Dispose()
        }
    }
    if ([string]::IsNullOrWhiteSpace($value)) {
        $source = "process environment"
        $value = [Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITEW6432")
        if ([string]::IsNullOrWhiteSpace($value)) {
            $value = [Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")
        }
    }
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Windows did not report its native processor architecture"
    }
    $normalized = switch -Regex ($value.Trim()) {
        "^(AMD64|X64)$" { "X64"; break }
        "^ARM64$" { "Arm64"; break }
        "^(X86|I[3-6]86)$" { "X86"; break }
        default { $value.Trim() }
    }
    return [pscustomobject]@{
        architecture = $normalized
        source = $source
    }
}

function Read-MiniMaxBundleConfig {
    param([string]$Path)
    $stream = [IO.File]::OpenRead($Path)
    $reader = New-Object IO.BinaryReader($stream)
    try {
        $expectedMagic = [byte[]](0x42, 0x55, 0x4e, 0x44, 0x4c, 0x45, 0x01, 0x00)
        $magic = $reader.ReadBytes($expectedMagic.Length)
        if ($magic.Length -ne $expectedMagic.Length) {
            throw "MiniMax-H3 bundle has a truncated magic header"
        }
        for ($index = 0; $index -lt $expectedMagic.Length; ++$index) {
            if ($magic[$index] -ne $expectedMagic[$index]) {
                throw "MiniMax-H3 bundle has an invalid magic header"
            }
        }
        $headerSize = $reader.ReadUInt64()
        if ($headerSize -eq 0 -or $headerSize -gt (64MB)) {
            throw "MiniMax-H3 bundle has an invalid header size"
        }
        $headerBytes = $reader.ReadBytes([int]$headerSize)
        if ($headerBytes.Length -ne $headerSize) {
            throw "MiniMax-H3 bundle has a truncated header"
        }
        $header = [Text.Encoding]::UTF8.GetString($headerBytes) | ConvertFrom-Json
        $section = $header.sections.'config.json'
        if ($null -eq $section) {
            throw "MiniMax-H3 bundle is missing config.json"
        }
        $offset = [int64]$section.offset
        $size = [int64]$section.size
        $dataStart = [int64]$expectedMagic.Length + 8 + [int64]$headerSize
        if ($offset -lt 0 -or $size -le 0 -or $dataStart + $offset + $size -gt $stream.Length) {
            throw "MiniMax-H3 bundle config.json has invalid bounds"
        }
        $stream.Position = $dataStart + $offset
        $configBytes = $reader.ReadBytes([int]$size)
        if ($configBytes.Length -ne $size) {
            throw "MiniMax-H3 bundle config.json is truncated"
        }
        return [Text.Encoding]::UTF8.GetString($configBytes) | ConvertFrom-Json
    } finally {
        $reader.Dispose()
        $stream.Dispose()
    }
}

function Get-TensorRtRtxVersion {
    param([string]$Root)
    $header = Get-Content -LiteralPath (Join-Path $Root "include\NvInferVersion.h") -Raw
    $parts = @()
    foreach ($name in @("MAJOR", "MINOR", "PATCH", "BUILD")) {
        $match = [regex]::Match(
            $header,
            "(?m)^#define[ \t]+TRT_${name}_RTX[ \t]+([0-9]+)[ \t]*\r?$"
        )
        if (-not $match.Success) {
            throw "TensorRT-RTX NvInferVersion.h is missing TRT_${name}_RTX"
        }
        $parts += $match.Groups[1].Value
    }
    return $parts -join "."
}

function Get-ObjectProperty {
    param([object]$Value, [string]$Name)
    $property = $Value.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Get-Median {
    param([double[]]$Values)
    if ($Values.Count -eq 0) {
        throw "Cannot compute a median from an empty sample set"
    }
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[$middle]
    }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

if ($Warmup -lt 0) {
    throw "Warmup must be nonnegative"
}
if ($Iterations -le 0) {
    throw "Iterations must be positive"
}
if ([double]::IsNaN($FirstBlockCacheThreshold) -or
    [double]::IsInfinity($FirstBlockCacheThreshold) -or
    $FirstBlockCacheThreshold -le 0.0) {
    throw "FirstBlockCacheThreshold must be finite and positive"
}
if ($TailWeightBudgetGiB -le 0) {
    throw "TailWeightBudgetGiB must be positive"
}
if ($GpuIndex -lt 0) {
    throw "GpuIndex must be nonnegative"
}

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$BaselinePromptFile = Join-Path $RepositoryRoot `
    "tests\e2e\models\minimax_h3\prompts\t2va-example-1.json"
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $RepositoryRoot "build-windows-h3"
}
if ([string]::IsNullOrWhiteSpace($PromptFile)) {
    $PromptFile = $BaselinePromptFile
}

$BundlePath = Resolve-RequiredFile $Bundle "MiniMax-H3 bundle"
$BuildRoot = Resolve-RequiredDirectory $BuildDirectory "build directory"
$CudaSdk = Resolve-RequiredDirectory $CudaRoot "CUDA root"
$RtxSdk = Resolve-RequiredDirectory $TensorRtRtxRoot "TensorRT-RTX root"
$PromptPath = Resolve-RequiredFile $PromptFile "prompt file"
$PluginRoot = Resolve-RequiredDirectory (Join-Path $BuildRoot "models\minimax_h3") `
    "MiniMax-H3 plugin directory"
$Worker = Resolve-RequiredFile (Join-Path $BuildRoot "trtmc_benchmark_worker.exe") `
    "benchmark worker"
$Core = Resolve-RequiredFile (Join-Path $BuildRoot "trtmc_core.dll") "core runtime"
$Backend = Resolve-RequiredFile (Join-Path $BuildRoot "trtmc_backend_trt_rtx.dll") `
    "TensorRT-RTX backend"
$Plugin = Resolve-RequiredFile (Join-Path $PluginRoot "trtmc_model_minimax_h3.dll") `
    "MiniMax-H3 plugin"
$CudaBin = Find-RuntimeDirectory $CudaSdk "cudart64_12.dll" "CUDA"
$RtxBin = Find-RuntimeDirectory $RtxSdk "tensorrt_rtx*.dll" "TensorRT-RTX"
$SelectedCudart = Resolve-SingleRuntimeFile $CudaBin "cudart64_12.dll" "CUDA runtime"
$SelectedRtxRuntime = Resolve-SingleRuntimeFile $RtxBin "tensorrt_rtx*.dll" `
    "TensorRT-RTX runtime"
$BuildCudart = Resolve-RequiredFile (Join-Path $BuildRoot "cudart64_12.dll") `
    "build CUDA runtime"
$BuildRtxRuntime = Resolve-RequiredFile (
    Join-Path $BuildRoot (Split-Path -Leaf $SelectedRtxRuntime)
) "build TensorRT-RTX runtime"

foreach ($tool in @("git", "cl.exe", "nvidia-smi.exe", "python")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required for a provenance-bound benchmark run"
    }
}

$SourceRevision = (& git -C $RepositoryRoot rev-parse HEAD 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceRevision -notmatch "^[0-9a-f]{40}$") {
    throw "Run this benchmark from a Git checkout with a valid HEAD revision"
}
if ((& git -C $RepositoryRoot status --porcelain | Out-String).Trim()) {
    throw "The benchmark requires a clean source checkout"
}

$WorkerMetadataResult = Invoke-NativeProcess $Worker @("--metadata")
if ($WorkerMetadataResult.exit_code -ne 0) {
    throw "Benchmark worker metadata query failed with exit code $($WorkerMetadataResult.exit_code)"
}
$WorkerMetadata = $WorkerMetadataResult.stdout | ConvertFrom-Json
if ($WorkerMetadata.build.source_revision -ne $SourceRevision -or
    $WorkerMetadata.build.configuration -ne "Release") {
    throw "Worker must be a Release build from the current checkout; rebuild it"
}

$BundleConfig = Read-MiniMaxBundleConfig $BundlePath
if ($BundleConfig.model_type -ne "minimax_h3" -or
    $BundleConfig.runtime_strategy -ne "diffusion_minimax_h3" -or
    $BundleConfig.engine_backend -ne "trt_rtx" -or
    $BundleConfig.cuda_major -ne 12 -or
    $BundleConfig.first_block_cache -ne $true -or
    $BundleConfig.denoiser_cache_mode -ne "first_block" -or
    $BundleConfig.height -ne 768 -or
    $BundleConfig.width -ne 1344 -or
    $BundleConfig.num_frames -ne 124 -or
    $BundleConfig.num_inference_steps -ne 50 -or
    $BundleConfig.context_parallel_size -ne 1 -or
    $BundleConfig.padded_sequence_length -ne 38247 -or
    $BundleConfig.precision -ne "bf16" -or
    $BundleConfig.vae_tile_batch -ne 28 -or
    $BundleConfig.runtime_memory.mode -ne "staged") {
    throw "MiniMax-H3 bundle does not match the fixed Windows hot profile"
}
$BundleWeightBudgetBytes = [int64]$BundleConfig.runtime_memory.weight_streaming_budget_bytes
if ($BundleWeightBudgetBytes -le 0) {
    throw "MiniMax-H3 bundle has an invalid weight-streaming budget"
}
$ExpectedPlans = @(
    "adaln_precompute.plan",
    "denoiser_finish.plan",
    "denoiser_head.plan",
    "denoiser_tail.plan",
    "text_encoder.plan",
    "vae_tile_decoder.plan"
)
$BundlePlans = @($BundleConfig.plan_sha256.PSObject.Properties.Name | Sort-Object)
if (($BundlePlans -join "|") -ne ($ExpectedPlans -join "|")) {
    throw "MiniMax-H3 bundle does not contain the required six-plan profile"
}
$BundleSourceRevision = [string](Get-ObjectProperty $BundleConfig "source_revision")
$CheckpointRevision = [string](Get-ObjectProperty $BundleConfig "checkpoint_revision")
$BuilderSourceSha256 = [string](Get-ObjectProperty $BundleConfig "builder_source_sha256")
$CheckpointInventorySha256 = [string](
    Get-ObjectProperty $BundleConfig "checkpoint_inventory_sha256"
)
$BundleValidator = @'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tensorrt_model_connect.families.minimax_h3.provenance import validate_native_bundle_config
validate_native_bundle_config(Path(sys.argv[2]), source_revision=sys.argv[3])
print("validated")
'@
$BundleValidatorPayload = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes($BundleValidator)
)
$BundleValidatorCommand =
    "import base64;exec(base64.b64decode('$BundleValidatorPayload'))"
$PythonPath = (Get-Command python -ErrorAction Stop).Source
$BundleValidationResult = Invoke-NativeProcess $PythonPath @(
    "-c",
    $BundleValidatorCommand,
    (Join-Path $RepositoryRoot "python"),
    $BundlePath,
    $SourceRevision
)
$BundleValidationStdout = $BundleValidationResult.stdout.Trim()
$BundleValidationStderrLines = @(
    $BundleValidationResult.stderr -split "`r?`n" | Where-Object { $_ }
)
$BundleValidationLastLine = if ($BundleValidationStderrLines.Count -gt 0) {
    [string]$BundleValidationStderrLines[-1]
} else {
    "validator produced no output"
}
$BundleProvenanceVerified =
    $BundleValidationResult.exit_code -eq 0 -and $BundleValidationStdout -eq "validated"
$BundleProvenanceValidationError = if ($BundleProvenanceVerified) {
    $null
} else {
    $BundleValidationLastLine
}
if (-not $BundleProvenanceVerified -and -not $AllowUnverifiedBundleProvenance) {
    throw "Bundle provenance does not match this checkout: $BundleProvenanceValidationError"
}

$TensorRtRtxVersion = Get-TensorRtRtxVersion $RtxSdk
if ([string]$BundleConfig.trt_version -ne $TensorRtRtxVersion) {
    throw "Bundle TensorRT-RTX version does not match the selected SDK"
}
$NvccVersionResult = Invoke-NativeProcess (Join-Path $CudaSdk "bin\nvcc.exe") @("--version")
$NvccVersionText = $NvccVersionResult.stdout + $NvccVersionResult.stderr
$NvccVersionMatch = [regex]::Match($NvccVersionText, "V([0-9]+\.[0-9]+\.[0-9]+)")
if ($NvccVersionResult.exit_code -ne 0 -or -not $NvccVersionMatch.Success) {
    throw "Unable to identify the CUDA Toolkit version"
}
$CudaToolkitVersion = $NvccVersionMatch.Groups[1].Value
if ($CudaToolkitVersion -notmatch "^12\.9\.") {
    throw "The fixed MiniMax-H3 Windows hot profile requires CUDA Toolkit 12.9"
}
$CompilerPath = (Get-Command cl.exe -ErrorAction Stop).Source
$CompilerVersion = (Get-Item -LiteralPath $CompilerPath).VersionInfo.ProductVersion
$PythonVersionResult = Invoke-NativeProcess $PythonPath @("--version")
$PythonVersion = ($PythonVersionResult.stdout + $PythonVersionResult.stderr).Trim()
if ($PythonVersionResult.exit_code -ne 0 -or [string]::IsNullOrWhiteSpace($PythonVersion)) {
    throw "Unable to identify the Python interpreter used for provenance validation"
}
$NativeWindowsArchitecture = Get-NativeWindowsArchitecture

$NvidiaSmiPath = (Get-Command nvidia-smi.exe -ErrorAction Stop).Source
$GpuIdentityResult = Invoke-NativeProcess $NvidiaSmiPath @(
    "--id=$GpuIndex",
    "--query-gpu=index,name,compute_cap,driver_version,memory.total",
    "--format=csv,noheader,nounits"
)
$GpuIdentityText = $GpuIdentityResult.stdout.Trim()
if ($GpuIdentityResult.exit_code -ne 0 -or [string]::IsNullOrWhiteSpace($GpuIdentityText)) {
    throw "nvidia-smi could not identify GPU index $GpuIndex"
}
$GpuFields = @($GpuIdentityText.Split(",") | ForEach-Object { $_.Trim() })
if ($GpuFields.Count -ne 5) {
    throw "nvidia-smi returned an unexpected GPU identity"
}
$GpuMemoryMiB = [double]::Parse(
    $GpuFields[4],
    [Globalization.CultureInfo]::InvariantCulture
)
if ($GpuFields[2] -notmatch "^12\." -or $GpuMemoryMiB -lt 60000) {
    throw "The fixed MiniMax-H3 hot profile requires a compute-capability 12.x GPU with about 64 GiB of visible memory"
}

$PromptSpec = Get-Content -LiteralPath $PromptPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace([string]$PromptSpec.prompt)) {
    throw "The selected prompt file has no prompt"
}
$BaselinePromptSpec = Get-Content -LiteralPath $BaselinePromptFile -Raw | ConvertFrom-Json
$PromptText = [string]$PromptSpec.prompt
$BaselinePromptText = [string]$BaselinePromptSpec.prompt
$PromptSha256 = Get-TextSha256 $PromptText
$BaselinePromptSha256 = Get-TextSha256 $BaselinePromptText
$PromptMatchesBaseline = $PromptSha256 -eq $BaselinePromptSha256 -and
    [string]::Equals($PromptText, $BaselinePromptText, [StringComparison]::Ordinal)

$CaseProfile = [ordered]@{
    model_id = "MiniMaxAI/MiniMax-H3"
    seed = 0
    batch_size = 1
    height = 768
    width = 1344
    video_num_frames = 124
    num_inference_steps = 50
    warmup = $Warmup
    iterations = $Iterations
    first_block_cache_threshold = $FirstBlockCacheThreshold
    retain_engines = $true
    retained_tail_weight_budget_gib = $TailWeightBudgetGiB
    cuda_graphs = $false
    prompt = $PromptText
    prompt_sha256 = $PromptSha256
}
$Invariant = [Globalization.CultureInfo]::InvariantCulture
$CaseIdentity = @(
    "schema=trtmc.minimax-h3-windows-hot-case/v1"
    "model_id=MiniMaxAI/MiniMax-H3"
    "seed=0"
    "batch_size=1"
    "height=768"
    "width=1344"
    "video_num_frames=124"
    "num_inference_steps=50"
    "media_type=video"
    "warmup=$($Warmup.ToString($Invariant))"
    "iterations=$($Iterations.ToString($Invariant))"
    "first_block_cache_threshold=$($FirstBlockCacheThreshold.ToString('R', $Invariant))"
    "retain_engines=true"
    "retained_tail_weight_budget_gib=$($TailWeightBudgetGiB.ToString($Invariant))"
    "cuda_graphs=false"
    "prompt_sha256=$PromptSha256"
) -join "`n"
$CaseDigest = Get-TextSha256 $CaseIdentity

$RunId = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$OutputRoot = [IO.Path]::GetFullPath($OutputDirectory)
$RepositoryPrefix = $RepositoryRoot.TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar
if ($OutputRoot.Equals($RepositoryRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $OutputRoot.StartsWith($RepositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDirectory must be outside the source checkout"
}
$RunRoot = Join-Path $OutputRoot "minimax-h3-hot-$RunId"

$RequestPath = Join-Path $RunRoot "worker-request.json"
$ResultPath = Join-Path $RunRoot "worker-result.json"
$LogPath = Join-Path $RunRoot "worker.log"
$WorkerStdoutPath = Join-Path $RunRoot "worker.stdout.log"
$TelemetryPath = Join-Path $RunRoot "gpu-telemetry.csv"
$ReceiptPath = Join-Path $RunRoot "environment.json"
$SummaryPath = Join-Path $RunRoot "summary.json"

$Request = [ordered]@{
    schema_version = 1
    case_name = "minimax-h3-windows-hot-768p"
    case_digest = $CaseDigest
    bundle = $BundlePath
    operation = "generate_image"
    request = [ordered]@{
        prompt = $PromptText
        seed = 0
        batch_size = 1
        height = 768
        width = 1344
        video_height = 768
        video_width = 1344
        video_num_frames = 124
        num_inference_steps = 50
        media_type = "video"
    }
    runtime = [ordered]@{
        cuda_graphs = $false
        config = [ordered]@{
            "minimax_h3.first_block_cache_threshold" = $FirstBlockCacheThreshold
            "minimax_h3.retain_engines" = $true
            "minimax_h3.retained_tail_weight_budget_gib" = $TailWeightBudgetGiB
        }
        backend_search_paths = @($BuildRoot)
        model_plugin_search_paths = @($PluginRoot)
    }
    measurement = [ordered]@{
        warmup = $Warmup
        iterations = $Iterations
        timing_scope = "public_pipeline_call_wall"
        asset_loading_included = $false
    }
}

$SelectedCudartSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $SelectedCudart).Hash.ToLowerInvariant()
$BuildCudartSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $BuildCudart).Hash.ToLowerInvariant()
$SelectedRtxRuntimeSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $SelectedRtxRuntime
).Hash.ToLowerInvariant()
$BuildRtxRuntimeSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $BuildRtxRuntime
).Hash.ToLowerInvariant()
if ($BuildCudartSha256 -ne $SelectedCudartSha256 -or
    $BuildRtxRuntimeSha256 -ne $SelectedRtxRuntimeSha256) {
    throw "Build runtime DLLs do not match the selected CUDA and TensorRT-RTX SDKs; rebuild from this checkout"
}

$ArtifactHashes = [ordered]@{
    worker_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Worker).Hash.ToLowerInvariant()
    core_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Core).Hash.ToLowerInvariant()
    backend_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Backend).Hash.ToLowerInvariant()
    plugin_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Plugin).Hash.ToLowerInvariant()
    cudart_runtime_sha256 = $BuildCudartSha256
    tensorrt_rtx_runtime_sha256 = $BuildRtxRuntimeSha256
}
if (-not $SkipBundleHash) {
    $ArtifactHashes.bundle_sha256 =
        (Get-FileHash -Algorithm SHA256 -LiteralPath $BundlePath).Hash.ToLowerInvariant()
}
$RequestedTailBudgetBytes = [int64]$TailWeightBudgetGiB * (1GB)
$EffectiveTailBudgetBytes = [Math]::Min(
    $RequestedTailBudgetBytes,
    $BundleWeightBudgetBytes
)
$TelemetryState = [ordered]@{
    requested = -not [bool]$DisableTelemetry
    started = $false
    status = if ($DisableTelemetry) { "disabled" } else { "pending" }
    exit_code = $null
    sample_count = 0
    finalization_error = $null
}
$Receipt = [ordered]@{
    schema_version = "trtmc.minimax-h3-windows-hot-environment/v1"
    status = "running"
    source_revision = $SourceRevision
    started_utc = [DateTimeOffset]::UtcNow.ToString("o")
    ended_utc = $null
    worker_exit_code = $null
    process_architecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
    os_architecture = $NativeWindowsArchitecture.architecture
    os_architecture_source = $NativeWindowsArchitecture.source
    runtime_reported_os_architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    os_description = [Runtime.InteropServices.RuntimeInformation]::OSDescription
    powershell_version = $PSVersionTable.PSVersion.ToString()
    python_version = $PythonVersion
    compiler_version = $CompilerVersion
    cuda_toolkit_version = $CudaToolkitVersion
    tensorrt_rtx_version = $TensorRtRtxVersion
    gpu = [ordered]@{
        index = [int]$GpuFields[0]
        name = $GpuFields[1]
        compute_capability = $GpuFields[2]
        driver_version = $GpuFields[3]
        memory_total_mib = $GpuMemoryMiB
    }
    workload = $CaseProfile
    fast_exit_requested = [bool]$UseFastExit
    bundle_provenance_verified = [bool]$BundleProvenanceVerified
    bundle_provenance_validation_error = $BundleProvenanceValidationError
    unverified_bundle_provenance_allowed = [bool]$AllowUnverifiedBundleProvenance
    bundle_source_revision = $BundleSourceRevision
    checkpoint_revision = $CheckpointRevision
    builder_source_sha256 = $BuilderSourceSha256
    checkpoint_inventory_sha256 = $CheckpointInventorySha256
    bundle_weight_streaming_budget_bytes = $BundleWeightBudgetBytes
    requested_tail_weight_budget_bytes = $RequestedTailBudgetBytes
    effective_tail_weight_budget_bytes = $EffectiveTailBudgetBytes
    bundle_bytes = (Get-Item -LiteralPath $BundlePath).Length
    artifact_hashes = $ArtifactHashes
    telemetry = $TelemetryState
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $RunRoot | Out-Null
Write-JsonFile $WorkerMetadata (Join-Path $RunRoot "worker-metadata.json")
Write-JsonFile $Request $RequestPath
Write-JsonFile $Receipt $ReceiptPath

$EnvironmentNames = @(
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "TRTMC_MODEL_PLUGIN_DIR",
    "TRTMC_MODEL_PLUGIN_STRICT",
    "TRTMC_BENCHMARK_FAST_EXIT",
    "TRTMC_NCCL_RENDEZVOUS",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_JOBID",
    "PMI_SIZE",
    "PMI_RANK",
    "WORLD_SIZE",
    "RANK",
    "PATH"
)
$OriginalEnvironment = @{}
foreach ($Name in $EnvironmentNames) {
    $Item = Get-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
    $OriginalEnvironment[$Name] = if ($null -eq $Item) { $null } else { $Item.Value }
}

$Telemetry = $null
$WorkerExitCode = -1
$RunSucceeded = $false
$RunError = $null
try {
    try {
        $env:CUDA_PATH = $CudaSdk
        $env:CUDA_VISIBLE_DEVICES = [string]$GpuIndex
        $env:TRTMC_MODEL_PLUGIN_DIR = $PluginRoot
        $env:TRTMC_MODEL_PLUGIN_STRICT = "1"
        $env:WORLD_SIZE = "1"
        $env:RANK = "0"
        foreach ($Name in @(
                "TRTMC_NCCL_RENDEZVOUS",
                "OMPI_COMM_WORLD_SIZE",
                "OMPI_COMM_WORLD_RANK",
                "OMPI_COMM_WORLD_JOBID",
                "PMI_SIZE",
                "PMI_RANK"
            )) {
            Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
        }
        if ($UseFastExit) {
            $env:TRTMC_BENCHMARK_FAST_EXIT = "1"
        } else {
            Remove-Item -LiteralPath "Env:TRTMC_BENCHMARK_FAST_EXIT" -ErrorAction SilentlyContinue
        }
        $env:PATH = @($BuildRoot, $PluginRoot, $RtxBin, $CudaBin, $env:PATH) -join `
            [IO.Path]::PathSeparator

        if (-not $DisableTelemetry) {
            $TelemetryArguments = @(
                "--id=$GpuIndex"
                "--query-gpu=timestamp,index,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm"
                "--format=csv"
                "--loop-ms=500"
            )
            try {
                $Telemetry = Start-Process -FilePath "nvidia-smi.exe" `
                    -ArgumentList $TelemetryArguments `
                    -RedirectStandardOutput $TelemetryPath `
                    -RedirectStandardError (Join-Path $RunRoot "gpu-telemetry.stderr.log") `
                    -WindowStyle Hidden `
                    -PassThru
                $TelemetryState.started = $true
                $TelemetryState.status = "running"
            } catch {
                $TelemetryState.status = "start_failed"
                $TelemetryState.finalization_error = $_.Exception.Message
                throw
            }
        }

        $WorkerProcessResult = Invoke-NativeProcess $Worker @(
            "--request",
            $RequestPath,
            "--output",
            $ResultPath
        )
        $WorkerExitCode = $WorkerProcessResult.exit_code
        $textEncoding = New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($LogPath, $WorkerProcessResult.stderr, $textEncoding)
        [IO.File]::WriteAllText(
            $WorkerStdoutPath,
            $WorkerProcessResult.stdout,
            $textEncoding
        )
    } finally {
        if ($null -ne $Telemetry) {
            try {
                if (-not $Telemetry.HasExited) {
                    Stop-Process -Id $Telemetry.Id -ErrorAction SilentlyContinue
                }
                if (-not $Telemetry.WaitForExit(10000)) {
                    throw "nvidia-smi telemetry did not exit within 10 seconds"
                }
                $TelemetryState.exit_code = $Telemetry.ExitCode
            } catch {
                $TelemetryState.finalization_error = $_.Exception.Message
                $TelemetryState.status = "finalization_failed"
                Write-Warning "Unable to finalize GPU telemetry: $($_.Exception.Message)"
            }
            if (Test-Path -LiteralPath $TelemetryPath -PathType Leaf) {
                $TelemetryLines = @(Get-Content -LiteralPath $TelemetryPath)
                $TelemetryState.sample_count = [Math]::Max(0, $TelemetryLines.Count - 1)
            }
            if ($null -eq $TelemetryState.finalization_error) {
                $TelemetryState.status = if ($TelemetryState.sample_count -gt 0) {
                    "captured"
                } else {
                    "empty"
                }
            }
        }
        foreach ($Name in $EnvironmentNames) {
            if ($null -eq $OriginalEnvironment[$Name]) {
                Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
            } else {
                Set-Item -LiteralPath "Env:$Name" -Value $OriginalEnvironment[$Name]
            }
        }
    }

    if ($WorkerExitCode -ne 0) {
        throw "MiniMax-H3 benchmark worker failed with exit code $WorkerExitCode; inspect $LogPath"
    }
    if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
        throw "MiniMax-H3 benchmark did not write $ResultPath"
    }

    $Result = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
    if ($Result.status -ne "completed" -or $Result.case_digest -ne $CaseDigest) {
        throw "MiniMax-H3 benchmark result status or case digest does not match the request"
    }
    $Observations = @($Result.observations)
    if ($Observations.Count -ne $Iterations) {
        throw "MiniMax-H3 benchmark returned the wrong number of observations"
    }
    $ExpectedFrames = 124
    $ExpectedElements = 383975424
    if (@($Observations | Where-Object {
                $_.generated_images -ne 1 -or
                $_.generated_frames -ne $ExpectedFrames -or
                $_.generated_pixels -ne $ExpectedElements
            }).Count -ne 0) {
        throw "MiniMax-H3 benchmark returned an unexpected output shape"
    }
    if (@($Observations | Where-Object { $_.nonfinite_elements -ne 0 }).Count -ne 0 -or
        $Result.output_summary.nonfinite_elements -ne 0) {
        throw "MiniMax-H3 benchmark produced non-finite output"
    }
    if ($Result.output_summary.height -ne 768 -or
        $Result.output_summary.width -ne 1344 -or
        $Result.output_summary.channels -ne 3 -or
        $Result.output_summary.num_frames -ne $ExpectedFrames -or
        $Result.output_summary.element_count -ne $ExpectedElements) {
        throw "MiniMax-H3 benchmark output summary does not match the fixed workload"
    }

    $Samples = @($Observations | ForEach-Object { [double]$_.measured_wall_ms })
    if (@($Samples | Where-Object { -not (Test-FinitePositive $_) }).Count -ne 0) {
        throw "MiniMax-H3 benchmark returned an invalid measured time"
    }

    $LogText = Get-Content -LiteralPath $LogPath -Raw
    $CacheMisses = [regex]::Matches($LogText, "\[trtmc\.rtx_engine_cache\] hit=0").Count
    $CacheHits = [regex]::Matches($LogText, "\[trtmc\.rtx_engine_cache\] hit=1").Count
    $TotalRequests = $Warmup + $Iterations
    if ($CacheMisses -ne 4 -or $CacheHits -lt (4 * [Math]::Max(0, $TotalRequests - 1))) {
        throw "MiniMax-H3 retained-engine cache evidence is incomplete"
    }
    $BudgetPattern =
        "\[trtmc\.rtx_weight_budget\] requested_bytes=([0-9]+) streamable_bytes=([0-9]+) applied_bytes=([0-9]+) streaming_scratch_bytes=([0-9]+)"
    $TailBudgetEvidence = @(
        [regex]::Matches($LogText, $BudgetPattern) |
            Where-Object { [int64]$_.Groups[1].Value -eq $EffectiveTailBudgetBytes } |
            ForEach-Object {
                [ordered]@{
                    requested_bytes = [int64]$_.Groups[1].Value
                    streamable_bytes = [int64]$_.Groups[2].Value
                    applied_bytes = [int64]$_.Groups[3].Value
                    streaming_scratch_bytes = [int64]$_.Groups[4].Value
                }
            }
    )
    if ($TailBudgetEvidence.Count -eq 0) {
        throw "MiniMax-H3 effective tail weight budget was not observed in the runtime log"
    }

    $BaselineProfile =
        $Warmup -eq 1 -and
        $Iterations -eq 2 -and
        [Math]::Abs($FirstBlockCacheThreshold - 0.30) -lt 1.0e-12 -and
        $TailWeightBudgetGiB -eq 24 -and
        $PromptMatchesBaseline -and
        -not $SkipBundleHash -and
        $CudaToolkitVersion -match "^12\.9\." -and
        -not $UseFastExit -and
        $BundleProvenanceVerified
    $Summary = [ordered]@{
        schema_version = "trtmc.minimax-h3-windows-hot-summary/v1"
        status = if ($UseFastExit) { "inference_completed" } else { "completed" }
        inference_status = "completed"
        lifecycle_status = if ($UseFastExit) { "bypassed" } else { "normal_teardown" }
        baseline_profile = [bool]$BaselineProfile
        worker_exit_code = $WorkerExitCode
        fast_exit_used = [bool]$UseFastExit
        bundle_provenance_verified = [bool]$BundleProvenanceVerified
        timing_scope = "public_pipeline_call_wall"
        samples_ms = $Samples
        median_ms = Get-Median $Samples
        retained_engine_cache = [ordered]@{
            misses = $CacheMisses
            hits = $CacheHits
        }
        effective_tail_weight_budget_bytes = $EffectiveTailBudgetBytes
        tail_weight_budget_evidence = $TailBudgetEvidence
        output_summary = $Result.output_summary
        result_file = "worker-result.json"
        log_file = "worker.log"
    }
    Write-JsonFile $Summary $SummaryPath
    $RunSucceeded = $true
} catch {
    $RunError = $_.Exception.Message
    $FailureSummary = [ordered]@{
        schema_version = "trtmc.minimax-h3-windows-hot-summary/v1"
        status = "failed"
        inference_status = "failed_or_unverified"
        lifecycle_status = "unknown"
        worker_exit_code = $WorkerExitCode
        error = $RunError
        result_file = if (Test-Path -LiteralPath $ResultPath) { "worker-result.json" } else { $null }
        log_file = if (Test-Path -LiteralPath $LogPath) { "worker.log" } else { $null }
    }
    Write-JsonFile $FailureSummary $SummaryPath
    throw
} finally {
    $Receipt.status = if ($RunSucceeded) { "completed" } else { "failed" }
    $Receipt.ended_utc = [DateTimeOffset]::UtcNow.ToString("o")
    $Receipt.worker_exit_code = $WorkerExitCode
    $Receipt.telemetry = $TelemetryState
    if ($null -ne $RunError) {
        $Receipt.error = $RunError
    }
    Write-JsonFile $Receipt $ReceiptPath
}

Write-Host "MiniMax-H3 hot benchmark completed: $RunRoot"
Write-Host ("Median public pipeline call: {0:N3} ms ({1:N3} min)" -f `
    $Summary.median_ms, ($Summary.median_ms / 60000.0))
