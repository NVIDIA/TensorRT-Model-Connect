# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
Creates the double-click native Windows MiniMax-H3 installation layout.

.DESCRIPTION
This is a release-engineering step, not part of video generation. The resulting
layout contains MiniMaxH3Setup.exe, a SHA-256 manifest, and a payload made only
of ModelConnect, the MiniMax-H3 bundle, and TensorRT-RTX. The setup executable
uses Win32 APIs and does not invoke PowerShell or any subprocess at install or
generation time.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BuildDirectory,

    [Parameter(Mandatory = $true)]
    [string]$BundlePath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RequiredFile {
    param([string]$Path, [string]$Label)
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "$Label is not a regular file: $Path"
    }
    return $resolved
}

function Copy-PayloadFile {
    param([string]$Source, [string]$Destination)
    $parent = Split-Path -Parent $Destination
    [IO.Directory]::CreateDirectory($parent) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
}

function Link-OrCopyLargeFile {
    param([string]$Source, [string]$Destination)
    $parent = Split-Path -Parent $Destination
    [IO.Directory]::CreateDirectory($parent) | Out-Null
    try {
        New-Item -ItemType HardLink -Path $Destination -Target $Source -ErrorAction Stop | Out-Null
    } catch {
        Copy-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
    }
}

function Assert-NativeDependencyBoundary {
    param([string[]]$PeFiles)
    if (-not (Get-Command dumpbin.exe -ErrorAction SilentlyContinue)) {
        throw "dumpbin.exe is required; run from an x64 Visual Studio developer PowerShell"
    }
    $forbidden = @(
        '^python[0-9_]*\.dll$',
        '^torch.*\.dll$',
        '^cudart.*\.dll$',
        '^msvcp[0-9_]*\.dll$',
        '^vcruntime[0-9_]*\.dll$',
        '^ucrtbase.*\.dll$',
        '^api-ms-win-crt-.*\.dll$',
        '^av(codec|device|filter|format|util).*\.dll$',
        '^sw(resample|scale).*\.dll$',
        '^nvinfer.*\.dll$',
        '^nvonnxparser.*\.dll$',
        '^cublas.*\.dll$',
        '^cufft.*\.dll$',
        '^curand.*\.dll$',
        '^cusolver.*\.dll$',
        '^cusparse.*\.dll$',
        '^nvrtc.*\.dll$',
        'fastvideo',
        'triton'
    )

    # ModelConnect is built /MT with the static CUDA runtime. These are the
    # only non-system direct imports permitted in the installed runtime.
    $allowedLocal = @{
        'trtmc_core.dll' = $true
    }
    $allowedWindows = @{
        'advapi32.dll' = $true
        'bcrypt.dll' = $true
        'combase.dll' = $true
        'crypt32.dll' = $true
        'd3d11.dll' = $true
        'dxgi.dll' = $true
        'gdi32.dll' = $true
        'kernel32.dll' = $true
        'mf.dll' = $true
        'mfplat.dll' = $true
        'mfreadwrite.dll' = $true
        'mfuuid.dll' = $true
        'ntdll.dll' = $true
        'ole32.dll' = $true
        'oleaut32.dll' = $true
        'propsys.dll' = $true
        'rpcrt4.dll' = $true
        'secur32.dll' = $true
        'setupapi.dll' = $true
        'shell32.dll' = $true
        'shlwapi.dll' = $true
        'user32.dll' = $true
        'userenv.dll' = $true
        'version.dll' = $true
        'winmm.dll' = $true
        'windowscodecs.dll' = $true
        'ws2_32.dll' = $true
    }
    foreach ($file in $PeFiles) {
        $dumpbinOutput = @(& dumpbin.exe /dependents $file 2>&1)
        if ($LASTEXITCODE -ne 0) {
            throw "dumpbin dependency scan failed: $file"
        }
        $imports = @(
            $dumpbinOutput |
                ForEach-Object { ([string]$_).Trim().ToLowerInvariant() } |
                Where-Object { $_ -match '^[a-z0-9_.-]+\.dll$' } |
                Sort-Object -Unique
        )
        foreach ($import in $imports) {
            foreach ($pattern in $forbidden) {
                if ($import -match $pattern) {
                    throw "Forbidden runtime dependency '$import' found in $file"
                }
            }
            if ($allowedLocal.ContainsKey($import) -or
                $allowedWindows.ContainsKey($import) -or
                $import -eq 'nvcuda.dll' -or
                $import -match '^tensorrt_rtx_[0-9_]+\.dll$') {
                continue
            }
            throw "Unapproved runtime dependency '$import' found in $file"
        }
    }
}

function Assert-CompleteMiniMaxH3Bundle {
    param(
        [string]$CliPath,
        [string]$BundleFile
    )
    $expectedSections = [Collections.Generic.List[string]]::new()
    foreach ($name in @(
        'text_encoder_plan',
        'vision_encoder_plan',
        'adaln_precompute_plan',
        'denoiser_entry_plan'
    )) {
        $expectedSections.Add($name)
    }
    for ($index = 0; $index -lt 49; ++$index) {
        $expectedSections.Add(('denoiser_transition_{0:D2}_plan' -f $index))
    }
    foreach ($name in @(
        'denoiser_finish_plan',
        'fl2va_keyframe_vae_encoder_plan',
        'vae_tile_decoder_plan',
        'audio_vae_decoder_plan',
        'ref2va_denoiser_plan',
        'ref2va_adaln_precompute_plan',
        'ref2va_video_vae_encoder_plan',
        'ref2va_audio_vae_encoder_plan'
    )) {
        $expectedSections.Add($name)
    }
    if ($expectedSections.Count -ne 61) {
        throw "Internal packaging error: expected MiniMax-H3 plan count is not 61"
    }

    $inspectOutput = @(& $CliPath inspect $BundleFile --validate-runtime --list-engines 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "ModelConnect rejected the complete MiniMax-H3 bundle:`n$($inspectOutput -join [Environment]::NewLine)"
    }
    $inspectText = $inspectOutput -join [Environment]::NewLine
    if ($inspectText -notmatch '(?m)^Runtime validation:\s+passed \(MiniMaxH3Pipeline\)\s*$') {
        throw "The native MiniMax-H3 runtime plugin did not validate the bundle"
    }
    $actualSections = @(
        $inspectOutput |
            ForEach-Object { [string]$_ } |
            Where-Object { $_ -match '^([^\s]+)\s+[0-9]+(?:\.[0-9]+)? MB\s+' } |
            ForEach-Object { $Matches[1] } |
            Sort-Object -Unique
    )
    $missing = @($expectedSections | Where-Object { $_ -notin $actualSections })
    $unexpected = @($actualSections | Where-Object { $_ -notin $expectedSections })
    if ($actualSections.Count -ne 61 -or $missing.Count -ne 0 -or $unexpected.Count -ne 0) {
        throw "MiniMax-H3 bundle plan set is incomplete or unexpected. Missing=[$($missing -join ', ')]; unexpected=[$($unexpected -join ', ')]"
    }
}

function Assert-RuntimeOnlyCli {
    param([string]$CliPath)
    $helpOutput = @(& $CliPath --help 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to attest the runtime-only ModelConnect CLI"
    }
    $helpText = $helpOutput -join [Environment]::NewLine
    foreach ($required in @('trtmc generate-video', 'trtmc inspect', 'trtmc version')) {
        if (-not $helpText.Contains($required)) {
            throw "ModelConnect CLI is missing required runtime command: $required"
        }
    }
    foreach ($forbidden in @(
        'trtmc build',
        'trtmc graph',
        'trtmc run',
        'trtmc generate-audio',
        '--hf-python',
        '--backend-dir',
        '--model-plugin-dir',
        '--kernel-bindings',
        'Build uses a sibling Python interpreter'
    )) {
        if ($helpText.Contains($forbidden)) {
            throw "Full/development ModelConnect CLI cannot be packaged: found '$forbidden'"
        }
    }
}

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = (Resolve-Path -LiteralPath $BuildDirectory -ErrorAction Stop).Path
$Bundle = Resolve-RequiredFile $BundlePath "MiniMax-H3 bundle"
$OutputRoot = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $OutputRoot) {
    throw "OutputDirectory already exists; select a new empty path: $OutputRoot"
}

$Cli = Resolve-RequiredFile (Join-Path $BuildRoot "trtmc.exe") "runtime-only CLI"
$Core = Resolve-RequiredFile (Join-Path $BuildRoot "trtmc_core.dll") "ModelConnect core"
$Backend = Resolve-RequiredFile (Join-Path $BuildRoot "trtmc_backend_trt_rtx.dll") `
    "TensorRT-RTX backend"
$ModelPlugin = Resolve-RequiredFile `
    (Join-Path $BuildRoot "trtmc\models\minimax_h3\trtmc_model_minimax_h3.dll") `
    "MiniMax-H3 model plugin"
$Setup = Resolve-RequiredFile (Join-Path $BuildRoot "MiniMaxH3Setup.exe") `
    "native MiniMax-H3 setup executable"
$RtxRuntimeCandidates = @(Get-ChildItem -LiteralPath $BuildRoot -File -Filter "tensorrt_rtx_*.dll")
if ($RtxRuntimeCandidates.Count -ne 1) {
    throw "Expected exactly one TensorRT-RTX runtime DLL in $BuildRoot"
}
$RtxRuntime = $RtxRuntimeCandidates[0].FullName

Assert-RuntimeOnlyCli $Cli
Assert-NativeDependencyBoundary @($Cli, $Core, $Backend, $ModelPlugin, $Setup, $RtxRuntime)

$PayloadRoot = Join-Path $OutputRoot "payload"
[IO.Directory]::CreateDirectory($PayloadRoot) | Out-Null
Copy-PayloadFile $Setup (Join-Path $OutputRoot "MiniMaxH3Setup.exe")
Copy-PayloadFile $Setup (Join-Path $PayloadRoot "UninstallMiniMaxH3.exe")
Copy-PayloadFile $Cli (Join-Path $PayloadRoot "bin\trtmc.exe")
Copy-PayloadFile $Core (Join-Path $PayloadRoot "bin\trtmc_core.dll")
Copy-PayloadFile $Backend (Join-Path $PayloadRoot "bin\trtmc_backend_trt_rtx.dll")
Copy-PayloadFile $RtxRuntime (Join-Path $PayloadRoot ("bin\" + [IO.Path]::GetFileName($RtxRuntime)))
Copy-PayloadFile $ModelPlugin `
    (Join-Path $PayloadRoot "bin\trtmc\models\minimax_h3\trtmc_model_minimax_h3.dll")
Link-OrCopyLargeFile $Bundle (Join-Path $PayloadRoot "models\MiniMax-H3.bundle")

foreach ($legalName in @("LICENSE", "NOTICE")) {
    $legalPath = Join-Path $RepositoryRoot $legalName
    if (Test-Path -LiteralPath $legalPath -PathType Leaf) {
        Copy-PayloadFile $legalPath (Join-Path $PayloadRoot ("licenses\" + $legalName))
    }
}

$Utf8NoBom = [Text.UTF8Encoding]::new($false)
$MarkerPath = Join-Path $PayloadRoot ".minimax-h3-install-id"
[IO.File]::WriteAllText($MarkerPath, "trtmc-minimax-h3-native-install-v1`n", $Utf8NoBom)

$Rows = [Collections.Generic.List[string]]::new()
$PayloadPrefixLength = $PayloadRoot.TrimEnd('\').Length + 1
$PayloadFiles = @(Get-ChildItem -LiteralPath $PayloadRoot -Recurse -File | `
    Sort-Object { $_.FullName.Substring($PayloadPrefixLength).Replace('\', '/') })
foreach ($file in $PayloadFiles) {
    $relative = $file.FullName.Substring($PayloadPrefixLength).Replace('\', '/')
    $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $Rows.Add("$digest`t$($file.Length)`t$relative")
}
$ManifestPath = Join-Path $OutputRoot "payload.manifest"
[IO.File]::WriteAllText($ManifestPath, (($Rows -join "`n") + "`n"), $Utf8NoBom)

# Validate through the exact SHA-256-manifested package layout. The locked
# runtime discovers only the backend next to trtmc.exe and the fixed
# bin/trtmc/models/minimax_h3 plugin path; no loader override is accepted.
$PackagedCli = Join-Path $PayloadRoot "bin\trtmc.exe"
$PackagedBundle = Join-Path $PayloadRoot "models\MiniMax-H3.bundle"
Assert-RuntimeOnlyCli $PackagedCli
Assert-CompleteMiniMaxH3Bundle $PackagedCli $PackagedBundle

$PackagedPeFiles = @(
    Get-ChildItem -LiteralPath $OutputRoot -Recurse -File |
        Where-Object { $_.Extension -in @('.exe', '.dll') } |
        ForEach-Object { $_.FullName }
)
Assert-NativeDependencyBoundary $PackagedPeFiles

# Re-run the native installer verifier after every package validation step.
# It rejects any payload file that is not represented by the SHA-256 manifest.
$PackagedSetup = Join-Path $OutputRoot "MiniMaxH3Setup.exe"
$VerifyOutput = @(& $PackagedSetup --payload-dir $PayloadRoot --verify-only --quiet 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "Native installer rejected the exact package payload:`n$($VerifyOutput -join [Environment]::NewLine)"
}

Write-Host "Native MiniMax-H3 installation layout created: $OutputRoot"
Write-Host "End users install by double-clicking MiniMaxH3Setup.exe."
