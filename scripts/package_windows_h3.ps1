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
    [string]$OutputDirectory,

    # Save one bundle-sized allocation by atomically moving the input bundle
    # into the package. This is opt-in because it consumes BundlePath.
    [switch]$ConsumeBundle
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

if (-not ('TrtmcPackage.NativeMethods' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace TrtmcPackage {
    public static class NativeMethods {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr BeginUpdateResource(string fileName, bool deleteExisting);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool UpdateResource(IntPtr update, IntPtr type, IntPtr name,
                                                 ushort language, byte[] data, uint size);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool EndUpdateResource(IntPtr update, bool discard);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool MoveFileEx(string existingName, string newName, uint flags);
    }
}
'@
}

function Move-BundleIntoPackage {
    param([string]$Source, [string]$Destination)
    $parent = Split-Path -Parent $Destination
    [IO.Directory]::CreateDirectory($parent) | Out-Null
    if ([IO.File]::Exists($Destination)) {
        throw "Bundle package destination already exists: $Destination"
    }
    # Flags=0 is a rename-only Win32 move. It fails across volumes and never
    # falls back to copy-then-delete, so the source is consumed atomically.
    if (-not [TrtmcPackage.NativeMethods]::MoveFileEx($Source, $Destination, 0)) {
        $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw "Unable to atomically consume bundle. BundlePath and OutputDirectory must be on the same volume. Windows error $code; source='$Source'; destination='$Destination'"
    }
}

function Set-EmbeddedManifestResource {
    param([string]$SetupPath, [string]$ManifestPath)
    $bytes = [IO.File]::ReadAllBytes($ManifestPath)
    if ($bytes.Length -eq 0) {
        throw "Refusing to stamp an empty payload manifest: $ManifestPath"
    }
    $update = [TrtmcPackage.NativeMethods]::BeginUpdateResource($SetupPath, $false)
    if ($update -eq [IntPtr]::Zero) {
        $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw "Unable to open Setup for manifest resource stamping. Windows error $code; path='$SetupPath'"
    }
    $ended = $false
    try {
        $resourceType = [IntPtr]10 # RT_RCDATA
        $resourceName = [IntPtr]241 # kPayloadManifestResourceId
        if (-not [TrtmcPackage.NativeMethods]::UpdateResource(
                $update, $resourceType, $resourceName, 0, $bytes,
                [Convert]::ToUInt32($bytes.Length))) {
            $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            throw "Unable to stamp the payload manifest into Setup. Windows error $code; setup='$SetupPath'; manifest='$ManifestPath'"
        }
        if (-not [TrtmcPackage.NativeMethods]::EndUpdateResource($update, $false)) {
            $ended = $true
            $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            throw "Unable to commit the Setup manifest resource. Windows error $code; setup='$SetupPath'"
        }
        $ended = $true
    } catch {
        $primary = $_.Exception.Message
        if (-not $ended) {
            if (-not [TrtmcPackage.NativeMethods]::EndUpdateResource($update, $true)) {
                $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
                throw "$primary; resource rollback also failed with Windows error $code; setup='$SetupPath'"
            }
        }
        throw $primary
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
    $forbiddenProcessImports = @(
        '(?im)\bCreateProcess(?:A|W)?\b',
        '(?im)\bCreateProcessAsUser(?:A|W)?\b',
        '(?im)\bCreateProcessWithToken(?:A|W)?\b',
        '(?im)\bCreateProcessWithLogon(?:A|W)?\b',
        '(?im)\bNtCreateUserProcess\b',
        '(?im)\bRtlCreateUserProcess\b',
        '(?im)\bShellExecute(?:Ex)?(?:A|W)?\b',
        '(?im)\bWinExec\b',
        '(?im)\b_popen\b',
        '(?im)\b_(?:w)?spawn[a-z0-9_]*\b',
        '(?im)\bsystem\b'
    )
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

        # DLL-name checks alone would not catch a future direct process-launch
        # import from an otherwise allowed Windows system DLL. TensorRT-RTX is
        # the one vendor binary exception: it currently imports CreateProcessW,
        # but the locked runtime establishes and tests a one-process Job before
        # loading it. ModelConnect-owned executables and DLLs get no exception.
        $fileName = [IO.Path]::GetFileName($file).ToLowerInvariant()
        if ($fileName -notmatch '^tensorrt_rtx_[0-9_]+\.dll$') {
            $importsOutput = @(& dumpbin.exe /imports $file 2>&1)
            if ($LASTEXITCODE -ne 0) {
                throw "dumpbin import-symbol scan failed: $file"
            }
            $importsText = $importsOutput -join [Environment]::NewLine
            foreach ($pattern in $forbiddenProcessImports) {
                if ($importsText -match $pattern) {
                    throw "Forbidden process-launch import '$($Matches[0])' found in $file"
                }
            }
        }
    }
}

function Assert-CompleteMiniMaxH3Bundle {
    param(
        [string]$CliPath,
        [string]$BundleFile
    )
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

    # The runtime deliberately supports three authenticated denoiser layouts:
    # the production singular FirstBlockCache path, the legacy segmented
    # FastH3 VSA path, and legacy monolithic dense bundles.
    # Keep the package boundary exact for each layout instead of weakening it
    # to a broad prefix check.
    $commonSections = @(
        'text_encoder_plan',
        'vae_tile_decoder_plan',
        'audio_vae_decoder_plan'
    )
    $singularFirstBlockCache = @(
        $commonSections
        'adaln_precompute_plan'
        'denoiser_head_plan'
        'denoiser_tail_plan'
        'denoiser_finish_plan'
    )
    $legacyMonolithicDense = @(
        $commonSections
        'adaln_precompute_plan'
        'denoiser_plan'
    )

    $segmentedVsa = [Collections.Generic.List[string]]::new()
    foreach ($name in $commonSections) {
        $segmentedVsa.Add($name)
    }
    $segmentedVsa.Add('adaln_precompute_plan')
    $segmentedVsa.Add('denoiser_entry_plan')
    for ($index = 0; $index -lt 49; ++$index) {
        $segmentedVsa.Add(('denoiser_transition_{0:D2}_plan' -f $index))
    }
    $segmentedVsa.Add('denoiser_finish_plan')

    $conditioningSections = @(
        'vision_encoder_plan',
        'fl2va_keyframe_vae_encoder_plan'
    )
    $ref2vaSections = @(
        'ref2va_denoiser_plan',
        'ref2va_adaln_precompute_plan',
        'ref2va_video_vae_encoder_plan',
        'ref2va_audio_vae_encoder_plan'
    )
    $presentConditioning = @(
        $conditioningSections | Where-Object { $_ -in $actualSections }
    )
    if ($presentConditioning.Count -ne 0 -and
        $presentConditioning.Count -ne $conditioningSections.Count) {
        throw "MiniMax-H3 FL2VA conditioning plan set must be all-or-none"
    }
    $presentRef2va = @($ref2vaSections | Where-Object { $_ -in $actualSections })
    if ($presentRef2va.Count -ne 0 -and $presentRef2va.Count -ne $ref2vaSections.Count) {
        throw "MiniMax-H3 Ref2VA plan set must be all-or-none"
    }
    if ($presentRef2va.Count -ne 0 -and
        $presentConditioning.Count -ne $conditioningSections.Count) {
        throw "MiniMax-H3 Ref2VA requires both shared conditioning plans"
    }

    $optionalSections = @($presentConditioning) + @($presentRef2va)
    $baseLayouts = [ordered]@{
        singular_first_block_cache = @($singularFirstBlockCache)
        segmented_vsa = @($segmentedVsa)
        legacy_monolithic_dense = @($legacyMonolithicDense)
    }
    $matchedLayout = $null
    foreach ($layout in $baseLayouts.GetEnumerator()) {
        $expectedSections = @($layout.Value) + $optionalSections
        $missing = @($expectedSections | Where-Object { $_ -notin $actualSections })
        $unexpected = @($actualSections | Where-Object { $_ -notin $expectedSections })
        if ($actualSections.Count -eq $expectedSections.Count -and
            $missing.Count -eq 0 -and $unexpected.Count -eq 0) {
            $matchedLayout = $layout.Key
            break
        }
    }
    if ($null -eq $matchedLayout) {
        $accepted = @(
            $baseLayouts.GetEnumerator() | ForEach-Object {
                "$($_.Key)=$(@($_.Value).Count + $optionalSections.Count)"
            }
        )
        throw "MiniMax-H3 bundle plan set is incomplete or unexpected. Accepted layouts=[$($accepted -join ', ')]; actual=[$($actualSections -join ', ')]"
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
$PackagedBundle = Join-Path $PayloadRoot "models\MiniMax-H3.bundle"
if ($ConsumeBundle) {
    Move-BundleIntoPackage $Bundle $PackagedBundle
    Write-Warning "BundlePath was atomically consumed. The only package-owned copy is now '$PackagedBundle'."
} else {
    # The release package owns independent bytes. Never hard-link the large
    # bundle to a mutable build artifact.
    Copy-PayloadFile $Bundle $PackagedBundle
}

foreach ($legalName in @("LICENSE", "NOTICE", "ASSET_LICENSES.md")) {
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

# Anchor the exact external manifest bytes into the package's outer Setup.
# Release Authenticode signing (and publication of the official Setup SHA-256)
# happens after this resource update.
$PackagedSetup = Join-Path $OutputRoot "MiniMaxH3Setup.exe"
Set-EmbeddedManifestResource $PackagedSetup $ManifestPath

# Validate through the exact SHA-256-manifested package layout. The locked
# runtime discovers only the backend next to trtmc.exe and the fixed
# bin/trtmc/models/minimax_h3 plugin path; no loader override is accepted.
$PackagedCli = Join-Path $PayloadRoot "bin\trtmc.exe"
Assert-RuntimeOnlyCli $PackagedCli
Assert-CompleteMiniMaxH3Bundle $PackagedCli $PackagedBundle

$PackagedPeFiles = @(
    Get-ChildItem -LiteralPath $OutputRoot -Recurse -File |
        Where-Object { $_.Extension -in @('.exe', '.dll') } |
        ForEach-Object { $_.FullName }
)
Assert-NativeDependencyBoundary $PackagedPeFiles

# Re-run the native installer verifier after every package validation step.
# MiniMaxH3Setup.exe uses the Windows GUI subsystem, so an ordinary PowerShell
# invocation can return before it exits. Explicitly wait for --verify-only --quiet
# and hide its non-interactive window before accepting the package.
$QuotedPayloadRoot = '"' + $PayloadRoot + '"'
$VerifyProcess = Start-Process `
    -FilePath $PackagedSetup `
    -ArgumentList @("--payload-dir", $QuotedPayloadRoot, "--verify-only", "--quiet") `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
if ($VerifyProcess.ExitCode -ne 0) {
    throw "Native installer rejected the exact package payload with exit code $($VerifyProcess.ExitCode)"
}

Write-Host "Native MiniMax-H3 installation layout created: $OutputRoot"
Write-Host "End users install by double-clicking MiniMaxH3Setup.exe."
