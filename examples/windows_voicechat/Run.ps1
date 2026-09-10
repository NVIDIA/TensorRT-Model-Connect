# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [switch]$Rehearsal,
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
if (-not $WorkspaceRoot) { $WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path }
$WorkspaceRoot = [IO.Path]::GetFullPath($WorkspaceRoot)
$required = @('Nemotron Voice Lab\Nemotron Voice Lab.exe')
if (-not $Rehearsal) {
    $required += @('runtime\trtmc_voicechat_bridge.exe', 'models\nemotron-voicechat-rtx.bundle')
}
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $WorkspaceRoot $relative) -PathType Leaf)) {
        throw "Missing $relative under $WorkspaceRoot. Run Setup.ps1 with this WorkspaceRoot first."
    }
}
if ($ValidateOnly) {
    $components = if ($Rehearsal) { 'The rehearsal app is' } else { 'App, runtime and model bundle are' }
    Write-Host "$components present under $WorkspaceRoot. No application was started."
    return
}
$previousWorkspace = $env:VOICE_LAB_WORKSPACE
$previousElectronMode = $env:ELECTRON_RUN_AS_NODE
try {
    $env:VOICE_LAB_WORKSPACE = $WorkspaceRoot
    Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
    $appRoot = Join-Path $WorkspaceRoot 'Nemotron Voice Lab'
    Start-Process -FilePath (Join-Path $appRoot 'Nemotron Voice Lab.exe') -WorkingDirectory $appRoot -WindowStyle Normal
} finally {
    $env:VOICE_LAB_WORKSPACE = $previousWorkspace
    $env:ELECTRON_RUN_AS_NODE = $previousElectronMode
}
