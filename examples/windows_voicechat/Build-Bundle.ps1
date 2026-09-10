# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [string]$Python,
    [string]$ModelPath,
    [string]$OutputPath
)
$ErrorActionPreference = 'Stop'
if (-not $WorkspaceRoot) { $WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (!$Python) { $Python = Join-Path $WorkspaceRoot 'dependencies/python/Scripts/python.exe' }
if (!$ModelPath) { $ModelPath = Join-Path $WorkspaceRoot 'models/Nemotron-VoiceChat-11B' }
if (!$OutputPath) { $OutputPath = Join-Path $WorkspaceRoot 'models/nemotron-voicechat-rtx.bundle' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python is missing. Run Setup.ps1 -Stage Python,Dependencies first.' }
if (-not (Test-Path -LiteralPath (Join-Path $ModelPath 'model.safetensors') -PathType Leaf)) {
    throw 'The VoiceChat checkpoint is missing. Run Setup.ps1 -Stage Model after installing the Python dependencies.'
}
New-Item -ItemType Directory -Force -Path (Split-Path ([IO.Path]::GetFullPath($OutputPath)) -Parent) | Out-Null
$previousPythonPath = $env:PYTHONPATH
$previousHfHome = $env:HF_HOME
try {
    $env:PYTHONPATH = "$(Join-Path $repoRoot 'core/builder');$repoRoot"
    $env:HF_HOME = Join-Path $WorkspaceRoot 'models/huggingface'
    & $Python -m tensorrt_model_connect build $ModelPath --backend trt_rtx --precision fp32 --quantization int8 --max-sequence-length 512 --output $OutputPath
    if ($LASTEXITCODE -ne 0) { throw "TensorRT-RTX bundle build failed (exit $LASTEXITCODE)." }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:HF_HOME = $previousHfHome
}
