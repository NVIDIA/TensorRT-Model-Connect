# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#Requires -Version 5.1

[CmdletBinding()]
param([string]$WorkspaceRoot)
$ErrorActionPreference = 'Stop'
if (-not $WorkspaceRoot) { $WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path }
$dependencyRoot = Join-Path $WorkspaceRoot 'dependencies'
New-Item -ItemType Directory -Force $dependencyRoot | Out-Null
$archive = Join-Path $dependencyRoot 'electron-v44.3.0-win32-x64.zip'
$electronRoot = Join-Path $dependencyRoot 'electron'
$expectedHash = '26bf9a617d58d81772b3d68305d59ee48272969c15083c06db634a77358a8d9d'
if (!(Test-Path -LiteralPath $archive)) {
    & curl.exe --fail --location --retry 3 --silent --show-error 'https://github.com/electron/electron/releases/download/v44.3.0/electron-v44.3.0-win32-x64.zip' --output "$archive.partial"
    if ($LASTEXITCODE -ne 0) { throw "Electron download failed (exit $LASTEXITCODE)." }
    if ((Get-FileHash -LiteralPath "$archive.partial" -Algorithm SHA256).Hash -ne $expectedHash) {
        throw 'Electron download checksum mismatch. The archive has not been extracted.'
    }
    Move-Item -LiteralPath "$archive.partial" -Destination $archive -Force
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Electron archive checksum mismatch. The app has not been assembled.'
}
if (!(Test-Path -LiteralPath (Join-Path $electronRoot 'electron.exe'))) {
    Expand-Archive -LiteralPath $archive -DestinationPath $electronRoot -Force
}
& (Join-Path $PSScriptRoot 'Package-App.ps1') -WorkspaceRoot $WorkspaceRoot -ElectronRoot $electronRoot
