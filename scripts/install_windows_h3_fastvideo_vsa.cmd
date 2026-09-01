@REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
@REM SPDX-License-Identifier: Apache-2.0

@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows_h3_fastvideo_vsa.ps1" %*
set "INSTALL_EXIT_CODE=%ERRORLEVEL%"

if not "%INSTALL_EXIT_CODE%"=="0" (
  echo.
  echo FastVideo VSA installation failed. Review the message above.
) else (
  echo.
  echo FastVideo VSA installation completed successfully.
)

pause
exit /b %INSTALL_EXIT_CODE%
