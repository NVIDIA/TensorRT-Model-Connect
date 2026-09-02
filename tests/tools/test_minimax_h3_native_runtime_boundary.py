# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
H3_RUNTIME = ROOT / "src" / "runtime" / "models" / "minimax_h3"


def test_minimax_h3_runtime_has_no_sidecar_or_framework_escape_hatch() -> None:
    forbidden = {
        "python.h": "embedded Python",
        "fastvideo": "FastVideo sidecar",
        "triton": "Triton runtime",
        "ffmpeg": "FFmpeg runtime",
        "torch/": "LibTorch runtime",
        "torch::": "LibTorch runtime",
        "createprocess": "subprocess launch",
        "_popen": "subprocess launch",
        "popen(": "subprocess launch",
        "std::system": "shell launch",
        "system(": "shell launch",
    }
    violations: list[str] = []
    for path in sorted((*H3_RUNTIME.glob("*.cpp"), *H3_RUNTIME.glob("*.cu"), *H3_RUNTIME.glob("*.h"))):
        text = path.read_text(encoding="utf-8").lower()
        # Architecture comments may name a reference implementation while
        # documenting a native parity point. Only compiled source counts as a
        # runtime dependency, so remove C/C++ comments before scanning.
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"//[^\n]*", "", text)
        for needle, label in forbidden.items():
            if needle in text:
                violations.append(f"{path.relative_to(ROOT)}: {label} ({needle!r})")
    assert not violations, "MiniMax-H3 runtime boundary violations:\n" + "\n".join(violations)


def test_windows_h3_distribution_disables_optional_runtime_frameworks() -> None:
    build_script = (ROOT / "scripts" / "build_windows_h3.ps1").read_text(encoding="utf-8")
    required_flags = {
        "-DTRTMC_BUILD_BACKEND_RTX=ON",
        "-DCMAKE_CUDA_RUNTIME_LIBRARY=Static",
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
        "-DTRTMC_ENABLE_TRT=OFF",
        "-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF",
        "-DTRTMC_ENABLE_TVM_FFI=OFF",
        "-DTRTMC_RUNTIME_MODELS=minimax_h3",
        "-DTRTMC_DISTRIBUTABLE_BUILD=ON",
        "-DTRTMC_RUNTIME_ONLY_CLI=ON",
        "-DTRTMC_LOCKED_H3_RUNTIME=ON",
    }
    assert required_flags <= set(build_script.splitlines()) or all(
        flag in build_script for flag in required_flags
    )
    assert "cudart_static.lib" in build_script
    assert "cudart64_12.dll" not in build_script

    package_script = (ROOT / "scripts" / "package_windows_h3.ps1").read_text(
        encoding="utf-8"
    )
    for forbidden_import in (
        r"^python[0-9_]*\.dll$",
        r"^torch.*\.dll$",
        r"^cudart.*\.dll$",
        r"^msvcp[0-9_]*\.dll$",
        r"^vcruntime[0-9_]*\.dll$",
        r"^ucrtbase.*\.dll$",
        r"^api-ms-win-crt-.*\.dll$",
        r"^nvinfer.*\.dll$",
        r"^nvonnxparser.*\.dll$",
        r"^cublas.*\.dll$",
        r"^nvrtc.*\.dll$",
        "fastvideo",
        "triton",
    ):
        assert forbidden_import in package_script.lower()
    assert "unapproved runtime dependency" in package_script.lower()
    assert "--validate-runtime" in package_script
    inspect_invocation = re.search(
        r"\$inspectOutput\s*=\s*@\((.*?)\)\s*\n", package_script, re.DOTALL
    )
    assert inspect_invocation is not None
    for forbidden_override in (
        "--backend-dir",
        "--model-plugin-dir",
        "--kernel-bindings",
    ):
        assert forbidden_override not in inspect_invocation.group(1)
    assert 'Join-Path $PayloadRoot "bin\\trtmc.exe"' in package_script
    assert 'Join-Path $PayloadRoot "models\\MiniMax-H3.bundle"' in package_script
    assert "trtmc\\models\\minimax_h3\\trtmc_model_minimax_h3.dll" in package_script
    assert "--verify-only --quiet" in package_script
    assert "expectedSections.Count -ne 61" in package_script
    assert "Assert-RuntimeOnlyCli" in package_script
    assert "trtmc build" in package_script
    assert "--hf-python" in package_script

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"' in cmake
    assert "TRTMC_LOCKED_H3_RUNTIME requires exactly" in cmake


def test_locked_windows_h3_runtime_is_fail_closed() -> None:
    args = (ROOT / "src" / "cli" / "args.cpp").read_text(encoding="utf-8")
    for option in ("--backend-dir", "--model-plugin-dir", "--kernel-bindings"):
        assert f'{option} is disabled in the locked MiniMax-H3 runtime' in args

    lockdown = (
        ROOT / "src" / "runtime" / "platform" / "windows_process_lockdown.cpp"
    ).read_text(encoding="utf-8")
    for required in (
        "CreateJobObjectW",
        "SetInformationJobObject",
        "AssignProcessToJobObject",
        "JOB_OBJECT_LIMIT_ACTIVE_PROCESS",
        "ActiveProcessLimit = 1",
        "TRTMC_BACKEND_DIR",
        "TRTMC_MODEL_PLUGIN_DIR",
        "TRTMC_MODEL_PLUGIN_STRICT",
        "TRTMC_KERNEL_BINDINGS",
        "TRTMC_KERNEL_BINDINGS_PATH",
    ):
        assert required in lockdown
    for forbidden in ("CreateProcess", "ShellExecute", "system(", "popen("):
        assert forbidden not in lockdown

    backend_loader = (
        ROOT / "src" / "runtime" / "backend" / "backend_loader.cpp"
    ).read_text(encoding="utf-8")
    assert "locked MiniMax-H3 runtime rejects backend search path overrides" in backend_loader
    assert 'backend_name != "trt_rtx"' in backend_loader
    assert "return try_open_backend_dso(path, path, tried);" in backend_loader
    assert "internal::current_module_path().parent_path()" in backend_loader

    pipeline_factory = (
        ROOT / "src" / "runtime" / "registry" / "pipeline_factory.cpp"
    ).read_text(encoding="utf-8")
    early_guard = pipeline_factory.index(
        "locked MiniMax-H3 runtime requires engine_backend=trt_rtx before backend discovery"
    )
    standard_discovery = pipeline_factory.index("load_standard_trt_backend_for_bundle(")
    assert early_guard > standard_discovery
    backend_dispatch = pipeline_factory.index(
        'if (logical_backend == "trt_rtx")', early_guard
    )
    assert early_guard < backend_dispatch
    assert pipeline_factory.count("enforce_locked_h3_process_policy();") >= 3

    trt_version = (
        ROOT / "src" / "runtime" / "backend" / "trt_version.cpp"
    ).read_text(encoding="utf-8")
    assert "locked MiniMax-H3 runtime disables standard TensorRT discovery" in trt_version
    assert "locked MiniMax-H3 runtime disables standard TensorRT library loading" in trt_version
    assert "internal::enforce_locked_h3_process_policy();" in backend_loader

    plugin_loader = (
        ROOT / "src" / "runtime" / "registry" / "pipeline_plugin_loader.cpp"
    ).read_text(encoding="utf-8")
    assert "locked MiniMax-H3 runtime rejects model plugin search path overrides" in plugin_loader
    assert (
        'core_module.parent_path() / "trtmc" / "models" / "minimax_h3"'
        in plugin_loader
    )
    assert "internal::current_module_path()" in plugin_loader

    # The Job APIs add no third-party import; the package PE allowlist already
    # permits only Kernel32 for this policy implementation.
    package_script = (ROOT / "scripts" / "package_windows_h3.ps1").read_text(
        encoding="utf-8"
    )
    assert "'kernel32.dll' = $true" in package_script.lower()
    assert "unapproved runtime dependency" in package_script.lower()

    pipeline_factory = (
        ROOT / "src" / "runtime" / "registry" / "pipeline_factory.cpp"
    ).read_text(encoding="utf-8")
    assert "#if !defined(TRTMC_LOCKED_H3_RUNTIME)" in pipeline_factory
    assert "try_write_effective_config_next_to" in pipeline_factory

    cli_main = (ROOT / "src" / "cli" / "main.cpp").read_text(encoding="utf-8")
    assert "#if defined(TRTMC_LOCKED_H3_RUNTIME)" in cli_main
    assert "must not emit effective-config" in cli_main

    installer = (
        ROOT / "src" / "installer" / "windows_h3_installer.cpp"
    ).read_text(encoding="utf-8")
    assert "Payload contains a file absent from the manifest" in installer
    assert "MiniMax-H3 payload manifest contains an unexpected file" in installer
    assert "exactly one versioned TensorRT-RTX runtime DLL" in installer


def test_minimax_h3_uses_bundle_owned_native_tokenizer() -> None:
    plugin = (H3_RUNTIME / "plugin.cpp").read_text(encoding="utf-8")
    assert 'find_section(bundle, "tokenizer.json")' in plugin
    assert "CreateBpeTokenizer" in plugin


def test_minimax_h3_exposes_video_not_image_generation() -> None:
    pipeline = (H3_RUNTIME / "pipeline.h").read_text(encoding="utf-8")
    assert "generate_video(" in pipeline
    assert "supports_image_generation" not in pipeline
    assert "generate_image(" not in pipeline
