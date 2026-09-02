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
    assert "expectedSections.Count -ne 61" in package_script
    assert "Assert-RuntimeOnlyCli" in package_script
    assert "trtmc build" in package_script
    assert "--hf-python" in package_script

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"' in cmake


def test_minimax_h3_uses_bundle_owned_native_tokenizer() -> None:
    plugin = (H3_RUNTIME / "plugin.cpp").read_text(encoding="utf-8")
    assert 'find_section(bundle, "tokenizer.json")' in plugin
    assert "CreateBpeTokenizer" in plugin


def test_minimax_h3_exposes_video_not_image_generation() -> None:
    pipeline = (H3_RUNTIME / "pipeline.h").read_text(encoding="utf-8")
    assert "generate_video(" in pipeline
    assert "supports_image_generation" not in pipeline
    assert "generate_image(" not in pipeline
