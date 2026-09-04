# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
H3_RUNTIME = ROOT / "src" / "runtime" / "models" / "minimax_h3"


def test_runtime_has_no_external_framework_or_subprocess() -> None:
    forbidden = {
        "python.h": "embedded Python",
        "fastvideo": "FastVideo",
        "triton": "Triton",
        "ffmpeg": "FFmpeg",
        "torch/": "LibTorch",
        "torch::": "LibTorch",
        "createprocess": "subprocess",
        "_popen": "subprocess",
        "popen(": "subprocess",
        "std::system": "shell",
        "system(": "shell",
    }
    violations: list[str] = []
    sources = (*H3_RUNTIME.glob("*.cpp"), *H3_RUNTIME.glob("*.cu"), *H3_RUNTIME.glob("*.h"))
    for path in sorted(sources):
        text = path.read_text(encoding="utf-8").lower()
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"//[^\n]*", "", text)
        for needle, label in forbidden.items():
            if needle in text:
                violations.append(f"{path.relative_to(ROOT)}: {label}")
    assert not violations, "\n".join(violations)


def test_windows_helper_builds_only_native_rtx_runtime() -> None:
    script = (H3_RUNTIME / "build_windows.ps1").read_text(encoding="utf-8")
    for flag in (
        "-DTRTMC_BUILD_BACKEND_RTX=ON",
        "-DTRTMC_BUILD_BACKEND_TRT=OFF",
        "-DTRTMC_ENABLE_TRT=OFF",
        "-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF",
        "-DTRTMC_ENABLE_TVM_FFI=OFF",
        "-DTRTMC_BUILD_DIFFUSION_KERNELS=OFF",
        "-DTRTMC_RUNTIME_MODELS=minimax_h3",
        "-DTRTMC_RUNTIME_ONLY_CLI=ON",
    ):
        assert flag in script
    assert "cudart_static.lib" in script


def test_runtime_owns_tokenization_and_video_generation() -> None:
    plugin = (H3_RUNTIME / "plugin.cpp").read_text(encoding="utf-8")
    pipeline = (H3_RUNTIME / "pipeline.h").read_text(encoding="utf-8")
    assert 'find_section(bundle, "tokenizer.json")' in plugin
    assert "CreateBpeTokenizer" in plugin
    assert "generate_video(" in pipeline
    assert "generate_image(" not in pipeline


def test_documented_speed_command_keeps_qualified_settings() -> None:
    guide = (ROOT / "website" / "docs" / "getting-started" / "windows-native-h3.md").read_text(
        encoding="utf-8"
    )
    for setting in (
        "--num-frames 120 --height 768 --width 1344 --seed 0",
        '--set "minimax_h3.retain_engines=true"',
        '--set "minimax_h3.retained_tail_weight_budget_gib=24"',
        '--set "minimax_h3.first_block_cache_threshold=0.30"',
        "--warmup 1 --benchmark 1",
        "555,000 ms",
    ):
        assert setting in guide
