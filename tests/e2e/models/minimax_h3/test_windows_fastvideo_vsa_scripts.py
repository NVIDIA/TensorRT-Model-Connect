# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
GITATTRIBUTES = REPOSITORY_ROOT / ".gitattributes"
PRE_COMMIT_CONFIG = REPOSITORY_ROOT / ".pre-commit-config.yaml"
PROFILE_PATH = Path(__file__).with_name("fastvideo_windows_vsa_profile.json")
BENCHMARK_PATH = Path(__file__).with_name("fastvideo_windows_vsa_benchmark.json")
INSTALLER = REPOSITORY_ROOT / "scripts" / "install_windows_h3_fastvideo_vsa.ps1"
DOUBLE_CLICK_INSTALLER = REPOSITORY_ROOT / "scripts" / "install_windows_h3_fastvideo_vsa.cmd"
RUNNER = REPOSITORY_ROOT / "scripts" / "run_windows_h3_fastvideo_vsa.ps1"
REQUIREMENTS = REPOSITORY_ROOT / "requirements" / "windows-h3-fastvideo-vsa.txt"
DOCUMENTATION = REPOSITORY_ROOT / "website" / "docs" / "getting-started" / "windows-fastvideo-h3-vsa.md"
SIDEBAR = REPOSITORY_ROOT / "website" / "sidebars.js"
PATCH_PLACEHOLDER = "__FASTVIDEO_WINDOWS_H3_VSA_PATCH_SHA256__"


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _benchmark() -> dict:
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def test_fastvideo_windows_vsa_profile_is_public_and_revision_bound() -> None:
    profile = _profile()

    assert profile["schema_version"] == 1
    assert profile["backend"] == "fastvideo-pytorch-triton-windows"
    assert profile["support_status"] == "experimental-reproduction"
    assert profile["hardware"] == {
        "operating_system": "Windows 11",
        "minimum_os_build": 22000,
        "tested_os_build": 28000,
        "os_architecture": "Arm64",
        "process_architecture": "X64",
        "python_architecture": "AMD64",
        "python_bits": 64,
        "gpu_count": 1,
        "compute_capability": "12.1",
    }
    assert profile["fastvideo"]["repository"] == "https://github.com/hao-ai-lab/FastVideo.git"
    assert profile["fastvideo"]["revision"] == "3d8ac9d14bd697a89ede8f170cbfbca012a9edcc"
    assert profile["fastvideo"]["kernel_python_path"] == "fastvideo-kernel/python"
    assert profile["base_model"]["repository"] == "MiniMaxAI/MiniMax-H3"
    assert profile["base_model"]["revision"] == "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
    assert profile["adapter"]["repository"] == "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA"
    assert re.fullmatch(r"[0-9a-f]{40}", profile["adapter"]["revision"])
    assert profile["adapter"]["file"] == "vsa-datafree/adapter_model.safetensors"
    assert profile["adapter"]["size_bytes"] == 5_339_117_712
    assert profile["adapter"]["sha256"] == "42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470"


def test_fastvideo_patch_contract_is_fail_closed_until_finalized() -> None:
    profile = _profile()
    attributes = GITATTRIBUTES.read_text(encoding="utf-8")
    pre_commit = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    patch_relative = Path(profile["fastvideo"]["patch_path"])
    assert not patch_relative.is_absolute()
    assert patch_relative.as_posix() == "third_party/fastvideo/windows-h3-vsa.patch"
    assert "third_party/fastvideo/windows-h3-vsa.patch text eol=lf -whitespace" in attributes
    assert "^third_party/fastvideo/windows-h3-vsa\\.patch$" in pre_commit
    assert profile["fastvideo"]["patch_expected_paths"] == sorted(profile["fastvideo"]["patch_expected_paths"])
    assert len(profile["fastvideo"]["patch_expected_paths"]) == 14

    cuda_toolchain = profile["cuda_toolchain"]
    assert cuda_toolchain["release"] == "12.9.1"
    assert cuda_toolchain["nvcc_version"] == "12.9.86"
    assert cuda_toolchain["archive_url"].startswith(
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/"
    )
    assert cuda_toolchain["archive_size_bytes"] == 126_917_884
    assert cuda_toolchain["archive_sha256"] == (
        "227b109663b5e57d2718bcabb24a4ba0d9d4e52d958e327dc476f7c28691be85"
    )
    assert cuda_toolchain["ptxas_relative_path"] == "bin/ptxas.exe"
    assert cuda_toolchain["ptxas_size_bytes"] == 26_767_872
    assert cuda_toolchain["ptxas_sha256"] == (
        "bb89f22b3afe5d63141c5d83838c7aff9df66171a02d3c501376c8d7904eae0b"
    )

    expected_sha = profile["fastvideo"]["patch_sha256"]
    if expected_sha == PATCH_PLACEHOLDER:
        return

    assert re.fullmatch(r"[0-9a-f]{64}", expected_sha)
    patch_path = REPOSITORY_ROOT / patch_relative
    assert patch_path.is_file()
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == expected_sha


def test_fastvideo_workload_is_one_native_four_forward_request() -> None:
    workload = _profile()["workload"]

    assert workload["prompt"] == {
        "minimum_tokens": 1,
        "maximum_tokens": 1024,
        "default_prompt_tokens": 537,
    }
    length = workload["video_length"]
    assert length == {
        "minimum_requested_seconds": 5.0,
        "maximum_requested_seconds": 15.0,
        "default_num_frames": 124,
        "minimum_num_frames": 124,
        "maximum_num_frames": 345,
        "frame_alignment": 17,
        "frame_offset": 5,
        "validated_num_frames": [124, 345],
    }
    supported_frames = list(
        range(length["minimum_num_frames"], length["maximum_num_frames"] + 1, length["frame_alignment"])
    )
    assert supported_frames == [124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345]
    assert all((frames - length["frame_offset"]) % length["frame_alignment"] == 0 for frames in supported_frames)
    assert workload["height"] == 768
    assert workload["width"] == 1344
    assert workload["fps"] == 24
    assert workload["seed"] == 0
    assert workload["scheduler_grid_points"] == workload["transformer_forwards"] + 1 == 5
    assert workload["warmup_requests"] == 0
    assert workload["measured_requests"] == 1
    assert workload["attention_backend"] == "VIDEO_SPARSE_ATTN_H3"
    assert workload["vsa_sparsity"] == 0.9
    assert workload["vsa_tile_size"] == 64
    assert workload["vsa_kernel"] == "triton"
    assert workload["fa4"] is False
    assert workload["video_decode_backend"] == "h3-vae"
    assert workload["long_sequence_memory_policy"] == "no-grad-inplace-gate-merge"
    assert workload["cuda_allocator"] == "platform-default"
    assert workload["execution"] == {
        "profile": "all",
        "num_gpus": 1,
        "execution_backend": "mp",
        "regional_torch_compile": True,
        "whole_model_torch_compile": False,
        "compile_vae": True,
        "parallel_vae": False,
        "replicated_dit": True,
        "h3_sequential_load": True,
        "lazy_module_load": True,
        "pin_cpu_memory": True,
    }


@pytest.mark.parametrize(
    ("requested_seconds", "expected_frames"),
    ((5.0, 124), (8.0, 192), (10.0, 243), (14.375, 345), (15.0, 345)),
)
def test_nominal_duration_resolves_to_a_supported_native_bucket(
    requested_seconds: float,
    expected_frames: int,
) -> None:
    workload = _profile()["workload"]
    length = workload["video_length"]
    requested_frames = math.ceil(requested_seconds * workload["fps"])
    steps = math.ceil((requested_frames - length["frame_offset"]) / length["frame_alignment"])
    aligned = length["frame_offset"] + length["frame_alignment"] * steps

    assert min(aligned, length["maximum_num_frames"]) == expected_frames


def test_validated_benchmark_is_sanitized_and_profile_bound() -> None:
    profile = _profile()
    benchmark = _benchmark()

    assert benchmark["schema_version"] == 1
    assert benchmark["status"] == "completed"
    assert benchmark["profile_id"] == profile["profile_id"]
    assert benchmark["hardware"]["compute_capability"] == profile["hardware"]["compute_capability"]
    assert benchmark["hardware"]["os_build"] == profile["hardware"]["tested_os_build"]
    assert benchmark["hardware"]["os_architecture"] == profile["hardware"]["os_architecture"]
    assert benchmark["hardware"]["process_architecture"] == profile["hardware"]["process_architecture"]
    assert benchmark["software"]["fastvideo_revision"] == profile["fastvideo"]["revision"]
    assert benchmark["software"]["fastvideo_patch_sha256"] == profile["fastvideo"]["patch_sha256"]
    assert benchmark["software"]["python_version"] == profile["python"]["validated_version"]
    assert benchmark["software"]["ptxas_sha256"] == profile["cuda_toolchain"]["ptxas_sha256"]
    assert benchmark["software"]["base_model_revision"] == profile["base_model"]["revision"]
    assert benchmark["software"]["adapter_revision"] == profile["adapter"]["revision"]
    assert benchmark["workload"]["execution"] | {
        "warmup_requests": 0,
        "measured_requests": 1,
    } == profile["workload"]["execution"] | {
        "warmup_requests": profile["workload"]["warmup_requests"],
        "measured_requests": profile["workload"]["measured_requests"],
    }

    timing = benchmark["timing_seconds"]
    stage_sum = (
        timing["conditioning"]
        + timing["denoising_and_cold_regional_compile"]
        + timing["full_h3_video_vae"]
        + timing["audio_decode"]
    )
    assert timing["stage_sum"] == pytest.approx(stage_sum, abs=1e-6)
    assert benchmark["target_comparison"]["request_wall_minutes"] == pytest.approx(
        timing["request_wall"] / 60,
        abs=1e-6,
    )
    assert timing["denoising_and_cold_regional_compile"] > 600
    assert benchmark["target_comparison"]["achieved"] is False

    media = benchmark["media"]
    assert media["sha256"] == "4e48850904ce696e97ec9ed5b05f9d6d6ac2628e29018d2d34a6ddee061670e8"
    assert (media["video_frames"], media["video_width"], media["video_height"], media["video_fps"]) == (
        345,
        1344,
        768,
        24,
    )
    assert media["audio_channels"] == 2
    assert media["audio_sample_rate_hz"] == 32000


def test_installer_pins_source_patch_and_inference_dependencies() -> None:
    installer = INSTALLER.read_text(encoding="utf-8")
    requirements = REQUIREMENTS.read_text(encoding="utf-8")

    for required in (
        "SupportsShouldProcess = $true",
        "fastvideo_windows_vsa_profile.json",
        "The FastVideo patch SHA-256 placeholder must be replaced before installation",
        '"apply", "--check", "--index"',
        '"apply", "--index"',
        '"diff", "--cached", "--check"',
        "patch_expected_paths",
        '"--requirement", $RequirementsPath',
        '"--no-build-isolation", "--no-deps", "--editable", $FastVideoRoot',
        'SetEnvironmentVariable("PYTHONPATH", $KernelPythonRoot, "Process")',
        'SetEnvironmentVariable("TRITON_PTXAS.EXE_PATH", $PtxasPath, "Process")',
        "Invoke-PinnedDownload",
        "cuda_toolchain.archive_sha256",
        "Triton did not select CUDA 12.9 ptxas",
        "Triton did not select the pinned ptxas path",
        "The extracted ptxas.exe does not match the pinned size and SHA-256",
        "The validated cohort requires Arm64 Windows with an x64 PowerShell process",
        "The validated cohort requires 64-bit x64 Python running under Windows emulation",
        "The validated profile requires exact Python",
        '"PROCESSOR_ARCHITECTURE"',
        '"Machine"',
        "from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton",
        "from fastvideo_kernel.triton_kernels.index import map_to_index",
        "InstallRoot must be outside the TensorRT-Model-Connect source checkout",
    ):
        assert required in installer

    assert "Invoke-Expression" not in installer
    assert "Get-FileHash" not in installer
    assert "RuntimeInformation]::OSArchitecture" not in installer
    assert "Remove-Item" not in installer
    assert "reset --hard" not in installer
    assert "file:///" not in requirements
    package_lines = [line for line in requirements.splitlines() if line and not line.startswith("#")]
    assert package_lines
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s]+", line) for line in package_lines)
    for externally_pinned in ("torch==", "torchvision==", "triton-windows==", "fastvideo=="):
        assert not any(line.lower().startswith(externally_pinned) for line in package_lines)


def test_runner_uses_only_the_single_gpu_windows_vsa_profile() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    for fixed_argument in (
        '"--num-gpus", [string]$Execution.num_gpus',
        '"--execution-backend", [string]$Execution.execution_backend',
        '"--vsa"',
        '"--vsa-kernel", "triton"',
        '"--no-fa4"',
        '"--video-decode-backend", "h3-vae"',
        '"--h3-sequential-load"',
        '"--lazy-module-load"',
        '"--compile-vae"',
        '"--no-parallel-vae"',
        '"--inference-torch-compile"',
        '"--no-warmup"',
        '"--repeats", "1"',
    ):
        assert fixed_argument in runner

    assert "[ValidateSet(124, 345)]" not in runner
    assert 'throw "Prompt and PromptFile are mutually exclusive"' in runner
    assert 'ContainsKey("Prompt")' in runner
    assert 'ContainsKey("PromptFile")' in runner
    assert "Get-Content -Raw -Encoding UTF8 -LiteralPath $PromptPath" in runner
    assert "$PromptSpec.prompt -isnot [string]" in runner
    assert "from transformers import AutoTokenizer" in runner
    assert "supports 1 to 1024 prompt tokens without truncation" in runner
    assert "$PromptPayloadPath" in runner
    assert '[IO.File]::Delete($PromptPayloadPath)' in runner
    assert 'read_text(encoding="utf-8")' in runner
    assert "runpy.run_path" in runner
    assert 'f"--prompt={prompt}"' in runner
    assert "$PromptBase64" not in runner
    assert '"--prompt", $PromptText' not in runner
    assert '-Include "processor/*"' in runner
    assert runner.index("supports 1 to 1024 prompt tokens without truncation") < runner.index(
        "Resolving the authorized pinned MiniMax H3 snapshot."
    )
    assert 'ContainsKey("DurationSeconds")' in runner
    assert "NumFrames must be a native 17n+5 value from 124 through 345" in runner
    assert "caps this request at 345 frames (14.375 seconds)" in runner
    assert "requested_duration_seconds" in runner
    assert "aligned_duration_seconds" in runner
    assert "prompt_tokens" in runner
    assert "prompt_sha256" in runner
    assert 'Set-ProcessEnvironment "TRITON_PTXAS.EXE_PATH" $PtxasPath' in runner
    assert 'Set-ProcessEnvironment "TRITON_PTXAS_PATH"' not in runner
    assert 'Set-ProcessEnvironment "FASTVIDEO_VSA_TRITON" "1"' in runner
    assert "The validated cohort requires Arm64 Windows with an x64 PowerShell process" in runner
    assert '"PROCESSOR_ARCHITECTURE"' in runner
    assert '"Machine"' in runner
    assert "RuntimeInformation]::OSArchitecture" not in runner
    assert '"python_architecture": platform.machine()' in runner
    assert '"python_bits": struct.calcsize("P") * 8' in runner
    assert '"ptxas_path": knobs.nvidia.ptxas.path' in runner
    assert "The checked-in execution contract is not supported by this runner" in runner
    assert "5 scheduler grid points = 4 DiT forwards" in runner
    assert "OutputDirectory must be outside the source checkout" in runner
    assert "remote get-url origin" in runner
    assert "ls-files --others --exclude-standard" in runner
    assert "apply --reverse --check --index" in runner
    assert 'Set-ProcessEnvironment "PYTHONPATH" $KernelPythonRoot' in runner
    assert "from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton" in runner
    assert "from fastvideo_kernel.triton_kernels.index import map_to_index" in runner
    assert "hf auth login" in runner
    assert "--token" not in runner
    assert '"--repeats", "2"' not in runner
    assert '"--video-decode-backend", "taeh3"' not in runner


@pytest.mark.parametrize(
    "prompt",
    ((" " * 130_000) + "cat", "-hello"),
    ids=("long-valid-token-shape", "leading-option-marker"),
)
def test_generation_launcher_keeps_prompt_out_of_process_arguments(
    tmp_path: Path,
    prompt: str,
) -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    launcher_match = re.search(
        r"\$GenerationLauncher = @'\r?\n(.*?)\r?\n'@",
        runner,
        re.DOTALL,
    )
    assert launcher_match is not None

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    target_script = tmp_path / "target.py"
    target_script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--sentinel")\n'
        'parser.add_argument("--prompt")\n'
        "arguments = parser.parse_args()\n"
        'assert arguments.sentinel == "ok"\n'
        "print(len(arguments.prompt))\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-c",
        launcher_match.group(1),
        str(target_script),
        str(prompt_path),
        "--sentinel",
        "ok",
    ]
    assert all(prompt not in argument for argument in command)

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(len(prompt))


def test_double_click_installer_is_a_thin_non_admin_wrapper() -> None:
    wrapper = DOUBLE_CLICK_INSTALLER.read_text(encoding="utf-8")

    assert "%~dp0install_windows_h3_fastvideo_vsa.ps1" in wrapper
    assert "powershell.exe -NoLogo -NoProfile" in wrapper
    assert "%*" in wrapper
    assert "pause" in wrapper
    assert "runas" not in wrapper.lower()
    assert "http://" not in wrapper
    assert "https://" not in wrapper


def test_documentation_keeps_fastvideo_separate_from_native_trt() -> None:
    documentation = DOCUMENTATION.read_text(encoding="utf-8")
    sidebar = SIDEBAR.read_text(encoding="utf-8")

    assert "separate from the" in documentation
    assert "native TensorRT-RTX MiniMax H3 path" in documentation
    assert "does not add a" in documentation
    assert ".\\scripts\\install_windows_h3_fastvideo_vsa.ps1" in documentation
    assert ".\\scripts\\run_windows_h3_fastvideo_vsa.ps1" in documentation
    assert "five scheduler grid points" in documentation
    assert "exactly four DiT forwards" in documentation
    assert "--no-warmup --repeats 1" in documentation
    assert "full MiniMax H3 video-VAE" in documentation
    assert "345 frames" in documentation
    assert "CUDA 12.9.86" in documentation
    assert "Windows 11 Arm64" in documentation
    assert "x64 Python emulation" in documentation
    assert "no-grad in-place gate merge" in documentation
    assert "1358.751" in documentation
    assert "did not reach 10 minutes on this single RTX" in documentation
    assert "getting-started/windows-fastvideo-h3-vsa" in sidebar
    assert "does not contain tokens" not in documentation


def test_helper_sources_contain_no_local_identity_or_secret() -> None:
    paths = (
        PROFILE_PATH,
        BENCHMARK_PATH,
        INSTALLER,
        DOUBLE_CLICK_INSTALLER,
        RUNNER,
        REQUIREMENTS,
        DOCUMENTATION,
    )
    forbidden = (
        re.compile(r"(?i)[A-Z]:\\Users\\"),
        re.compile(r"(?i)/home/[^/]+"),
        re.compile(r"(?i)hf_[A-Za-z0-9]{20,}"),
        re.compile(r"(?i)(?:access|auth|api)[_-]?token\s*[=:]\s*['\"][^'\"]+"),
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern.search(text) is None, f"forbidden local or secret pattern in {path.name}"


@pytest.mark.parametrize("script", [INSTALLER, RUNNER])
def test_powershell_script_parses_when_powershell_is_available(script: Path) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell parser is unavailable")

    escaped_script = str(script).replace("'", "''")
    command = f"""
$tokens = $null
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    '{escaped_script}', [ref]$tokens, [ref]$errors
) | Out-Null
if ($errors.Count -ne 0) {{
    $errors | ForEach-Object {{ Write-Error $_.Message }}
    exit 1
}}
"""
    result = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
