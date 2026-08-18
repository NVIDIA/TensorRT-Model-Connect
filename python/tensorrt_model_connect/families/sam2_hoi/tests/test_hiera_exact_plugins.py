# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct


NATIVE_PLUGINS = Path(__file__).resolve().parent.parent / "native_plugins"


def _source(name: str) -> str:
    return (NATIVE_PLUGINS / name).read_text(encoding="utf-8")


def _normalized(name: str) -> str:
    return " ".join(_source(name).split())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        entries[relative.removeprefix("./")] = digest
    return entries


def test_exact_hiera_plugin_abi_is_additive_empty_and_fail_closed() -> None:
    plugins = {
        "hiera_layer_norm": "Sam2HoiHieraLayerNorm",
        "hiera_flash_attention": "Sam2HoiHieraFlashAttention96",
        "hiera_gelu": "Sam2HoiHieraGeluErfBF16",
        "hiera_block1415_projection": "Sam2HoiHieraBlock1415Projection",
    }
    for stem, plugin_name in plugins.items():
        header = _source(f"{stem}_plugin.h")
        implementation_suffix = "cu" if stem == "hiera_layer_norm" else "cpp"
        implementation = _source(f"{stem}_plugin.{implementation_suffix}")
        creator = _normalized(f"{stem}_creator.cpp")

        assert f'kPLUGIN_NAME = "{plugin_name}"' in " ".join(header.split())
        assert 'kPLUGIN_VERSION = "1"' in " ".join(header.split())
        assert "fields_.nbFields = 0;" in creator
        assert "fields_.fields = nullptr;" in creator
        assert "fields == nullptr || fields->nbFields != 0" in creator
        assert "if (length != 0) { return nullptr; }" in creator
        assert "getSerializationSize() const noexcept" in implementation
        assert "return 0;" in implementation


def test_exact_hiera_abi_entrypoints_guard_descriptors_types_and_pointers() -> None:
    layer_norm = _normalized("hiera_layer_norm_plugin.cu")
    flash = _normalized("hiera_flash_attention_plugin.cpp")
    gelu = _normalized("hiera_gelu_plugin.cpp")

    for source in (layer_norm, flash, gelu):
        assert "inputs_outputs == nullptr" in source
        assert "inputs == nullptr || outputs == nullptr" in source
        assert "inputs[0] == nullptr" in source
        assert "outputs[0] == nullptr" in source

    assert "input_types != nullptr" in flash
    assert "input_types != nullptr" in gelu
    assert "inputs[1] == nullptr || inputs[2] == nullptr" in layer_norm
    assert "valid_descriptors(input_descriptors, output_descriptors)" in layer_norm
    assert "outputs[0].type != nvinfer1::DataType::kFLOAT" in layer_norm
    assert "outputs[0].format != nvinfer1::TensorFormat::kLINEAR" in layer_norm

    assert "inputs[index].type != nvinfer1::DataType::kBF16" in flash
    assert "inputs[index].format != nvinfer1::TensorFormat::kLINEAR" in flash
    assert "outputs[0].type != nvinfer1::DataType::kBF16" in flash
    assert "outputs[0].format != nvinfer1::TensorFormat::kLINEAR" in flash


def test_hiera_layer_norm_has_only_reviewed_fp32_contracts() -> None:
    source = _normalized("hiera_layer_norm_plugin.cu")

    assert "width == 96 || width == 192 || width == 384 || width == 768" in source
    assert "constexpr float kEpsilon = 1.0e-6F" in source
    assert source.count("nvinfer1::DataType::kFLOAT") >= 4
    assert "nvinfer1::TensorFormat::kLINEAR" in source


def test_flash_attention_has_only_reviewed_bf16_d96_shapes() -> None:
    launcher = _source("hiera_flash_attention_launcher.cu")
    plugin = _normalized("hiera_flash_attention_plugin.cpp")
    allowed = {
        tuple(map(int, match))
        for match in re.findall(
            r"b == (\d+) && h == (\d+) && sq == (\d+) && sk == (\d+)",
            launcher,
        )
    }

    assert allowed == {
        (1024, 1, 64, 64),
        (1024, 2, 16, 64),
        (1024, 2, 16, 16),
        (1024, 4, 4, 16),
        (25, 4, 196, 196),
        (1, 4, 4096, 4096),
        (25, 8, 49, 196),
        (25, 8, 49, 49),
    }
    assert "constexpr int kHeadDim = 96" in launcher
    assert "qd == 96 && kd == 96 && vd == 96 && od == 96" in plugin
    assert plugin.count("nvinfer1::DataType::kBF16") >= 6
    assert "nvinfer1::TensorFormat::kLINEAR" in plugin


def test_gelu_lut_covers_complete_bf16_domain_and_special_values() -> None:
    binary = NATIVE_PLUGINS / "hiera_gelu_bf16_lut_cuda128_exact.bin"
    metadata = json.loads(
        (NATIVE_PLUGINS / "hiera_gelu_bf16_lut_cuda128_exact.json").read_text(encoding="utf-8")
    )
    raw = binary.read_bytes()
    values = struct.unpack("<65536H", raw)

    assert len(raw) == 131072
    assert metadata["table"] == {
        "entries": 65536,
        "entry_format": "little-endian raw IEEE BF16 uint16",
        "size_bytes": 131072,
        "binary_sha256": "c577423f9580e2d2e5943d57e92770a98d875d95a87861e32525d3db97df811f",
        "cuda_include_sha256": ("e5d85ba5af299f03678e876b6e0d26d640a55f9273d7e8da39007d54e01b69f1"),
    }
    assert _sha256(binary) == metadata["table"]["binary_sha256"]
    assert values[0x0000] == 0x0000
    assert values[0x8000] == 0x8000
    assert values[0x7F80] == 0x7F80
    assert values[0xFF80] == 0x7FFF
    assert all(values[index] == 0x7FFF for index in range(0x7F81, 0x8000))
    assert all(values[index] == 0x7FFF for index in range(0xFF81, 0x10000))

    launcher = _source("hiera_gelu_launcher.cu")
    assert "kGeluBf16Lut[1 << 16]" in launcher
    assert "std::uint16_t gelu_lookup(std::uint16_t input)" in launcher
    assert "erf(" not in launcher


def test_gelu_has_only_reviewed_bf16_nhwc_shapes() -> None:
    launcher = _source("hiera_gelu_launcher.cu")
    allowed = {
        tuple(map(int, match))
        for match in re.findall(
            r"h == (\d+) && w == (\d+) && c == (\d+)",
            launcher,
        )
    }

    assert allowed == {
        (256, 256, 384),
        (128, 128, 768),
        (64, 64, 1536),
        (32, 32, 3072),
    }
    assert "return b == 1" in launcher


def test_cmake_scopes_exact_flash_flags_and_uses_local_vendor_closure() -> None:
    cmake = _source("CMakeLists.txt")
    start = cmake.index("set_source_files_properties(hiera_flash_attention_launcher.cu")
    end = cmake.index("\n)", start) + 2
    flash_scope = cmake[start:end]

    for source in (
        "hiera_layer_norm_plugin.cu",
        "hiera_layer_norm_creator.cpp",
        "hiera_flash_attention_launcher.cu",
        "hiera_flash_attention_plugin.cpp",
        "hiera_flash_attention_creator.cpp",
        "hiera_gelu_launcher.cu",
        "hiera_gelu_plugin.cpp",
        "hiera_gelu_creator.cpp",
        "hiera_block1415_projection_plugin.cpp",
        "hiera_block1415_projection_creator.cpp",
    ):
        expected_count = 2 if source == "hiera_flash_attention_launcher.cu" else 1
        assert cmake.count(source) == expected_count
    for flag in (
        "UNFUSE_FMA=1",
        "-O2",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ):
        assert flag in flash_scope
        assert cmake.count(flag) == 1
    active_cmake = "\n".join(
        line for line in cmake.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--use_fast_math" not in active_cmake
    assert "set_source_files_properties(hiera_block1415_projection" not in cmake
    assert "CUDA::cublasLt" in cmake
    assert "PROPERTY CUDA_ARCHITECTURES 89 100" in " ".join(cmake.split())
    assert "${CMAKE_CURRENT_SOURCE_DIR}/vendor/flash_attention/include" in cmake
    assert "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cutlass/include" in cmake
    for remote_build in ("FetchContent", "ExternalProject", "curl", "wget", "git clone"):
        assert remote_build not in cmake


def test_vendor_pins_and_manifests_are_complete_and_exact() -> None:
    vendors = {
        "flash_attention": (14, 12, "979702c87a8713a8e0a5e9fee122b90d2ef13be5"),
        "cutlass": (132, 130, "afa1772203677c5118fcd82537a9c8fefbcc7008"),
    }
    for name, (entry_count, header_count, commit) in vendors.items():
        root = NATIVE_PLUGINS / "vendor" / name
        manifest = _manifest(root)
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}

        assert len(manifest) == entry_count
        assert sum(relative.startswith("include/") for relative in manifest) == header_count
        assert actual == {*manifest, "MANIFEST.sha256"}
        assert commit in (root / "SOURCE_COMMIT").read_text(encoding="utf-8")
        for relative, digest in manifest.items():
            assert _sha256(root / relative) == digest


def test_gelu_asset_manifest_matches_every_runtime_member() -> None:
    manifest_path = NATIVE_PLUGINS / "hiera_gelu_bf16_lut_cuda128_exact.MANIFEST.sha256"
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        entries[relative] = digest

    assert set(entries) == {
        "hiera_gelu_bf16_lut_cuda128_exact.bin",
        "hiera_gelu_bf16_lut_cuda128_exact.inc",
        "hiera_gelu_bf16_lut_cuda128_exact.json",
        "hiera_gelu_launcher.cu",
    }
    for relative, digest in entries.items():
        assert _sha256(NATIVE_PLUGINS / relative) == digest


def test_projection_contract_is_exact_and_lifecycle_is_fail_closed() -> None:
    header = _normalized("hiera_block1415_projection_plugin.h")
    source = _normalized("hiera_block1415_projection_plugin.cpp")

    for constant in (
        "kM = 1225",
        "kN = 768",
        "kK = 768",
        "kWORKSPACE_BYTES = std::size_t{1} << 20",
        "kALGORITHM_ID = 30",
        "kCTA_SWIZZLING = 1",
    ):
        assert constant in header
    assert "CUBLASLT_MATMUL_TILE_64x128" in source
    assert "CUBLAS_COMPUTE_32F, CUDA_R_32F" in source
    assert "CUBLAS_OP_T" in source
    assert "CUBLASLT_EPILOGUE_BIAS" in source
    assert "cublasLtMatmulAlgoGetHeuristic" in source
    assert "cublasLtMatmulAlgoCheck" in source
    assert "candidate.workspaceSize > kWORKSPACE_BYTES" in source
    assert "checked.workspaceSize > kWORKSPACE_BYTES" in source
    assert "dynamicTensorContract(inputs[0], {1, kM, kK})" in source
    assert "dynamicTensorContract(inputs[1], {kN, kK})" in source
    assert "dynamicTensorContract(inputs[2], {kN})" in source
    assert "dynamicTensorContract(outputs[0], {1, kM, kN})" in source
    assert "std::atomic_flag lock_ = ATOMIC_FLAG_INIT" in header
    assert "LockGuard guard(lock_)" in source
    assert "releaseLocked(); return initializeLocked() ? 0 : 1;" in source
    assert "void HieraBlock1415ProjectionPlugin::terminate() noexcept" in source
    assert "result->namespace_ = namespace_" in source
    assert "result->configured_ = configured_" in source
    assert "result->serialization_valid_ = serialization_valid_" in source
    assert "result->handle_" not in source
    assert "if (!selectExactAlgorithmLocked())" in source
    assert "return false;" in source
