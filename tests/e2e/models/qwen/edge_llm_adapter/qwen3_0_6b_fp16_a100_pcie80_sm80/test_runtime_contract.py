# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and exercise this capsule's DSO contract without CUDA or Edge-LLM."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
CAPSULE_ROOT = (
    REPOSITORY_ROOT
    / "python"
    / "tensorrt_model_connect"
    / "families"
    / "qwen"
    / "edge_llm_adapter"
    / "qwen3_0_6b_fp16_a100_pcie80_sm80"
)
RUNTIME_ROOT = (
    REPOSITORY_ROOT
    / "src"
    / "runtime"
    / "models"
    / "qwen"
    / "edge_llm_adapter"
    / "qwen3_0_6b_fp16_a100_pcie80_sm80"
)
PYTHON_ROOT = REPOSITORY_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from tensorrt_model_connect.runtime_provider.provider_process import (  # noqa: E402
    ImplementationRequest,
    run_build,
    run_probe,
)
from tensorrt_model_connect.runtime_provider.manifest import (  # noqa: E402
    load_implementation_manifest,
    manifest_contract_sha256,
)


IMPLEMENTATION_ID = "qwen3-0.6b-fp16.tensorrt-edge-llm-v0.9.trt11.a100-pcie80-sm80"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROFILE_ID = "qwen3-0.6b-fp16--a100-pcie80-sm80--edgellm0.9-trt11"
RUNTIME_LIBRARY = "libtrtmc_impl_qwen3_0_6b_fp16_tensorrt_edge_llm_v0_9_trt11.so"


def test_real_runtime_delegates_graph_capture_and_generation_to_edgellm() -> None:
    source = (RUNTIME_ROOT / "adapter.cpp").read_text(encoding="utf-8")

    assert "RTLD_NOW | RTLD_LOCAL" in source
    assert source.index("configure_edge_logging();") < source.index("open_edge_plugin(plugin);")
    assert "captureDecodingCUDAGraph(handle.stream)" in source
    assert "decoding CUDA graph capture failed" in source
    assert "handle.runtime->handleRequest(edge_request, edge_response, handle.stream)" in source


@pytest.fixture(autouse=True)
def _enable_test_only_payload_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("_TRTMC_INTERNAL_QWEN3_06B_ALLOW_FAKE_RUNTIME_BUILD", "1")


class CreateRequest(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("implementation_id", ctypes.c_char_p),
        ("model_id", ctypes.c_char_p),
        ("profile_id", ctypes.c_char_p),
        ("bundle_path", ctypes.c_char_p),
        ("artifact_path", ctypes.c_char_p),
        ("implementation_metadata", ctypes.c_char_p),
        ("implementation_metadata_size", ctypes.c_size_t),
        ("load_options", ctypes.c_void_p),
    ]


class ToolchainAbiV1(ctypes.Structure):
    _fields_ = [
        ("compiler_family", ctypes.c_uint32),
        ("compiler_major_version", ctypes.c_uint32),
        ("cxx_abi_family", ctypes.c_uint32),
        ("cxx_abi_version", ctypes.c_uint32),
        ("cxx_language_standard", ctypes.c_uint32),
        ("standard_library_family", ctypes.c_uint32),
        ("standard_library_version", ctypes.c_uint32),
        ("standard_library_abi", ctypes.c_uint32),
        ("pointer_size", ctypes.c_uint32),
        ("string_size", ctypes.c_uint32),
        ("string_alignment", ctypes.c_uint32),
    ]


class FactoryV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("implementation_id", ctypes.c_char_p),
        ("runtime_name", ctypes.c_char_p),
        ("runtime_version", ctypes.c_char_p),
        ("runtime_commit", ctypes.c_char_p),
        ("create", ctypes.c_void_p),
        ("pipeline_abi_version", ctypes.c_uint32),
        ("toolchain_abi", ToolchainAbiV1),
        ("process_compatibility_namespace", ctypes.c_char_p),
        ("process_compatibility_fingerprint", ctypes.c_char_p),
    ]


CreateFn = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.POINTER(CreateRequest),
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_size_t,
)


def _nlohmann_json_include() -> Path:
    configured = os.environ.get("_TRTMC_INTERNAL_QWEN3_06B_NLOHMANN_JSON_INCLUDE_DIR", "")
    trtmc_binary = os.environ.get("TRTMC_BINARY", "")
    candidates = (
        ([Path(configured)] if configured else [])
        + (
            [Path(trtmc_binary).resolve().parent / "_deps/nlohmann_json-src/include"]
            if trtmc_binary
            else []
        )
        + [Path("/usr/include")]
        + sorted(REPOSITORY_ROOT.glob("build*/_deps/nlohmann_json-src/include"))
    )
    for candidate in candidates:
        if (candidate / "nlohmann" / "json.hpp").is_file():
            return candidate.resolve()
    pytest.fail(
        "Qwen fake-runtime tests require MC's provisioned nlohmann_json include; "
        "set _TRTMC_INTERNAL_QWEN3_06B_NLOHMANN_JSON_INCLUDE_DIR"
    )


def _run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _build_fake_runtime(
    build: Path, *, tensorrt_build: int = 113, cuda_runtime_version: int = 13030
) -> Path:
    json_include = _nlohmann_json_include()
    manifest = load_implementation_manifest(CAPSULE_ROOT / "IMPLEMENTATION.toml")
    _run(
        [
            "cmake",
            "-S",
            str(RUNTIME_ROOT),
            "-B",
            str(build),
            "-DTRTMC_QWEN_EDGE_FAKE_RUNTIME=ON",
            (f"-DTRTMC_SDK_INCLUDE_DIRS={REPOSITORY_ROOT / 'src'};{REPOSITORY_ROOT / 'include'}"),
            f"-DTRTMC_NLOHMANN_JSON_INCLUDE_DIR={json_include}",
            f"-DTRTMC_QWEN_EDGE_FAKE_TENSORRT_BUILD={tensorrt_build}",
            f"-DTRTMC_QWEN_EDGE_FAKE_CUDA_RUNTIME_VERSION={cuda_runtime_version}",
            (f"-DTRTMC_QWEN_EDGE_MANIFEST_SHA256={manifest_contract_sha256(manifest)}"),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    _run(["cmake", "--build", str(build), "--parallel", "2"])
    runtime = build / RUNTIME_LIBRARY
    assert runtime.is_file()
    (build / "libNvInfer_edgellm_plugin.so").write_bytes(b"fake-plugin-contract")
    return runtime


@pytest.fixture(scope="module")
def fake_runtime(tmp_path_factory) -> Path:
    return _build_fake_runtime(tmp_path_factory.mktemp("qwen-edge-runtime-build"))


def _fake_engine(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model": "qwen3",
                "spec_decode_type": "none",
                "engine_role": "llm",
                "edgellm_version": "0.9.0",
                "vocab_size": 151936,
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 40960,
                "rope_theta": 1000000.0,
                "rope_scaling": None,
                "partial_rotary_factor": 1.0,
                "num_deepstack_features": 0,
                "ple_enabled": False,
                "num_ple_inputs": 0,
                "ple_hidden_size": 0,
                "kv_cache_dtype": "fp16",
                "builder_config": {
                    "max_input_len": 1024,
                    "max_kv_cache_capacity": 4096,
                    "max_batch_size": 4,
                    "spec_draft": False,
                    "spec_base": False,
                    "max_lora_rank": 0,
                    "trt_native_ops": False,
                },
            }
        ),
        encoding="utf-8",
    )
    for filename in (
        "llm.engine",
        "embedding.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "processed_chat_template.json",
    ):
        (root / filename).write_bytes(filename.encode("utf-8"))
    return root


def _staged_capsule(tmp_path: Path, fake_runtime: Path):
    engine = _fake_engine(tmp_path / "engine")
    plugin = fake_runtime.parent / "libNvInfer_edgellm_plugin.so"
    manifest = load_implementation_manifest(CAPSULE_ROOT / "IMPLEMENTATION.toml")
    request = ImplementationRequest(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        target={
            "os": "linux",
            "architecture": "x86_64",
            "platform_kind": "discrete",
            "gpu_architecture": "sm80",
            "gpu_memory_mib": 81152,
            "gpu_count": 8,
            "gpu_name": "NVIDIA A100 80GB PCIe",
        },
        parameters={
            "engine_dir": str(engine),
            "runtime_library": str(fake_runtime),
            "runtime_plugin": str(plugin),
            "runtime_build": {"fake": True},
        },
    )
    probe = run_probe(manifest, request)
    assert probe.supported
    return run_build(manifest, request, tmp_path / "capsule", probe=probe), request


def _create_request(artifact, metadata: dict) -> tuple[CreateRequest, bytes]:
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    request = CreateRequest(
        abi_version=1,
        struct_size=ctypes.sizeof(CreateRequest),
        implementation_id=IMPLEMENTATION_ID.encode(),
        model_id=MODEL_ID.encode(),
        profile_id=PROFILE_ID.encode(),
        bundle_path=b"fake.trtfb",
        artifact_path=str(artifact.artifacts_path).encode(),
        implementation_metadata=metadata_bytes,
        implementation_metadata_size=len(metadata_bytes),
        load_options=None,
    )
    return request, metadata_bytes


def _mutable_descriptor(artifact) -> dict:
    return json.loads(json.dumps(dict(artifact.descriptor)))


def _rejected_create(artifact, metadata: dict) -> bytes:
    library = ctypes.CDLL(str(artifact.artifacts_path / RUNTIME_LIBRARY))
    library.trtmc_get_optimized_runtime_factory_v1.restype = ctypes.POINTER(FactoryV1)
    factory = library.trtmc_get_optimized_runtime_factory_v1().contents
    create = CreateFn(factory.create)
    create_request, metadata_bytes = _create_request(artifact, metadata)
    assert metadata_bytes
    error = ctypes.create_string_buffer(2048)
    assert not create(ctypes.byref(create_request), error, len(error))
    return error.value


def test_fake_runtime_exports_exact_identity_and_validates_create_metadata(
    tmp_path: Path, fake_runtime: Path
) -> None:
    artifact, _ = _staged_capsule(tmp_path, fake_runtime)
    dso_path = artifact.artifacts_path / RUNTIME_LIBRARY
    library = ctypes.CDLL(str(dso_path))
    library.trtmc_get_optimized_runtime_factory_v1.restype = ctypes.POINTER(FactoryV1)
    factory = library.trtmc_get_optimized_runtime_factory_v1().contents

    assert factory.abi_version == 1
    assert factory.implementation_id.decode() == IMPLEMENTATION_ID
    assert factory.runtime_name == b"tensorrt-edge-llm"
    assert factory.runtime_version == b"0.9.0"
    assert factory.runtime_commit == b"1ac0f2b99642045125e1c5ac7b109434ba3b36c7"
    assert factory.pipeline_abi_version == 1
    assert factory.process_compatibility_namespace == b"tensorrt-edge-llm.plugin-registry"
    assert factory.process_compatibility_fingerprint == (
        b"edgellm-1ac0f2b99642045125e1c5ac7b109434ba3b36c7-trt11.2.0.113-cuda13.3-sm80"
    )

    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(dso_path)],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "trtmc_get_optimized_runtime_factory_v1@@TRTMC_QWEN3_EDGE_FACTORY_1" in exported
    assert "trtmc_get_runtime_provider" not in exported

    metadata_bytes = json.dumps(dict(artifact.descriptor), sort_keys=True).encode("utf-8")
    invalid_metadata = json.loads(metadata_bytes)
    invalid_metadata["target"]["gpu_name"] = "NVIDIA H100 80GB HBM3"
    assert b"gpu_name" in _rejected_create(artifact, invalid_metadata)


def test_runtime_compile_binding_matches_selected_manifest_and_profile(
    tmp_path: Path, fake_runtime: Path
) -> None:
    artifact, _ = _staged_capsule(tmp_path, fake_runtime)
    manifest = load_implementation_manifest(CAPSULE_ROOT / "IMPLEMENTATION.toml")
    binding = artifact.descriptor["build_binding"]

    assert set(binding) == {
        "schema_version",
        "implementation_id",
        "manifest_sha256",
        "request_sha256",
        "profile_id",
    }
    assert binding["schema_version"] == 1
    assert binding["manifest_sha256"] == manifest_contract_sha256(manifest)
    assert binding["implementation_id"] == IMPLEMENTATION_ID
    assert binding["profile_id"] == PROFILE_ID


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", 2),
        ("implementation_id", IMPLEMENTATION_ID[:-1] + "x"),
        ("manifest_sha256", "0" * 64),
        ("profile_id", PROFILE_ID[:-1] + "x"),
    ),
)
def test_runtime_rejects_stable_build_binding_tamper(
    tmp_path: Path,
    fake_runtime: Path,
    field: str,
    replacement: object,
) -> None:
    artifact, _ = _staged_capsule(tmp_path, fake_runtime)
    metadata = _mutable_descriptor(artifact)
    original = metadata["build_binding"][field]
    assert replacement != original
    if isinstance(original, str) and isinstance(replacement, str):
        assert len(replacement) == len(original)
    metadata["build_binding"][field] = replacement

    error = _rejected_create(artifact, metadata)
    assert field.encode() in error


@pytest.mark.parametrize("request_sha256", ("A" * 64, "0" * 63, "g" * 64))
def test_runtime_rejects_noncanonical_request_digest(
    tmp_path: Path,
    fake_runtime: Path,
    request_sha256: str,
) -> None:
    artifact, _ = _staged_capsule(tmp_path, fake_runtime)
    metadata = _mutable_descriptor(artifact)
    metadata["build_binding"]["request_sha256"] = request_sha256

    error = _rejected_create(artifact, metadata)
    assert error == (
        b"implementation metadata build_binding field 'request_sha256' "
        b"must be a lowercase SHA-256 digest"
    )


@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_runtime_rejects_nonexact_build_binding_field_set(
    tmp_path: Path,
    fake_runtime: Path,
    mutation: str,
) -> None:
    artifact, _ = _staged_capsule(tmp_path, fake_runtime)
    metadata = _mutable_descriptor(artifact)
    if mutation == "extra":
        metadata["build_binding"]["unexpected"] = "value"
    else:
        del metadata["build_binding"]["request_sha256"]

    error = _rejected_create(artifact, metadata)
    assert error == b"implementation metadata build_binding has an unexpected field set"


def test_runtime_rejects_the_wrong_loaded_tensorrt_build_before_construction(
    tmp_path: Path,
) -> None:
    wrong_runtime = _build_fake_runtime(tmp_path / "wrong-trt-build", tensorrt_build=112)
    artifact, _ = _staged_capsule(tmp_path, wrong_runtime)
    assert _rejected_create(artifact, dict(artifact.descriptor)) == (
        b"loaded TensorRT runtime version 11.2.0.112 is unsupported; expected 11.2.0.113"
    )


def test_runtime_rejects_wrong_cuda_before_plugin_validation(tmp_path: Path) -> None:
    wrong_runtime = _build_fake_runtime(tmp_path / "wrong-cuda-runtime", cuda_runtime_version=13020)
    artifact, _ = _staged_capsule(tmp_path, wrong_runtime)
    (artifact.artifacts_path / "libNvInfer_edgellm_plugin.so").unlink()

    assert _rejected_create(artifact, dict(artifact.descriptor)) == (
        b"loaded CUDA runtime version 13020 is unsupported; expected 13030 (CUDA 13.3)"
    )


def test_fake_runtime_factory_returns_an_ipipeline_that_owns_generation(
    tmp_path: Path, fake_runtime: Path
) -> None:
    artifact, _ = _staged_capsule(tmp_path, fake_runtime)
    metadata = tmp_path / "implementation.json"
    metadata.write_text(json.dumps(dict(artifact.descriptor), sort_keys=True), encoding="utf-8")
    client = tmp_path / "factory_client.cpp"
    client.write_text(
        r"""
#include "runtime/providers/optimized_runtime_factory.h"

#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4)
        return 10;
    void* dso = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (dso == nullptr)
        return 11;
    auto getter = reinterpret_cast<trtmc::internal::GetOptimizedRuntimeFactoryV1>(
        dlsym(dso, trtmc::internal::kOptimizedRuntimeFactoryEntrypointV1));
    if (getter == nullptr)
        return 12;
    const auto* factory = getter();
    std::ifstream input(argv[3]);
    std::string metadata((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
    trtmc::internal::OptimizedRuntimePipelineCreateRequestV1 request{};
    request.abi_version = trtmc::internal::kOptimizedRuntimeFactoryAbiVersionV1;
    request.struct_size = sizeof(request);
    request.implementation_id = factory->implementation_id;
    request.model_id = "Qwen/Qwen3-0.6B";
    request.profile_id = "qwen3-0.6b-fp16--a100-pcie80-sm80--edgellm0.9-trt11";
    request.bundle_path = "fake.trtfb";
    request.artifact_path = argv[2];
    request.implementation_metadata = metadata.data();
    request.implementation_metadata_size = metadata.size();
    char error[2048]{};
    std::unique_ptr<trtmc::IPipeline> pipeline(factory->create(&request, error, sizeof(error)));
    if (!pipeline)
        throw std::runtime_error(error);
    trtmc::GenerateConfig config;
    config.max_new_tokens = 4;
    config.temperature = 0.0F;
    config.top_k = 1;
    config.top_p = 1.0F;
    config.use_chat_template = true;
    config.enable_thinking = false;
    const auto first = pipeline->generate("hello", config);
    const auto second = pipeline->generate("world", config);
    if (first.text != "fake:hello" || second.text != "fake:world" ||
        first.token_ids.size() != 4 || first.prefill_ms != 0.0 || first.decode_ms != 0.0 ||
        second.prefill_ms != 0.0 || second.decode_ms != 0.0 ||
        pipeline->model_id() != std::string("Qwen/Qwen3-0.6B") ||
        pipeline->pipeline_type() != std::string("QwenTextGenerationPipeline"))
        return 13;
    config.top_k = -1;
    config.top_p = 0.8F;
    const auto normalized = pipeline->generate("normalized", config);
    if (normalized.text != "fake:normalized")
        return 14;
    config.top_k = 1;
    const auto nucleus = pipeline->generate("inspect-sampling", config);
    if (nucleus.text != "fake-sampling:0:0.800000")
        return 15;
    config.top_p = 1.0F;
    const auto greedy_defaults = pipeline->generate("greedy-defaults", config);
    if (greedy_defaults.text != "fake:greedy-defaults")
        return 16;
    config.temperature = 0.5F;
    const auto greedy_temperature = pipeline->generate("greedy-temperature", config);
    if (greedy_temperature.text != "fake:greedy-temperature")
        return 17;
    config.temperature = 0.0F;
    config.top_k = 1;
    config.min_p = 0.1F;
    try {
        (void)pipeline->generate("rejected", config);
        return 18;
    } catch (const std::invalid_argument&) {
    }
    config.min_p = 0.0F;
    config.lora_adapter_id = "unsupported-adapter";
    try {
        (void)pipeline->generate("rejected-lora", config);
        return 19;
    } catch (const std::invalid_argument&) {
    }
    std::cout << first.text << "\n" << second.text << "\n";
    return 0;
}
""".lstrip(),
        encoding="utf-8",
    )
    binary = tmp_path / "factory-client"
    _run(
        [
            "c++",
            "-std=c++17",
            f"-I{REPOSITORY_ROOT / 'src'}",
            f"-I{REPOSITORY_ROOT / 'include'}",
            str(client),
            "-ldl",
            "-pthread",
            "-o",
            str(binary),
        ]
    )
    completed = subprocess.run(
        [
            str(binary),
            str(artifact.artifacts_path / RUNTIME_LIBRARY),
            str(artifact.artifacts_path),
            str(metadata),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "fake:hello\nfake:world\n"
