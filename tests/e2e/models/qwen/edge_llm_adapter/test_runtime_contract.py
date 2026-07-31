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

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
CAPSULE_ROOT = (
    REPOSITORY_ROOT / "python" / "tensorrt_model_connect" / "families" / "qwen" / "edge_llm_adapter"
)
RUNTIME_ROOT = REPOSITORY_ROOT / "src" / "runtime" / "models" / "qwen" / "edge_llm_adapter"
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


IMPLEMENTATION_ID = "qwen.tensorrt-edge-llm"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROFILE_ID = "qwen3-0.6b-fp16--a100-pcie80-sm80--edgellm0.9-trt11.1"
RUNTIME_LIBRARY = "libtrtmc_impl_qwen_tensorrt_edge_llm.so"
ALLOW_TEST_PAYLOAD_ENV = "_TRTMC_INTERNAL_QWEN_EDGE_LLM_ALLOW_FAKE_RUNTIME_BUILD"
PROFILE_PATH = CAPSULE_ROOT / "profiles" / "qwen3-0.6b-a100-sm80-fp16.toml"
DEPENDENCY_LOCK_PATH = CAPSULE_ROOT / "dependency.lock"
with DEPENDENCY_LOCK_PATH.open("rb") as _dependency_lock_source:
    DEPENDENCY_LOCK = tomllib.load(_dependency_lock_source)
EDGE_RUNTIME_VERSION = str(DEPENDENCY_LOCK["downstream"]["version"])
EDGE_RUNTIME_COMMIT = str(DEPENDENCY_LOCK["downstream"]["commit"])
TENSORRT_VERSION = str(DEPENDENCY_LOCK["tensorrt"]["version"])
TENSORRT_VERSION_PARTS = tuple(int(part) for part in TENSORRT_VERSION.split("."))
CUDA_VERSION = str(DEPENDENCY_LOCK["cuda"]["version"])
_CUDA_MAJOR, _CUDA_MINOR = (int(part) for part in CUDA_VERSION.split("."))
CUDA_RUNTIME_VERSION = _CUDA_MAJOR * 1000 + _CUDA_MINOR * 10


def _profile_cases() -> tuple[tuple[str, str, str], ...]:
    cases = []
    for path in sorted((CAPSULE_ROOT / "profiles").glob("*.toml")):
        with path.open("rb") as source:
            data = tomllib.load(source)
        revisions = tuple(data["model"]["revisions"])
        if len(revisions) != 1:
            raise RuntimeError(f"test profile must declare one exact revision: {path}")
        cases.append((data["model"]["id"], revisions[0], path.name))
    return tuple(cases)


PROFILE_CASES = _profile_cases()


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
    ]


CreateFn = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.POINTER(CreateRequest),
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_size_t,
)


def test_real_runtime_forces_cuda_device_link_for_edgellm_core() -> None:
    cmake = (RUNTIME_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    stub = RUNTIME_ROOT / "device_link_stub.cu"

    assert "cmake_minimum_required(VERSION 3.20)" in cmake
    assert stub.is_file()
    assert "target_sources(${_target} PRIVATE device_link_stub.cu)" in cmake
    assert "CUDA_RESOLVE_DEVICE_SYMBOLS ON" in cmake
    assert "-Wl,--no-undefined" in cmake


def _nlohmann_json_include() -> Path:
    configured = os.environ.get("_TRTMC_INTERNAL_QWEN_EDGE_LLM_NLOHMANN_JSON_INCLUDE_DIR", "")
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
        "set _TRTMC_INTERNAL_QWEN_EDGE_LLM_NLOHMANN_JSON_INCLUDE_DIR"
    )


def _run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _build_fake_runtime(
    build: Path,
    *,
    tensorrt_build: int = TENSORRT_VERSION_PARTS[3],
    cuda_runtime_version: int = CUDA_RUNTIME_VERSION,
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
            f"-DTRTMC_EDGE_LLM_VERSION={EDGE_RUNTIME_VERSION}",
            f"-DTRTMC_EDGE_LLM_COMMIT={EDGE_RUNTIME_COMMIT}",
            f"-DTRTMC_TENSORRT_VERSION={TENSORRT_VERSION}",
            f"-DTRTMC_CUDA_VERSION={CUDA_VERSION}",
            f"-DTRTMC_CUDA_RUNTIME_VERSION={CUDA_RUNTIME_VERSION}",
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


@pytest.fixture(autouse=True)
def authorize_test_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_TEST_PAYLOAD_ENV, "1")


def _fake_engine(root: Path, profile_path: Path = PROFILE_PATH) -> Path:
    root.mkdir()
    with profile_path.open("rb") as source:
        profile = tomllib.load(source)
    (root / "config.json").write_text(
        json.dumps(
            {
                **profile["engine"],
                "edgellm_version": EDGE_RUNTIME_VERSION,
                "builder_config": profile["builder"],
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


def _staged_capsule(
    tmp_path: Path,
    fake_runtime: Path,
    *,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    profile_path: Path = PROFILE_PATH,
):
    engine = _fake_engine(tmp_path / "engine", profile_path)
    plugin = fake_runtime.parent / "libNvInfer_edgellm_plugin.so"
    manifest = load_implementation_manifest(CAPSULE_ROOT / "IMPLEMENTATION.toml")
    request = ImplementationRequest(
        model_id=model_id,
        model_revision=model_revision,
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
            "public_options": {
                "precision": "fp32",
                "max_cache_length": 256,
                "max_batch_size": 1,
            },
        },
    )
    probe = run_probe(manifest, request)
    assert probe.supported
    return (
        run_build(
            manifest,
            request,
            tmp_path / "capsule",
            probe=probe,
        ),
        request,
    )


def _create_request(artifact, metadata: dict) -> tuple[CreateRequest, bytes]:
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    request = CreateRequest(
        abi_version=1,
        struct_size=ctypes.sizeof(CreateRequest),
        implementation_id=IMPLEMENTATION_ID.encode(),
        model_id=str(artifact.descriptor["model"]["id"]).encode(),
        profile_id=str(artifact.descriptor["profile_id"]).encode(),
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
    assert factory.runtime_version.decode() == EDGE_RUNTIME_VERSION
    assert factory.runtime_commit.decode() == EDGE_RUNTIME_COMMIT
    assert factory.pipeline_abi_version == 1

    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(dso_path)],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "trtmc_get_optimized_runtime_factory_v1@@TRTMC_QWEN_EDGE_FACTORY_1" in exported
    assert "trtmc_get_runtime_provider" not in exported

    metadata_bytes = json.dumps(dict(artifact.descriptor), sort_keys=True).encode("utf-8")
    invalid_metadata = json.loads(metadata_bytes)
    invalid_metadata["target"]["gpu_name"] = "NVIDIA H100 80GB HBM3"
    assert b"gpu_name" in _rejected_create(artifact, invalid_metadata)


@pytest.mark.parametrize(("model_id", "revision", "profile_file"), PROFILE_CASES)
def test_one_runtime_dso_accepts_every_qwen_profile(
    tmp_path: Path,
    fake_runtime: Path,
    model_id: str,
    revision: str,
    profile_file: str,
) -> None:
    artifact, _ = _staged_capsule(
        tmp_path,
        fake_runtime,
        model_id=model_id,
        model_revision=revision,
        profile_path=CAPSULE_ROOT / "profiles" / profile_file,
    )
    library = ctypes.CDLL(str(artifact.artifacts_path / RUNTIME_LIBRARY))
    library.trtmc_get_optimized_runtime_factory_v1.restype = ctypes.POINTER(FactoryV1)
    factory = library.trtmc_get_optimized_runtime_factory_v1().contents
    create = CreateFn(factory.create)
    request, metadata_bytes = _create_request(artifact, dict(artifact.descriptor))
    error = ctypes.create_string_buffer(2048)

    pipeline = create(ctypes.byref(request), error, len(error))

    assert metadata_bytes
    assert pipeline, error.value.decode()
    assert artifact.descriptor["model"] == {"id": model_id, "revision": revision}


def test_runtime_compile_binding_matches_selected_manifest_and_profile(
    tmp_path: Path, fake_runtime: Path
) -> None:
    artifact, request = _staged_capsule(tmp_path, fake_runtime)
    manifest = load_implementation_manifest(CAPSULE_ROOT / "IMPLEMENTATION.toml")
    binding = artifact.descriptor["build_binding"]

    assert set(binding) == {
        "schema_version",
        "implementation_id",
        "manifest_sha256",
        "request_sha256",
        "profile_id",
        "profile_sha256",
    }
    assert binding["schema_version"] == 1
    assert binding["manifest_sha256"] == manifest_contract_sha256(manifest)
    assert binding["implementation_id"] == IMPLEMENTATION_ID
    assert binding["profile_id"] == PROFILE_ID
    assert binding["profile_sha256"] == run_probe(manifest, request).profile_sha256


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


@pytest.mark.parametrize("field", ("request_sha256", "profile_sha256"))
@pytest.mark.parametrize("digest", ("A" * 64, "0" * 63, "g" * 64))
def test_runtime_rejects_noncanonical_build_binding_digest(
    tmp_path: Path,
    fake_runtime: Path,
    field: str,
    digest: str,
) -> None:
    artifact, _ = _staged_capsule(tmp_path, fake_runtime)
    metadata = _mutable_descriptor(artifact)
    metadata["build_binding"][field] = digest

    error = _rejected_create(artifact, metadata)
    assert (
        error
        == (
            f"implementation metadata build_binding field '{field}' "
            "must be a lowercase SHA-256 digest"
        ).encode()
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
    wrong_build = TENSORRT_VERSION_PARTS[3] + 1
    wrong_runtime = _build_fake_runtime(tmp_path / "wrong-trt-build", tensorrt_build=wrong_build)
    artifact, _ = _staged_capsule(tmp_path, wrong_runtime)
    observed_version = ".".join(str(part) for part in (*TENSORRT_VERSION_PARTS[:3], wrong_build))
    assert _rejected_create(artifact, dict(artifact.descriptor)).decode() == (
        f"loaded TensorRT runtime version {observed_version} is unsupported; "
        f"expected {TENSORRT_VERSION}"
    )


def test_runtime_rejects_wrong_cuda_before_plugin_validation(tmp_path: Path) -> None:
    wrong_cuda_runtime = CUDA_RUNTIME_VERSION - 10
    wrong_runtime = _build_fake_runtime(
        tmp_path / "wrong-cuda-runtime", cuda_runtime_version=wrong_cuda_runtime
    )
    artifact, _ = _staged_capsule(tmp_path, wrong_runtime)
    (artifact.artifacts_path / "libNvInfer_edgellm_plugin.so").unlink()

    assert _rejected_create(artifact, dict(artifact.descriptor)).decode() == (
        f"loaded CUDA runtime version {wrong_cuda_runtime} is unsupported; "
        f"expected {CUDA_RUNTIME_VERSION} (CUDA {CUDA_VERSION})"
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
    request.profile_id = "qwen3-0.6b-fp16--a100-pcie80-sm80--edgellm0.9-trt11.1";
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
    config.cfg_scale = -2.0F;
    config.sde_gamma = -2.0F;
    config.height = -1;
    config.width = -1;
    config.text_generation_mode.clear();
    config.block_length = -1;
    config.confidence_threshold = -2.0F;
    const auto first = pipeline->generate("hello", config);
    const auto second = pipeline->generate("world", config);
    if (first.text != "fake:hello" || second.text != "fake:world" ||
        first.token_ids.size() != 4 || first.prefill_ms != 0.0 || first.decode_ms != 0.0 ||
        second.prefill_ms != 0.0 || second.decode_ms != 0.0 ||
        pipeline->model_id() != std::string("Qwen/Qwen3-0.6B") ||
        pipeline->pipeline_type() != std::string("QwenTextGenerationPipeline"))
        return 13;
    config.temperature = 0.7F;
    config.top_k = -1;
    config.top_p = 0.8F;
    const auto nucleus = pipeline->generate("inspect-sampling", config);
    if (nucleus.text != "fake-sampling:0:0.800000")
        return 14;
    config.top_p = 1.0F;
    try {
        (void)pipeline->generate("unsupported-unbounded-sampling", config);
        return 15;
    } catch (const std::invalid_argument&) {
    }
    config.top_k = 50;
    config.top_p = 0.0F;
    const auto top_p_greedy = pipeline->generate("inspect-sampling", config);
    if (top_p_greedy.text != "fake-sampling:1:1.000000")
        return 16;
    config.top_p = 0.8F;
    config.temperature = 0.0F;
    const auto temperature_greedy = pipeline->generate("inspect-sampling", config);
    if (temperature_greedy.text != "fake-sampling:1:1.000000")
        return 17;
    config.temperature = 0.7F;
    config.top_p = 1.0F;
    config.top_k = 1;
    const auto greedy = pipeline->generate("inspect-sampling", config);
    if (greedy.text != "fake-sampling:1:1.000000")
        return 18;
    config.temperature = 0.0005F;
    config.top_k = 50;
    try {
        (void)pipeline->generate("unsupported-low-temperature-sampling", config);
        return 19;
    } catch (const std::invalid_argument&) {
    }
    config.temperature = 0.7F;
    config.top_k = 1;
    config.top_p = 0.9999995F;
    try {
        (void)pipeline->generate("unsupported-near-one-top-p", config);
        return 20;
    } catch (const std::invalid_argument&) {
    }
    config.top_p = 1.0F;
    config.top_k = 1025;
    try {
        (void)pipeline->generate("rejected-top-k", config);
        return 21;
    } catch (const std::invalid_argument&) {
    }
    config.top_k = 1;
    config.min_p = 0.1F;
    try {
        (void)pipeline->generate("rejected", config);
        return 22;
    } catch (const std::invalid_argument&) {
    }
    config.min_p = 0.0F;
    config.lora_adapter_id = "unsupported-adapter";
    try {
        (void)pipeline->generate("rejected-lora", config);
        return 23;
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
