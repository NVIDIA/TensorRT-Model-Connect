# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-local contracts for the Qwen3-1.7B TensorRT Edge-LLM capsule."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
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
    / "qwen3_1_7b_fp16_a100_pcie80_sm80"
)
RUNTIME_ROOT = (
    REPOSITORY_ROOT
    / "src"
    / "runtime"
    / "models"
    / "qwen"
    / "edge_llm_adapter"
    / "qwen3_1_7b_fp16_a100_pcie80_sm80"
)
PYTHON_ROOT = REPOSITORY_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

from tensorrt_model_connect.runtime_provider.provider_process import (  # noqa: E402
    BuildAdapterError,
    ImplementationRequest,
    run_build,
    run_probe,
)
from tensorrt_model_connect.runtime_provider.bundle import (  # noqa: E402
    write_optimized_bundle,
)
from tensorrt_model_connect.runtime_provider.manifest import (  # noqa: E402
    load_implementation_manifest,
    manifest_contract_sha256,
)
from tensorrt_model_connect.runtime_provider.orchestrator import (  # noqa: E402
    discover_family_implementations_for_model,
    family_implementation_root,
)


MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
IMPLEMENTATION_ID = "qwen3-1.7b-fp16.tensorrt-edge-llm-v0.9.trt11.a100-pcie80-sm80"
PROFILE_ID = "qwen3-1.7b-fp16--a100-pcie80-sm80--edgellm0.9-trt11"
EDGE_COMMIT = "1ac0f2b99642045125e1c5ac7b109434ba3b36c7"
EDGE_SOURCE = "https://github.com/NVIDIA/TensorRT-Edge-LLM.git"
RUNTIME_LIBRARY = "libtrtmc_impl_qwen3_1_7b_fp16_tensorrt_edge_llm_v0_9_trt11.so"
RUNTIME_PLUGIN = "libNvInfer_edgellm_plugin.so"
MANIFEST_PATH = CAPSULE_ROOT / "IMPLEMENTATION.toml"
PROFILE_PATH = CAPSULE_ROOT / "profiles" / "a100-pcie80-sm80-fp16.toml"
ADAPTER_PATH = CAPSULE_ROOT / "adapter.py"
ENGINE_MODEL_CONFIG: dict[str, object] = {
    "model": "qwen3",
    "spec_decode_type": "none",
    "engine_role": "llm",
    "edgellm_version": "0.9.0",
    "vocab_size": 151936,
    "hidden_size": 2048,
    "intermediate_size": 6144,
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
}
ENGINE_BUILDER_CONFIG: dict[str, object] = {
    "max_input_len": 1024,
    "max_kv_cache_capacity": 4096,
    "max_batch_size": 4,
    "spec_draft": False,
    "spec_base": False,
    "max_lora_rank": 0,
    "trt_native_ops": False,
}
REQUIRED_ENGINE_FILES = (
    "llm.engine",
    "embedding.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "processed_chat_template.json",
)


def test_qwen_adapter_package_inventory_is_model_owned() -> None:
    def source_files(root: Path) -> set[str]:
        return {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }

    assert source_files(CAPSULE_ROOT) | {
        f"runtime/{relative}" for relative in source_files(RUNTIME_ROOT)
    } == {
        "IMPLEMENTATION.toml",
        "adapter.py",
        "dependency.lock",
        "profiles/a100-pcie80-sm80-fp16.toml",
        "runtime/CMakeLists.txt",
        "runtime/adapter.cpp",
        "runtime/exports.map",
    }


def test_source_checkout_discovers_edge_llm_only_from_qwen_builder() -> None:
    root = family_implementation_root("qwen")
    discovered = discover_family_implementations_for_model("qwen", MODEL_ID)

    assert root == CAPSULE_ROOT.parents[1]
    assert [manifest.implementation_id for manifest in discovered] == [IMPLEMENTATION_ID]


def test_runtime_source_resolves_from_qwen_runtime_folder_in_a_checkout() -> None:
    adapter = _load_adapter_module()

    assert adapter._runtime_source_root() == RUNTIME_ROOT


def test_installed_layout_compiles_the_packaged_qwen_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _load_adapter_module()
    package = tmp_path / "tensorrt_model_connect"
    packaged_builder = (
        package / "families" / "qwen" / "edge_llm_adapter" / "qwen3_1_7b_fp16_a100_pcie80_sm80"
    )
    packaged_runtime = packaged_builder / "runtime"
    private_sdk = package / "runtime_provider" / "_sdk" / "include"
    shutil.copytree(RUNTIME_ROOT, packaged_runtime)
    for source, relative in (
        (
            REPOSITORY_ROOT / "src/runtime/providers/optimized_runtime_factory.h",
            Path("runtime/providers/optimized_runtime_factory.h"),
        ),
        (REPOSITORY_ROOT / "include/trtmc/pipeline.h", Path("trtmc/pipeline.h")),
    ):
        destination = private_sdk / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    monkeypatch.setattr(adapter, "CAPSULE_ROOT", packaged_builder)
    monkeypatch.setattr(adapter, "REPOSITORY_ROOT", package)
    monkeypatch.setenv("_TRTMC_INTERNAL_QWEN3_1_7B_ALLOW_FAKE_RUNTIME_BUILD", "1")
    manifest = load_implementation_manifest(MANIFEST_PATH)
    runtime, plugin, dependency = adapter._build_runtime_dso(
        tmp_path / "output",
        {
            "runtime_build": {
                "fake": True,
                "nlohmann_json_include_dir": str(_nlohmann_json_include()),
                "parallel": 2,
            }
        },
        manifest_contract_sha256(manifest),
    )

    assert adapter._runtime_source_root() == packaged_runtime
    assert adapter._private_sdk_include_roots() == (private_sdk,)
    assert runtime.read_bytes().startswith(b"\x7fELF")
    assert plugin is None
    assert dependency is None


def _nlohmann_json_include() -> Path:
    configured = os.environ.get("_TRTMC_INTERNAL_QWEN3_1_7B_NLOHMANN_JSON_INCLUDE_DIR", "")
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
        "set _TRTMC_INTERNAL_QWEN3_1_7B_NLOHMANN_JSON_INCLUDE_DIR"
    )


def _load_adapter_module():
    name = f"trtmc_qwen_edge_adapter_test_{id(object())}"
    specification = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _qualify_exporter_host(adapter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter.platform, "machine", lambda: "x86_64")


def _target(**changes: object) -> dict[str, object]:
    target: dict[str, object] = {
        "os": "linux",
        "architecture": "x86_64",
        "platform_kind": "discrete",
        "gpu_architecture": "sm80",
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "gpu_count": 1,
        "gpu_memory_mib": 81152,
    }
    target.update(changes)
    return target


def _request(
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    target: dict[str, object] | None = None,
    parameters: dict[str, object] | None = None,
) -> ImplementationRequest:
    return ImplementationRequest(
        model_id=model_id,
        model_revision=revision,
        target=target or _target(),
        parameters=parameters,
    )


def _fake_engine(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                **ENGINE_MODEL_CONFIG,
                "builder_config": ENGINE_BUILDER_CONFIG,
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
        (root / filename).write_bytes(f"fake-{filename}".encode())
    return root


def _prebuilt_runtime_parameters(tmp_path: Path) -> dict[str, object]:
    runtime = tmp_path / "runtime-source.so"
    plugin = tmp_path / "plugin-source.so"
    runtime.write_bytes(b"fake-runtime-dso")
    plugin.write_bytes(b"fake-edge-plugin")
    return {
        "engine_dir": str(_fake_engine(tmp_path / "prebuilt-engine")),
        "runtime_library": str(runtime),
        "runtime_plugin": str(plugin),
        "runtime_build": {"fake": True},
        "precision": "fp16",
        "quantization": "none",
        "max_input_length": 1024,
        "max_cache_length": 4096,
        "max_batch_size": 4,
    }


def _wrong_engine_value(value: object) -> object:
    if type(value) is bool:
        return not value
    if type(value) in (int, float):
        return value + 1
    if value is None:
        return "not-null"
    return f"{value}-wrong"


@pytest.mark.parametrize("field", tuple(ENGINE_MODEL_CONFIG))
def test_engine_validation_rejects_each_wrong_model_fingerprint_field(
    tmp_path: Path, field: str
) -> None:
    adapter = _load_adapter_module()
    engine = _fake_engine(tmp_path / "wrong-engine")
    config_path = engine / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = _wrong_engine_value(config[field])
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(adapter.AdapterError, match=rf"config\.{field} must be exactly"):
        adapter._validate_engine_directory(engine)


@pytest.mark.parametrize("field", tuple(ENGINE_MODEL_CONFIG))
def test_engine_validation_rejects_each_missing_model_field(tmp_path: Path, field: str) -> None:
    adapter = _load_adapter_module()
    engine = _fake_engine(tmp_path / "missing-model-field-engine")
    config_path = engine / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config[field]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(adapter.AdapterError, match=rf"config\.{field} must be exactly"):
        adapter._validate_engine_directory(engine)


@pytest.mark.parametrize("reduced_vocab_size", (None, 0, 1, False, "0"))
def test_engine_validation_allows_only_null_reduced_vocab_size(
    tmp_path: Path, reduced_vocab_size: object
) -> None:
    adapter = _load_adapter_module()
    engine = _fake_engine(tmp_path / "reduced-vocab-engine")
    config_path = engine / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["reduced_vocab_size"] = reduced_vocab_size
    config_path.write_text(json.dumps(config), encoding="utf-8")

    if reduced_vocab_size is None:
        assert adapter._validate_engine_directory(engine)[1] == ENGINE_MODEL_CONFIG["vocab_size"]
    else:
        with pytest.raises(
            adapter.AdapterError, match=r"reduced_vocab_size must be absent or null"
        ):
            adapter._validate_engine_directory(engine)


@pytest.mark.parametrize("field,value", (("tp_size", 1), ("tp_rank", 0)))
def test_engine_validation_rejects_tensor_parallel_metadata(
    tmp_path: Path, field: str, value: int
) -> None:
    adapter = _load_adapter_module()
    engine = _fake_engine(tmp_path / f"{field}-engine")
    config_path = engine / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(adapter.AdapterError, match=rf"config\.{field} must be absent"):
        adapter._validate_engine_directory(engine)


@pytest.mark.parametrize("field", tuple(ENGINE_BUILDER_CONFIG))
def test_engine_validation_rejects_each_wrong_builder_field(tmp_path: Path, field: str) -> None:
    adapter = _load_adapter_module()
    engine = _fake_engine(tmp_path / "wrong-builder-engine")
    config_path = engine / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["builder_config"][field] = _wrong_engine_value(config["builder_config"][field])
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(adapter.AdapterError, match=rf"builder_config\.{field} must be exactly"):
        adapter._validate_engine_directory(engine)


@pytest.mark.parametrize("field", tuple(ENGINE_BUILDER_CONFIG))
def test_engine_validation_rejects_each_missing_builder_field(tmp_path: Path, field: str) -> None:
    adapter = _load_adapter_module()
    engine = _fake_engine(tmp_path / "missing-builder-engine")
    config_path = engine / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["builder_config"][field]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(adapter.AdapterError, match=rf"builder_config\.{field} must be exactly"):
        adapter._validate_engine_directory(engine)


@pytest.mark.parametrize("filename", REQUIRED_ENGINE_FILES)
def test_engine_validation_rejects_each_missing_required_artifact(
    tmp_path: Path, filename: str
) -> None:
    adapter = _load_adapter_module()
    engine = _fake_engine(tmp_path / "missing-artifact-engine")
    (engine / filename).unlink()

    with pytest.raises(adapter.AdapterError, match=rf"missing required artifact {filename}"):
        adapter._validate_engine_directory(engine)


@pytest.mark.parametrize(
    "sibling_config,field,expected",
    (
        ({"hidden_size": 1024, "intermediate_size": 3072}, "hidden_size", 2048),
        ({"hidden_size": 2560, "num_hidden_layers": 36}, "hidden_size", 2048),
    ),
)
def test_engine_validation_rejects_qwen3_sibling_engines(
    tmp_path: Path, sibling_config: dict[str, object], field: str, expected: object
) -> None:
    adapter = _load_adapter_module()
    engine = _fake_engine(tmp_path / "sibling-engine")
    config_path = engine / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(sibling_config)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        adapter.AdapterError, match=rf"config\.{field} must be exactly {expected!r}"
    ):
        adapter._validate_engine_directory(engine)


def _run_build_after_probe(manifest, request, output: Path):
    probe = run_probe(manifest, request)
    assert probe.supported, probe.reason
    return run_build(manifest, request, output, probe=probe)


def _make_cuda_toolkit(
    root: Path,
    *,
    encoded_header_version: int = 13030,
    compiler_version: str = "13.3",
) -> tuple[Path, Path, Path, Path]:
    include = root / "include"
    include.mkdir(parents=True)
    (include / "cuda_runtime_api.h").write_text("/* fixture */\n", encoding="utf-8")
    (include / "cuda.h").write_text(
        f"#define CUDA_VERSION {encoded_header_version}\n", encoding="utf-8"
    )
    compiler = root / "bin" / "nvcc"
    compiler.parent.mkdir()
    compiler.write_text(
        "#!/bin/sh\n"
        f"printf 'Cuda compilation tools, release {compiler_version}, "
        f"V{compiler_version}.0\\n'\n",
        encoding="utf-8",
    )
    compiler.chmod(0o755)
    cudart = root / "lib64" / "libcudart.so"
    cudart.parent.mkdir()
    cudart.write_bytes(b"test cudart")
    driver = root / "lib64" / "stubs" / "libcuda.so"
    driver.parent.mkdir()
    driver.write_bytes(b"test CUDA driver stub")
    return include, compiler, cudart, driver


def _make_edge_source(root: Path) -> Path:
    for relative in (
        "CMakeLists.txt",
        "cpp/common/version.h",
        "3rdParty/nlohmannJson/include/nlohmann/json.hpp",
        "tensorrt_edgellm/scripts/export.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    return root


def test_manifest_profile_and_dependency_pins_are_exact_and_capsule_owned() -> None:
    manifest = load_implementation_manifest(MANIFEST_PATH)
    with PROFILE_PATH.open("rb") as profile_file:
        profile = tomllib.load(profile_file)
    with (CAPSULE_ROOT / "dependency.lock").open("rb") as dependency_file:
        dependency = tomllib.load(dependency_file)

    assert manifest.implementation_id == IMPLEMENTATION_ID
    assert manifest.runtime_library == RUNTIME_LIBRARY
    assert manifest.runtime_abi == 1
    assert manifest.matches(_request())
    assert manifest.matches(_request(target=_target(gpu_count=8)))
    assert not manifest.matches(_request(model_id="Qwen/Qwen3-0.6B"))
    assert not manifest.matches(_request(model_id="Qwen/Qwen3-4B-Instruct-2507"))
    assert not manifest.matches(_request(revision="0" * 40))
    assert not manifest.matches(_request(target=_target(gpu_architecture="sm90")))
    assert not manifest.matches(_request(target=_target(platform_kind="soc")))
    assert not manifest.matches(_request(target=_target(gpu_name="NVIDIA A100-SXM4-80GB")))

    assert dependency["downstream"] == {
        "name": "tensorrt-edge-llm",
        "source": EDGE_SOURCE,
        "version": "0.9.0",
        "tag": "v0.9.0",
        "commit": EDGE_COMMIT,
        "source_mode": "git",
    }
    assert dependency["tensorrt"] == {"version": "11.2.0.113"}
    assert dependency["cuda"] == {"version": "13.3"}
    exporter_python = dependency["exporter_python"]
    assert exporter_python["direct"] == {
        "torch": "2.12.0",
        "transformers": "5.9.0",
        "onnx": "1.19.0",
        "onnxscript": "0.7.0",
        "safetensors": "0.7.0",
        "numpy": "2.4.6",
        "onnx-graphsurgeon": "0.6.1",
    }
    assert {
        field: exporter_python[field]
        for field in (
            "implementation",
            "version",
            "platform",
            "architecture",
            "abi",
            "wheel_target",
            "lock_format",
            "resolver",
            "package_count",
        )
    } == {
        "implementation": "CPython",
        "version": "3.12",
        "platform": "linux",
        "architecture": "x86_64",
        "abi": "cp312",
        "wheel_target": "x86_64-manylinux_2_28",
        "lock_format": "pip-require-hashes-v1",
        "resolver": "uv==0.11.29",
        "package_count": 60,
    }
    _requirements, locked_packages = _load_adapter_module()._parse_exporter_lock(exporter_python)
    assert len(locked_packages) == 60
    assert locked_packages["filelock"] == "3.31.1"
    assert locked_packages["nvidia-cudnn-cu13"] == "9.20.0.48"
    assert locked_packages["pip"] == "26.1.2"
    assert profile["profile_id"] == PROFILE_ID
    assert "model" not in profile
    assert "target" not in profile
    assert "versions" not in profile
    _load_adapter_module()._validate_capsule_data(profile)


def test_adapter_restores_parent_active_device_for_heterogeneous_visible_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _load_adapter_module()
    calls: list[tuple[str, int]] = []
    selected = {"ordinal": 0}

    class FakeCudaRuntime:
        class cudaError_t:
            cudaSuccess = 0

        @staticmethod
        def cudaSetDevice(ordinal: int):
            calls.append(("set", ordinal))
            selected["ordinal"] = ordinal
            return (0,)

        @staticmethod
        def cudaGetDevice():
            calls.append(("get", selected["ordinal"]))
            return 0, selected["ordinal"]

    monkeypatch.setattr(adapter, "_cuda_runtime", lambda: FakeCudaRuntime)
    monkeypatch.setenv("TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-slow,GPU-a100")

    adapter._select_parent_active_cuda_device()

    assert calls == [("set", 1), ("get", 1)]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-a100"


def test_supported_probe_selects_profile_without_hidden_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("_TRTMC_INTERNAL_QWEN3_1_7B_ALLOW_FAKE_RUNTIME_BUILD", "1")
    manifest = load_implementation_manifest(MANIFEST_PATH)
    parameters = _prebuilt_runtime_parameters(tmp_path)

    probe = run_probe(manifest, _request(parameters=parameters))

    assert probe.supported
    assert probe.profile_id == PROFILE_ID
    assert not probe.reason

    unsupported = run_probe(
        manifest,
        _request(parameters={**parameters, "precision": "fp32"}),
    )
    assert not unsupported.supported
    assert "precision='fp32'" in unsupported.reason

    unsupported_option = run_probe(
        manifest,
        _request(parameters={**parameters, "public_options": {"dynamic_kv_cache": True}}),
    )
    assert not unsupported_option.supported
    assert "dynamic_kv_cache=true" in unsupported_option.reason

    with pytest.raises(BuildAdapterError, match="unsupported parameters: deployment"):
        run_probe(
            manifest,
            _request(parameters={**parameters, "deployment": {"target": "current"}}),
        )


def test_established_mc_defaults_do_not_change_the_requested_profile() -> None:
    import inspect

    from tensorrt_model_connect.engine_builder import build

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(build).parameters.items()
        if name not in {"model_id_or_path", "output_path"}
    }

    reason = _load_adapter_module()._public_option_reason(defaults)
    assert "requires public option max_batch_size=4; got 1" in reason


@pytest.mark.parametrize("value", (None, False, "", (), [], {}))
def test_future_inert_public_option_does_not_couple_to_this_adapter(value) -> None:
    reason = _load_adapter_module()._public_option_reason({"future_option": value})
    assert reason == ""


def test_future_non_inert_public_option_remains_fail_closed() -> None:
    reason = _load_adapter_module()._public_option_reason({"future_option": "requested"})
    assert "does not recognize public option(s): future_option" in reason


def test_public_cli_requires_the_exact_qualified_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.build_cli as build_cli

    captured = {}

    def capture(args) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(build_cli, "_cmd_build", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        ["trtmc", "build", MODEL_ID, "-o", "/tmp/qwen-edge-test.trtfb"],
    )
    with pytest.raises(SystemExit) as exit_info:
        build_cli.main()

    assert exit_info.value.code == 0
    options = build_cli._optimized_cli_public_options(captured["args"])
    reason = _load_adapter_module()._public_option_reason(options)
    assert "requires public option max_batch_size=4; got 1" in reason

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trtmc",
            "build",
            MODEL_ID,
            "-o",
            "/tmp/qwen-edge-test.trtfb",
            "--precision",
            "fp16",
            "--max-cache-length",
            "4096",
            "--max-batch-size",
            "4",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        build_cli.main()

    assert exit_info.value.code == 0
    options = build_cli._optimized_cli_public_options(captured["args"])
    assert _load_adapter_module()._public_option_reason(options) == ""


def test_qualified_probe_is_side_effect_free_and_does_not_require_ambient_exporter_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = _load_adapter_module()
    _qualify_exporter_host(adapter, monkeypatch)
    monkeypatch.setattr(adapter, "_select_parent_active_cuda_device", lambda: None)
    monkeypatch.setattr(adapter.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        adapter.importlib.util,
        "find_spec",
        lambda module: object() if module in {"venv", "ensurepip"} else None,
    )
    monkeypatch.setattr(adapter, "_resolve_tensorrt", lambda _settings: object())
    monkeypatch.setattr(adapter, "_resolve_cuda", lambda _settings: object())

    payload = _request().to_json()
    payload["implementation_id"] = adapter.IMPLEMENTATION_ID
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    profile_root = tmp_path / "python-profiles"
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_ROOT", str(profile_root))
    monkeypatch.setattr(
        adapter,
        "_materialize_exporter_python",
        lambda: pytest.fail("probe must not materialize the exporter environment"),
    )

    assert adapter.main(["probe", "--request", str(request_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "profile_id": PROFILE_ID,
        "schema_version": 1,
        "supported": True,
    }
    assert not captured.err
    assert not profile_root.exists()


def test_exporter_profile_identity_binds_the_exact_leaf_dependency_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    lock = tmp_path / "dependency.lock"
    lock.write_bytes((CAPSULE_ROOT / "dependency.lock").read_bytes())
    monkeypatch.setattr(adapter, "DEPENDENCY_PATH", lock)

    first = adapter._exporter_profile_identity()
    lock.write_bytes(lock.read_bytes() + b"\n# identity mutation\n")
    second = adapter._exporter_profile_identity()

    assert first != second


def test_exporter_profile_materializes_atomically_once_and_reuses_the_ready_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    _qualify_exporter_host(adapter, monkeypatch)
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_ROOT", str(tmp_path / "profiles"))
    calls: list[list[str]] = []
    verified: list[Path] = []

    def fake_run(
        command: list[str],
        _description: str,
        *,
        environment=None,
        cwd=None,
    ) -> None:
        assert environment is not None
        assert cwd is not None
        calls.append(command)
        if command[1:4] == ["-I", "-m", "venv"]:
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)

    monkeypatch.setattr(adapter, "_run_checked", fake_run)
    monkeypatch.setattr(
        adapter,
        "_verify_exporter_python",
        lambda python: verified.append(python),
    )

    cold = adapter._materialize_exporter_python()
    warm = adapter._materialize_exporter_python()

    assert cold == warm
    assert cold.is_file()
    assert (cold.parents[1] / ".ready").is_file()
    assert len(calls) == 2
    assert calls[0][1:4] == ["-I", "-m", "venv"]
    assert calls[1][1:6] == ["-I", "-m", "pip", "--isolated", "install"]
    assert {"--only-binary=:all:", "--require-hashes", "--no-deps"} <= set(calls[1])
    requirements = cold.parents[1] / "requirements.lock.txt"
    assert "filelock==3.31.1 --hash=sha256:" in requirements.read_text(encoding="utf-8")
    assert len(verified) == 2
    assert not list((tmp_path / "profiles").glob(f"{adapter._EXPORTER_PROFILE_NAME}-*/.ready.tmp"))


def test_concurrent_exporter_profile_materialization_publishes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    _qualify_exporter_host(adapter, monkeypatch)
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_ROOT", str(tmp_path / "profiles"))
    venv_started = threading.Event()
    finish_first = threading.Event()
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        _description: str,
        *,
        environment=None,
        cwd=None,
    ) -> None:
        del environment, cwd
        calls.append(command)
        if command[1:4] == ["-I", "-m", "venv"]:
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
            venv_started.set()
            assert finish_first.wait(timeout=10)

    monkeypatch.setattr(adapter, "_run_checked", fake_run)
    monkeypatch.setattr(adapter, "_verify_exporter_python", lambda _python: None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(adapter._materialize_exporter_python)
        assert venv_started.wait(timeout=10)
        second = executor.submit(adapter._materialize_exporter_python)
        finish_first.set()
        resolved = [first.result(timeout=10), second.result(timeout=10)]

    assert resolved[0] == resolved[1]
    assert sum(command[1:4] == ["-I", "-m", "venv"] for command in calls) == 1
    assert sum("install" in command for command in calls) == 1
    profile_root = tmp_path / "profiles"
    assert len(list(profile_root.glob(f"{adapter._EXPORTER_PROFILE_NAME}-*/.ready"))) == 1
    assert not [
        path for path in profile_root.iterdir() if path.is_dir() and not (path / ".ready").is_file()
    ]


def test_exporter_profile_install_failure_never_publishes_a_ready_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    _qualify_exporter_host(adapter, monkeypatch)
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_ROOT", str(tmp_path / "profiles"))

    attempts = {"pip": 0}

    def fail_install(
        command: list[str],
        _description: str,
        *,
        environment=None,
        cwd=None,
    ) -> None:
        del environment, cwd
        if command[1:4] == ["-I", "-m", "venv"]:
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
            return
        attempts["pip"] += 1
        if attempts["pip"] == 1:
            raise adapter.AdapterError("synthetic pip failure")

    monkeypatch.setattr(adapter, "_run_checked", fail_install)
    monkeypatch.setattr(adapter, "_verify_exporter_python", lambda _python: None)

    with pytest.raises(adapter.AdapterError, match="synthetic pip failure"):
        adapter._materialize_exporter_python()

    profile_root = tmp_path / "profiles"
    assert not list(profile_root.glob("*/.ready"))
    assert not [path for path in profile_root.iterdir() if path.is_dir()]

    recovered = adapter._materialize_exporter_python()

    assert recovered.is_file()
    assert (recovered.parents[1] / ".ready").is_file()
    assert attempts["pip"] == 2


def test_exporter_profile_prebuilt_only_probe_is_read_only_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    _qualify_exporter_host(adapter, monkeypatch)
    profile_root = tmp_path / "profiles"
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_ROOT", str(profile_root))
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_PREBUILT_ONLY", "1")

    with pytest.raises(adapter.AdapterError, match="not prebuilt or is corrupt"):
        adapter._probe_exporter_python()

    assert not profile_root.exists()


@pytest.mark.parametrize(
    ("libc", "confstr", "accepted"),
    (
        (("glibc", "2.28"), None, True),
        (("", ""), "glibc 2.39", True),
        (("glibc", "2.27"), "glibc 2.39", False),
        (("", ""), None, False),
    ),
)
def test_exporter_host_enforces_manylinux_2_28_glibc_floor(
    monkeypatch: pytest.MonkeyPatch,
    libc: tuple[str, str],
    confstr: str | None,
    accepted: bool,
) -> None:
    adapter = _load_adapter_module()
    _qualify_exporter_host(adapter, monkeypatch)
    monkeypatch.setattr(adapter.platform, "libc_ver", lambda: libc)
    monkeypatch.setattr(adapter.os, "confstr", lambda _name: confstr)

    if accepted:
        adapter._validate_exporter_host()
    else:
        with pytest.raises(adapter.AdapterError, match=r"glibc >=2\.28"):
            adapter._validate_exporter_host()


def test_exporter_python_rejects_wrong_exact_package_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _load_adapter_module()
    _requirements, actual = adapter._load_exporter_lock()
    actual["transformers"] = "5.2.0"
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["python"], 0, json.dumps(actual), ""
        ),
    )

    with pytest.raises(adapter.AdapterError, match='"actual": "5.2.0"'):
        adapter._verify_exporter_python(Path(sys.executable))


def test_exporter_python_rejects_transitive_dependency_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _load_adapter_module()
    _requirements, actual = adapter._load_exporter_lock()
    actual["filelock"] = "999.0"
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["python"], 0, json.dumps(actual), ""
        ),
    )

    with pytest.raises(adapter.AdapterError, match='"filelock"'):
        adapter._verify_exporter_python(Path(sys.executable))


def test_edge_export_bootstrap_ignores_caller_cwd_and_pythonpath_poison(
    tmp_path: Path,
) -> None:
    adapter = _load_adapter_module()
    pinned_source = tmp_path / "pinned-edge"
    poison_source = tmp_path / "caller-cwd"
    output = tmp_path / "output"
    output.mkdir()

    for source, selected in ((pinned_source, "pinned"), (poison_source, "poison")):
        package = source / "tensorrt_edgellm" / "scripts"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "export.py").write_text(
            "from pathlib import Path\n"
            "import sys\n"
            f"Path(sys.argv[2], 'selected.txt').write_text('{selected}', encoding='utf-8')\n",
            encoding="utf-8",
        )

    command = adapter._edge_export_command(
        Path(sys.executable),
        pinned_source,
        tmp_path / "model",
        output,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(poison_source)
    environment["PYTHONHOME"] = str(poison_source)

    result = subprocess.run(
        command,
        cwd=poison_source,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert command[1:3] == ["-I", "-c"]
    assert (output / "selected.txt").read_text(encoding="utf-8") == "pinned"


def test_production_probe_rejects_unqualified_prebuilt_payload_overrides(
    tmp_path: Path,
) -> None:
    manifest = load_implementation_manifest(MANIFEST_PATH)
    with pytest.raises(BuildAdapterError, match="payload overrides are test-only"):
        run_probe(manifest, _request(parameters=_prebuilt_runtime_parameters(tmp_path)))


def test_fake_probe_and_build_require_the_same_internal_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    monkeypatch.delenv("_TRTMC_INTERNAL_QWEN3_1_7B_ALLOW_FAKE_RUNTIME_BUILD", raising=False)
    reason = "Fake runtime builds require _TRTMC_INTERNAL_QWEN3_1_7B_ALLOW_FAKE_RUNTIME_BUILD=1"
    manifest = load_implementation_manifest(MANIFEST_PATH)
    probe = run_probe(
        manifest,
        _request(parameters={"runtime_build": {"fake": True}}),
    )
    assert not probe.supported
    assert probe.reason == reason
    with pytest.raises(adapter.AdapterError, match=reason):
        adapter._build_runtime_dso(
            tmp_path / "output",
            {"runtime_build": {"fake": True}},
            "0" * 64,
        )


def test_build_stages_test_only_engine_runtime_and_plugin_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("_TRTMC_INTERNAL_QWEN3_1_7B_ALLOW_FAKE_RUNTIME_BUILD", "1")
    parameters = _prebuilt_runtime_parameters(tmp_path)
    manifest = load_implementation_manifest(MANIFEST_PATH)
    request = _request(parameters=parameters)

    build = _run_build_after_probe(manifest, request, tmp_path / "capsule-output")

    assert (build.artifacts_path / "engine.dir" / "llm.engine").read_bytes() == (b"fake-llm.engine")
    assert (build.artifacts_path / RUNTIME_LIBRARY).read_bytes() == b"fake-runtime-dso"
    assert (build.artifacts_path / RUNTIME_PLUGIN).read_bytes() == b"fake-edge-plugin"
    assert build.descriptor["profile_id"] == PROFILE_ID
    assert build.descriptor["operation"] == "text-generation-v1"
    assert build.descriptor["runtime"] == {
        "abi": 1,
        "library": RUNTIME_LIBRARY,
        "plugin": RUNTIME_PLUGIN,
    }
    assert build.descriptor["limits"] == {
        "max_batch_size": 4,
        "max_cache_length": 4096,
        "max_input_length": 1024,
        "vocab_size": 151936,
    }
    assert build.descriptor["versions"]["edge_llm"] == "0.9.0"
    assert build.descriptor["versions"]["cuda"] == "13.3"
    assert build.descriptor["bundle_info"]["family"] == "qwen"
    assert build.descriptor["bundle_info"]["model_type"] == "qwen3"
    assert build.descriptor["bundle_config"]["family"] == "qwen"
    assert build.descriptor["bundle_config"]["model_type"] == "qwen3"
    assert build.descriptor["bundle_config"]["runtime_provider"] == IMPLEMENTATION_ID
    assert "metadata" not in build.descriptor

    bundle = write_optimized_bundle(tmp_path / "qwen3-edge.trtfb", manifest, request, build)
    assert bundle.is_file()
    assert bundle.stat().st_size > sum(
        path.stat().st_size for path in build.artifacts_path.rglob("*") if path.is_file()
    )


def test_qualified_probe_reports_wrong_tensorrt_without_bootstrapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _load_adapter_module()
    _qualify_exporter_host(adapter, monkeypatch)

    def wrong_tensorrt(_runtime_build):
        raise adapter.AdapterError("found TensorRT 10.14.1.48, not 11.2.0.113")

    def forbidden_bootstrap(*_args, **_kwargs):
        raise AssertionError("unsupported probe must not bootstrap Edge-LLM")

    monkeypatch.setattr(adapter, "_resolve_tensorrt", wrong_tensorrt)
    monkeypatch.setattr(adapter.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda _module: object())
    monkeypatch.setattr(adapter, "_resolve_cuda", forbidden_bootstrap)
    monkeypatch.setattr(adapter, "_resolve_edge_source", forbidden_bootstrap)
    monkeypatch.setattr(adapter, "_resolve_edge_dependency", forbidden_bootstrap)
    monkeypatch.setattr(adapter, "_run_checked", forbidden_bootstrap)

    payload = _request().to_json()
    payload["implementation_id"] = adapter.IMPLEMENTATION_ID
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    assert adapter.main(["probe", "--request", str(request_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "reason": "Qwen Edge-LLM software profile is unavailable: "
        "found TensorRT 10.14.1.48, not 11.2.0.113",
        "schema_version": 1,
        "supported": False,
    }
    assert not captured.err


def test_tensorrt_resolution_requires_one_exact_core_and_parser_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    include = tmp_path / "include"
    include.mkdir()
    (include / "NvInfer.h").write_text("// fixture\n", encoding="utf-8")
    version_header = include / "NvInferVersion.h"
    version_header.write_text(
        "#define NV_TENSORRT_MAJOR 11\n"
        "#define NV_TENSORRT_MINOR 2\n"
        "#define NV_TENSORRT_PATCH 0\n"
        "#define NV_TENSORRT_BUILD 113\n",
        encoding="utf-8",
    )
    (include / "NvOnnxParser.h").write_text("// fixture\n", encoding="utf-8")
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    core = library_dir / "libnvinfer.so.11.2.0.113"
    parser = library_dir / "libnvonnxparser.so.11"
    core.write_bytes(b"exact TensorRT core")
    parser.write_bytes(b"exact TensorRT parser")
    runtime_build = {
        "tensorrt_include_dir": str(include),
        "tensorrt_library": str(core),
        "onnx_parser_include_dir": str(include),
        "onnx_parser_library": str(parser),
    }
    monkeypatch.setattr(adapter, "_library_tensorrt_version", lambda _library: (11, 2, 0, 113))

    resolved = adapter._resolve_tensorrt(runtime_build)
    assert resolved.library == core
    assert resolved.onnx_parser_library == parser

    version_header.write_text(
        version_header.read_text(encoding="utf-8").replace(
            "#define NV_TENSORRT_BUILD 113", "#define NV_TENSORRT_BUILD 112"
        ),
        encoding="utf-8",
    )
    with pytest.raises(adapter.AdapterError, match="11.2.0.112, not 11.2.0.113"):
        adapter._resolve_tensorrt(runtime_build)


@pytest.mark.parametrize(
    ("header_version", "compiler_version"),
    ((12080, "13.3"), (13030, "12.8")),
)
def test_cuda_resolution_rejects_unsupported_toolkit_components(
    tmp_path: Path,
    header_version: int,
    compiler_version: str,
) -> None:
    adapter = _load_adapter_module()
    include, _compiler, cudart, _driver = _make_cuda_toolkit(
        tmp_path / "cuda",
        encoded_header_version=header_version,
        compiler_version=compiler_version,
    )

    with pytest.raises(adapter.AdapterError, match=r"not 13\.3"):
        adapter._resolve_cuda({"cuda_include_dir": str(include), "cudart_library": str(cudart)})


def test_cuda_resolution_pins_one_coherent_13_3_toolkit(tmp_path: Path) -> None:
    adapter = _load_adapter_module()
    cuda_root = tmp_path / "cuda"
    include, compiler, cudart, driver = _make_cuda_toolkit(cuda_root)

    resolved = adapter._resolve_cuda(
        {"cuda_include_dir": str(include), "cudart_library": str(cudart)}
    )

    assert resolved.root == cuda_root.resolve()
    assert resolved.include_dir == include.resolve()
    assert resolved.cudart_library == cudart.resolve()
    assert resolved.driver_library == driver.resolve()
    assert resolved.compiler == compiler.resolve()
    assert resolved.version == "13.3"

    explicit_driver = cuda_root / "lib64" / "libcuda.so"
    explicit_driver.write_bytes(b"explicit CUDA driver library")
    explicitly_resolved = adapter._resolve_cuda(
        {
            "cuda_include_dir": str(include),
            "cudart_library": str(cudart),
            "cuda_driver_library": str(explicit_driver),
        }
    )
    assert explicitly_resolved.driver_library == explicit_driver.resolve()

    other_root = tmp_path / "other-cuda"
    _other_include, _other_compiler, other_cudart, other_driver = _make_cuda_toolkit(other_root)
    with pytest.raises(adapter.AdapterError, match="same exact CUDA 13.3 toolkit"):
        adapter._resolve_cuda(
            {"cuda_include_dir": str(include), "cudart_library": str(other_cudart)}
        )
    with pytest.raises(adapter.AdapterError, match="same exact CUDA 13.3 toolkit"):
        adapter._resolve_cuda(
            {
                "cuda_include_dir": str(include),
                "cudart_library": str(cudart),
                "cuda_driver_library": str(other_driver),
            }
        )


def test_cuda_resolution_requires_toolkit_driver_payload(tmp_path: Path) -> None:
    adapter = _load_adapter_module()
    include, _compiler, cudart, driver = _make_cuda_toolkit(tmp_path / "cuda")
    driver.unlink()

    with pytest.raises(adapter.AdapterError, match="CUDA driver stub or library"):
        adapter._resolve_cuda({"cuda_include_dir": str(include), "cudart_library": str(cudart)})


def test_selected_build_lazily_clones_exact_edge_tag_and_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("_TRTMC_INTERNAL_OPTIMIZED_RUNTIME_DEPENDENCY_ROOT", raising=False)
    adapter = _load_adapter_module()
    output = tmp_path / "output"
    output.mkdir()
    acquired = output / ".edge-source"
    commands: list[list[str]] = []

    def validate(path: Path) -> Path:
        if path == acquired and len(commands) == 3:
            return acquired
        raise adapter.AdapterError("source not present")

    monkeypatch.setattr(adapter, "_validate_edge_source", validate)
    monkeypatch.setattr(
        adapter, "_run_checked", lambda command, _description: commands.append(command)
    )

    assert commands == []
    assert adapter._resolve_edge_source(output, {}) == acquired
    assert commands[0] == [
        "git",
        "clone",
        "--branch",
        "v0.9.0",
        "--single-branch",
        "--no-checkout",
        EDGE_SOURCE,
        str(acquired),
    ]
    assert commands[1][-2:] == ["--detach", EDGE_COMMIT]
    assert commands[2][-4:] == ["submodule", "update", "--init", "--recursive"]


def test_selected_build_uses_ci_staged_pinned_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    dependency_root = tmp_path / "dependencies"
    cached = dependency_root / "tensorrt-edge-llm" / EDGE_COMMIT
    cached.mkdir(parents=True)
    monkeypatch.setenv("_TRTMC_INTERNAL_OPTIMIZED_RUNTIME_DEPENDENCY_ROOT", str(dependency_root))
    monkeypatch.setattr(adapter, "_validate_edge_source", lambda path: path.resolve())
    monkeypatch.setattr(
        adapter,
        "_run_checked",
        lambda *_args: pytest.fail("a staged CI dependency must not be fetched again"),
    )

    assert adapter._resolve_edge_source(tmp_path / "output", {}) == cached.resolve()


def test_edge_source_requires_the_pinned_clean_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    source = _make_edge_source(tmp_path / "edge-source")
    source_status = {"value": ""}

    def inspect(command: list[str], _description: str) -> str:
        if command[-2:] == ["rev-parse", "HEAD"]:
            return EDGE_COMMIT
        if command[-3:] == ["remote", "get-url", "origin"]:
            return EDGE_SOURCE
        if "status" in command and "submodule" not in command:
            return source_status["value"]
        return ""

    monkeypatch.setattr(adapter, "_run_capture", inspect)
    assert adapter._validate_edge_source(source) == source.resolve()

    source_status["value"] = " M cpp/common/version.h"
    with pytest.raises(adapter.AdapterError, match="must have no tracked, untracked, or ignored"):
        adapter._validate_edge_source(source)


def test_build_can_compile_the_capsule_runtime_only_after_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _fake_engine(tmp_path / "prebuilt-engine")
    plugin = tmp_path / "plugin-source.so"
    plugin.write_bytes(b"fake-edge-plugin")
    monkeypatch.setenv("_TRTMC_INTERNAL_QWEN3_1_7B_ALLOW_FAKE_RUNTIME_BUILD", "1")
    manifest = load_implementation_manifest(MANIFEST_PATH)
    request = _request(
        parameters={
            "engine_dir": str(engine),
            "runtime_plugin": str(plugin),
            "runtime_build": {
                "fake": True,
                "parallel": 2,
                "nlohmann_json_include_dir": str(_nlohmann_json_include()),
            },
        }
    )

    output = tmp_path / "capsule-output"
    build = _run_build_after_probe(manifest, request, output)

    runtime = build.artifacts_path / RUNTIME_LIBRARY
    assert runtime.read_bytes().startswith(b"\x7fELF")
    assert not (output / ".runtime-build").exists()


def test_edge_toolchain_identity_hashes_tools_dependencies_headers_and_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()

    def fixture(path: Path, content: bytes | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content or path.name.encode("utf-8"))
        return path

    source = tmp_path / "edge"
    for relative in (
        "CMakeLists.txt",
        "cpp/runtime/llmInferenceRuntime.h",
        "cpp/common/version.h",
    ):
        fixture(source / relative)
    trt_include = tmp_path / "trt" / "include"
    for name in ("NvInfer.h", "NvInferVersion.h", "NvOnnxParser.h"):
        fixture(trt_include / name)
    cuda_root = tmp_path / "cuda"
    cuda_include = cuda_root / "include"
    for name in ("cuda.h", "cuda_runtime_api.h"):
        fixture(cuda_include / name)

    cc = fixture(tmp_path / "tools" / "cc")
    cxx = fixture(tmp_path / "tools" / "c++")
    cmake = fixture(tmp_path / "tools" / "cmake")
    linker = fixture(tmp_path / "tools" / "ld")
    archiver = fixture(tmp_path / "tools" / "ar")
    libstdcxx = fixture(tmp_path / "tools" / "libstdc++.so")
    nvcc = fixture(cuda_root / "bin" / "nvcc", b"nvcc-v1")
    trt_library = fixture(tmp_path / "trt" / "libnvinfer.so.11")
    onnx_library = fixture(tmp_path / "trt" / "libnvonnxparser.so.11")
    cudart = fixture(cuda_root / "libcudart.so")
    driver = fixture(cuda_root / "libcuda.so")
    tensorrt = adapter._TensorRtInstallation(trt_include, trt_library, trt_include, onnx_library)
    cuda = adapter._CudaInstallation(cuda_root, cuda_include, cudart, driver, nvcc, "13.3")

    def resolved_compiler(environment: str, _default: str) -> Path:
        return {"CC": cc, "CXX": cxx, "CMAKE_COMMAND": cmake}[environment]

    monkeypatch.setattr(adapter, "_resolved_compiler", resolved_compiler)
    monkeypatch.setattr(
        adapter,
        "_resolved_build_tool",
        lambda _environment, _compiler, program: {"ld": linker, "ar": archiver}[program],
    )
    monkeypatch.setattr(
        adapter,
        "_run_capture",
        lambda command, _description: (
            str(libstdcxx) if "-print-file-name=libstdc++.so" in command else "version"
        ),
    )

    baseline = adapter._host_toolchain_identity(source, tensorrt, cuda)
    nvcc.write_bytes(b"nvcc-v2")
    assert adapter._host_toolchain_identity(source, tensorrt, cuda).sha256 != baseline.sha256

    nvcc.write_bytes(b"nvcc-v1")
    (trt_include / "NvInfer.h").write_bytes(b"mutated header")
    assert adapter._host_toolchain_identity(source, tensorrt, cuda).sha256 != baseline.sha256

    (trt_include / "NvInfer.h").write_bytes(b"NvInfer.h")
    monkeypatch.setenv("CXXFLAGS", "-fno-semantic-interposition")
    assert adapter._host_toolchain_identity(source, tensorrt, cuda).sha256 != baseline.sha256


def test_selected_build_builds_only_required_edge_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    source = tmp_path / "edge-source"
    source.mkdir()
    trt_include = tmp_path / "trt-include"
    trt_include.mkdir()
    trt_library = tmp_path / "libnvinfer.so.11"
    trt_library.write_bytes(b"trt")
    onnx_library = tmp_path / "libnvonnxparser.so.11"
    onnx_library.write_bytes(b"onnx")
    cuda_root = tmp_path / "cuda"
    cuda_include = cuda_root / "include"
    cuda_include.mkdir(parents=True)
    cudart = cuda_root / "libcudart.so"
    cudart.write_bytes(b"cudart")
    cuda_driver = cuda_root / "libcuda.so"
    cuda_driver.write_bytes(b"cuda driver")
    cuda_compiler = cuda_root / "bin" / "nvcc"
    cuda_compiler.parent.mkdir()
    cuda_compiler.write_bytes(b"nvcc")
    tensorrt = adapter._TensorRtInstallation(trt_include, trt_library, trt_include, onnx_library)
    cuda = adapter._CudaInstallation(
        cuda_root, cuda_include, cudart, cuda_driver, cuda_compiler, "13.3"
    )
    calls: list[list[str]] = []
    validated_sources: list[Path] = []
    cc = Path(shutil.which("cc") or "/usr/bin/cc").resolve()
    cxx = Path(shutil.which("c++") or "/usr/bin/c++").resolve()
    linker = Path(shutil.which("ld") or "/usr/bin/ld").resolve()
    archiver = Path(shutil.which("ar") or "/usr/bin/ar").resolve()
    toolchain_identity = "1" * 64
    architecture = "x86_64"
    toolchain = adapter._EdgeBuildToolchain(
        cc,
        cxx,
        linker,
        archiver,
        Path("cmake"),
        toolchain_identity,
        architecture,
    )

    monkeypatch.setattr(adapter, "_resolve_edge_source", lambda *_args: source)
    monkeypatch.setattr(adapter, "_resolve_tensorrt", lambda _settings: tensorrt)
    monkeypatch.setattr(adapter, "_resolve_cuda", lambda _settings: cuda)
    monkeypatch.setattr(
        adapter,
        "_host_toolchain_identity",
        lambda _source, _tensorrt, _cuda: toolchain,
    )
    monkeypatch.setattr(
        adapter,
        "_validate_edge_source",
        lambda path: validated_sources.append(path) or path,
    )

    def fake_run(command: list[str], _description: str) -> None:
        calls.append(command)
        build_dir = tmp_path / "output" / ".edge-dependency-build"
        if command[:2] == ["cmake", "-S"]:
            build_dir.mkdir(parents=True)
            definitions = {
                argument[2:].split("=", 1)[0]: argument[2:].split("=", 1)[1]
                for argument in command
                if argument.startswith("-D") and "=" in argument
            }
            (build_dir / "CMakeCache.txt").write_text(
                "\n".join(
                    [
                        f"CMAKE_HOME_DIRECTORY:INTERNAL={source}",
                        *(f"{name}:STRING={value}" for name, value in definitions.items()),
                    ]
                ),
                encoding="utf-8",
            )
            return
        for product in (
            build_dir / "cpp" / "libedgellmCore.a",
            build_dir / "libNvInfer_edgellm_plugin.so.1.0",
            build_dir / "examples" / "llm" / "llm_build",
        ):
            product.parent.mkdir(parents=True, exist_ok=True)
            product.write_bytes(b"product")

    monkeypatch.setattr(adapter, "_run_checked", fake_run)
    output = tmp_path / "output"

    assert calls == []
    dependency = adapter._resolve_edge_dependency(output, {"parallel": 3})

    assert dependency.source_dir == source
    assert dependency.build_tool.name == "llm_build"
    assert validated_sources == [source, source]
    configure = calls[0]
    assert "-DCMAKE_CUDA_ARCHITECTURES=80" in configure
    assert "-DAARCH64_BUILD=OFF" in configure
    assert "-DCMAKE_SKIP_RPATH=ON" in configure
    assert "-DBUILD_PYTHON_BINDINGS=OFF" in configure
    assert "-DENABLE_CUTE_DSL=OFF" in configure
    assert f"-DTensorRT_INCLUDE_DIR={trt_include}" in configure
    assert f"-DTensorRT_LIBRARY={trt_library}" in configure
    assert f"-DTensorRT_OnnxParser_INCLUDE_DIR={trt_include}" in configure
    assert f"-DTensorRT_OnnxParser_LIBRARY={onnx_library}" in configure
    assert "-DCUDA_CTK_VERSION=13.3" in configure
    assert f"-DCUDA_DRIVER_LIB={cuda_driver}" in configure
    assert f"-DCMAKE_AR={archiver}" in configure
    assert f"-DCMAKE_LINKER={linker}" in configure
    assert calls[1][-4:] == [
        "--target",
        "edgellmCore",
        "NvInfer_edgellm_plugin",
        "llm_build",
    ]
    reused = adapter._resolve_edge_dependency(
        output,
        {"parallel": 3, "edge_llm_build_dir": str(dependency.build_dir)},
    )
    assert reused.build_dir == dependency.build_dir
    assert len(calls) == 2

    stamp = dependency.build_dir / adapter._EDGE_BUILD_STAMP
    stamp_text = stamp.read_text(encoding="utf-8")
    stamp_data = json.loads(stamp_text)
    assert set(stamp_data["products"]) == {
        "cpp/libedgellmCore.a",
        "examples/llm/llm_build",
        "libNvInfer_edgellm_plugin.so.1.0",
    }
    assert stamp_data["recipe_sha256"] == adapter._canonical_sha256(stamp_data["recipe"])

    dependency.plugin.write_bytes(b"mutated plugin")
    with pytest.raises(adapter.AdapterError, match="does not match the pinned source"):
        adapter._resolve_edge_dependency(
            output,
            {"parallel": 3, "edge_llm_build_dir": str(dependency.build_dir)},
        )
    dependency.plugin.write_bytes(b"product")

    stamp.unlink()
    with pytest.raises(adapter.AdapterError, match="does not match the pinned source"):
        adapter._resolve_edge_dependency(
            output,
            {"parallel": 3, "edge_llm_build_dir": str(dependency.build_dir)},
        )
    stamp.write_text(stamp_text, encoding="utf-8")

    mutated_stamp = json.loads(stamp_text)
    mutated_stamp["recipe_sha256"] = "0" * 64
    stamp.write_text(json.dumps(mutated_stamp), encoding="utf-8")
    with pytest.raises(adapter.AdapterError, match="does not match the pinned source"):
        adapter._resolve_edge_dependency(
            output,
            {"parallel": 3, "edge_llm_build_dir": str(dependency.build_dir)},
        )
    stamp.write_text(stamp_text, encoding="utf-8")

    monkeypatch.setenv("CXXFLAGS", "-fno-semantic-interposition")
    with pytest.raises(adapter.AdapterError, match="does not match the pinned source"):
        adapter._resolve_edge_dependency(
            output,
            {"parallel": 3, "edge_llm_build_dir": str(dependency.build_dir)},
        )


def test_runtime_dso_build_receives_exact_cuda_driver_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    source = tmp_path / "edge-source"
    source.mkdir()
    edge_build = tmp_path / "edge-build"
    edge_build.mkdir()
    include = tmp_path / "include"
    include.mkdir()
    placeholder = tmp_path / "placeholder"
    placeholder.write_bytes(b"fixture")
    driver = tmp_path / "cuda" / "lib64" / "stubs" / "libcuda.so"
    driver.parent.mkdir(parents=True)
    driver.write_bytes(b"CUDA driver stub")
    selected_cc = Path(shutil.which("cc") or "/usr/bin/cc").resolve()
    selected_cxx = Path(shutil.which("c++") or "/usr/bin/c++").resolve()
    toolchain = adapter._EdgeBuildToolchain(
        selected_cc,
        selected_cxx,
        Path(shutil.which("ld") or "/usr/bin/ld").resolve(),
        Path(shutil.which("ar") or "/usr/bin/ar").resolve(),
        Path("cmake"),
        "2" * 64,
        "x86_64",
    )
    dependency = adapter._EdgeDependency(
        source,
        edge_build,
        placeholder,
        placeholder,
        adapter._TensorRtInstallation(include, placeholder, include, placeholder),
        adapter._CudaInstallation(
            tmp_path / "cuda", include, placeholder, driver, placeholder, "13.3"
        ),
        toolchain,
    )
    calls: list[list[str]] = []

    monkeypatch.setattr(adapter, "_resolve_edge_dependency", lambda *_args: dependency)
    monkeypatch.setattr(adapter, "_validate_edge_source", lambda path: path)
    monkeypatch.setattr(adapter, "_validate_packaged_elf", lambda _path: None)

    def fake_run(command: list[str], _description: str) -> None:
        calls.append(command)
        if command[:2] != ["cmake", "--build"]:
            return
        runtime_build = tmp_path / "output" / ".runtime-build"
        runtime_build.mkdir(parents=True)
        (runtime_build / RUNTIME_LIBRARY).write_bytes(b"runtime")
        (runtime_build / RUNTIME_PLUGIN).write_bytes(b"plugin")

    monkeypatch.setattr(adapter, "_run_checked", fake_run)
    runtime, plugin, resolved = adapter._build_runtime_dso(
        tmp_path / "output",
        {
            "runtime_build": {
                "sdk_include_dir": str(tmp_path / "sdk"),
                "nlohmann_json_include_dir": str(tmp_path / "json"),
            }
        },
        "0" * 64,
    )

    assert runtime.name == RUNTIME_LIBRARY
    assert plugin is not None and plugin.name == RUNTIME_PLUGIN
    assert resolved is dependency
    assert f"-DTRTMC_CUDA_DRIVER_LIBRARY={driver}" in calls[0]
    assert f"-DCMAKE_C_COMPILER={selected_cc}" in calls[0]
    assert f"-DCMAKE_CXX_COMPILER={selected_cxx}" in calls[0]
    assert f"-DCMAKE_CUDA_HOST_COMPILER={selected_cxx}" in calls[0]


@pytest.mark.parametrize("runpath", ("/build/tensorrt", "$ORIGIN", "../lib"))
def test_real_payload_validation_rejects_every_runpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runpath: str
) -> None:
    adapter = _load_adapter_module()
    payload = tmp_path / "runtime.so"
    payload.write_bytes(b"ELF fixture")
    header = """
  Class:                             ELF64
  Machine:                           Advanced Micro Devices X86-64
"""
    dynamic = f"""
 0x0000000000000001 (NEEDED) Shared library: [libcuda.so.1]
 0x0000000000000001 (NEEDED) Shared library: [libcudart.so.13]
 0x0000000000000001 (NEEDED) Shared library: [libnvinfer.so.11]
 0x000000000000001d (RUNPATH) Library runpath: [{runpath}]
"""
    monkeypatch.setattr(
        adapter,
        "_run_capture",
        lambda command, _description: header if "-h" in command else dynamic,
    )

    with pytest.raises(adapter.AdapterError, match="forbidden RPATH/RUNPATH"):
        adapter._validate_packaged_elf(payload)


def test_engine_export_uses_pinned_dependency_tools_despite_ambient_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    source = tmp_path / "edge-source"
    export_script = source / "tensorrt_edgellm" / "scripts" / "export.py"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("# pinned exporter\n", encoding="utf-8")
    build_dir = tmp_path / "edge-build"
    build_tool = build_dir / "examples" / "llm" / "llm_build"
    build_tool.parent.mkdir(parents=True)
    build_tool.write_bytes(b"tool")
    plugin = build_dir / "libNvInfer_edgellm_plugin.so.1.0"
    plugin.write_bytes(b"plugin")
    placeholder = tmp_path / "placeholder"
    placeholder.write_bytes(b"x")
    include = tmp_path / "include"
    include.mkdir()
    dependency = adapter._EdgeDependency(
        source,
        build_dir,
        build_tool,
        plugin,
        adapter._TensorRtInstallation(include, placeholder, include, placeholder),
        adapter._CudaInstallation(tmp_path, include, placeholder, placeholder, placeholder, "13.3"),
    )
    model_source = tmp_path / "model"
    model_source.mkdir()
    poison_bin = tmp_path / "poison-bin"
    poison_bin.mkdir()
    for name in ("tensorrt-edgellm-export", "llm_build"):
        poison_tool = poison_bin / name
        poison_tool.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        poison_tool.chmod(0o755)
    ambient_python = tmp_path / "ambient-python"
    ambient_package = ambient_python / "tensorrt_edgellm" / "__init__.py"
    ambient_package.parent.mkdir(parents=True)
    ambient_package.write_text("raise RuntimeError('ambient poison')\n", encoding="utf-8")
    invocations: list[tuple[list[str], object, Path | None]] = []
    validated_sources: list[Path] = []

    monkeypatch.setattr(adapter, "_probe_build_device", lambda: None)
    monkeypatch.setattr(adapter, "_materialize_exporter_python", lambda: Path(sys.executable))
    monkeypatch.setattr(adapter, "_materialize_model_source", lambda _source: model_source)
    monkeypatch.setattr(
        adapter,
        "_validate_edge_source",
        lambda path: validated_sources.append(path) or path,
    )
    monkeypatch.setenv("PATH", str(poison_bin))
    monkeypatch.setenv("PYTHONPATH", str(ambient_python))

    def fake_tool(command, *, verbose, environment=None, cwd=None):
        del verbose
        invocations.append((command, environment, cwd))
        engine_argument = next((item for item in command if item.startswith("--engineDir=")), None)
        if engine_argument is None:
            return
        engine = Path(engine_argument.split("=", 1)[1])
        _fake_engine_contents = {
            **ENGINE_MODEL_CONFIG,
            "edgellm_version": "0.9.0",
            "builder_config": ENGINE_BUILDER_CONFIG,
        }
        (engine / "config.json").write_text(json.dumps(_fake_engine_contents), encoding="utf-8")
        for filename in (
            "llm.engine",
            "embedding.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "processed_chat_template.json",
        ):
            (engine / filename).write_bytes(b"artifact")

    monkeypatch.setattr(adapter, "_run_tool", fake_tool)
    engine, _vocab_size, attempt = adapter._build_or_resolve_engine(
        {}, tmp_path / "output", dependency, plugin
    )

    assert engine.name == "engine.dir"
    assert attempt is not None
    assert invocations[0][0] == adapter._edge_export_command(
        Path(sys.executable), source, model_source, attempt / "output_root"
    )
    assert "PYTHONPATH" not in invocations[0][1]
    assert invocations[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert invocations[0][1]["PYTHONNOUSERSITE"] == "1"
    assert invocations[0][1]["PIP_CONFIG_FILE"] == os.devnull
    assert invocations[0][2] == attempt / ".tool-cwd"
    assert invocations[1][0][0] == str(build_tool)
    assert f"--onnxDir={attempt / 'output_root' / 'llm'}" in invocations[1][0]
    assert invocations[1][1]["EDGELLM_PLUGIN_PATH"] == str(plugin)
    assert invocations[1][2] == attempt / ".tool-cwd"
    assert validated_sources == [source, source, source]


def test_engine_build_revalidates_source_after_each_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    source = tmp_path / "edge-source"
    export_script = source / "tensorrt_edgellm" / "scripts" / "export.py"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("# pinned exporter\n", encoding="utf-8")
    build_dir = tmp_path / "edge-build"
    build_tool = build_dir / "examples" / "llm" / "llm_build"
    build_tool.parent.mkdir(parents=True)
    build_tool.write_bytes(b"tool")
    plugin = build_dir / "libNvInfer_edgellm_plugin.so.1.0"
    plugin.write_bytes(b"plugin")
    placeholder = tmp_path / "placeholder"
    placeholder.write_bytes(b"x")
    include = tmp_path / "include"
    include.mkdir()
    dependency = adapter._EdgeDependency(
        source,
        build_dir,
        build_tool,
        plugin,
        adapter._TensorRtInstallation(include, placeholder, include, placeholder),
        adapter._CudaInstallation(tmp_path, include, placeholder, placeholder, placeholder, "13.3"),
    )
    model_source = tmp_path / "model"
    model_source.mkdir()
    mutation = source / "post-build-poison.py"
    validations: list[Path] = []
    tool_calls: list[list[str]] = []

    monkeypatch.setattr(adapter, "_probe_build_device", lambda: None)
    monkeypatch.setattr(adapter, "_materialize_exporter_python", lambda: Path(sys.executable))
    monkeypatch.setattr(adapter, "_materialize_model_source", lambda _source: model_source)

    def validate(path: Path) -> Path:
        validations.append(path)
        if mutation.exists():
            raise adapter.AdapterError("post-build pinned-source mutation")
        return path

    def mutate_after_build(command, *, verbose, environment=None, cwd=None):
        del verbose, environment, cwd
        tool_calls.append(command)
        if command[0] == str(build_tool):
            mutation.write_text("poison\n", encoding="utf-8")

    monkeypatch.setattr(adapter, "_validate_edge_source", validate)
    monkeypatch.setattr(adapter, "_run_tool", mutate_after_build)

    output = tmp_path / "output"
    with pytest.raises(adapter.AdapterError, match="post-build pinned-source mutation"):
        adapter._build_or_resolve_engine({}, output, dependency, plugin)

    assert len(tool_calls) == 2
    assert validations == [source, source, source]
    workspace = output / ".edge-build-workspace"
    assert workspace.is_dir()
    assert not any(workspace.iterdir())


def test_verbose_edge_tool_logs_never_pollute_json_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _load_adapter_module()
    completed = subprocess.CompletedProcess(
        ["edge-tool"],
        returncode=0,
        stdout="child stdout\n",
        stderr="child stderr\n",
    )
    monkeypatch.setattr(adapter.subprocess, "run", lambda *_args, **_kwargs: completed)

    adapter._run_tool(["edge-tool"], verbose=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "child stdout\nchild stderr\n"
