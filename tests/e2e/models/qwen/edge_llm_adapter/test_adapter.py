# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-local contracts for the Qwen3-0.6B TensorRT Edge-LLM capsule."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
CAPSULE_ROOT = (
    REPOSITORY_ROOT / "python" / "tensorrt_model_connect" / "families" / "qwen" / "edge_llm_adapter"
)
RUNTIME_ROOT = REPOSITORY_ROOT / "src" / "runtime" / "models" / "qwen" / "edge_llm_adapter"
PYTHON_ROOT = REPOSITORY_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

from tensorrt_model_connect.optimized_runtime.build_adapter import (  # noqa: E402
    BuildAdapterError,
    ImplementationRequest,
    run_build,
    run_probe,
)
from tensorrt_model_connect.optimized_runtime.bundle import (  # noqa: E402
    write_optimized_bundle,
)
from tensorrt_model_connect.optimized_runtime.manifest import (  # noqa: E402
    load_implementation_manifest,
    manifest_contract_sha256,
)
from tensorrt_model_connect.optimized_runtime.orchestrator import (  # noqa: E402
    discover_family_implementations_for_model,
    family_implementation_root,
)


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
IMPLEMENTATION_ID = "qwen3-0.6b-fp16.tensorrt-edge-llm.a100-pcie80-sm80"
PROFILE_ID = "qwen3-0.6b-fp16--a100-pcie80-sm80"
EDGE_COMMIT = "2620a9768022f25dff18912db2fb92b2ef264a70"
EDGE_SOURCE = "https://github.com/NVIDIA/TensorRT-Edge-LLM.git"
RUNTIME_LIBRARY = "libtrtmc_impl_qwen3_0_6b_fp16_tensorrt_edge_llm.so"
RUNTIME_PLUGIN = "libNvInfer_edgellm_plugin.so"
MANIFEST_PATH = CAPSULE_ROOT / "IMPLEMENTATION.toml"
PROFILE_PATH = CAPSULE_ROOT / "profiles" / "a100-pcie80-sm80-fp16.toml"
ADAPTER_PATH = CAPSULE_ROOT / "adapter.py"


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

    assert root == CAPSULE_ROOT.parent
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
    packaged_builder = package / "families" / "qwen" / "edge_llm_adapter"
    packaged_runtime = packaged_builder / "runtime"
    private_sdk = package / "optimized_runtime" / "_sdk" / "include"
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
    monkeypatch.setenv("_TRTMC_INTERNAL_QWEN3_06B_ALLOW_FAKE_RUNTIME_BUILD", "1")
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


def _load_adapter_module():
    name = f"trtmc_qwen_edge_adapter_test_{id(object())}"
    specification = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


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
                "vocab_size": 151936,
                "edgellm_version": "0.6.1",
                "builder_config": {
                    "max_input_len": 1024,
                    "max_kv_cache_capacity": 4096,
                    "max_batch_size": 4,
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
        "precision": "fp16",
        "quantization": "none",
        "max_input_length": 1024,
        "max_cache_length": 4096,
        "max_batch_size": 4,
    }


def _run_build_after_probe(manifest, request, output: Path):
    probe = run_probe(manifest, request)
    assert probe.supported, probe.reason
    return run_build(manifest, request, output, probe=probe)


def _make_cuda_toolkit(
    root: Path,
    *,
    encoded_header_version: int = 12080,
    compiler_version: str = "12.8",
) -> tuple[Path, Path, Path]:
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
    return include, compiler, cudart


def _make_edge_source(root: Path) -> Path:
    for relative in (
        "CMakeLists.txt",
        "cpp/common/version.h",
        "3rdParty/nlohmannJson/include/nlohmann/json.hpp",
        "tensorrt_edgellm/scripts/export_llm.py",
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
    assert not manifest.matches(_request(model_id="Qwen/Qwen3-1.7B"))
    assert not manifest.matches(_request(revision="0" * 40))
    assert not manifest.matches(_request(target=_target(gpu_architecture="sm90")))
    assert not manifest.matches(_request(target=_target(platform_kind="soc")))
    assert not manifest.matches(_request(target=_target(gpu_name="NVIDIA A100-SXM4-80GB")))

    assert dependency["downstream"] == {
        "name": "tensorrt-edge-llm",
        "source": EDGE_SOURCE,
        "version": "0.6.1",
        "tag": "v0.6.1",
        "commit": EDGE_COMMIT,
        "source_mode": "git",
    }
    assert dependency["tensorrt"] == {"version": "10.14.1.48"}
    assert dependency["cuda"] == {"version": "12.8"}
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


def test_supported_probe_selects_profile_without_hidden_opt_in(tmp_path: Path) -> None:
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


@pytest.mark.parametrize("missing_module", ("modelopt", "onnx_graphsurgeon", "torch"))
def test_qualified_probe_reports_missing_exporter_prerequisite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing_module: str,
) -> None:
    adapter = _load_adapter_module()
    monkeypatch.setattr(adapter, "_select_parent_active_cuda_device", lambda: None)
    monkeypatch.setattr(adapter.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        adapter.importlib.util,
        "find_spec",
        lambda module: None if module == missing_module else object(),
    )
    monkeypatch.setattr(adapter, "_resolve_tensorrt", lambda _settings: object())
    monkeypatch.setattr(adapter, "_resolve_cuda", lambda _settings: object())

    payload = _request().to_json()
    payload["implementation_id"] = adapter.IMPLEMENTATION_ID
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    assert adapter.main(["probe", "--request", str(request_path)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert (
        f"Qwen Edge-LLM build prerequisites are unavailable: {missing_module}"
        in captured.err
    )


def test_build_stages_explicit_engine_runtime_and_plugin_payloads(tmp_path: Path) -> None:
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
    assert build.descriptor["versions"]["edge_llm"] == "0.6.1"
    assert build.descriptor["versions"]["cuda"] == "12.8"
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

    def wrong_tensorrt(_runtime_build):
        raise adapter.AdapterError("found TensorRT 11.2.0.113, not 10.14.1.48")

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

    assert adapter.main(["probe", "--request", str(request_path)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "TensorRT 11.2.0.113" in captured.err


def test_tensorrt_resolution_requires_one_exact_core_and_parser_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    include = tmp_path / "include"
    include.mkdir()
    (include / "NvInfer.h").write_text("// fixture\n", encoding="utf-8")
    version_header = include / "NvInferVersion.h"
    version_header.write_text(
        "#define NV_TENSORRT_MAJOR 10\n"
        "#define NV_TENSORRT_MINOR 14\n"
        "#define NV_TENSORRT_PATCH 1\n"
        "#define NV_TENSORRT_BUILD 48\n",
        encoding="utf-8",
    )
    (include / "NvOnnxParser.h").write_text("// fixture\n", encoding="utf-8")
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    core = library_dir / "libnvinfer.so.10.14.1.48"
    parser = library_dir / "libnvonnxparser.so.10"
    core.write_bytes(b"exact TensorRT core")
    parser.write_bytes(b"exact TensorRT parser")
    runtime_build = {
        "tensorrt_include_dir": str(include),
        "tensorrt_library": str(core),
        "onnx_parser_include_dir": str(include),
        "onnx_parser_library": str(parser),
    }
    monkeypatch.setattr(adapter, "_library_tensorrt_version", lambda _library: (10, 14, 1, 48))

    resolved = adapter._resolve_tensorrt(runtime_build)
    assert resolved.library == core
    assert resolved.onnx_parser_library == parser

    version_header.write_text(
        version_header.read_text(encoding="utf-8").replace(
            "#define NV_TENSORRT_BUILD 48", "#define NV_TENSORRT_BUILD 47"
        ),
        encoding="utf-8",
    )
    with pytest.raises(adapter.AdapterError, match="10.14.1.47, not 10.14.1.48"):
        adapter._resolve_tensorrt(runtime_build)


@pytest.mark.parametrize(
    ("header_version", "compiler_version"),
    ((13000, "12.8"), (12080, "13.0")),
)
def test_cuda_resolution_rejects_unsupported_toolkit_components(
    tmp_path: Path,
    header_version: int,
    compiler_version: str,
) -> None:
    adapter = _load_adapter_module()
    include, _compiler, cudart = _make_cuda_toolkit(
        tmp_path / "cuda",
        encoded_header_version=header_version,
        compiler_version=compiler_version,
    )

    with pytest.raises(adapter.AdapterError, match=r"not 12\.8"):
        adapter._resolve_cuda({"cuda_include_dir": str(include), "cudart_library": str(cudart)})


def test_cuda_resolution_pins_one_coherent_12_8_toolkit(tmp_path: Path) -> None:
    adapter = _load_adapter_module()
    cuda_root = tmp_path / "cuda"
    include, compiler, cudart = _make_cuda_toolkit(cuda_root)

    resolved = adapter._resolve_cuda(
        {"cuda_include_dir": str(include), "cudart_library": str(cudart)}
    )

    assert resolved.root == cuda_root.resolve()
    assert resolved.include_dir == include.resolve()
    assert resolved.cudart_library == cudart.resolve()
    assert resolved.compiler == compiler.resolve()
    assert resolved.version == "12.8"

    other_root = tmp_path / "other-cuda"
    _other_include, _other_compiler, other_cudart = _make_cuda_toolkit(other_root)
    with pytest.raises(adapter.AdapterError, match="same exact CUDA 12.8 toolkit"):
        adapter._resolve_cuda(
            {"cuda_include_dir": str(include), "cudart_library": str(other_cudart)}
        )


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
        "v0.6.1",
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
    monkeypatch.setenv("_TRTMC_INTERNAL_QWEN3_06B_ALLOW_FAKE_RUNTIME_BUILD", "1")
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


def test_selected_build_builds_only_required_edge_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    source = tmp_path / "edge-source"
    source.mkdir()
    trt_include = tmp_path / "trt-include"
    trt_include.mkdir()
    trt_library = tmp_path / "libnvinfer.so.10"
    trt_library.write_bytes(b"trt")
    onnx_library = tmp_path / "libnvonnxparser.so.10"
    onnx_library.write_bytes(b"onnx")
    cuda_root = tmp_path / "cuda"
    cuda_include = cuda_root / "include"
    cuda_include.mkdir(parents=True)
    cudart = cuda_root / "libcudart.so"
    cudart.write_bytes(b"cudart")
    cuda_compiler = cuda_root / "bin" / "nvcc"
    cuda_compiler.parent.mkdir()
    cuda_compiler.write_bytes(b"nvcc")
    tensorrt = adapter._TensorRtInstallation(trt_include, trt_library, trt_include, onnx_library)
    cuda = adapter._CudaInstallation(cuda_root, cuda_include, cudart, cuda_compiler, "12.8")
    calls: list[list[str]] = []
    validated_sources: list[Path] = []

    monkeypatch.setattr(adapter, "_resolve_edge_source", lambda *_args: source)
    monkeypatch.setattr(adapter, "_resolve_tensorrt", lambda _settings: tensorrt)
    monkeypatch.setattr(adapter, "_resolve_cuda", lambda _settings: cuda)
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
            (build_dir / "CMakeCache.txt").write_text(
                "\n".join(
                    (
                        f"CMAKE_HOME_DIRECTORY:INTERNAL={source}",
                        "CMAKE_BUILD_TYPE:STRING=Release",
                        f"TRT_INCLUDE_DIR:PATH={trt_include}",
                        f"NVINFER_LIB:FILEPATH={trt_library}",
                        f"ONNX_PARSER_INCLUDE_DIR:PATH={trt_include}",
                        f"NV_ONNX_PARSER_LIB:FILEPATH={onnx_library}",
                        f"CUDA_RUNTIME_API_INCLUDE_DIR:PATH={cuda_include}",
                        f"CUDART_LIB:FILEPATH={cudart}",
                        f"CMAKE_CUDA_COMPILER:FILEPATH={cuda_compiler}",
                    )
                ),
                encoding="utf-8",
            )
            return
        for product in (
            build_dir / "cpp" / "libedgellmCore.a",
            build_dir / "cpp" / "libedgellmTokenizer.a",
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
    assert calls[1][-5:] == [
        "--target",
        "edgellmCore",
        "edgellmTokenizer",
        "NvInfer_edgellm_plugin",
        "llm_build",
    ]
    reused = adapter._resolve_edge_dependency(
        output,
        {"parallel": 3, "edge_llm_build_dir": str(dependency.build_dir)},
    )
    assert reused.build_dir == dependency.build_dir
    assert len(calls) == 2


def test_engine_export_uses_pinned_dependency_tools_despite_ambient_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    source = tmp_path / "edge-source"
    export_script = source / "tensorrt_edgellm" / "scripts" / "export_llm.py"
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
        adapter._CudaInstallation(tmp_path, include, placeholder, placeholder, "12.8"),
    )
    model_source = tmp_path / "model"
    model_source.mkdir()
    poison_bin = tmp_path / "poison-bin"
    poison_bin.mkdir()
    for name in ("tensorrt-edgellm-export-llm", "llm_build"):
        poison_tool = poison_bin / name
        poison_tool.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        poison_tool.chmod(0o755)
    ambient_python = tmp_path / "ambient-python"
    ambient_package = ambient_python / "tensorrt_edgellm" / "__init__.py"
    ambient_package.parent.mkdir(parents=True)
    ambient_package.write_text("raise RuntimeError('ambient poison')\n", encoding="utf-8")
    invocations: list[tuple[list[str], object]] = []
    validated_sources: list[Path] = []

    monkeypatch.setattr(adapter, "_probe_build_device", lambda: None)
    monkeypatch.setattr(adapter, "_materialize_model_source", lambda _source: model_source)
    monkeypatch.setattr(
        adapter,
        "_validate_edge_source",
        lambda path: validated_sources.append(path) or path,
    )
    monkeypatch.setenv("PATH", str(poison_bin))
    monkeypatch.setenv("PYTHONPATH", str(ambient_python))

    def fake_tool(command, *, verbose, environment=None):
        del verbose
        invocations.append((command, environment))
        engine_argument = next((item for item in command if item.startswith("--engineDir=")), None)
        if engine_argument is None:
            return
        engine = Path(engine_argument.split("=", 1)[1])
        _fake_engine_contents = {
            "vocab_size": 151936,
            "edgellm_version": "0.6.1",
            "builder_config": {
                "max_input_len": 1024,
                "max_kv_cache_capacity": 4096,
                "max_batch_size": 4,
            },
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
    assert invocations[0][0][:2] == [sys.executable, str(export_script)]
    assert invocations[0][1]["PYTHONPATH"] == str(source)
    assert str(ambient_python) not in invocations[0][1]["PYTHONPATH"]
    assert invocations[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert invocations[0][1]["PYTHONNOUSERSITE"] == "1"
    assert invocations[1][0][0] == str(build_tool)
    assert invocations[1][1]["EDGELLM_PLUGIN_PATH"] == str(plugin)
    assert validated_sources == [source, source, source]
    assert attempt is not None


def test_engine_build_revalidates_source_after_each_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_adapter_module()
    source = tmp_path / "edge-source"
    export_script = source / "tensorrt_edgellm" / "scripts" / "export_llm.py"
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
        adapter._CudaInstallation(tmp_path, include, placeholder, placeholder, "12.8"),
    )
    model_source = tmp_path / "model"
    model_source.mkdir()
    mutation = source / "post-build-poison.py"
    validations: list[Path] = []
    tool_calls: list[list[str]] = []

    monkeypatch.setattr(adapter, "_probe_build_device", lambda: None)
    monkeypatch.setattr(adapter, "_materialize_model_source", lambda _source: model_source)

    def validate(path: Path) -> Path:
        validations.append(path)
        if mutation.exists():
            raise adapter.AdapterError("post-build pinned-source mutation")
        return path

    def mutate_after_build(command, *, verbose, environment=None):
        del verbose, environment
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
