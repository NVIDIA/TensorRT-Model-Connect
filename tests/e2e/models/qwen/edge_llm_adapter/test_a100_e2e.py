# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Model Connect E2E for the qualified Qwen EdgeLLM profiles."""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from tensorrt_model_connect import Pipeline
from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC

from tests.e2e.models.qwen.edge_llm_adapter import performance_harness

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from tensorrt_model_connect.runtime_provider.target import (
    TargetResolutionError,
    _probe_current_target_with_device,
)


_REPO_ROOT = Path(__file__).resolve().parents[5]
_PROFILES_ROOT = (
    _REPO_ROOT / "python/tensorrt_model_connect/families/qwen/edge_llm_adapter/profiles"
)
_QUALIFICATION_DESCRIPTOR = Path(__file__).with_name("QUALIFICATION.a100.toml")
_PROFILE_FILES_ENV = "TRTMC_QUALIFICATION_PROFILE_FILES"
with _QUALIFICATION_DESCRIPTOR.open("rb") as _descriptor_stream:
    _A100_TARGET = dict(tomllib.load(_descriptor_stream)["profile_target"])


@dataclass(frozen=True)
class _Profile:
    file_name: str
    profile_id: str
    model_id: str
    revision: str
    qualification_state: str
    required_files: tuple[str, ...]


def _requested_profile_paths(profile_files: str, profiles_root: Path) -> tuple[Path, ...]:
    available = {path.name: path for path in sorted(profiles_root.glob("*.toml"))}
    if not profile_files:
        return tuple(available.values())

    names = profile_files.split(",")
    invalid = next(
        (
            name
            for name in names
            if not name
            or name != name.strip()
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.toml", name) is None
        ),
        None,
    )
    if invalid is not None:
        raise RuntimeError(
            f"{_PROFILE_FILES_ENV} contains an invalid profile basename: {invalid!r}"
        )
    if len(names) != len(set(names)):
        raise RuntimeError(f"{_PROFILE_FILES_ENV} contains duplicate profile basenames")
    missing = [name for name in names if name not in available]
    if missing:
        raise RuntimeError(
            f"{_PROFILE_FILES_ENV} names unavailable Qwen EdgeLLM profile(s): " + ", ".join(missing)
        )
    return tuple(available[name] for name in names)


def _profiles(
    profile_files: str | None = None,
    *,
    profiles_root: Path | None = None,
) -> tuple[_Profile, ...]:
    root = _PROFILES_ROOT if profiles_root is None else profiles_root
    raw_selection = (
        os.environ.get(_PROFILE_FILES_ENV, "") if profile_files is None else profile_files
    )
    explicit_selection = bool(raw_selection)
    profiles = []
    for path in _requested_profile_paths(raw_selection, root):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Qwen EdgeLLM qualification profile must be a regular file: {path}")
        with path.open("rb") as stream:
            data = tomllib.load(stream)
        target = data.get("target")
        target_matches = isinstance(target, dict) and all(
            target.get(name) == value for name, value in _A100_TARGET.items()
        )
        if not target_matches:
            if explicit_selection:
                raise RuntimeError(
                    f"{_PROFILE_FILES_ENV} selected a profile outside the exact A100 target: "
                    f"{path.name}"
                )
            continue
        qualification_state = str(data.get("qualification_state", ""))
        if qualification_state != "qualified":
            if explicit_selection:
                raise RuntimeError(
                    f"{_PROFILE_FILES_ENV} selected non-qualified profile {path.name}: "
                    f"{qualification_state or '<missing>'}"
                )
            continue
        revisions = tuple(str(revision) for revision in data["model"]["revisions"])
        if len(revisions) != 1:
            raise RuntimeError(f"A100 qualification requires one exact revision in {path}")
        profiles.append(
            _Profile(
                file_name=path.name,
                profile_id=str(data["profile_id"]),
                model_id=str(data["model"]["id"]),
                revision=revisions[0],
                qualification_state=qualification_state,
                required_files=tuple(str(name) for name in data["artifacts"]["required_files"]),
            )
        )
    if not profiles:
        raise RuntimeError(f"no qualified Qwen EdgeLLM A100 profiles selected from {root}")
    return tuple(profiles)


_PROFILE_FILE_SELECTION = os.environ.get(_PROFILE_FILES_ENV, "")
_PROFILES = _profiles(_PROFILE_FILE_SELECTION)
_QUALIFIED_PROFILES = _PROFILES
_PROFILE_PARAMETERS = tuple(pytest.param(profile, id=profile.profile_id) for profile in _PROFILES)


def _coexistence_profiles(
    profiles: tuple[_Profile, ...], profile_file_selection: str
) -> tuple[_Profile, ...]:
    return profiles[:2] if not profile_file_selection and len(profiles) >= 2 else ()


_COEXISTENCE_PROFILES = _coexistence_profiles(_QUALIFIED_PROFILES, _PROFILE_FILE_SELECTION)


def _run(
    command: list[str],
    *,
    timeout: int,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _require_supported_a100() -> None:
    try:
        target, _ = _probe_current_target_with_device()
    except TargetResolutionError as exc:
        pytest.fail(f"the selected A100 proof could not inspect its CUDA target: {exc}")
    mismatches = {
        field: {"expected": expected, "actual": target.get(field)}
        for field, expected in _A100_TARGET.items()
        if type(target.get(field)) is not type(expected) or target.get(field) != expected
    }
    if mismatches:
        pytest.fail(
            "the selected A100 proof requires its exact descriptor target; "
            + json.dumps(mismatches, sort_keys=True)
        )


def _required_path(environment_name: str, description: str) -> Path:
    value = os.environ.get(environment_name, "").strip()
    if not value:
        pytest.fail(f"{environment_name} is required for {description}")
    return Path(value).resolve(strict=True)


def _read_json_section(bundle: Path, name: str) -> dict[str, object]:
    with bundle.open("rb") as source:
        assert source.read(8) == BUNDLE_MAGIC
        header_length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_length))
        metadata = header["sections"][name]
        source.seek(16 + header_length + int(metadata["offset"]))
        return json.loads(source.read(int(metadata["size"])))


def _build_bundle(binary: Path, profile: _Profile, output: Path) -> None:
    _run(
        [str(binary), "build", profile.model_id, "-o", str(output)],
        timeout=21_600,
    )


def _compile_cpp_client(root: Path, core: Path, include: Path) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.fail("a C++ compiler is required for the installed C++ API proof")
    source = root / "installed-cpp-client.cpp"
    source.write_text(
        r"""
#include <trtmc/pipeline.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3)
        return 10;
    trtmc::LoadOptions options;
    options.runtime_cache_path = argv[argc - 1];
    trtmc::GenerateConfig config;
    config.max_new_tokens = 16;
    config.temperature = 0.0F;
    config.top_k = 1;
    config.top_p = 1.0F;
    config.use_chat_template = true;
    config.enable_thinking = false;

    std::vector<std::unique_ptr<trtmc::IPipeline>> pipelines;
    pipelines.reserve(static_cast<std::size_t>(argc - 2));
    for (int index = 1; index < argc - 1; ++index) {
        auto pipeline = trtmc::load(argv[index], options);
        const auto result = pipeline->generate("Reply with one word: CUDA", config);
        if (result.text.empty() || std::string(pipeline->model_id()).empty())
            return 11;
        std::cout << "TRTMC_CPP_OK:" << pipeline->model_id() << '\n';
        pipelines.push_back(std::move(pipeline));
    }
    return 0;
}
""".lstrip(),
        encoding="utf-8",
    )
    client = root / "installed-cpp-client"
    _run(
        [
            compiler,
            "-std=c++17",
            f"-I{include}",
            str(source),
            str(core),
            f"-Wl,-rpath,{core.parent}",
            "-pthread",
            "-ldl",
            "-o",
            str(client),
        ],
        timeout=120,
    )
    return client


def _run_cpp_client(client: Path, bundles: list[Path], cache: Path) -> list[str]:
    result = _run(
        [str(client), *map(str, bundles), str(cache)],
        timeout=1_200,
        environment={
            **os.environ,
            "LD_LIBRARY_PATH": (f"{client.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"),
        },
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def qualified_bundles(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build each qualified profile once for every public/runtime proof."""

    _require_supported_a100()
    binary = _required_path("TRTMC_BINARY", "the A100 EdgeLLM E2E")
    root = tmp_path_factory.mktemp("qwen-edge-qualified-bundles")
    bundles = {}
    for profile in _QUALIFIED_PROFILES:
        bundle = root / f"{profile.profile_id}.trtfb"
        _build_bundle(binary, profile, bundle)
        bundles[profile.profile_id] = bundle
    return bundles


@pytest.fixture(scope="module")
def edge_inference_binary(qualified_bundles: dict[str, Path]) -> Path:
    """Resolve or build the official runner from the dependency used by MC."""

    assert qualified_bundles
    configured = os.environ.get("TRTMC_EDGE_LLM_INFERENCE_BINARY", "").strip()
    if configured:
        return Path(configured).resolve(strict=True)
    build_directory = _required_path(
        "_TRTMC_INTERNAL_QWEN_EDGE_LLM_BUILD_DIR",
        "the deterministic Qwen EdgeLLM qualification build",
    )
    return performance_harness.prepare_official_inference_binary(build_directory)


@pytest.fixture(scope="module")
def performance_runners(
    tmp_path_factory: pytest.TempPathFactory,
    edge_inference_binary: Path,
) -> performance_harness.PerformanceRunners:
    """Build one family-owned runner pair for every qualified Qwen profile."""

    core_library = _required_path(
        "TRTMC_CORE_LIBRARY", "the installed Model Connect performance path"
    )
    include_directory = _required_path("TRTMC_INCLUDE_DIR", "the installed Model Connect C++ API")
    assert (include_directory / "trtmc/pipeline.h").is_file()
    return performance_harness.build_performance_runners(
        source_directory=Path(__file__).resolve().parent,
        build_directory=tmp_path_factory.mktemp("qwen-edge-performance-runners"),
        inference_binary=edge_inference_binary,
        mc_core_library=core_library,
        mc_include_directory=include_directory,
    )


def _run_c_abi_probe(core: Path, bundle: Path, cache: Path) -> None:
    script = r"""
import ctypes
import sys

class Options(ctypes.Structure):
    _fields_ = [
        ("max_new_tokens", ctypes.c_int),
        ("hf_python", ctypes.c_char_p),
        ("image_path", ctypes.c_char_p),
        ("runtime_cache", ctypes.c_char_p),
        ("cuda_graphs", ctypes.c_int),
    ]

library = ctypes.CDLL(sys.argv[1])
create = library.trtmc_create_pipeline_ex
create.argtypes = [ctypes.c_char_p, ctypes.POINTER(Options)]
create.restype = ctypes.c_void_p
last_error = library.trtmc_last_error
last_error.restype = ctypes.c_char_p
options = Options(runtime_cache=sys.argv[3].encode())
pipeline = create(sys.argv[2].encode(), ctypes.byref(options))
if not pipeline:
    raise RuntimeError(last_error().decode())
"""
    _run([sys.executable, "-c", script, str(core), str(bundle), str(cache)], timeout=600)


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.slow
@pytest.mark.parametrize("profile", _PROFILE_PARAMETERS)
def test_public_build_inspect_and_run_delegate_to_edgellm(
    tmp_path: Path,
    profile: _Profile,
    qualified_bundles: dict[str, Path],
    edge_inference_binary: Path,
) -> None:
    """Build and execute one request through unchanged public commands."""

    _require_supported_a100()
    binary = _required_path("TRTMC_BINARY", "the A100 EdgeLLM E2E")
    core_library = _required_path("TRTMC_CORE_LIBRARY", "the installed C and C++ API proof")
    include_directory = _required_path("TRTMC_INCLUDE_DIR", "the installed C++ API header proof")
    assert (include_directory / "trtmc/pipeline.h").is_file()

    bundle = qualified_bundles[profile.profile_id]
    cache = tmp_path / "runtime-cache"
    output = tmp_path / "generated.jsonl"
    warm_output = tmp_path / "generated-warm.jsonl"

    inspect = _run([str(binary), "inspect", str(bundle)], timeout=60)
    assert "optimized_runtime.json" in inspect.stdout
    assert "optimized_runtime_artifacts/engine.dir/" in inspect.stdout
    assert "libtrtmc_impl_qwen_tensorrt_edge_llm.so" in inspect.stdout
    descriptor = _read_json_section(bundle, "optimized_runtime.json")
    implementation = _read_json_section(bundle, "implementation.json")
    assert descriptor["implementation_id"] == "qwen.tensorrt-edge-llm"
    assert descriptor["profile_id"] == profile.profile_id
    assert descriptor["model_id"] == profile.model_id
    assert descriptor["runtime"] == {
        "name": "tensorrt-edge-llm",
        "version": "0.9.0",
        "commit": "1ac0f2b99642045125e1c5ac7b109434ba3b36c7",
    }
    assert implementation["model"] == {
        "id": profile.model_id,
        "revision": profile.revision,
    }

    prompt = "Reply with one short sentence about accelerated computing."
    run_command = [
        str(binary),
        "run",
        str(bundle),
        "--prompt",
        prompt,
        "--greedy",
        "--top-p",
        "1",
        "--top-k",
        "1",
        "--chat-template",
        "--no-thinking",
        "--max-new-tokens",
        "32",
        "--runtime-cache",
        str(cache),
        "--output",
        str(output),
    ]
    _run(run_command, timeout=600)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["generated"].strip()

    python_result = Pipeline(str(bundle), binary=str(binary))(prompt, max_new_tokens=16)
    assert python_result.strip()
    _run_c_abi_probe(core_library, bundle, cache)
    cpp_client = _compile_cpp_client(tmp_path, core_library, include_directory)
    assert _run_cpp_client(cpp_client, [bundle], cache) == [f"TRTMC_CPP_OK:{profile.model_id}"]

    engine_dirs = [path for path in cache.rglob("engine.dir") if path.is_dir()]
    assert len(engine_dirs) == 1
    cold_files: dict[str, tuple[int, int]] = {}
    for relative_path in profile.required_files:
        artifact = engine_dirs[0] / relative_path
        assert artifact.is_file()
        stat = artifact.stat()
        cold_files[relative_path] = (stat.st_size, stat.st_mtime_ns)

    warm_command = list(run_command)
    warm_command[warm_command.index(str(output))] = str(warm_output)
    _run(warm_command, timeout=600)
    warm_rows = [json.loads(line) for line in warm_output.read_text(encoding="utf-8").splitlines()]
    assert warm_rows == rows

    repeated = _run(
        [
            *run_command[: run_command.index("--output")],
            "--benchmark",
            "2",
            "--warmup",
            "1",
        ],
        timeout=1_800,
    )
    assert repeated.stdout.strip()
    for relative_path, expected in cold_files.items():
        stat = (engine_dirs[0] / relative_path).stat()
        assert (stat.st_size, stat.st_mtime_ns) == expected

    direct_input = tmp_path / "direct-input.json"
    direct_output = tmp_path / "direct-output.json"
    direct_input.write_text(
        json.dumps(
            {
                "batch_size": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_generate_length": 32,
                "apply_chat_template": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
                "requests": [
                    {"messages": [{"role": "user", "content": prompt}]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _run(
        [
            str(edge_inference_binary),
            "--engineDir",
            str(engine_dirs[0]),
            "--inputFile",
            str(direct_input),
            "--outputFile",
            str(direct_output),
        ],
        timeout=600,
        environment={
            **os.environ,
            "EDGELLM_PLUGIN_PATH": str(engine_dirs[0].parent / "libNvInfer_edgellm_plugin.so"),
        },
    )
    direct = json.loads(direct_output.read_text(encoding="utf-8"))
    assert direct["responses"][0]["output_text"].strip() == rows[0]["generated"].strip()


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.slow
@pytest.mark.skipif(
    bool(_PROFILE_FILE_SELECTION),
    reason="the full qualification owns the process-wide x86 TensorRT native regression",
)
def test_x86_tensorrt_cohort_preserves_native_qwen_execution(tmp_path: Path) -> None:
    """Prove the x86 TensorRT cohort still builds and runs the native Qwen path."""

    _require_supported_a100()
    binary = _required_path("TRTMC_BINARY", "the native x86 TensorRT regression")
    profile = _QUALIFIED_PROFILES[0]
    bundle = tmp_path / "native-qwen.trtfb"
    _run(
        [
            str(binary),
            "build",
            profile.model_id,
            "-o",
            str(bundle),
            "--precision",
            "fp16",
            "--max-cache-length",
            "256",
            "--max-batch-size",
            "1",
        ],
        timeout=21_600,
    )
    inspect = _run([str(binary), "inspect", str(bundle)], timeout=60)
    assert "optimized_runtime.json" not in inspect.stdout

    output = tmp_path / "native-output.jsonl"
    _run(
        [
            str(binary),
            "run",
            str(bundle),
            "--prompt",
            "Reply with one word: CUDA",
            "--greedy",
            "--max-new-tokens",
            "8",
            "--runtime-cache",
            str(tmp_path / "native-runtime-cache"),
            "--output",
            str(output),
        ],
        timeout=1_200,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["generated"].strip()


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.slow
@pytest.mark.skipif(
    len(_COEXISTENCE_PROFILES) < 2,
    reason="the complete A100 profile selection is required for coexistence",
)
def test_installed_cpp_api_keeps_two_profile_runtimes_alive_in_one_process(
    tmp_path: Path,
    qualified_bundles: dict[str, Path],
) -> None:
    """Prove two family profiles coexist through the unchanged public C++ API."""

    _require_supported_a100()
    assert len(_COEXISTENCE_PROFILES) >= 2
    core_library = _required_path("TRTMC_CORE_LIBRARY", "the installed C++ API proof")
    include_directory = _required_path("TRTMC_INCLUDE_DIR", "the installed C++ API header proof")
    profiles = _COEXISTENCE_PROFILES
    bundles = [qualified_bundles[profile.profile_id] for profile in profiles]

    client = _compile_cpp_client(tmp_path, core_library, include_directory)
    output = _run_cpp_client(client, bundles, tmp_path / "runtime-cache")
    assert output == [f"TRTMC_CPP_OK:{profile.model_id}" for profile in profiles]


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.slow
@pytest.mark.parametrize("profile", _PROFILE_PARAMETERS)
def test_model_connect_performance_matches_direct_edgellm(
    tmp_path: Path,
    profile: _Profile,
    qualified_bundles: dict[str, Path],
    performance_runners: performance_harness.PerformanceRunners,
) -> None:
    """Qualify adapter overhead against direct EdgeLLM on the same engine."""

    _require_supported_a100()
    tested_source = performance_harness.source_identity(_REPO_ROOT)
    artifact_root = performance_harness.artifact_root(_REPO_ROOT)
    core_library = _required_path(
        "TRTMC_CORE_LIBRARY", "the installed Model Connect performance path"
    )
    bundle = qualified_bundles[profile.profile_id]
    runtime_cache = tmp_path / "runtime-cache"
    _run_c_abi_probe(core_library, bundle, runtime_cache)

    engine_directories = [path for path in runtime_cache.rglob("engine.dir") if path.is_dir()]
    assert len(engine_directories) == 1
    engine_directory = engine_directories[0]
    plugin = engine_directory.parent / "libNvInfer_edgellm_plugin.so"
    assert plugin.is_file()

    artifact_directory = (
        artifact_root
        / f"{performance_harness.safe_profile_name(profile.profile_id)}-{uuid.uuid4().hex}"
    )
    direct_results, mc_results, raw_paths = performance_harness.run_repetitions(
        performance_runners,
        bundle=bundle,
        engine_directory=engine_directory,
        plugin=plugin,
        runtime_cache=runtime_cache,
        artifact_directory=artifact_directory,
    )
    summary = performance_harness.evaluate_performance(
        profile_id=profile.profile_id,
        model_id=profile.model_id,
        revision=profile.revision,
        source_identity=tested_source,
        runtime_identity=performance_runners.runtime_identity,
        direct_results=direct_results,
        mc_results=mc_results,
        raw_paths=raw_paths,
    )
    summary_path = artifact_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "TRTMC_QWEN_EDGELLM_PERFORMANCE "
        + json.dumps(
            {
                "artifact": str(summary_path),
                "metrics": summary["metrics"],
                "passed": summary["passed"],
                "profile_id": profile.profile_id,
            },
            sort_keys=True,
        )
    )
    assert summary["passed"], summary
