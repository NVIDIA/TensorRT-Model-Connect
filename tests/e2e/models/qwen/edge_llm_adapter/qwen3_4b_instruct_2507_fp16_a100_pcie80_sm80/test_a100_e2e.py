# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public A100 qualification for the Qwen3 4B Instruct 2507 EdgeLLM capsule."""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from tensorrt_model_connect.runtime_provider.target import (
    TargetResolutionError,
    _probe_current_target_with_device,
)


_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
_IMPLEMENTATION_ID = "qwen3-4b-instruct-2507-fp16.tensorrt-edge-llm-v0.9.trt11.a100-pcie80-sm80"
_PROMPT = "Reply with one short sentence about accelerated computing."
_MAX_NEW_TOKENS = 32
_WARMUPS = 5
_MEASURED_REQUESTS = 30
_PERFORMANCE_REPETITIONS = 3
_REPO_ROOT = Path(__file__).resolve().parents[6]
_DEPENDENCY_LOCK = (
    _REPO_ROOT
    / "python/tensorrt_model_connect/families/qwen/edge_llm_adapter"
    / "qwen3_4b_instruct_2507_fp16_a100_pcie80_sm80/dependency.lock"
)
_EXPORT_REQUIREMENT = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9._+!-]*) "
    r"--hash=sha256:(?P<sha256>[0-9a-f]{64})"
)
_PUBLIC_EDGE_BUILD_ENVIRONMENT = {
    "TRTMC_EDGE_LLM_SOURCE_DIR": "_TRTMC_INTERNAL_QWEN3_4B_INSTRUCT_2507_EDGE_LLM_SOURCE_DIR",
    "TRTMC_EDGE_LLM_BUILD_DIR": "_TRTMC_INTERNAL_QWEN3_4B_INSTRUCT_2507_EDGE_LLM_BUILD_DIR",
}
_EDGE_BUILD_PRODUCT_LAYOUTS = tuple(
    frozenset(
        {
            f"cpp/{configuration + '/' if configuration else ''}libedgellmCore.a",
            f"{configuration + '/' if configuration else ''}libNvInfer_edgellm_plugin.so.1.0",
            f"examples/llm/{configuration + '/' if configuration else ''}llm_build",
        }
    )
    for configuration in ("", "Release", "RelWithDebInfo", "Debug")
)


def _run(
    command: list[str],
    *,
    timeout: int,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        env=env,
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
    if target["gpu_name"] != "NVIDIA A100 80GB PCIe":
        pytest.fail(
            f"the selected A100 proof requires NVIDIA A100 80GB PCIe; found {target['gpu_name']}"
        )


def _required_executable(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.fail(f"{name} is required for the A100 EdgeLLM qualification")
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        pytest.fail(f"{name} is not executable: {path}")
    return path


def _read_bundle_header(bundle: Path) -> tuple[int, dict[str, Any]]:
    with bundle.open("rb") as stream:
        assert stream.read(8) == b"TRTFB\x00\x01\x00"
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return header_size, header


def _read_bundle_section(bundle: Path, name: str) -> bytes:
    header_size, header = _read_bundle_header(bundle)
    with bundle.open("rb") as stream:
        section = header["sections"][name]
        stream.seek(16 + header_size + section["offset"])
        return stream.read(section["size"])


def _tree_inventory(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _verify_edge_build_stamp(build_dir: Path) -> dict[str, Any]:
    stamp_path = build_dir / ".trtmc-edge-build-stamp.json"
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    assert set(stamp) == {"products", "recipe", "recipe_sha256", "schema_version"}
    assert stamp["schema_version"] == 1
    recipe = stamp["recipe"]
    assert isinstance(recipe, dict)
    assert stamp["recipe_sha256"] == _canonical_sha256(recipe)

    products = stamp["products"]
    assert isinstance(products, dict)
    assert frozenset(products) in _EDGE_BUILD_PRODUCT_LAYOUTS
    build_root = build_dir.resolve(strict=True)
    for relative_path, expected_sha256 in products.items():
        assert isinstance(expected_sha256, str)
        assert len(expected_sha256) == 64
        assert all(character in "0123456789abcdef" for character in expected_sha256)
        product = (build_root / relative_path).resolve(strict=True)
        product.relative_to(build_root)
        assert product.is_file() and product.stat().st_size > 0
        assert _file_sha256(product) == expected_sha256
    return stamp


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _mc_run_command(binary: Path, bundle: Path, cache: Path, output: Path) -> list[str]:
    return [
        str(binary),
        "run",
        str(bundle),
        "--prompt",
        _PROMPT,
        "--greedy",
        "--top-p",
        "1",
        "--top-k",
        "1",
        "--no-thinking",
        "--max-new-tokens",
        str(_MAX_NEW_TOKENS),
        "--runtime-cache",
        str(cache),
        "--output",
        str(output),
    ]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_exporter_packages() -> dict[str, str]:
    dependency = tomllib.loads(_DEPENDENCY_LOCK.read_text(encoding="utf-8"))
    exporter = dependency["exporter_python"]
    requirements = exporter["requirements"]
    packages: dict[str, str] = {}
    for raw_line in requirements.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _EXPORT_REQUIREMENT.fullmatch(line)
        assert match is not None, f"invalid locked exporter requirement: {line!r}"
        name = _canonical_package_name(match.group("name"))
        assert name not in packages, f"duplicate locked exporter requirement: {name}"
        packages[name] = match.group("version")
    assert len(packages) == exporter["package_count"]
    assert list(packages) == sorted(packages)
    return packages


def _verify_exporter_profile(profile_root: Path) -> dict[str, tuple[int, int]]:
    profiles = sorted(path for path in profile_root.iterdir() if path.is_dir())
    assert len(profiles) == 1, f"expected one model-owned exporter profile, found {profiles}"
    profile = profiles[0]
    ready = (profile / ".ready").read_text(encoding="utf-8").strip()
    assert len(ready) == 64 and all(character in "0123456789abcdef" for character in ready)
    profile_python = profile / "bin" / "python"
    version_script = (
        "import importlib.metadata as m,json,re; "
        "canonical=lambda name:re.sub(r'[-_.]+','-',name).lower(); "
        "print(json.dumps({canonical(dist.metadata['Name']):dist.version "
        "for dist in m.distributions()},sort_keys=True))"
    )
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    result = _run(
        [
            str(profile_python),
            "-I",
            "-c",
            version_script,
        ],
        timeout=300,
        env=clean_env,
    )
    assert json.loads(result.stdout) == _locked_exporter_packages()
    return _tree_inventory(profile)


def _load_only_with_c_api(
    core_library: Path,
    bundle: Path,
    cache: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    # A disposable process makes the public C ABI a load-only probe. It never
    # calls generate, so an existing materialized engine after this command is
    # proof that capsule materialization and EdgeLLM initialization happen in load().
    script = """
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

library = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
library.trtmc_create_pipeline_ex.argtypes = [ctypes.c_char_p, ctypes.POINTER(Options)]
library.trtmc_create_pipeline_ex.restype = ctypes.c_void_p
library.trtmc_last_error.restype = ctypes.c_char_p
options = Options(0, None, None, sys.argv[3].encode(), 0)
handle = library.trtmc_create_pipeline_ex(sys.argv[2].encode(), ctypes.byref(options))
if not handle:
    error = library.trtmc_last_error()
    raise SystemExit(error.decode() if error else "trtmc_create_pipeline_ex failed")
print("load-only-ok")
"""
    return _run(
        [sys.executable, "-c", script, str(core_library), str(bundle), str(cache)],
        timeout=600,
        cwd=cwd,
        env=env,
    )


def _installed_package_payload(cwd: Path) -> tuple[Path, Path, Path]:
    installed_python = _required_executable("TRTMC_INSTALLED_PYTHON")
    script = """
import json
import pathlib
import sys
import tensorrt_model_connect

repo = pathlib.Path(sys.argv[1]).resolve()
module = pathlib.Path(tensorrt_model_connect.__file__).resolve()
if module == repo or repo in module.parents:
    raise SystemExit(f"source checkout leaked into installed-package proof: {module}")
install_root = pathlib.Path(sys.prefix).resolve(strict=True)
if install_root not in module.parents:
    raise SystemExit(
        f"installed module is outside the selected Python environment: {module} not below {install_root}"
    )
package = module.parent
pipeline = tensorrt_model_connect.Pipeline("unused-provenance-probe.trtfb")
binary = pathlib.Path(pipeline.binary).resolve(strict=True)
expected_binary = (package / "bin" / "trtmc").resolve(strict=True)
if binary != expected_binary:
    raise SystemExit(
        f"installed Python API did not select its bundled executable: {binary} != {expected_binary}"
    )
cores = {path.resolve(strict=True) for path in binary.parent.glob("libtrtmc_core.so*")}
if len(cores) != 1:
    raise SystemExit(f"installed package has no bundled libtrtmc_core beside {binary}")
core = cores.pop()
for path in (binary, core):
    if (
        path == repo
        or repo in path.parents
        or install_root not in path.parents
        or package not in path.parents
    ):
        raise SystemExit(f"installed-package native payload escaped its package root: {path}")
print(json.dumps({
    "module": str(module),
    "install_root": str(install_root),
    "binary": str(binary),
    "core": str(core),
}, sort_keys=True))
"""
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    result = _run(
        [str(installed_python), "-c", script, str(_REPO_ROOT)],
        timeout=300,
        cwd=cwd,
        env=clean_env,
    )
    proof = json.loads(result.stdout.splitlines()[-1])
    binary = Path(proof["binary"]).resolve(strict=True)
    core = Path(proof["core"]).resolve(strict=True)
    for native_path in (binary, core):
        assert native_path != _REPO_ROOT and _REPO_ROOT not in native_path.parents
    return installed_python, binary, core


def _build_and_verify_installed_python_api(
    model_id: str,
    bundle: Path,
    cwd: Path,
    installed_python: Path,
    expected_binary: Path,
    expected_core: Path,
    build_env: dict[str, str],
) -> None:
    script = """
import json
import os
import pathlib
import re
import subprocess
import sys
import tensorrt_model_connect

repo = pathlib.Path(sys.argv[1]).resolve()
module = pathlib.Path(tensorrt_model_connect.__file__).resolve()
install_root = pathlib.Path(sys.prefix).resolve(strict=True)
if module == repo or repo in module.parents or install_root not in module.parents:
    raise SystemExit(f"installed build imported an invalid package: {module}")
bundle = pathlib.Path(sys.argv[3])
if bundle.exists():
    raise SystemExit(f"installed Python build output already exists: {bundle}")
tensorrt_model_connect.build(
    sys.argv[2],
    str(bundle),
    max_cache_length=4096,
    precision="fp16",
    max_batch_size=4,
)
if not bundle.is_file():
    raise SystemExit(f"installed Python build produced no bundle: {bundle}")
pipeline = tensorrt_model_connect.Pipeline(str(bundle))
binary = pathlib.Path(pipeline.binary).resolve(strict=True)
expected_binary = pathlib.Path(sys.argv[5]).resolve(strict=True)
expected_core = pathlib.Path(sys.argv[6]).resolve(strict=True)
if binary != expected_binary:
    raise SystemExit(f"installed Python API selected the wrong executable: {binary}")

# Do not repair the wheel's loader path. Remove source-tree entries while
# preserving the host TensorRT/CUDA paths, then require the executable's own
# $ORIGIN RUNPATH to resolve its bundled core.
clean_library_path = []
for raw_path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
    if not raw_path:
        continue
    path = pathlib.Path(raw_path).expanduser().resolve()
    if path == repo or repo in path.parents:
        continue
    clean_library_path.append(str(path))
os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(clean_library_path)
linked = subprocess.run(
    ["ldd", str(binary)], check=True, capture_output=True, text=True
).stdout.splitlines()
pattern = re.compile(r"^\\s*libtrtmc_core\\.so(?:\\.[^ ]+)?\\s+=>\\s+")
core_lines = [line for line in linked if pattern.match(line)]
if len(core_lines) != 1 or " (" not in core_lines[0]:
    raise SystemExit(
        "installed executable did not resolve exactly one bundled libtrtmc_core: "
        + repr(linked)
    )
loaded_core = pathlib.Path(core_lines[0].split("=>", 1)[1].rsplit(" (", 1)[0].strip()).resolve(
    strict=True
)
if loaded_core != expected_core:
    raise SystemExit(f"installed executable loaded the wrong libtrtmc_core: {loaded_core}")

inspection = pipeline.inspect()
if "optimized_runtime.json" not in inspection:
    raise SystemExit("installed Python build did not produce a delegated bundle")
generated = pipeline(sys.argv[4], max_new_tokens=8, timeout=600)
if not generated.strip():
    raise SystemExit("installed Python API returned an empty response")
print(json.dumps({
    "bundle": str(bundle.resolve(strict=True)),
    "binary": str(binary),
    "core": str(loaded_core),
    "generated": generated,
    "module": str(module),
}, sort_keys=True))
"""
    clean_env = dict(build_env)
    clean_env.pop("PYTHONPATH", None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    clean_env["XDG_CACHE_HOME"] = str(cwd / "installed-python-cache")
    result = _run(
        [
            str(installed_python),
            "-c",
            script,
            str(_REPO_ROOT),
            model_id,
            str(bundle),
            _PROMPT,
            str(expected_binary),
            str(expected_core),
        ],
        timeout=21_600,
        cwd=cwd,
        env=clean_env,
    )
    proof = json.loads(result.stdout.splitlines()[-1])
    assert proof["generated"].strip()
    assert Path(proof["bundle"]).resolve(strict=True) == bundle.resolve(strict=True)
    module = Path(proof["module"]).resolve(strict=True)
    assert module != _REPO_ROOT and _REPO_ROOT not in module.parents
    assert Path(proof["binary"]).resolve(strict=True) == expected_binary
    assert Path(proof["core"]).resolve(strict=True) == expected_core
    print("installed-package proof:", json.dumps(proof, sort_keys=True))


def _runner_request(kind: str, **runtime: str) -> dict[str, Any]:
    return {
        "schema": "trtmc.edgellm.long-lived.v1",
        "runtime": {"kind": kind, **runtime},
        "prompt": _PROMPT,
        "generation": {
            "max_new_tokens": _MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "use_chat_template": False,
            "enable_thinking": False,
        },
        "warmups_per_repetition": _WARMUPS,
        "measured_requests_per_repetition": _MEASURED_REQUESTS,
        "require_native_token_ids": True,
        "synchronize_each_request": True,
    }


def _run_qualification_runner(
    runner: Path,
    request: dict[str, Any],
    stem: str,
    *,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    request_path = cwd / f"{stem}-request.json"
    output_path = cwd / f"{stem}-output.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    _run(
        [str(runner), "--request", str(request_path), "--output", str(output_path)],
        timeout=7_200,
        cwd=cwd,
        env=env,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(result) == {
        "schema",
        "runtime_kind",
        "runtime_initializations",
        "decoding_cuda_graph_captured",
        "observed_tensorrt_version",
        "observed_cuda_runtime_version",
        "native_token_ids",
        "synchronized_each_request",
        "warmups_completed",
        "measured_elapsed_ms",
        "iterations",
    }
    assert result["schema"] == request["schema"]
    assert result["runtime_kind"] == request["runtime"]["kind"]
    assert result["runtime_initializations"] == 1
    assert result["decoding_cuda_graph_captured"] is True
    assert result["observed_tensorrt_version"] == "11.2.0.113"
    assert result["observed_cuda_runtime_version"] == 13030
    assert result["native_token_ids"] is True
    assert result["synchronized_each_request"] is True
    assert result["warmups_completed"] == _WARMUPS
    assert result["measured_elapsed_ms"] > 0.0
    iterations = result["iterations"]
    assert len(iterations) == _MEASURED_REQUESTS
    for iteration in iterations:
        assert set(iteration) == {"latency_ms", "generated", "token_ids"}
        assert iteration["latency_ms"] > 0.0
        assert iteration["generated"].strip()
        token_ids = iteration["token_ids"]
        assert token_ids
        assert all(isinstance(token, int) and token >= 0 for token in token_ids)
    return result


def _verify_direct_runtime_and_performance(
    direct_runner: Path,
    mc_runner: Path,
    bundle: Path,
    engine_dir: Path,
    runtime_cache: Path,
    mc_row: dict[str, Any],
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    direct_results: list[dict[str, Any]] = []
    mc_results: list[dict[str, Any]] = []
    direct_request = _runner_request("edgellm-direct", engine_dir=str(engine_dir))
    mc_request = _runner_request(
        "model-connect", bundle=str(bundle), runtime_cache=str(runtime_cache)
    )
    for repetition in range(_PERFORMANCE_REPETITIONS):
        order = (
            (
                (direct_runner, direct_request, "edge-direct", direct_results),
                (mc_runner, mc_request, "model-connect", mc_results),
            )
            if repetition % 2 == 0
            else (
                (mc_runner, mc_request, "model-connect", mc_results),
                (direct_runner, direct_request, "edge-direct", direct_results),
            )
        )
        for runner, request, stem, destination in order:
            destination.append(
                _run_qualification_runner(
                    runner,
                    request,
                    f"{stem}-{repetition}",
                    cwd=cwd,
                    env=env,
                )
            )

    direct_iterations = [item for result in direct_results for item in result["iterations"]]
    mc_iterations = [item for result in mc_results for item in result["iterations"]]
    assert (
        len(direct_iterations)
        == len(mc_iterations)
        == (_MEASURED_REQUESTS * _PERFORMANCE_REPETITIONS)
    )
    reference_text = direct_iterations[0]["generated"]
    reference_tokens = direct_iterations[0]["token_ids"]
    assert reference_text == mc_row["generated"]
    assert reference_tokens == mc_row["token_ids"]
    for direct_iteration, mc_iteration in zip(direct_iterations, mc_iterations, strict=True):
        assert direct_iteration["generated"] == reference_text
        assert mc_iteration["generated"] == reference_text
        assert direct_iteration["token_ids"] == reference_tokens
        assert mc_iteration["token_ids"] == reference_tokens

    direct_latencies = [item["latency_ms"] for item in direct_iterations]
    mc_latencies = [item["latency_ms"] for item in mc_iterations]
    mc_median = statistics.median(mc_latencies)
    direct_median = statistics.median(direct_latencies)
    median_ratio = mc_median / direct_median
    p95_ratio = _percentile(mc_latencies, 0.95) / _percentile(direct_latencies, 0.95)
    mc_elapsed = sum(result["measured_elapsed_ms"] for result in mc_results)
    direct_elapsed = sum(result["measured_elapsed_ms"] for result in direct_results)
    throughput_ratio = direct_elapsed / mc_elapsed
    performance = {
        "warmups_per_repetition": _WARMUPS,
        "measured_requests_per_repetition": _MEASURED_REQUESTS,
        "repetitions": _PERFORMANCE_REPETITIONS,
        "mc_median_ms": mc_median,
        "direct_median_ms": direct_median,
        "mc_p95_ms": _percentile(mc_latencies, 0.95),
        "direct_p95_ms": _percentile(direct_latencies, 0.95),
        "mc_measured_elapsed_ms": mc_elapsed,
        "direct_measured_elapsed_ms": direct_elapsed,
        "mc_to_direct_median_ratio": median_ratio,
        "mc_to_direct_p95_ratio": p95_ratio,
        "mc_to_direct_throughput_ratio": throughput_ratio,
    }
    print("EdgeLLM performance qualification:", json.dumps(performance, sort_keys=True))
    assert median_ratio <= 1.05, performance
    assert p95_ratio <= 1.10, performance
    assert throughput_ratio >= 0.95, performance


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.slow
def test_public_build_inspect_and_run_delegate_to_edgellm(tmp_path: Path) -> None:
    """Qualify the unchanged public workflow and its direct EdgeLLM delegation."""

    _require_supported_a100()
    outside_checkout = tmp_path / "outside-checkout"
    outside_checkout.mkdir()
    installed_python, installed_binary, installed_core = _installed_package_payload(
        outside_checkout
    )
    binary = _required_executable("TRTMC_BINARY")
    assert binary == installed_binary, (
        "TRTMC_BINARY must be the executable bundled in the installed wheel: "
        f"{binary} != {installed_binary}"
    )

    build_env = os.environ.copy()
    configured_edge_build: Path | None = None
    for public_name, internal_name in _PUBLIC_EDGE_BUILD_ENVIRONMENT.items():
        value = build_env.get(public_name, "").strip()
        if value:
            build_env[internal_name] = value
            if public_name == "TRTMC_EDGE_LLM_BUILD_DIR":
                configured_edge_build = Path(value).expanduser().resolve()
    exporter_root = tmp_path / "exporter-profiles"
    build_env["TRTMC_PYTHON_PROFILE_ROOT"] = str(exporter_root)
    bundle = tmp_path / "qwen3-4b-instruct-2507-edge.trtfb"
    build_command = [
        str(binary),
        "build",
        _MODEL_ID,
        "-o",
        str(bundle),
        "--precision",
        "fp16",
        "--max-cache-length",
        "4096",
        "--max-batch-size",
        "4",
    ]
    assert not exporter_root.exists()
    _run(build_command, timeout=21_600, cwd=outside_checkout, env=build_env)

    _header_size, header = _read_bundle_header(bundle)
    assert header["model_type"] == "qwen3"
    assert header["family"] == "qwen"
    descriptor = json.loads(_read_bundle_section(bundle, "optimized_runtime.json"))
    assert descriptor["implementation_id"] == _IMPLEMENTATION_ID
    inspect = _run(
        [str(binary), "inspect", str(bundle)],
        timeout=60,
        cwd=outside_checkout,
        env=build_env,
    )
    assert "optimized_runtime.json" in inspect.stdout
    assert "optimized_runtime_artifacts/engine.dir/" in inspect.stdout

    exporter_inventory = _verify_exporter_profile(exporter_root)
    edge_build_inventory: dict[str, tuple[int, int]] | None = None
    edge_build_stamp: dict[str, Any] | None = None
    if configured_edge_build is not None:
        edge_build_stamp = _verify_edge_build_stamp(configured_edge_build)
        edge_build_inventory = _tree_inventory(configured_edge_build)
        assert edge_build_inventory
    warm_bundle = tmp_path / "qwen3-4b-instruct-2507-edge-warm-profile.trtfb"
    warm_build_command = list(build_command)
    warm_build_command[warm_build_command.index(str(bundle))] = str(warm_bundle)
    _run(warm_build_command, timeout=21_600, cwd=outside_checkout, env=build_env)
    warm_exporter_inventory = _verify_exporter_profile(exporter_root)
    assert warm_exporter_inventory == exporter_inventory
    if configured_edge_build is not None:
        assert _verify_edge_build_stamp(configured_edge_build) == edge_build_stamp
        assert _tree_inventory(configured_edge_build) == edge_build_inventory
    warm_descriptor = json.loads(_read_bundle_section(warm_bundle, "optimized_runtime.json"))
    assert warm_descriptor["implementation_id"] == _IMPLEMENTATION_ID

    load_cache = tmp_path / "load-only-cache"
    expected_load_cache = (
        load_cache
        / "optimized-runtimes"
        / _IMPLEMENTATION_ID
        / f"{descriptor['profile_id']}-{descriptor['artifact']['tree_sha256']}"
    )
    core_library_value = os.environ.get("TRTMC_CORE_LIBRARY", "").strip()
    core_library = Path(core_library_value or installed_core).resolve(strict=True)
    assert core_library == installed_core, (
        "TRTMC_CORE_LIBRARY must be the core bundled in the installed wheel: "
        f"{core_library} != {installed_core}"
    )
    runtime_env = os.environ.copy()
    runtime_env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (str(core_library.parent), runtime_env.get("LD_LIBRARY_PATH", "")))
    )
    assert not load_cache.exists()
    load_only = _load_only_with_c_api(
        core_library,
        bundle,
        load_cache,
        cwd=outside_checkout,
        env=runtime_env,
    )
    assert load_only.stdout.strip() == "load-only-ok"
    assert "Optimized-runtime implementation initialized during load" in load_only.stderr
    assert expected_load_cache.is_dir()
    assert (expected_load_cache / "engine.dir").is_dir()
    assert _tree_inventory(expected_load_cache)

    runtime_cache = tmp_path / "runtime-cache"
    expected_runtime_cache = (
        runtime_cache
        / "optimized-runtimes"
        / _IMPLEMENTATION_ID
        / f"{descriptor['profile_id']}-{descriptor['artifact']['tree_sha256']}"
    )
    cold_output = tmp_path / "generated-cold.jsonl"
    warm_output = tmp_path / "generated-warm.jsonl"
    assert not runtime_cache.exists()
    _run(
        _mc_run_command(binary, bundle, runtime_cache, cold_output),
        timeout=600,
        cwd=outside_checkout,
        env=runtime_env,
    )
    assert expected_runtime_cache.is_dir()
    assert (expected_runtime_cache / "engine.dir").is_dir()
    cold_inventory = _tree_inventory(expected_runtime_cache)
    assert cold_inventory
    cold_rows = _rows(cold_output)
    assert len(cold_rows) == 1
    assert cold_rows[0]["generated"].strip()
    assert cold_rows[0]["token_ids"]

    _run(
        _mc_run_command(binary, bundle, runtime_cache, warm_output),
        timeout=600,
        cwd=outside_checkout,
        env=runtime_env,
    )
    assert _tree_inventory(expected_runtime_cache) == cold_inventory
    warm_rows = _rows(warm_output)
    assert len(warm_rows) == 1
    assert warm_rows[0]["token_ids"] == cold_rows[0]["token_ids"]
    assert warm_rows[0]["generated"] == cold_rows[0]["generated"]

    installed_python_bundle = tmp_path / "qwen3-4b-instruct-2507-edge-installed-python.trtfb"
    _build_and_verify_installed_python_api(
        _MODEL_ID,
        installed_python_bundle,
        outside_checkout,
        installed_python,
        installed_binary,
        installed_core,
        build_env,
    )
    installed_descriptor = json.loads(
        _read_bundle_section(installed_python_bundle, "optimized_runtime.json")
    )
    assert installed_descriptor["implementation_id"] == _IMPLEMENTATION_ID

    if configured_edge_build is None:
        pytest.fail(
            "TRTMC_EDGE_LLM_BUILD_DIR is required with qualification runners so the exact "
            "stamped EdgeLLM build products can be revalidated before execution"
        )
    assert _verify_edge_build_stamp(configured_edge_build) == edge_build_stamp
    direct_runner = _required_executable("TRTMC_EDGELLM_DIRECT_RUNNER")
    mc_runner = _required_executable("TRTMC_EDGELLM_MC_RUNNER")
    _verify_direct_runtime_and_performance(
        direct_runner,
        mc_runner,
        bundle,
        expected_runtime_cache / "engine.dir",
        runtime_cache,
        cold_rows[0],
        cwd=outside_checkout,
        env=runtime_env,
    )
