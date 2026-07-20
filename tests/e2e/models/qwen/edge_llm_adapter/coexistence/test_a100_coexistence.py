# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A100 same-process qualification for the Qwen EdgeLLM adapter leaves."""

from __future__ import annotations

import ast
import json
import os
import struct
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pytest

from tensorrt_model_connect.runtime_provider.target import (
    TargetResolutionError,
    _probe_current_target_with_device,
)


_PROMPT = "Reply with one short sentence about accelerated computing."
_EDGE_COMMIT = "1ac0f2b99642045125e1c5ac7b109434ba3b36c7"
_REPOSITORY = Path(__file__).resolve().parents[6]
_LEAF = Path(__file__).resolve().parent
_BUILDER_ROOT = _REPOSITORY / "python/tensorrt_model_connect/families/qwen/edge_llm_adapter"
_TEST_ROOT = _LEAF.parent
_BUNDLES_ENVIRONMENT = "TRTMC_QWEN_EDGELLM_BUNDLES_JSON"


@dataclass(frozen=True)
class _Profile:
    leaf: str
    model_id: str
    implementation_id: str
    runtime_library: str
    internal_source_environment: str
    internal_build_environment: str


def _literal_assignment(path: Path, name: str) -> object:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            values.append(ast.literal_eval(statement.value))
    if len(values) != 1:
        raise ValueError(f"{path} must declare exactly one literal {name}")
    return values[0]


def _build_environment_mapping(strict_test: Path) -> tuple[str, str]:
    mapping = _literal_assignment(strict_test, "_PUBLIC_EDGE_BUILD_ENVIRONMENT")
    expected = {"TRTMC_EDGE_LLM_SOURCE_DIR", "TRTMC_EDGE_LLM_BUILD_DIR"}
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise ValueError(f"{strict_test} must map the two public EdgeLLM build inputs")
    if not all(
        isinstance(value, str) and value.startswith("_TRTMC_INTERNAL_")
        for value in mapping.values()
    ) or len(set(mapping.values())) != len(mapping):
        raise ValueError(f"{strict_test} contains invalid internal build mappings")
    return mapping["TRTMC_EDGE_LLM_SOURCE_DIR"], mapping["TRTMC_EDGE_LLM_BUILD_DIR"]


def _profile_from_manifest(manifest: Path, test_root: Path) -> _Profile:
    leaf = manifest.parent.name
    strict_test = test_root / leaf / "test_a100_e2e.py"
    if not strict_test.is_file():
        raise ValueError(f"Qwen EdgeLLM profile {leaf} has no strict A100 test")
    with manifest.open("rb") as source:
        descriptor = tomllib.load(source)
    source_environment, build_environment = _build_environment_mapping(strict_test)
    model = descriptor.get("model")
    runtime = descriptor.get("runtime")
    if (
        descriptor.get("schema_version") != 1
        or descriptor.get("downstream_runtime") != "tensorrt-edge-llm"
        or descriptor.get("downstream_version") != "0.9.0"
        or descriptor.get("downstream_commit") != _EDGE_COMMIT
        or not isinstance(model, dict)
        or not isinstance(runtime, dict)
        or runtime.get("abi") != 1
    ):
        raise ValueError(f"Qwen EdgeLLM profile has an unsupported descriptor: {manifest}")
    model_id = model.get("id")
    implementation_id = descriptor.get("implementation_id")
    runtime_library = runtime.get("library")
    if not all(
        isinstance(value, str) and value for value in (model_id, implementation_id, runtime_library)
    ):
        raise ValueError(f"Qwen EdgeLLM profile has incomplete identity fields: {manifest}")
    return _Profile(
        leaf,
        model_id,
        implementation_id,
        runtime_library,
        source_environment,
        build_environment,
    )


def _discover_profiles(
    builder_root: Path = _BUILDER_ROOT, test_root: Path = _TEST_ROOT
) -> tuple[_Profile, ...]:
    profiles = tuple(
        _profile_from_manifest(manifest, test_root)
        for manifest in sorted(builder_root.glob("*/IMPLEMENTATION.toml"))
    )
    if not profiles:
        raise ValueError(f"no Qwen EdgeLLM profiles were found below {builder_root}")
    identities = (
        ("model IDs", [profile.model_id for profile in profiles]),
        ("implementation IDs", [profile.implementation_id for profile in profiles]),
        ("runtime libraries", [profile.runtime_library for profile in profiles]),
        (
            "internal build environment names",
            [
                name
                for profile in profiles
                for name in (
                    profile.internal_source_environment,
                    profile.internal_build_environment,
                )
            ],
        ),
    )
    for description, values in identities:
        if len(set(values)) != len(values):
            raise ValueError(f"Qwen EdgeLLM profiles have duplicate {description}")
    return profiles


def _representative_load_orders(
    profiles: Sequence[_Profile],
) -> tuple[tuple[_Profile, ...], ...]:
    ordered = tuple(profiles)
    if len(ordered) < 2:
        raise ValueError("coexistence requires at least two profiles")
    candidates: list[tuple[_Profile, ...]] = []
    for direction in (ordered, tuple(reversed(ordered))):
        for offset in range(len(direction)):
            candidate = direction[offset:] + direction[:offset]
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


_PROFILES = _discover_profiles()


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
        pytest.fail(f"the Qwen coexistence proof could not inspect its CUDA target: {exc}")
    if target["gpu_name"] != "NVIDIA A100 80GB PCIe":
        pytest.fail(
            f"the Qwen coexistence proof requires NVIDIA A100 80GB PCIe; found {target['gpu_name']}"
        )


def _required_file(environment_name: str, *, executable: bool = False) -> Path:
    value = os.environ.get(environment_name, "").strip()
    if not value:
        pytest.fail(f"{environment_name} is required for Qwen EdgeLLM coexistence")
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        pytest.fail(f"{environment_name} is not a file: {path}")
    if executable and not os.access(path, os.X_OK):
        pytest.fail(f"{environment_name} is not executable: {path}")
    return path


def _read_bundle_header(bundle: Path) -> tuple[int, dict[str, Any]]:
    with bundle.open("rb") as stream:
        assert stream.read(8) == b"TRTFB\x00\x01\x00"
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return header_size, header


def _read_bundle_section(bundle: Path, name: str) -> bytes:
    header_size, header = _read_bundle_header(bundle)
    section = header["sections"][name]
    with bundle.open("rb") as stream:
        stream.seek(16 + header_size + section["offset"])
        return stream.read(section["size"])


def _validate_delegated_bundle(bundle: Path, profile: _Profile) -> None:
    _header_size, header = _read_bundle_header(bundle)
    assert header["model_id"] == profile.model_id
    sections = header["sections"]
    assert "optimized_runtime.json" in sections
    descriptor = json.loads(_read_bundle_section(bundle, "optimized_runtime.json"))
    assert descriptor["schema_version"] == 2
    assert descriptor["implementation_id"] == profile.implementation_id
    assert descriptor["model_id"] == profile.model_id
    assert descriptor["runtime_library"] == profile.runtime_library
    assert descriptor["factory_abi"] == 1
    assert descriptor["runtime"] == {
        "name": "tensorrt-edge-llm",
        "version": "0.9.0",
        "commit": _EDGE_COMMIT,
    }
    artifact = descriptor["artifact"]
    assert artifact["section_prefix"] == "optimized_runtime_artifacts"
    assert "engine.dir" in artifact["directories"]
    assert artifact["file_count"] > 0
    assert artifact["total_size"] > 0
    assert len(artifact["tree_sha256"]) == 64
    artifact_sections = [
        name for name in sections if name.startswith("optimized_runtime_artifacts/")
    ]
    assert len(artifact_sections) == artifact["file_count"]
    assert f"optimized_runtime_artifacts/{profile.runtime_library}" in artifact_sections
    assert any(
        name.startswith("optimized_runtime_artifacts/engine.dir/") for name in artifact_sections
    )


def _build_or_resolve_bundles(tmp_path: Path) -> dict[_Profile, Path]:
    configured = os.environ.get(_BUNDLES_ENVIRONMENT, "").strip()
    if configured:
        try:
            paths = json.loads(configured)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{_BUNDLES_ENVIRONMENT} is not valid JSON: {exc}")
        expected = {profile.leaf for profile in _PROFILES}
        if not isinstance(paths, dict) or set(paths) != expected:
            pytest.fail(
                f"{_BUNDLES_ENVIRONMENT} must map exactly the discovered leaves: "
                + ", ".join(sorted(expected))
            )
        if not all(isinstance(value, str) and value for value in paths.values()):
            pytest.fail(f"{_BUNDLES_ENVIRONMENT} values must be non-empty paths")
        bundles = {
            profile: Path(paths[profile.leaf]).expanduser().resolve(strict=True)
            for profile in _PROFILES
        }
    else:
        binary = _required_file("TRTMC_BINARY", executable=True)
        build_environment = os.environ.copy()
        edge_source = build_environment.get("TRTMC_EDGE_LLM_SOURCE_DIR", "").strip()
        edge_build = build_environment.get("TRTMC_EDGE_LLM_BUILD_DIR", "").strip()
        if not edge_source or not edge_build:
            pytest.fail(
                "TRTMC_EDGE_LLM_SOURCE_DIR and TRTMC_EDGE_LLM_BUILD_DIR are required "
                "when the coexistence gate builds its own bundles"
            )
        for profile in _PROFILES:
            build_environment[profile.internal_source_environment] = edge_source
            build_environment[profile.internal_build_environment] = edge_build
        build_environment["TRTMC_PYTHON_PROFILE_ROOT"] = str(tmp_path / "exporter-profiles")
        outside_checkout = tmp_path / "outside-checkout"
        outside_checkout.mkdir()
        bundles = {}
        for profile in _PROFILES:
            bundle = tmp_path / f"{profile.leaf}.trtfb"
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
                    "4096",
                    "--max-batch-size",
                    "4",
                ],
                timeout=21_600,
                cwd=outside_checkout,
                env=build_environment,
            )
            bundles[profile] = bundle.resolve(strict=True)
    for profile, bundle in bundles.items():
        _validate_delegated_bundle(bundle, profile)
    return bundles


def _build_runner(tmp_path: Path, core_library: Path) -> Path:
    build = tmp_path / "coexistence-runner-build"
    _run(
        [
            "cmake",
            "-S",
            str(_LEAF),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DTRTMC_MC_INCLUDE_DIR={_REPOSITORY / 'include'}",
            f"-DTRTMC_MC_CORE_LIBRARY={core_library}",
        ],
        timeout=300,
    )
    _run(
        [
            "cmake",
            "--build",
            str(build),
            "--parallel",
            "4",
            "--target",
            "trtmc_qwen_edgellm_coexistence_runner",
        ],
        timeout=600,
    )
    runner = build / "trtmc_qwen_edgellm_coexistence_runner"
    assert runner.is_file() and os.access(runner, os.X_OK)
    return runner


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.slow
def test_discovered_delegated_bundles_coexist_in_representative_load_orders(
    tmp_path: Path,
) -> None:
    """Keep all discovered runtimes alive across bounded representative orders."""

    _require_supported_a100()
    bundles = _build_or_resolve_bundles(tmp_path)
    core_library = _required_file("TRTMC_CORE_LIBRARY")
    runner = _build_runner(tmp_path, core_library)
    runtime_environment = os.environ.copy()
    runtime_environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(
            None,
            (str(core_library.parent), runtime_environment.get("LD_LIBRARY_PATH", "")),
        )
    )
    runtime_cache = tmp_path / "runtime-cache"
    expected: dict[Path, tuple[str, tuple[int, ...], str, str]] = {}

    for order_index, order in enumerate(_representative_load_orders(_PROFILES)):
        output = tmp_path / f"coexistence-{order_index}.json"
        ordered_bundles = [bundles[profile] for profile in order]
        result = _run(
            [
                str(runner),
                "--runtime-cache",
                str(runtime_cache),
                "--output",
                str(output),
                "--prompt",
                _PROMPT,
                "--max-new-tokens",
                "8",
                *(str(bundle) for bundle in ordered_bundles),
            ],
            timeout=1_800,
            cwd=tmp_path,
            env=runtime_environment,
        )
        assert result.stdout.strip() == "coexistence-ok"
        proof = json.loads(output.read_text(encoding="utf-8"))
        assert set(proof) == {"schema", "load_order", "forward", "reverse", "concurrent"}
        assert proof["schema"] == "trtmc.qwen.edgellm.coexistence.v1"
        assert proof["load_order"] == [str(bundle) for bundle in ordered_bundles]
        assert [row["bundle"] for row in proof["forward"]] == proof["load_order"]
        assert [row["bundle"] for row in proof["reverse"]] == list(reversed(proof["load_order"]))
        assert [row["bundle"] for row in proof["concurrent"]] == proof["load_order"]
        for profile, row in zip(order, proof["forward"], strict=True):
            assert set(row) == {
                "bundle",
                "model_id",
                "pipeline_type",
                "generated",
                "token_ids",
            }
            assert row["model_id"] == profile.model_id
            assert row["pipeline_type"]
            assert row["generated"].strip()
            assert row["token_ids"]
            observation = (
                row["generated"],
                tuple(row["token_ids"]),
                row["model_id"],
                row["pipeline_type"],
            )
            bundle = bundles[profile]
            if bundle in expected:
                assert observation == expected[bundle]
            else:
                expected[bundle] = observation
        reverse_by_bundle = {Path(row["bundle"]): row for row in proof["reverse"]}
        concurrent_by_bundle = {Path(row["bundle"]): row for row in proof["concurrent"]}
        for bundle, observation in expected.items():
            if bundle not in ordered_bundles:
                continue
            reverse = reverse_by_bundle[bundle]
            assert (
                reverse["generated"],
                tuple(reverse["token_ids"]),
                reverse["model_id"],
                reverse["pipeline_type"],
            ) == observation
            concurrent = concurrent_by_bundle[bundle]
            assert (
                concurrent["generated"],
                tuple(concurrent["token_ids"]),
                concurrent["model_id"],
                concurrent["pipeline_type"],
            ) == observation

    assert set(expected) == set(bundles.values())
