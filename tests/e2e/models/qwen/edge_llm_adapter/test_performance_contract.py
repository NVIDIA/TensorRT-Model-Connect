# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for the family-owned EdgeLLM performance qualification."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e.models.qwen.edge_llm_adapter import performance_harness
from tests.e2e.models.qwen.edge_llm_adapter import test_a100_e2e


_ADAPTER_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[5]
_SOURCE_IDENTITY = {
    "kind": "archive_sha256",
    "value": "a" * 64,
    "dirty": False,
}
_RUNTIME_IDENTITY = performance_harness.EdgeRuntimeIdentity(
    performance_harness.EDGE_NAME,
    performance_harness.EDGE_VERSION,
    performance_harness.EDGE_COMMIT,
)


def _runner_result(
    kind: str,
    *,
    latencies: list[float] | None = None,
    elapsed_ms: float | None = None,
    generated: str = "Accelerated computing is fast.",
    token_ids: list[int] | None = None,
) -> dict[str, Any]:
    samples = latencies or [100.0] * performance_harness.MEASURED_REQUESTS
    assert len(samples) == performance_harness.MEASURED_REQUESTS
    return {
        "schema": performance_harness.SCHEMA,
        "runtime_kind": kind,
        "runtime_initializations": 1,
        "decoding_cuda_graph_captured": True,
        "observed_tensorrt_version": performance_harness.TENSORRT_VERSION,
        "observed_cuda_runtime_version": performance_harness.CUDA_RUNTIME_VERSION,
        "native_token_ids": True,
        "synchronized_each_request": True,
        "warmups_completed": performance_harness.WARMUPS,
        "measured_elapsed_ms": elapsed_ms or sum(samples),
        "iterations": [
            {
                "latency_ms": latency,
                "generated": generated,
                "token_ids": token_ids or [1, 2, 3],
            }
            for latency in samples
        ],
    }


def _results(
    kind: str,
    *,
    latencies: list[float] | None = None,
    elapsed_ms: float | None = None,
    generated: str = "Accelerated computing is fast.",
    token_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    return [
        _runner_result(
            kind,
            latencies=latencies,
            elapsed_ms=elapsed_ms,
            generated=generated,
            token_ids=token_ids,
        )
        for _ in range(performance_harness.REPETITIONS)
    ]


def _evaluate(
    direct_results: list[dict[str, Any]],
    mc_results: list[dict[str, Any]],
    *,
    source_identity: dict[str, Any] = _SOURCE_IDENTITY,
) -> dict[str, Any]:
    return performance_harness.evaluate_performance(
        profile_id="profile",
        model_id="Qwen/model",
        revision="revision",
        source_identity=source_identity,
        runtime_identity=_RUNTIME_IDENTITY,
        direct_results=direct_results,
        mc_results=mc_results,
    )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_repository(path: Path, *, origin: str | None = None) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "TRTMC contract test")
    _git(path, "config", "user.email", "trtmc-contract@example.invalid")
    if origin is not None:
        _git(path, "remote", "add", "origin", origin)
    return path


def test_performance_cohort_is_loaded_from_the_model_owned_dependency_lock() -> None:
    pins = performance_harness.load_dependency_pins()

    assert performance_harness.DEPENDENCY_LOCK_PATH == (
        _REPO_ROOT / "python/tensorrt_model_connect/families/qwen/edge_llm_adapter/dependency.lock"
    )
    assert pins == performance_harness.DEPENDENCY_PINS
    assert performance_harness.EDGE_NAME == pins.edge_name
    assert performance_harness.EDGE_SOURCE == pins.edge_source
    assert performance_harness.EDGE_VERSION == pins.edge_version
    assert performance_harness.EDGE_COMMIT == pins.edge_commit
    assert performance_harness.TENSORRT_VERSION == pins.tensorrt_version
    assert performance_harness.TENSORRT_VERSION_PARTS == pins.tensorrt_version_parts
    assert performance_harness.CUDA_VERSION == pins.cuda_version
    assert performance_harness.CUDA_RUNTIME_VERSION == pins.cuda_runtime_version


@pytest.mark.parametrize(
    ("field", "replacement", "expected"),
    (
        ("tag", 'tag = "v9.9.8"', "tag must match"),
        ("commit", 'commit = "not-a-commit"', "lowercase Git commit"),
        ("tensorrt", 'version = "9.9"', "major.minor.patch.build"),
        ("cuda", 'version = "9.9.1"', "major.minor"),
    ),
)
def test_dependency_lock_parser_fails_closed(
    tmp_path: Path, field: str, replacement: str, expected: str
) -> None:
    sections = {
        "tag": 'tag = "v9.9.9"',
        "commit": f'commit = "{"a" * 40}"',
        "tensorrt": 'version = "9.9.9.9"',
        "cuda": 'version = "9.9"',
    }
    sections[field] = replacement
    path = tmp_path / "dependency.lock"
    path.write_text(
        inspect.cleandoc(
            f"""
            schema_version = 1

            [downstream]
            name = "tensorrt-edge-llm"
            source = "https://example.invalid/edge.git"
            version = "9.9.9"
            {sections["tag"]}
            {sections["commit"]}
            source_mode = "git"

            [tensorrt]
            {sections["tensorrt"]}

            [cuda]
            {sections["cuda"]}
            """
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(performance_harness.PerformanceContractError, match=expected):
        performance_harness.load_dependency_pins(path)


def _edge_provenance_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_submodule: bool = False,
) -> SimpleNamespace:
    source = _git_repository(tmp_path / "edge-source", origin=performance_harness.EDGE_SOURCE)
    (source / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n")
    submodule_checkout = None
    if with_submodule:
        nested = _git_repository(tmp_path / "nested-source")
        (nested / "nested.txt").write_text("clean\n", encoding="utf-8")
        _git(nested, "add", "nested.txt")
        _git(nested, "commit", "-q", "-m", "nested fixture")
        _git(
            source,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(nested),
            "3rdParty/nested",
        )
        submodule_checkout = source / "3rdParty" / "nested"
    _git(source, "add", "-A")
    _git(source, "commit", "-q", "-m", "EdgeLLM fixture")
    revision = _git(source, "rev-parse", "HEAD")
    monkeypatch.setattr(performance_harness, "EDGE_COMMIT", revision)

    build = tmp_path / "edge-build"
    inference = build / "examples" / "llm" / "llm_inference"
    inference.parent.mkdir(parents=True)
    inference.write_bytes(b"official inference target built after the stamped targets")
    product_paths = (
        build / "cpp" / "libedgellmCore.a",
        build / "libNvInfer_edgellm_plugin.so.1.0",
        build / "examples" / "llm" / "llm_build",
    )
    for product in product_paths:
        product.parent.mkdir(parents=True, exist_ok=True)
        product.write_bytes(product.name.encode("utf-8"))
    products = {
        product.relative_to(build).as_posix(): performance_harness._sha256(product)
        for product in product_paths
    }
    recipe = {
        "configure_definitions": {"CMAKE_BUILD_TYPE": "Release"},
        "edge_commit": revision,
        "edge_version": performance_harness.EDGE_VERSION,
        "schema_version": 1,
        "source": str(source.resolve()),
        "targets": list(performance_harness._EDGE_BUILD_TARGETS),
        "toolchain_sha256": "d" * 64,
    }
    recipe_sha256 = performance_harness._canonical_sha256(recipe)
    stamp = {
        "products": products,
        "recipe": recipe,
        "recipe_sha256": recipe_sha256,
        "schema_version": 1,
    }
    stamp_path = build / performance_harness._EDGE_BUILD_STAMP
    stamp_path.write_text(json.dumps(stamp) + "\n", encoding="utf-8")
    cache_path = build / "CMakeCache.txt"
    cache_path.write_text(
        "\n".join(
            (
                f"CMAKE_HOME_DIRECTORY:INTERNAL={source.resolve()}",
                f"TRTMC_EDGE_BUILD_RECIPE_SHA256:STRING={recipe_sha256}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        build=build,
        cache_path=cache_path,
        inference=inference,
        revision=revision,
        source=source,
        stamp_path=stamp_path,
        submodule=submodule_checkout,
    )


def _validate_provenance(fixture: SimpleNamespace) -> performance_harness.EdgeRuntimeIdentity:
    assert performance_harness.edge_build_root(fixture.inference) == fixture.build
    return performance_harness.validate_edge_build_provenance(
        fixture.build,
        performance_harness.parse_cmake_cache(fixture.cache_path),
    )


def test_direct_edgellm_build_provenance_is_bound_to_pinned_clean_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _edge_provenance_fixture(tmp_path, monkeypatch)
    identity = _validate_provenance(fixture)
    assert identity == performance_harness.EdgeRuntimeIdentity(
        "tensorrt-edge-llm", performance_harness.EDGE_VERSION, fixture.revision
    )
    stamp = json.loads(fixture.stamp_path.read_text(encoding="utf-8"))
    assert "examples/llm/llm_inference" not in stamp["products"]


def test_official_inference_binary_is_built_from_the_qualified_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _edge_provenance_fixture(tmp_path, monkeypatch)
    cmake = tmp_path / "cmake"
    cmake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cmake.chmod(0o755)
    monkeypatch.setattr(performance_harness.shutil, "which", lambda name: str(cmake))

    assert performance_harness.prepare_official_inference_binary(fixture.build) == (
        fixture.inference.resolve()
    )


def test_official_inference_binary_fails_when_target_produces_no_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _edge_provenance_fixture(tmp_path, monkeypatch)
    fixture.inference.unlink()
    cmake = tmp_path / "cmake"
    cmake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cmake.chmod(0o755)
    monkeypatch.setattr(performance_harness.shutil, "which", lambda name: str(cmake))

    with pytest.raises(performance_harness.PerformanceContractError, match="exactly one"):
        performance_harness.prepare_official_inference_binary(fixture.build)


@pytest.mark.parametrize(
    ("failure", "expected"),
    (("missing", "missing a regular"), ("malformed", "stamp is malformed"), ("wrong", "digest")),
)
def test_direct_edgellm_build_stamp_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected: str,
) -> None:
    fixture = _edge_provenance_fixture(tmp_path, monkeypatch)
    if failure == "missing":
        fixture.stamp_path.unlink()
    elif failure == "malformed":
        fixture.stamp_path.write_text("{\n", encoding="utf-8")
    else:
        stamp = json.loads(fixture.stamp_path.read_text(encoding="utf-8"))
        stamp["products"]["cpp/libedgellmCore.a"] = "0" * 64
        fixture.stamp_path.write_text(json.dumps(stamp) + "\n", encoding="utf-8")
    with pytest.raises(performance_harness.PerformanceContractError, match=expected):
        _validate_provenance(fixture)


def test_direct_edgellm_build_rejects_cmake_and_stamp_source_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _edge_provenance_fixture(tmp_path, monkeypatch)
    other_source = tmp_path / "other-source"
    other_source.mkdir()
    fixture.cache_path.write_text(
        fixture.cache_path.read_text(encoding="utf-8").replace(
            str(fixture.source.resolve()), str(other_source.resolve())
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        performance_harness.PerformanceContractError, match="pinned source and build"
    ):
        _validate_provenance(fixture)


def test_direct_edgellm_build_rejects_wrong_git_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _edge_provenance_fixture(tmp_path, monkeypatch)
    _git(fixture.source, "commit", "-q", "--allow-empty", "-m", "wrong commit")
    with pytest.raises(performance_harness.PerformanceContractError, match="must be pinned"):
        _validate_provenance(fixture)


@pytest.mark.parametrize("dirty_kind", ("tracked", "untracked", "ignored", "submodule"))
def test_direct_edgellm_build_rejects_dirty_source_and_submodules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirty_kind: str
) -> None:
    fixture = _edge_provenance_fixture(
        tmp_path, monkeypatch, with_submodule=dirty_kind == "submodule"
    )
    if dirty_kind == "tracked":
        (fixture.source / "CMakeLists.txt").write_text("dirty\n", encoding="utf-8")
    elif dirty_kind == "untracked":
        (fixture.source / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    elif dirty_kind == "ignored":
        (fixture.source / "source.ignored").write_text("dirty\n", encoding="utf-8")
    else:
        (fixture.submodule / "nested.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(performance_harness.PerformanceContractError, match="must have no tracked"):
        _validate_provenance(fixture)


def test_performance_summary_requires_verified_runtime_identity() -> None:
    wrong = performance_harness.EdgeRuntimeIdentity(
        "tensorrt-edge-llm", performance_harness.EDGE_VERSION, "0" * 40
    )
    with pytest.raises(
        performance_harness.PerformanceContractError, match="not the pinned EdgeLLM release"
    ):
        performance_harness.evaluate_performance(
            profile_id="profile",
            model_id="Qwen/model",
            revision="revision",
            source_identity=_SOURCE_IDENTITY,
            runtime_identity=wrong,
            direct_results=_results("edgellm-direct"),
            mc_results=_results("model-connect"),
        )


def _write_qualification_profile(
    root: Path,
    name: str,
    *,
    qualification_state: str = "qualified",
    target_overrides: dict[str, str] | None = None,
) -> Path:
    target = {**test_a100_e2e._A100_TARGET, **(target_overrides or {})}
    path = root / name
    path.write_text(
        inspect.cleandoc(
            f'''
            schema_version = 1
            profile_id = "{name.removesuffix(".toml")}"
            qualification_state = "{qualification_state}"

            [model]
            id = "Qwen/Test"
            revisions = ["{"a" * 40}"]

            [target]
            os = "{target["os"]}"
            architecture = "{target["architecture"]}"
            platform_kind = "{target["platform_kind"]}"
            gpu_architecture = "{target["gpu_architecture"]}"
            gpu_name = "{target["gpu_name"]}"

            [artifacts]
            required_files = ["config.json"]
            '''
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("target_key", "wrong_value"),
    (
        ("os", "windows"),
        ("architecture", "aarch64"),
        ("platform_kind", "integrated"),
        ("gpu_architecture", "sm120"),
        ("gpu_name", "NVIDIA GeForce RTX 5090"),
    ),
)
def test_a100_profile_selection_uses_exact_target_and_qualified_state(
    tmp_path: Path, target_key: str, wrong_value: str
) -> None:
    selected = _write_qualification_profile(tmp_path, "a.toml")
    _write_qualification_profile(
        tmp_path,
        "candidate.toml",
        qualification_state="candidate",
    )
    _write_qualification_profile(
        tmp_path,
        "other-target.toml",
        target_overrides={target_key: wrong_value},
    )

    profiles = test_a100_e2e._profiles("", profiles_root=tmp_path)
    assert [profile.file_name for profile in profiles] == [selected.name]
    assert test_a100_e2e._profiles(selected.name, profiles_root=tmp_path) == profiles
    with pytest.raises(RuntimeError, match="outside the exact A100 target"):
        test_a100_e2e._profiles("other-target.toml", profiles_root=tmp_path)
    with pytest.raises(RuntimeError, match="selected non-qualified profile"):
        test_a100_e2e._profiles("candidate.toml", profiles_root=tmp_path)


@pytest.mark.parametrize(
    "selection",
    (
        "missing.toml",
        "../a.toml",
        "/tmp/a.toml",
        "nested/a.toml",
        "a.txt",
        "a.toml,a.toml",
        ",a.toml",
        "a.toml,",
        "a.toml,,b.toml",
        " a.toml",
        "a.toml ",
    ),
)
def test_a100_profile_file_override_is_strict(tmp_path: Path, selection: str) -> None:
    _write_qualification_profile(tmp_path, "a.toml")
    _write_qualification_profile(tmp_path, "b.toml")
    with pytest.raises(RuntimeError, match="TRTMC_QUALIFICATION_PROFILE_FILES"):
        test_a100_e2e._profiles(selection, profiles_root=tmp_path)


def test_a100_profile_selection_preserves_order_and_rejects_symlinks(tmp_path: Path) -> None:
    _write_qualification_profile(tmp_path, "a.toml")
    _write_qualification_profile(tmp_path, "b.toml")
    assert [
        profile.file_name
        for profile in test_a100_e2e._profiles("b.toml,a.toml", profiles_root=tmp_path)
    ] == ["b.toml", "a.toml"]

    (tmp_path / "link.toml").symlink_to(tmp_path / "a.toml")
    with pytest.raises(RuntimeError, match="must be a regular file"):
        test_a100_e2e._profiles("link.toml", profiles_root=tmp_path)


def test_a100_profile_selection_and_coexistence_fail_closed(tmp_path: Path) -> None:
    _write_qualification_profile(
        tmp_path,
        "candidate.toml",
        qualification_state="candidate",
    )
    with pytest.raises(RuntimeError, match="no qualified Qwen EdgeLLM A100 profiles"):
        test_a100_e2e._profiles("", profiles_root=tmp_path)

    full = test_a100_e2e._PROFILES
    assert len(full) >= 2
    assert test_a100_e2e._coexistence_profiles(full, "") == full[:2]
    assert test_a100_e2e._coexistence_profiles(full, full[0].file_name) == ()


def test_a100_producer_exports_one_source_bound_consumer_transfer(tmp_path: Path) -> None:
    profile = test_a100_e2e._PROFILES[0]
    bundle = tmp_path / "source.trtfb"
    bundle.write_bytes(b"real-producer-bundle-fixture")
    wheel = tmp_path / "tensorrt_model_connect-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"exact-wheel-fixture")
    export = tmp_path / "transfer"

    test_a100_e2e._export_consumer_transfer(profile, bundle, wheel, export, "a" * 40)

    assert sorted(path.name for path in export.iterdir()) == [
        "SHA256SUMS",
        "delegated.trtfb",
        "tensorrt_model_connect-1.0.0-py3-none-any.whl",
        "transfer-manifest.json",
    ]
    manifest = json.loads((export / "transfer-manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_revision"] == "a" * 40
    assert manifest["model_id"] == profile.model_id
    assert manifest["model_revision"] == profile.revision
    assert manifest["profile_id"] == profile.profile_id
    assert manifest["bundle_sha256"] == test_a100_e2e._sha256(export / "delegated.trtfb")
    subprocess.run(
        ["sha256sum", "--check", "--strict", "SHA256SUMS"],
        cwd=export,
        check=True,
        capture_output=True,
        text=True,
    )


def test_family_performance_test_is_parameterized_by_selected_a100_profiles() -> None:
    profile_ids = tuple(profile.profile_id for profile in test_a100_e2e._PROFILES)
    parameter_ids = tuple(
        parameter.values[0].profile_id for parameter in test_a100_e2e._PROFILE_PARAMETERS
    )
    assert all(profile.qualification_state == "qualified" for profile in test_a100_e2e._PROFILES)
    assert parameter_ids == profile_ids

    parameter_marks = [
        mark
        for mark in test_a100_e2e.test_model_connect_performance_matches_direct_edgellm.pytestmark
        if mark.name == "parametrize"
    ]
    assert len(parameter_marks) == 1
    assert parameter_marks[0].args == ("profile", test_a100_e2e._PROFILE_PARAMETERS)


def test_request_contract_is_identical_except_for_runtime_location() -> None:
    direct = performance_harness.runner_request(
        "edgellm-direct", engine_dir="/engine.dir", plugin="/plugin.so"
    )
    model_connect = performance_harness.runner_request(
        "model-connect", bundle="/model.trtfb", runtime_cache="/cache"
    )
    assert direct["prompt"] == model_connect["prompt"]
    assert (
        direct["generation"]
        == model_connect["generation"]
        == {
            "max_new_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "use_chat_template": True,
            "enable_thinking": False,
        }
    )
    for request in (direct, model_connect):
        assert request["warmups_per_repetition"] == 5
        assert request["measured_requests_per_repetition"] == 30
        assert request["require_native_token_ids"] is True
        assert request["synchronize_each_request"] is True


def test_three_repetitions_alternate_execution_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "model.trtfb"
    engine = tmp_path / "engine.dir"
    plugin = tmp_path / "plugin.so"
    for path in (bundle, plugin):
        path.write_bytes(b"x")
    engine.mkdir()
    observed: list[str] = []

    def fake_run_one(
        runner: Path,
        request: dict[str, Any],
        stem: str,
        directory: Path,
        environment: dict[str, str],
    ) -> tuple[dict[str, Any], Path, Path]:
        del runner, request, environment
        observed.append(stem)
        request_path = directory / f"{stem}-request.json"
        result_path = directory / f"{stem}-result.json"
        request_path.write_text("{}\n", encoding="utf-8")
        result_path.write_text("{}\n", encoding="utf-8")
        kind = "edgellm-direct" if stem.startswith("direct") else "model-connect"
        return _runner_result(kind), request_path, result_path

    monkeypatch.setattr(performance_harness, "_run_one", fake_run_one)
    direct, model_connect, raw_paths = performance_harness.run_repetitions(
        performance_harness.PerformanceRunners(Path("direct"), Path("mc"), {}, _RUNTIME_IDENTITY),
        bundle=bundle,
        engine_directory=engine,
        plugin=plugin,
        runtime_cache=tmp_path / "cache",
        artifact_directory=tmp_path / "artifacts",
    )
    assert observed == [
        "direct-repetition-1",
        "mc-repetition-1",
        "mc-repetition-2",
        "direct-repetition-2",
        "direct-repetition-3",
        "mc-repetition-3",
    ]
    assert len(direct) == len(model_connect) == performance_harness.REPETITIONS
    assert len(raw_paths) == performance_harness.REPETITIONS * 4


def test_all_performance_gates_pass_at_the_boundary() -> None:
    direct = _results("edgellm-direct", latencies=[100.0] * 30, elapsed_ms=3_000.0)
    model_connect = _results(
        "model-connect",
        latencies=[105.0] * 24 + [110.0] * 6,
        elapsed_ms=3_000.0 / 0.95,
    )
    summary = _evaluate(direct, model_connect)
    assert summary["parity"]["passed"] is True
    assert summary["metrics"]["mc_to_direct_median_ratio"] == pytest.approx(1.05)
    assert summary["metrics"]["mc_to_direct_p95_ratio"] == pytest.approx(1.10)
    assert summary["metrics"]["mc_to_direct_throughput_ratio"] == pytest.approx(0.95)
    assert summary["failures"] == []
    assert summary["passed"] is True


@pytest.mark.parametrize(
    ("direct", "model_connect", "expected_failure"),
    [
        (
            _results("edgellm-direct", latencies=[100.0] * 30, elapsed_ms=3_000.0),
            _results("model-connect", latencies=[106.0] * 30, elapsed_ms=3_000.0),
            "median_latency_ratio",
        ),
        (
            _results("edgellm-direct", latencies=[100.0] * 30, elapsed_ms=3_000.0),
            _results(
                "model-connect",
                latencies=[104.0] * 24 + [120.0] * 6,
                elapsed_ms=3_000.0,
            ),
            "p95_latency_ratio",
        ),
        (
            _results("edgellm-direct", latencies=[100.0] * 30, elapsed_ms=3_000.0),
            _results("model-connect", latencies=[104.0] * 30, elapsed_ms=3_300.0),
            "throughput_ratio",
        ),
        (
            _results("edgellm-direct", elapsed_ms=3_000.0),
            _results("model-connect", elapsed_ms=3_000.0, token_ids=[1, 2, 4]),
            "output_or_token_parity",
        ),
    ],
)
def test_each_gate_fails_closed(
    direct: list[dict[str, Any]],
    model_connect: list[dict[str, Any]],
    expected_failure: str,
) -> None:
    summary = _evaluate(direct, model_connect)
    assert expected_failure in summary["failures"]
    assert summary["passed"] is False


def test_malformed_measurements_fail_before_metrics_are_calculated() -> None:
    result = _runner_result("model-connect")
    result["iterations"].pop()
    with pytest.raises(performance_harness.PerformanceContractError, match="30 measurements"):
        _evaluate(_results("edgellm-direct"), [result] * performance_harness.REPETITIONS)

    result = _runner_result("model-connect")
    result["measured_elapsed_ms"] = float("inf")
    with pytest.raises(performance_harness.PerformanceContractError, match="invalid elapsed"):
        _evaluate(_results("edgellm-direct"), [result] * performance_harness.REPETITIONS)

    result = _runner_result("model-connect")
    result["iterations"][0]["latency_ms"] = float("nan")
    with pytest.raises(performance_harness.PerformanceContractError, match="invalid measured"):
        _evaluate(_results("edgellm-direct"), [result] * performance_harness.REPETITIONS)

    overflow = _results("model-connect", elapsed_ms=sys.float_info.max)
    with pytest.raises(performance_harness.PerformanceContractError, match="aggregates"):
        _evaluate(_results("edgellm-direct"), overflow)


def test_source_identity_accepts_an_exact_archive_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRTMC_TESTED_SOURCE_SHA256", "b" * 64)
    assert performance_harness.source_identity(tmp_path) == {
        "kind": "archive_sha256",
        "value": "b" * 64,
        "dirty": False,
    }


def test_source_identity_rejects_ambiguous_archive_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRTMC_TESTED_SOURCE_SHA256", "B" * 64)
    with pytest.raises(performance_harness.PerformanceContractError, match="64 lowercase"):
        performance_harness.source_identity(tmp_path)
    monkeypatch.delenv("TRTMC_TESTED_SOURCE_SHA256")
    with pytest.raises(performance_harness.PerformanceContractError, match="no .git metadata"):
        performance_harness.source_identity(tmp_path)


def test_source_identity_records_git_revision_and_dirty_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.delenv("TRTMC_TESTED_SOURCE_SHA256", raising=False)
    responses = iter((SimpleNamespace(stdout="c" * 40 + "\n"), SimpleNamespace(stdout=" M file\n")))
    monkeypatch.setattr(performance_harness, "_run", lambda *args, **kwargs: next(responses))
    identity = performance_harness.source_identity(tmp_path)
    assert identity == {
        "kind": "git_revision",
        "value": "c" * 40,
        "dirty": True,
    }
    summary = _evaluate(
        _results("edgellm-direct"),
        _results("model-connect"),
        source_identity=identity,
    )
    assert summary["failures"] == ["source_checkout_dirty"]
    assert summary["passed"] is False


def test_artifacts_are_required_outside_the_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.delenv("TRTMC_PERF_ARTIFACT_DIR", raising=False)
    with pytest.raises(performance_harness.PerformanceContractError, match="is required"):
        performance_harness.artifact_root(source)

    monkeypatch.setenv("TRTMC_PERF_ARTIFACT_DIR", str(source / "evidence"))
    with pytest.raises(performance_harness.PerformanceContractError, match="outside"):
        performance_harness.artifact_root(source)

    external = tmp_path / "artifacts"
    monkeypatch.setenv("TRTMC_PERF_ARTIFACT_DIR", str(external))
    assert performance_harness.artifact_root(source) == external
    assert external.is_dir()


def test_runner_sources_keep_the_two_execution_paths_explicit() -> None:
    model_connect = (_ADAPTER_ROOT / "mc_performance_runner.cpp").read_text(encoding="utf-8")
    direct = (_ADAPTER_ROOT / "direct_performance_runner.cpp").read_text(encoding="utf-8")
    header = (_ADAPTER_ROOT / "performance_runner.h").read_text(encoding="utf-8")
    cmake = (_ADAPTER_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "cmake_minimum_required(VERSION 3.20)" in cmake
    assert "trtmc::load" in model_connect
    assert "pipeline->generate" in model_connect
    assert "LLMInferenceRuntime" in direct
    assert "captureDecodingCUDAGraph" in direct
    assert "handleRequest" in direct
    assert "cudaDeviceSynchronize" in model_connect
    assert "cudaDeviceSynchronize" in direct
    assert "kWarmups = 5" in header
    assert "kMeasuredRequests = 30" in header
    assert cmake.count("add_executable(") == 2
    assert "performance_device_link_stub.cu" in cmake
    assert "must be supplied from the model-owned dependency.lock" in cmake
    for variable in (
        "TRTMC_EDGE_LLM_SOURCE",
        "TRTMC_EDGE_LLM_VERSION",
        "TRTMC_EDGE_LLM_COMMIT",
        "TRTMC_TENSORRT_VERSION",
        "TRTMC_CUDA_VERSION",
        "TRTMC_CUDA_RUNTIME_VERSION",
    ):
        assert variable in cmake
    for source in (model_connect, direct):
        assert "TRTMC_EXPECTED_TENSORRT_VERSION" in source
        assert "TRTMC_EXPECTED_CUDA_RUNTIME_VERSION" in source
        assert performance_harness.TENSORRT_VERSION not in source
        assert str(performance_harness.CUDA_RUNTIME_VERSION) not in source
    assert performance_harness.EDGE_COMMIT not in cmake
    assert performance_harness.EDGE_VERSION not in cmake
    for warning in ("-Wall", "-Wextra", "-Wpedantic", "-Werror"):
        assert f"$<$<COMPILE_LANGUAGE:CXX>:{warning}>" in cmake
    assert "target_compile_options(${_target} PRIVATE -Wall" not in cmake


def test_cpp_measurement_loop_executes_without_a_gpu(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    json_include = next(
        (
            candidate
            for candidate in sorted(_REPO_ROOT.glob("build*/_deps/nlohmann_json-src/include"))
            if (candidate / "nlohmann/json.hpp").is_file()
        ),
        None,
    )
    if compiler is None or json_include is None:
        pytest.skip("a C++ compiler and configured nlohmann-json dependency are required")

    source = tmp_path / "measurement_contract.cpp"
    source.write_text(
        inspect.cleandoc(
            r"""
            #include "performance_runner.h"
            #include <iostream>

            int main() {
                int generated = 0;
                int synchronized = 0;
                auto generate = [&]() {
                    ++generated;
                    return qwen_edge_performance::Sample{"ok", {1}};
                };
                auto synchronize = [&]() { ++synchronized; };
                const auto result = qwen_edge_performance::measure(generate, synchronize);
                std::cout << generated << ' ' << synchronized << ' '
                          << result.iterations.size() << '\n';
            }
            """
        )
        + "\n",
        encoding="utf-8",
    )
    binary = tmp_path / "measurement_contract"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Werror",
            f"-I{_ADAPTER_ROOT}",
            f"-I{json_include}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.stdout == "35 66 30\n"
