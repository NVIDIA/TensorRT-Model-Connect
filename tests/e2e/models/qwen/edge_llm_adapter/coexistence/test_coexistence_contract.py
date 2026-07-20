# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-GPU contract checks for Qwen EdgeLLM same-process qualification."""

from __future__ import annotations

import hashlib
import itertools
import json
import runpy
import shutil
import struct
import subprocess
from pathlib import Path

import pytest


_LEAF = Path(__file__).resolve().parent
_REPOSITORY = Path(__file__).resolve().parents[6]
_RUNNER = _LEAF / "coexistence_runner.cpp"


def test_runner_compiles_and_executes_with_four_live_public_pipelines(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    fake_core = tmp_path / "fake_core.cpp"
    fake_core.write_text(
        r"""
#include <trtmc/pipeline.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

int live_pipelines = 0;

class FakePipeline final : public IPipeline {
  public:
    explicit FakePipeline(std::string bundle)
        : model_(std::filesystem::path(std::move(bundle)).stem().string()) {
        ++live_pipelines;
    }

    ~FakePipeline() override { --live_pipelines; }

    TextResult generate(const std::string& prompt, const GenerateConfig& config) override {
        if (live_pipelines != 4)
            throw std::runtime_error("generation ran before all four pipelines were live");
        return TextResult{model_ + ":" + prompt,
                          {static_cast<int32_t>(model_.size()), config.max_new_tokens}};
    }

    const char* model_id() const override { return model_.c_str(); }
    const char* pipeline_type() const override { return "fake-text"; }

  private:
    std::string model_;
};

} // namespace

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const LoadOptions&) {
    return std::make_unique<FakePipeline>(bundle_path);
}

} // namespace trtmc
""".lstrip(),
        encoding="utf-8",
    )
    binary = tmp_path / "coexistence-runner"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-pthread",
            f"-I{_REPOSITORY / 'include'}",
            str(_RUNNER),
            str(fake_core),
            "-o",
            str(binary),
        ],
        check=True,
    )
    bundles = []
    for name in (
        "qwen3-0.6b.trtfb",
        "qwen3-1.7b.trtfb",
        "qwen3-4b.trtfb",
        "qwen3-next.trtfb",
    ):
        bundle = tmp_path / name
        bundle.write_bytes(b"test")
        bundles.append(bundle.resolve())
    output = tmp_path / "proof.json"
    result = subprocess.run(
        [
            str(binary),
            "--runtime-cache",
            str(tmp_path / "cache"),
            "--output",
            str(output),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "7",
            *(str(bundle) for bundle in bundles),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "coexistence-ok"
    proof = json.loads(output.read_text(encoding="utf-8"))
    assert proof["load_order"] == [str(bundle) for bundle in bundles]
    assert [row["bundle"] for row in proof["forward"]] == proof["load_order"]
    assert [row["bundle"] for row in proof["reverse"]] == list(reversed(proof["load_order"]))
    assert [row["bundle"] for row in proof["concurrent"]] == proof["load_order"]
    assert {row["pipeline_type"] for row in proof["forward"]} == {"fake-text"}
    assert {tuple(row["token_ids"])[1] for row in proof["forward"]} == {7}

    duplicate = subprocess.run(
        [
            str(binary),
            "--runtime-cache",
            str(tmp_path / "duplicate-cache"),
            "--output",
            str(tmp_path / "duplicate.json"),
            "--prompt",
            "hello",
            str(bundles[0]),
            str(bundles[0]),
            str(bundles[2]),
            str(bundles[3]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode != 0
    assert "bundle paths must be distinct" in duplicate.stderr

    too_few = subprocess.run(
        [
            str(binary),
            "--runtime-cache",
            str(tmp_path / "single-cache"),
            "--output",
            str(tmp_path / "single.json"),
            "--prompt",
            "hello",
            str(bundles[0]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert too_few.returncode != 0
    assert "at least two bundle paths are required" in too_few.stderr


def _write_bundle(path: Path, profile: object, *, implementation_id: str) -> None:
    runtime_library = profile.runtime_library
    descriptor = {
        "schema_version": 2,
        "implementation_id": implementation_id,
        "model_id": profile.model_id,
        "profile_id": "a100-pcie80-sm80-fp16",
        "runtime_library": runtime_library,
        "factory_abi": 1,
        "implementation_metadata_section": "implementation.json",
        "runtime": {
            "name": "tensorrt-edge-llm",
            "version": "0.9.0",
            "commit": "1ac0f2b99642045125e1c5ac7b109434ba3b36c7",
        },
        "artifact": {
            "section_prefix": "optimized_runtime_artifacts",
            "directories": ["engine.dir"],
            "file_count": 2,
            "total_size": 10,
            "tree_sha256": "1" * 64,
        },
    }
    payloads = (
        (
            "optimized_runtime.json",
            json.dumps(descriptor, separators=(",", ":")).encode("utf-8"),
        ),
        (f"optimized_runtime_artifacts/{runtime_library}", b"runtime"),
        ("optimized_runtime_artifacts/engine.dir/llm.engine", b"engine"),
    )
    offset = 0
    sections = {}
    for name, payload in payloads:
        sections[name] = {
            "offset": offset,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        offset += len(payload)
    header = json.dumps(
        {"model_id": profile.model_id, "sections": sections},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(
        b"TRTFB\x00\x01\x00"
        + struct.pack("<Q", len(header))
        + header
        + b"".join(payload for _name, payload in payloads)
    )


def test_a100_gate_requires_exact_delegated_descriptors_and_bounded_orders(
    tmp_path: Path,
) -> None:
    scope = runpy.run_path(str(_LEAF / "test_a100_coexistence.py"))
    profiles = scope["_PROFILES"]
    validate = scope["_validate_delegated_bundle"]
    representative_load_orders = scope["_representative_load_orders"]
    if len(profiles) < 2:
        with pytest.raises(ValueError, match="at least two profiles"):
            representative_load_orders(profiles)
    else:
        orders = representative_load_orders(profiles)
        assert len(orders) <= 2 * len(profiles)
        assert all(set(order) == set(profiles) for order in orders)
        if len(profiles) == 3:
            assert len(orders) == 6

    for profile in profiles:
        bundle = tmp_path / f"{profile.leaf}.trtfb"
        _write_bundle(bundle, profile, implementation_id=profile.implementation_id)
        validate(bundle, profile)

        _write_bundle(bundle, profile, implementation_id="wrong-implementation")
        with pytest.raises(AssertionError):
            validate(bundle, profile)


def _write_synthetic_profile(root: Path, index: int) -> None:
    leaf = f"qwen3_{index}b_fp16_a100_pcie80_sm80"
    builder = root / "python/tensorrt_model_connect/families/qwen/edge_llm_adapter" / leaf
    tests = root / "tests/e2e/models/qwen/edge_llm_adapter" / leaf
    builder.mkdir(parents=True)
    tests.mkdir(parents=True)
    (builder / "IMPLEMENTATION.toml").write_text(
        f"""schema_version = 1
implementation_id = "qwen3-{index}b-edge"
downstream_runtime = "tensorrt-edge-llm"
downstream_version = "0.9.0"
downstream_commit = "1ac0f2b99642045125e1c5ac7b109434ba3b36c7"

[model]
id = "Qwen/Qwen3-{index}B"

[runtime]
library = "libqwen3_{index}b_edge.so"
abi = 1
""",
        encoding="utf-8",
    )
    (tests / "test_a100_e2e.py").write_text(
        f"""_PUBLIC_EDGE_BUILD_ENVIRONMENT = {{
    "TRTMC_EDGE_LLM_SOURCE_DIR": "_TRTMC_INTERNAL_QWEN3_{index}B_SOURCE_DIR",
    "TRTMC_EDGE_LLM_BUILD_DIR": "_TRTMC_INTERNAL_QWEN3_{index}B_BUILD_DIR",
}}
""",
        encoding="utf-8",
    )
    (tests / "build_runners.py").write_text("", encoding="utf-8")


def test_fourth_profile_is_discovered_without_family_gate_changes(tmp_path: Path) -> None:
    for index in range(1, 5):
        _write_synthetic_profile(tmp_path, index)

    gate = runpy.run_path(str(_LEAF / "test_a100_coexistence.py"))
    builder_root = tmp_path / "python/tensorrt_model_connect/families/qwen/edge_llm_adapter"
    test_root = tmp_path / "tests/e2e/models/qwen/edge_llm_adapter"
    profiles = gate["_discover_profiles"](builder_root, test_root)
    orders = gate["_representative_load_orders"](profiles)
    three_orders = gate["_representative_load_orders"](profiles[:3])

    assert len(profiles) == 4
    assert {profile.model_id for profile in profiles} == {
        "Qwen/Qwen3-1B",
        "Qwen/Qwen3-2B",
        "Qwen/Qwen3-3B",
        "Qwen/Qwen3-4B",
    }
    assert len(orders) == 8
    assert len(set(orders)) == 8
    assert all(set(order) == set(profiles) for order in orders)
    assert set(three_orders) == set(itertools.permutations(profiles[:3]))

    qualifier = runpy.run_path(str(_LEAF.parent / "qualify_a100.py"))
    qualified = qualifier["_discover_profiles"](tmp_path)
    assert len(qualified) == 4


def test_runner_is_test_only_and_uses_no_private_or_edgellm_api() -> None:
    source = _RUNNER.read_text(encoding="utf-8")
    cmake = (_LEAF / "CMakeLists.txt").read_text(encoding="utf-8")
    gate = (_LEAF / "test_a100_coexistence.py").read_text(encoding="utf-8")
    assert "#include <trtmc/pipeline.h>" in source
    assert "trtmc::load(bundle.string(), load_options)" in source
    assert "std::vector<LoadedPipeline> loaded" in source
    assert "for (auto iterator = loaded.rbegin()" in source
    assert "std::async(std::launch::async" in source
    assert "require_deterministic(forward, concurrent)" in source
    for forbidden in (
        "optimized_runtime_factory.h",
        "optimized_runtime_host.h",
        "tensorrt_llm",
        "edgellm::",
        "dlopen(",
        "dlsym(",
    ):
        assert forbidden not in source
    assert "TRTMC_MC_INCLUDE_DIR" in cmake
    assert "TRTMC_MC_CORE_LIBRARY" in cmake
    assert "_PROFILES = _discover_profiles()" in gate
    assert "_representative_load_orders(_PROFILES)" in gate
    assert "TRTMC_QWEN_EDGELLM_BUNDLES_JSON" in gate
    assert "itertools.permutations" not in gate
    assert "TRTMC_QWEN3_06B_EDGELLM_BUNDLE" not in gate
