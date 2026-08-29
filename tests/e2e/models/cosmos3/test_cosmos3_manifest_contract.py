# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for the Cosmos3-Nano 720p proof."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tests.e2e.models.cosmos3.e2e_plugins.runners import diffusion
from tests.e2e_harness.manifest_loader import load_manifest


MODEL_ID = "nvidia/Cosmos3-Nano"
MODEL_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
MANIFEST_FILENAMES = ("cosmos3-nano-l0.json", "cosmos3-nano-l0-cp2.json")
FIXED_PROFILE = {
    "video_num_frames": 189,
    "video_height": 720,
    "video_width": 1280,
    "num_inference_steps": 35,
    "guidance_scale": 6.0,
}
FIXED_OUTPUT_THRESHOLDS = {
    "exact_num_frames": 189,
    "exact_video_height": 720,
    "exact_video_width": 1280,
}


@pytest.mark.parametrize(
    "filename",
    MANIFEST_FILENAMES,
)
def test_cosmos3_manifests_declare_public_pinned_checkpoint(filename: str) -> None:
    manifest_path = Path(__file__).with_name("manifests") / filename
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)

    assert raw["hf_id"] == MODEL_ID
    assert raw["hf_revision"] == MODEL_REVISION
    assert "gated" not in raw
    assert all(item.kind != "hf_auth_token_present" for item in case.preflight)


@pytest.mark.parametrize("filename", MANIFEST_FILENAMES)
def test_cosmos3_manifests_share_fixed_720p_profile(filename: str) -> None:
    model_dir = Path(__file__).parent
    case = load_manifest(model_dir / "manifests" / filename)
    thresholds = json.loads(
        (model_dir / "thresholds" / filename).read_text(encoding="utf-8")
    )["threshold_overrides"]

    assert {name: case.inputs[name] for name in FIXED_PROFILE} == FIXED_PROFILE
    assert {
        name: thresholds[name] for name in FIXED_OUTPUT_THRESHOLDS
    } == FIXED_OUTPUT_THRESHOLDS


def test_cosmos3_cp1_manifest_remains_single_device() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "cosmos3-nano-l0.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "build_args" not in raw
    assert "distributed_runtime" not in raw


def test_cosmos3_cp2_manifest_declares_context_parallel_runtime() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "cosmos3-nano-l0-cp2.json"
    case = load_manifest(manifest_path)
    parallel = case.metadata["build_args"]["parallel"]
    distributed = case.metadata["distributed_runtime"]

    assert parallel == {"mode": "context_parallel", "cp_size": 2}
    assert distributed["enabled"] is True
    assert distributed["launcher"] == "mpirun"
    assert distributed["world_size"] == parallel["cp_size"]


def test_cosmos3_native_attention_allows_decomposition_fallback() -> None:
    source_path = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/cosmos3/trt_ops.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "decomposable"
            for target in node.targets
        )
    ]

    assert len(values) == 2
    assert all(isinstance(value, ast.Constant) and value.value is True for value in values)


def test_cosmos3_context_parallel_attention_restores_bf16_after_routing() -> None:
    source_path = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/cosmos3/trt_ops.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    seq_to_heads = functions["ulysses_seq_to_heads"]
    dual_attention = functions["ulysses_dual_attention"]
    assert "trt.float16" in seq_to_heads
    assert "cast(network, exchanged, trt.bfloat16)" in seq_to_heads
    assert "trt.float16" not in dual_attention
    assert "trt.bfloat16" in dual_attention


def _runner_case(*, distributed: bool) -> SimpleNamespace:
    metadata = {"runtime_timeout_s": 1}
    if distributed:
        metadata["distributed_runtime"] = {
            "enabled": True,
            "launcher": "mpirun",
            "world_size": 2,
            "export_env": ["TRTMC_NCCL_RENDEZVOUS"],
        }
    return SimpleNamespace(
        name="cosmos3-contract",
        task_strategy="diffusion_media_generation",
        bundle="cosmos3.trtfb",
        metadata=metadata,
        inputs={
            "prompt": "test",
            "num_inference_steps": 1,
            "guidance_scale": 1.0,
            "seed": 7,
            "video_height": 720,
            "video_width": 1280,
        },
    )


def _runner_context(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifacts_dir=str(tmp_path),
        binary_path="/opt/trtmc/bin/trtmc",
        engine_dir="/models",
        ld_library_path="/opt/trtmc/lib",
        model_plugin_dir="/opt/trtmc/lib/models",
    )


def test_cosmos3_distributed_repro_preserves_rendezvous_and_unique_output(
    tmp_path: Path,
) -> None:
    case = _runner_case(distributed=True)
    ctx = _runner_context(tmp_path)

    first = diffusion.plugin.build_trt_inference_command(case, ctx, "/models/cosmos3.trtfb")
    second = diffusion.plugin.build_trt_inference_command(case, ctx, "/models/cosmos3.trtfb")

    assert first is not None and second is not None
    exported = first[first.index("-x") + 1]
    assert exported.startswith("TRTMC_NCCL_RENDEZVOUS=")
    assert first[first.index("--output") + 1] != second[second.index("--output") + 1]


def test_cosmos3_timeout_returns_failure_and_persists_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _runner_case(distributed=False)
    ctx = _runner_context(tmp_path)

    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(
            command,
            1,
            output="partial output",
            stderr="timeout diagnostic",
        )

    monkeypatch.setattr(diffusion.subprocess, "run", timeout)
    result = diffusion.plugin.run_stage(case, SimpleNamespace(name="end_to_end"), ctx)

    assert result.data["returncode"] == 124
    assert result.text == "partial output"
    assert result.metadata["error"] == "timed out after 1 seconds"
    stderr_log = Path(result.metadata["stderr_log"])
    assert stderr_log.read_text(encoding="utf-8") == "timeout diagnostic"
