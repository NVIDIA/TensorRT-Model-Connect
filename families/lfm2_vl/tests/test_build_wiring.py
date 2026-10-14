# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise owner CLI/request wiring without creating an engine or using CUDA."""

import json
from pathlib import Path

import pytest

pytest.importorskip("tensorrt")

import numpy as np
from families.lfm2_vl import cli, model, vision_builder
from tensorrt_model_connect.build import BuildRequest as LegacyBuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect import model_support


@pytest.mark.parametrize(
    "precision,capacity,verbose", [("bf16", 32768, False), ("fp16", 256, True)]
)
def test_owned_cli_preserves_builder_arguments_and_bundle_sections(
    monkeypatch, tmp_path, precision, capacity, verbose
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint.joinpath("config.json").write_bytes(
        (Path(__file__).parent / "fixtures/config.json").read_bytes()
    )
    checkpoint.joinpath("generation_config.json").write_text(
        json.dumps({"bos_token_id": 1, "eos_token_id": 7, "pad_token_id": 0})
    )
    resolved = []

    def resolve(name, revision):
        resolved.append((name, revision))
        return checkpoint

    monkeypatch.setattr(model_support, "resolve_model", resolve)
    monkeypatch.setattr(model, "load_lfm2_vl_weights", lambda *args, **kwargs: {})
    decoder_calls, vision_calls = [], []

    def decoder(config, weights, max_cache_length, **options):
        decoder_calls.append((config.raw, weights, max_cache_length, options))
        return b"decoder placeholder"

    def vision(model_dir, config, dtype):
        vision_calls.append((model_dir, config, dtype))
        return b"vision placeholder", np.zeros(1, dtype=np.float32)

    monkeypatch.setattr(model, "build_lfm2_vl_engine", decoder)
    monkeypatch.setattr(vision_builder, "build_vision_engine", vision)
    owned_path, legacy_path = tmp_path / "owned.bundle", tmp_path / "legacy.bundle"
    assert (
        cli.build(
            model="LiquidAI/LFM2.5-VL-1.6B",
            revision="919fde3d022e3f90a4716006f993938ee8c2eb97",
            output=owned_path,
            precision=precision,
            max_sequence_length=capacity,
            verbose=verbose,
        )
        == 0
    )
    writer = BundleWriter(legacy_path)
    model.build(
        LegacyBuildRequest(
            model_dir=checkpoint,
            output_path=legacy_path,
            family="lfm2_vl",
            task="images_text_to_text",
            precision=precision,
            max_sequence_length=capacity,
            verbose=verbose,
        ),
        writer,
    )
    writer.finish()
    assert resolved == [("LiquidAI/LFM2.5-VL-1.6B", "919fde3d022e3f90a4716006f993938ee8c2eb97")]
    assert decoder_calls[0] == decoder_calls[1]
    assert decoder_calls[0][2:] == (
        capacity,
        {"precision": precision, "verbose": verbose, "debug_layer_outputs": False},
    )
    assert vision_calls[0] == vision_calls[1]
    assert vision_calls[0][0] == checkpoint and vision_calls[0][2] == precision
    assert owned_path.read_bytes() == legacy_path.read_bytes()


@pytest.mark.parametrize("capacity", [0, -1, True])
def test_owned_request_rejects_invalid_capacity(tmp_path, capacity):
    with pytest.raises(ValueError, match="positive integer"):
        model.BuildRequest(tmp_path, max_sequence_length=capacity)


def test_backend_label_cannot_disagree_with_the_tensor_rt_builder(tmp_path):
    with pytest.raises(ValueError, match="only the trt backend"):
        model.BuildRequest(tmp_path, backend="trt_rtx")
    request = LegacyBuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unsupported.bundle",
        family="lfm2_vl",
        task="images_text_to_text",
        precision="bf16",
        backend="trt_rtx",
    )
    # Reject before reading a checkpoint, invoking an engine builder, or writing
    # any bundle metadata, including for compatibility callers.
    with pytest.raises(ValueError, match="only the trt backend"):
        model.build(request, None)
