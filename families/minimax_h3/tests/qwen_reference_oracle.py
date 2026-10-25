# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native Qwen reference inputs with pinned upstream processors.

This is a host-input boundary oracle, not a model, resize, or video test.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLVisionRotaryEmbedding,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


class FixtureTokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False)
        return dict(input_ids=[ord(value) for value in text])

    def convert_tokens_to_ids(self, token):
        return {"<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|image_pad|>": 151655}[
            token
        ]


class PositionOracle:
    get_rope_index = Qwen3VLModel.get_rope_index
    get_vision_position_ids = Qwen3VLModel.get_vision_position_ids
    config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert importlib.metadata.version("transformers") == "5.15.0"
    assert importlib.metadata.version("diffusers") == "0.40.0"

    def native(field, dtype, shape):
        raw = subprocess.check_output([str(args.executable), "--dump-fixture", field])
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    comparisons = []

    def compare(field, expected, dtype, *, exact=True):
        value = np.asarray(expected)
        actual = native(field, dtype, value.shape)
        error = np.abs(actual.astype(np.float64) - value.astype(np.float64))
        comparisons.append(
            dict(
                field=field, shape=list(value.shape), exact=exact, max_abs_error=float(error.max())
            )
        )
        if exact:
            np.testing.assert_array_equal(actual, value)
        else:
            np.testing.assert_allclose(actual, value, rtol=1.3e-6, atol=1e-5, equal_nan=False)

    grid = torch.tensor([[1, 2, 4]])
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, 48, mode="bilinear", align_corners=True, spatial_merge_size=2
    )
    compare("indices", indices.numpy(), np.int32)
    compare("weights", weights.numpy(), np.float32, exact=False)
    positions = get_vision_position_ids(grid, 2)
    frequencies = Qwen3VLVisionRotaryEmbedding(36)(positions)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    compare("cos", frequencies.cos().numpy(), np.float32, exact=False)
    compare("sin", frequencies.sin().numpy(), np.float32, exact=False)
    rgb = (np.arange(32 * 64 * 3) % 256).astype(np.uint8).reshape(32, 64, 3)
    processor = Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        size=dict(shortest_edge=65536, longest_edge=16777216),
    )
    pixels = processor(images=[rgb], do_resize=False, return_tensors="pt")["pixel_values"]
    compare("pixels", pixels.numpy(), np.float32)
    token_ids, tags = MiniMaxH3Ref2VATextEncoderStep._build_presentation(
        FixtureTokenizer(),
        "walk",
        [SimpleNamespace(kind="image", has_audio=False)] * 2,
        image_token_counts=[4, 3],
        video_block_token_counts=[],
        video_block_timestamps=[],
    )
    compare("ids", token_ids, np.int32)
    compare("tags", tags, np.int32)
    mask = np.asarray(token_ids) == 151655
    compare("mask", mask.astype(np.int32), np.int32)
    compare("visual_rows", np.flatnonzero(mask).astype(np.int32), np.int32)
    input_ids = torch.tensor([token_ids])
    types = torch.from_numpy(mask.astype(np.int64))[None]
    positions, delta = PositionOracle().get_rope_index(
        input_ids, types, image_grid_thw=torch.tensor([[1, 4, 4], [1, 2, 6]])
    )
    assert int(delta.item()) == -2
    compare("positions", positions[:, 0].numpy(), np.int32)
    args.output.mkdir(mode=0o700)
    with (args.output / "report.json").open("x") as file:
        json.dump(
            dict(
                comparisons=comparisons,
                host_inputs_qualified=True,
                resize_qualified=False,
                tokenizer_algorithm_qualified=False,
                native_Ref2VA_video_qualified=False,
                full_app_qualified=False,
            ),
            file,
            indent=2,
            allow_nan=False,
        )
    print(json.dumps(comparisons, allow_nan=False))


if __name__ == "__main__":
    main()
