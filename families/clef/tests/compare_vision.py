# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the complete native vision encoder on original processor tensors."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from families.clef.model import Checkpoint
from families.clef.tests.compare_head import dump_inputs
from families.clef.vision import build_vision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoProcessor
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from transformers.vision_utils import (
        get_vision_position_ids,
        get_vision_bilinear_indices_and_weights,
    )

    config = json.loads((args.checkpoint / "config.json").read_text())["vision_config"]
    weights = Checkpoint(args.checkpoint).select("model.visual.")
    plan = args.output / "vision.plan"
    plan.write_bytes(build_vision(weights, config, max_patches=2048))
    hf_config = Qwen3_5VisionConfig(**config)
    hf_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3_5VisionModel(hf_config)
    model.load_state_dict(
        {name.removeprefix("model.visual."): weight for name, weight in weights.items()},
        assign=True,
    )
    # Rotary frequency is a nonpersistent buffer initialized on meta above.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding

    model.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(
        config["hidden_size"] // config["num_heads"] // 2
    )
    model = model.cuda().eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    receipts = []
    with torch.inference_mode():
        for index, (height, width) in enumerate(((256, 256), (181, 347))):
            rng = np.random.default_rng(2026 + index)
            image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
            batch = processor.image_processor(images=[image], return_tensors="pt").to("cuda")
            grid = batch["image_grid_thw"]
            indices, coefficients = get_vision_bilinear_indices_and_weights(grid, 48, 2)
            positions = (model.pos_embed(indices) * coefficients[:, :, None]).sum(0)
            rotary = model.rotary_pos_emb(get_vision_position_ids(grid, 2))
            rotary = torch.cat([rotary, rotary], -1)
            expected = model(batch["pixel_values"], grid).pooler_output
            root = args.output / str(index)
            dump_inputs(
                root,
                {
                    "patches": batch["pixel_values"],
                    "positions": positions,
                    "rope_cos": rotary.cos()[:, None],
                    "rope_sin": rotary.sin()[:, None],
                    "frame_ids": torch.zeros(batch["pixel_values"].shape[0], dtype=torch.int32),
                },
            )
            subprocess.run([str(args.probe), str(plan), str(root)], check=True)
            actual = (
                torch.from_numpy(np.fromfile(root / "visual_embeddings.out.bin", np.int16))
                .view(torch.bfloat16)
                .reshape(expected.shape)
                .cuda()
            )
            difference = (actual.float() - expected.float()).abs()
            print(
                height,
                width,
                "max",
                difference.max().item(),
                "mean",
                difference.mean().item(),
                flush=True,
            )
            torch.testing.assert_close(actual.float(), expected.float(), atol=0.125, rtol=0.02)
            receipts.append(
                {
                    "height": height,
                    "width": width,
                    "max_abs_error": difference.max().item(),
                    "mean_abs_error": difference.mean().item(),
                }
            )
    (args.output / "comparison.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
