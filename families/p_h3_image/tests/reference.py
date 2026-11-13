# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream full-image oracle, isolated to release CUDA state on exit."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import ModularPipeline


def main():
    directory, input_path, tmp_path = map(Path, sys.argv[1:])
    manifest, case = json.loads(input_path.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    height, width = manifest["image_height"], manifest["image_width"]
    assert not case["prompt_upsampler"]
    blocks = ModularPipeline.from_pretrained(str(directory), local_files_only=True).blocks
    # The documented upstream mechanism for literal prompt rendering.
    blocks.sub_blocks.pop("prompt_upsampler")
    reference = blocks.init_pipeline(str(directory))
    reference.load_components(
        names=[
            "text_encoder",
            "text_connector",
            "tokenizer",
            "processor",
            "vae",
            "transformer",
            "scheduler",
            "audio_scheduler",
        ],
        pretrained_model_name_or_path=str(directory),
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    reference.text_encoder.set_attn_implementation("eager")
    reference.to("cuda")
    assert reference.vae.use_tiling
    assert reference.vae.tile_sample_min_height == reference.vae.tile_sample_min_width == 256
    assert (
        reference.vae.tile_sample_min_overlap_height
        == reference.vae.tile_sample_min_overlap_width
        == 64
    )
    assert len(reference.config.sample_sigmas) - 1 == case["num_inference_steps"]
    requests, expected = [], []
    for index, prompt in enumerate(case["prompts"]):
        seed = case["seed"] + index
        initial = torch.randn(
            (1, 24, 1, height // 16, width // 16),
            generator=torch.Generator().manual_seed(seed + 1000),
        )
        # Providing image latents leaves only the source's 4x32 audio draw on
        # this request generator. Replay those exact two public input streams.
        audio = torch.randn((4, 32), generator=torch.Generator().manual_seed(seed))
        noise = tmp_path / f"noise-{index}.f32"
        np.concatenate([initial.numpy().reshape(-1), audio.numpy().reshape(-1)]).astype(
            "<f4"
        ).tofile(noise)
        with torch.no_grad():
            images = reference(
                prompt=prompt,
                height=height,
                width=width,
                latents=initial,
                generator=torch.Generator().manual_seed(seed),
                output_type="np",
                output="images",
            )
        pixels = np.asarray(images[0], dtype=np.float32)
        assert pixels.shape == (height, width, 3)
        assert np.isfinite(pixels).all()
        assert pixels.std() > 0.05
        expected.append(pixels)
        pixels.tofile(tmp_path / f"reference-{index}.f32")
        requests.append(
            dict(prompt=prompt, noise=str(noise), output=str(tmp_path / f"native-{index}.f32"))
        )
        print(f"Official P-H3-Image completed prompt {index}", flush=True)
    (tmp_path / "reference-requests.json").write_text(json.dumps(requests))


if __name__ == "__main__":
    main()
