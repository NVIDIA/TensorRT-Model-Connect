# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pinned official six-step Pika pipeline, in its own CUDA process."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import ModularPipeline


def main():
    directory, input_path, output_dir = map(Path, sys.argv[1:])
    manifest, case = json.loads(input_path.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipeline = ModularPipeline.from_pretrained(
        str(directory), workflow="t2va", local_files_only=True
    )
    pipeline.load_components(
        pretrained_model_name_or_path=str(directory),
        local_files_only=True,
        dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16},
    )
    pipeline.load_lora_weights(
        str(directory), weight_name="fl2va/adapter.safetensors", local_files_only=True
    )
    pipeline.text_encoder.set_attn_implementation("eager")
    pipeline.to("cuda")
    assert pipeline.vae.use_tiling
    assert pipeline.vae.tile_sample_min_height == pipeline.vae.tile_sample_min_width == 256
    assert pipeline.config.denoising_step_list == [1000, 750, 500, 250, 100, 40, 0]
    for index, prompt in enumerate(case["prompts"]):
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                height=manifest["image_height"],
                width=manifest["image_width"],
                num_frames=manifest["video_num_frames"],
                generator=torch.Generator("cpu").manual_seed(case["seed"] + index),
                output_type="np",
                output=["videos", "audio", "sampling_rate"],
            )
        video = np.asarray(result["videos"][0], dtype=np.float32)
        audio = result["audio"][0].detach().float().cpu().numpy().T.copy()
        assert video.shape == (124, 480, 832, 3)
        assert audio.shape == (165333, 2) and result["sampling_rate"] == 32000
        assert np.isfinite(video).all() and np.isfinite(audio).all()
        assert video.std() > 0.05 and audio.std() > 1e-6
        video.astype("<f4").tofile(output_dir / f"reference-video-{index}.f32")
        audio.astype("<f4").tofile(output_dir / f"reference-audio-{index}.f32")
        print(f"Official Pika H3 Flex completed prompt {index}", flush=True)


if __name__ == "__main__":
    main()
