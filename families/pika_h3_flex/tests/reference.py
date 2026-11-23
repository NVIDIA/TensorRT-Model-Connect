# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pinned official six-step Pika pipeline, in its own CUDA process."""

import gc
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import transformers
import diffusers
from diffusers import ModularPipeline
from diffusers.hooks.group_offloading import apply_group_offloading


@contextmanager
def _reference_pipeline(directory, output_dir):
    # Load and stage one component at a time so the reference does not retain
    # the complete text encoder, transformer and decoders in resident CPU RAM.
    with tempfile.TemporaryDirectory(prefix="reference-offload-", dir=output_dir) as scratch:
        pipeline = ModularPipeline.from_pretrained(
            str(directory), workflow="t2va", local_files_only=True
        )
        options = {"pretrained_model_name_or_path": str(directory), "local_files_only": True}
        for name in ("transformer", "text_encoder", "vae", "audio_vae"):
            dtype = torch.float32 if name in {"vae", "audio_vae"} else torch.bfloat16
            pipeline.load_components(names=name, dtype=dtype, **options)
            component = getattr(pipeline, name)
            if name == "transformer":
                pipeline.load_lora_weights(
                    str(directory), weight_name="fl2va/adapter.safetensors", local_files_only=True
                )
            if name == "text_encoder":
                component.set_attn_implementation("eager")
            if name == "audio_vae":
                # weight_norm pre-hooks require this complete FP32 component
                # on CUDA; the audio VAE opts out of group offloading.
                component.to("cuda")
            else:
                apply_group_offloading(
                    component,
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="leaf_level",
                    use_stream=False,
                    offload_to_disk_path=str(Path(scratch) / name),
                )
            gc.collect()
        # Tokenizers, processors and schedulers do not retain neural weights.
        pipeline.load_components(
            dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16},
            **options,
        )
        yield pipeline


def _generate(pipeline, manifest, case, output_dir):
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
        del result, video, audio
        gc.collect()


def main():
    directory, input_path, output_dir = map(Path, sys.argv[1:])
    manifest, case = json.loads(input_path.read_text())
    assert transformers.__version__ == "5.15.0"
    assert torch.__version__.split("+")[0] == "2.12.0"
    (output_dir / "reference-environment.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "diffusers": diffusers.__version__,
                "weight_offload": "leaf_level_temporary_disk",
            }
        )
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with _reference_pipeline(directory, output_dir) as pipeline:
        _generate(pipeline, manifest, case, output_dir)


if __name__ == "__main__":
    main()
