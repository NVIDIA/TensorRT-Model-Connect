# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build Flash-Next native TensorRT plans and self-contained request assets."""

import json

from transformers import PreTrainedTokenizerFast

from .checkpoint import FlashCheckpoint
from .native_decoder import write_language_decoder
from .prompt import provider_recipes
from .vision import VisionConfig, build_vision_engine, load_vision_weights


def write_native(request, writer, cfg, input_limit, capacity):
    root = request.model_dir
    source = json.loads((root / "generation_config.json").read_text())
    if set(source["eos_token_id"]) != {248046, 248044}:
        raise ValueError("Flash-Next needs both checkpoint EOS IDs")
    # The released tokenizer JSON and template are sufficient; no custom HF model
    # code or Transformers 5.17 runtime is needed for native packaging.
    template = (root / "chat_template.jinja").read_text()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(root / "tokenizer.json"), chat_template=template
    )
    meta = dict(
        format=1,
        weight_format="prepared_flash_next_nvfp4",
        task=request.task,
        max_input_length=input_limit,
        capacity=capacity,
        eos=source["eos_token_id"],
        prompts=provider_recipes(tokenizer),
    )
    for marker in ("image_pad", "video_pad", "vision_start", "vision_end"):
        field = {"image_pad": "image_token_id", "video_pad": "video_token_id"}.get(
            marker, marker + "_token_id"
        )
        if tokenizer.encode("<|" + marker + "|>", add_special_tokens=False) != [cfg[field]]:
            raise ValueError("Flash-Next tokenizer disagrees with checkpoint " + field)
        meta[field] = cfg[field]
    with FlashCheckpoint(root) as ck:
        if request.task == "vision_language_generation":
            v = VisionConfig.from_dict(cfg["vision_config"], cfg["text_config"]["hidden_size"])
            processors = json.loads((root / "processor_config.json").read_text())
            image, video = (processors[k] for k in ("image_processor", "video_processor"))
            for p in (image, video):
                if any(
                    p.get(k) != val
                    for k, val in {
                        "patch_size": v.patch,
                        "merge_size": v.merge,
                        "temporal_patch_size": v.temporal,
                        "do_resize": True,
                        "do_rescale": True,
                        "do_normalize": True,
                        "do_convert_rgb": True,
                        "resample": 3,
                        "image_mean": [0.5] * 3,
                        "image_std": [0.5] * 3,
                        "rescale_factor": 1 / 255,
                    }.items()
                ):
                    raise ValueError("Unsupported Flash-Next visual processor")
            # A visual profile bounds allocation, not source geometry. Never silently shrink
            # the provider's chosen resize to make a request fit this plan.
            max_visual = min(1024, input_limit)
            meta["visual"] = dict(
                max_tokens=max_visual,
                image_min_pixels=image["size"]["shortest_edge"],
                image_max_pixels=image["size"]["longest_edge"],
                video_min_pixels=video["size"]["shortest_edge"],
                video_max_pixels=video["size"]["longest_edge"],
                hidden=v.output,
                patch=v.patch,
                temporal=v.temporal,
                merge=v.merge,
                positions=v.positions,
                head_dim=v.head_dim,
                theta=v.theta,
            )
            plan = build_vision_engine(v, load_vision_weights(ck, v), max_visual * v.merge**2)
            with writer.open_section("visual.plan") as out:
                out.write(plan)
            del plan
        write_language_decoder(
            ck, writer, cfg["text_config"], chunk=min(16, input_limit), capacity=capacity
        )
    with writer.open_section("tokenizer.json") as out, (root / "tokenizer.json").open("rb") as src:
        import shutil

        shutil.copyfileobj(src, out, length=1 << 20)
    writer.add_json("native.json", meta)
