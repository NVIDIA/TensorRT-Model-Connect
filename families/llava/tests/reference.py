# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent HF reference, runnable in an isolated CUDA PyTorch interpreter."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils import logging

logging.disable_progress_bar()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "LLaVA reference requires CUDA"
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float16, attn_implementation="eager",
    ).eval().to("cuda")
    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=False)
    results = {}
    for case in cases:
        image_path = Path(case["image"])
        image = Image.open(image_path).convert("RGB")
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": case["prompt"]}]}],
            add_generation_prompt=True,
        )
        inputs = processor(text=text, images=image, return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=case["max_new_tokens"], do_sample=False)
            features = model.model.get_image_features(
                inputs["pixel_values"], vision_feature_layer=model.config.vision_feature_layer,
                vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            ).pooler_output
        ids = output[0, inputs["input_ids"].shape[1]:].tolist()
        # Retain independent intermediate evidence for mismatch localization.
        np.save(args.output / (case["name"] + "-pixels.npy"), inputs["pixel_values"].float().cpu().numpy())
        if isinstance(features, (list, tuple)):
            features = torch.cat(features, dim=0)
        np.save(args.output / (case["name"] + "-features.npy"), features.float().cpu().numpy())
        results[case["name"]] = {
            "token_ids": ids, "text": processor.decode(ids, skip_special_tokens=True),
            "input_ids": inputs["input_ids"][0].tolist(), "image": str(image_path),
        }
        print(json.dumps({"case": case["name"], "text": results[case["name"]]["text"],
                          "token_ids": ids}), flush=True)
    (args.output / "reference.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
