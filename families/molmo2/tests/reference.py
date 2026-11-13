# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the unmodified upstream processor and model in an isolated environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(args.output / "hf-modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    assert transformers.__version__ == "4.57.1", (
        "use tests/requirements.txt in an isolated reference environment"
    )
    manifest = json.loads((Path(__file__).parent / "manifests/molmo2-8b.json").read_text())
    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    model = None
    if not args.preprocess_only:
        print("Loading pinned upstream Molmo2-8B in BF16", flush=True)
        model = (
            AutoModelForImageTextToText.from_pretrained(
                args.model_dir,
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch.bfloat16,
            )
            .to("cuda")
            .eval()
        )
    results = []
    for case in manifest["testcases"]:
        if "test_image" in case:
            image = Image.open(Path(__file__).parent / case["test_image"]).convert("RGB")
        else:
            image = Image.new("RGB", tuple(case["size"]), tuple(case["color"]))
        name = case["name"]
        image.save(args.output / (name + ".png"))
        np.asarray(image, dtype=np.float32).tofile(args.output / (name + ".f32"))
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": case["prompt"]}],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        for key, value in inputs.items():
            np.save(args.output / (name + "-" + key + ".npy"), value.numpy())
        record = {
            "name": name,
            "input_ids": inputs["input_ids"][0].tolist(),
            "width": image.width,
            "height": image.height,
        }
        if model is not None:
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs, use_cache=True)
                np.save(
                    args.output / (name + "-features.npy"),
                    output.image_hidden_states.float().cpu().numpy(),
                )
                np.save(
                    args.output / (name + "-logits.npy"), output.logits[0, -1].float().cpu().numpy()
                )
                ids = model.generate(
                    **inputs, max_new_tokens=case["max_new_tokens"], do_sample=False
                )
            record["tokens"] = ids[0, inputs["input_ids"].shape[1] :].tolist()
            record["text"] = processor.tokenizer.decode(record["tokens"], skip_special_tokens=True)
        results.append(record)
        print(json.dumps(record), flush=True)
    (args.output / "reference.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
