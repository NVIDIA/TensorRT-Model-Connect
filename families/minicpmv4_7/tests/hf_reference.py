# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official Transformers oracle for the MiniCPM-V 4.7 family E2E.

Runs in the reference environment (``requirements.txt``) and writes one JSON record per request:

* ``prepare``: the provider processor's ``input_ids``, crop grids, per-crop pixel patches (``.f32``
  file in the runtime's [patches, 3 * p * p] layout) and the canvas M-RoPE positions of
  ``get_rope_index`` -- the host-side contract the native runtime must reproduce exactly.
* ``generate``: greedy ``generate`` token ids.
* ``score``: teacher-forced logits over ``--candidate`` tokens: for each generated position the
  reference argmax, top-2 gap and the candidate's distance below the maximum.

Usage:
  python -m families.minicpmv4_7.tests.hf_reference --checkpoint DIR --request REQUEST.json
      --output OUT.json [--dtype float32|bfloat16] [--device cuda] [--candidate IDS.json]
      [--max-new-tokens N]
REQUEST.json: {"enable_thinking": false, "messages": [{"role": "user", "parts": [{"image": "a.png"},
               {"text": "..."}]}]} (image paths relative to the request file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _messages(request: dict, base: Path):
    from PIL import Image

    images, messages = [], []
    for message in request["messages"]:
        content = []
        for part in message["parts"]:
            if "image" in part:
                image = Image.open(base / part["image"]).convert("RGB")
                images.append(image)
                content.append({"type": "image", "image": image})
            else:
                content.append({"type": "text", "text": part["text"]})
        messages.append({"role": message["role"], "content": content})
    return messages, images


def _patches(pixel_values, target_sizes, patch: int):
    """Provider NaViT packing [1, C, p, total * p] -> per-crop rows [patches, C * p * p]."""
    import torch

    values = pixel_values[0].to(torch.float32).cpu()  # [C, p, total * p]
    channels = values.shape[0]
    total = values.shape[-1] // patch
    rows = values.reshape(channels, patch, total, patch).permute(2, 0, 1, 3).reshape(total, -1)
    counts = [int(h) * int(w) for h, w in target_sizes.tolist()]
    if sum(counts) != total:
        raise ValueError("pixel_values and target_sizes disagree")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--candidate", type=Path, help="JSON list of generated token ids to score")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    request = json.loads(args.request.read_text(encoding="utf-8"))
    messages, _ = _messages(request, args.request.parent)
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    dtype = getattr(torch, args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(str(args.checkpoint), dtype=dtype).to(args.device).eval()
    if dtype == torch.float32:
        assert torch.get_float32_matmul_precision() == "highest"
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        enable_thinking=bool(request.get("enable_thinking", False)))
    inputs = inputs.to(args.device)
    ids = inputs["input_ids"]
    record = {"checkpoint": str(args.checkpoint), "dtype": args.dtype, "input_ids": ids[0].tolist()}

    if "pixel_values" in inputs:
        sizes = inputs["target_sizes"]
        patch = model.config.vision_config.patch_size
        rows = _patches(inputs["pixel_values"], sizes, patch)
        rows.numpy().astype("<f4").tofile(args.output.with_suffix(".patches.f32"))
        record["crop_grids"] = sizes.tolist()
        positions, deltas = model.model.get_rope_index(
            ids, mm_token_type_ids=inputs["mm_token_type_ids"], target_sizes=sizes,
            attention_mask=inputs["attention_mask"])
        record["positions"] = positions[:, 0, :].transpose(0, 1).tolist()
        record["decode_position"] = int(positions.max()) + 1
        record["rope_delta"] = int(deltas[0, 0])
        # Vision-tower position ids, per crop, exactly as the model computes them on its device.
        from transformers.vision_utils import get_vision_nearest_position_ids

        per_side = model.config.vision_config.image_size // patch
        record["crop_position_ids"] = [
            get_vision_nearest_position_ids(sizes[i:i + 1], per_side).tolist() for i in range(sizes.shape[0])]

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   temperature=None, top_p=None, top_k=None)
    record["generated_ids"] = generated[0, ids.shape[1]:].tolist()
    record["generated_text"] = processor.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)

    if args.candidate is not None:
        candidate = json.loads(args.candidate.read_text())
        sequence = torch.cat([ids, torch.tensor([candidate], device=ids.device, dtype=ids.dtype)], dim=1)
        forward = dict(inputs)
        forward["input_ids"] = sequence
        forward["attention_mask"] = torch.ones_like(sequence)
        forward["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(sequence[:, ids.shape[1]:])], dim=1)
        with torch.no_grad():
            logits = model(**forward).logits[0, ids.shape[1] - 1:-1].to(torch.float32)
        top = torch.topk(logits, 2, dim=-1)
        chosen = logits.gather(1, torch.tensor(candidate, device=logits.device)[:, None])[:, 0]
        record["teacher_forced"] = {
            "candidate": candidate,
            "argmax": top.indices[:, 0].tolist(),
            "top2_gap": (top.values[:, 0] - top.values[:, 1]).tolist(),
            "candidate_gap": (top.values[:, 0] - chosen).tolist(),
        }
    args.output.write_text(json.dumps(record))


if __name__ == "__main__":
    main()
