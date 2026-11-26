# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the independent pinned official model and retain full OCR observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRANSFORMERS_VERSION = "5.2.0"


def run(checkpoint: Path, image_path: Path, output: Path, *, max_new_tokens: int = 1024,
        device: str = "cuda") -> None:
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"The official reference requires Transformers {TRANSFORMERS_VERSION}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The selected official reference requires a CUDA device")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=True)
    processor = LightOnOcrProcessor.from_pretrained(
        checkpoint, local_files_only=True, use_fast=False,
    )
    model, loading = LightOnOcrForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise RuntimeError(f"Official checkpoint loading was incomplete: {key}: {loading[key]}")
    loading = {key: sorted(value) if isinstance(value, set) else value
               for key, value in loading.items()}
    model = model.to(device).eval()
    embedding = model.get_input_embeddings()
    if embedding.weight.dtype != torch.float32:
        raise RuntimeError("The official reference did not load FP32 embeddings")
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    prompt_ids = inputs["input_ids"][0].tolist()
    metadata = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "use_fast": False,
        "precision": "fp32",
        "embedding_dtype": "float32",
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "processor_config": json.loads((checkpoint / "processor_config.json").read_text()),
        "loading_info": loading,
        "image_dimensions": [image.height, image.width],
        "prompt_ids": prompt_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "device": device,
        "tensors": {},
    }

    def save(name, tensor):
        value = tensor.detach().float().cpu().contiguous().numpy()
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite official observation: {name}")
        filename = name + ".f32"
        value.tofile(output / filename)
        metadata["tensors"][name] = {
            "file": filename, "shape": list(value.shape), "dtype": "float32",
        }

    save("pixel_values", inputs["pixel_values"])
    with torch.inference_mode():
        first = model(**inputs, use_cache=True, return_dict=True, logits_to_keep=1)
        features = first.image_hidden_states
        if isinstance(features, (list, tuple)):
            features = torch.cat(features, dim=0)
        save("image_features", features.reshape(-1, model.config.text_config.hidden_size))
        logits = first.logits[:, -1, :]
        save("step_0_logits", logits)
        forced_id = int(logits.argmax(-1).item())
        metadata["teacher_forced_ids"] = [forced_id]
        previous = [(layer.keys.clone(), layer.values.clone())
                    for layer in first.past_key_values.layers]
        second = model(
            input_ids=torch.tensor([[forced_id]], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, len(prompt_ids) + 1), dtype=torch.long, device=device),
            past_key_values=first.past_key_values,
            use_cache=True, return_dict=True, logits_to_keep=1,
        )
        save("step_1_logits", second.logits[:, -1, :])
        prefix_unchanged = all(
            torch.equal(layer.keys[..., :len(prompt_ids), :], saved_key)
            and torch.equal(layer.values[..., :len(prompt_ids), :], saved_value)
            for layer, (saved_key, saved_value) in zip(
                second.past_key_values.layers, previous, strict=True,
            )
        )
        if not prefix_unchanged:
            raise RuntimeError("The official autoregressive step mutated an existing cache prefix")
        metadata["cache_prefix_unchanged"] = True
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    token_ids = generated[0, len(prompt_ids):].tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    metadata["generation"] = {
        "text": processor.decode(token_ids, skip_special_tokens=True),
        "token_ids": token_ids,
        "eos_token_ids": eos_ids,
        "reached_eos": bool(token_ids and token_ids[-1] in eos_ids),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.checkpoint, args.image, args.output,
        max_new_tokens=args.max_new_tokens, device=args.device)


if __name__ == "__main__":
    main()
