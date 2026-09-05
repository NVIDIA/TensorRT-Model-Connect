# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Hugging Face reference for LocateAnything."""

from __future__ import annotations

import json
from pathlib import Path

from families.locateanything.tests.vision_oracle import preprocess_image_inputs_for_trt


class _LocalTokenizer:
    """Small tokenizer.json adapter for the methods used by LocateAnything."""

    def __init__(self, model_dir: Path) -> None:
        from tokenizers import Tokenizer

        tokenizer_path = model_dir / "tokenizer.json"
        config_path = model_dir / "tokenizer_config.json"
        if not tokenizer_path.is_file() or not config_path.is_file():
            raise FileNotFoundError("LocateAnything requires local tokenizer files")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.model_max_length = int(config.get("model_max_length", 16384))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text: str, return_tensors: str | None = None) -> dict:
        ids = self.encode(text)
        if return_tensors is None:
            return {"input_ids": ids}
        if return_tensors != "pt":
            raise ValueError("LocateAnything reference only supports PyTorch tensors")
        import torch

        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().reshape(-1).tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self._tokenizer.decode(
            [int(token) for token in ids], skip_special_tokens=skip_special_tokens
        )

    def batch_decode(self, batch_ids, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]


def _rope_theta(raw_config: dict) -> float:
    text_config = raw_config.get("text_config", {})
    if not isinstance(text_config, dict):
        text_config = {}
    value = text_config.get("rope_theta")
    for field in ("rope_parameters", "rope_scaling"):
        nested = text_config.get(field, {})
        if value is None and isinstance(nested, dict):
            value = nested.get("rope_theta")
    if value is None:
        value = raw_config.get("rope_theta", 10000.0)
    return float(value)


def _load_config(model_dir: Path):
    from transformers import AutoConfig

    raw_path = model_dir / "config.json"
    if not raw_path.is_file():
        raise FileNotFoundError("LocateAnything requires a local config.json")
    raw_config = json.loads(raw_path.read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    if hasattr(config, "text_config") and not hasattr(config.text_config, "rope_theta"):
        config.text_config.rope_theta = _rope_theta(raw_config)
    return config


def _repair_rotary_buffers(model) -> None:
    import torch

    repaired = 0
    model_device = next(model.parameters()).device
    for module in model.language_model.modules():
        if not all(hasattr(module, field) for field in ("_set_cos_sin_cache", "base", "dim")):
            continue
        device = getattr(getattr(module, "inv_freq", None), "device", model_device)
        dimension = int(module.dim)
        inv_freq = 1.0 / (
            float(module.base)
            ** (torch.arange(0, dimension, 2, device=device, dtype=torch.float32) / dimension)
        )
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        sequence_length = int(
            getattr(
                module,
                "max_position_embeddings",
                getattr(model.config.text_config, "max_position_embeddings", 32768),
            )
        )
        module._set_cos_sin_cache(
            seq_len=sequence_length, device=inv_freq.device, dtype=torch.float32
        )
        repaired += 1
    if repaired == 0:
        raise RuntimeError("LocateAnything reference did not find RoPE buffers")


def manual_chat_prompt(prompt: str) -> str:
    image_context = "<IMG_CONTEXT>" * 256
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<img>{image_context}</img>{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def official_reference(
    model_dir: Path,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    reference_precision: str,
) -> dict[str, str]:
    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("LocateAnything HF reference requires CUDA")
    torch.backends.cudnn.enabled = False
    config = _load_config(model_dir)
    tokenizer = _LocalTokenizer(model_dir)
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[reference_precision]
    model = AutoModel.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    ).to("cuda")
    model.eval()
    _repair_rotary_buffers(model)

    image_inputs = preprocess_image_inputs_for_trt(
        image_path,
        fixed_image_size=448,
        patch_size=14,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        interpolation="bicubic",
    )
    pixel_values = torch.from_numpy(image_inputs["pixel_values"]).to("cuda")
    image_grid_hws = torch.from_numpy(image_inputs["image_grid_hws"]).to(
        device="cuda", dtype=torch.int32
    )
    inputs = tokenizer(manual_chat_prompt(prompt), return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    with torch.no_grad():
        output = model.generate(
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode="slow",
            do_sample=False,
        )
    if isinstance(output, str):
        text = output
    elif isinstance(output, (list, tuple)) and output and isinstance(output[0], str):
        text = output[0]
    else:
        token_ids = output[0] if output.ndim > 1 else output
        if token_ids.numel() > input_ids.shape[-1]:
            token_ids = token_ids[input_ids.shape[-1] :]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    text = text.strip()
    if not text:
        raise RuntimeError("HF LocateAnything reference produced empty text")
    del model, output, inputs, input_ids, attention_mask, pixel_values, image_grid_hws
    torch.cuda.empty_cache()
    return {"text": text}
