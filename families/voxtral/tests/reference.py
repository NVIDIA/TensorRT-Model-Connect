# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official Transformers reference for the Voxtral E2E.

Transformers cannot load MLX exports, so this module dequantizes the
checkpoint itself and loads the dense float32 weights into the official
``VoxtralForConditionalGeneration``. The dequantization is written
independently of the family builder (bit matrix in torch, not the builder's
eight-code blocks in NumPy) so that one cannot hide the other's mistakes.
Prompt ids and log-mel features come from the official ``VoxtralProcessor``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

_BLOCK_ROWS = 8192


def mlx_dequantize(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, bits: int,
                   group_size: int) -> torch.Tensor:
    """Dense float32 weight of one MLX affine-quantized matrix.

    Every output row is a little-endian bit stream of ``bits``-wide codes
    stored in uint32 words; ``w = scale * code + bias`` per ``group_size`` columns.
    """
    words = packed.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rows = words.shape[0]
    columns = words.shape[1] * 32 // bits
    place = (2 ** torch.arange(bits, dtype=torch.int64)).to(words.device)
    shifts = torch.arange(32, dtype=torch.int64, device=words.device)
    output = torch.empty((rows, columns), dtype=torch.float32, device=words.device)
    for start in range(0, rows, _BLOCK_ROWS):
        block = words[start : start + _BLOCK_ROWS]
        stream = ((block[:, :, None] >> shifts) & 1).reshape(block.shape[0], columns, bits)
        codes = (stream * place).sum(-1).to(torch.float32)
        codes = codes.reshape(block.shape[0], -1, group_size)
        scale = scales[start : start + _BLOCK_ROWS].to(torch.float32)[:, :, None]
        bias = biases[start : start + _BLOCK_ROWS].to(torch.float32)[:, :, None]
        output[start : start + _BLOCK_ROWS] = (codes * scale + bias).reshape(block.shape[0], columns)
    return output


def dense_state_dict(model_dir: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """The export as a float32 Transformers state dict (duplicate embedding dropped)."""
    from safetensors.torch import load_file

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    quantization = config["quantization"]
    tensors = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        tensors.update(load_file(str(path), device=device))
    state = {}
    for name, value in tensors.items():
        module, _, suffix = name.rpartition(".")
        if module == "embed_tokens" or suffix in ("scales", "biases"):
            continue
        spec = quantization.get(module, False)
        if spec and suffix == "weight":
            value = mlx_dequantize(value, tensors[f"{module}.scales"], tensors[f"{module}.biases"],
                                   int(spec["bits"]), int(spec["group_size"]))
        elif module in ("audio_tower.conv1", "audio_tower.conv2") and suffix == "weight":
            value = value.permute(0, 2, 1)  # MLX [out, kernel, in] -> torch [out, in, kernel]
        state[name] = value.to(torch.float32).contiguous()
    return state


class VoxtralReference:
    """Float32 official model and processor; TF32 is disabled for exact float32 products."""

    def __init__(self, model_dir: Path, device: str = "cuda"):
        from transformers import (
            GenerationConfig,
            VoxtralConfig,
            VoxtralForConditionalGeneration,
            VoxtralProcessor,
        )

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.get_float32_matmul_precision() == "highest"
        self.model_dir = Path(model_dir)
        raw = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        for key in ("quantization", "quantization_config"):
            raw.pop(key, None)
        config = VoxtralConfig.from_dict(raw)
        with torch.device(device):
            model = VoxtralForConditionalGeneration(config).to(torch.float32)
        # Transformers 5 nests the towers under ``model.``; the export uses the flat names.
        expected = set(model.state_dict())
        state = {(name if name in expected else f"model.{name}"): value
                 for name, value in dense_state_dict(self.model_dir, device).items()}
        model.load_state_dict(state, strict=True)
        model.generation_config = GenerationConfig.from_pretrained(str(self.model_dir))
        self.model = model.eval()
        self.processor = VoxtralProcessor.from_pretrained(str(self.model_dir))
        self.device = device

    def inputs(self, wav: Path, language: str | None) -> dict:
        batch = self.processor.apply_transcription_request(
            audio=str(wav), model_id=str(self.model_dir), language=language)
        return {key: value.to(self.device) for key, value in batch.items()}

    @torch.inference_mode()
    def transcribe(self, wav: Path, language: str | None, max_new_tokens: int):
        """Greedy ids (end-of-sequence dropped) and the processor's decoded text."""
        inputs = self.inputs(wav, language)
        prompt_length = inputs["input_ids"].shape[1]
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     num_beams=1)
        ids = output[0, prompt_length:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        if ids and ids[-1] in eos:
            ids = ids[:-1]
        text = self.processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0]
        return ids, text, inputs["input_ids"][0].tolist()

    @torch.inference_mode()
    def teacher_forced_logits(self, wav: Path, language: str | None,
                              tokens: list[int]) -> torch.Tensor:
        """Float32 logits that predict each of ``tokens`` given the prompt and its predecessors."""
        inputs = self.inputs(wav, language)
        prompt = inputs["input_ids"]
        forced = torch.tensor([tokens], dtype=prompt.dtype, device=prompt.device)
        ids = torch.cat([prompt, forced], dim=1)
        attention = torch.ones_like(ids)
        logits = self.model(input_ids=ids, input_features=inputs["input_features"],
                            attention_mask=attention).logits[0]
        start = prompt.shape[1] - 1
        return logits[start : start + len(tokens) + 1].float()
