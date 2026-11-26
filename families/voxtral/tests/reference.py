# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official Transformers reference for the Voxtral E2E.

The checkpoint is loaded by Transformers' own ``VoxtralForConditionalGeneration``
loader (not the family's reader) and run in float32; prompt ids and log-mel
features come from the official ``VoxtralProcessor`` (mistral_common).
"""

from __future__ import annotations

from pathlib import Path

import torch


class VoxtralReference:
    """Float32 official model and processor; TF32 is disabled for exact float32 products."""

    def __init__(self, model_dir: Path, device: str = "cuda"):
        from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

        self.model_dir = Path(model_dir)
        model = VoxtralForConditionalGeneration.from_pretrained(
            str(self.model_dir), dtype=torch.float32, local_files_only=True)
        self.model = model.to(device).eval()
        self.processor = VoxtralProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        self.device = device

    def inputs(self, wav: Path, language: str | None) -> dict:
        batch = self.processor.apply_transcription_request(
            audio=str(wav), model_id=str(self.model_dir), language=language)
        return {key: value.to(self.device) for key, value in batch.items()}

    def _run(self, function, *args):
        """Run ``function`` with TF32 off and restore the caller's settings, also on failure."""
        saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        try:
            with torch.inference_mode():
                return function(*args)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[:2]
            torch.set_float32_matmul_precision(saved[2])

    def transcribe(self, wav: Path, language: str | None, max_new_tokens: int):
        """Greedy ids (end-of-sequence dropped), the processor's decoded text and the prompt ids."""
        return self._run(self._transcribe, wav, language, max_new_tokens)

    def _transcribe(self, wav: Path, language: str | None, max_new_tokens: int):
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

    def decode(self, ids: list[int]) -> str:
        """The official processor's text for ``ids`` (special tokens skipped)."""
        return self.processor.batch_decode([list(ids)], skip_special_tokens=True)[0]

    def teacher_forced_logits(self, wav: Path, language: str | None,
                              tokens: list[int]) -> torch.Tensor:
        """Float32 logits that predict each of ``tokens``, and the token after them, given the prompt."""
        return self._run(self._teacher_forced_logits, wav, language, tokens)

    def _teacher_forced_logits(self, wav: Path, language: str | None, tokens: list[int]) -> torch.Tensor:
        inputs = self.inputs(wav, language)
        prompt = inputs["input_ids"]
        forced = torch.tensor([tokens], dtype=prompt.dtype, device=prompt.device)
        ids = torch.cat([prompt, forced], dim=1)
        attention = torch.ones_like(ids)
        logits = self.model(input_ids=ids, input_features=inputs["input_features"],
                            attention_mask=attention).logits[0]
        start = prompt.shape[1] - 1
        return logits[start : start + len(tokens) + 1].float()
