# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ESM-2's Transformers masked LM for ``token_scores``, as tests/test_e2e.py runs it: the sequence padded to
the bundle's fixed window (``options.max_sequence_length``) with its attention mask, the vocabulary logits of
every real token (special tokens included)."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.spec, self.host = spec, host
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = host.maybe_compile(AutoModelForMaskedLM.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)
        self.window = int(spec.options["max_sequence_length"])

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        ids = self.tokenizer(str(self.host.required(request, "prompt")))["input_ids"]
        if len(ids) > self.window:
            raise ValueError(f"{len(ids)} tokens exceed the bundle window of {self.window}")
        padded = ids + [self.tokenizer.pad_token_id] * (self.window - len(ids))
        mask = [1] * len(ids) + [0] * (self.window - len(ids))
        tokens = torch.tensor([padded], device=self.spec.device)
        attention = torch.tensor([mask], device=self.spec.device)
        logits, model_ms = self.host.timed(
            lambda: self.model(input_ids=tokens, attention_mask=attention).logits[0, : len(ids)].float())
        return self.host.invocation({"scored_sequences": 1, "logits": logits.flatten().cpu().tolist(),
                                     "logits_shape": list(logits.shape)}, model_ms)
