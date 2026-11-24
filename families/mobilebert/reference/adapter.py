# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MobileBERT's Transformers pretraining model for ``token_scores``, as tests/test_e2e.py runs it: the text lowercased
with accents stripped (an uncased checkpoint) and framed as [CLS] text [SEP] without padding, the masked-LM logits of the [MASK] positions, or of every position when the text
has no [MASK] (the runtime's selection). A text longer than the bundle window is refused, as the runtime does."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import AutoTokenizer, MobileBertForPreTraining

        self.spec, self.host = spec, host
        # An uncased checkpoint: its tokenizer.json does not lowercase, so the BERT normalizer is asked to (lowercase and
        # strip accents, as the runtime does); otherwise a capital such as "A" becomes [UNK].
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model, **{**spec.pretrained_kwargs(), "do_lower_case": True})
        self.model = host.maybe_compile(MobileBertForPreTraining.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)
        self.window = int(spec.options["max_sequence_length"])

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        ids = self.tokenizer(str(self.host.required(request, "prompt")))["input_ids"]
        if len(ids) > self.window:
            raise ValueError(f"{len(ids)} tokens exceed the bundle window of {self.window}")
        masked = [index for index, value in enumerate(ids) if value == self.tokenizer.mask_token_id]
        rows = masked or list(range(len(ids)))
        tokens = torch.tensor([ids], device=self.spec.device)
        logits, model_ms = self.host.timed(
            lambda: self.model(input_ids=tokens).prediction_logits[0, rows].float())
        return self.host.invocation({"scored_sequences": 1, "logits": logits.flatten().cpu().tolist(),
                                     "logits_shape": list(logits.shape)}, model_ms)
