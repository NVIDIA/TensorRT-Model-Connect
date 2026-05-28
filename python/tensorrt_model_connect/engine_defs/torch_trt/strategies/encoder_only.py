"""Encoder-only build strategy — models like BERT that produce embeddings.

No KV cache, no autoregressive loop. The wrapper accepts input_ids and
attention_mask, and returns the last hidden state.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EncoderOnlyWrapper(nn.Module):
    """Wraps an encoder-only HF model with raw TRT-compatible I/O.

    Inputs:
      - input_ids:      int32 [1, seq_len]
      - attention_mask:  float32 [1, seq_len]

    Outputs:
      - output0:  float32 [1, seq_len, hidden_size]  (last hidden state)
    """

    def __init__(self, model: nn.Module, config, max_seq_length: int):
        super().__init__()
        self.model = model
        self.hidden_size = config.hidden_size
        self.max_seq_length = max_seq_length

    def forward(self, input_ids, attention_mask):
        # input_ids: int32 [1, seq_len] -> int64 [1, seq_len]
        ids = input_ids.to(torch.int64)

        # attention_mask: float32 [1, seq_len] -> int64 [1, seq_len]
        # HF expects 1/0 integer mask, not float
        mask = attention_mask.to(torch.int64)

        outputs = self.model(input_ids=ids, attention_mask=mask)

        # last_hidden_state: [1, seq_len, hidden_size] -> float32
        return (outputs.last_hidden_state.to(torch.float32),)


class EncoderOnlyBuildStrategy:
    """Build strategy for encoder-only models (BERT, etc.)."""

    name = "encoder_only"
    runtime_strategy = "torchtrt_encoder"

    def wrap_model(
        self,
        model: nn.Module,
        config,
        max_cache_length: int,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> nn.Module:
        return EncoderOnlyWrapper(model, config, max_cache_length)

    def make_export_args(
        self,
        config,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        # max_cache_length serves as max_seq_length for encoder-only models
        input_ids = torch.zeros(
            (1, max_cache_length), dtype=torch.int32, device="cuda")
        attention_mask = torch.ones(
            (1, max_cache_length), dtype=torch.float32, device="cuda")
        return (input_ids, attention_mask)

    def pre_export_setup(self) -> None:
        pass  # No global patches needed for encoder-only models
