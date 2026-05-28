"""BERT family plugin for Torch-TRT — encoder-only models.

Uses the encoder_only build strategy: no KV cache, no autoregressive loop.
The wrapper accepts input_ids + attention_mask and returns the last hidden state.

Note: C++ runtime does not yet handle torchtrt_encoder bundles.
This plugin enables Python-side bundle building only.
"""

from __future__ import annotations

import torch

from ..config import ModelConfig


class BertTorchTrtPlugin:
    name = "bert"
    runtime_strategy = "encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "bert"

    def load_model(
        self,
        model_dir: str,
        config: ModelConfig,
        max_cache_length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.nn.Module:
        from transformers import AutoModel

        if dtype is None:
            dtype = torch.float16

        model = AutoModel.from_pretrained(
            model_dir,
            dtype=dtype,
            device_map="cuda",
        )
        model.eval()
        return model

    def get_export_args(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        from ..strategies.encoder_only import EncoderOnlyBuildStrategy
        strategy = EncoderOnlyBuildStrategy()
        return strategy.make_export_args(
            model.config, max_cache_length, precision=precision)


plugin = BertTorchTrtPlugin()
