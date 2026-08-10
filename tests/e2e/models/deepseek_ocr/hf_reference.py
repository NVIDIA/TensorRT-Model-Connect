#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official DeepSeek-OCR-2 Hugging Face inference path."""

from __future__ import annotations

import argparse
import json
import math
import tempfile

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _install_transformers_5_legacy_model_compat() -> None:
    """Bridge only the cache/import APIs used by the pinned upstream model code."""
    from transformers.cache_utils import Cache, DynamicCache
    import transformers.models.llama.modeling_llama as llama
    import transformers.utils.import_utils as import_utils

    if not hasattr(Cache, "get_usable_length"):
        def get_usable_length(self, new_seq_length, layer_idx=0):
            previous = self.get_seq_length(layer_idx)
            maximum = self.get_max_cache_shape()
            if maximum is None or maximum < 0:
                return previous
            if previous + new_seq_length > maximum:
                return max(0, maximum - new_seq_length)
            return previous

        Cache.get_usable_length = get_usable_length
    if not hasattr(Cache, "get_max_length"):
        def get_max_length(self):
            maximum = self.get_max_cache_shape()
            return None if maximum is None or maximum < 0 else maximum

        Cache.get_max_length = get_max_length
    if not hasattr(Cache, "seen_tokens"):
        Cache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "to_legacy_cache"):
        def to_legacy_cache(self):
            return tuple((layer.keys, layer.values) for layer in self.layers)

        DynamicCache.to_legacy_cache = to_legacy_cache
    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            for layer_idx, (key_states, value_states) in enumerate(
                past_key_values or ()
            ):
                cache.update(key_states, value_states, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(llama, "LlamaFlashAttention2"):
        from transformers.models.llama.configuration_llama import LlamaConfig

        class _LlamaAttentionCompat(torch.nn.Module):
            """Transformers 4.46 eager attention contract used by DeepSeek-OCR-2."""

            def __init__(self, config, layer_idx=None):
                super().__init__()
                self.config = config
                self.layer_idx = layer_idx
                self.attention_dropout = config.attention_dropout
                self.hidden_size = config.hidden_size
                self.num_heads = config.num_attention_heads
                self.head_dim = self.hidden_size // self.num_heads
                self.num_key_value_heads = config.num_key_value_heads
                self.num_key_value_groups = (
                    self.num_heads // self.num_key_value_heads
                )
                self.q_proj = torch.nn.Linear(
                    self.hidden_size,
                    self.num_heads * self.head_dim,
                    bias=config.attention_bias,
                )
                self.k_proj = torch.nn.Linear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=config.attention_bias,
                )
                self.v_proj = torch.nn.Linear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=config.attention_bias,
                )
                self.o_proj = torch.nn.Linear(
                    self.num_heads * self.head_dim,
                    self.hidden_size,
                    bias=config.attention_bias,
                )
                if getattr(config, "rope_scaling", None) is not None:
                    raise ValueError(
                        "DeepSeek-OCR-2 compatibility expects unscaled RoPE"
                    )
                rope_config = LlamaConfig(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    max_position_embeddings=config.max_position_embeddings,
                    rope_parameters={
                        "rope_theta": getattr(config, "rope_theta", 10000.0),
                        "rope_type": "default",
                    },
                )
                self.rotary_emb = llama.LlamaRotaryEmbedding(rope_config)

            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs,
            ):
                del use_cache, kwargs
                batch_size, query_length, _ = hidden_states.size()
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
                query_states = query_states.view(
                    batch_size, query_length, self.num_heads, self.head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    batch_size,
                    query_length,
                    self.num_key_value_heads,
                    self.head_dim,
                ).transpose(1, 2)
                value_states = value_states.view(
                    batch_size,
                    query_length,
                    self.num_key_value_heads,
                    self.head_dim,
                ).transpose(1, 2)

                if position_embeddings is None:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                else:
                    cos, sin = position_embeddings
                query_states, key_states = llama.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
                if past_key_value is not None:
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    key_states, value_states = past_key_value.update(
                        key_states,
                        value_states,
                        self.layer_idx,
                        cache_kwargs,
                    )
                key_states = llama.repeat_kv(
                    key_states, self.num_key_value_groups
                )
                value_states = llama.repeat_kv(
                    value_states, self.num_key_value_groups
                )
                attention_weights = torch.matmul(
                    query_states, key_states.transpose(2, 3)
                ) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    attention_weights = attention_weights + attention_mask[
                        :, :, :, : key_states.shape[-2]
                    ]
                attention_weights = F.softmax(
                    attention_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                attention_weights = F.dropout(
                    attention_weights,
                    p=self.attention_dropout,
                    training=self.training,
                )
                attention_output = torch.matmul(
                    attention_weights, value_states
                )
                attention_output = attention_output.transpose(1, 2).contiguous()
                attention_output = attention_output.reshape(
                    batch_size, query_length, -1
                )
                attention_output = self.o_proj(attention_output)
                if not output_attentions:
                    attention_weights = None
                return attention_output, attention_weights, past_key_value

        llama.LlamaAttention = _LlamaAttentionCompat
        llama.LlamaFlashAttention2 = _LlamaAttentionCompat
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    _install_transformers_5_legacy_model_compat()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().cuda()

    prompt = args.prompt
    if "<image>" not in prompt:
        prompt = f"<image>\n{prompt}"

    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_hf_") as output_dir:
        text = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=args.image,
            output_path=output_dir,
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=False,
            eval_mode=True,
        )

    print(json.dumps({"text": str(text or "")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
