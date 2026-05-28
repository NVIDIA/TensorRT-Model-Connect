"""Decoder build strategy — standard CausalLM with StatelessCacheWrapper.

Handles decoder-only models (Qwen, LLaMA, Mistral, Phi, etc.) that use
KV cache for autoregressive generation. The wrapper converts between
raw TRT I/O format and HF's native format.
"""

from __future__ import annotations

import threading

import torch
import torch.nn as nn

from ..cache_config import make_export_args as _make_export_args

_patch_lock = threading.Lock()


def patch_static_cache_scatter():
    """Patch StaticLayer.update to use torch.scatter instead of index_copy_.

    index_copy_ is an in-place op that fragments the TRT graph into hundreds
    of fallback modules. scatter is functional (no in-place mutation) and
    produces a single clean TRT engine.

    Thread-safe: uses a lock to prevent concurrent check-then-patch races.
    """
    from transformers.cache_utils import StaticLayer

    with _patch_lock:
        if getattr(StaticLayer.update, '_scatter_patched', False):
            return

        def _functional_update(self, key_states, value_states, cache_kwargs=None):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)

            cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
            if cache_position is None:
                cache_position = torch.arange(
                    key_states.shape[-2], device=key_states.device)

            idx = cache_position.view(1, 1, -1, 1).expand_as(key_states)
            self.keys = self.keys.scatter(2, idx, key_states)
            self.values = self.values.scatter(2, idx, value_states)
            return self.keys, self.values

        _functional_update._scatter_patched = True
        StaticLayer.update = _functional_update


class StatelessCacheWrapper(nn.Module):
    """Wraps HF CausalLM with raw TRT-compatible I/O.

    Accepts inputs in the exact format used by the C++ DeviceKvCache runtime:
      - token_id:      int32 [1]
      - position_id:   int32 [1]
      - attention_mask: float32 [1, max_cache_length + 1]
      - cache_kv_{2i}: float32 [max_cache_length, kv_dim]  (keys layer i)
      - cache_kv_{2i+1}: float32 [max_cache_length, kv_dim]  (values layer i)
    where kv_dim = num_key_value_heads * head_dim.

    Produces outputs matching the raw TRT engine convention:
      - output0:       float32 [1, vocab_size]  (logits)
      - output{2i+1}:  float32 [1, kv_dim]  (present_k layer i)
      - output{2i+2}:  float32 [1, kv_dim]  (present_v layer i)

    Internally converts between raw TRT fp32 I/O and HF's native 4D cache
    format at the configured compute dtype.
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        max_cache_length: int,
        compute_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.compute_dtype = compute_dtype
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, 'head_dim',
            config.hidden_size // config.num_attention_heads)
        self.max_cache_length = max_cache_length
        self.attention_size = config.num_attention_heads * self.head_dim
        self.kv_dim = config.num_key_value_heads * self.head_dim

    def forward(self, token_id, position_id, attention_mask, *cache_kv):
        from transformers.cache_utils import StaticCache

        compute_dtype = self.compute_dtype

        # --- Input conversion: raw TRT format -> HF format ---

        # token_id: int32 [1] -> int64 [1, 1]
        input_ids = token_id.to(torch.int64).unsqueeze(0)

        # position_id: int32 [1] -> int64 [1]
        cache_position = position_id.to(torch.int64)
        position_ids = cache_position.unsqueeze(0)  # [1, 1]

        # attention_mask: float32 [1, max_cache_length+1] -> compute_dtype [1, 1, 1, max_cache_length]
        hf_mask = attention_mask[:, :self.max_cache_length]  # [1, max_cache_length]
        idx = cache_position.view(1, 1)  # [1, 1] -- current position to unmask
        hf_mask = hf_mask.scatter(1, idx, 0.0)  # unmask current cache position
        hf_mask = hf_mask.to(compute_dtype).unsqueeze(1).unsqueeze(1)  # [1, 1, 1, max_cache_length]

        # cache_kv: float32 [max_cache_length, kv_dim] -> compute_dtype [1, num_kv_heads, max_cache_length, head_dim]
        cache = StaticCache(
            self.config, max_cache_len=self.max_cache_length,
            dtype=compute_dtype, device=token_id.device)

        for i in range(self.num_layers):
            raw_k = cache_kv[2 * i]   # [max_cache_length, kv_dim]
            raw_v = cache_kv[2 * i + 1]

            k_compact = raw_k.view(
                self.max_cache_length, self.num_kv_heads, self.head_dim)
            v_compact = raw_v.view(
                self.max_cache_length, self.num_kv_heads, self.head_dim)

            # Reshape to HF format: [1, num_kv_heads, max_cache_length, head_dim] compute_dtype
            layer = cache.layers[i]
            layer.keys = k_compact.permute(1, 0, 2).unsqueeze(0).to(compute_dtype)
            layer.values = v_compact.permute(1, 0, 2).unsqueeze(0).to(compute_dtype)
            layer.is_initialized = True

        # --- Run HF model ---
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=hf_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # --- Output conversion: HF format -> raw TRT format ---

        # logits: [1, 1, vocab_size] -> [1, vocab_size] float32
        logits = outputs.logits.squeeze(1).to(torch.float32)

        # present KV: [1, num_kv_heads, 1, head_dim] -> [1, kv_dim] float32
        present_outputs = []
        updated_cache = outputs.past_key_values
        for i in range(self.num_layers):
            idx = cache_position.view(1, 1, -1, 1).expand(
                1, self.num_kv_heads, 1, self.head_dim)
            new_k = updated_cache.layers[i].keys.gather(2, idx)
            new_v = updated_cache.layers[i].values.gather(2, idx)

            present_outputs.append(new_k.reshape(1, self.kv_dim).to(torch.float32))
            present_outputs.append(new_v.reshape(1, self.kv_dim).to(torch.float32))

        return (logits, *present_outputs)


class DecoderBuildStrategy:
    """Build strategy for standard decoder-only models with KV cache."""

    name = "decoder"
    runtime_strategy = "torchtrt_decoder"

    def wrap_model(
        self,
        model: nn.Module,
        config,
        max_cache_length: int,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> nn.Module:
        if compute_dtype is None:
            compute_dtype = torch.float16
        return StatelessCacheWrapper(
            model, config, max_cache_length, compute_dtype=compute_dtype)

    def make_export_args(
        self,
        config,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        return _make_export_args(config, max_cache_length, precision=precision)

    def pre_export_setup(self) -> None:
        patch_static_cache_scatter()
