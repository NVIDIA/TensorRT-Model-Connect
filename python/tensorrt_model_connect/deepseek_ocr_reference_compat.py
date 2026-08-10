# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Portions of the cache, RoPE, and eager-attention compatibility behavior are
# adapted from Hugging Face Transformers v4.46.3 `cache_utils.py` and
# `models/llama/modeling_llama.py`.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.

"""Transformers v5 adapters for the pinned DeepSeek-OCR-2 reference.

The exact upstream source was authored against Transformers 4.46.3. The
reference runs in its own Python process, so this module restores only the
removed import, config, call, and cache surfaces that source consumes. The
adapter retains the pinned Transformers 5.5.4 package while preserving the
v4 eager operation order where BF16 GPU arithmetic is observably different.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib.metadata import version
import math
from typing import Any


_SUPPORTED_TRANSFORMERS_VERSION = "5.5.4"
_COMPAT_STATE_ATTRIBUTE = "_trtmc_deepseek_ocr_compat_state"
_CONFIG_COMPAT_ATTRIBUTE = "_trtmc_deepseek_ocr_v4_config_compat"
_QWEN_CONFIG_COMPAT_ATTRIBUTE = "_trtmc_deepseek_ocr_qwen_config_compat"
_QWEN_MODEL_COMPAT_ATTRIBUTE = "_trtmc_deepseek_ocr_qwen_model_compat"
DEEPSEEK_OCR_REFERENCE_REVISION = "aaa02f3811945a91062062994c5c4a3f4c0af2b0"


@dataclass(frozen=True)
class DeepSeekOcrTransformersCompat:
    """Types installed into the isolated DeepSeek reference process."""

    eager_attention_type: type
    flash_attention_type: type
    dynamic_cache_type: type


def configure_deepseek_ocr_legacy_generation_cache(model: Any) -> None:
    """Keep the pinned remote model's v4 tuple-cache generation path."""
    model_type = type(model)
    if model_type.__name__ != "DeepseekOCR2ForCausalLM" or not model_type.__module__.startswith(
        "transformers_modules."
    ):
        raise RuntimeError(
            "DeepSeek-OCR generation cache compatibility received an unexpected model type: "
            f"{model_type.__module__}.{model_type.__name__}"
        )

    @classmethod
    def supports_default_dynamic_cache(cls) -> bool:
        del cls
        return False

    model_type._supports_default_dynamic_cache = supports_default_dynamic_cache
    if model_type._supports_default_dynamic_cache():
        raise RuntimeError("DeepSeek-OCR v5 dynamic-cache initialization remained enabled")

    original_prepare = model_type.prepare_inputs_for_generation

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        prepared = original_prepare(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        prepared_ids = prepared.get("input_ids")
        if past_key_values is not None and prepared_ids is not None:
            if isinstance(past_key_values, tuple):
                cached = past_key_values[0][0].shape[-2]
            else:
                cached = past_key_values.get_seq_length()
            if cached and prepared_ids.shape[1] == input_ids.shape[1] > 1:
                raise RuntimeError(
                    "DeepSeek-OCR generation did not trim cached input tokens: "
                    f"cache={type(past_key_values).__name__}, cached={cached}, "
                    f"input={input_ids.shape[1]}, prepared={prepared_ids.shape[1]}, "
                    f"mask={getattr(attention_mask, 'shape', None)}"
                )
            for name in ("position_ids", "cache_position"):
                value = prepared.get(name)
                if value is not None and value.shape[-1] > prepared_ids.shape[1]:
                    prepared[name] = value[..., -prepared_ids.shape[1] :]
        return prepared

    model_type.prepare_inputs_for_generation = prepare_inputs_for_generation


def configure_deepseek_ocr_rotary_embeddings(
    model: Any,
    compat: DeepSeekOcrTransformersCompat,
) -> tuple[int, int]:
    """Materialize v5 RoPE buffers skipped by the remote-code meta loader."""
    language_layers = 0
    visual_encoders = 0
    for _, module in model.named_modules():
        is_language_attention = isinstance(module, compat.eager_attention_type)
        module_type = type(module)
        is_visual_encoder = (
            module_type.__name__ == "CustomQwen2ModelInner"
            and module_type.__module__.startswith("transformers_modules.")
        )
        if not is_language_attention and not is_visual_encoder:
            continue

        rotary = getattr(module, "rotary_emb", None)
        config = getattr(module, "config", None)
        if rotary is None or config is None:
            raise RuntimeError(
                "Pinned DeepSeek-OCR module is missing its rotary embedding or config: "
                f"{module_type.__module__}.{module_type.__name__}"
            )
        parameter = next(module.parameters(), None)
        device = parameter.device if parameter is not None else None
        module.rotary_emb = type(rotary)(config=config, device=device)
        inv_freq = module.rotary_emb.inv_freq
        if inv_freq.device.type == "meta" or not bool((inv_freq > 0).all().item()):
            raise RuntimeError(
                "Pinned DeepSeek-OCR rotary frequencies were not materialized safely"
            )
        if is_language_attention:
            language_layers += 1
        else:
            visual_encoders += 1

    if language_layers == 0 or visual_encoders != 1:
        raise RuntimeError(
            "Pinned DeepSeek-OCR rotary module count mismatch: "
            f"language={language_layers}, visual={visual_encoders}"
        )
    return language_layers, visual_encoders


def _build_legacy_dynamic_cache(cache_base_type: type, torch) -> type:
    """Build the small v4 dynamic cache consumed by the pinned remote source."""

    class DeepSeekOcrLegacyDynamicCache(cache_base_type):
        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__(layers=[])
            self._seen_tokens = 0
            self.key_cache = []
            self.value_cache = []

        def __getitem__(self, layer_idx):
            if layer_idx >= len(self):
                raise KeyError(f"Cache has {len(self)} layers; cannot access layer {layer_idx}")
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def __iter__(self):
            return iter(zip(self.key_cache, self.value_cache))

        def __len__(self):
            return len(self.key_cache)

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            del cache_kwargs
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
            while len(self.key_cache) < layer_idx:
                self.key_cache.append([])
                self.value_cache.append([])
            if len(self.key_cache) == layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif len(self.key_cache[layer_idx]) == 0:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat(
                    (self.key_cache[layer_idx], key_states), dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    (self.value_cache[layer_idx], value_states), dim=-2
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx=0):
            if layer_idx >= len(self.key_cache) or len(self.key_cache[layer_idx]) == 0:
                return 0
            return self.key_cache[layer_idx].shape[-2]

        def get_max_cache_shape(self):
            return None

        def get_max_length(self):
            return None

        def get_usable_length(self, new_seq_length, layer_idx=0):
            del new_seq_length
            return self.get_seq_length(layer_idx)

        @property
        def seen_tokens(self):
            return self._seen_tokens

        def to_legacy_cache(self):
            return tuple(zip(self.key_cache, self.value_cache))

        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                    cache.update(key_states, value_states, layer_idx)
            return cache

        def reorder_cache(self, beam_indices):
            for layer_idx in range(len(self)):
                device_indices = beam_indices.to(self.key_cache[layer_idx].device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                    0, device_indices
                )
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                    0, device_indices
                )

    DeepSeekOcrLegacyDynamicCache.__name__ = "DynamicCache"
    DeepSeekOcrLegacyDynamicCache.__qualname__ = "DynamicCache"
    DeepSeekOcrLegacyDynamicCache.__module__ = "transformers.cache_utils"
    return DeepSeekOcrLegacyDynamicCache


def _install_config_compat(pretrained_config_type: type) -> None:
    """Keep v4 inherited remote-config construction and token attributes."""
    if getattr(pretrained_config_type, _CONFIG_COMPAT_ATTRIBUTE, False):
        return

    original_post_init = pretrained_config_type.__post_init__

    def post_init(self, **kwargs):
        token_ids = {
            name: kwargs[name]
            for name in ("pad_token_id", "bos_token_id", "eos_token_id")
            if name in kwargs
        }
        original_post_init(self, **kwargs)
        for name, value in token_ids.items():
            setattr(self, name, value)

    # Transformers 5 dataclass-wraps a remote subclass with no local __init__,
    # which bypasses its inherited custom constructor. The pinned source relies
    # on normal Python inheritance to retain DeepseekV2Config's v4 defaults.
    def legacy_init_subclass(cls, *args, **kwargs):
        del cls, args, kwargs

    pretrained_config_type.__post_init__ = post_init
    pretrained_config_type.__init_subclass__ = classmethod(legacy_init_subclass)
    setattr(pretrained_config_type, _CONFIG_COMPAT_ATTRIBUTE, True)


def _install_qwen_config_compat(qwen_config_type: type) -> None:
    """Keep the requested Qwen vision-encoder layer count under v5 strict config."""
    if getattr(qwen_config_type, _QWEN_CONFIG_COMPAT_ATTRIBUTE, False):
        return

    original_init = qwen_config_type.__init__

    def init(self, *args, **kwargs):
        if not args and "num_hidden_layers" in kwargs and "layer_types" not in kwargs:
            kwargs["layer_types"] = ["full_attention"] * int(kwargs["num_hidden_layers"])
        original_init(self, *args, **kwargs)

    qwen_config_type.__init__ = init
    setattr(qwen_config_type, _QWEN_CONFIG_COMPAT_ATTRIBUTE, True)


def _install_qwen_model_compat(qwen_model_type: type) -> None:
    """Route the pinned visual encoder through its source-owned custom mask."""
    if getattr(qwen_model_type, _QWEN_MODEL_COMPAT_ATTRIBUTE, False):
        return

    original_forward = qwen_model_type.forward

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        model_type = type(self)
        is_pinned_visual_encoder = (
            model_type.__name__ == "CustomQwen2ModelInner"
            and model_type.__module__.startswith("transformers_modules.")
        )
        token_type_ids = getattr(self, "_current_token_type_ids", None)
        if is_pinned_visual_encoder:
            if inputs_embeds is None or token_type_ids is None:
                raise RuntimeError(
                    "Pinned DeepSeek-OCR Qwen2 visual encoder requires inputs_embeds "
                    "and token_type_ids"
                )
            if set(self.config.layer_types) != {"full_attention"}:
                raise RuntimeError(
                    "Pinned DeepSeek-OCR Qwen2 visual encoder requires full-attention layers"
                )
            custom_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                kwargs.get("cache_position"),
                past_key_values,
                kwargs.get("output_attentions"),
            )
            attention_mask = {"full_attention": custom_mask}

        return original_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    qwen_model_type.forward = forward
    setattr(qwen_model_type, _QWEN_MODEL_COMPAT_ATTRIBUTE, True)


def install_deepseek_ocr_transformers_compat() -> DeepSeekOcrTransformersCompat:
    """Install the exact legacy surfaces required by DeepSeek-OCR-2."""
    installed_version = version("transformers")
    if installed_version != _SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "DeepSeek-OCR compatibility requires Transformers "
            f"{_SUPPORTED_TRANSFORMERS_VERSION}, found {installed_version}"
        )

    modeling_llama = importlib.import_module("transformers.models.llama.modeling_llama")
    existing = getattr(modeling_llama, _COMPAT_STATE_ATTRIBUTE, None)
    if existing is not None:
        return existing

    cache_utils = importlib.import_module("transformers.cache_utils")
    configuration_utils = importlib.import_module("transformers.configuration_utils")
    import_utils = importlib.import_module("transformers.utils.import_utils")
    torch = importlib.import_module("torch")
    if not hasattr(import_utils, "is_torch_fx_available"):

        def is_torch_fx_available() -> bool:
            return hasattr(torch, "fx")

        import_utils.is_torch_fx_available = is_torch_fx_available

    _install_config_compat(configuration_utils.PreTrainedConfig)
    dynamic_cache_type = _build_legacy_dynamic_cache(cache_utils.Cache, torch)
    cache_utils.DynamicCache = dynamic_cache_type
    qwen_configuration = importlib.import_module("transformers.models.qwen2.configuration_qwen2")
    _install_qwen_config_compat(qwen_configuration.Qwen2Config)
    qwen_modeling = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    _install_qwen_model_compat(qwen_modeling.Qwen2Model)

    current_attention_type = modeling_llama.LlamaAttention
    current_rotary_type = modeling_llama.LlamaRotaryEmbedding
    repeat_kv = modeling_llama.repeat_kv

    def apply_rotary_pos_emb(query, key, cos, sin):
        """Use the pinned v4 eager RoPE operations, without v5 kernel dispatch."""
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        def rotate_half(value):
            first = value[..., : value.shape[-1] // 2]
            second = value[..., value.shape[-1] // 2 :]
            return torch.cat((-second, first), dim=-1)

        return (
            (query * cos) + (rotate_half(query) * sin),
            (key * cos) + (rotate_half(key) * sin),
        )

    class DeepSeekOcrEagerLlamaAttention(current_attention_type):
        """Translate the v4 call contract to Transformers 5 eager attention."""

        _trtmc_deepseek_ocr_eager_compat = True

        def __init__(self, config, layer_idx=None):
            if getattr(config, "pretraining_tp", None) != 1:
                raise RuntimeError("Pinned DeepSeek-OCR requires pretraining_tp=1")
            if getattr(config, "_attn_implementation", None) != "eager":
                raise RuntimeError("Pinned DeepSeek-OCR requires eager attention")
            super().__init__(config=config, layer_idx=layer_idx)
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.rotary_emb = current_rotary_type(config=config)

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
            del use_cache
            batch_size, query_length, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = query_states.view(
                batch_size, query_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                batch_size, query_length, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                batch_size, query_length, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            if position_embeddings is None:
                if position_ids is None:
                    raise RuntimeError(
                        "DeepSeek-OCR eager attention requires explicit position_ids"
                    )
                position_embeddings = self.rotary_emb(value_states, position_ids)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_length = query_length
            previous_length = 0
            if past_key_value is not None:
                previous_length = past_key_value.get_usable_length(query_length, self.layer_idx)
                key_length += previous_length
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
            if attention_mask is not None:
                if attention_mask.shape[-1] != key_length:
                    raise RuntimeError(
                        "DeepSeek-OCR eager mask/cache mismatch before attention: "
                        f"layer={self.layer_idx}, query={query_length}, "
                        f"previous={previous_length}, mask={attention_mask.shape[-1]}"
                    )
                attention_mask = attention_mask[..., :key_length]

            del kwargs
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            # Keep v4's division operation: folding this into a reciprocal
            # multiply changes BF16 GPU logits and the qualified OCR output.
            weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            if attention_mask is not None:
                weights = weights + attention_mask
            weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            weights = torch.nn.functional.dropout(
                weights,
                p=self.attention_dropout,
                training=self.training,
            )
            output = torch.matmul(weights, value_states)
            expected_shape = (
                batch_size,
                self.num_heads,
                query_length,
                self.head_dim,
            )
            if output.size() != expected_shape:
                raise RuntimeError(
                    "DeepSeek-OCR eager attention output shape mismatch: "
                    f"expected={expected_shape}, found={tuple(output.size())}"
                )
            output = output.transpose(1, 2).contiguous()
            output = output.reshape(batch_size, query_length, -1)
            output = self.o_proj(output)
            if not output_attentions:
                weights = None
            return output, weights, past_key_value

    class DeepSeekOcrFlashAttentionCompat(DeepSeekOcrEagerLlamaAttention):
        """Import-compatible marker; selecting it is rejected before inference."""

        _trtmc_deepseek_ocr_flash_compat = True

    DeepSeekOcrEagerLlamaAttention.__name__ = "LlamaAttention"
    DeepSeekOcrEagerLlamaAttention.__qualname__ = "LlamaAttention"
    DeepSeekOcrEagerLlamaAttention.__module__ = modeling_llama.__name__
    DeepSeekOcrFlashAttentionCompat.__name__ = "LlamaFlashAttention2"
    DeepSeekOcrFlashAttentionCompat.__qualname__ = "LlamaFlashAttention2"
    DeepSeekOcrFlashAttentionCompat.__module__ = modeling_llama.__name__

    modeling_llama.LlamaAttention = DeepSeekOcrEagerLlamaAttention
    modeling_llama.LlamaFlashAttention2 = DeepSeekOcrFlashAttentionCompat
    state = DeepSeekOcrTransformersCompat(
        eager_attention_type=DeepSeekOcrEagerLlamaAttention,
        flash_attention_type=DeepSeekOcrFlashAttentionCompat,
        dynamic_cache_type=dynamic_cache_type,
    )
    setattr(modeling_llama, _COMPAT_STATE_ATTRIBUTE, state)
    return state


def assert_deepseek_ocr_eager_attention(
    model: Any,
    compat: DeepSeekOcrTransformersCompat,
) -> int:
    """Fail closed unless the loaded model selected pinned eager MHA layers."""
    config = getattr(model, "config", None)
    implementation = getattr(config, "_attn_implementation", None)
    if implementation != "eager":
        raise RuntimeError(
            f"DeepSeek-OCR reference must use eager attention; loaded {implementation!r}"
        )
    if not callable(getattr(model, "named_modules", None)):
        raise RuntimeError("DeepSeek-OCR reference model cannot be inspected")

    eager_modules: list[tuple[str, Any]] = []
    flash_modules: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, compat.flash_attention_type):
            flash_modules.append(name)
        elif isinstance(module, compat.eager_attention_type):
            eager_modules.append((name, module))
    if flash_modules:
        raise RuntimeError(
            "DeepSeek-OCR selected unsupported flash-attention compatibility "
            f"layers: {flash_modules}"
        )
    if not eager_modules:
        raise RuntimeError("DeepSeek-OCR did not instantiate the pinned eager MHA compatibility")

    expected_layer_counts: set[int] = set()
    for name, module in eager_modules:
        module_config = getattr(module, "config", None)
        if getattr(module_config, "_attn_implementation", None) != "eager":
            raise RuntimeError(f"DeepSeek-OCR layer {name!r} is not configured for eager attention")
        if getattr(module_config, "use_mla", None) is not False:
            raise RuntimeError(f"DeepSeek-OCR layer {name!r} did not select pinned MHA mode")
        expected_layer_counts.add(int(module_config.num_hidden_layers))
    if expected_layer_counts != {len(eager_modules)}:
        raise RuntimeError(
            "DeepSeek-OCR eager attention layer count mismatch: "
            f"expected {sorted(expected_layer_counts)}, found {len(eager_modules)}"
        )
    return len(eager_modules)
