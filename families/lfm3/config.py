# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import math


def _integer(raw, name, default=None, minimum=1):
    value = raw.get(name, default)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _boolean(raw, name, default):
    value = raw.get(name, default)
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class DenseLfm3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    norm_eps: float
    conv_l_cache: int
    conv_bias: bool
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int | tuple[int, ...]
    pad_token_id: int
    layer_types: tuple[str, ...]
    full_rope: tuple[int, float]
    sliding_rope: tuple[int, float]
    sliding_window: int
    use_sandwich_norm: bool

    @property
    def attention_size(self):
        return self.num_attention_heads * self.head_dim

    @property
    def kv_attention_size(self):
        return self.num_key_value_heads * self.head_dim

    @property
    def num_attention_layers(self):
        return sum(t != "conv" for t in self.layer_types)

    @property
    def num_conv_layers(self):
        return self.layer_types.count("conv")

    @property
    def default_cache_length(self):
        return min(self.max_position_embeddings, 4096)

    def rope_spec(self, layer_type):
        if layer_type == "full_attention":
            return self.full_rope
        if layer_type == "sliding_attention":
            return self.sliding_rope
        raise ValueError("RoPE requires an attention layer")

    def bundle_overrides(self):
        return dict(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            norm_eps=self.norm_eps,
            rope_parameters={
                t: dict(rotary_dim=self.rope_spec(t)[0], rope_theta=self.rope_spec(t)[1])
                for t in ("full_attention", "sliding_attention")
            },
            layer_types=list(self.layer_types),
            num_attention_layers=self.num_attention_layers,
            num_conv_layers=self.num_conv_layers,
            conv_L_cache=self.conv_l_cache,
            conv_dim=self.hidden_size,
            tie_word_embeddings=self.tie_word_embeddings,
            native_kv_cache=True,
            native_kv_contract_version=1,
            sliding_window=self.sliding_window,
            use_sandwich_norm=self.use_sandwich_norm,
        )


def validate_dense_lfm3_config(config):
    raw = getattr(config, "raw", None)
    if not isinstance(raw, dict) or raw.get("model_type") != "lfm3":
        raise ValueError("LFM3 requires model_type='lfm3'")
    if raw.get("architectures", ["Lfm3ForCausalLM"]) != ["Lfm3ForCausalLM"]:
        raise ValueError("LFM3 requires architectures=['Lfm3ForCausalLM']")
    if any(raw.get(k) for k in ("vision_config", "num_experts", "num_local_experts")):
        raise ValueError("LFM3 does not support vision or MoE variants")
    if raw.get("quantization_config") is not None:
        raise ValueError("LFM3 does not support quantized checkpoints")
    fields = {
        k: _integer(raw, k)
        for k in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        )
    }
    head = _integer(raw, "head_dim", fields["hidden_size"] // fields["num_attention_heads"])
    if head % 2 or fields["num_attention_heads"] % fields["num_key_value_heads"]:
        raise ValueError("LFM3 requires even head_dim and integral grouped-query heads")
    if fields["hidden_size"] != fields["num_attention_heads"] * head:
        raise ValueError("hidden_size must equal num_attention_heads * head_dim")
    layers = raw.get("layer_types")
    if not isinstance(layers, list) or len(layers) != fields["num_hidden_layers"]:
        raise ValueError("layer_types must cover all layers")
    if any(t not in ("conv", "full_attention", "sliding_attention") for t in layers):
        raise ValueError("unsupported LFM3 layer type")
    if "conv" not in layers or not any(t != "conv" for t in layers):
        raise ValueError("LFM3 requires convolution and attention layers")
    rope = raw.get("rope_parameters", {})
    if not isinstance(rope, dict):
        raise ValueError("rope_parameters must be an object")
    specs = []
    for name, theta, factor in [("full_attention", 1e6, 0.5), ("sliding_attention", 1e4, 1.0)]:
        item = rope.get(name, dict(rope_theta=theta, partial_rotary_factor=factor))
        if not isinstance(item, dict) or item.get("rope_type", "default") != "default":
            raise ValueError("LFM3 supports only default RoPE")
        partial = _positive(item.get("partial_rotary_factor", factor), "partial_rotary_factor")
        dimension = head * partial
        if partial > 1 or not dimension.is_integer() or int(dimension) < 2 or int(dimension) % 2:
            raise ValueError("partial_rotary_factor must select an even rotary dimension")
        specs.append((int(dimension), _positive(item.get("rope_theta", theta), "rope_theta")))
    ids = {}
    for name, default in [("bos_token_id", 0), ("pad_token_id", 0)]:
        value = _integer(raw, name, default, minimum=0)
        if value >= fields["vocab_size"]:
            raise ValueError(f"{name} must index the vocabulary")
        ids[name] = value
    eos = raw.get("eos_token_id")
    values = eos if isinstance(eos, list) else [eos]
    if not values or any(type(v) is not int or not 0 <= v < fields["vocab_size"] for v in values):
        raise ValueError("eos_token_id must index the vocabulary")
    ids["eos_token_id"] = tuple(values) if isinstance(eos, list) else eos
    return DenseLfm3Config(
        **fields,
        head_dim=head,
        layer_types=tuple(layers),
        norm_eps=_positive(raw.get("norm_eps", 1e-5), "norm_eps"),
        conv_l_cache=_integer(raw, "conv_L_cache", 7),
        conv_bias=_boolean(raw, "conv_bias", False),
        tie_word_embeddings=_boolean(raw, "tie_word_embeddings", True),
        sliding_window=_integer(raw, "sliding_window", 256),
        use_sandwich_norm=_boolean(raw, "use_sandwich_norm", True),
        full_rope=specs[0],
        sliding_rope=specs[1],
        **ids,
    )
