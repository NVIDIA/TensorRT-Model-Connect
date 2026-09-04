# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT EAR-TTS step graph for NemotronLabs VoiceChat 11B.

The runtime contract deliberately contains no Python-framework seam.  Python is
used while building the engine only to read safetensors and tokenizer JSON.  At
runtime one TensorRT enqueue consumes the previous 31-code RVQ frame, the
current text token, compact conditional/unconditional KV caches, and explicit
random samples.  It produces the next 31-code frame and one compact K/V row per
decoder layer.

The graph is a literal port of the public checkpoint's ``RVQEARTTSModel``:

* one character-aware T5Gemma encoder layer for the current subword;
* gated audio/text fusion;
* a 28-layer Gemma3-text decoder with Q/K RMSNorm and four norms per layer;
* classifier-free guidance; and
* the eight-point EAR masking schedule followed by residual RVQ encoding.

TensorRT does not expose attention-logit soft-capping on ``IAttentionLayer``.
The character encoder therefore uses a narrow native-layer decomposition
(``MatMul -> tanh soft-cap -> Softmax -> MatMul``).  The 28-layer causal decoder
uses native ``IRotaryEmbeddingLayer`` and ``IAttentionLayer`` throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorrt as trt
from safetensors import safe_open


CHECKPOINT_PREFIX = "tts_model.tts_model."
NUM_REFINEMENT_STEPS = 8
FRAME_SECONDS = 0.08
TEXT_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
TEXT_MODEL_REVISION = "6533e8de2c68e4536bf7c411d7a3ce5734111476"


@dataclass(frozen=True)
class NativeTTSConfig:
    """Exact architecture encoded by the public VoiceChat 11B checkpoint."""

    hidden_size: int = 1152
    intermediate_size: int = 4608
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    latent_size: int = 512
    codebook_size: int = 1024
    num_quantizers: int = 31
    rms_norm_eps: float = 1.0e-6
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 7500
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 131072
    rope_theta: float = 1_000_000.0
    rope_local_theta: float = 10_000.0
    char_num_hidden_layers: int = 1
    char_vocab_size: int = 257
    char_rope_theta: float = 10_000.0
    char_attention_softcap: float = 50.0
    char_query_pre_attn_scalar: float = 256.0
    mog_num_layers: int = 3
    mog_num_predictions: int = 1024
    mog_low_rank: int = 64
    mog_min_log_std: float = -4.0
    guidance_scale: float = 0.2
    top_p: float = 0.95
    noise_scale: float = 0.001
    masking_exponent: float = 3.0

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        """Compact K/V width; never expand this to a different head layout."""
        return self.num_key_value_heads * self.head_dim

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(
            "full_attention" if (i + 1) % self.sliding_window_pattern == 0 else "sliding_attention"
            for i in range(self.num_hidden_layers)
        )

    @property
    def refinement_widths(self) -> tuple[int, ...]:
        rates = np.linspace(0.0, 1.0, NUM_REFINEMENT_STEPS + 1, dtype=np.float64)[:-1]
        masking_rates = np.power(
            1.0 - np.power(rates, self.masking_exponent), 1.0 / self.masking_exponent
        )
        num_maskings = np.ceil(masking_rates * self.num_quantizers).astype(np.int64)
        following = np.pad(num_maskings[1:], (0, 1))
        return tuple(int(x) for x in num_maskings - following)

    def validate(self) -> None:
        if self.attention_size != self.hidden_size:
            raise ValueError(
                "VoiceChat EAR-TTS requires num_attention_heads * head_dim "
                f"== hidden_size, got {self.attention_size} and {self.hidden_size}"
            )
        if self.kv_width != self.num_key_value_heads * self.head_dim:
            raise AssertionError("invalid compact K/V width")
        if len(self.layer_types) != self.num_hidden_layers:
            raise AssertionError("one attention type is required per decoder layer")
        if len(self.refinement_widths) != NUM_REFINEMENT_STEPS:
            raise AssertionError("EAR-TTS requires an eight-point masking schedule")
        if sum(self.refinement_widths) != self.num_quantizers:
            raise AssertionError("the refinement schedule must emit every RVQ codebook")


EXACT_CONFIG = NativeTTSConfig()
EXACT_CONFIG.validate()


@dataclass(frozen=True)
class SubwordTables:
    """Tokenizer-derived lookup tables consumed as TensorRT constants."""

    char_ids: np.ndarray
    char_lengths: np.ndarray
    char_padding_id: int
    vocab_size: int
    max_chars: int


class NativeTTSWeights(dict[str, np.ndarray]):
    """Checkpoint tensors keyed relative to :data:`CHECKPOINT_PREFIX`."""


def required_checkpoint_shapes(
    config: NativeTTSConfig = EXACT_CONFIG,
) -> dict[str, tuple[int, ...]]:
    """Return the complete, exact ``tts_model.tts_model.*`` tensor contract."""
    h = config.hidden_size
    m = config.intermediate_size
    d = config.head_dim
    q = config.num_quantizers
    v = config.codebook_size
    z = config.latent_size
    n = config.mog_num_predictions
    r = config.mog_low_rank

    shapes: dict[str, tuple[int, ...]] = {
        "audio_prompt_projection_W": (h, h),
        "backbone.norm.weight": (h,),
        "bos_emb": (h,),
        "embed_code.weight": (h, z),
        "embed_subword.backbone.encoder.norm.weight": (h,),
        "embed_subword.bos_eos_emb.pad_tensor": (),
        "embed_subword.bos_eos_emb.special_emb.weight": (3, h),
        "embed_subword.bos_eos_emb.special_flags": (131072,),
        "embed_subword.embed_tokens.weight": (config.char_vocab_size, h),
        "embed_subword.proj_embedding.weight": (h, h),
        "embed_subword.subword_flag_emb.cont_emb.weight": (2, h),
        "embed_subword.subword_flag_emb.is_continuation": (131073,),
        "embed_subword.subword_flag_emb.pad_tensor": (),
        "gated_fusion_audio_text.audio_proj.bias": (h,),
        "gated_fusion_audio_text.audio_proj.weight": (h, h),
        "gated_fusion_audio_text.final_norm.weight": (h,),
        "gated_fusion_audio_text.gate": (h,),
        "gated_fusion_audio_text.residual_scale": (),
        "gated_fusion_audio_text.text_proj.bias": (h,),
        "gated_fusion_audio_text.text_proj.weight": (h, h),
        "mog_head.low_mat": (n, z, r),
        "mog_head.mlp_stack.3.weight": (h,),
        "mog_head.proj_else.weight": (z, h),
        "mog_head.proj_logits.weight": (n, h),
        "mog_head.proj_logs.weight": (1, h),
        "mog_head.proj_mus.weight": (n * r, h),
        "null_emb": (h,),
        "rvq_embs": (q, v, z),
    }

    decoder_layer_shapes = {
        "input_layernorm.weight": (h,),
        "mlp.down_proj.weight": (h, m),
        "mlp.gate_proj.weight": (m, h),
        "mlp.up_proj.weight": (m, h),
        "post_attention_layernorm.weight": (h,),
        "post_feedforward_layernorm.weight": (h,),
        "pre_feedforward_layernorm.weight": (h,),
        "self_attn.k_norm.weight": (d,),
        "self_attn.k_proj.weight": (config.kv_width, h),
        "self_attn.o_proj.weight": (h, config.attention_size),
        "self_attn.q_norm.weight": (d,),
        "self_attn.q_proj.weight": (config.attention_size, h),
        "self_attn.v_proj.weight": (config.kv_width, h),
    }
    for layer_idx in range(config.num_hidden_layers):
        for suffix, shape in decoder_layer_shapes.items():
            shapes[f"backbone.layers.{layer_idx}.{suffix}"] = shape

    char_layer_shapes = {
        "mlp.down_proj.weight": (h, m),
        "mlp.gate_proj.weight": (m, h),
        "mlp.up_proj.weight": (m, h),
        "post_feedforward_layernorm.weight": (h,),
        "post_self_attn_layernorm.weight": (h,),
        "pre_feedforward_layernorm.weight": (h,),
        "pre_self_attn_layernorm.weight": (h,),
        "self_attn.k_proj.weight": (config.kv_width, h),
        "self_attn.o_proj.weight": (h, config.attention_size),
        "self_attn.q_proj.weight": (config.attention_size, h),
        "self_attn.v_proj.weight": (config.kv_width, h),
    }
    for suffix, shape in char_layer_shapes.items():
        shapes[f"embed_subword.backbone.encoder.layers.0.{suffix}"] = shape

    for layer_idx in range(config.mog_num_layers):
        prefix = f"mog_head.mlp_stack.{layer_idx}"
        shapes[f"{prefix}.mlp.down_proj.weight"] = (h, m)
        shapes[f"{prefix}.mlp.gate_proj.weight"] = (m, h)
        shapes[f"{prefix}.mlp.up_proj.weight"] = (m, h)
        shapes[f"{prefix}.post_norm.weight"] = (h,)
        shapes[f"{prefix}.pre_norm.weight"] = (h,)

    return shapes


def _resolve_safetensors(model_dir: str | Path) -> Path:
    path = Path(model_dir)
    if path.is_file():
        if path.name != "model.safetensors":
            raise ValueError(f"expected model.safetensors, got {path}")
        return path
    candidate = path / "model.safetensors"
    if not candidate.is_file():
        raise FileNotFoundError(f"VoiceChat model.safetensors not found in {path}")
    return candidate


def load_native_tts_weights(
    model_dir: str | Path,
    *,
    config: NativeTTSConfig = EXACT_CONFIG,
) -> NativeTTSWeights:
    """Load exactly the native EAR-TTS tensors without importing Torch or NeMo."""
    config.validate()
    required = required_checkpoint_shapes(config)
    weights = NativeTTSWeights()
    checkpoint = _resolve_safetensors(model_dir)

    with safe_open(str(checkpoint), framework="np") as reader:
        available = set(reader.keys())
        expected_full = {CHECKPOINT_PREFIX + key for key in required}
        observed_tts = {key for key in available if key.startswith(CHECKPOINT_PREFIX)}
        missing = sorted(expected_full - available)
        unexpected = sorted(observed_tts - expected_full)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing[:8]}" + ("..." if len(missing) > 8 else ""))
            if unexpected:
                details.append(
                    f"unexpected={unexpected[:8]}" + ("..." if len(unexpected) > 8 else "")
                )
            raise ValueError(
                "VoiceChat EAR-TTS checkpoint contract mismatch: " + "; ".join(details)
            )

        for relative_name, expected_shape in required.items():
            source_name = CHECKPOINT_PREFIX + relative_name
            array = np.asarray(reader.get_tensor(source_name))
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"{source_name} has shape {tuple(array.shape)}, expected {expected_shape}"
                )
            if array.dtype.kind in "iu":
                mapped = np.ascontiguousarray(array, dtype=np.int32)
            else:
                mapped = np.ascontiguousarray(array, dtype=np.float32)
            weights[relative_name] = mapped

    return weights


def _tokenizer_vocab(tokenizer_dir: str | Path) -> dict[str, int]:
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"pinned Nemotron tokenizer.json not found in {tokenizer_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab = payload.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"tokenizer model.vocab is missing or invalid in {path}")
    mapped = {str(token): int(token_id) for token, token_id in vocab.items()}
    ids = sorted(mapped.values())
    if ids != list(range(len(ids))):
        raise ValueError("VoiceChat tokenizer IDs must be dense from zero")
    return mapped


def build_subword_tables(
    tokenizer_dir: str | Path,
    *,
    expected_vocab_size: int = 131072,
    expected_char_vocab_size: int = 256,
) -> SubwordTables:
    """Recreate NeMo's character-aware subword mapping from tokenizer JSON."""
    vocab = _tokenizer_vocab(tokenizer_dir)
    if len(vocab) != expected_vocab_size:
        raise ValueError(
            f"VoiceChat tokenizer has {len(vocab)} entries, expected {expected_vocab_size}"
        )

    single_chars = {token: token_id for token, token_id in vocab.items() if len(token) == 1}
    sorted_chars = sorted(single_chars, key=single_chars.__getitem__)
    char_vocab = {char: index for index, char in enumerate(sorted_chars)}
    if len(char_vocab) != expected_char_vocab_size:
        raise ValueError(
            f"VoiceChat character vocabulary has {len(char_vocab)} entries, "
            f"expected {expected_char_vocab_size}"
        )

    encoded: list[tuple[int, ...]] = [()] * len(vocab)
    for token, token_id in vocab.items():
        encoded[token_id] = tuple(char_vocab[char] for char in token if char in char_vocab)
    if any(not chars for chars in encoded):
        raise ValueError("every VoiceChat subword must contain at least one checkpoint character")

    max_chars = max(len(chars) for chars in encoded)
    padding_id = len(char_vocab)
    char_ids = np.full((len(vocab), max_chars), padding_id, dtype=np.int32)
    char_lengths = np.empty((len(vocab),), dtype=np.int32)
    for token_id, chars in enumerate(encoded):
        char_lengths[token_id] = len(chars)
        char_ids[token_id, : len(chars)] = chars

    return SubwordTables(
        char_ids=np.ascontiguousarray(char_ids),
        char_lengths=np.ascontiguousarray(char_lengths),
        char_padding_id=padding_id,
        vocab_size=len(vocab),
        max_chars=max_chars,
    )


@dataclass
class _GraphContext:
    network: Any
    trt: Any
    weights: NativeTTSWeights
    work_trt_dtype: Any
    work_np_dtype: Any
    constants: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def constant(
        self,
        key: tuple[Any, ...],
        values: np.ndarray | float | int,
        *,
        dtype: Any | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> Any:
        cached = self.constants.get(key)
        if cached is not None:
            return cached
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        if shape is not None:
            array = np.ascontiguousarray(array.reshape(shape))
        output = self.network.add_constant(tuple(array.shape), self.trt.Weights(array)).get_output(
            0
        )
        self.constants[key] = output
        return output

    def work_constant(
        self,
        key: tuple[Any, ...],
        values: np.ndarray | float,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Any:
        tensor = self.constant(key, values, dtype=self.work_np_dtype, shape=shape)
        if tensor.dtype != self.work_trt_dtype:
            tensor = self.network.add_cast(tensor, self.work_trt_dtype).get_output(0)
        return tensor


def _cast(ctx: _GraphContext, tensor: Any, dtype: Any) -> Any:
    if tensor.dtype == dtype:
        return tensor
    return ctx.network.add_cast(tensor, dtype).get_output(0)


def _shuffle(
    ctx: _GraphContext,
    tensor: Any,
    shape: tuple[int, ...],
    permutation: tuple[int, ...] | None = None,
) -> Any:
    layer = ctx.network.add_shuffle(tensor)
    if permutation is not None:
        layer.first_transpose = ctx.trt.Permutation(list(permutation))
    layer.reshape_dims = shape
    return layer.get_output(0)


def _linear(ctx: _GraphContext, tensor: Any, weight_name: str, bias_name: str | None = None) -> Any:
    weight = ctx.weights[weight_name]
    if weight.ndim != 2:
        raise ValueError(f"linear weight {weight_name} must be rank two")
    out_size, in_size = weight.shape
    rank = len(tuple(tensor.shape))
    rhs_shape = (1,) * max(rank - 2, 0) + (in_size, out_size)
    rhs = ctx.work_constant(
        ("linear", weight_name, rhs_shape),
        weight.T,
        shape=rhs_shape,
    )
    output = ctx.network.add_matrix_multiply(
        tensor,
        ctx.trt.MatrixOperation.NONE,
        rhs,
        ctx.trt.MatrixOperation.NONE,
    ).get_output(0)
    output = _cast(ctx, output, ctx.work_trt_dtype)
    if bias_name is not None:
        bias = ctx.weights[bias_name]
        bias_shape = (1,) * (rank - 1) + (out_size,)
        bias_tensor = ctx.work_constant(
            ("bias", bias_name, bias_shape),
            bias,
            shape=bias_shape,
        )
        output = ctx.network.add_elementwise(
            output, bias_tensor, ctx.trt.ElementWiseOperation.SUM
        ).get_output(0)
    return output


def _rms_norm(ctx: _GraphContext, tensor: Any, weight_name: str, eps: float) -> Any:
    """Gemma RMSNorm in FP32; TensorRT normalization is LayerNorm, not RMSNorm."""
    output_dtype = tensor.dtype
    work = _cast(ctx, tensor, ctx.trt.float32)
    rank = len(tuple(work.shape))
    axis = 1 << (rank - 1)
    squared = ctx.network.add_elementwise(work, work, ctx.trt.ElementWiseOperation.PROD).get_output(
        0
    )
    mean = ctx.network.add_reduce(squared, ctx.trt.ReduceOperation.AVG, axis, True).get_output(0)
    eps_tensor = ctx.constant(
        ("rms_eps", rank, eps),
        np.array(eps, dtype=np.float32),
        dtype=np.float32,
        shape=(1,) * rank,
    )
    denominator = ctx.network.add_elementwise(
        mean, eps_tensor, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)
    denominator = ctx.network.add_unary(denominator, ctx.trt.UnaryOperation.SQRT).get_output(0)
    reciprocal = ctx.network.add_unary(denominator, ctx.trt.UnaryOperation.RECIP).get_output(0)
    normalized = ctx.network.add_elementwise(
        work, reciprocal, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    gamma = np.asarray(ctx.weights[weight_name], dtype=np.float32) + 1.0
    gamma_shape = (1,) * (rank - 1) + (gamma.size,)
    gamma_tensor = ctx.constant(
        ("rms_gamma", weight_name, rank), gamma, dtype=np.float32, shape=gamma_shape
    )
    output = ctx.network.add_elementwise(
        normalized, gamma_tensor, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    return _cast(ctx, output, output_dtype)


def _gelu_tanh(ctx: _GraphContext, tensor: Any) -> Any:
    output_dtype = tensor.dtype
    work = _cast(ctx, tensor, ctx.trt.float32)
    output = ctx.network.add_activation(work, ctx.trt.ActivationType.GELU_TANH).get_output(0)
    return _cast(ctx, output, output_dtype)


def _gated_mlp(ctx: _GraphContext, tensor: Any, prefix: str) -> Any:
    gate = _linear(ctx, tensor, f"{prefix}.gate_proj.weight")
    up = _linear(ctx, tensor, f"{prefix}.up_proj.weight")
    activated = _gelu_tanh(ctx, gate)
    product = ctx.network.add_elementwise(
        activated, up, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    return _linear(ctx, product, f"{prefix}.down_proj.weight")


def _rope_table(length: int, head_dim: int, theta: float, *, cosine: bool) -> np.ndarray:
    half = head_dim // 2
    positions = np.arange(length, dtype=np.float32)[:, None]
    dimensions = np.arange(half, dtype=np.float32)[None, :]
    frequencies = np.power(np.float32(theta), -(2.0 * dimensions) / float(head_dim))
    angles = positions * frequencies
    return np.cos(angles).astype(np.float32) if cosine else np.sin(angles).astype(np.float32)


def _apply_rope(
    ctx: _GraphContext,
    tensor: Any,
    position_ids: Any,
    *,
    table_name: str,
    table_length: int,
    head_dim: int,
    theta: float,
) -> Any:
    cos = ctx.work_constant(
        ("rope_cos", table_name, table_length, head_dim, theta),
        _rope_table(table_length, head_dim, theta, cosine=True),
    )
    sin = ctx.work_constant(
        ("rope_sin", table_name, table_length, head_dim, theta),
        _rope_table(table_length, head_dim, theta, cosine=False),
    )
    layer = ctx.network.add_rotary_embedding(tensor, cos, sin, False, head_dim)
    layer.set_input(3, position_ids)
    return layer.get_output(0)


def _rows_to_heads(ctx: _GraphContext, tensor: Any, batch: int, heads: int, head_dim: int) -> Any:
    reshaped = _shuffle(ctx, tensor, (batch, -1, heads, head_dim))
    return _shuffle(ctx, reshaped, (batch, heads, -1, head_dim), (0, 2, 1, 3))


def _heads_to_rows(ctx: _GraphContext, tensor: Any, batch: int, width: int) -> Any:
    transposed = _shuffle(ctx, tensor, (batch, -1, width), (0, 2, 1, 3))
    return transposed


def _native_attention(ctx: _GraphContext, q: Any, k: Any, v: Any, mask: Any, scale: float) -> Any:
    scale_tensor = ctx.work_constant(
        ("attention_scale", scale),
        np.array(scale, dtype=ctx.work_np_dtype),
        shape=(1, 1, 1, 1),
    )
    scaled_q = ctx.network.add_elementwise(
        q, scale_tensor, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    layer = ctx.network.add_attention(
        scaled_q,
        k,
        v,
        ctx.trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    layer.decomposable = True
    layer.mask = mask
    return layer.get_output(0)


def _char_attention(
    ctx: _GraphContext, q: Any, k: Any, v: Any, mask: Any, config: NativeTTSConfig
) -> Any:
    """Exact T5Gemma soft-capped attention using only native TRT primitives."""
    q_fp32 = _cast(ctx, q, ctx.trt.float32)
    k_fp32 = _cast(ctx, k, ctx.trt.float32)
    scores = ctx.network.add_matrix_multiply(
        q_fp32,
        ctx.trt.MatrixOperation.NONE,
        k_fp32,
        ctx.trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    scale = 1.0 / math.sqrt(config.char_query_pre_attn_scalar)
    scale_tensor = ctx.constant(
        ("char_attention_scale",),
        np.array(scale, dtype=np.float32),
        dtype=np.float32,
        shape=(1, 1, 1, 1),
    )
    scores = ctx.network.add_elementwise(
        scores, scale_tensor, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    cap = ctx.constant(
        ("char_attention_cap",),
        np.array(config.char_attention_softcap, dtype=np.float32),
        dtype=np.float32,
        shape=(1, 1, 1, 1),
    )
    scores = ctx.network.add_elementwise(scores, cap, ctx.trt.ElementWiseOperation.DIV).get_output(
        0
    )
    scores = ctx.network.add_activation(scores, ctx.trt.ActivationType.TANH).get_output(0)
    scores = ctx.network.add_elementwise(scores, cap, ctx.trt.ElementWiseOperation.PROD).get_output(
        0
    )
    scores = ctx.network.add_elementwise(
        scores, _cast(ctx, mask, ctx.trt.float32), ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)
    softmax = ctx.network.add_softmax(scores)
    softmax.axes = 1 << 3
    probabilities = _cast(ctx, softmax.get_output(0), v.dtype)
    return ctx.network.add_matrix_multiply(
        probabilities,
        ctx.trt.MatrixOperation.NONE,
        v,
        ctx.trt.MatrixOperation.NONE,
    ).get_output(0)


def _add_char_encoder(
    ctx: _GraphContext,
    subword_id: Any,
    subword_mask: Any,
    tables: SubwordTables,
    config: NativeTTSConfig,
) -> Any:
    ids_table = ctx.constant(("subword_char_ids",), tables.char_ids, dtype=np.int32)
    lengths_table = ctx.constant(("subword_char_lengths",), tables.char_lengths, dtype=np.int32)
    char_ids = ctx.network.add_gather(ids_table, subword_id, 0).get_output(0)
    char_length = ctx.network.add_gather(lengths_table, subword_id, 0).get_output(0)

    char_embedding = ctx.work_constant(
        ("weight", "embed_subword.embed_tokens.weight"),
        ctx.weights["embed_subword.embed_tokens.weight"],
    )
    hidden = ctx.network.add_gather(char_embedding, char_ids, 0).get_output(0)
    hidden = _shuffle(ctx, hidden, (1, tables.max_chars, config.hidden_size))
    embed_scale = ctx.work_constant(
        ("char_embed_scale",),
        np.array(math.sqrt(config.hidden_size), dtype=ctx.work_np_dtype),
        shape=(1, 1, 1),
    )
    hidden = ctx.network.add_elementwise(
        hidden, embed_scale, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)

    positions_np = np.arange(tables.max_chars, dtype=np.int32).reshape(1, -1)
    positions = ctx.constant(("char_positions", tables.max_chars), positions_np, dtype=np.int32)
    length_2d = _shuffle(ctx, char_length, (1, 1))
    valid = ctx.network.add_elementwise(
        positions, length_2d, ctx.trt.ElementWiseOperation.LESS
    ).get_output(0)
    valid_fp32 = _cast(ctx, valid, ctx.trt.float32)
    one = ctx.constant(
        ("char_mask_one",), np.array(1.0, dtype=np.float32), dtype=np.float32, shape=(1, 1)
    )
    invalid = ctx.network.add_elementwise(
        one, valid_fp32, ctx.trt.ElementWiseOperation.SUB
    ).get_output(0)
    neg = ctx.constant(
        ("char_mask_neg",), np.array(-1.0e4, dtype=np.float32), dtype=np.float32, shape=(1, 1)
    )
    additive_mask = ctx.network.add_elementwise(
        invalid, neg, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    additive_mask = _shuffle(ctx, additive_mask, (1, 1, 1, tables.max_chars))

    prefix = "embed_subword.backbone.encoder.layers.0"
    residual = hidden
    normed = _rms_norm(ctx, hidden, f"{prefix}.pre_self_attn_layernorm.weight", config.rms_norm_eps)
    q = _linear(ctx, normed, f"{prefix}.self_attn.q_proj.weight")
    k = _linear(ctx, normed, f"{prefix}.self_attn.k_proj.weight")
    v = _linear(ctx, normed, f"{prefix}.self_attn.v_proj.weight")
    q = _rows_to_heads(ctx, q, 1, config.num_attention_heads, config.head_dim)
    k = _rows_to_heads(ctx, k, 1, config.num_key_value_heads, config.head_dim)
    v = _rows_to_heads(ctx, v, 1, config.num_key_value_heads, config.head_dim)
    q = _apply_rope(
        ctx,
        q,
        positions,
        table_name="char",
        table_length=tables.max_chars,
        head_dim=config.head_dim,
        theta=config.char_rope_theta,
    )
    k = _apply_rope(
        ctx,
        k,
        positions,
        table_name="char",
        table_length=tables.max_chars,
        head_dim=config.head_dim,
        theta=config.char_rope_theta,
    )
    attention = _char_attention(ctx, q, k, v, additive_mask, config)
    attention = _heads_to_rows(ctx, attention, 1, config.attention_size)
    attention = _linear(ctx, attention, f"{prefix}.self_attn.o_proj.weight")
    attention = _rms_norm(
        ctx, attention, f"{prefix}.post_self_attn_layernorm.weight", config.rms_norm_eps
    )
    hidden = ctx.network.add_elementwise(
        residual, attention, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)

    residual = hidden
    hidden = _rms_norm(
        ctx, hidden, f"{prefix}.pre_feedforward_layernorm.weight", config.rms_norm_eps
    )
    hidden = _gated_mlp(ctx, hidden, f"{prefix}.mlp")
    hidden = _rms_norm(
        ctx, hidden, f"{prefix}.post_feedforward_layernorm.weight", config.rms_norm_eps
    )
    hidden = ctx.network.add_elementwise(
        residual, hidden, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)
    hidden = _rms_norm(
        ctx, hidden, "embed_subword.backbone.encoder.norm.weight", config.rms_norm_eps
    )

    valid_work = _cast(ctx, valid, ctx.work_trt_dtype)
    valid_work = _shuffle(ctx, valid_work, (1, tables.max_chars, 1))
    masked = ctx.network.add_elementwise(
        hidden, valid_work, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    summed = ctx.network.add_reduce(masked, ctx.trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    denominator = _cast(ctx, char_length, ctx.work_trt_dtype)
    denominator = _shuffle(ctx, denominator, (1, 1, 1))
    pooled = ctx.network.add_elementwise(
        summed, denominator, ctx.trt.ElementWiseOperation.DIV
    ).get_output(0)
    text = _linear(ctx, pooled, "embed_subword.proj_embedding.weight")
    enabled = _cast(ctx, subword_mask, ctx.work_trt_dtype)
    enabled = _shuffle(ctx, enabled, (1, 1, 1))
    text = ctx.network.add_elementwise(text, enabled, ctx.trt.ElementWiseOperation.PROD).get_output(
        0
    )

    # NeMo masks only the character-encoder scatter.  Its learned continuation
    # and BOS/EOS flag embeddings are added afterwards, including at the
    # mask-false PAD positions in the batched Aria warmup.
    continuation_table = ctx.constant(
        ("weight", "embed_subword.subword_flag_emb.is_continuation"),
        ctx.weights["embed_subword.subword_flag_emb.is_continuation"],
        dtype=np.int32,
    )
    continuation_id = ctx.network.add_gather(continuation_table, subword_id, 0).get_output(0)
    continuation_embedding = ctx.work_constant(
        ("weight", "embed_subword.subword_flag_emb.cont_emb.weight"),
        ctx.weights["embed_subword.subword_flag_emb.cont_emb.weight"],
    )
    continuation = ctx.network.add_gather(continuation_embedding, continuation_id, 0).get_output(0)
    continuation = _shuffle(ctx, continuation, (1, 1, config.hidden_size))
    text = ctx.network.add_elementwise(
        text, continuation, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)

    special_table = ctx.constant(
        ("weight", "embed_subword.bos_eos_emb.special_flags"),
        ctx.weights["embed_subword.bos_eos_emb.special_flags"],
        dtype=np.int32,
    )
    special_id = ctx.network.add_gather(special_table, subword_id, 0).get_output(0)
    special_embedding = ctx.work_constant(
        ("weight", "embed_subword.bos_eos_emb.special_emb.weight"),
        ctx.weights["embed_subword.bos_eos_emb.special_emb.weight"],
    )
    special = ctx.network.add_gather(special_embedding, special_id, 0).get_output(0)
    special = _shuffle(ctx, special, (1, 1, config.hidden_size))
    text = ctx.network.add_elementwise(text, special, ctx.trt.ElementWiseOperation.SUM).get_output(
        0
    )
    return text


def _depthsum_embedding(
    ctx: _GraphContext,
    codes: Any,
    config: NativeTTSConfig,
) -> Any:
    rvq = ctx.weights["rvq_embs"]
    total = ctx.work_constant(
        ("depthsum_zero",),
        np.zeros((1, 1, config.latent_size), dtype=ctx.work_np_dtype),
    )
    padding = np.zeros((1, config.latent_size), dtype=rvq.dtype)
    for codebook in range(config.num_quantizers):
        table_np = np.concatenate((rvq[codebook], padding), axis=0)
        table = ctx.work_constant(("rvq_table", codebook), table_np)
        code = ctx.network.add_slice(codes, (codebook,), (1,), (1,)).get_output(0)
        embedding = ctx.network.add_gather(table, code, 0).get_output(0)
        embedding = _shuffle(ctx, embedding, (1, 1, config.latent_size))
        total = ctx.network.add_elementwise(
            total, embedding, ctx.trt.ElementWiseOperation.SUM
        ).get_output(0)
    return total


def _audio_conditioning(
    ctx: _GraphContext,
    prev_codes: Any,
    audio_prompt_latent: Any,
    audio_prompt_mode: Any,
    bos_flag: Any,
    config: NativeTTSConfig,
) -> Any:
    latent = _depthsum_embedding(ctx, prev_codes, config)
    normal = _linear(ctx, latent, "embed_code.weight")

    # The public checkpoint carries a pre-baked Aria prompt latent.  Keeping it
    # as an explicit input avoids a Python/codec-encoder dependency at runtime
    # and permits a native warmup to stream one prompt row per cache step.
    prompt = _cast(ctx, audio_prompt_latent, ctx.work_trt_dtype)
    prompt = _shuffle(ctx, prompt, (1, 1, config.hidden_size))
    prompt_enabled = _cast(ctx, audio_prompt_mode, ctx.work_trt_dtype)
    prompt_enabled = _shuffle(ctx, prompt_enabled, (1, 1, 1))
    one = ctx.work_constant(("one_3d",), np.array(1.0, dtype=ctx.work_np_dtype), shape=(1, 1, 1))
    normal_enabled = ctx.network.add_elementwise(
        one, prompt_enabled, ctx.trt.ElementWiseOperation.SUB
    ).get_output(0)
    normal = ctx.network.add_elementwise(
        normal, normal_enabled, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    prompt = ctx.network.add_elementwise(
        prompt, prompt_enabled, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    audio = ctx.network.add_elementwise(
        normal, prompt, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)

    bos = ctx.work_constant(
        ("weight", "bos_emb"), ctx.weights["bos_emb"], shape=(1, 1, config.hidden_size)
    )
    bos_enabled = _cast(ctx, bos_flag, ctx.work_trt_dtype)
    bos_enabled = _shuffle(ctx, bos_enabled, (1, 1, 1))
    bos = ctx.network.add_elementwise(
        bos, bos_enabled, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    return ctx.network.add_elementwise(audio, bos, ctx.trt.ElementWiseOperation.SUM).get_output(0)


def _gated_fusion(
    ctx: _GraphContext,
    audio: Any,
    text: Any,
    config: NativeTTSConfig,
) -> Any:
    divisor = ctx.work_constant(
        ("num_quantizers",),
        np.array(float(config.num_quantizers), dtype=ctx.work_np_dtype),
        shape=(1, 1, 1),
    )
    audio = ctx.network.add_elementwise(
        audio, divisor, ctx.trt.ElementWiseOperation.DIV
    ).get_output(0)
    audio = _linear(
        ctx,
        audio,
        "gated_fusion_audio_text.audio_proj.weight",
        "gated_fusion_audio_text.audio_proj.bias",
    )
    text = _linear(
        ctx,
        text,
        "gated_fusion_audio_text.text_proj.weight",
        "gated_fusion_audio_text.text_proj.bias",
    )

    gate = ctx.constant(
        ("fusion_gate",),
        ctx.weights["gated_fusion_audio_text.gate"],
        dtype=np.float32,
        shape=(1, 1, config.hidden_size),
    )
    gate = ctx.network.add_activation(gate, ctx.trt.ActivationType.SIGMOID).get_output(0)
    gate = _cast(ctx, gate, ctx.work_trt_dtype)
    one = ctx.work_constant(
        ("fusion_one",), np.array(1.0, dtype=ctx.work_np_dtype), shape=(1, 1, 1)
    )
    inverse_gate = ctx.network.add_elementwise(
        one, gate, ctx.trt.ElementWiseOperation.SUB
    ).get_output(0)
    audio = ctx.network.add_elementwise(audio, gate, ctx.trt.ElementWiseOperation.PROD).get_output(
        0
    )
    text = ctx.network.add_elementwise(
        text, inverse_gate, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    fused = ctx.network.add_elementwise(audio, text, ctx.trt.ElementWiseOperation.SUM).get_output(0)

    residual_scale = ctx.constant(
        ("fusion_residual_scale",),
        ctx.weights["gated_fusion_audio_text.residual_scale"],
        dtype=np.float32,
        shape=(1, 1, 1),
    )
    residual_scale = ctx.network.add_activation(
        residual_scale, ctx.trt.ActivationType.SIGMOID
    ).get_output(0)
    residual_scale = _cast(ctx, residual_scale, ctx.work_trt_dtype)
    fused = ctx.network.add_elementwise(
        fused, residual_scale, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    return _rms_norm(ctx, fused, "gated_fusion_audio_text.final_norm.weight", config.rms_norm_eps)


def _duplicate_cfg_conditioning(
    ctx: _GraphContext, audio: Any, text: Any, config: NativeTTSConfig
) -> Any:
    null = ctx.work_constant(
        ("weight", "null_emb"),
        ctx.weights["null_emb"],
        shape=(1, 1, config.hidden_size),
    )
    conditional = _gated_fusion(ctx, audio, text, config)
    unconditional = _gated_fusion(ctx, audio, null, config)
    concat = ctx.network.add_concatenation([conditional, unconditional])
    concat.axis = 0
    return concat.get_output(0)


def _decoder_layer(
    ctx: _GraphContext,
    hidden: Any,
    cache_k: Any,
    cache_v: Any,
    position_ids: Any,
    attention_mask: Any,
    layer_idx: int,
    config: NativeTTSConfig,
) -> tuple[Any, Any, Any]:
    prefix = f"backbone.layers.{layer_idx}"
    residual = hidden
    normed = _rms_norm(ctx, hidden, f"{prefix}.input_layernorm.weight", config.rms_norm_eps)
    q = _linear(ctx, normed, f"{prefix}.self_attn.q_proj.weight")
    k_rows = _linear(ctx, normed, f"{prefix}.self_attn.k_proj.weight")
    v_rows = _linear(ctx, normed, f"{prefix}.self_attn.v_proj.weight")
    q = _rows_to_heads(ctx, q, 2, config.num_attention_heads, config.head_dim)
    k = _rows_to_heads(ctx, k_rows, 2, config.num_key_value_heads, config.head_dim)
    v = _rows_to_heads(ctx, v_rows, 2, config.num_key_value_heads, config.head_dim)
    q = _rms_norm(ctx, q, f"{prefix}.self_attn.q_norm.weight", config.rms_norm_eps)
    k = _rms_norm(ctx, k, f"{prefix}.self_attn.k_norm.weight", config.rms_norm_eps)

    local = config.layer_types[layer_idx] == "sliding_attention"
    theta = config.rope_local_theta if local else config.rope_theta
    table_name = "decoder_local" if local else "decoder_global"
    q = _apply_rope(
        ctx,
        q,
        position_ids,
        table_name=table_name,
        table_length=config.max_position_embeddings,
        head_dim=config.head_dim,
        theta=theta,
    )
    k = _apply_rope(
        ctx,
        k,
        position_ids,
        table_name=table_name,
        table_length=config.max_position_embeddings,
        head_dim=config.head_dim,
        theta=theta,
    )
    present_k = _heads_to_rows(ctx, k, 2, config.kv_width)
    present_v = v_rows

    cached_k = _rows_to_heads(ctx, cache_k, 2, config.num_key_value_heads, config.head_dim)
    cached_v = _rows_to_heads(ctx, cache_v, 2, config.num_key_value_heads, config.head_dim)
    all_k_layer = ctx.network.add_concatenation([cached_k, k])
    all_k_layer.axis = 2
    all_v_layer = ctx.network.add_concatenation([cached_v, v])
    all_v_layer.axis = 2
    attention = _native_attention(
        ctx,
        q,
        all_k_layer.get_output(0),
        all_v_layer.get_output(0),
        attention_mask,
        1.0 / math.sqrt(config.query_pre_attn_scalar),
    )
    attention = _heads_to_rows(ctx, attention, 2, config.attention_size)
    attention = _linear(ctx, attention, f"{prefix}.self_attn.o_proj.weight")
    attention = _rms_norm(
        ctx, attention, f"{prefix}.post_attention_layernorm.weight", config.rms_norm_eps
    )
    hidden = ctx.network.add_elementwise(
        residual, attention, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)

    residual = hidden
    hidden = _rms_norm(
        ctx, hidden, f"{prefix}.pre_feedforward_layernorm.weight", config.rms_norm_eps
    )
    hidden = _gated_mlp(ctx, hidden, f"{prefix}.mlp")
    hidden = _rms_norm(
        ctx, hidden, f"{prefix}.post_feedforward_layernorm.weight", config.rms_norm_eps
    )
    hidden = ctx.network.add_elementwise(
        residual, hidden, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)
    return hidden, present_k, present_v


def _mog_stack(ctx: _GraphContext, hidden: Any, config: NativeTTSConfig) -> Any:
    for layer_idx in range(config.mog_num_layers):
        prefix = f"mog_head.mlp_stack.{layer_idx}"
        residual = hidden
        hidden = _rms_norm(ctx, hidden, f"{prefix}.pre_norm.weight", config.rms_norm_eps)
        hidden = _gated_mlp(ctx, hidden, f"{prefix}.mlp")
        hidden = _rms_norm(ctx, hidden, f"{prefix}.post_norm.weight", config.rms_norm_eps)
        hidden = ctx.network.add_elementwise(
            residual, hidden, ctx.trt.ElementWiseOperation.SUM
        ).get_output(0)
    return _rms_norm(ctx, hidden, "mog_head.mlp_stack.3.weight", config.rms_norm_eps)


def _slice_refinement_input(ctx: _GraphContext, tensor: Any, step: int, width: int) -> Any:
    return ctx.network.add_slice(tensor, (step, 0), (1, width), (1, 1)).get_output(0)


def _gumbel_from_uniform(ctx: _GraphContext, uniform: Any) -> Any:
    eps = ctx.constant(
        ("gumbel_eps",), np.array(1.0e-8, dtype=np.float32), dtype=np.float32, shape=(1, 1)
    )
    value = ctx.network.add_elementwise(uniform, eps, ctx.trt.ElementWiseOperation.SUM).get_output(
        0
    )
    value = ctx.network.add_unary(value, ctx.trt.UnaryOperation.LOG).get_output(0)
    minus_one = ctx.constant(
        ("minus_one_2d",), np.array(-1.0, dtype=np.float32), dtype=np.float32, shape=(1, 1)
    )
    value = ctx.network.add_elementwise(
        value, minus_one, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    value = ctx.network.add_elementwise(value, eps, ctx.trt.ElementWiseOperation.SUM).get_output(0)
    value = ctx.network.add_unary(value, ctx.trt.UnaryOperation.LOG).get_output(0)
    return ctx.network.add_elementwise(
        value, minus_one, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)


def _sample_mixture(
    ctx: _GraphContext,
    logits: Any,
    uniform: Any,
    config: NativeTTSConfig,
) -> Any:
    """Top-p filter and Gumbel-max sampling, matching TopPLogitsWarper."""
    logits = _cast(ctx, logits, ctx.trt.float32)
    logits = _shuffle(ctx, logits, (1, config.mog_num_predictions))
    sorted_layer = ctx.network.add_topk(
        logits,
        ctx.trt.TopKOperation.MAX,
        config.mog_num_predictions,
        1 << 1,
    )
    sorted_logits = sorted_layer.get_output(0)
    sorted_indices = sorted_layer.get_output(1)
    softmax = ctx.network.add_softmax(sorted_logits)
    softmax.axes = 1 << 1
    probabilities = softmax.get_output(0)

    # Descending equivalent of HF's ascending-tail TopPLogitsWarper:
    # retain an item iff cumulative probability before that item is < top_p.
    cumulative_matrix = np.triu(
        np.ones((config.mog_num_predictions, config.mog_num_predictions), dtype=np.float32)
    )
    cumulative_matrix = cumulative_matrix.reshape(
        1, config.mog_num_predictions, config.mog_num_predictions
    )
    cumulative_tensor = ctx.constant(
        ("top_p_cumulative", config.mog_num_predictions),
        cumulative_matrix,
        dtype=np.float32,
    )
    probabilities_3d = _shuffle(ctx, probabilities, (1, 1, config.mog_num_predictions))
    cumulative = ctx.network.add_matrix_multiply(
        probabilities_3d,
        ctx.trt.MatrixOperation.NONE,
        cumulative_tensor,
        ctx.trt.MatrixOperation.NONE,
    ).get_output(0)
    cumulative = _shuffle(ctx, cumulative, (1, config.mog_num_predictions))
    before = ctx.network.add_elementwise(
        cumulative, probabilities, ctx.trt.ElementWiseOperation.SUB
    ).get_output(0)
    top_p = ctx.constant(
        ("top_p",), np.array(config.top_p, dtype=np.float32), dtype=np.float32, shape=(1, 1)
    )
    keep = ctx.network.add_elementwise(before, top_p, ctx.trt.ElementWiseOperation.LESS).get_output(
        0
    )
    filtered_value = ctx.constant(
        ("top_p_filter",), np.array(-1.0e9, dtype=np.float32), dtype=np.float32, shape=(1, 1)
    )
    filtered = ctx.network.add_select(keep, sorted_logits, filtered_value).get_output(0)

    gumbel = _gumbel_from_uniform(ctx, uniform)
    flat_sorted_indices = _shuffle(ctx, sorted_indices, (config.mog_num_predictions,))
    sorted_gumbel = ctx.network.add_gather(gumbel, flat_sorted_indices, 1).get_output(0)
    scores = ctx.network.add_elementwise(
        filtered, sorted_gumbel, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)
    winner = ctx.network.add_topk(scores, ctx.trt.TopKOperation.MAX, 1, 1 << 1).get_output(1)
    mixture = ctx.network.add_gather(sorted_indices, winner, 1).get_output(0)
    return _shuffle(ctx, mixture, (1,))


def _selected_mog_projection(
    ctx: _GraphContext,
    hidden: Any,
    mixture: Any,
    config: NativeTTSConfig,
) -> Any:
    low_rank = config.mog_low_rank
    mus = ctx.work_constant(
        ("mog_proj_mus",),
        ctx.weights["mog_head.proj_mus.weight"],
        shape=(config.mog_num_predictions, low_rank, config.hidden_size),
    )
    selected_mus = ctx.network.add_gather(mus, mixture, 0).get_output(0)
    selected_mus = _shuffle(ctx, selected_mus, (1, low_rank, config.hidden_size))
    low_mu = ctx.network.add_matrix_multiply(
        hidden,
        ctx.trt.MatrixOperation.NONE,
        selected_mus,
        ctx.trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)

    low_mats = ctx.work_constant(("mog_low_mat",), ctx.weights["mog_head.low_mat"])
    selected_low_mat = ctx.network.add_gather(low_mats, mixture, 0).get_output(0)
    selected_low_mat = _shuffle(ctx, selected_low_mat, (1, config.latent_size, config.mog_low_rank))
    return ctx.network.add_matrix_multiply(
        low_mu,
        ctx.trt.MatrixOperation.NONE,
        selected_low_mat,
        ctx.trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


def _mog_sample(
    ctx: _GraphContext,
    conditional_hidden: Any,
    unconditional_hidden: Any,
    quantized: Any,
    uniform: Any,
    noise: Any,
    config: NativeTTSConfig,
) -> Any:
    code_embed = _linear(ctx, quantized, "embed_code.weight")
    conditional = ctx.network.add_elementwise(
        code_embed, conditional_hidden, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)
    unconditional = ctx.network.add_elementwise(
        code_embed, unconditional_hidden, ctx.trt.ElementWiseOperation.SUM
    ).get_output(0)
    paired = ctx.network.add_concatenation([conditional, unconditional])
    paired.axis = 0
    paired = _mog_stack(ctx, paired.get_output(0), config)

    cond = ctx.network.add_slice(
        paired, (0, 0, 0), (1, 1, config.hidden_size), (1, 1, 1)
    ).get_output(0)
    uncond = ctx.network.add_slice(
        paired, (1, 0, 0), (1, 1, config.hidden_size), (1, 1, 1)
    ).get_output(0)
    delta = ctx.network.add_elementwise(cond, uncond, ctx.trt.ElementWiseOperation.SUB).get_output(
        0
    )
    guidance = ctx.work_constant(
        ("guidance_scale",),
        np.array(config.guidance_scale, dtype=ctx.work_np_dtype),
        shape=(1, 1, 1),
    )
    delta = ctx.network.add_elementwise(
        delta, guidance, ctx.trt.ElementWiseOperation.PROD
    ).get_output(0)
    guided = ctx.network.add_elementwise(cond, delta, ctx.trt.ElementWiseOperation.SUM).get_output(
        0
    )

    logits = _linear(ctx, guided, "mog_head.proj_logits.weight")
    mixture = _sample_mixture(ctx, logits, uniform, config)
    mu = _selected_mog_projection(ctx, guided, mixture, config)
    mu_residual = _linear(ctx, guided, "mog_head.proj_else.weight")
    logs = _linear(ctx, guided, "mog_head.proj_logs.weight")
    minimum = ctx.work_constant(
        ("mog_min_log_std",),
        np.array(config.mog_min_log_std, dtype=ctx.work_np_dtype),
        shape=(1, 1, 1),
    )
    logs = ctx.network.add_elementwise(logs, minimum, ctx.trt.ElementWiseOperation.MAX).get_output(
        0
    )
    std = ctx.network.add_unary(logs, ctx.trt.UnaryOperation.EXP).get_output(0)
    mu = ctx.network.add_elementwise(mu, std, ctx.trt.ElementWiseOperation.PROD).get_output(0)
    mu = ctx.network.add_elementwise(mu, mu_residual, ctx.trt.ElementWiseOperation.SUM).get_output(
        0
    )

    noise = _cast(ctx, noise, ctx.work_trt_dtype)
    noise = _shuffle(ctx, noise, (1, 1, config.latent_size))
    scale = ctx.work_constant(
        ("mog_noise_scale",),
        np.array(config.noise_scale, dtype=ctx.work_np_dtype),
        shape=(1, 1, 1),
    )
    noise = ctx.network.add_elementwise(noise, scale, ctx.trt.ElementWiseOperation.PROD).get_output(
        0
    )
    noise = ctx.network.add_elementwise(noise, std, ctx.trt.ElementWiseOperation.PROD).get_output(0)
    return ctx.network.add_elementwise(mu, noise, ctx.trt.ElementWiseOperation.SUM).get_output(0)


def _rvq_refinement(
    ctx: _GraphContext,
    decoder_hidden: Any,
    mixture_uniform: Any,
    mog_noise: Any,
    config: NativeTTSConfig,
) -> Any:
    conditional = ctx.network.add_slice(
        decoder_hidden, (0, 0, 0), (1, 1, config.hidden_size), (1, 1, 1)
    ).get_output(0)
    unconditional = ctx.network.add_slice(
        decoder_hidden, (1, 0, 0), (1, 1, config.hidden_size), (1, 1, 1)
    ).get_output(0)
    quantized = ctx.work_constant(
        ("rvq_quantized_zero",),
        np.zeros((1, 1, config.latent_size), dtype=ctx.work_np_dtype),
    )
    selected_codes: list[Any] = []
    codebook_index = 0
    rvq = ctx.weights["rvq_embs"]

    for step, width in enumerate(config.refinement_widths):
        if width == 0:
            continue
        uniform = _slice_refinement_input(ctx, mixture_uniform, step, config.mog_num_predictions)
        noise = _slice_refinement_input(ctx, mog_noise, step, config.latent_size)
        residual = _mog_sample(
            ctx,
            conditional,
            unconditional,
            quantized,
            uniform,
            noise,
            config,
        )
        residual = _cast(ctx, residual, ctx.trt.float32)
        for _ in range(width):
            embedding_np = np.asarray(rvq[codebook_index], dtype=np.float32)
            embedding = ctx.constant(("rvq_search", codebook_index), embedding_np, dtype=np.float32)
            norms_np = np.square(embedding_np).sum(axis=-1, dtype=np.float32).reshape(1, 1, -1)
            norms = ctx.constant(("rvq_norms", codebook_index), norms_np, dtype=np.float32)
            embedding_t = ctx.constant(
                ("rvq_search_t", codebook_index),
                embedding_np.T,
                dtype=np.float32,
                shape=(1, config.latent_size, config.codebook_size),
            )
            dot = ctx.network.add_matrix_multiply(
                residual,
                ctx.trt.MatrixOperation.NONE,
                embedding_t,
                ctx.trt.MatrixOperation.NONE,
            ).get_output(0)
            minus_two = ctx.constant(
                ("minus_two",), np.array(-2.0, dtype=np.float32), dtype=np.float32, shape=(1, 1, 1)
            )
            dot = ctx.network.add_elementwise(
                dot, minus_two, ctx.trt.ElementWiseOperation.PROD
            ).get_output(0)
            distance = ctx.network.add_elementwise(
                dot, norms, ctx.trt.ElementWiseOperation.SUM
            ).get_output(0)
            selected = ctx.network.add_topk(
                distance, ctx.trt.TopKOperation.MIN, 1, 1 << 2
            ).get_output(1)
            selected_codes.append(selected)
            flat_selected = _shuffle(ctx, selected, (1,))
            chosen = ctx.network.add_gather(embedding, flat_selected, 0).get_output(0)
            chosen = _shuffle(ctx, chosen, (1, 1, config.latent_size))
            residual = ctx.network.add_elementwise(
                residual, chosen, ctx.trt.ElementWiseOperation.SUB
            ).get_output(0)
            chosen_work = _cast(ctx, chosen, ctx.work_trt_dtype)
            quantized = ctx.network.add_elementwise(
                quantized, chosen_work, ctx.trt.ElementWiseOperation.SUM
            ).get_output(0)
            codebook_index += 1

    if codebook_index != config.num_quantizers:
        raise AssertionError(
            f"refinement emitted {codebook_index} codes, expected {config.num_quantizers}"
        )
    concat = ctx.network.add_concatenation(selected_codes)
    concat.axis = 2
    return _shuffle(ctx, concat.get_output(0), (config.num_quantizers,))


def add_native_tts_step_graph(
    network: Any,
    trt: Any,
    weights: NativeTTSWeights,
    tables: SubwordTables,
    *,
    max_cache_length: int,
    config: NativeTTSConfig = EXACT_CONFIG,
) -> dict[str, Any]:
    """Populate a strongly typed network with one 80 ms EAR-TTS frame step."""
    config.validate()
    if max_cache_length < 1:
        raise ValueError("EAR-TTS max_cache_length must be positive")
    if max_cache_length > config.sliding_window:
        raise ValueError(
            "EAR-TTS step engine currently requires max_cache_length <= "
            f"sliding_window ({config.sliding_window}) so local and global "
            "layers share one compact runtime cache contract"
        )
    if tables.vocab_size != 131072 or tables.char_padding_id + 1 != config.char_vocab_size:
        raise ValueError("subword tables do not match the checkpoint character encoder")
    if set(weights) != set(required_checkpoint_shapes(config)):
        missing = sorted(set(required_checkpoint_shapes(config)) - set(weights))
        extra = sorted(set(weights) - set(required_checkpoint_shapes(config)))
        raise ValueError(
            f"native TTS weights are incomplete: missing={missing[:4]}, extra={extra[:4]}"
        )

    ctx = _GraphContext(network, trt, weights, trt.float32, np.float32)

    prev_codes = network.add_input("prev_codes", trt.int32, (config.num_quantizers,))
    subword_id = network.add_input("subword_id", trt.int32, (1,))
    subword_mask = network.add_input("subword_mask", trt.float32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, 1, 1, -1))
    mixture_uniform = network.add_input(
        "mixture_uniform", trt.float32, (NUM_REFINEMENT_STEPS, config.mog_num_predictions)
    )
    mog_noise = network.add_input(
        "mog_noise", trt.float32, (NUM_REFINEMENT_STEPS, config.latent_size)
    )
    audio_prompt_latent = network.add_input(
        "audio_prompt_latent", trt.float32, (config.hidden_size,)
    )
    audio_prompt_mode = network.add_input("audio_prompt_mode", trt.float32, (1,))
    bos_flag = network.add_input("bos_flag", trt.float32, (1,))

    cache_k_inputs = []
    cache_v_inputs = []
    for layer_idx in range(config.num_hidden_layers):
        cache_k_inputs.append(
            network.add_input(f"cache_k_{layer_idx}", trt.float32, (2, -1, config.kv_width))
        )
        cache_v_inputs.append(
            network.add_input(f"cache_v_{layer_idx}", trt.float32, (2, -1, config.kv_width))
        )

    text = _add_char_encoder(ctx, subword_id, subword_mask, tables, config)
    audio = _audio_conditioning(
        ctx,
        prev_codes,
        audio_prompt_latent,
        audio_prompt_mode,
        bos_flag,
        config,
    )
    hidden = _duplicate_cfg_conditioning(ctx, audio, text, config)
    position_ids = network.add_concatenation([position_id, position_id])
    position_ids.axis = 0
    position_ids = _shuffle(ctx, position_ids.get_output(0), (2, 1))
    mask = attention_mask
    mask_layer = network.add_concatenation([mask, mask])
    mask_layer.axis = 0
    mask = mask_layer.get_output(0)

    present_k: list[Any] = []
    present_v: list[Any] = []
    for layer_idx in range(config.num_hidden_layers):
        hidden, key, value = _decoder_layer(
            ctx,
            hidden,
            cache_k_inputs[layer_idx],
            cache_v_inputs[layer_idx],
            position_ids,
            mask,
            layer_idx,
            config,
        )
        present_k.append(key)
        present_v.append(value)
    hidden = _rms_norm(ctx, hidden, "backbone.norm.weight", config.rms_norm_eps)
    codes = _rvq_refinement(ctx, hidden, mixture_uniform, mog_noise, config)
    codes.name = "rvq_codes"
    network.mark_output(codes)

    for layer_idx, (key, value) in enumerate(zip(present_k, present_v, strict=True)):
        key.name = f"present_k_{layer_idx}"
        value.name = f"present_v_{layer_idx}"
        network.mark_output(key)
        network.mark_output(value)

    return {
        "rvq_codes": codes,
        **{f"present_k_{i}": tensor for i, tensor in enumerate(present_k)},
        **{f"present_v_{i}": tensor for i, tensor in enumerate(present_v)},
    }


def build_native_tts_engine_from_weights(
    weights: NativeTTSWeights,
    tables: SubwordTables,
    *,
    max_cache_length: int,
    config: NativeTTSConfig = EXACT_CONFIG,
    verbose: bool = False,
) -> bytes:
    """Build a serialized strongly typed TensorRT EAR-TTS step engine."""
    severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 1
    builder_config.max_num_tactics = 1
    builder_config.clear_flag(trt.BuilderFlag.TF32)

    add_native_tts_step_graph(
        network,
        trt,
        weights,
        tables,
        max_cache_length=max_cache_length,
        config=config,
    )

    profile = builder.create_optimization_profile()
    opt_cache_length = min(max_cache_length, 256)
    profile.set_shape(
        "attention_mask",
        (1, 1, 1, 2),
        (1, 1, 1, opt_cache_length + 1),
        (1, 1, 1, max_cache_length + 1),
    )
    for layer_idx in range(config.num_hidden_layers):
        minimum = (2, 1, config.kv_width)
        optimum = (2, opt_cache_length, config.kv_width)
        maximum = (2, max_cache_length, config.kv_width)
        profile.set_shape(f"cache_k_{layer_idx}", minimum, optimum, maximum)
        profile.set_shape(f"cache_v_{layer_idx}", minimum, optimum, maximum)
    builder_config.add_optimization_profile(profile)

    if verbose:
        print(
            "[trtmc build] VoiceChat native EAR-TTS: "
            f"layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
            f"kv={config.kv_width}, cache={max_cache_length}",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the VoiceChat EAR-TTS engine")
    return bytes(plan)


def build_native_tts_engine(
    model_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    max_cache_length: int,
    verbose: bool = False,
) -> bytes:
    """Load public assets and build the runtime-only TensorRT TTS engine."""
    weights = load_native_tts_weights(model_dir)
    tables = build_subword_tables(tokenizer_dir)
    return build_native_tts_engine_from_weights(
        weights,
        tables,
        max_cache_length=max_cache_length,
        verbose=verbose,
    )


def _resolve_tokenizer_snapshot(tokenizer_dir: str | Path | None) -> Path:
    if tokenizer_dir is not None:
        return Path(tokenizer_dir)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=TEXT_MODEL_ID,
            revision=TEXT_MODEL_REVISION,
            allow_patterns=["tokenizer.json"],
        )
    )


def _load_runtime_code_assets(model_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = _resolve_safetensors(model_dir)
    with safe_open(str(checkpoint), framework="np") as reader:
        silence = np.asarray(reader.get_tensor("tts_model.codec_silence_tokens"))
        control = np.asarray(reader.get_tensor("tts_model._control_codes"))
    if silence.shape != (EXACT_CONFIG.num_quantizers,):
        raise ValueError("tts_model.codec_silence_tokens must contain one token per RVQ codebook")
    if control.shape != (3,):
        raise ValueError("tts_model._control_codes must contain BOS, EOS, and PAD")
    silence = np.ascontiguousarray(silence, dtype="<i4")
    control = np.ascontiguousarray(control, dtype="<i4")
    expected_control = np.array([1026, 1025, 1024], dtype="<i4")
    if not np.array_equal(control, expected_control):
        raise ValueError(f"unexpected VoiceChat TTS control codes: {control.tolist()}")
    if np.any(silence < 0) or np.any(silence >= EXACT_CONFIG.codebook_size):
        raise ValueError("VoiceChat TTS silence codes must be ordinary codec IDs")
    return silence, control


def _load_aria_warmup_assets(
    model_dir: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the checkpoint's pre-baked Aria state and exact 37-step recipe.

    NeMo applies the cached latent only before the final BOS frame.  Rows 0-35
    therefore use this checkpoint tensor directly; row 36 switches back to
    the ordinary silence-code embedding and adds ``bos_emb``.
    """
    checkpoint = _resolve_safetensors(model_dir)
    with safe_open(str(checkpoint), framework="np") as reader:
        prompt = np.asarray(
            reader.get_tensor("tts_model.audio_prompt_latents.Aria"),
            dtype=np.float32,
        )
    expected_shape = (1, 37, EXACT_CONFIG.hidden_size)
    if prompt.shape != expected_shape:
        raise ValueError(
            f"Aria audio prompt latent has shape {prompt.shape}, expected {expected_shape}"
        )
    warmup = np.ascontiguousarray(prompt[0], dtype="<f4")

    # target_text = [EOS] + ([PAD] * 36 + [EOS]); the TTS model shifts it
    # left and keeps the first 37 entries for warmup.
    subword_ids = [12] * 36 + [2]
    subword_mask = [0] * 35 + [1, 1]
    bos_flags = [0] * 36 + [1]
    recipe: dict[str, Any] = {
        "num_steps": 37,
        "subword_ids": subword_ids,
        "subword_mask": subword_mask,
        "audio_prompt_mode": [1] * 36 + [0],
        "bos_flags": bos_flags,
        "position_ids": list(range(37)),
        "first_generation_position_id": 37,
        "tts_max_cache_length": EXACT_CONFIG.sliding_window,
    }
    return warmup, recipe


def build_tts_sections(
    model_dir: str | Path,
    raw_config: dict[str, Any] | None = None,
    *,
    tokenizer_dir: str | Path | None = None,
    max_cache_length: int = EXACT_CONFIG.sliding_window,
    verbose: bool = False,
) -> list[tuple[str, bytes]]:
    """Model.py integration entrypoint for the native VoiceChat TTS sections.

    ``raw_config`` is accepted so the model-owned build orchestrator can use a
    uniform component signature.  Architecture truth is deliberately checked
    against the checkpoint tensor contract rather than reinterpreted from that
    nested training configuration.
    """
    del raw_config
    resolved_tokenizer = _resolve_tokenizer_snapshot(tokenizer_dir)
    engine = build_native_tts_engine(
        model_dir,
        resolved_tokenizer,
        max_cache_length=max_cache_length,
        verbose=verbose,
    )
    silence, control = _load_runtime_code_assets(model_dir)
    aria_warmup, prompt_recipe = _load_aria_warmup_assets(model_dir)
    prompt_recipe["tts_max_cache_length"] = max_cache_length
    first_code = np.full(
        (EXACT_CONFIG.num_quantizers,),
        int(control[2]),
        dtype="<i4",
    )
    return [
        ("tts.plan", engine),
        ("tts_prompt.silence_codes", silence.tobytes()),
        ("tts_prompt.control_codes", control.tobytes()),
        ("tts_prompt.first_codes", first_code.tobytes()),
        ("tts_prompt.embeddings", aria_warmup.tobytes()),
        (
            "tts_prompt.json",
            json.dumps(prompt_recipe, separators=(",", ":")).encode("utf-8"),
        ),
    ]


__all__ = [
    "CHECKPOINT_PREFIX",
    "EXACT_CONFIG",
    "FRAME_SECONDS",
    "NUM_REFINEMENT_STEPS",
    "NativeTTSConfig",
    "NativeTTSWeights",
    "SubwordTables",
    "add_native_tts_step_graph",
    "build_native_tts_engine",
    "build_native_tts_engine_from_weights",
    "build_tts_sections",
    "build_subword_tables",
    "load_native_tts_weights",
    "required_checkpoint_shapes",
]
