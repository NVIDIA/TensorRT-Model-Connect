# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native FP32 TensorRT encoder for MiniMax-H3 Ref2VA soundtracks.

The plan consumes stereo as two batch items, follows the released DAC encoder
and causal attention projection, and emits the posterior mean.  Ref2VA never
evaluates ``logs_proj`` and never samples this posterior; C++ applies the
checkpoint's 32-channel mean/std normalization and then packs channel-major
rows.  No Torch, audio framework, or process sidecar is part of the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .audio_vae_builder import (
    _conv1d,
    _fold_checkpoint_weight,
    _require_array,
)
from .fl2va_contract import PlanAbi, TensorAbi


trt = trt_compat.get_trt()

AUDIO_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30


@dataclass(frozen=True)
class Ref2VAAudioEncoderProfile:
    batch_size: int = 2
    sampling_rate: int = 32_000
    min_samples: int = 64_000
    opt_samples: int = 165_600
    max_samples: int = 480_000
    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    latent_dim: int = 2048
    latent_channels: int = 32
    num_attention_heads: int = 8

    @property
    def hop_length(self) -> int:
        return math.prod(self.encoder_rates)

    @property
    def sample_profile(self) -> tuple[int, int, int]:
        return self.min_samples, self.opt_samples, self.max_samples

    @property
    def latent_profile(self) -> tuple[int, int, int]:
        return tuple(value // self.hop_length for value in self.sample_profile)

    def validate(self) -> None:
        values = (
            self.batch_size,
            self.sampling_rate,
            self.min_samples,
            self.opt_samples,
            self.max_samples,
            self.encoder_dim,
            self.latent_dim,
            self.latent_channels,
            self.num_attention_heads,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
        ):
            raise ValueError("MiniMax-H3 Ref2VA audio-encoder dimensions must be positive integers")
        if self.batch_size != 2 or self.sampling_rate != 32_000:
            raise ValueError("MiniMax-H3 Ref2VA audio encoder requires stereo at 32 kHz")
        if not self.min_samples <= self.opt_samples <= self.max_samples:
            raise ValueError("MiniMax-H3 Ref2VA audio samples must satisfy min <= opt <= max")
        if any(value % self.hop_length for value in self.sample_profile):
            raise ValueError(
                "MiniMax-H3 Ref2VA audio inputs must be right-padded to the 800-sample hop"
            )
        if self.latent_dim % self.num_attention_heads:
            raise ValueError("MiniMax-H3 Ref2VA audio attention width must divide its heads")


DEFAULT_REF2VA_AUDIO_ENCODER_PROFILE = Ref2VAAudioEncoderProfile()


def ref2va_audio_encoder_abi(
    profile: Ref2VAAudioEncoderProfile = DEFAULT_REF2VA_AUDIO_ENCODER_PROFILE,
) -> PlanAbi:
    """Strict native ABI for explicit audio and attached video soundtracks.

    The C++ caller resamples/interleaves to stereo 32 kHz, separates the two
    channels into this batch-of-two representation, and right-pads each clip
    to the 800-sample hop.  The plan emits the released posterior mode.  The
    caller then applies the 32 per-channel ``latents_mean``/``latents_std``
    values pinned in bundle metadata and preserves channel-major row order.
    """

    profile.validate()
    sample_shapes = tuple((profile.batch_size, 1, samples) for samples in profile.sample_profile)
    latent_shapes = tuple(
        (profile.batch_size, profile.latent_channels, frames) for frames in profile.latent_profile
    )
    return PlanAbi(
        filename="ref2va_audio_vae_encoder.plan",
        inputs=(TensorAbi("audio_samples", "float32", *sample_shapes),),
        outputs=(TensorAbi("posterior_mean", "float32", *latent_shapes),),
    )


def checkpoint_keys(
    profile: Ref2VAAudioEncoderProfile = DEFAULT_REF2VA_AUDIO_ENCODER_PROFILE,
) -> tuple[str, ...]:
    """Return exactly the weights evaluated by ``posterior.mode()``."""

    profile.validate()
    names = [
        "encoder.block.0.weight_g",
        "encoder.block.0.weight_v",
        "encoder.block.0.bias",
    ]
    channels = profile.encoder_dim
    for stage, _stride in enumerate(profile.encoder_rates, start=1):
        prefix = f"encoder.block.{stage}.block"
        for residual in range(3):
            residual_prefix = f"{prefix}.{residual}.block"
            names.extend(
                (
                    f"{residual_prefix}.0.alpha",
                    f"{residual_prefix}.1.weight_g",
                    f"{residual_prefix}.1.weight_v",
                    f"{residual_prefix}.1.bias",
                    f"{residual_prefix}.2.alpha",
                    f"{residual_prefix}.3.weight_g",
                    f"{residual_prefix}.3.weight_v",
                    f"{residual_prefix}.3.bias",
                )
            )
        names.extend(
            (
                f"{prefix}.3.alpha",
                f"{prefix}.4.weight_g",
                f"{prefix}.4.weight_v",
                f"{prefix}.4.bias",
            )
        )
        channels *= 2
    names.extend(
        (
            f"encoder.block.{len(profile.encoder_rates) + 1}.alpha",
            f"encoder.block.{len(profile.encoder_rates) + 2}.weight_g",
            f"encoder.block.{len(profile.encoder_rates) + 2}.weight_v",
            f"encoder.block.{len(profile.encoder_rates) + 2}.bias",
            "pre_block.norm1.weight",
            "pre_block.norm1.bias",
            "pre_block.attn.qkv.weight",
            "pre_block.attn.q_bias",
            "pre_block.attn.v_bias",
            "pre_block.attn.zero_k_bias",
            "pre_block.attn.proj.weight",
            "pre_block.attn.proj.bias",
            "pre_block.proj.weight",
            "pre_block.proj.bias",
            "pre_block.norm3.weight",
            "pre_block.norm3.bias",
            "pre_block.norm2.weight",
            "pre_block.norm2.bias",
            "pre_block.mlp.norm.weight",
            "pre_block.mlp.norm.bias",
            "pre_block.mlp.w0.weight",
            "pre_block.mlp.w0.bias",
            "pre_block.mlp.w1.weight",
            "pre_block.mlp.w1.bias",
            "pre_block.mlp.w2.weight",
            "pre_block.mlp.w2.bias",
            "mean_proj.weight",
            "mean_proj.bias",
        )
    )
    return tuple(names)


def _snake(network, hidden, weights, name: str):
    channels = int(tuple(hidden.shape)[1])
    alpha = _require_array(weights, f"{name}.alpha", (1, channels, 1)).reshape(1, channels, 1, 1)
    alpha_tensor = op.weight_constant(network, alpha)
    phase = network.add_elementwise(hidden, alpha_tensor, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    sine = network.add_unary(phase, trt.UnaryOperation.SIN).get_output(0)
    square = network.add_elementwise(sine, sine, trt.ElementWiseOperation.PROD).get_output(0)
    epsilon = op.constant(network, np.full((1, 1, 1, 1), 1.0e-9, np.float32))
    denominator = network.add_elementwise(
        alpha_tensor, epsilon, trt.ElementWiseOperation.SUM
    ).get_output(0)
    periodic = network.add_elementwise(
        square, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)
    return network.add_elementwise(hidden, periodic, trt.ElementWiseOperation.SUM).get_output(0)


def _residual_unit(
    network,
    hidden,
    weights,
    prefix: str,
    *,
    channels: int,
    dilation: int,
    owned_weights: list[np.ndarray],
):
    update = _snake(network, hidden, weights, f"{prefix}.0")
    conv1 = f"{prefix}.1"
    update = _conv1d(
        network,
        update,
        _fold_checkpoint_weight(weights, conv1, (channels, channels, 7)),
        _require_array(weights, f"{conv1}.bias", (channels,)),
        padding=3 * dilation,
        dilation=dilation,
        name=conv1,
        owned_weights=owned_weights,
    )
    update = _snake(network, update, weights, f"{prefix}.2")
    conv2 = f"{prefix}.3"
    update = _conv1d(
        network,
        update,
        _fold_checkpoint_weight(weights, conv2, (channels, channels, 1)),
        _require_array(weights, f"{conv2}.bias", (channels,)),
        name=conv2,
        owned_weights=owned_weights,
    )
    return network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)


def _layer_norm(network, hidden, weights, prefix: str, width: int):
    rank = len(tuple(hidden.shape))
    shape = (1,) * (rank - 1) + (width,)
    gamma = op.weight_constant(
        network, _require_array(weights, f"{prefix}.weight", (width,)).reshape(shape)
    )
    beta = op.weight_constant(
        network, _require_array(weights, f"{prefix}.bias", (width,)).reshape(shape)
    )
    layer = network.add_normalization_v2(hidden, gamma, beta, 1 << (rank - 1))
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 audio LayerNorm {prefix}")
    layer.name = prefix
    layer.epsilon = 1.0e-5
    return layer.get_output(0)


def _linear(network, hidden, weights, prefix: str):
    weight = _require_array(
        weights,
        f"{prefix}.weight",
        tuple(np.asarray(weights[f"{prefix}.weight"]).shape),
    )
    # ``mean_proj`` is a checkpoint Conv1d(k=1).  The projection is evaluated
    # on [B,T,C] rows here, so its singleton kernel axis is removed without
    # changing either the values or their input/output ordering.
    if weight.ndim == 3:
        if weight.shape[-1] != 1:
            raise ValueError(f"MiniMax-H3 audio linear {prefix} must have a unit kernel")
        weight = np.ascontiguousarray(weight[..., 0])
    return op.linear(
        network,
        hidden,
        weight,
        _require_array(
            weights,
            f"{prefix}.bias",
            tuple(np.asarray(weights[f"{prefix}.bias"]).shape),
        ),
        bf16=False,
    )


def _causal_attention_projection(network, hidden, weights, profile: Ref2VAAudioEncoderProfile):
    batch = profile.batch_size
    in_width = profile.latent_dim
    out_width = profile.latent_channels
    heads = profile.num_attention_heads
    head_dim = in_width // heads

    normalized = _layer_norm(network, hidden, weights, "pre_block.norm1", in_width)
    qkv_weight = _require_array(weights, "pre_block.attn.qkv.weight", (3 * in_width, in_width))
    q_bias = _require_array(weights, "pre_block.attn.q_bias", (in_width,))
    k_bias = _require_array(weights, "pre_block.attn.zero_k_bias", (in_width,))
    if np.any(k_bias != 0.0):
        raise ValueError("MiniMax-H3 audio frozen key bias must be exactly zero")
    v_bias = _require_array(weights, "pre_block.attn.v_bias", (in_width,))
    qkv = op.linear(
        network,
        normalized,
        qkv_weight,
        np.concatenate((q_bias, k_bias, v_bias)),
        bf16=False,
    )

    tensors = []
    for index in range(3):
        tensor = op.dynamic_slice(
            network,
            qkv,
            (0, 0, index * in_width),
            (batch, None, in_width),
        )
        reshape = network.add_shuffle(tensor)
        reshape.reshape_dims = (batch, -1, heads, head_dim)
        reshape.second_transpose = trt.Permutation((0, 2, 1, 3))
        tensors.append(reshape.get_output(0))
    query, key, value = tensors
    scale = op.constant(
        network,
        np.full((1, 1, 1, 1), 1.0 / math.sqrt(head_dim), np.float32),
    )
    query = network.add_elementwise(query, scale, trt.ElementWiseOperation.PROD).get_output(0)
    attention = network.add_attention(
        query,
        key,
        value,
        trt.AttentionNormalizationOp.SOFTMAX,
        True,
    )
    if attention is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 audio causal attention")
    attention.name = "pre_block.attn.native_causal_attention"
    attention.metadata = "trtmc.native_op=IAttention;causal=true;decomposable=true"
    # TRT-RTX 1.6 has no dedicated causal kernel covering the complete
    # dynamic 80..600-row public profile.  Allow TRT itself to lower the
    # IAttention composite into native layers; this remains inside the plan
    # and introduces no plugin, framework, or runtime dependency.
    attention.decomposable = True

    # [B,H,T,D] -> mean heads -> [B,T,D] -> average each exact group
    # of D/out_width=8 channels, matching adaptive_avg_pool1d.
    pooled_heads = network.add_reduce(
        attention.get_output(0),
        trt.ReduceOperation.AVG,
        1 << 1,
        False,
    )
    reshape = network.add_shuffle(pooled_heads.get_output(0))
    reshape.reshape_dims = (batch, -1, out_width, head_dim // out_width)
    pooled_width = network.add_reduce(
        reshape.get_output(0),
        trt.ReduceOperation.AVG,
        1 << 3,
        False,
    ).get_output(0)
    attended = _linear(network, pooled_width, weights, "pre_block.attn.proj")

    residual = _layer_norm(network, hidden, weights, "pre_block.norm3", in_width)
    residual = _linear(network, residual, weights, "pre_block.proj")
    hidden = network.add_elementwise(residual, attended, trt.ElementWiseOperation.SUM).get_output(0)
    update = _layer_norm(network, hidden, weights, "pre_block.norm2", out_width)
    update = _layer_norm(network, update, weights, "pre_block.mlp.norm", out_width)
    left = _linear(network, update, weights, "pre_block.mlp.w0")
    gelu = network.add_activation(left, trt.ActivationType.GELU_TANH)
    if gelu is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 audio GeGLU activation")
    right = _linear(network, update, weights, "pre_block.mlp.w1")
    update = network.add_elementwise(
        gelu.get_output(0), right, trt.ElementWiseOperation.PROD
    ).get_output(0)
    update = _linear(network, update, weights, "pre_block.mlp.w2")
    return network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)


@op.cleanup_failed_build
def build_ref2va_audio_encoder_engine(
    weights: dict,
    profile: Ref2VAAudioEncoderProfile = DEFAULT_REF2VA_AUDIO_ENCODER_PROFILE,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build the released stereo soundtrack posterior-mean encoder."""

    profile.validate()
    expected = set(checkpoint_keys(profile))
    missing = sorted(expected - set(weights))
    unexpected = sorted(set(weights) - expected)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 Ref2VA audio encoder checkpoint partition mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config, weight_streaming=weight_streaming)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=AUDIO_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    )
    abi = ref2va_audio_encoder_abi(profile)
    samples = network.add_input(
        abi.inputs[0].name,
        trt.float32,
        (profile.batch_size, 1, -1),
    )
    optimization = builder.create_optimization_profile()
    optimization.set_shape(
        "audio_samples",
        min=(profile.batch_size, 1, profile.min_samples),
        opt=(profile.batch_size, 1, profile.opt_samples),
        max=(profile.batch_size, 1, profile.max_samples),
    )
    if config.add_optimization_profile(optimization) != 0:
        raise RuntimeError("TensorRT rejected MiniMax-H3 Ref2VA audio profile")
    expand = network.add_shuffle(samples)
    expand.reshape_dims = (profile.batch_size, 1, 1, -1)
    hidden = expand.get_output(0)
    owned_weights: list[np.ndarray] = []

    hidden = _conv1d(
        network,
        hidden,
        _fold_checkpoint_weight(
            weights,
            "encoder.block.0",
            (profile.encoder_dim, 1, 7),
        ),
        _require_array(weights, "encoder.block.0.bias", (profile.encoder_dim,)),
        padding=3,
        name="encoder.block.0",
        owned_weights=owned_weights,
    )
    channels = profile.encoder_dim
    for stage, stride in enumerate(profile.encoder_rates, start=1):
        prefix = f"encoder.block.{stage}.block"
        for residual, dilation in enumerate((1, 3, 9)):
            hidden = _residual_unit(
                network,
                hidden,
                weights,
                f"{prefix}.{residual}.block",
                channels=channels,
                dilation=dilation,
                owned_weights=owned_weights,
            )
        hidden = _snake(network, hidden, weights, f"{prefix}.3")
        down = f"{prefix}.4"
        hidden = _conv1d(
            network,
            hidden,
            _fold_checkpoint_weight(
                weights,
                down,
                (2 * channels, channels, 2 * stride),
            ),
            _require_array(weights, f"{down}.bias", (2 * channels,)),
            stride=stride,
            padding=math.ceil(stride / 2),
            name=down,
            owned_weights=owned_weights,
        )
        channels *= 2
    hidden = _snake(
        network,
        hidden,
        weights,
        f"encoder.block.{len(profile.encoder_rates) + 1}",
    )
    final = f"encoder.block.{len(profile.encoder_rates) + 2}"
    hidden = _conv1d(
        network,
        hidden,
        _fold_checkpoint_weight(weights, final, (profile.latent_dim, channels, 3)),
        _require_array(weights, f"{final}.bias", (profile.latent_dim,)),
        padding=1,
        name=final,
        owned_weights=owned_weights,
    )
    rows = network.add_shuffle(hidden)
    rows.first_transpose = trt.Permutation((0, 3, 1, 2))
    rows.reshape_dims = (profile.batch_size, -1, profile.latent_dim)
    hidden = _causal_attention_projection(network, rows.get_output(0), weights, profile)
    posterior_mean = _linear(network, hidden, weights, "mean_proj")
    output = network.add_shuffle(posterior_mean)
    output.first_transpose = trt.Permutation((0, 2, 1))
    value = output.get_output(0)
    value.name = abi.outputs[0].name
    network.mark_output(value)
    op.validate_native_network(
        network,
        expected_attentions=1,
        label="Ref2VA audio VAE encoder",
    )

    plan = None
    record = None
    try:
        if output_path is None:
            plan = builder.build_serialized_network(network, config)
        else:
            record = trt_compat.build_serialized_network_to_file(
                builder, network, config, output_path
            )
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if output_path is None and plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 Ref2VA audio encoder")
    del network, config, builder, logger, owned_weights
    gc.collect()
    return record if record is not None else bytes(plan)
