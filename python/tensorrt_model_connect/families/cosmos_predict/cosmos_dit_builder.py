"""Cosmos-Predict2 14B Video2World DiT engine builder.

Builds a single TRT engine for the dense (single-GPU) ``CosmosTransformer3DModel``
denoiser at the heart of ``nvidia/Cosmos-Predict2-14B-Video2World``.

Architecture (locked from HF ``transformer/config.json`` and the diffusers
reference impl ``diffusers.models.transformers.transformer_cosmos``)
----------------------------------------------------------------------

* ``in_channels=17``        — 16 VAE latent channels + 1 padding mask, concatenated
                              along the channel dimension by the C++ / Python
                              preprocessor before the engine runs.
* ``out_channels=16``       — denoising target (epsilon / velocity in the
                              VAE latent space).
* ``num_attention_heads=40`` x ``attention_head_dim=128`` -> ``hidden_size=5120``.
* ``num_layers=36``         — number of ``CosmosTransformerBlock``s.
* ``mlp_ratio=4.0``         -> ``ffn_dim = 4 * 5120 = 20480``.
* ``patch_size=(1, 2, 2)``  — temporal=1, spatial=2x2.
* ``text_embed_dim=1024``   — T5-11B encoder width (different from Wan/Cosmos1).
* ``adaln_lora_dim=256``    — low-rank channel of the per-block AdaLN-LoRA.
* ``concat_padding_mask=True`` — caller is responsible for the 16+1 channel
                              concat; the engine just expects 17 in-channels.
* ``extra_pos_embed_type=None`` — 14B does not use a learnable position
                              embedding (the 2B variant does). 3-axis RoPE
                              is the only positional signal.
* ``rope_scale=(0.8333, 2.0, 2.0)`` — per-axis (T, H, W) scaling of the RoPE
                              positions. Determines the (T, H, W) head_dim
                              split as (16, 56, 56) for the 14B model (the
                              same axes_dim used by ``QwenEmbedRope`` /
                              Cosmos diffusers reference).
* ``max_size=(128, 240, 240)`` — maximum patches along (T, H, W). Used to
                              pre-compute the RoPE frequency tables once at
                              build time.

Per-block structure (``CosmosTransformerBlock``)
-------------------------------------------------

Each block has three :class:`CosmosAdaLayerNormZero` modulators (``norm1``,
``norm2``, ``norm3``). One AdaLN-LoRA forward looks like::

    a = SiLU(temb)
    b = SiLU(embedded_timestep)
    delta_main      = norm.linear_2(SiLU(norm.linear_1(a)))     # [B, 3*dim]
    delta_residual  = norm.linear_3(b)                          # [B, 3*dim]
    shift, scale, gate = (delta_main + delta_residual).chunk(3, dim=-1)
    h = LayerNorm(x, eps=1e-6, affine=False)
    h = h * (1 + scale) + shift
    # gate is returned separately for the gated residual that follows the
    # block sub-layer (self-attn / cross-attn / FFN).

The block forward then runs::

    h1, gate1 = norm1(x, temb, embedded_timestep)
    x = x + gate1 * self_attention(h1)
    h2, gate2 = norm2(x, temb, embedded_timestep)
    x = x + gate2 * cross_attention(h2, encoder_hidden_states)
    h3, gate3 = norm3(x, temb, embedded_timestep)
    x = x + gate3 * ffn(h3)

* Self-attention: RMSNorm over each head's ``D`` dim on Q and K, 3-axis
  RoPE applied to Q and K, no biases on the projections.
* Cross-attention: RMSNorm on Q only (the diffusers source has no
  ``norm_k`` for ``attn2``). No RoPE. K/V projected from the 1024-dim T5
  output via dedicated ``to_k``/``to_v`` weights of shape ``[text_dim, hidden]``.
* FFN: ``Linear(hidden, ffn_dim) -> gelu_pytorch_tanh -> Linear(ffn_dim, hidden)``,
  no SwiGLU gating.

Final-output head (``norm_out`` + ``proj_out``)
------------------------------------------------

A single :class:`CosmosAdaLayerNorm` head produces ``(shift, scale)`` only::

    delta_main     = norm_out.linear_2(SiLU(norm_out.linear_1(SiLU(temb))))
    delta_residual = norm_out.linear_3(SiLU(embedded_timestep))
    shift, scale   = (delta_main + delta_residual).chunk(2, dim=-1)
    h = LayerNorm(x, eps=1e-6, affine=False) * (1 + scale) + shift
    output = proj_out(h)        # [num_patches, out_channels * pt * ph * pw]

Engine I/O contract
-------------------

Inputs (all batch=1, fixed shape at build time):

    hidden_states         [num_patches, hidden_size]              fp32
        Patchified latent. The runner is responsible for the
        ``[B, 17, T_lat, H_lat, W_lat] -> [num_patches, hidden_size]``
        reshape (matches the Wan T2V engine convention).
    encoder_hidden_states [text_seq_len, text_embed_dim]          fp32
        T5-11B encoder output. ``text_embed_dim`` is 1024 for Cosmos-Predict2.
    temb                  [1, hidden_size]                        fp32
        Per-step timestep embedding feeding the LoRA "main" path of every
        :class:`CosmosAdaLayerNormZero` modulator.
    embedded_timestep     [1, hidden_size]                        fp32
        Per-step embedding feeding the LoRA "residual" path (``linear_3``)
        of every modulator. In diffusers this is the SiLU-activated
        sinusoidal embedding that bypasses the time-MLP. Pre-activated
        (i.e. NO SiLU applied here) by the caller, matching what the
        diffusers ``CosmosEmbedding`` module returns.

Output:

    noise_pred            [num_patches, out_channels * pt * ph * pw]  fp32
        Pre-unpatchify denoising output. The runner reshapes to
        ``[1, 16, T_lat, H_lat, W_lat]``.

Single-GPU only. Multi-GPU sequence-parallel and tensor-parallel variants
are out of scope for this builder (the plugin guards with NotImplementedError
when ``parallel_config.tp_size != 1``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tensorrt_model_connect import trt_compat

from ... import graph_ops

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# ---------------------------------------------------------------------------
# Cosmos-Predict2 14B architecture constants
# ---------------------------------------------------------------------------

COSMOS_14B_IN_CHANNELS = 17           # 16 VAE z + 1 padding mask
COSMOS_14B_OUT_CHANNELS = 16
COSMOS_14B_NUM_HEADS = 40
COSMOS_14B_HEAD_DIM = 128
COSMOS_14B_HIDDEN_SIZE = 5120         # 40 * 128
COSMOS_14B_NUM_LAYERS = 36
COSMOS_14B_FFN_DIM = 20480            # mlp_ratio=4.0
COSMOS_14B_PATCH_SIZE = (1, 2, 2)
COSMOS_14B_TEXT_EMBED_DIM = 1024
COSMOS_14B_ADALN_LORA_DIM = 256
COSMOS_14B_TEXT_SEQ_LEN = 512
COSMOS_14B_ROPE_THETA = 10000.0
# Per-axis RoPE head-dim split (T, H, W). 16 + 56 + 56 = 128.
COSMOS_14B_ROPE_AXES_DIM = (16, 56, 56)
# Per-axis position scaling factor (matches diffusers ``rope_scale``).
COSMOS_14B_ROPE_SCALE = (0.8333, 2.0, 2.0)
# Maximum supported patch grid; needed to bound the RoPE precompute table.
COSMOS_14B_MAX_SIZE = (128, 240, 240)
COSMOS_14B_NORM_EPS = 1e-6
COSMOS_14B_VAE_SCALE_SPATIAL = 8
COSMOS_14B_VAE_SCALE_TEMPORAL = 4


# ---------------------------------------------------------------------------
# Small TRT helpers
# ---------------------------------------------------------------------------

def _silu_2d(network: "trt.INetworkDefinition", x: "trt.ITensor") -> "trt.ITensor":
    """SiLU on a 2-D tensor (broadcast-friendly variant of graph_ops.add_silu)."""
    sig = network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(x, sig, trt.ElementWiseOperation.PROD).get_output(0)


def _linear_2d(
    network: "trt.INetworkDefinition",
    x: "trt.ITensor",
    in_dim: int,
    out_dim: int,
    weight_t: np.ndarray,
    bias: np.ndarray | None,
) -> "trt.ITensor":
    """Linear(in_dim -> out_dim) on [N, in_dim] input.

    ``weight_t`` must be transposed to ``[in_dim, out_dim]`` order so that
    ``y = x @ weight_t`` produces ``[N, out_dim]``. The matching
    ``load_cosmos_dit_weights`` helper does the transpose at load time.
    """
    y = graph_ops.add_matmul_rhs_constant(network, x, in_dim, out_dim, weight_t)
    if bias is not None:
        y = graph_ops.add_bias_sum(network, y, out_dim, bias)
    return y


def _layer_norm_no_affine_2d(
    network: "trt.INetworkDefinition",
    x: "trt.ITensor",
    hidden_size: int,
    eps_tensor: "trt.ITensor",
) -> "trt.ITensor":
    """LayerNorm with elementwise_affine=False over the last axis."""
    # Reuse the shared no-affine helper.
    return graph_ops.add_layer_norm_no_affine(network, x, hidden_size, eps_tensor)


# ---------------------------------------------------------------------------
# CosmosAdaLayerNormZero (per-block: norm1/norm2/norm3)
#
# Returns ``(modulated_x, gate)`` so the caller can splice in the sub-layer
# (self-attn / cross-attn / FFN) and finish the gated residual outside.
# ---------------------------------------------------------------------------

def _adaln_zero(
    network: "trt.INetworkDefinition",
    x: "trt.ITensor",
    temb_silu: "trt.ITensor",
    embedded_silu: "trt.ITensor",
    *,
    hidden_size: int,
    adaln_lora_dim: int,
    eps_tensor: "trt.ITensor",
    w_l1: np.ndarray,
    b_l1: np.ndarray | None,
    w_l2: np.ndarray,
    b_l2: np.ndarray | None,
    w_l3: np.ndarray,
    b_l3: np.ndarray | None,
) -> tuple["trt.ITensor", "trt.ITensor"]:
    """One CosmosAdaLayerNormZero step.

    Produces 3 ``hidden_size``-wide modulation chunks ``(shift, scale, gate)``
    and applies ``LayerNorm * (1 + scale) + shift`` to ``x``. The gate is
    returned so the caller can use it for the residual add after the
    following sub-layer.

    ``temb_silu`` / ``embedded_silu`` are [1, hidden_size] tensors already
    pre-activated with SiLU by ``build_cosmos_dit_engine``.
    """
    # LoRA "main" path: SiLU was already applied to temb. We do
    # Linear-1, SiLU, Linear-2 here.
    a = _linear_2d(network, temb_silu, hidden_size, adaln_lora_dim, w_l1, b_l1)
    a = _silu_2d(network, a)
    a = _linear_2d(network, a, adaln_lora_dim, 3 * hidden_size, w_l2, b_l2)
    # LoRA "residual" path: Linear-3 from the SiLU-activated embedded_timestep.
    r = _linear_2d(network, embedded_silu, hidden_size, 3 * hidden_size, w_l3, b_l3)
    modulation = network.add_elementwise(
        a, r, trt.ElementWiseOperation.SUM).get_output(0)
    # Chunk along axis=-1 into (shift, scale, gate), each [1, hidden_size].
    shift = network.add_slice(
        modulation,
        start=(0, 0),
        shape=(1, hidden_size),
        stride=(1, 1)).get_output(0)
    scale = network.add_slice(
        modulation,
        start=(0, hidden_size),
        shape=(1, hidden_size),
        stride=(1, 1)).get_output(0)
    gate = network.add_slice(
        modulation,
        start=(0, 2 * hidden_size),
        shape=(1, hidden_size),
        stride=(1, 1)).get_output(0)
    # LayerNorm(x, affine=False), then modulate.
    h = _layer_norm_no_affine_2d(network, x, hidden_size, eps_tensor)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    scale_plus_one = network.add_elementwise(
        scale, one, trt.ElementWiseOperation.SUM).get_output(0)
    h = network.add_elementwise(
        h, scale_plus_one, trt.ElementWiseOperation.PROD).get_output(0)
    h = network.add_elementwise(
        h, shift, trt.ElementWiseOperation.SUM).get_output(0)
    return h, gate


# ---------------------------------------------------------------------------
# CosmosAdaLayerNorm (final norm_out; 2 chunks instead of 3)
# ---------------------------------------------------------------------------

def _adaln_final(
    network: "trt.INetworkDefinition",
    x: "trt.ITensor",
    temb_silu: "trt.ITensor",
    embedded_silu: "trt.ITensor",
    *,
    hidden_size: int,
    adaln_lora_dim: int,
    eps_tensor: "trt.ITensor",
    w_l1: np.ndarray,
    b_l1: np.ndarray | None,
    w_l2: np.ndarray,
    b_l2: np.ndarray | None,
    w_l3: np.ndarray,
    b_l3: np.ndarray | None,
) -> "trt.ITensor":
    """Final AdaLN modulator. Same LoRA layout as block norms but 2 chunks.

    ``linear_2.weight`` has shape ``[adaln_lora_dim, 2*hidden_size]`` and
    ``linear_3.weight`` has shape ``[hidden_size, 2*hidden_size]``.
    """
    a = _linear_2d(network, temb_silu, hidden_size, adaln_lora_dim, w_l1, b_l1)
    a = _silu_2d(network, a)
    a = _linear_2d(network, a, adaln_lora_dim, 2 * hidden_size, w_l2, b_l2)
    r = _linear_2d(network, embedded_silu, hidden_size, 2 * hidden_size, w_l3, b_l3)
    modulation = network.add_elementwise(
        a, r, trt.ElementWiseOperation.SUM).get_output(0)
    shift = network.add_slice(
        modulation, start=(0, 0),
        shape=(1, hidden_size), stride=(1, 1)).get_output(0)
    scale = network.add_slice(
        modulation, start=(0, hidden_size),
        shape=(1, hidden_size), stride=(1, 1)).get_output(0)
    h = _layer_norm_no_affine_2d(network, x, hidden_size, eps_tensor)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    scale_plus_one = network.add_elementwise(
        scale, one, trt.ElementWiseOperation.SUM).get_output(0)
    h = network.add_elementwise(
        h, scale_plus_one, trt.ElementWiseOperation.PROD).get_output(0)
    h = network.add_elementwise(
        h, shift, trt.ElementWiseOperation.SUM).get_output(0)
    return h


# ---------------------------------------------------------------------------
# 3-axis RoPE precompute (T, H, W) — baked as a constant
#
# Mirrors diffusers ``CosmosRotaryPosEmbed``. Each head_dim chunk
# corresponds to one axis and is computed as the standard rotate-half RoPE
# inv_freq formulation, then concatenated along the last axis. ``rope_scale``
# rescales the position indices per axis before computing the angles.
# ---------------------------------------------------------------------------

def _rope_table_axis(
    *,
    num_positions: int,
    axis_dim: int,
    rope_theta: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cos, sin)`` of shape ``[num_positions, axis_dim // 2]``.

    Standard rotate-half layout (LLaMA/Qwen/Cosmos): the cos/sin tables
    store the *half*-dim frequencies. The application step splits the
    incoming head_dim into ``(x1, x2)`` halves and computes
    ``(x1 * c - x2 * s, x2 * c + x1 * s)``.
    """
    if axis_dim <= 0 or num_positions <= 0:
        return (np.zeros((max(num_positions, 1), max(axis_dim // 2, 1)),
                          dtype=np.float32),
                np.zeros((max(num_positions, 1), max(axis_dim // 2, 1)),
                          dtype=np.float32))
    half = axis_dim // 2
    # inv_freq[j] = 1 / theta ** (2j / axis_dim)  for j in [0, half)
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, axis_dim, 2, dtype=np.float64) / axis_dim))
    positions = np.arange(num_positions, dtype=np.float64) * float(scale)
    angles = np.outer(positions, inv_freq)  # [num_positions, half]
    return (np.cos(angles).astype(np.float32),
            np.sin(angles).astype(np.float32))


def _build_3axis_rope_tables(
    *,
    t_patches: int,
    h_patches: int,
    w_patches: int,
    axes_dim: tuple[int, int, int],
    rope_scale: tuple[float, float, float],
    rope_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``cos, sin`` tables of shape ``[num_patches, head_dim // 2]``.

    The per-axis tables are tiled across the 3-D grid (T, H, W) and
    concatenated along the dim axis so the final layout matches the per-token
    rotation pattern used inside ``CosmosAttnProcessor2_5``.
    """
    ad_t, ad_h, ad_w = axes_dim
    cos_t, sin_t = _rope_table_axis(
        num_positions=t_patches, axis_dim=ad_t,
        rope_theta=rope_theta, scale=rope_scale[0])
    cos_h, sin_h = _rope_table_axis(
        num_positions=h_patches, axis_dim=ad_h,
        rope_theta=rope_theta, scale=rope_scale[1])
    cos_w, sin_w = _rope_table_axis(
        num_positions=w_patches, axis_dim=ad_w,
        rope_theta=rope_theta, scale=rope_scale[2])

    # Broadcast over the 3-D grid: each token (t, h, w) gets
    #   cos_t[t, :] | cos_h[h, :] | cos_w[w, :]
    cos_t_grid = np.broadcast_to(
        cos_t[:, None, None, :], (t_patches, h_patches, w_patches, ad_t // 2))
    cos_h_grid = np.broadcast_to(
        cos_h[None, :, None, :], (t_patches, h_patches, w_patches, ad_h // 2))
    cos_w_grid = np.broadcast_to(
        cos_w[None, None, :, :], (t_patches, h_patches, w_patches, ad_w // 2))
    sin_t_grid = np.broadcast_to(
        sin_t[:, None, None, :], (t_patches, h_patches, w_patches, ad_t // 2))
    sin_h_grid = np.broadcast_to(
        sin_h[None, :, None, :], (t_patches, h_patches, w_patches, ad_h // 2))
    sin_w_grid = np.broadcast_to(
        sin_w[None, None, :, :], (t_patches, h_patches, w_patches, ad_w // 2))

    cos_table = np.concatenate(
        [cos_t_grid, cos_h_grid, cos_w_grid], axis=-1).reshape(
            t_patches * h_patches * w_patches, -1).astype(np.float32)
    sin_table = np.concatenate(
        [sin_t_grid, sin_h_grid, sin_w_grid], axis=-1).reshape(
            t_patches * h_patches * w_patches, -1).astype(np.float32)
    return (np.ascontiguousarray(cos_table),
            np.ascontiguousarray(sin_table))


def _apply_rope_3axis(
    network: "trt.INetworkDefinition",
    x: "trt.ITensor",
    *,
    num_heads: int,
    head_dim: int,
    num_patches: int,
    cos_3d: "trt.ITensor",
    sin_3d: "trt.ITensor",
) -> "trt.ITensor":
    """Apply rotate-half RoPE using full per-axis cos/sin tables.

    ``cos_3d`` / ``sin_3d`` are pre-built constants of shape
    ``[1, num_patches, head_dim // 2]`` (the half-dim cache layout expected
    by ``IRotaryEmbeddingLayer``). We delegate to
    ``graph_ops.add_apply_rope_native_sequence`` which handles the
    [Sq, H*D] -> [1, H, Sq, D] -> rotate -> [Sq, H*D] dance.

    The 3-axis layout is encoded in the *table contents*, not the math:
    ``_build_3axis_rope_tables`` concatenates per-axis frequencies along the
    dim axis, so each per-token rotation pair already targets the correct
    axis.
    """
    return graph_ops.add_apply_rope_native_sequence(
        network,
        x,
        num_heads=num_heads,
        head_dim=head_dim,
        cos_cache_3d=cos_3d,
        sin_cache_3d=sin_3d,
        rotary_embedding_dim=head_dim,
        interleaved=False,
        sequence_length=num_patches,
    )


# ---------------------------------------------------------------------------
# Per-block weight key fan-out
#
# Diffusers ``CosmosTransformer3DModel`` serializes weights under the
# ``transformer_blocks.{i}.`` prefix. The canonical sub-keys are:
#
#   transformer_blocks.{i}.norm1.linear_1.{weight,bias}    [hidden, lora]
#   transformer_blocks.{i}.norm1.linear_2.{weight,bias}    [lora,   3*hidden]
#   transformer_blocks.{i}.norm1.linear_3.{weight,bias}    [hidden, 3*hidden]
#   transformer_blocks.{i}.attn1.to_q.weight               [hidden, hidden]
#   transformer_blocks.{i}.attn1.to_k.weight               [hidden, hidden]
#   transformer_blocks.{i}.attn1.to_v.weight               [hidden, hidden]
#   transformer_blocks.{i}.attn1.to_out.0.weight           [hidden, hidden]
#   transformer_blocks.{i}.attn1.norm_q.weight             [head_dim]
#   transformer_blocks.{i}.attn1.norm_k.weight             [head_dim]
#   transformer_blocks.{i}.norm2.linear_{1,2,3}.{weight,bias}
#   transformer_blocks.{i}.attn2.to_q.weight               [hidden, hidden]
#   transformer_blocks.{i}.attn2.to_k.weight               [text_embed, hidden]
#   transformer_blocks.{i}.attn2.to_v.weight               [text_embed, hidden]
#   transformer_blocks.{i}.attn2.to_out.0.weight           [hidden, hidden]
#   transformer_blocks.{i}.attn2.norm_q.weight             [head_dim]
#   transformer_blocks.{i}.norm3.linear_{1,2,3}.{weight,bias}
#   transformer_blocks.{i}.ff.net.0.proj.{weight,bias}     [hidden, ffn]
#   transformer_blocks.{i}.ff.net.2.{weight,bias}          [ffn, hidden]
#
# Global keys:
#
#   patch_embed.proj.{weight,bias}    [in_channels * pt*ph*pw, hidden]
#   time_embed.timesteps_proj.linear_1.{weight,bias}  [256, hidden]
#   time_embed.timesteps_proj.linear_2.{weight,bias}  [hidden, hidden]
#   time_embed.t_embedder.linear_1.{weight,bias}      [256, hidden]
#   time_embed.t_embedder.linear_2.{weight,bias}      [hidden, hidden]
#   time_embed.norm.weight                            (optional RMSNorm on temb)
#   norm_out.linear_1.{weight,bias}                   [hidden, lora]
#   norm_out.linear_2.{weight,bias}                   [lora,   2*hidden]
#   norm_out.linear_3.{weight,bias}                   [hidden, 2*hidden]
#   proj_out.{weight,bias}                            [hidden, out_channels * pt*ph*pw]
# ---------------------------------------------------------------------------


def _block_weights_or_none(
    weights: "WeightDict",
    i: int,
    *,
    require_attn_biases: bool = False,
) -> dict:
    """Pull out all per-block weight tensors that ``build_cosmos_dit_engine`` needs."""
    p = f"transformer_blocks.{i}"
    w = {}
    # AdaLN-LoRA (3 norms per block, each with linear_1/2/3).
    for nm in ("norm1", "norm2", "norm3"):
        w[f"{nm}.linear_1.weight"] = weights[f"{p}.{nm}.linear_1.weight"]
        w[f"{nm}.linear_1.bias"] = weights.get(f"{p}.{nm}.linear_1.bias")
        w[f"{nm}.linear_2.weight"] = weights[f"{p}.{nm}.linear_2.weight"]
        w[f"{nm}.linear_2.bias"] = weights.get(f"{p}.{nm}.linear_2.bias")
        w[f"{nm}.linear_3.weight"] = weights[f"{p}.{nm}.linear_3.weight"]
        w[f"{nm}.linear_3.bias"] = weights.get(f"{p}.{nm}.linear_3.bias")
    # Self-attention (attn1).
    w["attn1.to_q.weight"] = weights[f"{p}.attn1.to_q.weight"]
    w["attn1.to_k.weight"] = weights[f"{p}.attn1.to_k.weight"]
    w["attn1.to_v.weight"] = weights[f"{p}.attn1.to_v.weight"]
    w["attn1.to_out.0.weight"] = weights[f"{p}.attn1.to_out.0.weight"]
    w["attn1.norm_q.weight"] = weights.get(f"{p}.attn1.norm_q.weight")
    w["attn1.norm_k.weight"] = weights.get(f"{p}.attn1.norm_k.weight")
    # Cross-attention (attn2).
    w["attn2.to_q.weight"] = weights[f"{p}.attn2.to_q.weight"]
    w["attn2.to_k.weight"] = weights[f"{p}.attn2.to_k.weight"]
    w["attn2.to_v.weight"] = weights[f"{p}.attn2.to_v.weight"]
    w["attn2.to_out.0.weight"] = weights[f"{p}.attn2.to_out.0.weight"]
    w["attn2.norm_q.weight"] = weights.get(f"{p}.attn2.norm_q.weight")
    # FFN (gelu_pytorch_tanh -> GELU tanh).
    w["ff.net.0.proj.weight"] = weights[f"{p}.ff.net.0.proj.weight"]
    w["ff.net.0.proj.bias"] = weights.get(f"{p}.ff.net.0.proj.bias")
    w["ff.net.2.weight"] = weights[f"{p}.ff.net.2.weight"]
    w["ff.net.2.bias"] = weights.get(f"{p}.ff.net.2.bias")
    return w


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------


def build_cosmos_dit_engine(
    weights: "WeightDict",
    *,
    video_height: int,
    video_width: int,
    video_num_frames: int,
    hidden_size: int = COSMOS_14B_HIDDEN_SIZE,
    num_heads: int = COSMOS_14B_NUM_HEADS,
    head_dim: int = COSMOS_14B_HEAD_DIM,
    num_layers: int = COSMOS_14B_NUM_LAYERS,
    ffn_dim: int = COSMOS_14B_FFN_DIM,
    out_channels: int = COSMOS_14B_OUT_CHANNELS,
    text_embed_dim: int = COSMOS_14B_TEXT_EMBED_DIM,
    text_seq_len: int = COSMOS_14B_TEXT_SEQ_LEN,
    adaln_lora_dim: int = COSMOS_14B_ADALN_LORA_DIM,
    patch_size: tuple[int, int, int] = COSMOS_14B_PATCH_SIZE,
    rope_axes_dim: tuple[int, int, int] = COSMOS_14B_ROPE_AXES_DIM,
    rope_scale: tuple[float, float, float] = COSMOS_14B_ROPE_SCALE,
    rope_theta: float = COSMOS_14B_ROPE_THETA,
    eps: float = COSMOS_14B_NORM_EPS,
    vae_scale_spatial: int = COSMOS_14B_VAE_SCALE_SPATIAL,
    vae_scale_temporal: int = COSMOS_14B_VAE_SCALE_TEMPORAL,
    verbose: bool = False,
) -> bytes:
    """Build the Cosmos-Predict2 14B DiT denoiser TRT engine.

    Args:
        weights: Output of :func:`load_cosmos_dit_weights`. The transformer
            weights are expected to be transposed for TRT matmul (i.e.
            ``[in_dim, out_dim]`` order). RMSNorm/LayerNorm weights are
            flat 1-D arrays.
        video_height, video_width, video_num_frames: Pixel-space target. Used
            to compute the static engine I/O shapes via the VAE compression
            factors.
        hidden_size, num_heads, head_dim, num_layers, ffn_dim, out_channels,
        text_embed_dim, text_seq_len, adaln_lora_dim, patch_size,
        rope_axes_dim, rope_scale, rope_theta, eps:
            Architecture constants. Defaults match the 14B Video2World variant.
        vae_scale_spatial, vae_scale_temporal: Wan-AI VAE compression ratios
            (8x spatial, 4x temporal). Used to derive latent dimensions.
        verbose: TRT builder verbose log level.

    Returns:
        Serialized TRT engine plan as ``bytes``.
    """
    if num_heads * head_dim != hidden_size:
        raise ValueError(
            f"hidden_size={hidden_size} must equal num_heads*head_dim "
            f"({num_heads}*{head_dim}={num_heads * head_dim})")
    if sum(rope_axes_dim) != head_dim:
        raise ValueError(
            f"sum(rope_axes_dim)={sum(rope_axes_dim)} must equal head_dim "
            f"({head_dim}); got {rope_axes_dim}")
    pt, ph, pw = patch_size

    # Latent dims after VAE encode.
    t_lat = (video_num_frames - 1) // vae_scale_temporal + 1
    h_lat = video_height // vae_scale_spatial
    w_lat = video_width // vae_scale_spatial
    if t_lat % pt != 0 or h_lat % ph != 0 or w_lat % pw != 0:
        raise ValueError(
            f"Latent dims (t={t_lat}, h={h_lat}, w={w_lat}) must be divisible "
            f"by patch_size {patch_size}")
    t_patches = t_lat // pt
    h_patches = h_lat // ph
    w_patches = w_lat // pw
    num_patches = t_patches * h_patches * w_patches
    patch_dim = out_channels * pt * ph * pw

    # --- Build the TRT network skeleton ---
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # --- Inputs ---
    hidden_inp = network.add_input(
        "hidden_states", trt.float32, (num_patches, hidden_size))
    encoder_hidden_inp = network.add_input(
        "encoder_hidden_states", trt.float32, (text_seq_len, text_embed_dim))
    temb_inp = network.add_input(
        "temb", trt.float32, (1, hidden_size))
    embedded_timestep_inp = network.add_input(
        "embedded_timestep", trt.float32, (1, hidden_size))

    # --- Constants ---
    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=np.float32))

    # 3-axis RoPE tables, precomputed once and baked as constants.
    cos_full, sin_full = _build_3axis_rope_tables(
        t_patches=t_patches,
        h_patches=h_patches,
        w_patches=w_patches,
        axes_dim=rope_axes_dim,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    cos_const = graph_ops.add_constant(
        network, (1, num_patches, head_dim // 2),
        cos_full.reshape(1, num_patches, head_dim // 2))
    sin_const = graph_ops.add_constant(
        network, (1, num_patches, head_dim // 2),
        sin_full.reshape(1, num_patches, head_dim // 2))

    # Pre-activate temb / embedded_timestep with SiLU.
    temb_silu = _silu_2d(network, temb_inp)
    embedded_silu = _silu_2d(network, embedded_timestep_inp)

    hidden = hidden_inp

    # --- Per-block forward ---
    for layer_idx in range(num_layers):
        bw = _block_weights_or_none(weights, layer_idx)
        prefix = f"transformer_blocks.{layer_idx}"

        # === 1. Self-attention with AdaLN-Zero (norm1) ===
        h1, gate1 = _adaln_zero(
            network, hidden, temb_silu, embedded_silu,
            hidden_size=hidden_size,
            adaln_lora_dim=adaln_lora_dim,
            eps_tensor=eps_t,
            w_l1=bw["norm1.linear_1.weight"], b_l1=bw["norm1.linear_1.bias"],
            w_l2=bw["norm1.linear_2.weight"], b_l2=bw["norm1.linear_2.bias"],
            w_l3=bw["norm1.linear_3.weight"], b_l3=bw["norm1.linear_3.bias"],
        )
        # Q/K/V projections (no biases).
        q = graph_ops.add_matmul_rhs_constant(
            network, h1, hidden_size, hidden_size, bw["attn1.to_q.weight"])
        k = graph_ops.add_matmul_rhs_constant(
            network, h1, hidden_size, hidden_size, bw["attn1.to_k.weight"])
        v = graph_ops.add_matmul_rhs_constant(
            network, h1, hidden_size, hidden_size, bw["attn1.to_v.weight"])
        # Per-head RMSNorm on Q and K.
        nq = bw["attn1.norm_q.weight"]
        if nq is not None:
            q = graph_ops.add_rms_norm_per_head(
                network, q, num_heads, head_dim, nq, eps_t,
                sequence_length=num_patches)
        nk = bw["attn1.norm_k.weight"]
        if nk is not None:
            k = graph_ops.add_rms_norm_per_head(
                network, k, num_heads, head_dim, nk, eps_t,
                sequence_length=num_patches)
        # 3-axis RoPE on Q and K.
        q = _apply_rope_3axis(
            network, q,
            num_heads=num_heads, head_dim=head_dim,
            num_patches=num_patches,
            cos_3d=cos_const, sin_3d=sin_const)
        k = _apply_rope_3axis(
            network, k,
            num_heads=num_heads, head_dim=head_dim,
            num_patches=num_patches,
            cos_3d=cos_const, sin_3d=sin_const)
        ctx = graph_ops.add_attention_from_rows(
            network, q, k, v,
            num_heads=num_heads, head_dim=head_dim,
            q_seq=num_patches, kv_seq=num_patches,
            tag=f"{prefix}.attn1")
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx, hidden_size, hidden_size,
            bw["attn1.to_out.0.weight"])
        gated = network.add_elementwise(
            attn_out, gate1, trt.ElementWiseOperation.PROD).get_output(0)
        hidden = network.add_elementwise(
            hidden, gated, trt.ElementWiseOperation.SUM).get_output(0)

        # === 2. Cross-attention with AdaLN-Zero (norm2) ===
        h2, gate2 = _adaln_zero(
            network, hidden, temb_silu, embedded_silu,
            hidden_size=hidden_size,
            adaln_lora_dim=adaln_lora_dim,
            eps_tensor=eps_t,
            w_l1=bw["norm2.linear_1.weight"], b_l1=bw["norm2.linear_1.bias"],
            w_l2=bw["norm2.linear_2.weight"], b_l2=bw["norm2.linear_2.bias"],
            w_l3=bw["norm2.linear_3.weight"], b_l3=bw["norm2.linear_3.bias"],
        )
        cq = graph_ops.add_matmul_rhs_constant(
            network, h2, hidden_size, hidden_size, bw["attn2.to_q.weight"])
        ck = graph_ops.add_matmul_rhs_constant(
            network, encoder_hidden_inp, text_embed_dim, hidden_size,
            bw["attn2.to_k.weight"])
        cv = graph_ops.add_matmul_rhs_constant(
            network, encoder_hidden_inp, text_embed_dim, hidden_size,
            bw["attn2.to_v.weight"])
        nq2 = bw["attn2.norm_q.weight"]
        if nq2 is not None:
            cq = graph_ops.add_rms_norm_per_head(
                network, cq, num_heads, head_dim, nq2, eps_t,
                sequence_length=num_patches)
        cross_ctx = graph_ops.add_attention_from_rows(
            network, cq, ck, cv,
            num_heads=num_heads, head_dim=head_dim,
            q_seq=num_patches, kv_seq=text_seq_len,
            tag=f"{prefix}.attn2")
        cross_out = graph_ops.add_matmul_rhs_constant(
            network, cross_ctx, hidden_size, hidden_size,
            bw["attn2.to_out.0.weight"])
        gated2 = network.add_elementwise(
            cross_out, gate2, trt.ElementWiseOperation.PROD).get_output(0)
        hidden = network.add_elementwise(
            hidden, gated2, trt.ElementWiseOperation.SUM).get_output(0)

        # === 3. FFN with AdaLN-Zero (norm3) ===
        h3, gate3 = _adaln_zero(
            network, hidden, temb_silu, embedded_silu,
            hidden_size=hidden_size,
            adaln_lora_dim=adaln_lora_dim,
            eps_tensor=eps_t,
            w_l1=bw["norm3.linear_1.weight"], b_l1=bw["norm3.linear_1.bias"],
            w_l2=bw["norm3.linear_2.weight"], b_l2=bw["norm3.linear_2.bias"],
            w_l3=bw["norm3.linear_3.weight"], b_l3=bw["norm3.linear_3.bias"],
        )
        ffn_h = graph_ops.add_matmul_rhs_constant(
            network, h3, hidden_size, ffn_dim, bw["ff.net.0.proj.weight"])
        if bw["ff.net.0.proj.bias"] is not None:
            ffn_h = graph_ops.add_bias_sum(
                network, ffn_h, ffn_dim, bw["ff.net.0.proj.bias"])
        # gelu_pytorch_tanh = tanh-approximation GELU.
        ffn_h = graph_ops.add_gelu_new(network, ffn_h)
        ffn_h = graph_ops.add_matmul_rhs_constant(
            network, ffn_h, ffn_dim, hidden_size, bw["ff.net.2.weight"])
        if bw["ff.net.2.bias"] is not None:
            ffn_h = graph_ops.add_bias_sum(
                network, ffn_h, hidden_size, bw["ff.net.2.bias"])
        gated3 = network.add_elementwise(
            ffn_h, gate3, trt.ElementWiseOperation.PROD).get_output(0)
        hidden = network.add_elementwise(
            hidden, gated3, trt.ElementWiseOperation.SUM).get_output(0)

    # --- Final norm_out (2 chunks: shift, scale) + proj_out ---
    h_final = _adaln_final(
        network, hidden, temb_silu, embedded_silu,
        hidden_size=hidden_size,
        adaln_lora_dim=adaln_lora_dim,
        eps_tensor=eps_t,
        w_l1=weights["norm_out.linear_1.weight"],
        b_l1=weights.get("norm_out.linear_1.bias"),
        w_l2=weights["norm_out.linear_2.weight"],
        b_l2=weights.get("norm_out.linear_2.bias"),
        w_l3=weights["norm_out.linear_3.weight"],
        b_l3=weights.get("norm_out.linear_3.bias"),
    )
    output = graph_ops.add_matmul_rhs_constant(
        network, h_final, hidden_size, patch_dim,
        weights["proj_out.weight"])
    proj_b = weights.get("proj_out.bias")
    if proj_b is not None:
        output = graph_ops.add_bias_sum(network, output, patch_dim, proj_b)

    # Mark output (fp32 by contract).
    out_cast = network.add_cast(output, trt.float32).get_output(0)
    out_cast.name = "noise_pred"
    network.mark_output(out_cast)

    print(
        f"[cosmos-dit] Building TRT engine (hidden={hidden_size}, "
        f"layers={num_layers}, num_patches={num_patches}, "
        f"t_lat={t_lat}, h_lat={h_lat}, w_lat={w_lat}, "
        f"axes_dim={rope_axes_dim}, rope_scale={rope_scale}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for Cosmos DiT")
    return bytes(plan)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_cosmos_dit_weights(
    model_dir: str,
    *,
    hidden_size: int = COSMOS_14B_HIDDEN_SIZE,
    num_heads: int = COSMOS_14B_NUM_HEADS,
    head_dim: int = COSMOS_14B_HEAD_DIM,
    num_layers: int = COSMOS_14B_NUM_LAYERS,
    ffn_dim: int = COSMOS_14B_FFN_DIM,
    text_embed_dim: int = COSMOS_14B_TEXT_EMBED_DIM,
    adaln_lora_dim: int = COSMOS_14B_ADALN_LORA_DIM,
    out_channels: int = COSMOS_14B_OUT_CHANNELS,
    in_channels: int = COSMOS_14B_IN_CHANNELS,
    patch_size: tuple[int, int, int] = COSMOS_14B_PATCH_SIZE,
) -> "WeightDict":
    """Load ``CosmosTransformer3DModel`` weights from the diffusers
    ``transformer/`` sub-directory of a Cosmos-Predict2 checkpoint.

    Walks the safetensors shards via the shared
    :func:`checkpoint_mapper._open_safetensors` helper. All projection
    weights are transposed at load time so the engine builder can use the
    plain ``x @ W`` form (``W`` in ``[in_dim, out_dim]`` order). Norm / RMS
    weights are returned as flat 1-D arrays.

    Args:
        model_dir: Path to the ``transformer/`` directory inside the
            diffusers checkpoint snapshot.
        hidden_size, num_heads, head_dim, num_layers, ffn_dim, text_embed_dim,
        adaln_lora_dim, out_channels, in_channels, patch_size:
            Architecture constants used to choose which tensors to load.
            Tensors that are absent from the safetensors index but referenced
            by the builder will raise ``KeyError`` here -- the builder also
            tolerates ``None`` for optional biases / norm-q-only attentions.

    Returns:
        A :class:`WeightDict` keyed by the names the builder expects.
    """
    from ...checkpoint_mapper import (
        WeightDict, _open_safetensors, _load_tensor, _has_tensor)

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    weights = WeightDict()

    def _t(name: str) -> np.ndarray:
        """Load a 2-D weight and transpose ``[out, in] -> [in, out]``."""
        w = _load_tensor(readers, name)
        if w.ndim != 2:
            raise ValueError(
                f"Expected 2-D weight for {name!r}, got shape {w.shape}")
        return np.ascontiguousarray(w.T, dtype=np.float32)

    def _f(name: str) -> np.ndarray:
        """Load a flat (1-D) tensor as fp32."""
        return _load_tensor(readers, name).astype(np.float32)

    def _maybe_t(name: str) -> np.ndarray | None:
        return _t(name) if _has_tensor(readers, name) else None

    def _maybe_f(name: str) -> np.ndarray | None:
        return _f(name) if _has_tensor(readers, name) else None

    # ---- per-block --------------------------------------------------------
    for i in range(num_layers):
        p = f"transformer_blocks.{i}"
        for nm in ("norm1", "norm2", "norm3"):
            weights[f"{p}.{nm}.linear_1.weight"] = _t(f"{p}.{nm}.linear_1.weight")
            weights[f"{p}.{nm}.linear_2.weight"] = _t(f"{p}.{nm}.linear_2.weight")
            weights[f"{p}.{nm}.linear_3.weight"] = _t(f"{p}.{nm}.linear_3.weight")
            b = _maybe_f(f"{p}.{nm}.linear_1.bias")
            if b is not None:
                weights[f"{p}.{nm}.linear_1.bias"] = b
            b = _maybe_f(f"{p}.{nm}.linear_2.bias")
            if b is not None:
                weights[f"{p}.{nm}.linear_2.bias"] = b
            b = _maybe_f(f"{p}.{nm}.linear_3.bias")
            if b is not None:
                weights[f"{p}.{nm}.linear_3.bias"] = b
        # Self-attention.
        weights[f"{p}.attn1.to_q.weight"] = _t(f"{p}.attn1.to_q.weight")
        weights[f"{p}.attn1.to_k.weight"] = _t(f"{p}.attn1.to_k.weight")
        weights[f"{p}.attn1.to_v.weight"] = _t(f"{p}.attn1.to_v.weight")
        weights[f"{p}.attn1.to_out.0.weight"] = _t(f"{p}.attn1.to_out.0.weight")
        nq = _maybe_f(f"{p}.attn1.norm_q.weight")
        if nq is not None:
            weights[f"{p}.attn1.norm_q.weight"] = nq
        nk = _maybe_f(f"{p}.attn1.norm_k.weight")
        if nk is not None:
            weights[f"{p}.attn1.norm_k.weight"] = nk
        # Cross-attention.
        weights[f"{p}.attn2.to_q.weight"] = _t(f"{p}.attn2.to_q.weight")
        weights[f"{p}.attn2.to_k.weight"] = _t(f"{p}.attn2.to_k.weight")
        weights[f"{p}.attn2.to_v.weight"] = _t(f"{p}.attn2.to_v.weight")
        weights[f"{p}.attn2.to_out.0.weight"] = _t(f"{p}.attn2.to_out.0.weight")
        nq2 = _maybe_f(f"{p}.attn2.norm_q.weight")
        if nq2 is not None:
            weights[f"{p}.attn2.norm_q.weight"] = nq2
        # FFN (gelu_pytorch_tanh).
        weights[f"{p}.ff.net.0.proj.weight"] = _t(f"{p}.ff.net.0.proj.weight")
        weights[f"{p}.ff.net.2.weight"] = _t(f"{p}.ff.net.2.weight")
        b = _maybe_f(f"{p}.ff.net.0.proj.bias")
        if b is not None:
            weights[f"{p}.ff.net.0.proj.bias"] = b
        b = _maybe_f(f"{p}.ff.net.2.bias")
        if b is not None:
            weights[f"{p}.ff.net.2.bias"] = b

    # ---- final norm_out + proj_out ---------------------------------------
    weights["norm_out.linear_1.weight"] = _t("norm_out.linear_1.weight")
    weights["norm_out.linear_2.weight"] = _t("norm_out.linear_2.weight")
    weights["norm_out.linear_3.weight"] = _t("norm_out.linear_3.weight")
    b = _maybe_f("norm_out.linear_1.bias")
    if b is not None:
        weights["norm_out.linear_1.bias"] = b
    b = _maybe_f("norm_out.linear_2.bias")
    if b is not None:
        weights["norm_out.linear_2.bias"] = b
    b = _maybe_f("norm_out.linear_3.bias")
    if b is not None:
        weights["norm_out.linear_3.bias"] = b

    weights["proj_out.weight"] = _t("proj_out.weight")
    b = _maybe_f("proj_out.bias")
    if b is not None:
        weights["proj_out.bias"] = b

    # ---- patch_embed + time_embed (loaded but consumed externally) -------
    # The patch embedding is a Linear(in_channels * pt*ph*pw, hidden_size) in
    # the diffusers reference. The C++ preprocessor / Python runner applies
    # this before the engine runs, so we expose the raw weights for callers
    # that need them (matches the Wan T2V plugin's preprocessor_weights
    # contract).
    if _has_tensor(readers, "patch_embed.proj.weight"):
        weights["patch_embed.proj.weight"] = _load_tensor(
            readers, "patch_embed.proj.weight").astype(np.float32)
    if _has_tensor(readers, "patch_embed.proj.bias"):
        weights["patch_embed.proj.bias"] = _load_tensor(
            readers, "patch_embed.proj.bias").astype(np.float32)

    # Time embedding (sinusoidal + 2x MLP). Two MLPs: `timesteps_proj`
    # produces the per-block ``temb``, and ``t_embedder`` produces the
    # ``embedded_timestep`` residual. Layout in the safetensors is the
    # diffusers default ``linear_1`` / ``linear_2`` naming.
    for prefix in ("time_embed.timesteps_proj", "time_embed.t_embedder"):
        for sub in ("linear_1", "linear_2"):
            wk = f"{prefix}.{sub}.weight"
            bk = f"{prefix}.{sub}.bias"
            if _has_tensor(readers, wk):
                weights[wk] = _load_tensor(readers, wk).astype(np.float32)
            if _has_tensor(readers, bk):
                weights[bk] = _load_tensor(readers, bk).astype(np.float32)
    # Optional RMSNorm on the temb output (some diffusers variants apply
    # a small norm; if absent the caller treats it as identity).
    if _has_tensor(readers, "time_embed.norm.weight"):
        weights["time_embed.norm.weight"] = _load_tensor(
            readers, "time_embed.norm.weight").astype(np.float32)

    return weights
