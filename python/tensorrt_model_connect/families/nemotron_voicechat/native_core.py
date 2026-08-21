# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoiceChat-owned native perception, thinker, and RNNT builders.

NemotronH (NVIDIA) uses a heterogeneous layer stack with three layer types
defined by hybrid_override_pattern (e.g. "M-M-M-MM-M-M-M*-..."):
  M = Mamba-2 SSM layer
  - = MLP layer (up_proj -> relu2 -> down_proj)
  * = Attention layer (GQA, no RoPE, no bias)

Key differences from Mamba-1 (existing mamba.py):
  Mamba-2 uses State Space Duality (SSD):
    - in_proj -> split into [gate, hidden_B_C, dt]
    - conv1d over hidden_B_C (d_inner + 2*n_groups*d_state channels)
    - After conv+SiLU, split hidden_B_C -> [hidden, B, C]
    - Multi-head SSM (nheads * headdim = d_inner)
    - A is a scalar per head (not per d_inner like Mamba-1)
    - dt from in_proj directly (no separate x_proj/dt_proj)
    - Gated RMSNorm on SSM output: norm(y) * silu(gate)
    - SSM state: [nheads, headdim, d_state] (headdim-aware)

NemotronH Nano 9B: 56 layers (27 mamba2 + 25 mlp + 4 attention)
  - MLP layers: up_proj -> relu2 -> down_proj (NO gate_proj)
  - Attention layers: q/k/v/o_proj (GQA, no RoPE, no bias)

Weight key mapping (HF -> engine):
  backbone.embeddings.weight                           -> embedding
  backbone.layers.{i}.norm.weight                      -> layer.{i}.norm
  backbone.layers.{i}.mixer.in_proj.weight             -> Mamba-2 in_proj
  backbone.layers.{i}.mixer.conv1d.weight/bias         -> Mamba-2 conv state
  backbone.layers.{i}.mixer.dt_bias                    -> Mamba-2 timestep bias
  backbone.layers.{i}.mixer.A_log                      -> Mamba-2 SSM A
  backbone.layers.{i}.mixer.D                          -> Mamba-2 skip connection
  backbone.layers.{i}.mixer.norm.weight                -> Mamba-2 gated RMSNorm
  backbone.layers.{i}.mixer.out_proj.weight            -> Mamba-2 output proj
  backbone.layers.{i}.mixer.up_proj.weight             -> MLP up
  backbone.layers.{i}.mixer.down_proj.weight           -> MLP down
  backbone.layers.{i}.mixer.q/k/v/o_proj.weight        -> Attention QKV + out
  backbone.norm_f.weight                               -> final_norm
  lm_head.weight                                       -> w_lm_head
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from . import graph_blocks


trt = trt_compat.get_trt()


def _disable_tf32(builder_config) -> None:
    if hasattr(builder_config, "clear_flag") and hasattr(trt, "BuilderFlag"):
        builder_config.clear_flag(trt.BuilderFlag.TF32)


def _parse_layer_types(pattern: str) -> list[str]:
    """Parse hybrid_override_pattern: M=mamba2, -=mlp, *=attention."""
    mapping = {"M": "mamba2", "-": "mlp", "*": "attention"}
    return [mapping[ch] for ch in pattern if ch in mapping]


def _load_optional_bias(readers: list, name: str, size: int) -> np.ndarray:
    if _has_tensor(readers, name):
        value = _load_tensor(readers, name)
        if value.shape != (size,):
            raise ValueError(f"VoiceChat tensor {name} has shape {value.shape}; expected {(size,)}")
        return value.astype(np.float32)
    return np.zeros(size, dtype=np.float32)


def load_perception_weights(
    model_dir: str,
    stt_config: dict,
) -> WeightDict:
    """Map the checkpoint's FastConformer, projection, and mel frontend."""
    readers = _open_safetensors(Path(model_dir))
    perception = stt_config["perception"]
    encoder = perception["encoder"]
    preprocessor = perception["preprocessor"]
    hidden = int(encoder["d_model"])
    output_dim = int(perception["output_dim"])
    layers = int(encoder["n_layers"])
    heads = int(encoder["n_heads"])
    mel_bins = int(preprocessor["features"])
    ffn = int(encoder["ff_expansion_factor"]) * hidden
    kernel = int(encoder["conv_kernel_size"])
    sub_channels = int(encoder["subsampling_conv_channels"])
    prefix = "stt_model.perception.encoder"
    weights = WeightDict()
    weights.update(
        {
            "_enc_layers": layers,
            "_enc_heads": heads,
            "_enc_ffn": ffn,
            "_hidden": hidden,
            "_output_dim": output_dim,
            "_mel_bins": mel_bins,
            "_kern": kernel,
            "_sub_ch": sub_channels,
            "_head_dim": hidden // heads,
            "_streaming_cache_left": int(encoder.get("att_context_size", [70, 0])[0]),
        }
    )
    weights["enc_sub_conv0_w"] = _load_tensor(readers, f"{prefix}.pre_encode.conv.0.weight").astype(
        np.float32
    )
    weights["enc_sub_conv0_b"] = _load_tensor(readers, f"{prefix}.pre_encode.conv.0.bias").astype(
        np.float32
    )
    for stage, (depthwise, pointwise) in enumerate(((2, 3), (5, 6))):
        weights[f"enc_sub_dw{stage}_w"] = _load_tensor(
            readers, f"{prefix}.pre_encode.conv.{depthwise}.weight"
        ).astype(np.float32)
        weights[f"enc_sub_dw{stage}_b"] = _load_tensor(
            readers, f"{prefix}.pre_encode.conv.{depthwise}.bias"
        ).astype(np.float32)
        weights[f"enc_sub_pw{stage}_w"] = _load_tensor(
            readers, f"{prefix}.pre_encode.conv.{pointwise}.weight"
        ).astype(np.float32)
        weights[f"enc_sub_pw{stage}_b"] = _load_tensor(
            readers, f"{prefix}.pre_encode.conv.{pointwise}.bias"
        ).astype(np.float32)
    weights["enc_sub_out_w"] = _transpose_2d(
        _load_tensor(readers, f"{prefix}.pre_encode.out.weight"), "subsampling"
    )
    weights["enc_sub_out_b"] = _load_tensor(readers, f"{prefix}.pre_encode.out.bias").astype(
        np.float32
    )

    for layer in range(layers):
        source = f"{prefix}.layers.{layer}"
        target = f"el.{layer}"
        for logical, checkpoint in (
            ("w_q", "linear_q"),
            ("w_k", "linear_k"),
            ("w_v", "linear_v"),
            ("w_o", "linear_out"),
        ):
            weights[f"{target}.{logical}"] = _transpose_2d(
                _load_tensor(readers, f"{source}.self_attn.{checkpoint}.weight"), checkpoint
            )
            weights[f"{target}.b_{logical[-1]}"] = _load_optional_bias(
                readers, f"{source}.self_attn.{checkpoint}.bias", hidden
            )
        weights[f"{target}.pos_bias_u"] = _load_tensor(
            readers, f"{source}.self_attn.pos_bias_u"
        ).astype(np.float32)
        weights[f"{target}.pos_bias_v"] = _load_tensor(
            readers, f"{source}.self_attn.pos_bias_v"
        ).astype(np.float32)
        weights[f"{target}.w_pos"] = _transpose_2d(
            _load_tensor(readers, f"{source}.self_attn.linear_pos.weight"), "linear_pos"
        )
        for logical, checkpoint in (
            ("norm_sa", "norm_self_att"),
            ("norm_conv", "norm_conv"),
            ("norm_out", "norm_out"),
        ):
            weights[f"{target}.{logical}"] = _load_tensor(
                readers, f"{source}.{checkpoint}.weight"
            ).astype(np.float32)
            weights[f"{target}.{logical}_b"] = _load_tensor(
                readers, f"{source}.{checkpoint}.bias"
            ).astype(np.float32)
        for logical, checkpoint, norm_name in (
            ("ff1", "feed_forward1", "norm_feed_forward1"),
            ("ff2", "feed_forward2", "norm_feed_forward2"),
        ):
            weights[f"{target}.{logical}.w1"] = _transpose_2d(
                _load_tensor(readers, f"{source}.{checkpoint}.linear1.weight"),
                f"{logical}.linear1",
            )
            weights[f"{target}.{logical}.b1"] = _load_optional_bias(
                readers, f"{source}.{checkpoint}.linear1.bias", ffn
            )
            weights[f"{target}.{logical}.w2"] = _transpose_2d(
                _load_tensor(readers, f"{source}.{checkpoint}.linear2.weight"),
                f"{logical}.linear2",
            )
            weights[f"{target}.{logical}.b2"] = _load_optional_bias(
                readers, f"{source}.{checkpoint}.linear2.bias", hidden
            )
            weights[f"{target}.{logical}.norm"] = _load_tensor(
                readers, f"{source}.{norm_name}.weight"
            ).astype(np.float32)
            weights[f"{target}.{logical}.norm_b"] = _load_tensor(
                readers, f"{source}.{norm_name}.bias"
            ).astype(np.float32)
        for logical, checkpoint, size in (
            ("cpw1", "pointwise_conv1", 2 * hidden),
            ("cdw", "depthwise_conv", hidden),
            ("cpw2", "pointwise_conv2", hidden),
        ):
            weights[f"{target}.{logical}_w"] = _load_tensor(
                readers, f"{source}.conv.{checkpoint}.weight"
            ).astype(np.float32)
            weights[f"{target}.{logical}_b"] = _load_optional_bias(
                readers, f"{source}.conv.{checkpoint}.bias", size
            )
        weights[f"{target}.bn_w"] = _load_tensor(
            readers, f"{source}.conv.batch_norm.weight"
        ).astype(np.float32)
        weights[f"{target}.bn_b"] = _load_tensor(readers, f"{source}.conv.batch_norm.bias").astype(
            np.float32
        )
        weights[f"{target}.bn_m"] = np.zeros(hidden, dtype=np.float32)
        weights[f"{target}.bn_v"] = np.ones(hidden, dtype=np.float32)

    weights["perception_proj"] = _transpose_2d(
        _load_tensor(readers, "stt_model.perception.proj.weight"), "perception.proj"
    )
    weights["perception_proj_bias"] = _load_tensor(
        readers, "stt_model.perception.proj.bias"
    ).astype(np.float32)
    filterbank = _load_tensor(readers, "stt_model.perception.preprocessor.featurizer.fb")
    if filterbank.shape != (1, mel_bins, 257):
        raise ValueError(f"VoiceChat mel filterbank shape mismatch: {filterbank.shape}")
    weights["mel_filterbank"] = np.ascontiguousarray(filterbank[0].T, dtype=np.float32)
    weights["mel_window"] = _load_tensor(
        readers, "stt_model.perception.preprocessor.featurizer.window"
    ).astype(np.float32)
    return weights


def load_rnnt_weights(model_dir: str) -> WeightDict:
    """Map the embedded 1024-token RNNT predictor and joint branches."""
    readers = _open_safetensors(Path(model_dir))
    weights = WeightDict()
    prefix = "stt_model.rnnt_decoder.prediction"
    embedding = _load_tensor(readers, f"{prefix}.embed.weight").astype(np.float32)
    pred_hidden = int(embedding.shape[1])
    vocab_total = int(embedding.shape[0])
    weights["pred_embedding"] = embedding
    layers = 0
    source = f"{prefix}.dec_rnn.lstm"
    while _has_tensor(readers, f"{source}.weight_ih_l{layers}"):
        weight_ih = _load_tensor(readers, f"{source}.weight_ih_l{layers}")
        weight_hh = _load_tensor(readers, f"{source}.weight_hh_l{layers}")
        bias = _load_tensor(readers, f"{source}.bias_ih_l{layers}") + _load_tensor(
            readers, f"{source}.bias_hh_l{layers}"
        )
        expected = (4 * pred_hidden, pred_hidden)
        if weight_ih.shape != expected or weight_hh.shape != expected:
            raise ValueError(
                f"VoiceChat RNNT LSTM layer {layers} shape mismatch: "
                f"{weight_ih.shape}, {weight_hh.shape} vs {expected}"
            )
        weights[f"pred.{layers}.w_ih_t"] = np.ascontiguousarray(weight_ih.T)
        weights[f"pred.{layers}.w_hh_t"] = np.ascontiguousarray(weight_hh.T)
        weights[f"pred.{layers}.bias"] = bias.astype(np.float32).reshape(1, -1)
        layers += 1
    if layers == 0:
        raise ValueError("VoiceChat checkpoint contains no RNNT predictor layers")
    joint = "stt_model.rnnt_joint"
    weights["joint_enc_w"] = _transpose_2d(
        _load_tensor(readers, f"{joint}.enc.weight"), "rnnt_joint.enc"
    )
    weights["joint_enc_b"] = _load_tensor(readers, f"{joint}.enc.bias").astype(np.float32)
    weights["joint_pred_w"] = _transpose_2d(
        _load_tensor(readers, f"{joint}.pred.weight"), "rnnt_joint.pred"
    )
    weights["joint_pred_b"] = _load_tensor(readers, f"{joint}.pred.bias").astype(np.float32)
    weights["joint_out_w"] = _transpose_2d(
        _load_tensor(readers, f"{joint}.joint_net.2.weight"), "rnnt_joint.output"
    )
    weights["joint_out_b"] = _load_tensor(readers, f"{joint}.joint_net.2.bias").astype(np.float32)
    weights.update(
        {
            "_pred_hidden": pred_hidden,
            "_pred_layers": layers,
            "_vocab_total": vocab_total,
            "_vocab": vocab_total - 1,
            "_blank_id": vocab_total - 1,
            "_encoder_hidden": int(weights["joint_enc_w"].shape[0]),
            "_joint_hidden": int(weights["joint_enc_w"].shape[1]),
            "_joint_activation": "relu",
        }
    )
    return weights


def _add_lstm_cell(
    network,
    value,
    previous_h,
    previous_c,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    *,
    dtype=np.float32,
):
    input_weights = graph_ops.add_constant(
        network, (hidden_size, 4 * hidden_size), weights[f"{prefix}.w_ih_t"], dtype=dtype
    )
    recurrent_weights = graph_ops.add_constant(
        network, (hidden_size, 4 * hidden_size), weights[f"{prefix}.w_hh_t"], dtype=dtype
    )
    bias = graph_ops.add_constant(
        network, (1, 4 * hidden_size), weights[f"{prefix}.bias"], dtype=dtype
    )
    projected = network.add_matrix_multiply(
        value, trt.MatrixOperation.NONE, input_weights, trt.MatrixOperation.NONE
    ).get_output(0)
    recurrent = network.add_matrix_multiply(
        previous_h,
        trt.MatrixOperation.NONE,
        recurrent_weights,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    gates = network.add_elementwise(projected, recurrent, trt.ElementWiseOperation.SUM).get_output(
        0
    )
    gates = network.add_elementwise(gates, bias, trt.ElementWiseOperation.SUM).get_output(0)
    slices = [
        network.add_slice(
            gates,
            start=(0, index * hidden_size),
            shape=(1, hidden_size),
            stride=(1, 1),
        ).get_output(0)
        for index in range(4)
    ]
    input_gate = network.add_activation(slices[0], trt.ActivationType.SIGMOID).get_output(0)
    forget_gate = network.add_activation(slices[1], trt.ActivationType.SIGMOID).get_output(0)
    cell_gate = network.add_activation(slices[2], trt.ActivationType.TANH).get_output(0)
    output_gate = network.add_activation(slices[3], trt.ActivationType.SIGMOID).get_output(0)
    retained = network.add_elementwise(
        forget_gate, previous_c, trt.ElementWiseOperation.PROD
    ).get_output(0)
    update = network.add_elementwise(
        input_gate, cell_gate, trt.ElementWiseOperation.PROD
    ).get_output(0)
    next_c = network.add_elementwise(retained, update, trt.ElementWiseOperation.SUM).get_output(0)
    activated_c = network.add_activation(next_c, trt.ActivationType.TANH).get_output(0)
    next_h = network.add_elementwise(
        output_gate, activated_c, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return next_h, next_c


def build_rnnt_predictor(
    weights: WeightDict,
    *,
    verbose: bool = False,
) -> bytes:
    hidden_size = int(weights["_pred_hidden"])
    num_layers = int(weights["_pred_layers"])
    vocab_size = int(weights["_vocab_total"])
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    _disable_tf32(builder_config)
    token_id = network.add_input("token_id", trt.int32, (1,))
    embedding = graph_ops.add_constant(
        network, (vocab_size, hidden_size), weights["pred_embedding"]
    )
    hidden = network.add_gather(embedding, token_id, 0).get_output(0)
    next_h = []
    next_c = []
    for layer in range(num_layers):
        state_h = network.add_input(f"state_h_{layer}", trt.float32, (1, hidden_size))
        state_c = network.add_input(f"state_c_{layer}", trt.float32, (1, hidden_size))
        hidden, cell = _add_lstm_cell(
            network,
            hidden,
            state_h,
            state_c,
            weights,
            f"pred.{layer}",
            hidden_size,
        )
        next_h.append(hidden)
        next_c.append(cell)
    # The predictor output equals the last layer's next hidden state, but both
    # are separate public bindings.  Marking the same ITensor twice renames the
    # first binding and drops ``pred_output`` from the serialized engine.
    output = network.add_identity(hidden).get_output(0)
    output.name = "pred_output"
    network.mark_output(output)
    for layer, (state_h, state_c) in enumerate(zip(next_h, next_c, strict=True)):
        state_h.name = f"next_h_{layer}"
        state_c.name = f"next_c_{layer}"
        network.mark_output(state_h)
        network.mark_output(state_c)
    if verbose:
        print(
            f"[trtmc build] Building VoiceChat RNNT predictor ({num_layers}L, h={hidden_size})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("VoiceChat RNNT predictor build failed")
    return bytes(plan)


def build_rnnt_joint(
    weights: WeightDict,
    *,
    verbose: bool = False,
) -> bytes:
    encoder_hidden = int(weights["_encoder_hidden"])
    predictor_hidden = int(weights["_pred_hidden"])
    joint_hidden = int(weights["_joint_hidden"])
    vocab_size = int(weights["_vocab_total"])
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    _disable_tf32(builder_config)
    encoder = network.add_input("encoder_frame", trt.float32, (1, encoder_hidden))
    predictor = network.add_input("pred_output", trt.float32, (1, predictor_hidden))
    enc_projection = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network,
            encoder,
            encoder_hidden,
            joint_hidden,
            weights["joint_enc_w"],
        ),
        joint_hidden,
        weights["joint_enc_b"],
    )
    pred_projection = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network,
            predictor,
            predictor_hidden,
            joint_hidden,
            weights["joint_pred_w"],
        ),
        joint_hidden,
        weights["joint_pred_b"],
    )
    joint = network.add_elementwise(
        enc_projection, pred_projection, trt.ElementWiseOperation.SUM
    ).get_output(0)
    joint = network.add_activation(joint, trt.ActivationType.RELU).get_output(0)
    logits = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network,
            joint,
            joint_hidden,
            vocab_size,
            weights["joint_out_w"],
        ),
        vocab_size,
        weights["joint_out_b"],
    )
    logits.name = "logits"
    network.mark_output(logits)
    if verbose:
        print(
            f"[trtmc build] Building VoiceChat RNNT joint "
            f"(enc={encoder_hidden}, pred={predictor_hidden}, joint={joint_hidden})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("VoiceChat RNNT joint build failed")
    return bytes(plan)


class VoiceChatThinkerBuilder:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        readers = _open_safetensors(Path(model_dir))

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        raw = config.raw

        # Parse layer types from hybrid_override_pattern
        pattern = raw.get("hybrid_override_pattern", "M" * num_layers)
        layer_types = _parse_layer_types(pattern)
        assert len(layer_types) == num_layers, (
            f"Pattern length {len(layer_types)} != num_hidden_layers {num_layers}"
        )

        # Mamba-2 dimensions
        mamba_num_heads = raw.get("mamba_num_heads", 64)
        mamba_head_dim = raw.get("mamba_head_dim", 64)
        d_inner = mamba_num_heads * mamba_head_dim
        n_groups = raw.get("n_groups", 8)
        d_state = raw.get("ssm_state_size", raw.get("mamba_state_dim", 128))
        d_conv = raw.get("conv_kernel", 4)
        conv_dim = d_inner + 2 * n_groups * d_state

        # MLP dimensions
        mlp_intermediate = config.intermediate_size

        # Attention dimensions
        q_dim = num_heads * head_dim
        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "stt_model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        mamba_count = 0
        attn_count = 0

        for layer_idx in range(num_layers):
            lt = layer_types[layer_idx]
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"stt_model.llm.layers.{layer_idx}"

            # RMSNorm (all layer types)
            norm = _load_tensor(readers, f"{hf_prefix}.norm.weight")
            weights[f"{prefix}.input_norm"] = norm.astype(np.float32)

            if lt == "mamba2":
                # in_proj: [proj_size, hidden] where proj_size = d_inner + conv_dim + mamba_num_heads
                in_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.in_proj.weight")
                weights[f"{prefix}.mamba_in_proj"] = _transpose_2d(in_proj_raw, "mamba_in_proj")

                # conv1d: [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
                conv1d_w = _load_tensor(readers, f"{hf_prefix}.mixer.conv1d.weight")
                weights[f"{prefix}.conv1d_weight"] = conv1d_w.reshape(conv_dim, d_conv).astype(
                    np.float32
                )

                conv1d_b = _load_tensor(readers, f"{hf_prefix}.mixer.conv1d.bias")
                weights[f"{prefix}.conv1d_bias"] = conv1d_b.astype(np.float32)

                # out_proj: [hidden, d_inner]
                out_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.out_proj.weight")
                weights[f"{prefix}.mamba_out_proj"] = _transpose_2d(out_proj_raw, "mamba_out_proj")

                # A_log: [mamba_num_heads]
                A_log = _load_tensor(readers, f"{hf_prefix}.mixer.A_log")
                A = -np.exp(A_log.astype(np.float32))
                weights[f"{prefix}.A"] = A

                # D: [mamba_num_heads]
                D = _load_tensor(readers, f"{hf_prefix}.mixer.D")
                weights[f"{prefix}.D"] = D.astype(np.float32)

                # dt_bias: [mamba_num_heads]
                dt_bias = _load_tensor(readers, f"{hf_prefix}.mixer.dt_bias")
                weights[f"{prefix}.dt_bias"] = dt_bias.astype(np.float32)

                # Gated RMSNorm: [d_inner]
                norm_key = f"{hf_prefix}.mixer.norm.weight"
                if _has_tensor(readers, norm_key):
                    weights[f"{prefix}.mamba_norm"] = _load_tensor(readers, norm_key).astype(
                        np.float32
                    )
                else:
                    weights[f"{prefix}.mamba_norm"] = np.ones(d_inner, dtype=np.float32)

                mamba_count += 1

            elif lt == "mlp":
                # MLP: up_proj -> relu2 -> down_proj (NO gate_proj)
                up_raw = _load_tensor(readers, f"{hf_prefix}.mixer.up_proj.weight")
                down_raw = _load_tensor(readers, f"{hf_prefix}.mixer.down_proj.weight")
                weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
                weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")

            elif lt == "attention":
                # Attention: q/k/v/o projections (no bias, no RoPE)
                q_raw = _load_tensor(readers, f"{hf_prefix}.mixer.q_proj.weight")
                k_raw = _load_tensor(readers, f"{hf_prefix}.mixer.k_proj.weight")
                v_raw = _load_tensor(readers, f"{hf_prefix}.mixer.v_proj.weight")
                o_raw = _load_tensor(readers, f"{hf_prefix}.mixer.o_proj.weight")

                q_t = _transpose_2d(q_raw, "q_proj")
                k_t = _transpose_2d(k_raw, "k_proj")
                v_t = _transpose_2d(v_raw, "v_proj")
                o_t = _transpose_2d(o_raw, "o_proj")

                # Compact GQA/MQA K/V

                weights[f"{prefix}.w_q"] = q_t
                weights[f"{prefix}.w_k"] = k_t
                weights[f"{prefix}.w_v"] = v_t
                weights[f"{prefix}.w_o"] = o_t

                attn_count += 1

        # Final norm
        final_norm_key = "stt_model.llm.norm_f.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "stt_model.lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_lm_head"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_lm_head"] = _transpose_2d(embedding.copy(), "embedding_tied")

        function_head = _load_tensor(readers, "stt_model.function_head.weight")
        if function_head.shape != (vocab, hidden):
            raise ValueError(
                "VoiceChat function head shape mismatch: "
                f"{function_head.shape} != {(vocab, hidden)}"
            )
        weights["w_function_head"] = _transpose_2d(function_head, "function_head")
        weights["_duplex_text_weight"] = 1.0
        weights["_duplex_audio_weight"] = 1.0
        weights["_duplex_function_weight"] = 2.0

        # Metadata for engine builder
        weights["_layer_types"] = layer_types
        weights["_d_inner"] = d_inner
        weights["_d_state"] = d_state
        weights["_d_conv"] = d_conv
        weights["_conv_dim"] = conv_dim
        weights["_mamba_num_heads"] = mamba_num_heads
        weights["_mamba_head_dim"] = mamba_head_dim
        weights["_n_groups"] = n_groups
        weights["_num_mamba_layers"] = mamba_count
        weights["_num_attention_layers"] = attn_count
        weights["_attention_size"] = q_dim
        weights["_mlp_size"] = mlp_intermediate

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        verbose: bool = False,
    ) -> bytes:
        """Build hybrid TRT engine with heterogeneous layer stack."""
        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers

        layer_types: list[str] = weights["_layer_types"]
        d_inner: int = weights["_d_inner"]
        d_state: int = weights["_d_state"]
        d_conv: int = weights["_d_conv"]
        conv_dim: int = weights["_conv_dim"]
        mamba_num_heads: int = weights["_mamba_num_heads"]
        mamba_head_dim: int = weights["_mamba_head_dim"]
        n_groups: int = weights["_n_groups"]
        num_mamba: int = weights["_num_mamba_layers"]
        num_attn: int = weights["_num_attention_layers"]
        attention_size: int = weights["_attention_size"]
        mlp_size: int = weights["_mlp_size"]

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attention_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        attention_window = max_cache_length + 1

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        _disable_tf32(trt_config)

        # --- Inputs ---
        # VoiceChat uses frame-aligned additive fusion. Prompt steps select a
        # tokenizer embedding for the timeline channel; audio steps select the
        # FastConformer projection while retaining prior text/function tokens.
        text_token_id = network.add_input("text_token_id", trt.int32, (1,))
        timeline_token_id = network.add_input("timeline_token_id", trt.int32, (1,))
        function_token_id = network.add_input("function_token_id", trt.int32, (1,))
        audio_embed = network.add_input("audio_embed", trt.float32, (1, hidden))
        use_audio_embed = network.add_input("use_audio_embed", trt.float32, (1, 1))
        attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))

        conv_state_inputs = []
        ssm_state_inputs = []
        for mi in range(num_mamba):
            cs = network.add_input(
                graph_ops.layer_tensor_name("conv_state", mi), trt.float32, (conv_dim, d_conv)
            )
            ss = network.add_input(
                graph_ops.layer_tensor_name("ssm_state", mi),
                trt.float32,
                (mamba_num_heads, mamba_head_dim, d_state),
            )
            conv_state_inputs.append(cs)
            ssm_state_inputs.append(ss)

        cache_k_inputs = []
        cache_v_inputs = []
        for ai in range(num_attn):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", ai),
                trt.float32,
                (max_cache_length, kv_attention_size),
            )
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", ai),
                trt.float32,
                (max_cache_length, kv_attention_size),
            )
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        # --- Shared constants ---
        embedding_table = graph_ops.add_constant(network, (vocab, hidden), weights["embedding"])
        eps_tensor = graph_ops.add_constant(
            network,
            (1, 1),
            np.array([config.rms_norm_eps], dtype=np.float32),
        )

        # --- AddFusion(text, prompt-or-audio timeline, function) ---
        text_embed = network.add_gather(embedding_table, text_token_id, 0).get_output(0)
        timeline_embed = network.add_gather(embedding_table, timeline_token_id, 0).get_output(0)
        function_embed = network.add_gather(embedding_table, function_token_id, 0).get_output(0)
        one = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
        inverse_audio = network.add_elementwise(
            one, use_audio_embed, trt.ElementWiseOperation.SUB
        ).get_output(0)
        token_timeline = network.add_elementwise(
            timeline_embed, inverse_audio, trt.ElementWiseOperation.PROD
        ).get_output(0)
        acoustic_timeline = network.add_elementwise(
            audio_embed, use_audio_embed, trt.ElementWiseOperation.PROD
        ).get_output(0)
        selected_timeline = network.add_elementwise(
            token_timeline, acoustic_timeline, trt.ElementWiseOperation.SUM
        ).get_output(0)

        def scale_channel(tensor, scale_value: float):
            scale = graph_ops.add_constant(
                network,
                (1, 1),
                np.array([scale_value], dtype=np.float32),
            )
            return network.add_elementwise(tensor, scale, trt.ElementWiseOperation.PROD).get_output(
                0
            )

        text_channel = scale_channel(text_embed, float(weights["_duplex_text_weight"]))
        audio_channel = scale_channel(selected_timeline, float(weights["_duplex_audio_weight"]))
        function_channel = scale_channel(function_embed, float(weights["_duplex_function_weight"]))
        text_audio = network.add_elementwise(
            text_channel, audio_channel, trt.ElementWiseOperation.SUM
        ).get_output(0)
        hidden_state = network.add_elementwise(
            text_audio, function_channel, trt.ElementWiseOperation.SUM
        ).get_output(0)

        # --- Layer stack ---
        present_conv_outputs = []
        present_ssm_outputs = []
        present_k_outputs = []
        present_v_outputs = []
        mamba_counter = 0
        attn_counter = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            lt = layer_types[layer_idx]
            layer_hidden = hidden_state
            layer_eps = eps_tensor

            if lt == "mamba2":
                conv_state = conv_state_inputs[mamba_counter]
                ssm_state = ssm_state_inputs[mamba_counter]
                result = _add_mamba2_layer(
                    network=network,
                    hidden=layer_hidden,
                    conv_state_in=conv_state,
                    ssm_state_in=ssm_state,
                    eps_tensor=layer_eps,
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    d_inner=d_inner,
                    d_state=d_state,
                    d_conv=d_conv,
                    conv_dim=conv_dim,
                    mamba_num_heads=mamba_num_heads,
                    mamba_head_dim=mamba_head_dim,
                    n_groups=n_groups,
                )
                hidden_state = result["hidden"]
                present_conv_outputs.append(result["present_conv"])
                present_ssm_outputs.append(result["present_ssm"])
                mamba_counter += 1

            elif lt == "mlp":
                result = _add_mlp_layer(
                    network=network,
                    hidden=layer_hidden,
                    eps_tensor=layer_eps,
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    mlp_size=mlp_size,
                )
                hidden_state = result["hidden"]

            elif lt == "attention":
                cache_k = cache_k_inputs[attn_counter]
                cache_v = cache_v_inputs[attn_counter]
                result = graph_blocks.add_attention_block(
                    network,
                    layer_hidden,
                    cache_k,
                    cache_v,
                    attention_mask,
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    attention_size=attention_size,
                    kv_attention_size=kv_attention_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    max_cache_length=max_cache_length,
                    eps_tensor=layer_eps,
                )
                # add_attention_block does NOT apply residual
                residual = network.add_elementwise(
                    layer_hidden, result["attn_out"], trt.ElementWiseOperation.SUM
                )
                hidden_state = residual.get_output(0)
                present_k_outputs.append(result["present_k"])
                present_v_outputs.append(result["present_v"])
                attn_counter += 1

        # --- Final norm ---
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = graph_ops.add_rms_norm(
                network, hidden_state, hidden, final_norm, eps_tensor
            )

        # --- LM head ---
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_lm_head"]
        )
        logits = graph_ops.add_bias_sum(network, logits, vocab, np.zeros(vocab, dtype=np.float32))
        logits.name = "logits"
        network.mark_output(logits)

        function_logits = graph_ops.add_matmul_rhs_constant(
            network,
            hidden_state,
            hidden,
            vocab,
            weights["w_function_head"],
        )
        function_logits = graph_ops.add_bias_sum(
            network,
            function_logits,
            vocab,
            np.zeros(vocab, dtype=np.float32),
        )
        function_logits.name = "function_logits"
        network.mark_output(function_logits)

        # --- Present state outputs ---
        for mi in range(num_mamba):
            pc = present_conv_outputs[mi]
            ps = present_ssm_outputs[mi]
            pc.name = graph_ops.layer_tensor_name("present_conv", mi)
            ps.name = graph_ops.layer_tensor_name("present_ssm", mi)
            network.mark_output(pc)
            network.mark_output(ps)

        for ai in range(num_attn):
            pk = present_k_outputs[ai]
            pv = present_v_outputs[ai]
            pk.name = graph_ops.layer_tensor_name("present_k", ai)
            pv.name = graph_ops.layer_tensor_name("present_v", ai)
            network.mark_output(pk)
            network.mark_output(pv)

        # --- Build ---
        if verbose:
            print(
                f"[trtmc build] Building NemotronH hybrid TRT engine "
                f"({num_layers} layers: {num_mamba} mamba2 + "
                f"{sum(1 for t in layer_types if t == 'mlp')} mlp + "
                f"{num_attn} attention, "
                f"hidden={hidden}, d_inner={d_inner}, "
                f"d_state={d_state}, nheads={mamba_num_heads}, "
                f"cache={max_cache_length}) ...",
                file=sys.stderr,
            )

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)


def _add_mamba2_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    d_inner: int,
    d_state: int,
    d_conv: int,
    conv_dim: int,
    mamba_num_heads: int,
    mamba_head_dim: int,
    n_groups: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one Mamba-2 SSD layer (single-step decode).

    Mamba-2 in_proj splits: [gate(d_inner), hidden_B_C(conv_dim), dt(nheads)]
    Conv1d operates on hidden_B_C (d_inner + 2*n_groups*d_state channels).
    After conv+SiLU, split: hidden[d_inner], B[n_groups*d_state], C[n_groups*d_state].
    SSM state shape: [nheads, headdim, d_state] for full headdim-aware state.

    Returns: {hidden, present_conv, present_ssm}
    """
    groups_state_size = n_groups * d_state

    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )

    # ===== 2. Input projection =====
    proj_dim = d_inner + conv_dim + mamba_num_heads
    projected = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, proj_dim, weights[f"{prefix}.mamba_in_proj"], dtype=dtype
    )  # [1, proj_dim]

    # Split: gate [d_inner], hidden_B_C [conv_dim], dt [nheads]
    offset = 0
    gate_slice = network.add_slice(projected, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    gate = gate_slice.get_output(0)
    offset += d_inner

    hbc_slice = network.add_slice(projected, start=(0, offset), shape=(1, conv_dim), stride=(1, 1))
    hidden_B_C = hbc_slice.get_output(0)
    offset += conv_dim

    dt_slice = network.add_slice(
        projected, start=(0, offset), shape=(1, mamba_num_heads), stride=(1, 1)
    )
    dt_raw = dt_slice.get_output(0)

    # ===== 3. Conv1d step on hidden_B_C =====
    # conv_state_in: [conv_dim, d_conv]
    # hidden_B_C: [1, conv_dim] -> [conv_dim, 1]
    hbc_col = network.add_shuffle(hidden_B_C)
    hbc_col.reshape_dims = (conv_dim, 1)

    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in, start=(0, 1), shape=(conv_dim, d_conv - 1), stride=(1, 1)
        )
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), hbc_col.get_output(0)]
        )
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = hbc_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"], dtype=dtype
    )
    conv_prod = network.add_elementwise(present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    )
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim, weights[f"{prefix}.conv1d_bias"], dtype=dtype
    )
    hbc_activated = graph_ops.add_activation(network, conv_out, "silu")

    # ===== 4. Split hidden, B, C from activated output =====
    hidden_x_slice = network.add_slice(
        hbc_activated, start=(0, 0), shape=(1, d_inner), stride=(1, 1)
    )
    hidden_x = hidden_x_slice.get_output(0)

    B_raw_slice = network.add_slice(
        hbc_activated, start=(0, d_inner), shape=(1, groups_state_size), stride=(1, 1)
    )
    B_raw = B_raw_slice.get_output(0)

    C_raw_slice = network.add_slice(
        hbc_activated,
        start=(0, d_inner + groups_state_size),
        shape=(1, groups_state_size),
        stride=(1, 1),
    )
    C_raw = C_raw_slice.get_output(0)

    # ===== 5. dt: add bias + softplus =====
    dt_bias_const = graph_ops.add_constant(
        network, (1, mamba_num_heads), weights[f"{prefix}.dt_bias"], dtype=dtype
    )
    dt_biased = network.add_elementwise(dt_raw, dt_bias_const, trt.ElementWiseOperation.SUM)
    # The checkpoint contains dt_bias values as large as 33.5.
    dt_for_state = dt_biased.get_output(0)
    dt_exp = network.add_unary(dt_for_state, trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32), dtype=np.float32
    )
    dt_exp_p1 = network.add_elementwise(dt_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    dt_softplus = network.add_unary(dt_exp_p1.get_output(0), trt.UnaryOperation.LOG)
    dt = dt_softplus.get_output(0)  # [1, mamba_num_heads]

    # ===== 6. Multi-head SSM step =====
    # A: [nheads] -> [nheads, 1, 1] for broadcast
    A_const = graph_ops.add_constant(
        network,
        (mamba_num_heads, 1, 1),
        weights[f"{prefix}.A"].reshape(mamba_num_heads, 1, 1),
        dtype=np.float32,
    )

    # dt: [1, nheads] -> [nheads, 1, 1]
    dt_col = network.add_shuffle(dt)
    dt_col.reshape_dims = (mamba_num_heads, 1, 1)

    # dA = exp(dt * A): broadcast to [nheads, headdim, d_state]
    dtA = network.add_elementwise(dt_col.get_output(0), A_const, trt.ElementWiseOperation.PROD)
    dA = network.add_unary(dtA.get_output(0), trt.UnaryOperation.EXP)

    # B: [1, n_groups*d_state] -> [n_groups, d_state] -> expand to [nheads, d_state]
    B_grouped = network.add_shuffle(B_raw)
    B_grouped.reshape_dims = (n_groups, d_state)
    heads_per_group = mamba_num_heads // n_groups

    if heads_per_group > 1:
        B_3d = network.add_shuffle(B_grouped.get_output(0))
        B_3d.reshape_dims = (n_groups, 1, d_state)
        tile_ones = graph_ops.add_constant(
            network,
            (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=dtype),
            dtype=dtype,
        )
        B_tiled = network.add_elementwise(
            B_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        B_heads_s = network.add_shuffle(B_tiled.get_output(0))
        B_heads_s.reshape_dims = (mamba_num_heads, d_state)
        B_heads = B_heads_s.get_output(0)
    else:
        B_heads = B_grouped.get_output(0)

    # C: same group expansion
    C_grouped = network.add_shuffle(C_raw)
    C_grouped.reshape_dims = (n_groups, d_state)

    if heads_per_group > 1:
        C_3d = network.add_shuffle(C_grouped.get_output(0))
        C_3d.reshape_dims = (n_groups, 1, d_state)
        C_tiled = network.add_elementwise(
            C_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        C_heads_s = network.add_shuffle(C_tiled.get_output(0))
        C_heads_s.reshape_dims = (mamba_num_heads, d_state)
        C_heads = C_heads_s.get_output(0)
    else:
        C_heads = C_grouped.get_output(0)

    # x: [1, d_inner] -> [nheads, headdim]
    x_heads = network.add_shuffle(hidden_x)
    x_heads.reshape_dims = (mamba_num_heads, mamba_head_dim)

    # dBx[h,d,s] = dt[h] * B[h,s] * x[h,d]
    # dt_B: [nheads, 1, 1] * [nheads, 1, d_state] -> [nheads, 1, d_state]
    B_3d_expand = network.add_shuffle(B_heads)
    B_3d_expand.reshape_dims = (mamba_num_heads, 1, d_state)
    B_for_state = B_3d_expand.get_output(0)
    dt_B = network.add_elementwise(dt_col.get_output(0), B_for_state, trt.ElementWiseOperation.PROD)

    # x: [nheads, headdim] -> [nheads, headdim, 1]
    x_3d = network.add_shuffle(x_heads.get_output(0))
    x_3d.reshape_dims = (mamba_num_heads, mamba_head_dim, 1)
    x_for_state = x_3d.get_output(0)

    # dBx: [nheads, headdim, 1] * [nheads, 1, d_state] -> [nheads, headdim, d_state]
    dBx = network.add_elementwise(x_for_state, dt_B.get_output(0), trt.ElementWiseOperation.PROD)

    # SSM update: new_ssm = dA * ssm_state + dBx
    # ssm_state_in: [nheads, headdim, d_state]
    decay = network.add_elementwise(dA.get_output(0), ssm_state_in, trt.ElementWiseOperation.PROD)
    new_ssm = network.add_elementwise(
        decay.get_output(0), dBx.get_output(0), trt.ElementWiseOperation.SUM
    )
    present_ssm = new_ssm.get_output(0)  # [nheads, headdim, d_state]

    # y[h,d] = sum_s(ssm_state[h,d,s] * C[h,s])
    # C: [nheads, d_state] -> [nheads, d_state, 1]
    C_col = network.add_shuffle(C_heads)
    C_col.reshape_dims = (mamba_num_heads, d_state, 1)
    C_for_state = C_col.get_output(0)
    # batch matmul: [nheads, headdim, d_state] @ [nheads, d_state, 1] -> [nheads, headdim, 1]
    y_matmul = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.NONE, C_for_state, trt.MatrixOperation.NONE
    )
    y_squeeze = network.add_shuffle(y_matmul.get_output(0))
    y_squeeze.reshape_dims = (mamba_num_heads, mamba_head_dim)

    # D skip: D[h] * x[h,d]
    D_const = graph_ops.add_constant(
        network,
        (mamba_num_heads, 1),
        weights[f"{prefix}.D"].reshape(mamba_num_heads, 1),
        dtype=np.float32,
    )
    x_for_skip = x_heads.get_output(0)
    Dx = network.add_elementwise(D_const, x_for_skip, trt.ElementWiseOperation.PROD)

    y_plus_D = network.add_elementwise(
        y_squeeze.get_output(0), Dx.get_output(0), trt.ElementWiseOperation.SUM
    )
    # [nheads, headdim] -> [1, d_inner]
    y_flat = network.add_shuffle(y_plus_D.get_output(0))
    y_flat.reshape_dims = (1, d_inner)
    y_for_gate = y_flat.get_output(0)

    # ===== 7. Gated Group RMSNorm (norm_before_gate=False) =====
    # HF: output = weight * group_rms_norm(y * silu(gate))
    # Gate is applied BEFORE normalization. RMSNorm is per-group,
    # with group_size = d_inner // n_groups.
    mamba_norm_w = weights[f"{prefix}.mamba_norm"]
    eps_small = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=np.float32), dtype=np.float32
    )

    # Step 1: Apply silu(gate) to y BEFORE norm
    gate_activated = graph_ops.add_activation(network, gate, "silu")
    y_gated = network.add_elementwise(y_for_gate, gate_activated, trt.ElementWiseOperation.PROD)

    # Step 2: Group RMSNorm — reshape to [n_groups, group_size], norm per group
    group_size = d_inner // n_groups
    y_grouped = network.add_shuffle(y_gated.get_output(0))
    y_grouped.reshape_dims = (n_groups, group_size)
    norm_input = y_grouped.get_output(0)

    sq = network.add_elementwise(norm_input, norm_input, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        norm_input, recip.get_output(0), trt.ElementWiseOperation.PROD
    )

    # Reshape back to [1, d_inner] and apply weight
    y_flat_normed = network.add_shuffle(normalized.get_output(0))
    y_flat_normed.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(network, (1, d_inner), mamba_norm_w, dtype=np.float32)
    gated = network.add_elementwise(
        y_flat_normed.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    gated_tensor = gated.get_output(0)

    # ===== 8. Output projection + residual =====
    out = graph_ops.add_matmul_rhs_constant(
        network,
        gated_tensor,
        d_inner,
        hidden_size,
        weights[f"{prefix}.mamba_out_proj"],
        dtype=dtype,
    )

    residual = network.add_elementwise(hidden, out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def _add_mlp_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add MLP layer: RMSNorm -> up -> relu2 -> down -> residual."""
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )

    up = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, mlp_size, weights[f"{prefix}.w_up"], dtype=dtype
    )
    activated = graph_ops.add_activation(network, up, "relu2")
    down = graph_ops.add_matmul_rhs_constant(
        network, activated, mlp_size, hidden_size, weights[f"{prefix}.w_down"], dtype=dtype
    )

    residual = network.add_elementwise(hidden, down, trt.ElementWiseOperation.SUM)

    return {"hidden": residual.get_output(0)}


def build_thinker_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    verbose: bool = False,
) -> bytes:
    """Build the strongly typed VoiceChat AddFusion + Nemotron-H engine."""
    return VoiceChatThinkerBuilder().build_engine(
        config,
        weights,
        max_cache_length,
        verbose=verbose,
    )
