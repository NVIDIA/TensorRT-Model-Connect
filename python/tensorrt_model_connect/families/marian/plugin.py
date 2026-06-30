"""Marian family plugin -- encoder-decoder machine translation model.

Marian is an encoder-decoder transformer for machine translation:
  - Encoder: token embeddings + sinusoidal positional embeddings (static)
             -> N self-attention layers -> encoder output
  - Decoder: autoregressive text generation with causal self-attention (KV cache)
             + cross-attention to encoder output + Swish (SiLU) MLP
  - Uses LayerNorm (not RMSNorm), Swish activation
  - scale_embedding: True (multiply embeddings by sqrt(d_model))
  - normalize_before: False (post-LayerNorm, like BART)
  - No final layer norm (add_final_layer_norm: False)
  - Static positional embeddings (sinusoidal, stored as embed_positions.weight)
  - Shared encoder/decoder vocabulary; tied embeddings for lm_head
  - model_type: "marian", architectures: ["MarianMTModel"]
  - decoder_start_token_id = pad_token_id (62517 for opus-mt-en-ru)
  - eos_token_id = 0

Cross-attention design:
  The cross_k/cross_v inputs to the decoder engine are the RAW encoder output
  (same tensor copied to all layers). The per-layer K/V projections are baked
  into the decoder TRT graph. The C++ runtime copies the raw encoder output
  to all cross_k/cross_v slots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .weights import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from .model import model as graph_ops
from ...parallel_config import normalize_parallel_config, require_tensorrt_11_for_tensor_parallel


trt = trt_compat.get_trt()


class MarianPlugin:
    name = "marian"
    runtime_strategy = "marian_translation"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "marian"

    def ensure_tokenizer_json(
        self,
        model_dir: str | Path,
        *,
        previous_error: str | None = None,
    ) -> bool:
        from .weights.tokenizer_json import ensure_tokenizer_json

        return ensure_tokenizer_json(model_dir, previous_error=previous_error)

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)
        raw = config.raw
        enc_layers = raw.get("encoder_layers", config.num_hidden_layers)
        dec_layers = raw.get("decoder_layers", config.num_hidden_layers)
        enc_heads = raw.get("encoder_attention_heads", config.num_attention_heads)
        dec_heads = raw.get("decoder_attention_heads", config.num_attention_heads)
        enc_ffn = raw.get("encoder_ffn_dim", config.intermediate_size)
        dec_ffn = raw.get("decoder_ffn_dim", config.intermediate_size)
        max_position_embeddings = raw.get("max_position_embeddings", 512)

        weights = WeightDict()
        weights["_enc_layers"] = enc_layers
        weights["_dec_layers"] = dec_layers
        weights["_enc_heads"] = enc_heads
        weights["_dec_heads"] = dec_heads
        weights["_enc_ffn"] = enc_ffn
        weights["_dec_ffn"] = dec_ffn
        weights["_max_position_embeddings"] = max_position_embeddings

        enc_embed = _load_tensor(readers, "model.encoder.embed_tokens.weight").astype(np.float32)
        weights["enc_embedding"] = enc_embed
        weights["enc_pos_embedding"] = _load_tensor(
            readers, "model.encoder.embed_positions.weight"
        ).astype(np.float32)

        for i in range(enc_layers):
            hf = f"model.encoder.layers.{i}"
            pfx = f"enc_layer.{i}"
            for proj in ("q", "k", "v"):
                weights[f"{pfx}.w_{proj}"] = _transpose_2d(
                    _load_tensor(readers, f"{hf}.self_attn.{proj}_proj.weight"), f"enc_{proj}"
                )
                weights[f"{pfx}.b_{proj}"] = _load_tensor(
                    readers, f"{hf}.self_attn.{proj}_proj.bias"
                ).astype(np.float32)
            weights[f"{pfx}.w_o"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.self_attn.out_proj.weight"), "enc_o"
            )
            weights[f"{pfx}.b_o"] = _load_tensor(readers, f"{hf}.self_attn.out_proj.bias").astype(
                np.float32
            )
            weights[f"{pfx}.attn_norm"] = _load_tensor(
                readers, f"{hf}.self_attn_layer_norm.weight"
            ).astype(np.float32)
            weights[f"{pfx}.attn_norm_beta"] = _load_tensor(
                readers, f"{hf}.self_attn_layer_norm.bias"
            ).astype(np.float32)
            weights[f"{pfx}.w_fc1"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.fc1.weight"), "enc_fc1"
            )
            weights[f"{pfx}.b_fc1"] = _load_tensor(readers, f"{hf}.fc1.bias").astype(np.float32)
            weights[f"{pfx}.w_fc2"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.fc2.weight"), "enc_fc2"
            )
            weights[f"{pfx}.b_fc2"] = _load_tensor(readers, f"{hf}.fc2.bias").astype(np.float32)
            weights[f"{pfx}.ffn_norm"] = _load_tensor(
                readers, f"{hf}.final_layer_norm.weight"
            ).astype(np.float32)
            weights[f"{pfx}.ffn_norm_beta"] = _load_tensor(
                readers, f"{hf}.final_layer_norm.bias"
            ).astype(np.float32)

        dec_embed = _load_tensor(readers, "model.decoder.embed_tokens.weight").astype(np.float32)
        weights["dec_embedding"] = dec_embed
        weights["dec_pos_embedding"] = _load_tensor(
            readers, "model.decoder.embed_positions.weight"
        ).astype(np.float32)

        for i in range(dec_layers):
            hf = f"model.decoder.layers.{i}"
            pfx = f"layer.{i}"
            for proj in ("q", "k", "v"):
                weights[f"{pfx}.w_{proj}"] = _transpose_2d(
                    _load_tensor(readers, f"{hf}.self_attn.{proj}_proj.weight"), f"dec_{proj}"
                )
                weights[f"{pfx}.{proj}_bias"] = _load_tensor(
                    readers, f"{hf}.self_attn.{proj}_proj.bias"
                ).astype(np.float32)
            weights[f"{pfx}.w_o"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.self_attn.out_proj.weight"), "dec_o"
            )
            weights[f"{pfx}.o_bias"] = _load_tensor(
                readers, f"{hf}.self_attn.out_proj.bias"
            ).astype(np.float32)
            weights[f"{pfx}.input_norm"] = _load_tensor(
                readers, f"{hf}.self_attn_layer_norm.weight"
            ).astype(np.float32)
            weights[f"{pfx}.input_norm_beta"] = _load_tensor(
                readers, f"{hf}.self_attn_layer_norm.bias"
            ).astype(np.float32)
            for proj in ("q", "k", "v"):
                weights[f"{pfx}.cross_w_{proj}"] = _transpose_2d(
                    _load_tensor(readers, f"{hf}.encoder_attn.{proj}_proj.weight"), f"xattn_{proj}"
                )
                weights[f"{pfx}.cross_b_{proj}"] = _load_tensor(
                    readers, f"{hf}.encoder_attn.{proj}_proj.bias"
                ).astype(np.float32)
            weights[f"{pfx}.cross_w_o"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.encoder_attn.out_proj.weight"), "xattn_o"
            )
            weights[f"{pfx}.cross_b_o"] = _load_tensor(
                readers, f"{hf}.encoder_attn.out_proj.bias"
            ).astype(np.float32)
            weights[f"{pfx}.cross_attn_norm"] = _load_tensor(
                readers, f"{hf}.encoder_attn_layer_norm.weight"
            ).astype(np.float32)
            weights[f"{pfx}.cross_attn_norm_beta"] = _load_tensor(
                readers, f"{hf}.encoder_attn_layer_norm.bias"
            ).astype(np.float32)
            weights[f"{pfx}.w_fc1"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.fc1.weight"), "dec_fc1"
            )
            weights[f"{pfx}.fc1_bias"] = _load_tensor(readers, f"{hf}.fc1.bias").astype(np.float32)
            weights[f"{pfx}.w_fc2"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.fc2.weight"), "dec_fc2"
            )
            weights[f"{pfx}.fc2_bias"] = _load_tensor(readers, f"{hf}.fc2.bias").astype(np.float32)
            weights[f"{pfx}.post_attn_norm"] = _load_tensor(
                readers, f"{hf}.final_layer_norm.weight"
            ).astype(np.float32)
            weights[f"{pfx}.post_attn_norm_beta"] = _load_tensor(
                readers, f"{hf}.final_layer_norm.bias"
            ).astype(np.float32)

        weights["w_out"] = _transpose_2d(dec_embed.copy(), "embedding_tied")

        if _has_tensor(readers, "final_logits_bias"):
            weights["final_logits_bias"] = (
                _load_tensor(readers, "final_logits_bias").astype(np.float32).flatten()
            )
        else:
            weights["final_logits_bias"] = np.zeros(config.vocab_size, dtype=np.float32)

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Marian tensor-parallel decoder builds"
            )
            from .model.parallel import build_marian_tp_decoder_engine

            return build_marian_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                parallel_config=parallel,
            )

        dec_layers = weights["_dec_layers"]
        dec_heads = weights["_dec_heads"]
        dec_ffn = weights["_dec_ffn"]
        max_pos = weights["_max_position_embeddings"]
        hidden = config.hidden_size
        vocab = config.vocab_size
        head_dim = hidden // dec_heads
        attention_window = max_cache_length + 1
        max_enc_seq_len = max_pos

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        trt_config.clear_flag(trt.BuilderFlag.TF32)

        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input("attention_mask", trt.float32, (attention_window,))

        cache_k_inputs, cache_v_inputs = [], []
        for i in range(dec_layers):
            cache_k_inputs.append(
                network.add_input(
                    graph_ops.layer_tensor_name("cache_k", i),
                    trt.float32,
                    (max_cache_length, hidden),
                )
            )
            cache_v_inputs.append(
                network.add_input(
                    graph_ops.layer_tensor_name("cache_v", i),
                    trt.float32,
                    (max_cache_length, hidden),
                )
            )

        cross_k_inputs, cross_v_inputs = [], []
        for i in range(dec_layers):
            cross_k_inputs.append(
                network.add_input(
                    graph_ops.layer_tensor_name("cross_k", i),
                    trt.float32,
                    (max_enc_seq_len, hidden),
                )
            )
            cross_v_inputs.append(
                network.add_input(
                    graph_ops.layer_tensor_name("cross_v", i),
                    trt.float32,
                    (max_enc_seq_len, hidden),
                )
            )

        encoder_mask = network.add_input("encoder_mask", trt.float32, (max_enc_seq_len,))

        embedding_table = graph_ops.add_constant(network, (vocab, hidden), weights["dec_embedding"])
        pos_embed_np = weights["dec_pos_embedding"]
        pos_embedding_table = graph_ops.add_constant(network, pos_embed_np.shape, pos_embed_np)

        token_embed = network.add_gather(embedding_table, token_id, 0).get_output(0)
        scale_val = np.sqrt(float(hidden))
        scale_const = graph_ops.add_constant(
            network, (1, 1), np.array([scale_val], dtype=np.float32)
        )
        token_embed = network.add_elementwise(
            token_embed, scale_const, trt.ElementWiseOperation.PROD
        ).get_output(0)
        pos_embed = network.add_gather(pos_embedding_table, position_id, 0).get_output(0)
        hidden_state = network.add_elementwise(
            token_embed, pos_embed, trt.ElementWiseOperation.SUM
        ).get_output(0)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        present_k_outputs, present_v_outputs = [], []
        for layer_idx in range(dec_layers):
            prefix = f"layer.{layer_idx}"
            result = _add_marian_decoder_layer(
                network=network,
                hidden=hidden_state,
                cache_k=cache_k_inputs[layer_idx],
                cache_v=cache_v_inputs[layer_idx],
                cross_k=cross_k_inputs[layer_idx],
                cross_v=cross_v_inputs[layer_idx],
                attention_mask=attention_mask,
                encoder_mask=encoder_mask,
                eps=config.rms_norm_eps,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                num_heads=dec_heads,
                head_dim=head_dim,
                ffn_dim=dec_ffn,
                max_cache_length=max_cache_length,
                max_enc_seq_len=max_enc_seq_len,
            )
            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            if debug_layer_outputs:
                _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_out"]
        )
        logits = graph_ops.add_bias_sum(network, logits, vocab, weights["final_logits_bias"])
        logits.name = "logits"
        network.mark_output(logits)

        for i in range(dec_layers):
            present_k_outputs[i].name = graph_ops.layer_tensor_name("present_k", i)
            present_v_outputs[i].name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(present_k_outputs[i])
            network.mark_output(present_v_outputs[i])

        if verbose:
            print(
                f"[trtmc build] Building Marian decoder ({dec_layers}L, h={hidden}, heads={dec_heads}, ffn={dec_ffn}, cache={max_cache_length})",
                file=sys.stderr,
            )
        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT decoder engine build failed")
        return bytes(plan)

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict, *, verbose: bool = False
    ) -> bytes | None:
        return _build_marian_encoder(config, weights, verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        raw = config.raw
        eos_id = raw.get("eos_token_id", 0)
        pad_id = raw.get("pad_token_id", 62517)
        decoder_start_token_id = raw.get("decoder_start_token_id", pad_id)
        return {
            "max_source_positions": raw.get("max_position_embeddings", 512),
            "encoder_layers": raw.get("encoder_layers", config.num_hidden_layers),
            "decoder_layers": raw.get("decoder_layers", config.num_hidden_layers),
            "encoder_ffn_dim": raw.get("encoder_ffn_dim", config.intermediate_size),
            "decoder_ffn_dim": raw.get("decoder_ffn_dim", config.intermediate_size),
            "encoder_attention_heads": raw.get(
                "encoder_attention_heads", config.num_attention_heads
            ),
            "decoder_attention_heads": raw.get(
                "decoder_attention_heads", config.num_attention_heads
            ),
            "has_vision_engine": True,
            "decoder_start_token_id": decoder_start_token_id,
            "eos_token_id": eos_id,
            "scale_embedding": raw.get("scale_embedding", True),
            "activation_function": raw.get("activation_function", "swish"),
        }


def _build_marian_encoder(config, weights, *, verbose=False):
    enc_layers = weights["_enc_layers"]
    enc_heads = weights["_enc_heads"]
    enc_ffn = weights["_enc_ffn"]
    max_pos = weights["_max_position_embeddings"]
    hidden = config.hidden_size
    vocab = config.vocab_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = builder.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    tc.clear_flag(trt.BuilderFlag.TF32)

    input_ids = network.add_input("input_ids", trt.int32, (max_pos,))
    attention_mask = network.add_input("attention_mask", trt.float32, (max_pos,))

    embed_table = graph_ops.add_constant(network, (vocab, hidden), weights["enc_embedding"])
    pos_embed_np = weights["enc_pos_embedding"]
    graph_ops.add_constant(network, pos_embed_np.shape, pos_embed_np)

    token_embed = network.add_gather(embed_table, input_ids, 0).get_output(0)
    scale_val = np.sqrt(float(hidden))
    scale_const = graph_ops.add_constant(network, (1, 1), np.array([scale_val], dtype=np.float32))
    token_embed = network.add_elementwise(
        token_embed, scale_const, trt.ElementWiseOperation.PROD
    ).get_output(0)

    # Position embeddings cover full sequence; add as constant (no Gather needed)
    pos_embed = graph_ops.add_constant(network, (max_pos, hidden), pos_embed_np)

    hs = network.add_elementwise(token_embed, pos_embed, trt.ElementWiseOperation.SUM).get_output(0)

    # Reshape attention mask [max_pos] -> [1, 1, 1, max_pos]
    # for native IAttention broadcast across heads and query positions.
    enc_mask_4d = network.add_shuffle(attention_mask)
    enc_mask_4d.reshape_dims = (1, 1, 1, max_pos)

    for li in range(enc_layers):
        pfx = f"enc_layer.{li}"
        head_dim = hidden // enc_heads

        q = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(network, hs, hidden, hidden, weights[f"{pfx}.w_q"]),
            hidden,
            weights[f"{pfx}.b_q"],
        )
        k = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(network, hs, hidden, hidden, weights[f"{pfx}.w_k"]),
            hidden,
            weights[f"{pfx}.b_k"],
        )
        v = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(network, hs, hidden, hidden, weights[f"{pfx}.w_v"]),
            hidden,
            weights[f"{pfx}.b_v"],
        )

        ctx_flat = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=enc_heads,
            head_dim=head_dim,
            q_seq=max_pos,
            kv_seq=max_pos,
            mask=enc_mask_4d.get_output(0),
        )

        out_proj = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, ctx_flat, hidden, hidden, weights[f"{pfx}.w_o"]
            ),
            hidden,
            weights[f"{pfx}.b_o"],
        )

        hs = network.add_elementwise(hs, out_proj, trt.ElementWiseOperation.SUM).get_output(0)
        hs = graph_ops.add_layer_norm_native(
            network,
            hs,
            hidden,
            weights[f"{pfx}.attn_norm"],
            weights[f"{pfx}.attn_norm_beta"],
            config.rms_norm_eps,
        )

        fc1 = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, hs, hidden, enc_ffn, weights[f"{pfx}.w_fc1"]
            ),
            enc_ffn,
            weights[f"{pfx}.b_fc1"],
        )
        act = graph_ops.add_activation(network, fc1, "silu")
        fc2 = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, act, enc_ffn, hidden, weights[f"{pfx}.w_fc2"]
            ),
            hidden,
            weights[f"{pfx}.b_fc2"],
        )

        hs = network.add_elementwise(hs, fc2, trt.ElementWiseOperation.SUM).get_output(0)
        hs = graph_ops.add_layer_norm_native(
            network,
            hs,
            hidden,
            weights[f"{pfx}.ffn_norm"],
            weights[f"{pfx}.ffn_norm_beta"],
            config.rms_norm_eps,
        )

    hs.name = "encoder_output"
    network.mark_output(hs)

    if verbose:
        print(
            f"[trtmc build] Building Marian encoder ({enc_layers}L, h={hidden}, heads={enc_heads})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, tc)
    if plan is None:
        raise RuntimeError("TensorRT encoder engine build failed")
    return bytes(plan)


def _add_marian_decoder_layer(
    *,
    network,
    hidden,
    cache_k,
    cache_v,
    cross_k,
    cross_v,
    attention_mask,
    encoder_mask,
    eps,
    weights,
    prefix,
    hidden_size,
    num_heads,
    head_dim,
    ffn_dim,
    max_cache_length,
    max_enc_seq_len,
):
    attention_size = hidden_size
    attention_window = max_cache_length + 1

    q = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_q"]
        ),
        attention_size,
        weights[f"{prefix}.q_bias"],
    )
    k = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_k"]
        ),
        attention_size,
        weights[f"{prefix}.k_bias"],
    )
    v = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_v"]
        ),
        attention_size,
        weights[f"{prefix}.v_bias"],
    )
    present_k, present_v = k, v

    kr = network.add_shuffle(k)
    kr.reshape_dims = (1, attention_size)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (1, attention_size)
    ak = network.add_concatenation([cache_k, kr.get_output(0)])
    ak.axis = 0
    av = network.add_concatenation([cache_v, vr.get_output(0)])
    av.axis = 0

    m4 = network.add_shuffle(attention_mask)
    m4.reshape_dims = (1, 1, 1, attention_window)
    cf = graph_ops.add_attention_from_rows(
        network,
        q,
        ak.get_output(0),
        av.get_output(0),
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=attention_window,
        mask=m4.get_output(0),
    )
    sa = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, cf, attention_size, hidden_size, weights[f"{prefix}.w_o"]
        ),
        hidden_size,
        weights[f"{prefix}.o_bias"],
    )

    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)
    psa = graph_ops.add_layer_norm_native(
        network,
        psa,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights[f"{prefix}.input_norm_beta"],
        eps,
    )

    cq = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, psa, hidden_size, attention_size, weights[f"{prefix}.cross_w_q"]
        ),
        attention_size,
        weights[f"{prefix}.cross_b_q"],
    )
    ck_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, cross_k, hidden_size, attention_size, weights[f"{prefix}.cross_w_k"]
        ),
        attention_size,
        weights[f"{prefix}.cross_b_k"],
    )
    cv_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, cross_v, hidden_size, attention_size, weights[f"{prefix}.cross_w_v"]
        ),
        attention_size,
        weights[f"{prefix}.cross_b_v"],
    )

    enc_mask_4d = network.add_shuffle(encoder_mask)
    enc_mask_4d.reshape_dims = (1, 1, 1, max_enc_seq_len)
    ccf = graph_ops.add_attention_from_rows(
        network,
        cq,
        ck_proj,
        cv_proj,
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=max_enc_seq_len,
        mask=enc_mask_4d.get_output(0),
    )
    ca = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, ccf, attention_size, hidden_size, weights[f"{prefix}.cross_w_o"]
        ),
        hidden_size,
        weights[f"{prefix}.cross_b_o"],
    )

    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)
    pca = graph_ops.add_layer_norm_native(
        network,
        pca,
        hidden_size,
        weights[f"{prefix}.cross_attn_norm"],
        weights[f"{prefix}.cross_attn_norm_beta"],
        eps,
    )

    fc1 = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, pca, hidden_size, ffn_dim, weights[f"{prefix}.w_fc1"]
        ),
        ffn_dim,
        weights[f"{prefix}.fc1_bias"],
    )
    act = graph_ops.add_activation(network, fc1, "silu")
    fc2 = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, act, ffn_dim, hidden_size, weights[f"{prefix}.w_fc2"]
        ),
        hidden_size,
        weights[f"{prefix}.fc2_bias"],
    )

    out = network.add_elementwise(pca, fc2, trt.ElementWiseOperation.SUM).get_output(0)
    out = graph_ops.add_layer_norm_native(
        network,
        out,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps,
    )

    return {"hidden": out, "present_k": present_k, "present_v": present_v}


def _mark_debug_output(network, tensor, name):
    out = tensor
    if out.dtype != trt.float32:
        out = network.add_cast(out, trt.float32).get_output(0)
    out.name = name
    network.mark_output(out)


plugin = MarianPlugin()
