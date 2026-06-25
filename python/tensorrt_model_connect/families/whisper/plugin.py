"""Whisper family plugin -- encoder-decoder ASR model (speech-to-text).

Whisper is an encoder-decoder transformer for automatic speech recognition:
  - Encoder: mel spectrogram -> Conv1d stem -> learned positional encoding
             -> N self-attention layers -> encoder output [1500, d_model]
  - Decoder: autoregressive text generation with causal self-attention (KV cache)
             + cross-attention to encoder output + GELU MLP
  - Uses LayerNorm (not RMSNorm), GELU activation, learned positional embeddings
  - model_type: "whisper", architectures: ["WhisperForConditionalGeneration"]

Cross-attention design:
  The cross_k/cross_v inputs to the decoder engine are the RAW encoder output
  (same tensor copied to all layers). The per-layer K/V projections are baked
  into the decoder TRT graph. The C++ runtime's compute_cross_kv() correctly
  copies the raw encoder output to all cross_k/cross_v slots.
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
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .decoder_tp_builder import build_whisper_tp_decoder_engine


trt = trt_compat.get_trt()

def _load_bias_or_zeros(readers, hf_key: str, size: int, dtype=np.float32) -> np.ndarray:
    """Load bias if it exists, otherwise return zeros."""
    if _has_tensor(readers, hf_key):
        return _load_tensor(readers, hf_key).astype(dtype)
    return np.zeros(size, dtype=dtype)


class WhisperPlugin:
    name = "whisper"
    runtime_strategy = "whisper_speech_to_text"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "whisper"

    def load_weights(self, model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)
        raw = config.raw
        hidden = config.hidden_size
        enc_layers = raw.get("encoder_layers", config.num_hidden_layers)
        dec_layers = raw.get("decoder_layers", config.num_hidden_layers)
        enc_heads = raw.get("encoder_attention_heads", config.num_attention_heads)
        dec_heads = raw.get("decoder_attention_heads", config.num_attention_heads)
        enc_ffn = raw.get("encoder_ffn_dim", config.intermediate_size)
        dec_ffn = raw.get("decoder_ffn_dim", config.intermediate_size)
        num_mel_bins = raw.get("num_mel_bins", 80)
        max_source_positions = raw.get("max_source_positions", 1500)
        max_target_positions = raw.get("max_target_positions", 448)

        # Projection weights use work dtype; norm weights stay FP32
        # (add_layer_norm handles FP32 precision boundaries internally).
        w_dtype = np.float16 if precision == "fp16" else np.float32

        weights = WeightDict()
        weights["_enc_layers"] = enc_layers
        weights["_dec_layers"] = dec_layers
        weights["_enc_heads"] = enc_heads
        weights["_dec_heads"] = dec_heads
        weights["_enc_ffn"] = enc_ffn
        weights["_dec_ffn"] = dec_ffn
        weights["_num_mel_bins"] = num_mel_bins
        weights["_max_source_positions"] = max_source_positions
        weights["_max_target_positions"] = max_target_positions

        # Encoder conv stem
        weights["enc_conv1_weight"] = _load_tensor(readers, "model.encoder.conv1.weight").astype(w_dtype)
        weights["enc_conv1_bias"] = _load_tensor(readers, "model.encoder.conv1.bias").astype(w_dtype)
        weights["enc_conv2_weight"] = _load_tensor(readers, "model.encoder.conv2.weight").astype(w_dtype)
        weights["enc_conv2_bias"] = _load_tensor(readers, "model.encoder.conv2.bias").astype(w_dtype)

        # [C2] Encoder learned positional embeddings
        weights["enc_pos_embedding"] = _load_tensor(
            readers, "model.encoder.embed_positions.weight").astype(w_dtype)

        # Encoder layers
        for i in range(enc_layers):
            hf = f"model.encoder.layers.{i}"
            pfx = f"enc_layer.{i}"
            # [C1] Whisper k_proj has no bias -- load conditionally
            for proj in ("q", "k", "v"):
                weights[f"{pfx}.w_{proj}"] = _transpose_2d(
                    _load_tensor(readers, f"{hf}.self_attn.{proj}_proj.weight"), f"enc_{proj}")
                weights[f"{pfx}.b_{proj}"] = _load_bias_or_zeros(
                    readers, f"{hf}.self_attn.{proj}_proj.bias", hidden, dtype=w_dtype)
            weights[f"{pfx}.w_o"] = _transpose_2d(_load_tensor(readers, f"{hf}.self_attn.out_proj.weight"), "enc_o")
            weights[f"{pfx}.b_o"] = _load_tensor(readers, f"{hf}.self_attn.out_proj.bias").astype(w_dtype)
            # Norm weights stay FP32 (add_layer_norm casts internally)
            weights[f"{pfx}.attn_norm"] = _load_tensor(readers, f"{hf}.self_attn_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.attn_norm_beta"] = _load_tensor(readers, f"{hf}.self_attn_layer_norm.bias").astype(np.float32)
            weights[f"{pfx}.w_fc1"] = _transpose_2d(_load_tensor(readers, f"{hf}.fc1.weight"), "enc_fc1")
            weights[f"{pfx}.b_fc1"] = _load_tensor(readers, f"{hf}.fc1.bias").astype(w_dtype)
            weights[f"{pfx}.w_fc2"] = _transpose_2d(_load_tensor(readers, f"{hf}.fc2.weight"), "enc_fc2")
            weights[f"{pfx}.b_fc2"] = _load_tensor(readers, f"{hf}.fc2.bias").astype(w_dtype)
            # Norm weights stay FP32
            weights[f"{pfx}.ffn_norm"] = _load_tensor(readers, f"{hf}.final_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.ffn_norm_beta"] = _load_tensor(readers, f"{hf}.final_layer_norm.bias").astype(np.float32)

        # Norm weights stay FP32
        weights["enc_final_norm"] = _load_tensor(readers, "model.encoder.layer_norm.weight").astype(np.float32)
        weights["enc_final_norm_beta"] = _load_tensor(readers, "model.encoder.layer_norm.bias").astype(np.float32)

        # Decoder embeddings
        dec_embed = _load_tensor(readers, "model.decoder.embed_tokens.weight")
        weights["dec_embedding"] = dec_embed.astype(w_dtype)
        weights["dec_pos_embedding"] = _load_tensor(readers, "model.decoder.embed_positions.weight").astype(w_dtype)

        # Decoder layers
        for i in range(dec_layers):
            hf = f"model.decoder.layers.{i}"
            pfx = f"layer.{i}"
            # [C1] Decoder self-attn: k_proj has no bias
            for proj in ("q", "k", "v"):
                weights[f"{pfx}.w_{proj}"] = _transpose_2d(
                    _load_tensor(readers, f"{hf}.self_attn.{proj}_proj.weight"), f"dec_{proj}")
                weights[f"{pfx}.{proj}_bias"] = _load_bias_or_zeros(
                    readers, f"{hf}.self_attn.{proj}_proj.bias", hidden, dtype=w_dtype)
            weights[f"{pfx}.w_o"] = _transpose_2d(_load_tensor(readers, f"{hf}.self_attn.out_proj.weight"), "dec_o")
            weights[f"{pfx}.o_bias"] = _load_tensor(readers, f"{hf}.self_attn.out_proj.bias").astype(w_dtype)
            # Norm weights stay FP32
            weights[f"{pfx}.input_norm"] = _load_tensor(readers, f"{hf}.self_attn_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.input_norm_beta"] = _load_tensor(readers, f"{hf}.self_attn_layer_norm.bias").astype(np.float32)
            # [C1] Decoder cross-attn: k_proj has no bias
            for proj in ("q", "k", "v"):
                weights[f"{pfx}.cross_w_{proj}"] = _transpose_2d(
                    _load_tensor(readers, f"{hf}.encoder_attn.{proj}_proj.weight"), f"xattn_{proj}")
                weights[f"{pfx}.cross_b_{proj}"] = _load_bias_or_zeros(
                    readers, f"{hf}.encoder_attn.{proj}_proj.bias", hidden, dtype=w_dtype)
            weights[f"{pfx}.cross_w_o"] = _transpose_2d(_load_tensor(readers, f"{hf}.encoder_attn.out_proj.weight"), "xattn_o")
            weights[f"{pfx}.cross_b_o"] = _load_tensor(readers, f"{hf}.encoder_attn.out_proj.bias").astype(w_dtype)
            # Norm weights stay FP32
            weights[f"{pfx}.cross_attn_norm"] = _load_tensor(readers, f"{hf}.encoder_attn_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.cross_attn_norm_beta"] = _load_tensor(readers, f"{hf}.encoder_attn_layer_norm.bias").astype(np.float32)
            weights[f"{pfx}.w_fc1"] = _transpose_2d(_load_tensor(readers, f"{hf}.fc1.weight"), "dec_fc1")
            weights[f"{pfx}.fc1_bias"] = _load_tensor(readers, f"{hf}.fc1.bias").astype(w_dtype)
            weights[f"{pfx}.w_fc2"] = _transpose_2d(_load_tensor(readers, f"{hf}.fc2.weight"), "dec_fc2")
            weights[f"{pfx}.fc2_bias"] = _load_tensor(readers, f"{hf}.fc2.bias").astype(w_dtype)
            # Norm weights stay FP32
            weights[f"{pfx}.post_attn_norm"] = _load_tensor(readers, f"{hf}.final_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.post_attn_norm_beta"] = _load_tensor(readers, f"{hf}.final_layer_norm.bias").astype(np.float32)

        # Norm weights stay FP32
        weights["final_norm"] = _load_tensor(readers, "model.decoder.layer_norm.weight").astype(np.float32)
        weights["final_norm_beta"] = _load_tensor(readers, "model.decoder.layer_norm.bias").astype(np.float32)

        if _has_tensor(readers, "proj_out.weight"):
            weights["w_out"] = _transpose_2d(_load_tensor(readers, "proj_out.weight"), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(dec_embed.copy(), "embedding_tied")
        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Whisper tensor-parallel decoder builds")
            if quant_ctx is not None:
                raise ValueError("Whisper tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "Whisper tensor-parallel builds do not support debug_layer_outputs")
            return build_whisper_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                verbose=verbose,
                parallel_config=parallel,
            )

        dec_layers = weights["_dec_layers"]
        dec_heads = weights["_dec_heads"]
        dec_ffn = weights["_dec_ffn"]
        max_source_positions = weights["_max_source_positions"]
        hidden = config.hidden_size
        vocab = config.vocab_size
        head_dim = hidden // dec_heads
        attention_window = max_cache_length + 1

        # Precision configuration
        if precision == "fp16":
            work_np_dtype = np.float16
            work_trt_dtype = trt.float16
        elif precision == "bf16":
            work_np_dtype = np.float16  # stored as float16, TRT uses bfloat16
            work_trt_dtype = trt.bfloat16
        else:
            work_np_dtype = np.float32
            work_trt_dtype = trt.float32

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))

        cache_k_inputs, cache_v_inputs = [], []
        for i in range(dec_layers):
            cache_k_inputs.append(network.add_input(graph_ops.layer_tensor_name("cache_k", i), work_trt_dtype, (max_cache_length, hidden)))
            cache_v_inputs.append(network.add_input(graph_ops.layer_tensor_name("cache_v", i), work_trt_dtype, (max_cache_length, hidden)))

        # [C3] Cross-attention inputs: raw encoder output (projections baked in graph)
        cross_k_inputs, cross_v_inputs = [], []
        for i in range(dec_layers):
            cross_k_inputs.append(network.add_input(graph_ops.layer_tensor_name("cross_k", i), trt.float32, (max_source_positions, hidden)))
            cross_v_inputs.append(network.add_input(graph_ops.layer_tensor_name("cross_v", i), trt.float32, (max_source_positions, hidden)))

        embedding_table = graph_ops.add_constant(network, (vocab, hidden), weights["dec_embedding"], dtype=work_np_dtype)
        pos_embed_np = weights["dec_pos_embedding"]
        pos_embedding_table = graph_ops.add_constant(network, pos_embed_np.shape, pos_embed_np, dtype=work_np_dtype)
        eps_tensor = graph_ops.add_constant(network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype)
        # Cast attention mask to work dtype for elementwise compatibility
        if work_trt_dtype != trt.float32:
            attention_mask = network.add_cast(attention_mask, work_trt_dtype).get_output(0)

        hidden_state = network.add_elementwise(
            network.add_gather(embedding_table, token_id, 0).get_output(0),
            network.add_gather(pos_embedding_table, position_id, 0).get_output(0),
            trt.ElementWiseOperation.SUM).get_output(0)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        present_k_outputs, present_v_outputs = [], []
        for layer_idx in range(dec_layers):
            prefix = f"layer.{layer_idx}"
            result = _add_whisper_decoder_layer(
                network=network, hidden=hidden_state,
                cache_k=cache_k_inputs[layer_idx], cache_v=cache_v_inputs[layer_idx],
                cross_k=cross_k_inputs[layer_idx], cross_v=cross_v_inputs[layer_idx],
                attention_mask=attention_mask, eps_tensor=eps_tensor, weights=weights, prefix=prefix,
                hidden_size=hidden, num_heads=dec_heads, head_dim=head_dim,
                ffn_dim=dec_ffn, max_cache_length=max_cache_length,
                max_source_positions=max_source_positions,
                dtype=work_np_dtype)
            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            if debug_layer_outputs:
                _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

        hidden_state = graph_ops.add_layer_norm(network, hidden_state, hidden, weights["final_norm"], weights["final_norm_beta"], eps_tensor, dtype=work_np_dtype)
        logits = graph_ops.add_matmul_rhs_constant(network, hidden_state, hidden, vocab, weights["w_out"], dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(network, logits, vocab, np.zeros(vocab, dtype=work_np_dtype), dtype=work_np_dtype)

        # Logits output: always FP32 for accurate argmax/sampling
        if work_trt_dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        for i in range(dec_layers):
            present_k_outputs[i].name = graph_ops.layer_tensor_name("present_k", i)
            present_v_outputs[i].name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(present_k_outputs[i])
            network.mark_output(present_v_outputs[i])

        if verbose:
            print(f"[trtmc build] Building Whisper decoder ({dec_layers}L, h={hidden}, heads={dec_heads}, ffn={dec_ffn}, cache={max_cache_length}, precision={precision})", file=sys.stderr)
        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT decoder engine build failed")
        return bytes(plan)

    def build_vision_engine(self, model_dir: str, config: ModelConfig, weights: WeightDict, *, precision: str = "fp32", verbose: bool = False) -> bytes | None:
        return _build_whisper_encoder(config, weights, precision=precision, verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        raw = config.raw
        return {
            "num_mel_bins": raw.get("num_mel_bins", 80),
            "max_source_positions": raw.get("max_source_positions", 1500),
            "max_target_positions": raw.get("max_target_positions", 448),
            "encoder_layers": raw.get("encoder_layers", config.num_hidden_layers),
            "decoder_layers": raw.get("decoder_layers", config.num_hidden_layers),
            "encoder_ffn_dim": raw.get("encoder_ffn_dim", config.intermediate_size),
            "decoder_ffn_dim": raw.get("decoder_ffn_dim", config.intermediate_size),
            "encoder_attention_heads": raw.get("encoder_attention_heads", config.num_attention_heads),
            "decoder_attention_heads": raw.get("decoder_attention_heads", config.num_attention_heads),
            "has_vision_engine": True,
        }

    def get_audio_config(self, config: ModelConfig) -> dict | None:
        raw = config.raw
        return {
            "mel_n_fft": raw.get("n_fft", 400),
            "mel_hop_length": raw.get("hop_length", 160),
            "mel_chunk_length": raw.get("chunk_length", 30),
            "mel_sampling_rate": raw.get("sampling_rate", 16000),
        }

    def build_extra_engines(self, config: ModelConfig, weights, max_cache_length: int, *, precision: str = "fp32", verbose: bool = False) -> dict | None:
        """Bake the mel filterbank matrix into the bundle as a binary section."""
        raw = config.raw
        num_mel_bins = raw.get("num_mel_bins", 80)
        n_fft = raw.get("n_fft", 400)
        sampling_rate = raw.get("sampling_rate", 16000)
        n_freq_bins = 1 + n_fft // 2  # 201 for n_fft=400

        try:
            from transformers.audio_utils import mel_filter_bank
        except ImportError:
            print("[trtmc build] Warning: transformers.audio_utils not available, "
                  "skipping mel filterbank embedding", file=sys.stderr)
            return None

        # Compute the Slaney mel filterbank (matches WhisperFeatureExtractor)
        # Returns shape [num_frequency_bins, num_mel_filters] = [201, 80]
        filters = mel_filter_bank(
            num_frequency_bins=n_freq_bins,
            num_mel_filters=num_mel_bins,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        # filters shape: [num_frequency_bins, num_mel_filters] = [n_freq_bins, n_mel_bins]
        # C++ expects: [n_freq_bins, n_mel_bins] (rows=freq, cols=mel) — same layout
        filters_flat = np.ascontiguousarray(filters, dtype=np.float32)

        # Pack as binary: [n_freq_bins(int32), n_mel_bins(int32), float32 data...]
        header = np.array([n_freq_bins, num_mel_bins], dtype=np.int32)
        mel_fb_bytes = header.tobytes() + filters_flat.tobytes()

        if verbose:
            print(f"[trtmc build] Mel filterbank: {n_freq_bins}x{num_mel_bins} "
                  f"({len(mel_fb_bytes)} bytes)", file=sys.stderr)

        return {"mel_filterbank": mel_fb_bytes}


def _build_whisper_encoder(config, weights, *, precision="fp32", verbose=False):
    enc_layers = weights["_enc_layers"]
    enc_heads = weights["_enc_heads"]
    enc_ffn = weights["_enc_ffn"]
    num_mel_bins = weights["_num_mel_bins"]
    max_source_positions = weights["_max_source_positions"]
    hidden = config.hidden_size
    mel_length = max_source_positions * 2

    # Precision configuration
    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = builder.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    eps_tensor = graph_ops.add_constant(network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype)
    mel_input = network.add_input("mel_features", work_trt_dtype, (num_mel_bins, mel_length))

    # TRT requires 2D+ convolutions; reshape 1D conv weights [out, in, k] -> [out, in, 1, k]
    # and input from [1, C, L] -> [1, C, 1, L]
    ri = network.add_shuffle(mel_input)
    ri.reshape_dims = (1, num_mel_bins, 1, mel_length)
    conv1_w = weights["enc_conv1_weight"]
    conv1_w_4d = np.ascontiguousarray(conv1_w.reshape(conv1_w.shape[0], conv1_w.shape[1], 1, conv1_w.shape[2]), dtype=work_np_dtype)
    c1 = network.add_convolution_nd(ri.get_output(0), num_output_maps=hidden, kernel_shape=(1, 3),
        kernel=trt.Weights(conv1_w_4d),
        bias=trt.Weights(np.ascontiguousarray(weights["enc_conv1_bias"], dtype=work_np_dtype)))
    c1.stride_nd = (1, 1)
    c1.padding_nd = (0, 1)
    # Conv1 output: [1, hidden, 1, mel_length]. Squeeze to 2D for GELU, then back to 4D.
    c1_sq = network.add_shuffle(c1.get_output(0))
    c1_sq.reshape_dims = (hidden, mel_length)
    c1o_2d = graph_ops.add_activation(network, c1_sq.get_output(0), "gelu_new", dtype=work_np_dtype)
    c1_unsq = network.add_shuffle(c1o_2d)
    c1_unsq.reshape_dims = (1, hidden, 1, mel_length)

    conv2_w = weights["enc_conv2_weight"]
    conv2_w_4d = np.ascontiguousarray(conv2_w.reshape(conv2_w.shape[0], conv2_w.shape[1], 1, conv2_w.shape[2]), dtype=work_np_dtype)
    c2 = network.add_convolution_nd(c1_unsq.get_output(0), num_output_maps=hidden, kernel_shape=(1, 3),
        kernel=trt.Weights(conv2_w_4d),
        bias=trt.Weights(np.ascontiguousarray(weights["enc_conv2_bias"], dtype=work_np_dtype)))
    c2.stride_nd = (1, 2)
    c2.padding_nd = (0, 1)
    # Conv2 output: [1, hidden, 1, max_source_positions]. Squeeze to 2D for GELU.
    c2_sq = network.add_shuffle(c2.get_output(0))
    c2_sq.reshape_dims = (hidden, max_source_positions)
    c2o_2d = graph_ops.add_activation(network, c2_sq.get_output(0), "gelu_new", dtype=work_np_dtype)

    # Transpose to [max_source_positions, hidden]
    cr = network.add_shuffle(c2o_2d)
    cr.first_transpose = trt.Permutation([1, 0])
    hs = cr.get_output(0)

    # [C2] Use LEARNED positional embeddings (not sinusoidal)
    enc_pos_np = weights["enc_pos_embedding"]
    hs = network.add_elementwise(
        hs,
        graph_ops.add_constant(network, (max_source_positions, hidden), enc_pos_np, dtype=work_np_dtype),
        trt.ElementWiseOperation.SUM).get_output(0)

    for li in range(enc_layers):
        pfx = f"enc_layer.{li}"
        normed = graph_ops.add_layer_norm(network, hs, hidden, weights[f"{pfx}.attn_norm"], weights[f"{pfx}.attn_norm_beta"], eps_tensor, dtype=work_np_dtype)
        attn = graph_ops.add_self_attention_block(network, normed,
            w_q=weights[f"{pfx}.w_q"], w_k=weights[f"{pfx}.w_k"], w_v=weights[f"{pfx}.w_v"], w_o=weights[f"{pfx}.w_o"],
            hidden_size=hidden, num_heads=enc_heads, seq_length=max_source_positions,
            q_bias=weights[f"{pfx}.b_q"], k_bias=weights[f"{pfx}.b_k"], v_bias=weights[f"{pfx}.b_v"], o_bias=weights[f"{pfx}.b_o"],
            dtype=work_np_dtype)
        hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
        n2 = graph_ops.add_layer_norm(network, hs, hidden, weights[f"{pfx}.ffn_norm"], weights[f"{pfx}.ffn_norm_beta"], eps_tensor, dtype=work_np_dtype)
        fc1 = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, n2, hidden, enc_ffn, weights[f"{pfx}.w_fc1"], dtype=work_np_dtype), enc_ffn, weights[f"{pfx}.b_fc1"], dtype=work_np_dtype)
        act = graph_ops.add_activation(network, fc1, "gelu_new", dtype=work_np_dtype)
        fc2 = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, act, enc_ffn, hidden, weights[f"{pfx}.w_fc2"], dtype=work_np_dtype), hidden, weights[f"{pfx}.b_fc2"], dtype=work_np_dtype)
        hs = network.add_elementwise(hs, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    hs = graph_ops.add_layer_norm(network, hs, hidden, weights["enc_final_norm"], weights["enc_final_norm_beta"], eps_tensor, dtype=work_np_dtype)

    # Encoder output: always FP32 for downstream compatibility
    if work_trt_dtype != trt.float32:
        hs = network.add_cast(hs, trt.float32).get_output(0)
    hs.name = "encoder_output"
    network.mark_output(hs)

    if verbose:
        print(f"[trtmc build] Building Whisper encoder ({enc_layers}L, h={hidden}, heads={enc_heads}, mel={num_mel_bins}, precision={precision})", file=sys.stderr)
    plan = builder.build_serialized_network(network, tc)
    if plan is None:
        raise RuntimeError("TensorRT encoder engine build failed")
    return bytes(plan)


def _add_whisper_decoder_layer(*, network, hidden, cache_k, cache_v, cross_k, cross_v,
    attention_mask, eps_tensor, weights, prefix,
    hidden_size, num_heads, head_dim, ffn_dim, max_cache_length, max_source_positions,
    dtype=np.float32):
    attention_size = hidden_size
    attention_window = max_cache_length + 1

    # Self-attention
    normed = graph_ops.add_layer_norm(network, hidden, hidden_size, weights[f"{prefix}.input_norm"], weights[f"{prefix}.input_norm_beta"], eps_tensor, dtype=dtype)
    q = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, normed, hidden_size, attention_size, weights[f"{prefix}.w_q"], dtype=dtype), attention_size, weights[f"{prefix}.q_bias"], dtype=dtype)
    k = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, normed, hidden_size, attention_size, weights[f"{prefix}.w_k"], dtype=dtype), attention_size, weights[f"{prefix}.k_bias"], dtype=dtype)
    v = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, normed, hidden_size, attention_size, weights[f"{prefix}.w_v"], dtype=dtype), attention_size, weights[f"{prefix}.v_bias"], dtype=dtype)
    present_k, present_v = k, v

    kr = network.add_shuffle(k)
    kr.reshape_dims = (1, attention_size)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (1, attention_size)
    ak = network.add_concatenation([cache_k, kr.get_output(0)])
    ak.axis = 0
    av = network.add_concatenation([cache_v, vr.get_output(0)])
    av.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    cf = graph_ops.add_attention_from_rows(
        network, q, ak.get_output(0), av.get_output(0),
        num_heads=num_heads, head_dim=head_dim,
        q_seq=1, kv_seq=attention_window,
        mask=mask_4d)
    sa = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, cf, attention_size, hidden_size, weights[f"{prefix}.w_o"], dtype=dtype), hidden_size, weights[f"{prefix}.o_bias"], dtype=dtype)
    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)

    # Cross-attention
    # [C3] Apply per-layer K/V projections to raw encoder output BEFORE multi-head reshape
    cn = graph_ops.add_layer_norm(network, psa, hidden_size, weights[f"{prefix}.cross_attn_norm"], weights[f"{prefix}.cross_attn_norm_beta"], eps_tensor, dtype=dtype)
    cq = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, cn, hidden_size, attention_size, weights[f"{prefix}.cross_w_q"], dtype=dtype), attention_size, weights[f"{prefix}.cross_b_q"], dtype=dtype)

    # Project raw encoder output through per-layer K/V weights.
    # Cross inputs are FP32 (from encoder output); cast to work dtype for matmul.
    cross_k_typed = cross_k
    cross_v_typed = cross_v
    if dtype == np.float16:
        cross_k_typed = network.add_cast(cross_k, trt.float16).get_output(0)
        cross_v_typed = network.add_cast(cross_v, trt.float16).get_output(0)
    ck_proj = graph_ops.add_bias_sum(network,
        graph_ops.add_matmul_rhs_constant(network, cross_k_typed, hidden_size, attention_size, weights[f"{prefix}.cross_w_k"], dtype=dtype),
        attention_size, weights[f"{prefix}.cross_b_k"], dtype=dtype)
    cv_proj = graph_ops.add_bias_sum(network,
        graph_ops.add_matmul_rhs_constant(network, cross_v_typed, hidden_size, attention_size, weights[f"{prefix}.cross_w_v"], dtype=dtype),
        attention_size, weights[f"{prefix}.cross_b_v"], dtype=dtype)

    ccf = graph_ops.add_attention_from_rows(
        network, cq, ck_proj, cv_proj,
        num_heads=num_heads, head_dim=head_dim,
        q_seq=1, kv_seq=max_source_positions,
        fp32_accumulation=True,
        tag=f"{prefix}.cross_attn")
    ca = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, ccf, attention_size, hidden_size, weights[f"{prefix}.cross_w_o"], dtype=dtype), hidden_size, weights[f"{prefix}.cross_b_o"], dtype=dtype)
    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)

    # GELU MLP
    fn = graph_ops.add_layer_norm(network, pca, hidden_size, weights[f"{prefix}.post_attn_norm"], weights[f"{prefix}.post_attn_norm_beta"], eps_tensor, dtype=dtype)
    mlp = graph_blocks.add_gelu_fc_mlp(network, fn, weights=weights, prefix=prefix, hidden_size=hidden_size, mlp_size=ffn_dim, activation="gelu_new", dtype=dtype)
    out = network.add_elementwise(pca, mlp, trt.ElementWiseOperation.SUM).get_output(0)
    return {"hidden": out, "present_k": present_k, "present_v": present_v}

def _mark_debug_output(network, tensor, name):
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


plugin = WhisperPlugin()
