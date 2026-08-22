# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni family plugin with a text-only native Thinker product path.

The active bundle contract contains the Thinker engine plus tokenizer assets.
Vision, audio, Talker, and Code2Wav helpers remain private below so future
native multimodal work can reuse them without exposing dormant build hooks.

Qwen3-Omni is a 3-stage multimodal model:
  1. Thinker: Multimodal MoE decoder (text + image + audio input -> text output)
     - Vision encoder (reuses Qwen VL pattern with 3D RoPE)
     - Audio encoder (Whisper-like mel -> transformer encoder)
     - MoE text decoder (Qwen3 MoE architecture)
  2. Talker: Text embeddings -> 16-group RVQ speech codec tokens
     - Runs the checkpoint's complete 20-layer MoE Talker and residual-code
       predictor through the model-owned runtime bridge
  3. Code2Wav: Codec tokens -> audio waveform
     - Exports the complete official pre-transformer, upsampler, and decoder

The active Thinker MoE decoder follows Qwen3 MoE (sibling model) with the same
top-k softmax routing and accepts token IDs only.

Weight key mapping:
  Thinker MoE decoder:
    model.thinker.layers.{i}.input_layernorm.weight
    model.thinker.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.thinker.layers.{i}.block_sparse_moe.gate.weight
    model.thinker.layers.{i}.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight

  Audio encoder:
    model.thinker.audio_tower.conv1.weight/bias
    model.thinker.audio_tower.conv2.weight/bias
    model.thinker.audio_tower.layers.{i}.*

  Code2Wav:
    model.code2wav.pre_transformer.*
    model.code2wav.upsample.*
    model.code2wav.decoder.*
"""

from __future__ import annotations

import io
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
    _target_np_dtype,
    _transpose_2d,
)
from ...build_timing import timed_trt_compile
from . import graph_ops
from . import graph_blocks
from .standard_decoder_builder import _mark_debug_output


trt = trt_compat.get_trt()


class Qwen3OmniPlugin:
    name = "qwen3_omni"
    runtime_strategy = "qwen3_omni_multimodal"
    embed_input = False

    def __init__(self):
        self._thinker_cfg: dict = {}
        self._talker_cfg: dict = {}
        self._audio_encoder_cfg: dict = {}
        self._code2wav_cfg: dict = {}

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in ("qwen3_omni", "qwen3_omni_moe", "qwen3omni")

    def load_weights(
        self, model_dir: str, config: ModelConfig, *, precision: str = "fp32",
    ) -> WeightDict:
        """Load Qwen3-Omni weights from safetensors.

        Loads only the Thinker MoE text-decoder weights consumed by the
        current native runtime. Private multimodal builders remain available
        for a future product path but are not part of today's bundle contract.
        """
        readers = _open_safetensors(Path(model_dir))

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        target_dtype = _target_np_dtype(precision)

        # MoE config lives in thinker_config.text_config for Qwen3-Omni
        thinker_text = (config.raw.get("thinker_config", {})
                        .get("text_config", {}))
        num_experts = (thinker_text.get("num_experts")
                       or config.raw.get("num_local_experts")
                       or config.raw.get("num_experts", 8))
        num_experts_per_tok = (thinker_text.get("num_experts_per_tok")
                               or config.raw.get("num_experts_per_tok", 2))
        # moe_intermediate_size is per-expert intermediate; intermediate_size
        # from config may be the same value for MoE models.
        intermediate_size = (thinker_text.get("moe_intermediate_size")
                             or config.intermediate_size)

        weights = WeightDict()

        # Helper: find first existing key from a list of candidates
        def _find_key(candidates: list[str]) -> str:
            for c in candidates:
                if _has_tensor(readers, c):
                    return c
            raise KeyError(
                f"None of {candidates} found in safetensors")

        # Embedding (thinker shared embedding)
        embed_key = _find_key([
            "thinker.model.embed_tokens.weight",
            "model.thinker.embed_tokens.weight",
            "model.embed_tokens.weight",
        ])
        embedding = _load_tensor(readers, embed_key)
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(target_dtype)

        attention_size = 0

        # Detect layer prefix: thinker.model.layers.{i} vs model.thinker.layers.{i}
        _layer_prefix_candidates = [
            "thinker.model.layers",
            "model.thinker.layers",
            "model.layers",
        ]
        hf_layer_base = _layer_prefix_candidates[0]
        for cand in _layer_prefix_candidates:
            if _has_tensor(readers, f"{cand}.0.input_layernorm.weight"):
                hf_layer_base = cand
                break

        # Load Thinker MoE decoder layers
        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"{hf_layer_base}.{layer_idx}"

            # RMSNorm weights
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)

            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # Q/K/V/O projections (separate, no biases)
            q_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.o_proj.weight")

            if attention_size == 0:
                attention_size = q_raw.shape[0]

            q_t = _transpose_2d(q_raw, "q_proj", precision)
            k_t = _transpose_2d(k_raw, "k_proj", precision)
            v_t = _transpose_2d(v_raw, "v_proj", precision)
            o_t = _transpose_2d(o_raw, "o_proj", precision)
            del q_raw, k_raw, v_raw, o_raw

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            # Optional per-head q/k norms (Qwen3)
            q_norm_key = f"{hf_prefix}.self_attn.q_norm.weight"
            if _has_tensor(readers, q_norm_key):
                q_norm = _load_tensor(readers, q_norm_key).astype(np.float32)
                weights[f"{prefix}.q_norm"] = np.tile(q_norm, num_heads)
            k_norm_key = f"{hf_prefix}.self_attn.k_norm.weight"
            if _has_tensor(readers, k_norm_key):
                k_norm = _load_tensor(readers, k_norm_key).astype(np.float32)
                weights[f"{prefix}.k_norm"] = np.tile(k_norm, num_kv_heads)

            # MoE block: router + per-expert weights
            # Try both naming conventions: mlp.gate (Qwen3-Omni) and
            # block_sparse_moe.gate (older Mixtral-style)
            router_key = f"{hf_prefix}.mlp.gate.weight"
            moe_expert_fmt = "mlp"
            if not _has_tensor(readers, router_key):
                router_key = f"{hf_prefix}.block_sparse_moe.gate.weight"
                moe_expert_fmt = "block_sparse_moe"
            if _has_tensor(readers, router_key):
                # MoE layer
                router_raw = _load_tensor(readers, router_key)
                weights[f"{prefix}.router"] = _transpose_2d(
                    router_raw, "router", precision)
                del router_raw

                expert_gate = np.empty(
                    (num_experts, hidden, intermediate_size), dtype=target_dtype)
                expert_up = np.empty(
                    (num_experts, hidden, intermediate_size), dtype=target_dtype)
                expert_down = np.empty(
                    (num_experts, intermediate_size, hidden), dtype=target_dtype)
                for e in range(num_experts):
                    if moe_expert_fmt == "mlp":
                        # Qwen3-Omni: mlp.experts.{e}.gate_proj/up_proj/down_proj
                        exp_prefix = f"{hf_prefix}.mlp.experts.{e}"
                        gate_raw = _load_tensor(
                            readers, f"{exp_prefix}.gate_proj.weight")
                        up_raw = _load_tensor(
                            readers, f"{exp_prefix}.up_proj.weight")
                        down_raw = _load_tensor(
                            readers, f"{exp_prefix}.down_proj.weight")
                    else:
                        # Mixtral-style: block_sparse_moe.experts.{e}.w1/w3/w2
                        exp_prefix = (
                            f"{hf_prefix}.block_sparse_moe.experts.{e}")
                        gate_raw = _load_tensor(
                            readers, f"{exp_prefix}.w1.weight")
                        up_raw = _load_tensor(
                            readers, f"{exp_prefix}.w3.weight")
                        down_raw = _load_tensor(
                            readers, f"{exp_prefix}.w2.weight")

                    expert_gate[e] = _transpose_2d(
                        gate_raw, f"expert_{e}_gate", precision)
                    expert_up[e] = _transpose_2d(
                        up_raw, f"expert_{e}_up", precision)
                    expert_down[e] = _transpose_2d(
                        down_raw, f"expert_{e}_down", precision)
                    del gate_raw, up_raw, down_raw
                weights[f"{prefix}.experts.w_gate"] = expert_gate
                weights[f"{prefix}.experts.w_up"] = expert_up
                weights[f"{prefix}.experts.w_down"] = expert_down
            else:
                # Dense SwiGLU MLP (some layers may not be MoE)
                gate_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.gate_proj.weight")
                up_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.up_proj.weight")
                down_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.down_proj.weight")

                weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate", precision)
                weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up", precision)
                weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down", precision)

        # Final norm
        final_norm_key = None
        for cand in ["thinker.model.norm.weight",
                      "model.thinker.norm.weight",
                      "model.norm.weight"]:
            if _has_tensor(readers, cand):
                final_norm_key = cand
                break
        if final_norm_key is not None:
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = None
        for cand in ["thinker.lm_head.weight",
                      "lm_head.weight",
                      "model.thinker.lm_head.weight"]:
            if _has_tensor(readers, cand):
                lm_head_key = cand
                break
        if lm_head_key is not None:
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head", precision)
        else:
            weights["w_out"] = _transpose_2d(
                weights["embedding"], "embedding_tied", precision)

        weights["_attention_size"] = attention_size
        weights["_num_experts"] = num_experts
        weights["_moe_intermediate_size"] = intermediate_size
        weights["_num_experts_per_tok"] = num_experts_per_tok

        # Store configs for extra engine builders
        self._thinker_cfg = {
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "intermediate_size": intermediate_size,
        }

        return weights

    def _detect_audio_encoder(self, readers) -> dict:
        """Detect audio encoder dimensions from weight shapes."""
        # Detect the first conv layer. Qwen3-Omni uses conv2d1, not conv1.
        conv_key = None
        for cand in ["thinker.audio_tower.conv2d1.weight",
                      "thinker.audio_tower.conv1.weight",
                      "model.thinker.audio_tower.conv1.weight"]:
            if _has_tensor(readers, cand):
                conv_key = cand
                break
        if conv_key is None:
            return {}
        conv_w = _load_tensor(readers, conv_key)
        # conv: [out_channels, in_channels, kernel_size, ...]
        embed_dim = conv_w.shape[0]
        num_mel = conv_w.shape[1]

        # Count encoder layers (try both prefixes)
        num_layers = 0
        audio_layer_base = "thinker.audio_tower.layers"
        if not _has_tensor(
            readers,
            f"{audio_layer_base}.0.self_attn.q_proj.weight"
        ):
            audio_layer_base = "model.thinker.audio_tower.layers"
        while _has_tensor(
            readers,
            f"{audio_layer_base}.{num_layers}.self_attn.q_proj.weight"
        ):
            num_layers += 1

        # Detect num_heads from Q projection shape
        num_heads = embed_dim // 64
        q_key = f"{audio_layer_base}.0.self_attn.q_proj.weight"
        if _has_tensor(readers, q_key):
            q_w = _load_tensor(readers, q_key)
            q_out_dim = q_w.shape[0]
            for hd in [64, 128, 96, 80]:
                if q_out_dim % hd == 0:
                    num_heads = q_out_dim // hd
                    break

        return {
            "embed_dim": int(embed_dim),
            "num_mel_bins": int(num_mel),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
        }

    def _detect_talker(self, readers, config: ModelConfig) -> dict:
        """Detect Talker decoder dimensions from weight shapes."""
        # Try both key prefixes
        embed_key = "talker.model.embed_tokens.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "model.talker.embed_tokens.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "talker.model.codec_embedding.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "model.talker.model.codec_embedding.weight"
        if not _has_tensor(readers, embed_key):
            return {}
        embed_w = _load_tensor(readers, embed_key)
        talker_vocab, talker_hidden = embed_w.shape

        talker_raw = config.raw.get("talker_config", {})
        talker_text = talker_raw.get("text_config", {})
        input_hidden = (
            talker_raw.get("thinker_hidden_size")
            or config.hidden_size
        )
        decoder_hidden = talker_text.get("hidden_size", talker_hidden)

        # Detect layer prefix
        talker_layer_base = "talker.model.layers"
        if not _has_tensor(
            readers,
            f"{talker_layer_base}.0.input_layernorm.weight"
        ):
            talker_layer_base = "model.talker.layers"
        num_layers = 0
        while _has_tensor(
            readers,
            f"{talker_layer_base}.{num_layers}.input_layernorm.weight"
        ):
            num_layers += 1

        # Count RVQ codebook heads (try both prefixes)
        codec_base = "talker.codec_head"
        if not _has_tensor(readers, f"{codec_base}.weight"):
            codec_base = "model.talker.codec_head"

        # Detect if codec_head is a single weight or indexed (0, 1, ...)
        n_codebooks = 0
        codebook_size = 0
        if _has_tensor(readers, f"{codec_base}.0.weight"):
            while _has_tensor(readers, f"{codec_base}.{n_codebooks}.weight"):
                n_codebooks += 1
            if n_codebooks > 0:
                head_w = _load_tensor(readers, f"{codec_base}.0.weight")
                codebook_size = head_w.shape[0]
        elif _has_tensor(readers, f"{codec_base}.weight"):
            # Single codec_head (Qwen3-Omni uses talker.codec_head.weight)
            head_w = _load_tensor(readers, f"{codec_base}.weight")
            codebook_size = head_w.shape[0]
            n_codebooks = 1

        # Also check for code_predictor.lm_head.* pattern
        # (Qwen3-Omni uses talker.code_predictor.lm_head.{i}.weight)
        if n_codebooks == 0:
            cp_base = "talker.code_predictor.lm_head"
            if not _has_tensor(readers, f"{cp_base}.0.weight"):
                cp_base = "model.talker.code_predictor.lm_head"
            while _has_tensor(readers, f"{cp_base}.{n_codebooks}.weight"):
                n_codebooks += 1
            if n_codebooks > 0:
                head_w = _load_tensor(readers, f"{cp_base}.0.weight")
                codebook_size = head_w.shape[0]
                # Qwen3-Omni predicts one semantic group with codec_head plus
                # the indexed code_predictor heads for the remaining groups.
                n_codebooks += 1

        n_codebooks = int(talker_raw.get("num_code_groups", n_codebooks))
        code_predictor = talker_raw.get("code_predictor_config", {})
        codebook_size = int(code_predictor.get("vocab_size", codebook_size))

        return {
            "vocab_size": int(talker_vocab),
            "hidden_size": int(input_hidden),
            "decoder_hidden_size": int(decoder_hidden),
            "num_layers": int(num_layers),
            "n_codebooks": int(n_codebooks),
            "codebook_size": int(codebook_size),
        }

    def _detect_code2wav(self, readers, config: ModelConfig) -> dict:
        """Detect the official Qwen3-Omni Code2Wav checkpoint layout."""
        raw = config.raw.get("code2wav_config", {})
        embed_key = "code2wav.code_embedding.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "model.code2wav.code_embedding.weight"
        if not _has_tensor(readers, embed_key):
            return {}

        embedding = _load_tensor(readers, embed_key)
        num_quantizers = int(raw.get("num_quantizers", 0))
        codebook_size = int(raw.get("codebook_size", 0))
        if (
            num_quantizers <= 0
            or codebook_size <= 0
            or embedding.shape != (num_quantizers * codebook_size, int(raw.get("hidden_size", 0)))
        ):
            raise RuntimeError("Qwen3-Omni Code2Wav checkpoint/config dimensions do not match")

        required = (
            "code2wav.pre_transformer.layers.0.self_attn.q_proj.weight",
            "code2wav.upsample.0.0.conv.weight",
            "code2wav.decoder.0.conv.weight",
            "code2wav.decoder.6.conv.weight",
        )
        if not all(_has_tensor(readers, name) for name in required):
            raise RuntimeError(
                "Qwen3-Omni Code2Wav checkpoint is incomplete: expected the "
                "official pre-transformer, nested upsampler, and decoder layout"
            )

        return {
            "available": True,
            "config": dict(raw),
            "max_frames": 32,
            "upsample_factor": 1920,
            "output_delay": 555,
        }

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        """Build TRT engine for Thinker MoE decoder (primary engine).

        This builds the main text decoder with MoE routing. Vision and audio
        features are injected via embed_input mode during prefill.
        """
        attention_size: int = weights.get("_attention_size", config.attention_size)
        num_experts: int = weights.get("_num_experts", 8)
        moe_intermediate: int = weights.get("_moe_intermediate_size",
                                             config.intermediate_size)
        top_k: int = weights.get("_num_experts_per_tok", 2)
        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attention_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
        attention_window = max_cache_length + 1
        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()

        if precision == "fp16":
            work_np_dtype = np.float16
            work_trt_dtype = trt.float16
        elif precision == "bf16":
            work_np_dtype = np.float16
            work_trt_dtype = trt.bfloat16
        else:
            work_np_dtype = np.float32
            work_trt_dtype = trt.float32

        # Inputs
        token_id = network.add_input("token_id", trt.int32, (-1,))
        position_id = network.add_input("position_id", trt.int32, (-1,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (-1, -1))

        # KV cache inputs
        cache_k_inputs = []
        cache_v_inputs = []
        for i in range(num_layers):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                work_trt_dtype, (max_cache_length, kv_attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                work_trt_dtype, (max_cache_length, kv_attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False) -> None:
            profile = builder.create_optimization_profile()
            min_sq = opt_sq if fixed else 1
            profile.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
            profile.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
            profile.set_shape(
                "attention_mask",
                (min_sq, max_cache_length + min_sq),
                (opt_sq, max_cache_length + opt_sq),
                (max_sq, max_cache_length + max_sq))
            trt_config.add_optimization_profile(profile)

        _add_profile(min(64, max_cache_length), max_cache_length)
        _add_profile(1, 1, fixed=True)

        # Shared constants
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)

        graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
        cos_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, True)
        sin_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, False)
        cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)

        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype)

        if work_trt_dtype != trt.float32:
            attention_mask = network.add_cast(attention_mask, work_trt_dtype).get_output(0)
            cos_half_tensor = network.add_cast(cos_half_tensor, work_trt_dtype).get_output(0)
            sin_half_tensor = network.add_cast(sin_half_tensor, work_trt_dtype).get_output(0)
            eps_tensor = network.add_cast(eps_tensor, work_trt_dtype).get_output(0)

        # Pure text-token embedding lookup.
        gather = network.add_gather(embedding_table, token_id, 0)
        token_embed = gather.get_output(0)
        if token_embed.dtype != work_trt_dtype:
            token_embed = network.add_cast(token_embed, work_trt_dtype).get_output(0)
        hidden_state = token_embed

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        # Decoder layers with MoE
        present_k_outputs = []
        present_v_outputs = []

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"

            # Attention block via graph_blocks
            attn = graph_blocks.add_attention_block(
                network, hidden_state, cache_k_inputs[layer_idx],
                cache_v_inputs[layer_idx], attention_mask, position_id,
                weights=weights, prefix=prefix,
                hidden_size=hidden, attention_size=attention_size,
                kv_attention_size=kv_attention_size,
                num_heads=num_heads, head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                max_cache_length=max_cache_length,
                eps_tensor=eps_tensor,
                norm_type="rmsnorm", position_type="rope",
                cos_half_tensor=cos_half_tensor,
                sin_half_tensor=sin_half_tensor,
                rotary_embedding_dim=head_dim,
                dtype=work_np_dtype,
                dynamic_kv_cache=True,
                sequence_length=None,
            )

            attn_out = attn["attn_out"]
            present_k_outputs.append(attn["present_k"])
            present_v_outputs.append(attn["present_v"])

            # Residual after attention
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            post_attn = residual1.get_output(0)

            # Post-attention norm
            norm2 = graph_blocks.apply_norm(
                network, post_attn, hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor, "rmsnorm", dtype=work_np_dtype)

            # Check if this is an MoE or dense layer
            if f"{prefix}.router" in weights:
                # MoE block
                moe_out = _add_omni_moe_block(
                    network, norm2, weights, prefix,
                    hidden, num_experts, top_k,
                    dtype=work_np_dtype)
            else:
                # Dense SwiGLU MLP
                moe_out = graph_blocks.add_swiglu_mlp(
                    network, norm2, weights=weights, prefix=prefix,
                    hidden_size=hidden,
                    mlp_size=weights[f"{prefix}.w_gate"].shape[1],
                    dtype=work_np_dtype)

            # Residual
            residual2 = network.add_elementwise(
                post_attn, moe_out, trt.ElementWiseOperation.SUM)
            hidden_state = residual2.get_output(0)

            if debug_layer_outputs:
                _mark_debug_output(
                    network, residual1.get_output(0),
                    f"debug_post_attn_{layer_idx}")
                _mark_debug_output(
                    network, hidden_state,
                    f"debug_hidden_{layer_idx}")

        # Final norm
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = graph_blocks.apply_norm(
                network, hidden_state, hidden, final_norm, None,
                eps_tensor, "rmsnorm", dtype=work_np_dtype)

        # Keep the output contract fixed at one row for both profiles. Only
        # the final prompt row can affect the first generated token.
        hidden_shape = network.add_shape(hidden_state).get_output(0)
        one_hidden = graph_ops.add_constant(
            network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
        last_start = network.add_elementwise(
            hidden_shape, one_hidden, trt.ElementWiseOperation.SUB).get_output(0)
        last_size = graph_ops.add_constant(
            network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
        last_slice = network.add_slice(
            hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
        last_slice.set_input(1, last_start)
        last_slice.set_input(2, last_size)
        last_hidden = last_slice.get_output(0)

        hidden_out = network.add_identity(last_hidden).get_output(0)
        hidden_out.name = "hidden_state"
        network.mark_output(hidden_out)

        # LM head
        logits = graph_ops.add_matmul_rhs_constant(
            network, last_hidden, hidden, vocab, weights["w_out"],
            dtype=work_np_dtype)
        b_out = np.zeros(vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(
            network, logits, vocab, b_out, dtype=work_np_dtype)

        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)

        logits.name = "logits"
        network.mark_output(logits)

        # Present K/V outputs
        for i in range(num_layers):
            pk = present_k_outputs[i]
            pv = present_v_outputs[i]
            pk.name = graph_ops.layer_tensor_name("present_k", i)
            pv.name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(pk)
            network.mark_output(pv)

        if verbose:
            print(f"[trtmc build] Building dual-profile Qwen3-Omni Thinker MoE engine "
                  f"({num_layers} layers, hidden={hidden}, "
                  f"attn={attention_size}, experts={num_experts}, "
                  f"top_k={top_k}, inter={moe_intermediate}, "
                  f"cache={max_cache_length}) ...", file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed for Qwen3-Omni Thinker")

        return bytes(plan)

    def _build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        """Build Thinker vision encoder (reuses Qwen VL vision builder)."""
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            # Qwen3-Omni nests vision_config inside thinker_config
            thinker = config.raw.get("thinker_config", {})
            vision_config = thinker.get("vision_config")
        if vision_config is None:
            return None

        # Load vision weights from safetensors
        from .checkpoint_mapper import _open_safetensors, _load_tensor

        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        vision_weights = WeightDict()
        for reader in readers:
            for key in reader.keys():
                if key.startswith("model.thinker.visual."):
                    canon = key[len("model.thinker."):]
                    vision_weights[canon] = _load_tensor([reader], key)
                elif key.startswith("visual."):
                    vision_weights[key] = _load_tensor([reader], key)

        if not vision_weights:
            return None

        # Determine which builder to use based on vision_config
        deepstack_indexes = vision_config.get("deepstack_visual_indexes")
        fixed_image_size = 448

        if deepstack_indexes:
            from .qwen_vl_vision_builder import build_qwen3_vl_vision_engine
            return build_qwen3_vl_vision_engine(
                vision_config, vision_weights,
                fixed_image_size=fixed_image_size,
                verbose=verbose)
        else:
            from .qwen_vl_vision_builder import build_qwen_vl_vision_engine
            return build_qwen_vl_vision_engine(
                vision_config, vision_weights,
                fixed_image_size=fixed_image_size,
                verbose=verbose)

    def _build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        build_timing: dict | None = None,
    ) -> dict:
        """Build the audio encoder and official Code2Wav engines."""
        result = {}

        # Build audio encoder engine
        audio_cfg = weights.get("_audio_encoder_cfg", {})
        if audio_cfg and audio_cfg.get("num_layers", 0) > 0:
            if verbose:
                print("[trtmc build]   Building audio encoder engine ...", file=sys.stderr)
            with timed_trt_compile(build_timing, "extra_omni_audio_encoder"):
                audio_plan = _build_audio_encoder_engine(weights, audio_cfg, verbose=verbose)
            if audio_plan is not None:
                result["audio_encoder_plan"] = audio_plan

        # Build Code2Wav engine
        code2wav_cfg = weights.get("_code2wav_cfg", {})
        if code2wav_cfg.get("available"):
            if verbose:
                print("[trtmc build]   Building Code2Wav engine ...", file=sys.stderr)
            with timed_trt_compile(build_timing, "extra_omni_code2wav_decoder"):
                code2wav_plan = _build_code2wav_engine(weights, code2wav_cfg, verbose=verbose)
            if code2wav_plan is not None:
                result["code2wav_engine_plan"] = code2wav_plan

        return result

    def _get_vl_config(self, config: ModelConfig) -> dict | None:
        """Return VL config for vision-language support."""
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            # Qwen3-Omni nests vision_config inside thinker_config
            thinker = config.raw.get("thinker_config", {})
            vision_config = thinker.get("vision_config")
        if vision_config is None:
            return None

        patch_size = vision_config.get("patch_size", 14)
        merge_size = vision_config.get("spatial_merge_size", 2)
        fixed_image_size = 448

        grid_h = fixed_image_size // patch_size
        grid_w = fixed_image_size // patch_size
        num_patches = grid_h * grid_w
        num_merged = num_patches // (merge_size * merge_size)

        return {
            "image_token_id": 151655,
            "fixed_image_size": fixed_image_size,
            "num_image_pad_tokens": num_merged,
            "vision_output_dim": config.hidden_size,
            "preprocessor_type": "merge_group_chw",
            "vl_prompt_template": (
                "<|im_start|>user\n"
                "<|vision_start|>{image_pads}<|vision_end|>\n"
                "{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "image_token_str": "<|image_pad|>",
        }

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        """Return extra config fields for the bundle config.json.

        All fields must be flat (no nested dicts) since the C++ parser
        uses extract_json_int/string on the top-level config.json.

        IMPORTANT: The C++ fast_path_config parser does flat text search
        (``extract_json_int(text, "hidden_size", 0)``) which finds the
        FIRST occurrence of the key in the JSON text.  For Qwen3-Omni the
        original HF config.json has these critical fields nested inside
        ``thinker_config.text_config``, so they must be injected as
        top-level overrides and placed before any nested dicts in the
        serialized JSON (engine_builder.py handles this).
        """
        overrides = {}

        # Critical decoder dimensions that the C++ runtime needs at the
        # top level.  Without these the C++ parser gets 0 for hidden_size,
        # num_hidden_layers, etc. which causes a segfault when constructing
        # DeviceKvCache.
        overrides["hidden_size"] = config.hidden_size
        overrides["num_hidden_layers"] = config.num_hidden_layers
        overrides["num_attention_heads"] = config.num_attention_heads
        overrides["num_key_value_heads"] = config.num_key_value_heads
        overrides["head_dim"] = config.head_dim
        overrides["vocab_size"] = config.vocab_size
        overrides["intermediate_size"] = config.intermediate_size

        # Thinker MoE config
        overrides["num_local_experts"] = self._thinker_cfg.get(
            "num_experts", 8)
        overrides["num_experts_per_tok"] = self._thinker_cfg.get(
            "num_experts_per_tok", 2)
        return overrides


# ---------------------------------------------------------------------------
# MoE block for Omni (standard top-k softmax, same as Mixtral pattern)
# ---------------------------------------------------------------------------

def _add_routed_swiglu_experts(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    top_indices: trt.ITensor,
    routing_weights: trt.ITensor,
    hidden_size: int,
    top_k: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Compute only the top-k routed experts for each token."""
    inp_4d = network.add_shuffle(inp)
    inp_4d.reshape_dims = (-1, 1, 1, hidden_size)

    def packed_weight(values: np.ndarray) -> trt.ITensor:
        tensor = graph_ops.add_constant(
            network, values.shape, values, dtype=dtype)
        if tensor.dtype != inp.dtype:
            tensor = network.add_cast(tensor, inp.dtype).get_output(0)
        return tensor

    gate_weights = packed_weight(w_gate)
    up_weights = packed_weight(w_up)
    down_weights = packed_weight(w_down)
    selected_gate = network.add_gather(gate_weights, top_indices, 0)
    selected_up = network.add_gather(up_weights, top_indices, 0)
    gate = network.add_matrix_multiply(
        inp_4d.get_output(0), trt.MatrixOperation.NONE,
        selected_gate.get_output(0), trt.MatrixOperation.NONE)
    up = network.add_matrix_multiply(
        inp_4d.get_output(0), trt.MatrixOperation.NONE,
        selected_up.get_output(0), trt.MatrixOperation.NONE)

    sigmoid = network.add_activation(
        gate.get_output(0), trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate.get_output(0), sigmoid.get_output(0),
        trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up.get_output(0),
        trt.ElementWiseOperation.PROD)

    selected_down = network.add_gather(down_weights, top_indices, 0)
    down = network.add_matrix_multiply(
        gated.get_output(0), trt.MatrixOperation.NONE,
        selected_down.get_output(0), trt.MatrixOperation.NONE)
    output = network.add_shuffle(down.get_output(0))
    output.reshape_dims = (-1, top_k, hidden_size)

    route_weights = network.add_shuffle(routing_weights)
    route_weights.reshape_dims = (-1, top_k, 1)
    weighted = network.add_elementwise(
        output.get_output(0), route_weights.get_output(0),
        trt.ElementWiseOperation.PROD)
    return network.add_reduce(
        weighted.get_output(0), trt.ReduceOperation.SUM, 1 << 1,
        keep_dims=False).get_output(0)


def _add_omni_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_experts: int,
    top_k: int = 2,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add MoE block with standard top-k softmax routing (same as Mixtral)."""
    # Router logits
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts,
        weights[f"{prefix}.router"], dtype=dtype)

    # Softmax over router logits
    sm = network.add_softmax(router_logits)
    sm.axes = 1 << 1

    # TopK selection
    topk = network.add_topk(sm.get_output(0), trt.TopKOperation.MAX,
                            top_k, 1 << 1)
    top_values = topk.get_output(0)
    top_indices = topk.get_output(1)

    # Renormalize
    sum_val = network.add_reduce(
        top_values, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
    norm_weights = network.add_elementwise(
        top_values, sum_val.get_output(0),
        trt.ElementWiseOperation.DIV)

    # Gather each token's routed expert weights before the expert matmuls.
    return _add_routed_swiglu_experts(
        network, inp, top_indices, norm_weights.get_output(0), hidden_size, top_k,
        weights[f"{prefix}.experts.w_gate"],
        weights[f"{prefix}.experts.w_up"],
        weights[f"{prefix}.experts.w_down"],
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Audio encoder builder (Whisper-like)
# ---------------------------------------------------------------------------

def _build_audio_encoder_engine(
    weights: WeightDict,
    audio_cfg: dict,
    verbose: bool = False,
) -> bytes | None:
    """Build Whisper-like audio encoder: mel -> conv -> transformer -> features.

    Input: mel_features [num_mel_bins, max_audio_len] float32
    Output: audio_features [num_frames, embed_dim] float32
    """
    embed_dim = audio_cfg.get("embed_dim", 1280)
    num_mel = audio_cfg.get("num_mel_bins", 128)
    num_layers = audio_cfg.get("num_layers", 0)
    num_heads = audio_cfg.get("num_heads", 20)

    if num_layers == 0:
        return None

    max_audio_len = 3000  # 30s * 100 mel frames/s

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=np.float32))

    # Input: mel spectrogram [1, num_mel, max_audio_len]
    mel_input = network.add_input(
        "mel_features", trt.float32, (1, num_mel, max_audio_len))

    # Conv1: [1, num_mel, T] -> [1, embed_dim, T]
    conv1_w = weights.get("audio.audio_tower.conv1.weight")
    conv1_b = weights.get("audio.audio_tower.conv1.bias")
    if conv1_w is None:
        return None

    conv1_w_np = conv1_w.astype(np.float32)
    conv1 = network.add_convolution_nd(
        mel_input, embed_dim, (3,),
        trt.Weights(np.ascontiguousarray(conv1_w_np)),
        trt.Weights(conv1_b.astype(np.float32) if conv1_b is not None
                    else np.zeros(embed_dim, dtype=np.float32)))
    conv1.padding_nd = (1,)

    gelu1 = network.add_activation(
        conv1.get_output(0), trt.ActivationType.RELU)

    # Conv2: stride 2 downsampling
    conv2_w = weights.get("audio.audio_tower.conv2.weight")
    conv2_b = weights.get("audio.audio_tower.conv2.bias")
    if conv2_w is not None:
        conv2_w_np = conv2_w.astype(np.float32)
        conv2 = network.add_convolution_nd(
            gelu1.get_output(0), embed_dim, (3,),
            trt.Weights(np.ascontiguousarray(conv2_w_np)),
            trt.Weights(conv2_b.astype(np.float32) if conv2_b is not None
                        else np.zeros(embed_dim, dtype=np.float32)))
        conv2.stride_nd = (2,)
        conv2.padding_nd = (1,)
        gelu2 = network.add_activation(
            conv2.get_output(0), trt.ActivationType.RELU)
        hidden = gelu2.get_output(0)
    else:
        hidden = gelu1.get_output(0)

    # Transpose [1, embed_dim, T/2] -> [T/2, embed_dim] for transformer
    num_frames = max_audio_len // 2
    squeeze = network.add_shuffle(hidden)
    squeeze.reshape_dims = (embed_dim, num_frames)
    squeeze.second_transpose = trt.Permutation([1, 0])
    hidden = squeeze.get_output(0)

    # Position embedding
    pos_key = "audio.audio_tower.embed_positions.weight"
    if pos_key in weights:
        pos_w = weights[pos_key].astype(np.float32)
        # Truncate to num_frames if needed
        if pos_w.shape[0] >= num_frames:
            pos_const = graph_ops.add_constant(
                network, (num_frames, embed_dim),
                pos_w[:num_frames])
            hidden = network.add_elementwise(
                hidden, pos_const, trt.ElementWiseOperation.SUM).get_output(0)

    # Transformer encoder layers (simplified: LayerNorm -> Self-Attn -> Residual -> LayerNorm -> MLP -> Residual)
    for layer_idx in range(num_layers):
        lp = f"audio.audio_tower.layers.{layer_idx}"

        # Pre-attention LayerNorm
        ln1_w = weights.get(f"{lp}.self_attn_layer_norm.weight")
        ln1_b = weights.get(f"{lp}.self_attn_layer_norm.bias")
        if ln1_w is None:
            continue
        normed = graph_ops.add_layer_norm(
            network, hidden, embed_dim,
            ln1_w.astype(np.float32),
            ln1_b.astype(np.float32) if ln1_b is not None
                else np.zeros(embed_dim, dtype=np.float32),
            eps_tensor)

        # Self-attention Q/K/V projections
        q_w = weights.get(f"{lp}.self_attn.q_proj.weight")
        k_w = weights.get(f"{lp}.self_attn.k_proj.weight")
        v_w = weights.get(f"{lp}.self_attn.v_proj.weight")
        o_w = weights.get(f"{lp}.self_attn.out_proj.weight")

        if q_w is None:
            continue

        attn_out = graph_ops.add_self_attention_block_with_rope(
            network, normed,
            w_q=q_w.astype(np.float32).T.copy(),
            w_k=k_w.astype(np.float32).T.copy(),
            w_v=v_w.astype(np.float32).T.copy(),
            w_o=o_w.astype(np.float32).T.copy() if o_w is not None
                else np.zeros((embed_dim, embed_dim), dtype=np.float32),
            hidden_size=embed_dim,
            num_heads=num_heads,
            seq_length=num_frames,
            cos_table=np.ones((num_frames, embed_dim), dtype=np.float32),
            sin_table=np.zeros((num_frames, embed_dim), dtype=np.float32),
            q_bias=(weights.get(f"{lp}.self_attn.q_proj.bias") or
                    np.zeros(0, dtype=np.float32)).astype(np.float32)
                if weights.get(f"{lp}.self_attn.q_proj.bias") is not None else None,
            k_bias=(weights.get(f"{lp}.self_attn.k_proj.bias") or
                    np.zeros(0, dtype=np.float32)).astype(np.float32)
                if weights.get(f"{lp}.self_attn.k_proj.bias") is not None else None,
            v_bias=(weights.get(f"{lp}.self_attn.v_proj.bias") or
                    np.zeros(0, dtype=np.float32)).astype(np.float32)
                if weights.get(f"{lp}.self_attn.v_proj.bias") is not None else None,
            o_bias=(weights.get(f"{lp}.self_attn.out_proj.bias") or
                    np.zeros(0, dtype=np.float32)).astype(np.float32)
                if weights.get(f"{lp}.self_attn.out_proj.bias") is not None else None,
        )

        # Residual
        hidden = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(0)

        # Post-attention LayerNorm
        ln2_w = weights.get(f"{lp}.final_layer_norm.weight")
        ln2_b = weights.get(f"{lp}.final_layer_norm.bias")
        normed2 = graph_ops.add_layer_norm(
            network, hidden, embed_dim,
            ln2_w.astype(np.float32) if ln2_w is not None
                else np.ones(embed_dim, dtype=np.float32),
            ln2_b.astype(np.float32) if ln2_b is not None
                else np.zeros(embed_dim, dtype=np.float32),
            eps_tensor)

        # MLP: fc1 -> GELU -> fc2
        fc1_w = weights.get(f"{lp}.fc1.weight")
        fc1_b = weights.get(f"{lp}.fc1.bias")
        fc2_w = weights.get(f"{lp}.fc2.weight")
        fc2_b = weights.get(f"{lp}.fc2.bias")

        if fc1_w is not None and fc2_w is not None:
            mlp_hidden = fc1_w.shape[0]
            fc1 = graph_ops.add_matmul_rhs_constant(
                network, normed2, embed_dim, mlp_hidden,
                fc1_w.astype(np.float32).T.copy())
            if fc1_b is not None:
                fc1 = graph_ops.add_bias_sum(
                    network, fc1, mlp_hidden, fc1_b.astype(np.float32))
            activated = graph_ops.add_gelu_new(network, fc1)
            fc2 = graph_ops.add_matmul_rhs_constant(
                network, activated, mlp_hidden, embed_dim,
                fc2_w.astype(np.float32).T.copy())
            if fc2_b is not None:
                fc2 = graph_ops.add_bias_sum(
                    network, fc2, embed_dim, fc2_b.astype(np.float32))

            # Residual
            hidden = network.add_elementwise(
                hidden, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    # Output
    hidden.name = "audio_features"
    network.mark_output(hidden)

    if verbose:
        print(f"[trtmc build] Building Qwen3-Omni audio encoder "
              f"({num_layers} layers, embed={embed_dim}, "
              f"mel={num_mel}, frames={num_frames}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT audio encoder build failed")
    return bytes(plan)


# ---------------------------------------------------------------------------
# Code2Wav engine builder (ConvNet waveform synthesizer)
# ---------------------------------------------------------------------------


def _build_code2wav_engine(
    weights: WeightDict,
    code2wav_cfg: dict,
    verbose: bool = False,
) -> bytes | None:
    """Build the official Code2Wav graph: RVQ codes -> speech waveform.

    The released checkpoint uses an eight-layer sliding-attention
    pre-transformer, two ConvNeXt upsamplers, and four causal HiFi-GAN-style
    decoder blocks. The old placeholder builder looked for flat
    ``upsample_blocks.N.weight`` tensors that do not exist in this checkpoint,
    so it silently omitted the engine. Exporting the upstream model-owned
    module to ONNX preserves all 230 checkpoint tensors and lets TensorRT fuse
    the complete static audio decoder.
    """
    if not code2wav_cfg.get("available"):
        return None

    import gc
    import torch
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeCode2WavConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    model_config = Qwen3OmniMoeCode2WavConfig(**code2wav_cfg["config"])
    model = Qwen3OmniMoeCode2Wav(model_config)

    state = {}
    for name in model.state_dict():
        key = f"code2wav.{name}"
        value = weights.get(key)
        if value is None:
            raise RuntimeError(f"Qwen3-Omni Code2Wav checkpoint tensor is missing: {key}")
        state[name] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
    model.load_state_dict(state, strict=True)
    model.eval()

    class _StaticCode2Wav(torch.nn.Module):
        """Static, export-safe equivalent of the official forward method."""

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, codes):
            codes = codes.to(torch.int64)
            hidden = self.module.code_embedding(codes + self.module.code_offset).mean(1)

            # Supplying the official 72-frame sliding causal mask explicitly
            # avoids the torch.diff-based dynamic mask helper, which is not
            # representable in ONNX opset 17. The engine shape is static.
            length = hidden.shape[1]
            indices = torch.arange(length, device=hidden.device)
            row = indices[:, None]
            col = indices[None, :]
            allowed = (col <= row) & (col > row - self.module.config.sliding_window)
            mask = torch.where(
                allowed,
                torch.zeros((), dtype=hidden.dtype, device=hidden.device),
                torch.full(
                    (),
                    torch.finfo(hidden.dtype).min,
                    dtype=hidden.dtype,
                    device=hidden.device,
                ),
            )[None, None]
            hidden = self.module.pre_transformer(
                inputs_embeds=hidden,
                attention_mask={
                    "sliding_attention": mask,
                    "full_attention": mask,
                },
            ).last_hidden_state
            hidden = hidden.permute(0, 2, 1)
            for blocks in self.module.upsample:
                for block in blocks:
                    hidden = block(hidden)
            for block in self.module.decoder:
                hidden = block(hidden)
            return hidden.clamp(min=-1, max=1)

    max_frames = int(code2wav_cfg["max_frames"])
    num_quantizers = int(model_config.num_quantizers)
    export_module = _StaticCode2Wav(model).eval()
    dummy_codes = torch.zeros((1, num_quantizers, max_frames), dtype=torch.int32)
    onnx_buffer = io.BytesIO()
    if verbose:
        print(
            "[trtmc build]   Exporting official Qwen3-Omni Code2Wav "
            f"({num_quantizers} codebooks x {max_frames} frames) ...",
            file=sys.stderr,
        )
    with torch.inference_mode():
        torch.onnx.export(
            export_module,
            dummy_codes,
            onnx_buffer,
            opset_version=17,
            input_names=["codec_tokens"],
            output_names=["waveform"],
            dynamo=False,
        )

    del export_module, model, state, dummy_codes
    gc.collect()

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(
            explicit_batch=True,
            strongly_typed=True,
        )
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_buffer.getvalue()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Qwen3-Omni Code2Wav ONNX parsing failed:\n" + "\n".join(errors))

    expected_samples = max_frames * int(code2wav_cfg["upsample_factor"]) - int(
        code2wav_cfg["output_delay"]
    )
    if tuple(network.get_input(0).shape) != (1, num_quantizers, max_frames) or tuple(
        network.get_output(0).shape
    ) != (1, 1, expected_samples):
        raise RuntimeError(
            "Qwen3-Omni Code2Wav ONNX shape contract mismatch: "
            f"input={tuple(network.get_input(0).shape)}, "
            f"output={tuple(network.get_output(0).shape)}"
        )

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if verbose:
        print(
            "[trtmc build]   Building complete Qwen3-Omni Code2Wav "
            f"TensorRT engine ({expected_samples} samples) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT Qwen3-Omni Code2Wav build failed")
    return bytes(plan)


plugin = Qwen3OmniPlugin()
