"""Qwen3-Omni family plugin -- multimodal Thinker-Talker-Code2Wav pipeline.

Qwen3-Omni is a 3-stage multimodal model:
  1. Thinker: Multimodal MoE decoder (text + image + audio input -> text output)
     - Vision encoder (reuses Qwen VL pattern with 3D RoPE)
     - Audio encoder (Whisper-like mel -> transformer encoder)
     - MoE text decoder (Qwen3 MoE architecture)
  2. Talker: Text embeddings -> RVQ speech codec tokens
     - Takes Thinker hidden states + text tokens
     - Produces 8-codebook RVQ tokens (codebook-by-codebook autoregressive)
  3. Code2Wav: Codec tokens -> audio waveform
     - ConvNet-based waveform synthesizer (transposed convolutions)

The Thinker MoE decoder follows Qwen3 MoE (sibling model) with the same
top-k softmax routing. Vision/audio features inject via embed_input mode
during prefill.

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

  Talker:
    model.talker.layers.{i}.*
    model.talker.codec_head.{cb}.weight

  Code2Wav:
    model.code2wav.upsample_blocks.{i}.weight/bias
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
from ...build_timing import timed_trt_compile
from . import graph_ops
from . import graph_blocks
from .standard_decoder_builder import _mark_debug_output


trt = trt_compat.get_trt()

class Qwen3OmniPlugin:
    name = "qwen3_omni"
    runtime_strategy = "qwen3_omni_multimodal"
    embed_input = True

    def __init__(self):
        self._thinker_cfg: dict = {}
        self._talker_cfg: dict = {}
        self._audio_encoder_cfg: dict = {}
        self._code2wav_cfg: dict = {}

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in ("qwen3_omni", "qwen3_omni_moe", "qwen3omni")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load Qwen3-Omni weights from safetensors.

        Loads:
          - Thinker MoE decoder weights (model.thinker.layers.*)
          - Thinker audio encoder weights (model.thinker.audio_tower.*)
          - Talker decoder weights (model.talker.*)
          - Code2Wav weights (model.code2wav.*)
        """
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

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
        weights["embedding"] = embedding.astype(np.float32)

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

            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            o_t = _transpose_2d(o_raw, "o_proj")
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
                    router_raw, "router")
                del router_raw

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

                    weights[f"{prefix}.expert.{e}.w_gate"] = _transpose_2d(
                        gate_raw, f"expert_{e}_gate")
                    weights[f"{prefix}.expert.{e}.w_up"] = _transpose_2d(
                        up_raw, f"expert_{e}_up")
                    weights[f"{prefix}.expert.{e}.w_down"] = _transpose_2d(
                        down_raw, f"expert_{e}_down")
                    del gate_raw, up_raw, down_raw
            else:
                # Dense SwiGLU MLP (some layers may not be MoE)
                gate_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.gate_proj.weight")
                up_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.up_proj.weight")
                down_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.down_proj.weight")

                weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate")
                weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up")
                weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down")

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
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied")

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

        # Detect audio encoder config
        audio_cfg = self._detect_audio_encoder(readers)
        self._audio_encoder_cfg = audio_cfg
        weights["_audio_encoder_cfg"] = audio_cfg

        # Detect Talker config
        talker_cfg = self._detect_talker(readers, config)
        self._talker_cfg = talker_cfg
        weights["_talker_cfg"] = talker_cfg

        # Detect Code2Wav config
        code2wav_cfg = self._detect_code2wav(readers)
        self._code2wav_cfg = code2wav_cfg
        weights["_code2wav_cfg"] = code2wav_cfg

        # Load audio encoder, talker, and code2wav weights.
        # Handle both prefixes: thinker.audio_tower.* and
        # model.thinker.audio_tower.* (similarly for talker/code2wav).
        for reader in readers:
            for key in reader.keys():
                # Audio tower weights
                if key.startswith("thinker.audio_tower."):
                    canon = key[len("thinker."):]
                    weights[f"audio.{canon}"] = _load_tensor([reader], key)
                elif key.startswith("model.thinker.audio_tower."):
                    canon = key[len("model.thinker."):]
                    weights[f"audio.{canon}"] = _load_tensor([reader], key)
                # Talker weights
                elif key.startswith("talker."):
                    weights[f"talker.{key[len('talker.'):]}"] = (
                        _load_tensor([reader], key))
                elif key.startswith("model.talker."):
                    weights[f"talker.{key[len('model.talker.'):]}"] = (
                        _load_tensor([reader], key))
                # Code2Wav weights
                elif key.startswith("code2wav."):
                    weights[f"code2wav.{key[len('code2wav.'):]}"] = (
                        _load_tensor([reader], key))
                elif key.startswith("model.code2wav."):
                    weights[f"code2wav.{key[len('model.code2wav.'):]}"] = (
                        _load_tensor([reader], key))
                # Vision encoder weights
                elif key.startswith("thinker.visual."):
                    canon = key[len("thinker."):]
                    weights[f"vision.{canon}"] = _load_tensor([reader], key)
                elif key.startswith("model.thinker.visual."):
                    canon = key[len("model.thinker."):]
                    weights[f"vision.{canon}"] = _load_tensor([reader], key)

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

    def _detect_code2wav(self, readers) -> dict:
        """Detect Code2Wav dimensions from weight shapes."""
        # Look for upsample blocks (try both prefixes)
        upsample_base = "code2wav.upsample"
        if not _has_tensor(readers, f"{upsample_base}.0.weight"):
            upsample_base = "code2wav.upsample_blocks"
        if not _has_tensor(readers, f"{upsample_base}.0.weight"):
            upsample_base = "model.code2wav.upsample_blocks"
        n_upsample = 0
        while _has_tensor(
            readers, f"{upsample_base}.{n_upsample}.weight"
        ):
            n_upsample += 1

        # Look for codebook embedding (try both prefixes)
        embed_key = "code2wav.code_embedding.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "code2wav.codebook_embed.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "model.code2wav.codebook_embed.weight"
        if _has_tensor(readers, embed_key):
            embed_w = _load_tensor(readers, embed_key)
            codebook_vocab, embed_dim = embed_w.shape
        else:
            codebook_vocab, embed_dim = 0, 0

        return {
            "n_upsample_blocks": int(n_upsample),
            "codebook_vocab": int(codebook_vocab),
            "embed_dim": int(embed_dim),
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
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # Inputs
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))

        # VL embed_input (for vision/audio feature injection)
        input_embed_tensor = network.add_input(
            "input_embed", trt.float32, (1, hidden))
        use_input_embed_tensor = network.add_input(
            "use_input_embed", trt.float32, (1,))

        # KV cache inputs
        cache_k_inputs = []
        cache_v_inputs = []
        for i in range(num_layers):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                trt.float32, (max_cache_length, kv_attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                trt.float32, (max_cache_length, kv_attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        # Shared constants
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"])

        graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
        cos_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, True)
        sin_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, False)
        cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np)

        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=np.float32))

        # Embedding lookup with input_embed override for VL/audio
        gather = network.add_gather(embedding_table, token_id, 0)
        token_embed = gather.get_output(0)

        # Conditional: (1 - flag) * token_embed + flag * input_embed
        flag_broadcast = network.add_shuffle(use_input_embed_tensor)
        flag_broadcast.reshape_dims = (1, 1)
        one_const = graph_ops.add_constant(
            network, (1, 1), np.array([1.0], dtype=np.float32))
        inv_flag = network.add_elementwise(
            one_const, flag_broadcast.get_output(0),
            trt.ElementWiseOperation.SUB)
        tok_part = network.add_elementwise(
            inv_flag.get_output(0), token_embed,
            trt.ElementWiseOperation.PROD)
        embed_part = network.add_elementwise(
            flag_broadcast.get_output(0), input_embed_tensor,
            trt.ElementWiseOperation.PROD)
        hidden_sum = network.add_elementwise(
            tok_part.get_output(0), embed_part.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = hidden_sum.get_output(0)

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
                eps_tensor, "rmsnorm")

            # Check if this is an MoE or dense layer
            if f"{prefix}.router" in weights:
                # MoE block
                moe_out = _add_omni_moe_block(
                    network, norm2, weights, prefix,
                    hidden, num_experts, moe_intermediate, top_k)
            else:
                # Dense SwiGLU MLP
                moe_out = graph_blocks.add_swiglu_mlp(
                    network, norm2, weights=weights, prefix=prefix,
                    hidden_size=hidden,
                    mlp_size=weights[f"{prefix}.w_gate"].shape[1])

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
                eps_tensor, "rmsnorm")

        hidden_out = network.add_identity(hidden_state).get_output(0)
        hidden_out.name = "hidden_state"
        network.mark_output(hidden_out)

        # LM head
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_out"])
        b_out = np.zeros(vocab, dtype=np.float32)
        logits = graph_ops.add_bias_sum(network, logits, vocab, b_out)

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
            print(f"[trtmc build] Building Qwen3-Omni Thinker MoE engine "
                  f"({num_layers} layers, hidden={hidden}, "
                  f"attn={attention_size}, experts={num_experts}, "
                  f"top_k={top_k}, inter={moe_intermediate}, "
                  f"cache={max_cache_length}) ...", file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed for Qwen3-Omni Thinker")

        return bytes(plan)

    def build_vision_engine(
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

    def build_extra_engines(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        verbose: bool = False,
        build_timing: dict | None = None,
    ) -> dict:
        """Build audio encoder, Talker, and Code2Wav engines."""
        result = {}

        # Build audio encoder engine
        audio_cfg = weights.get("_audio_encoder_cfg", {})
        if audio_cfg and audio_cfg.get("num_layers", 0) > 0:
            if verbose:
                print("[trtmc build]   Building audio encoder engine ...",
                      file=sys.stderr)
            with timed_trt_compile(build_timing, "extra_omni_audio_encoder"):
                audio_plan = _build_audio_encoder_engine(
                    weights, audio_cfg, verbose=verbose)
            if audio_plan is not None:
                result["audio_encoder_plan"] = audio_plan

        # Build Talker engine
        talker_cfg = weights.get("_talker_cfg", {})
        if talker_cfg and talker_cfg.get("num_layers", 0) > 0:
            if verbose:
                print("[trtmc build]   Building Talker engine ...",
                      file=sys.stderr)
            with timed_trt_compile(build_timing, "extra_omni_talker_decoder"):
                talker_plan = _build_talker_engine(
                    weights, talker_cfg, config, max_cache_length,
                    verbose=verbose)
            if talker_plan is not None:
                result["talker_engine_plan"] = talker_plan

        # Build Code2Wav engine
        code2wav_cfg = weights.get("_code2wav_cfg", {})
        if code2wav_cfg and code2wav_cfg.get("n_upsample_blocks", 0) > 0:
            if verbose:
                print("[trtmc build]   Building Code2Wav engine ...",
                      file=sys.stderr)
            with timed_trt_compile(build_timing, "extra_omni_code2wav_decoder"):
                code2wav_plan = _build_code2wav_engine(
                    weights, code2wav_cfg, verbose=verbose)
            if code2wav_plan is not None:
                result["code2wav_engine_plan"] = code2wav_plan

        return result

    def get_vl_config(self, config: ModelConfig) -> dict | None:
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

        # Audio output config (flat keys matching C++ fast_path_config parser)
        overrides["audio_sample_rate"] = 24000
        overrides["omni_n_codebooks"] = self._talker_cfg.get(
            "n_codebooks", 8)
        overrides["omni_codebook_size"] = self._talker_cfg.get(
            "codebook_size", 2048)

        # Talker config (flat keys for C++ parser)
        overrides["omni_talker_hidden_size"] = self._talker_cfg.get(
            "hidden_size", 0)
        overrides["omni_talker_num_layers"] = self._talker_cfg.get(
            "num_layers", 0)
        overrides["omni_talker_max_cache_length"] = 1024

        # Audio encoder config (flat keys for C++ parser)
        overrides["omni_audio_embed_dim"] = self._audio_encoder_cfg.get(
            "embed_dim", 1280)
        overrides["omni_audio_num_mel"] = self._audio_encoder_cfg.get(
            "num_mel_bins", 128)
        overrides["omni_audio_num_layers"] = self._audio_encoder_cfg.get(
            "num_layers", 0)
        overrides["omni_audio_num_frames"] = 1500

        return overrides


# ---------------------------------------------------------------------------
# MoE block for Omni (standard top-k softmax, same as Mixtral pattern)
# ---------------------------------------------------------------------------

def _add_swiglu_expert(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> trt.ITensor:
    """Compute a single SwiGLU expert: down(silu(gate(x)) * up(x))."""
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate)
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up)

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    down = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size, w_down)
    return down


def _add_omni_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_experts: int,
    moe_intermediate: int,
    top_k: int = 2,
) -> trt.ITensor:
    """Add MoE block with standard top-k softmax routing (same as Mixtral)."""
    # Router logits
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts,
        weights[f"{prefix}.router"])

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

    # Compute all expert outputs and stack
    expert_outputs = []
    for e in range(num_experts):
        exp_out = _add_swiglu_expert(
            network, inp, hidden_size, moe_intermediate,
            weights[f"{prefix}.expert.{e}.w_gate"],
            weights[f"{prefix}.expert.{e}.w_up"],
            weights[f"{prefix}.expert.{e}.w_down"],
        )
        expert_outputs.append(exp_out)

    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)

    # Gather and scale each selected expert, then sum
    result = None
    for k in range(top_k):
        idx_slice = network.add_slice(
            top_indices, start=(0, k), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)

        w_slice = network.add_slice(
            norm_weights.get_output(0),
            start=(0, k), shape=(1, 1), stride=(1, 1))

        expert_out = network.add_gather(
            stacked_out, idx_flat.get_output(0), 0)

        scaled_expert = network.add_elementwise(
            expert_out.get_output(0), w_slice.get_output(0),
            trt.ElementWiseOperation.PROD)

        if result is None:
            result = scaled_expert.get_output(0)
        else:
            sum_layer = network.add_elementwise(
                result, scaled_expert.get_output(0),
                trt.ElementWiseOperation.SUM)
            result = sum_layer.get_output(0)

    return result


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
# Talker engine builder (RVQ codec predictor)
# ---------------------------------------------------------------------------

def _build_talker_engine(
    weights: WeightDict,
    talker_cfg: dict,
    config: ModelConfig,
    max_cache_length: int,
    verbose: bool = False,
) -> bytes | None:
    """Build Talker decoder engine.

    The Talker takes Thinker hidden states and text tokens as input and
    produces RVQ codec tokens. It uses a standard KV-cache decoder architecture
    with codebook prediction heads.

    Input: token_id [1] int32, standard KV cache inputs
    Output: codec_logits [1, codebook_size] float32
    """
    talker_hidden = talker_cfg.get("hidden_size", 0)
    talker_vocab = talker_cfg.get("vocab_size", 0)
    num_layers = talker_cfg.get("num_layers", 0)
    n_codebooks = talker_cfg.get("n_codebooks", 8)
    codebook_size = talker_cfg.get("codebook_size", 2048)

    if num_layers == 0 or talker_hidden == 0:
        return None

    if verbose:
        print(f"[trtmc build]   Talker projection: layers={num_layers}, "
              f"hidden={talker_hidden}, vocab={talker_vocab}, "
              f"codebooks={n_codebooks}, cb_size={codebook_size}",
              file=sys.stderr)

    input_hidden = talker_cfg.get("hidden_size", talker_hidden)
    decoder_hidden = talker_cfg.get("decoder_hidden_size", talker_hidden)
    if (
        "talker.hidden_projection.linear_fc1.weight" not in weights
        or "talker.hidden_projection.linear_fc2.weight" not in weights
        or "talker.codec_head.weight" not in weights
    ):
        return None

    lm_head_keys = [
        f"talker.code_predictor.lm_head.{i}.weight"
        for i in range(max(n_codebooks - 1, 0))
    ]
    if any(k not in weights for k in lm_head_keys):
        return None

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    input_embed = network.add_input(
        "input_embed", trt.float32, (input_hidden,))
    hidden_in = network.add_shuffle(input_embed)
    hidden_in.reshape_dims = (1, input_hidden)
    hidden = hidden_in.get_output(0)

    fc1_w = _transpose_2d(
        weights["talker.hidden_projection.linear_fc1.weight"], "talker_fc1")
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, hidden, input_hidden, input_hidden, fc1_w)
    fc1_b = weights.get("talker.hidden_projection.linear_fc1.bias")
    if fc1_b is not None:
        fc1 = graph_ops.add_bias_sum(
            network, fc1, input_hidden, fc1_b.astype(np.float32))

    sigmoid = network.add_activation(fc1, trt.ActivationType.SIGMOID)
    silu = network.add_elementwise(
        fc1, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)

    fc2_w = _transpose_2d(
        weights["talker.hidden_projection.linear_fc2.weight"], "talker_fc2")
    hidden = graph_ops.add_matmul_rhs_constant(
        network, silu.get_output(0), input_hidden, decoder_hidden, fc2_w)
    fc2_b = weights.get("talker.hidden_projection.linear_fc2.bias")
    if fc2_b is not None:
        hidden = graph_ops.add_bias_sum(
            network, hidden, decoder_hidden, fc2_b.astype(np.float32))

    codec_head = weights["talker.codec_head.weight"]
    head_parts = []
    first_head_rows = min(codebook_size, codec_head.shape[0])
    if first_head_rows < codebook_size:
        return None
    first_head = _transpose_2d(
        codec_head[:codebook_size], "talker_codec_head")
    head_parts.append(graph_ops.add_matmul_rhs_constant(
        network, hidden, decoder_hidden, codebook_size, first_head))

    for i, key in enumerate(lm_head_keys):
        head = _transpose_2d(weights[key], f"talker_lm_head_{i}")
        head_parts.append(graph_ops.add_matmul_rhs_constant(
            network, hidden, decoder_hidden, codebook_size, head))

    if len(head_parts) == 1:
        logits = head_parts[0]
    else:
        concat = network.add_concatenation(head_parts)
        concat.axis = 1
        logits = concat.get_output(0)

    logits.name = "logits"
    network.mark_output(logits)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Qwen3-Omni Talker projection build failed")
    return bytes(plan)


# ---------------------------------------------------------------------------
# Code2Wav engine builder (ConvNet waveform synthesizer)
# ---------------------------------------------------------------------------

def _build_code2wav_engine(
    weights: WeightDict,
    code2wav_cfg: dict,
    verbose: bool = False,
) -> bytes | None:
    """Build Code2Wav engine: RVQ codec tokens -> audio waveform.

    Similar to EnCodec decoder: embedding lookup -> transposed convolutions.
    """
    n_upsample = code2wav_cfg.get("n_upsample_blocks", 0)
    codebook_vocab = code2wav_cfg.get("codebook_vocab", 0)
    embed_dim = code2wav_cfg.get("embed_dim", 0)

    if n_upsample == 0 or codebook_vocab == 0:
        return None

    # Max frames for the engine
    max_frames = 256

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # Input: codec tokens [n_codebooks, max_frames] int32
    n_codebooks = 8
    codec_input = network.add_input(
        "codec_tokens", trt.int32, (n_codebooks, max_frames))

    # Codebook embedding lookup
    embed_key = "code2wav.codebook_embed.weight"
    if embed_key not in weights:
        return None

    embed_w = weights[embed_key].astype(np.float32)
    embed_table = graph_ops.add_constant(
        network, (codebook_vocab, embed_dim), embed_w)

    # For now, just lookup first codebook and sum all
    # This is a simplified version — full implementation would handle
    # all codebooks with separate embedding tables
    # Flatten tokens, gather, reshape, sum across codebooks
    flat_tokens = network.add_shuffle(codec_input)
    flat_tokens.reshape_dims = (n_codebooks * max_frames,)

    gathered = network.add_gather(embed_table, flat_tokens.get_output(0), 0)
    reshaped = network.add_shuffle(gathered.get_output(0))
    reshaped.reshape_dims = (n_codebooks, max_frames, embed_dim)

    # Sum across codebooks
    summed = network.add_reduce(
        reshaped.get_output(0), trt.ReduceOperation.SUM, 1 << 0,
        keep_dims=False)
    hidden = summed.get_output(0)  # [max_frames, embed_dim]

    # Transpose to [1, embed_dim, max_frames] for convolutions
    transpose = network.add_shuffle(hidden)
    transpose.reshape_dims = (1, max_frames, embed_dim)
    transpose.second_transpose = trt.Permutation([0, 2, 1])
    hidden = transpose.get_output(0)

    # Upsampling transposed convolutions
    total_upsample = 1
    for i in range(n_upsample):
        up_w_key = f"code2wav.upsample_blocks.{i}.weight"
        up_b_key = f"code2wav.upsample_blocks.{i}.bias"
        up_w = weights.get(up_w_key)
        if up_w is None:
            break

        up_w_np = up_w.astype(np.float32)
        out_channels = up_w_np.shape[1]  # transposed conv: [in, out, K]
        kernel_size = up_w_np.shape[2] if up_w_np.ndim == 3 else 4
        stride = kernel_size // 2 if kernel_size > 1 else 1

        deconv = network.add_deconvolution_nd(
            hidden, out_channels, (kernel_size,),
            trt.Weights(np.ascontiguousarray(up_w_np)),
            trt.Weights(weights[up_b_key].astype(np.float32)
                        if up_b_key in weights
                        else np.zeros(out_channels, dtype=np.float32)))
        deconv.stride_nd = (stride,)
        deconv.padding_nd = ((kernel_size - stride) // 2,)

        relu = network.add_activation(
            deconv.get_output(0), trt.ActivationType.RELU)
        hidden = relu.get_output(0)
        total_upsample *= stride

    # Output: waveform [1, 1, num_samples]
    hidden.name = "waveform"
    network.mark_output(hidden)

    if verbose:
        print(f"[trtmc build] Building Qwen3-Omni Code2Wav engine "
              f"({n_upsample} upsample blocks, embed={embed_dim}, "
              f"max_frames={max_frames}, upsample={total_upsample}x) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Code2Wav build failed")
    return bytes(plan)


plugin = Qwen3OmniPlugin()
