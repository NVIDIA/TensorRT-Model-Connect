# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bark family plugin -- text-to-audio generation.

Bark is a multi-stage text-to-audio model:
  1. Semantic: text tokens -> semantic tokens (GPT decoder with learned positions)
  2. Coarse: semantic tokens -> coarse audio codes (GPT decoder)
  3. Fine: coarse codes -> fine audio codes (iterative refinement, no KV cache)
  4. Codec (EnCodec): audio codes -> waveform

HF Bark uses `layers.{i}.attn.att_proj.weight` in [3H, H] format (fused QKV).
This is NOT GPT-2 Conv1D -- the weights are already in standard linear format.
att_proj splits into Q, K, V each of dim H.

Weight key mapping:
  HF: semantic/coarse/fine.model.transformer.wte.weight
  HF: semantic/coarse/fine.model.transformer.wpe.weight
  HF: semantic/coarse/fine.model.transformer.h.{i}.layernorm_1/2.weight/bias
  HF: semantic/coarse/fine.model.transformer.h.{i}.attn.att_proj.weight
  HF: semantic/coarse/fine.model.transformer.h.{i}.attn.out_proj.weight/bias
  HF: semantic/coarse/fine.model.transformer.h.{i}.mlp.in_proj.weight/bias
  HF: semantic/coarse/fine.model.transformer.h.{i}.mlp.out_proj.weight/bias
  HF: semantic/coarse/fine.model.lm_head.weight
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import WeightDict
from ...build_timing import timed_trt_compile
from . import graph_ops
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .decoder_tp_builder import build_bark_tp_decoder_engine
from .timing_cache import build_bark_serialized_network


trt = trt_compat.get_trt()

def _load_bark_state_dict(model_dir: str) -> dict:
    """Load Bark state dict from pytorch_model.bin."""
    import torch
    model_path = Path(model_dir) / "pytorch_model.bin"
    if not model_path.exists():
        # Try safetensors
        from .checkpoint_mapper import _open_safetensors
        readers = _open_safetensors(Path(model_dir))
        state_dict = {}
        for reader in readers:
            for key in reader.keys():
                state_dict[key] = reader.get_tensor(key).numpy()
        return state_dict
    return torch.load(str(model_path), map_location="cpu", weights_only=True)


def _detect_sub_model_config(state_dict: dict, prefix: str) -> dict:
    """Auto-detect dimensions from state dict for a sub-model.

    HF Bark uses keys like:
      {prefix}.input_embeds_layer.weight  [vocab, hidden]
      {prefix}.position_embeds_layer.weight  [max_pos, hidden]
      {prefix}.layers.{i}.layernorm_1.weight
      {prefix}.layers.{i}.attn.att_proj.weight  [3*H, H]
      {prefix}.lm_head.weight  [output_vocab, hidden]  (semantic/coarse only)
    Fine model uses:
      {prefix}.input_embeds_layers.{i}.weight  [vocab, hidden]
      {prefix}.lm_heads.{i}.weight  [output_vocab, hidden]
    """
    # Find embedding weight to get vocab_size, hidden_size
    wte_key = f"{prefix}.input_embeds_layer.weight"
    if wte_key not in state_dict:
        # Fine model has multiple embedding tables
        wte_key = f"{prefix}.input_embeds_layers.0.weight"
    if wte_key not in state_dict:
        return {}
    wte = state_dict[wte_key]
    if hasattr(wte, 'numpy'):
        wte = wte.numpy()
    vocab_size, hidden_size = wte.shape

    # Count layers
    num_layers = 0
    while f"{prefix}.layers.{num_layers}.layernorm_1.weight" in state_dict:
        num_layers += 1

    # Detect num_heads from att_proj shape
    num_heads = hidden_size // 64  # Default guess
    att_key = f"{prefix}.layers.0.attn.att_proj.weight"
    if att_key in state_dict:
        for hd in [64, 128, 96]:
            if hidden_size % hd == 0:
                num_heads = hidden_size // hd
                break

    # Detect position embedding max length
    wpe_key = f"{prefix}.position_embeds_layer.weight"
    max_position = 1024
    if wpe_key in state_dict:
        wpe = state_dict[wpe_key]
        if hasattr(wpe, 'numpy'):
            wpe = wpe.numpy()
        max_position = wpe.shape[0]

    # Output vocab (from lm_head)
    lm_key = f"{prefix}.lm_head.weight"
    output_vocab = vocab_size
    if lm_key in state_dict:
        lm_w = state_dict[lm_key]
        if hasattr(lm_w, 'numpy'):
            lm_w = lm_w.numpy()
        output_vocab = lm_w.shape[0]

    # Count embedding tables (fine model has multiple: input_embeds_layers.0..N)
    n_embed_tables = 0
    while f"{prefix}.input_embeds_layers.{n_embed_tables}.weight" in state_dict:
        n_embed_tables += 1
    if n_embed_tables == 0:
        # Semantic/coarse: single embedding table
        n_embed_tables = 1

    # Count LM heads (fine model has multiple: lm_heads.0..N)
    n_lm_heads = 0
    while f"{prefix}.lm_heads.{n_lm_heads}.weight" in state_dict:
        n_lm_heads += 1
    if n_lm_heads == 0:
        n_lm_heads = 1  # Semantic/coarse: single lm_head

    # Detect codebook_size from first LM head (fine model)
    codebook_size = output_vocab
    lm0_key = f"{prefix}.lm_heads.0.weight"
    if lm0_key in state_dict:
        lm0_w = state_dict[lm0_key]
        if hasattr(lm0_w, 'numpy'):
            lm0_w = lm0_w.numpy()
        codebook_size = int(lm0_w.shape[0])

    return {
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "max_position": int(max_position),
        "output_vocab": int(output_vocab),
        "intermediate_size": int(hidden_size * 4),
        "n_embed_tables": int(n_embed_tables),
        "n_lm_heads": int(n_lm_heads),
        "codebook_size": int(codebook_size),
    }


def _map_bark_decoder_weights(
    state_dict: dict,
    prefix: str,
    sub_config: dict,
) -> WeightDict:
    """Map HF Bark decoder weights to standard decoder format.

    HF Bark key patterns:
      {prefix}.input_embeds_layer.weight  [vocab, hidden]
      {prefix}.position_embeds_layer.weight  [max_pos, hidden]
      {prefix}.layers.{i}.layernorm_1.weight  [hidden]  (no bias in bark-small)
      {prefix}.layers.{i}.attn.att_proj.weight  [3*H, H]  (fused QKV)
      {prefix}.layers.{i}.attn.out_proj.weight  [H, H]
      {prefix}.layers.{i}.mlp.in_proj.weight  [4*H, H]
      {prefix}.layers.{i}.mlp.out_proj.weight  [H, 4*H]
      {prefix}.layernorm_final.weight  [hidden]
      {prefix}.lm_head.weight  [output_vocab, hidden]

    Standard decoder builder expects:
      embedding [vocab, hidden], position_embedding [max_pos, hidden]
      layer.{i}.input_norm [hidden], layer.{i}.input_norm_beta [hidden]
      layer.{i}.w_q/w_k/w_v [hidden, hidden] (transposed: [in, out])
      layer.{i}.w_o [hidden, hidden]
      layer.{i}.post_attn_norm [hidden], layer.{i}.post_attn_norm_beta [hidden]
      layer.{i}.w_fc1 [hidden, 4*hidden], layer.{i}.w_fc2 [4*hidden, hidden]
      final_norm [hidden], final_norm_beta [hidden]
      w_out [hidden, output_vocab]
    """
    weights = WeightDict()
    hidden = sub_config["hidden_size"]
    num_layers = sub_config["num_layers"]

    def _to_np(t):
        if hasattr(t, 'numpy'):
            return t.numpy().astype(np.float32)
        return np.asarray(t, dtype=np.float32)

    def _t2d(w):
        """Transpose [out, in] -> [in, out] for matmul."""
        a = _to_np(w)
        if a.ndim == 2:
            return np.ascontiguousarray(a.T)
        return a

    # Embedding
    weights["embedding"] = _to_np(state_dict[f"{prefix}.input_embeds_layer.weight"])

    # Position embedding
    weights["position_embedding"] = _to_np(state_dict[f"{prefix}.position_embeds_layer.weight"])

    for i in range(num_layers):
        hf = f"{prefix}.layers.{i}"
        layer = f"layer.{i}"

        # Layer norms (bark-small has weight only, no bias)
        weights[f"{layer}.input_norm"] = _to_np(state_dict[f"{hf}.layernorm_1.weight"])
        if f"{hf}.layernorm_1.bias" in state_dict:
            weights[f"{layer}.input_norm_beta"] = _to_np(state_dict[f"{hf}.layernorm_1.bias"])
        else:
            weights[f"{layer}.input_norm_beta"] = np.zeros(hidden, dtype=np.float32)

        weights[f"{layer}.post_attn_norm"] = _to_np(state_dict[f"{hf}.layernorm_2.weight"])
        if f"{hf}.layernorm_2.bias" in state_dict:
            weights[f"{layer}.post_attn_norm_beta"] = _to_np(state_dict[f"{hf}.layernorm_2.bias"])
        else:
            weights[f"{layer}.post_attn_norm_beta"] = np.zeros(hidden, dtype=np.float32)

        # att_proj: [3*H, H] -> split into Q [H,H], K [H,H], V [H,H], transpose each
        att_w = _to_np(state_dict[f"{hf}.attn.att_proj.weight"])  # [3H, H]
        w_q = att_w[:hidden, :]           # [H, H] in [out, in]
        w_k = att_w[hidden:2*hidden, :]   # [H, H]
        w_v = att_w[2*hidden:, :]         # [H, H]
        weights[f"{layer}.w_q"] = _t2d(w_q)  # [in, out] = [H, H]
        weights[f"{layer}.w_k"] = _t2d(w_k)
        weights[f"{layer}.w_v"] = _t2d(w_v)

        # Output projection
        weights[f"{layer}.w_o"] = _t2d(state_dict[f"{hf}.attn.out_proj.weight"])

        # MLP: fc1 = in_proj, fc2 = out_proj
        weights[f"{layer}.w_fc1"] = _t2d(state_dict[f"{hf}.mlp.in_proj.weight"])
        weights[f"{layer}.w_fc2"] = _t2d(state_dict[f"{hf}.mlp.out_proj.weight"])

        # Biases (bark-small typically has no biases on attn/mlp, but handle if present)
        for bkey, wkey in [
            (f"{layer}.q_bias", f"{hf}.attn.att_proj.bias"),
            (f"{layer}.o_bias", f"{hf}.attn.out_proj.bias"),
            (f"{layer}.fc1_bias", f"{hf}.mlp.in_proj.bias"),
            (f"{layer}.fc2_bias", f"{hf}.mlp.out_proj.bias"),
        ]:
            if wkey in state_dict:
                b = _to_np(state_dict[wkey])
                if "q_bias" in bkey and len(b) == 3 * hidden:
                    # Fused QKV bias, split
                    weights[f"{layer}.q_bias"] = b[:hidden]
                    weights[f"{layer}.k_bias"] = b[hidden:2*hidden]
                    weights[f"{layer}.v_bias"] = b[2*hidden:]
                else:
                    weights[bkey] = b

    # Final layer norm
    ln_key = f"{prefix}.layernorm_final.weight"
    if ln_key in state_dict:
        weights["final_norm"] = _to_np(state_dict[ln_key])
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)
    ln_bias_key = f"{prefix}.layernorm_final.bias"
    if ln_bias_key in state_dict:
        weights["final_norm_beta"] = _to_np(state_dict[ln_bias_key])
    else:
        weights["final_norm_beta"] = np.zeros(hidden, dtype=np.float32)

    # LM head: standard_decoder_builder uses "w_out"
    lm_key = f"{prefix}.lm_head.weight"
    if lm_key in state_dict:
        weights["w_out"] = _t2d(state_dict[lm_key])  # [hidden, output_vocab]
    else:
        # Tied embeddings
        weights["w_out"] = _t2d(state_dict[f"{prefix}.input_embeds_layer.weight"])

    return weights


def _map_bark_fine_weights(
    state_dict: dict,
    fine_cfg: dict,
) -> WeightDict:
    """Map HF Bark fine model weights to the fine engine format.

    The fine model (BarkFineModel) differs from semantic/coarse:
      - 8 embedding tables (one per codebook): input_embeds_layers.{i}.weight
      - 1 position embedding: position_embeds_layer.weight
      - 12 transformer layers (same structure as semantic/coarse)
      - 7 LM heads (codebooks 1-7): lm_heads.{j}.weight
      - Final layernorm

    Weight mapping:
      fine_acoustics.input_embeds_layers.{i}.weight  -> fine.embedding_{i}
      fine_acoustics.position_embeds_layer.weight     -> fine.position_embedding
      fine_acoustics.layers.{i}.*                     -> fine.layer.{i}.*
      fine_acoustics.layernorm_final.weight/bias      -> fine.final_norm / fine.final_norm_beta
      fine_acoustics.lm_heads.{j}.weight              -> fine.w_lm_head_{j}  (transposed)
    """
    weights = WeightDict()
    prefix = "fine_acoustics"
    hidden = fine_cfg["hidden_size"]
    num_layers = fine_cfg["num_layers"]
    n_embed_tables = fine_cfg.get("n_embed_tables", 8)
    n_lm_heads = fine_cfg.get("n_lm_heads", 7)

    def _to_np(t):
        if hasattr(t, 'numpy'):
            return t.numpy().astype(np.float32)
        return np.asarray(t, dtype=np.float32)

    def _t2d(w):
        """Transpose [out, in] -> [in, out] for matmul."""
        a = _to_np(w)
        if a.ndim == 2:
            return np.ascontiguousarray(a.T)
        return a

    # 8 embedding tables
    for i in range(n_embed_tables):
        key = f"{prefix}.input_embeds_layers.{i}.weight"
        if key in state_dict:
            weights[f"fine.embedding_{i}"] = _to_np(state_dict[key])

    # Position embedding
    wpe_key = f"{prefix}.position_embeds_layer.weight"
    if wpe_key in state_dict:
        weights["fine.position_embedding"] = _to_np(state_dict[wpe_key])

    # Transformer layers (same pattern as _map_bark_decoder_weights)
    for i in range(num_layers):
        hf = f"{prefix}.layers.{i}"
        layer = f"fine.layer.{i}"

        # Layer norms
        weights[f"{layer}.input_norm"] = _to_np(state_dict[f"{hf}.layernorm_1.weight"])
        if f"{hf}.layernorm_1.bias" in state_dict:
            weights[f"{layer}.input_norm_beta"] = _to_np(state_dict[f"{hf}.layernorm_1.bias"])
        else:
            weights[f"{layer}.input_norm_beta"] = np.zeros(hidden, dtype=np.float32)

        weights[f"{layer}.post_attn_norm"] = _to_np(state_dict[f"{hf}.layernorm_2.weight"])
        if f"{hf}.layernorm_2.bias" in state_dict:
            weights[f"{layer}.post_attn_norm_beta"] = _to_np(state_dict[f"{hf}.layernorm_2.bias"])
        else:
            weights[f"{layer}.post_attn_norm_beta"] = np.zeros(hidden, dtype=np.float32)

        # att_proj: [3*H, H] -> split into Q, K, V, transpose each
        att_w = _to_np(state_dict[f"{hf}.attn.att_proj.weight"])  # [3H, H]
        w_q = att_w[:hidden, :]
        w_k = att_w[hidden:2*hidden, :]
        w_v = att_w[2*hidden:, :]
        weights[f"{layer}.w_q"] = _t2d(w_q)
        weights[f"{layer}.w_k"] = _t2d(w_k)
        weights[f"{layer}.w_v"] = _t2d(w_v)

        # Output projection
        weights[f"{layer}.w_o"] = _t2d(state_dict[f"{hf}.attn.out_proj.weight"])

        # MLP
        weights[f"{layer}.w_fc1"] = _t2d(state_dict[f"{hf}.mlp.in_proj.weight"])
        weights[f"{layer}.w_fc2"] = _t2d(state_dict[f"{hf}.mlp.out_proj.weight"])

        # Biases
        for bkey, wkey in [
            (f"{layer}.q_bias", f"{hf}.attn.att_proj.bias"),
            (f"{layer}.o_bias", f"{hf}.attn.out_proj.bias"),
            (f"{layer}.fc1_bias", f"{hf}.mlp.in_proj.bias"),
            (f"{layer}.fc2_bias", f"{hf}.mlp.out_proj.bias"),
        ]:
            if wkey in state_dict:
                b = _to_np(state_dict[wkey])
                if "q_bias" in bkey and len(b) == 3 * hidden:
                    weights[f"{layer}.q_bias"] = b[:hidden]
                    weights[f"{layer}.k_bias"] = b[hidden:2*hidden]
                    weights[f"{layer}.v_bias"] = b[2*hidden:]
                else:
                    weights[bkey] = b

    # Final layer norm
    ln_key = f"{prefix}.layernorm_final.weight"
    if ln_key in state_dict:
        weights["fine.final_norm"] = _to_np(state_dict[ln_key])
    else:
        weights["fine.final_norm"] = np.ones(hidden, dtype=np.float32)
    ln_bias_key = f"{prefix}.layernorm_final.bias"
    if ln_bias_key in state_dict:
        weights["fine.final_norm_beta"] = _to_np(state_dict[ln_bias_key])
    else:
        weights["fine.final_norm_beta"] = np.zeros(hidden, dtype=np.float32)

    # 7 LM heads: transposed to [hidden, codebook_size]
    for j in range(n_lm_heads):
        lm_key = f"{prefix}.lm_heads.{j}.weight"
        if lm_key in state_dict:
            weights[f"fine.w_lm_head_{j}"] = _t2d(state_dict[lm_key])

    return weights


class BarkPlugin:
    name = "bark"
    runtime_strategy = "text_to_audio_bark"

    def __init__(self):
        self._semantic_cfg: dict = {}
        self._coarse_cfg: dict = {}
        self._fine_cfg: dict = {}
        self._codec_seq_length: int = 0
        self._fine_seq_length: int = 0

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("bark",)

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load Bark weights from pytorch_model.bin."""
        state_dict = _load_bark_state_dict(model_dir)

        # Detect sub-model configs
        semantic_cfg = _detect_sub_model_config(state_dict, "semantic")
        coarse_cfg = _detect_sub_model_config(state_dict, "coarse_acoustics")
        fine_cfg = _detect_sub_model_config(state_dict, "fine_acoustics")

        weights = WeightDict()

        # Map each sub-model's weights
        sem_w = _map_bark_decoder_weights(state_dict, "semantic", semantic_cfg)
        for k, v in sem_w.items():
            weights[f"semantic.{k}"] = v

        coarse_w = _map_bark_decoder_weights(state_dict, "coarse_acoustics", coarse_cfg)
        for k, v in coarse_w.items():
            weights[f"coarse.{k}"] = v

        # Map fine model weights
        fine_w = _map_bark_fine_weights(state_dict, fine_cfg)
        for k, v in fine_w.items():
            weights[k] = v

        # Store raw state_dict for codec engine builds
        weights["_state_dict"] = state_dict

        # Store sub-model configs
        weights["_semantic_cfg"] = semantic_cfg
        weights["_coarse_cfg"] = coarse_cfg
        weights["_fine_cfg"] = fine_cfg
        self._semantic_cfg = semantic_cfg
        self._coarse_cfg = coarse_cfg
        self._fine_cfg = fine_cfg

        # Store codec info
        weights["_codec_model_id"] = "facebook/encodec_24khz"

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        """Build TRT engine for semantic decoder (primary engine)."""
        sem_cfg = weights["_semantic_cfg"]
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Bark semantic tensor-parallel decoder builds")
            if quant_ctx is not None:
                raise ValueError("Bark tensor-parallel builds do not support quantization")
            sem_weights = _extract_bark_sub_weights(weights, "semantic")
            return build_bark_tp_decoder_engine(
                config,
                sem_weights,
                max_cache_length,
                sub_model="semantic",
                sub_cfg=sem_cfg,
                precision=precision,
                verbose=verbose,
                parallel_config=parallel,
            )
        return _build_bark_standard_engine(
            weights, "semantic", sem_cfg, max_cache_length,
            embed_input=True, precision=precision, verbose=verbose)

    def build_extra_engines(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        verbose: bool = False,
        build_timing: dict | None = None,
        parallel_config=None,
    ) -> dict:
        """Build coarse, fine, codec engines + embedding tables for C++ runtime."""
        coarse_cfg = weights["_coarse_cfg"]
        fine_cfg = weights["_fine_cfg"]
        parallel = normalize_parallel_config(parallel_config)

        result = {}
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Bark coarse tensor-parallel decoder builds")
            coarse_weights = _extract_bark_sub_weights(weights, "coarse")
            with timed_trt_compile(build_timing, "extra_bark_coarse_decoder"):
                for rank in range(parallel.tp_size):
                    rank_parallel = parallel.for_rank(rank)
                    if verbose:
                        print(
                            f"[trtmc build]   Building coarse TP rank "
                            f"{rank}/{parallel.tp_size} ...",
                            file=sys.stderr,
                        )
                    result[f"coarse_engine_plan_tp_rank{rank}"] = (
                        build_bark_tp_decoder_engine(
                            config,
                            coarse_weights,
                            max_cache_length,
                            sub_model="coarse",
                            sub_cfg=coarse_cfg,
                            precision=precision,
                            verbose=verbose,
                            parallel_config=rank_parallel,
                        )
                    )
        else:
            with timed_trt_compile(build_timing, "extra_bark_coarse_decoder"):
                coarse_plan = _build_bark_standard_engine(
                    weights, "coarse", coarse_cfg, max_cache_length,
                    embed_input=True, precision=precision, verbose=verbose)
            result["coarse_engine_plan"] = coarse_plan

        # Add embedding tables as raw bundle sections.
        # The C++ runtime does host-side embedding lookup for embed_input mode.
        state_dict = weights.get("_state_dict")
        if state_dict is not None:
            sem_key = "semantic.input_embeds_layer.weight"
            if sem_key in state_dict:
                w = state_dict[sem_key]
                if hasattr(w, 'numpy'):
                    w = w.numpy()
                result["semantic_embed"] = np.asarray(w, dtype=np.float32).tobytes()

            coarse_key = "coarse_acoustics.input_embeds_layer.weight"
            if coarse_key in state_dict:
                w = state_dict[coarse_key]
                if hasattr(w, 'numpy'):
                    w = w.numpy()
                result["coarse_embed"] = np.asarray(w, dtype=np.float32).tobytes()

        # Calculate max codec frames from max_cache_length.
        # Semantic generates at most ~(max_cache_length - 257) tokens.
        # Coarse frames ≈ semantic_tokens * (75 / 49.9).
        max_semantic = max(max_cache_length - 257, 100)
        max_codec_frames = int(max_semantic * 75 / 49.9) + 1
        # Round up to multiple of 64 for TRT efficiency
        max_codec_frames = ((max_codec_frames + 63) // 64) * 64
        # Cap at 1024 frames — LSTM unrolling creates O(N) TRT layers per
        # timestep; very large values may make the TensorRT compiler OOM.
        # 1024 frames = 1024*320/24000 ~= 13.7s of audio.
        max_codec_frames = min(max_codec_frames, 1024)
        self._codec_seq_length = max_codec_frames

        # Build fine engine (non-autoregressive, bidirectional attention)
        fine_seq_length = max_codec_frames  # match codec_seq_length
        self._fine_seq_length = fine_seq_length
        if verbose:
            print(f"[trtmc build]   Building fine engine "
                  f"(seq_length={fine_seq_length}) ...",
                  file=sys.stderr)
        with timed_trt_compile(build_timing, "extra_bark_fine_decoder"):
            fine_plan = _build_bark_fine_engine(
                weights, fine_cfg, seq_length=fine_seq_length,
                precision=precision, verbose=verbose)
        result["fine_engine_plan"] = fine_plan

        # Add fine embedding tables as a single concatenated section.
        # Layout: table0 || table1 || ... || table7, each [codebook_size, hidden_size].
        # The C++ runtime indexes as: table[cb * codebook_size * hidden + code * hidden + h].
        n_embed_tables = fine_cfg.get("n_embed_tables", 8)
        embed_parts = []
        for i in range(n_embed_tables):
            embed_key = f"fine.embedding_{i}"
            if embed_key in weights:
                embed_parts.append(
                    np.asarray(weights[embed_key], dtype=np.float32))
        if embed_parts:
            result["fine_embed"] = np.concatenate(
                [e.ravel() for e in embed_parts]).tobytes()
        pos_key = "fine.position_embedding"
        if pos_key in weights:
            result["fine_position_embed"] = np.asarray(
                weights[pos_key], dtype=np.float32).ravel().tobytes()

        # Build codec (EnCodec) engine for waveform synthesis
        if state_dict is not None:
            from .encodec_builder import build_encodec_decoder_engine

            if verbose:
                print(f"[trtmc build]   Building codec engine "
                      f"(max_frames={max_codec_frames}) ...",
                      file=sys.stderr)

            with timed_trt_compile(build_timing, "extra_encodec_audio_decoder"):
                codec_plan = build_encodec_decoder_engine(
                    state_dict, seq_length=max_codec_frames,
                    precision=precision, verbose=verbose)
            result["codec_engine_plan"] = codec_plan

        return result

    def get_audio_config(self, config: ModelConfig) -> dict:
        """Return audio config for bundle config.json."""
        cfg = {
            "sample_rate": 24000,
            "semantic_vocab_size": 10000,
            "coarse_vocab_size": 1024,
            "fine_vocab_size": 1024,
            "n_coarse_codebooks": 2,
            "n_fine_codebooks": 8,
            "semantic_pad_token": 10000,
            "semantic_infer_token": 129599,
            "coarse_semantic_pad_token": 12048,
            "coarse_infer_token": 12050,
            "text_encoding_offset": 10048,
            "text_pad_token": 129595,
            "semantic_input_vocab": 129600,
            "coarse_input_vocab": 12096,
            "codebook_size": 1024,
        }
        # Inject semantic sub-model dimensions at top level.
        # HF Bark config.json nests these inside semantic_config/coarse_acoustics_config
        # but the C++ parser needs them at top level.
        if self._semantic_cfg:
            cfg["vocab_size"] = self._semantic_cfg["vocab_size"]
            cfg["hidden_size"] = self._semantic_cfg["hidden_size"]
            cfg["num_hidden_layers"] = self._semantic_cfg["num_layers"]
            cfg["num_attention_heads"] = self._semantic_cfg["num_heads"]
            cfg["num_key_value_heads"] = self._semantic_cfg["num_heads"]
        # Inject coarse sub-model dimensions (detected during load_weights)
        if self._coarse_cfg:
            cfg["coarse_hidden_size"] = self._coarse_cfg["hidden_size"]
            cfg["coarse_num_layers"] = self._coarse_cfg["num_layers"]
            cfg["coarse_num_heads"] = self._coarse_cfg["num_heads"]
        # Inject fine sub-model dimensions (detected during load_weights)
        if self._fine_cfg:
            cfg["fine_hidden_size"] = self._fine_cfg["hidden_size"]
            cfg["fine_num_layers"] = self._fine_cfg["num_layers"]
            cfg["fine_num_heads"] = self._fine_cfg["num_heads"]
            cfg["fine_codebook_size"] = self._fine_cfg.get("codebook_size", 1056)
            cfg["fine_n_lm_heads"] = self._fine_cfg.get("n_lm_heads", 7)
            if self._fine_seq_length > 0:
                cfg["fine_seq_length"] = self._fine_seq_length
        # Codec engine config
        if self._codec_seq_length > 0:
            cfg["codec_seq_length"] = self._codec_seq_length
            cfg["codec_upsample_factor"] = 320  # 8*5*4*2
            cfg["codec_n_codebooks"] = 8
        return cfg


def _extract_bark_sub_weights(weights: WeightDict, sub_model: str) -> WeightDict:
    """Extract semantic/coarse weights and strip the stored sub-model prefix."""
    prefix = f"{sub_model}."
    sub_weights = WeightDict()
    for k, v in weights.items():
        if k.startswith(prefix) and not k.startswith("_"):
            sub_weights[k[len(prefix):]] = v
    return sub_weights


def _build_bark_standard_engine(
    weights: WeightDict,
    sub_model: str,
    sub_cfg: dict,
    max_cache_length: int,
    embed_input: bool = True,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build a standard decoder engine for semantic or coarse using build_standard_decoder_engine."""
    from .standard_decoder_builder import build_standard_decoder_engine

    # Extract sub-model weights (strip prefix)
    sub_weights = _extract_bark_sub_weights(weights, sub_model)

    # Also need _attention_size and _mlp_size for the builder
    hidden = sub_cfg["hidden_size"]
    num_heads = sub_cfg["num_heads"]
    head_dim = hidden // num_heads
    sub_weights["_attention_size"] = num_heads * head_dim
    sub_weights["_mlp_size"] = sub_cfg.get("intermediate_size", hidden * 4)

    # Create a ModelConfig for this sub-model
    sub_mc = ModelConfig(
        model_type="bark",
        vocab_size=sub_cfg["vocab_size"],
        hidden_size=hidden,
        intermediate_size=sub_cfg.get("intermediate_size", hidden * 4),
        num_hidden_layers=sub_cfg["num_layers"],
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=sub_cfg.get("max_position", 1024),
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        raw={},
    )

    if verbose:
        print(f"[trtmc build]   Building {sub_model} engine: "
              f"layers={sub_cfg['num_layers']}, hidden={hidden}, "
              f"vocab={sub_cfg['vocab_size']}, output_vocab={sub_cfg.get('output_vocab', sub_cfg['vocab_size'])}",
              file=sys.stderr)

    return build_standard_decoder_engine(
        sub_mc, sub_weights, max_cache_length,
        norm_type="layernorm",
        mlp_type="gelu_fc",
        position_type="learned",
        activation="gelu_new",
        embed_input=embed_input,
        precision=precision,
        verbose=verbose,
    )


def _build_bark_decoder_engine(
    weights: WeightDict,
    sub_model: str,
    sub_cfg: dict,
    max_cache_length: int,
    embed_input: bool = True,
    verbose: bool = False,
) -> bytes:
    """Build a standard decoder engine for semantic or coarse sub-model."""
    hidden = sub_cfg["hidden_size"]
    vocab = sub_cfg["vocab_size"]
    num_layers = sub_cfg["num_layers"]
    num_heads = sub_cfg["num_heads"]
    head_dim = hidden // num_heads

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    attention_window = max_cache_length + 1
    attention_size = num_heads * head_dim
    prefix = f"{sub_model}."

    # Inputs
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, attention_window))

    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            trt.float32, (max_cache_length, attention_size))
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            trt.float32, (max_cache_length, attention_size))
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # Embedding tables
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights[f"{prefix}embedding"])
    position_table = graph_ops.add_constant(
        network, (sub_cfg["max_position"], hidden),
        weights[f"{prefix}position_embedding"])

    # Token + position embedding
    tok_embed = network.add_gather(embedding_table, token_id, 0)
    pos_embed = network.add_gather(position_table, position_id, 0)
    hidden_state = network.add_elementwise(
        tok_embed.get_output(0), pos_embed.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=np.float32))

    present_k_outputs = []
    present_v_outputs = []

    for layer_idx in range(num_layers):
        lp = f"{prefix}layer.{layer_idx}"

        # Pre-attention LayerNorm
        normed = graph_ops.add_layer_norm(
            network, hidden_state, hidden,
            weights[f"{lp}.norm1_gamma"],
            weights[f"{lp}.norm1_beta"], eps_t)

        # Q, K, V projections
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{lp}.w_q"])
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{lp}.w_k"])
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{lp}.w_v"])

        # Attention with KV cache
        # K -> present_k: [1, attention_size]
        # Gather from cache
        attn_scale = 1.0 / np.sqrt(head_dim)

        # present_k = k (single new row)
        present_k = k
        present_v = v

        # Full K = cache_k concat present_k for attention
        # [max_cache, attn] + [1, attn] -> [max_cache+1, attn]
        full_k = network.add_concatenation(
            [cache_k_inputs[layer_idx], present_k])
        full_k.axis = 0

        full_v = network.add_concatenation(
            [cache_v_inputs[layer_idx], present_v])
        full_v.axis = 0

        mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
        ctx_flat = graph_ops.add_attention_from_rows(
            network, q, full_k.get_output(0), full_v.get_output(0),
            num_heads=num_heads, head_dim=head_dim,
            q_seq=1, kv_seq=attention_window,
            mask=mask_4d,
            scale=attn_scale)

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx_flat, attention_size, hidden,
            weights[f"{lp}.w_o"])
        attn_out = graph_ops.add_bias_sum(
            network, attn_out, hidden, weights[f"{lp}.b_o"])

        # Residual
        res1 = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        hidden_state = res1.get_output(0)

        # Pre-FFN LayerNorm
        normed2 = graph_ops.add_layer_norm(
            network, hidden_state, hidden,
            weights[f"{lp}.norm2_gamma"],
            weights[f"{lp}.norm2_beta"], eps_t)

        # MLP: FC1 -> GELU -> FC2
        ffn_hidden = weights[f"{lp}.w_fc1"].shape[1]
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden, ffn_hidden,
            weights[f"{lp}.w_fc1"])
        fc1 = graph_ops.add_bias_sum(network, fc1, ffn_hidden,
            weights[f"{lp}.b_fc1"])
        # HF BarkMLP uses nn.GELU() with the exact erf formulation.
        gelu = graph_ops.add_gelu_erf(network, fc1)
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, gelu, ffn_hidden, hidden,
            weights[f"{lp}.w_fc2"])
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden,
            weights[f"{lp}.b_fc2"])

        # Residual
        res2 = network.add_elementwise(
            hidden_state, fc2, trt.ElementWiseOperation.SUM)
        hidden_state = res2.get_output(0)

        present_k_outputs.append(present_k)
        present_v_outputs.append(present_v)

    # Final LayerNorm
    hidden_state = graph_ops.add_layer_norm(
        network, hidden_state, hidden,
        weights[f"{prefix}final_norm_gamma"],
        weights[f"{prefix}final_norm_beta"], eps_t)

    # LM head
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab,
        weights[f"{prefix}w_lm_head"])

    logits.name = "logits"
    network.mark_output(logits)

    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    if verbose:
        print(f"[trtmc build] Building Bark {sub_model} engine "
              f"(layers={num_layers}, hidden={hidden}, vocab={vocab}) ...",
              file=sys.stderr)

    plan = build_bark_serialized_network(builder, network, trt_config)
    if plan is None:
        raise RuntimeError(f"TensorRT engine build failed for Bark {sub_model}")
    return bytes(plan)


def _build_bark_fine_engine(
    weights: WeightDict,
    fine_cfg: dict,
    seq_length: int = 1024,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build a non-autoregressive TRT engine for the Bark fine model.

    The fine model processes a full sequence at once with BIDIRECTIONAL
    self-attention (no causal mask, no KV cache). It predicts codebooks 1-7
    via 7 separate LM heads.

    Architecture:
      - Input: input_embeds [seq_length, hidden_size] float32 (pre-computed
        summed embeddings from C++, covering all 8 codebook embeddings +
        position embedding)
      - 12 transformer layers: LayerNorm -> bidirectional self-attention ->
        residual -> LayerNorm -> GELU MLP -> residual
      - Final LayerNorm
      - 7 LM heads, each producing [seq_length, codebook_size] logits

    Outputs: logits_cb1 through logits_cb7, each [seq_length, codebook_size].

    Args:
        weights: Weight dict with fine.* keys from _map_bark_fine_weights.
        fine_cfg: Fine sub-model config from _detect_sub_model_config.
        seq_length: Fixed sequence length (should match codec_seq_length).
        verbose: Print TRT builder logs.

    Returns:
        Serialized engine plan bytes.
    """
    hidden = fine_cfg["hidden_size"]
    num_layers = fine_cfg["num_layers"]
    num_heads = fine_cfg["num_heads"]
    head_dim = hidden // num_heads
    n_lm_heads = fine_cfg.get("n_lm_heads", 7)
    codebook_size = fine_cfg.get("codebook_size", 1056)
    prefix = "fine."
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(
            f"Unsupported Bark precision {precision!r}; expected fp32 or fp16")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Input: pre-computed summed embeddings [seq_length, hidden_size]
    # C++ runtime sums the 8 codebook embeddings + position embedding and
    # passes the result directly.
    input_embeds = network.add_input(
        "input_embeds", trt.float32, (seq_length, hidden))
    if work_trt_dtype != trt.float32:
        input_embeds = network.add_cast(
            input_embeds, work_trt_dtype).get_output(0)

    hidden_state = input_embeds

    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=work_np_dtype),
        dtype=work_np_dtype)

    for layer_idx in range(num_layers):
        lp = f"{prefix}layer.{layer_idx}"
        layer_np_dtype = work_np_dtype
        layer_trt_dtype = work_trt_dtype
        if hidden_state.dtype != layer_trt_dtype:
            hidden_state = network.add_cast(
                hidden_state, layer_trt_dtype).get_output(0)
        layer_eps_t = graph_ops.add_constant(
            network, (1, 1), np.array([1e-5], dtype=layer_np_dtype),
            dtype=layer_np_dtype)

        # Pre-attention LayerNorm
        normed = graph_ops.add_layer_norm(
            network, hidden_state, hidden,
            weights[f"{lp}.input_norm"],
            weights[f"{lp}.input_norm_beta"], layer_eps_t,
            dtype=layer_np_dtype)

        # Bidirectional self-attention (no causal mask, no KV cache)
        # normed: [seq_length, hidden]
        attention_size = num_heads * head_dim
        attn_scale = 1.0 / np.sqrt(head_dim)

        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size, weights[f"{lp}.w_q"],
            dtype=layer_np_dtype)
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size, weights[f"{lp}.w_k"],
            dtype=layer_np_dtype)
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size, weights[f"{lp}.w_v"],
            dtype=layer_np_dtype)

        # Optional QKV biases
        for bias_name, tensor_ref in [
            (f"{lp}.q_bias", "q"), (f"{lp}.k_bias", "k"), (f"{lp}.v_bias", "v")]:
            b = weights.get(bias_name)
            if b is not None:
                ref = {"q": q, "k": k, "v": v}[tensor_ref]
                ref_out = graph_ops.add_bias_sum(
                    network, ref, attention_size, b, dtype=layer_np_dtype)
                if tensor_ref == "q":
                    q = ref_out
                elif tensor_ref == "k":
                    k = ref_out
                else:
                    v = ref_out

        ctx_flat = graph_ops.add_attention_from_rows(
            network, q, k, v,
            num_heads=num_heads, head_dim=head_dim,
            q_seq=seq_length, kv_seq=seq_length,
            scale=attn_scale,
            fp32_accumulation=work_np_dtype != np.float32)

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx_flat, attention_size, hidden,
            weights[f"{lp}.w_o"], dtype=layer_np_dtype)
        o_bias = weights.get(f"{lp}.o_bias")
        if o_bias is not None:
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, o_bias, dtype=layer_np_dtype)

        # Residual
        hidden_state = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM).get_output(0)

        # Pre-FFN LayerNorm
        normed2 = graph_ops.add_layer_norm(
            network, hidden_state, hidden,
            weights[f"{lp}.post_attn_norm"],
            weights[f"{lp}.post_attn_norm_beta"], layer_eps_t,
            dtype=layer_np_dtype)

        # MLP: FC1 -> GELU -> FC2
        mlp_size = weights[f"{lp}.w_fc1"].shape[1]
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden, mlp_size, weights[f"{lp}.w_fc1"],
            dtype=layer_np_dtype)
        fc1_bias = weights.get(f"{lp}.fc1_bias")
        if fc1_bias is not None:
            fc1 = graph_ops.add_bias_sum(
                network, fc1, mlp_size, fc1_bias, dtype=layer_np_dtype)
        # HF BarkMLP uses nn.GELU() with the exact erf formulation.
        gelu = graph_ops.add_gelu_erf(network, fc1, dtype=layer_np_dtype)
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, gelu, mlp_size, hidden, weights[f"{lp}.w_fc2"],
            dtype=layer_np_dtype)
        fc2_bias = weights.get(f"{lp}.fc2_bias")
        if fc2_bias is not None:
            fc2 = graph_ops.add_bias_sum(
                network, fc2, hidden, fc2_bias, dtype=layer_np_dtype)

        # Residual
        hidden_state = network.add_elementwise(
            hidden_state, fc2, trt.ElementWiseOperation.SUM).get_output(0)
        if layer_np_dtype != work_np_dtype:
            hidden_state = network.add_cast(
                hidden_state, work_trt_dtype).get_output(0)

    # Final LayerNorm
    hidden_state = graph_ops.add_layer_norm(
        network, hidden_state, hidden,
        weights[f"{prefix}final_norm"],
        weights[f"{prefix}final_norm_beta"], eps_t,
        dtype=work_np_dtype)

    # 7 LM heads: each [seq_length, hidden] -> [seq_length, codebook_size]
    for j in range(n_lm_heads):
        logits_j = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, codebook_size,
            weights[f"{prefix}w_lm_head_{j}"], dtype=work_np_dtype)
        if logits_j.dtype != trt.float32:
            logits_j = network.add_cast(
                logits_j, trt.float32).get_output(0)
        logits_j.name = f"logits_cb{j + 1}"
        network.mark_output(logits_j)

    if verbose:
        print(f"[trtmc build] Building Bark fine engine "
              f"(layers={num_layers}, hidden={hidden}, heads={num_heads}, "
              f"seq_length={seq_length}, lm_heads={n_lm_heads}, "
              f"codebook_size={codebook_size}) ...",
              file=sys.stderr)

    plan = build_bark_serialized_network(builder, network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed for Bark fine model")
    return bytes(plan)


plugin = BarkPlugin()
