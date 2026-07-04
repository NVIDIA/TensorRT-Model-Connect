# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex family plugin -- speech-to-speech via Moshi architecture.

PersonaPlex (nvidia/personaplex-7b-v1) is based on the Moshi architecture:
  1. Temporal Transformer: 32-layer decoder processing joint listen + speak streams
  2. Depth Transformer: 6-layer per-timestep multi-codebook token generation
  3. Mimi Neural Codec: ConvNet encoder/decoder (placeholder engines for now)

Pipeline:
  audio_in -> Mimi_encode -> Temporal_process -> Depth_generate -> Mimi_decode -> audio_out

Bring-up CLI reference (used for PersonaPlex TRT/C++ iteration):
  # Rebuild C++ runtime after speech backend / CLI changes
  docker exec trtmc-dev-gb300 bash -lc 'cd /workspace/tensorrt-model-connect && cmake --build build -j'

  # Build a PersonaPlex bundle from HF (base engine bundle)
  docker exec trtmc-dev-gb300 bash -lc 'cd /workspace/tensorrt-model-connect && ./build/trtmc build nvidia/personaplex-7b-v1 -o /tmp/trtmc-ci/engines/personaplex-7b.trtfb --max-cache-length 256 --verbose'

  # Rebuild only Mimi decoder capacity (example: 320 frames) and repack bundle
  docker exec trtmc-dev-gb300 bash -lc 'cd /workspace/tensorrt-model-connect && .venv/bin/python <inline script calling _build_mimi_decoder_engine(..., num_input_codebooks=8, num_frames=320) and replacing mimi_decoder_plan in an existing .trtfb>'

  # Run speech inference (long-form test using tail frame extension)
  docker exec trtmc-dev-gb300 bash -lc 'cd /workspace/tensorrt-model-connect && ./build/trtmc speak /tmp/trtmc-ci/engines/personaplex-7b-ropefix2-dec320.trtfb --audio-in tests/e2e/models/personaplex/data/Recording.wav --audio-out /workspace/tensorrt-model-connect/personaplex_recording_ropefix2_dec320_tail300_max1000.wav --max-new-tokens 1000 --tail-frames 300'

Real weight key structure (nvidia/personaplex-7b-v1):
  Temporal Transformer (32 layers, hidden=4096):
    transformer.layers.{i}.self_attn.in_proj_weight: [12288, 4096]  # fused QKV
    transformer.layers.{i}.self_attn.out_proj.weight: [4096, 4096]
    transformer.layers.{i}.gating.linear_in.weight: [22528, 4096]   # fused gate+up
    transformer.layers.{i}.gating.linear_out.weight: [4096, 11264]  # down proj
    transformer.layers.{i}.norm1.alpha: [1, 1, 4096]                # RMSNorm
    transformer.layers.{i}.norm2.alpha: [1, 1, 4096]

  Depth Transformer (6 layers, hidden=1024, 16 codebooks):
    depformer.layers.{i}.self_attn.in_proj_weight: [49152, 1024]    # fused QKV * 16 codebooks
    depformer.layers.{i}.self_attn.out_proj.weight: [16384, 1024]   # out_proj * 16 codebooks
    depformer.layers.{i}.gating.{cb}.linear_in.weight: [5632, 1024] # per-codebook gated MLP
    depformer.layers.{i}.gating.{cb}.linear_out.weight: [1024, 2816]
    depformer.layers.{i}.norm1.alpha: [1, 1, 1024]
    depformer.layers.{i}.norm2.alpha: [1, 1, 1024]

  Embeddings:
    emb.{0-15}.weight: [2049, 4096]         # per-codebook audio embeddings (temporal)
    text_emb.weight: [32001, 4096]           # text embedding (temporal)
    text_linear.weight: [32000, 4096]        # text output projection
    depformer_emb.{0-14}.weight: [2049, 1024] # per-codebook embeddings (depth, 15 only)
    depformer_in.{0-15}.weight: [1024, 4096] # temporal->depth projections
    depformer_text_emb.weight: [32001, 1024] # text embedding (depth)
    linears.{0-15}.weight: [2048, 1024]      # per-codebook output heads
    out_norm.alpha: [1, 1, 4096]             # output norm
"""

from __future__ import annotations

import re
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
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)

trt = trt_compat.get_trt() if trt_compat.is_available() else None

_TEMPORAL_COMPONENT = 0
_DEPTH_COMPONENT = 1
_MIMI_ENCODER_COMPONENT = 2
_MIMI_DECODER_COMPONENT = 3
_TEMPORAL_LAYER_SELECTOR_BASE = 4


def _component_precision(
    config: ModelConfig,
    component: int,
    precision: str,
) -> str:
    selected = {int(index) for index in config.raw.get("_fp32_layers", ())}
    return "fp32" if component in selected else precision

# Conditionally import graph_ops -- it needs TRT
try:
    from . import graph_ops
    from .standard_decoder_builder import build_standard_decoder_engine
except ImportError:
    graph_ops = None
    build_standard_decoder_engine = None


def _infer_config_from_weights(readers) -> dict:
    """Infer all model dimensions from weight tensor shapes.

    The real personaplex config.json is minimal (just model_type + version),
    so we must derive everything from the weight shapes themselves.
    """
    # Collect all tensor keys across readers
    all_keys = set()
    for r in readers:
        all_keys.update(r.keys())

    # Temporal transformer dimensions
    norm1_key = "transformer.layers.0.norm1.alpha"
    if norm1_key not in all_keys:
        raise ValueError(f"Missing key {norm1_key} -- not a PersonaPlex model")
    norm1 = _load_tensor(readers, norm1_key)
    hidden = norm1.size  # flatten [1,1,4096] -> 4096

    # Count temporal layers
    num_layers = 0
    while f"transformer.layers.{num_layers}.norm1.alpha" in all_keys:
        num_layers += 1

    # Infer attention config from in_proj_weight [3*hidden, hidden]
    in_proj = _load_tensor(readers, "transformer.layers.0.self_attn.in_proj_weight")
    total_qkv = in_proj.shape[0]  # 12288 = 3 * 4096
    assert total_qkv == 3 * hidden, (
        f"in_proj_weight dim 0 ({total_qkv}) != 3 * hidden ({3 * hidden})")
    # Moshi uses 32 heads with head_dim=128 for hidden=4096
    num_heads = 32
    head_dim = hidden // num_heads  # 128

    # MLP dimensions from gating.linear_in [gate+up, hidden]
    gating_in = _load_tensor(readers, "transformer.layers.0.gating.linear_in.weight")
    fused_mlp_dim = gating_in.shape[0]  # 22528 = 2 * 11264
    intermediate_size = fused_mlp_dim // 2  # 11264

    # Depth transformer dimensions
    dep_norm1 = _load_tensor(readers, "depformer.layers.0.norm1.alpha")
    depth_hidden = dep_norm1.size  # 1024

    depth_num_layers = 0
    while f"depformer.layers.{depth_num_layers}.norm1.alpha" in all_keys:
        depth_num_layers += 1

    # Count codebooks from emb.{i}.weight
    num_codebooks = 0
    while f"emb.{num_codebooks}.weight" in all_keys:
        num_codebooks += 1

    # Vocab sizes
    emb0 = _load_tensor(readers, "emb.0.weight")
    audio_vocab = emb0.shape[0]  # 2049

    text_emb = _load_tensor(readers, "text_emb.weight")
    text_vocab = text_emb.shape[0]  # 32001

    # Output head vocab (codebook_size for audio)
    lin0 = _load_tensor(readers, "linears.0.weight")
    codebook_size = lin0.shape[0]  # 2048

    # Text output projection
    text_linear = _load_tensor(readers, "text_linear.weight")
    text_out_vocab = text_linear.shape[0]  # 32000

    # Depth MLP dimensions from per-codebook gating
    dep_gating_in = _load_tensor(readers, "depformer.layers.0.gating.0.linear_in.weight")
    depth_fused_mlp = dep_gating_in.shape[0]  # 5632
    depth_intermediate = depth_fused_mlp // 2  # 2816

    # Depth attention: in_proj [num_codebooks*3*depth_hidden, depth_hidden]
    dep_in_proj = _load_tensor(readers, "depformer.layers.0.self_attn.in_proj_weight")
    dep_total_qkv = dep_in_proj.shape[0]  # 49152
    # 49152 / 1024 = 48 = 16 codebooks * 3 (Q,K,V)
    dep_qkv_per_cb = dep_total_qkv // num_codebooks  # 3072 = 3 * 1024
    assert dep_qkv_per_cb == 3 * depth_hidden
    # Each codebook attention has 16 heads with head_dim=64
    depth_num_heads = 16
    depth_head_dim = depth_hidden // depth_num_heads  # 64

    # Count depformer_emb entries (may be fewer than num_codebooks)
    num_depformer_emb = 0
    while f"depformer_emb.{num_depformer_emb}.weight" in all_keys:
        num_depformer_emb += 1

    return {
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "num_codebooks": num_codebooks,
        "audio_vocab": audio_vocab,
        "text_vocab": text_vocab,
        "codebook_size": codebook_size,
        "text_out_vocab": text_out_vocab,
        "depth_hidden": depth_hidden,
        "depth_num_layers": depth_num_layers,
        "depth_num_heads": depth_num_heads,
        "depth_head_dim": depth_head_dim,
        "depth_intermediate": depth_intermediate,
        "num_depformer_emb": num_depformer_emb,
    }


class PersonaPlexPlugin:
    name = "personaplex"
    runtime_strategy = "personaplex_speech_to_speech"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in {"moshi", "personaplex", "personaplex_7b"}

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load PersonaPlex / Moshi weights from safetensors.

        Infers all dimensions from weight shapes since config.json is minimal.
        Returns a WeightDict with keys for all components:
          - temporal.*  (Temporal Transformer, standard decoder format)
          - depth.*     (Depth Transformer, per-codebook structure)
          - Various embedding and projection weights
        """
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        # Infer config from weight shapes
        inferred = _infer_config_from_weights(readers)

        weights = WeightDict()

        # ---------------------------------------------------------------
        # Temporal Transformer weights (standard decoder format)
        # ---------------------------------------------------------------
        _load_temporal_weights(
            weights, readers, inferred)

        # ---------------------------------------------------------------
        # Depth Transformer weights (stored as-is for custom engine)
        # ---------------------------------------------------------------
        _load_depth_weights(
            weights, readers, inferred)

        # ---------------------------------------------------------------
        # All embedding and projection weights
        # ---------------------------------------------------------------
        _load_embedding_weights(
            weights, readers, inferred)

        # ---------------------------------------------------------------
        # Store inferred metadata for engine builders
        # ---------------------------------------------------------------
        for k, v in inferred.items():
            weights[f"_{k}"] = v

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        """Build TRT engine for the Temporal Transformer.

        Uses standard decoder builder with:
          - No positional encoding (Moshi temporal uses no RoPE)
          - RMSNorm with alpha weights
          - SwiGLU MLP (gate+up fused, then down)
          - 32 heads, head_dim=128
        """
        hidden = weights["_hidden_size"]
        num_layers = weights["_num_hidden_layers"]
        num_heads = weights["_num_attention_heads"]
        head_dim = weights["_head_dim"]
        intermediate_size = weights["_intermediate_size"]
        selected_fp32 = {
            int(index) for index in config.raw.get("_fp32_layers", ())}
        valid_fp32 = set(range(_TEMPORAL_LAYER_SELECTOR_BASE + num_layers))
        invalid_fp32 = sorted(selected_fp32 - valid_fp32)
        if invalid_fp32:
            raise ValueError(
                "PersonaPlex fp32_layers contains unknown selectors: "
                f"{invalid_fp32}; expected 0=temporal, 1=depth, "
                "2=Mimi encoder, 3=Mimi decoder, or "
                f"4-{3 + num_layers}=temporal blocks 0-{num_layers - 1}")
        temporal_fp32_layers = tuple(sorted(
            selector - _TEMPORAL_LAYER_SELECTOR_BASE
            for selector in selected_fp32
            if selector >= _TEMPORAL_LAYER_SELECTOR_BASE))
        # Embedding table is text_emb [32001, 4096], output head is text_linear [32000, 4096]
        # vocab_size must match embedding table for the gather op
        text_vocab = weights["_text_vocab"]

        temporal_config = ModelConfig(
            model_type="personaplex",
            vocab_size=text_vocab,
            hidden_size=hidden,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            intermediate_size=intermediate_size,
            rms_norm_eps=1e-8,
            _head_dim=head_dim,
            raw={"hidden_size": hidden, "num_attention_heads": num_heads,
                 "head_dim": head_dim},
        )

        # Extract temporal.* keys to standard decoder format
        decoder_weights = WeightDict()
        for key, val in weights.items():
            if key.startswith("temporal."):
                decoder_weights[key[len("temporal."):]] = val
        decoder_weights["_kv_attention_size"] = hidden

        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            if precision == "fp16" and temporal_fp32_layers:
                raise NotImplementedError(
                    "PersonaPlex temporal per-layer FP32 selectors are not "
                    "supported with tensor parallelism")
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="PersonaPlex temporal tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("PersonaPlex tensor-parallel builds do not support quantization")
            from .decoder_tp_builder import build_personaplex_tp_decoder_engine
            return build_personaplex_tp_decoder_engine(
                temporal_config,
                decoder_weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type="rmsnorm",
                mlp_type="swiglu",
                position_type="rope",
                interleaved_rope=True,
                embed_input=True,
                verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                hidden_state_output=True,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            temporal_config, decoder_weights, max_cache_length,
            precision=_component_precision(
                config, _TEMPORAL_COMPONENT, precision),
            quant_ctx=quant_ctx,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            # Official PersonaPlex temporal transformer uses RoPE.
            position_type="rope",
            # Moshi RoPE rotates interleaved pairs: [d0,d1], [d2,d3], ...
            interleaved_rope=True,
            embed_input=True,  # Depth needs input_embed for temporal hidden injection  # Use learned position embeddings (zeros) as placeholder
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
            hidden_state_output=True,
            fp32_layers=(
                () if _TEMPORAL_COMPONENT in selected_fp32
                else temporal_fp32_layers),
        )  # Speech needs hidden state for depth transformer

    def build_extra_engines(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        verbose: bool = False,
        build_timing: dict | None = None,
    ) -> dict[str, bytes]:
        """Build additional engines: Depth Transformer, Mimi encoder, Mimi decoder.

        Returns dict mapping section name -> engine plan bytes.
        """
        extras = {}

        depth_hidden = weights["_depth_hidden"]
        depth_num_layers = weights["_depth_num_layers"]
        depth_num_heads = weights["_depth_num_heads"]
        depth_head_dim = weights["_depth_head_dim"]
        depth_intermediate = weights["_depth_intermediate"]
        num_codebooks = weights["_num_codebooks"]
        depth_precision = _component_precision(
            config, _DEPTH_COMPONENT, precision)
        mimi_encoder_precision = _component_precision(
            config, _MIMI_ENCODER_COMPONENT, precision)
        mimi_decoder_precision = _component_precision(
            config, _MIMI_DECODER_COMPONENT, precision)

        # --- Per-codebook Depth Transformer engines ---
        # The depth transformer has per-codebook attention (Q,K,V,O), MLP, and
        # output heads. We build 16 separate small engines (6 layers, 1024 hidden),
        # one per codebook, each with that codebook's specific weights baked in.
        # At runtime, the C++ backend selects the correct engine for each codebook step.
        audio_vocab = weights["_audio_vocab"]

        depth_config = ModelConfig(
            model_type="personaplex_depth",
            vocab_size=audio_vocab,
            hidden_size=depth_hidden,
            num_hidden_layers=depth_num_layers,
            num_attention_heads=depth_num_heads,
            num_key_value_heads=depth_num_heads,
            intermediate_size=depth_intermediate,
            rms_norm_eps=1e-8,
            _head_dim=depth_head_dim,
            raw={"hidden_size": depth_hidden,
                 "num_attention_heads": depth_num_heads,
                 "head_dim": depth_head_dim},
        )

        depth_cache_len = num_codebooks + 2

        for cb in range(num_codebooks):
            cb_prefix = f"depth_cb{cb}."
            depth_weights = WeightDict()
            for key, val in weights.items():
                if key.startswith(cb_prefix):
                    new_key = key[len(cb_prefix):]
                    depth_weights[new_key] = val

            if verbose:
                print(f"[trtmc build] Building depth engine for codebook {cb} "
                      f"({len(depth_weights)} weights)", file=sys.stderr)

            with timed_trt_compile(
                build_timing, f"extra_speech_depth_codebook_{cb:02d}"
            ):
                depth_plan = build_standard_decoder_engine(
                    depth_config, depth_weights, depth_cache_len,
                    precision=depth_precision,
                    norm_type="rmsnorm",
                    mlp_type="swiglu",
                    position_type="learned",
                    embed_input=True,
                    verbose=verbose)

            # Section name: depth_engine_plan_0 ... depth_engine_plan_15
            extras[f"depth_engine_plan_{cb}"] = depth_plan

        # --- Per-codebook temporal-to-depth projection matrices ---
        # Store all depformer_in.{cb}.weight [depth_hidden, temporal_hidden] for C++.
        # Concatenated into one blob: 16 x [depth_hidden * temporal_hidden] float32.
        proj_parts = []
        for cb in range(num_codebooks):
            key = f"depformer_in.{cb}"
            if key in weights:
                proj = weights[key].astype(np.float32)
                proj_parts.append(proj)
            else:
                break

        if proj_parts:
            all_proj = np.stack(proj_parts, axis=0)  # [num_cb, depth_hidden, temporal_hidden]
            extras["depth_projection"] = all_proj.tobytes()
            if verbose:
                print(f"[trtmc build] depth_projection: {all_proj.shape} "
                      f"({all_proj.nbytes} bytes)", file=sys.stderr)

        # --- Per-codebook audio embedding tables for temporal input ---
        # The temporal transformer input is the SUM of per-codebook embeddings.
        # emb.{0-15}.weight: [audio_vocab, temporal_hidden] = [2049, 4096]
        # Concatenated: [num_codebooks, audio_vocab, temporal_hidden] as float32.
        audio_vocab = weights["_audio_vocab"]
        emb_parts = []
        for cb in range(num_codebooks):
            key = f"audio_emb.{cb}"
            if key in weights:
                emb_parts.append(weights[key].astype(np.float32))
            else:
                break

        if emb_parts:
            all_emb = np.stack(emb_parts, axis=0)  # [num_cb, audio_vocab, temporal_hidden]
            extras["audio_embeddings"] = all_emb.tobytes()
            if verbose:
                print(f"[trtmc build] audio_embeddings: {all_emb.shape} "
                      f"({all_emb.nbytes / (1024*1024):.1f} MB)", file=sys.stderr)

        # --- Temporal text embedding table (text_emb.weight) ---
        # The official Moshi code adds text_emb(text_token) at every temporal
        # step: input = text_emb(text_token) + sum(emb[cb](audio[cb])).
        # During generation, text_token = padding_id (3).
        # Shape: [text_vocab, temporal_hidden] = [32001, 4096] as float32.
        if "text_emb" in weights:
            text_emb = weights["text_emb"].astype(np.float32)
            extras["temporal_text_embedding"] = text_emb.tobytes()
            if verbose:
                print(f"[trtmc build] temporal_text_embedding: {text_emb.shape} "
                      f"({text_emb.nbytes / (1024*1024):.1f} MB)", file=sys.stderr)

        # --- Depth text embedding table (depformer_text_emb) ---
        # The depth decoder needs this at position 0 (text token step).
        # Shape: [text_vocab, depth_hidden] = [32001, 1024] as float32.
        if "depformer_text_emb" in weights:
            depth_text_emb = weights["depformer_text_emb"].astype(np.float32)
            extras["depth_text_embedding"] = depth_text_emb.tobytes()
            if verbose:
                print(f"[trtmc build] depth_text_embedding: {depth_text_emb.shape} "
                      f"({depth_text_emb.nbytes / (1024*1024):.1f} MB)", file=sys.stderr)

        # --- Depth per-codebook audio embedding tables (depformer_emb) ---
        # The depth decoder uses depformer_emb.{i} at position i+1 to embed
        # the previous step's token. Shape: [num_emb, audio_vocab, depth_hidden].
        num_depformer_emb = weights.get("_num_depformer_emb", 0)
        dep_emb_parts = []
        for i in range(num_depformer_emb):
            key = f"depformer_emb.{i}"
            if key in weights:
                dep_emb_parts.append(weights[key].astype(np.float32))
            else:
                break
        if dep_emb_parts:
            all_dep_emb = np.stack(dep_emb_parts, axis=0)
            extras["depth_audio_embeddings"] = all_dep_emb.tobytes()
            if verbose:
                print(f"[trtmc build] depth_audio_embeddings: {all_dep_emb.shape} "
                      f"({all_dep_emb.nbytes / (1024*1024):.1f} MB)", file=sys.stderr)

        # --- Mimi Encoder engine (placeholder) ---
        with timed_trt_compile(build_timing, "extra_mimi_audio_encoder"):
            mimi_enc_plan = _build_mimi_encoder_engine(
                weights, precision=mimi_encoder_precision, verbose=verbose,
                model_dir=config.raw.get("_model_dir"))
        if mimi_enc_plan is not None:
            extras["mimi_encoder_plan"] = mimi_enc_plan

        # --- Mimi Decoder engine ---
        # Mimi uses 8 codebooks (1 semantic + 7 acoustic), matching the
        # official PersonaPlex code which calls model.set_num_codebooks(8)
        # and decodes only tokens[:, 1:9] (the first 8 audio codebooks,
        # i.e., the moshi stream).
        mimi_dec_codebooks = 8  # Mimi native: 1 semantic + 7 acoustic
        with timed_trt_compile(build_timing, "extra_mimi_audio_decoder"):
            mimi_dec_plan = _build_mimi_decoder_engine(
                weights, precision=mimi_decoder_precision, verbose=verbose,
                num_input_codebooks=mimi_dec_codebooks,
                num_frames=320, model_dir=config.raw.get("_model_dir"))
        if mimi_dec_plan is not None:
            extras["mimi_decoder_plan"] = mimi_dec_plan

        return extras

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        """Inject speech-to-speech config fields into bundle config.json."""
        # These will be overridden with real values from inferred config
        # since config.json is minimal. The engine_builder passes config
        # which is parsed from config.json. We use safe defaults here.
        # ropefix2 default (validated during bring-up):
        # - deterministic depth decoding (greedy)
        # - no injected text/system prompt
        prompt = ""
        return {
            "num_codebooks": 16,
            "codebook_size": 2048,
            "sample_rate": 24000,
            "frame_rate": 12.5,
            "depth_num_layers": 6,
            "depth_hidden_size": 1024,
            "depth_num_attention_heads": 16,
            "depth_num_key_value_heads": 16,
            # Override hidden_size/num_layers from the minimal config
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "eos_token_id": 2,
            # Delay pattern: delays[k] for k=0..16 (1 text + 16 audio)
            # Official Moshi: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
            "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            "text_initial_token_id": 32000,
            "audio_initial_token_id": 2048,
            "text_padding_id": 3,
            # ropefix2 decode policy: greedy + no sampling.
            "speech_depth_temperature": 0.0,
            "speech_depth_top_k": 0,
            # ropefix2 prompt policy: no prompt injection by default.
            "speech_system_prompt": prompt,
            "speech_text_prompt_ids": [],
        }

def _load_temporal_weights(
    weights: WeightDict,
    readers,
    inferred: dict,
) -> None:
    """Load temporal transformer weights in standard decoder format.

    Maps the Moshi weight naming to the standard decoder convention:
      - Splits fused in_proj_weight [3*hidden, hidden] into Q, K, V
      - Splits fused gating.linear_in [2*intermediate, hidden] into gate, up
      - Transposes all projections to [in, out] for TRT matmul
      - Flattens norm alpha from [1,1,hidden] to [hidden]
    """
    hidden = inferred["hidden_size"]
    num_layers = inferred["num_hidden_layers"]
    intermediate_size = inferred["intermediate_size"]

    # For the temporal transformer's text embedding:
    # Use text_emb as the token embedding (vocab -> hidden lookup)
    text_emb = _load_tensor(readers, "text_emb.weight")
    weights["temporal.embedding"] = text_emb.astype(np.float32)

    # Learned position embedding (zeros placeholder -- Moshi has no positional encoding)
    # The standard decoder builder requires position_embedding for position_type="learned"
    max_pos = 8192  # reasonable max
    weights["temporal.position_embedding"] = np.zeros(
        (max_pos, hidden), dtype=np.float32)

    attention_size = hidden  # Q,K,V are all hidden-dim

    for i in range(num_layers):
        hf_prefix = f"transformer.layers.{i}"
        out_prefix = f"temporal.layer.{i}"

        # --- Pre-attention norm (norm1.alpha) ---
        norm1 = _load_tensor(readers, f"{hf_prefix}.norm1.alpha")
        weights[f"{out_prefix}.input_norm"] = norm1.flatten().astype(np.float32)

        # --- Post-attention norm (norm2.alpha) ---
        norm2 = _load_tensor(readers, f"{hf_prefix}.norm2.alpha")
        weights[f"{out_prefix}.post_attn_norm"] = norm2.flatten().astype(np.float32)

        # --- Attention: split fused in_proj_weight [3*hidden, hidden] -> Q, K, V ---
        in_proj = _load_tensor(readers, f"{hf_prefix}.self_attn.in_proj_weight")
        # in_proj is [3*hidden, hidden] in row-major
        q_raw = in_proj[:hidden, :]          # [hidden, hidden]
        k_raw = in_proj[hidden:2*hidden, :]  # [hidden, hidden]
        v_raw = in_proj[2*hidden:, :]        # [hidden, hidden]

        # Transpose [out, in] -> [in, out] for TRT matmul
        weights[f"{out_prefix}.w_q"] = _transpose_2d(q_raw, "q_proj")
        weights[f"{out_prefix}.w_k"] = _transpose_2d(k_raw, "k_proj")
        weights[f"{out_prefix}.w_v"] = _transpose_2d(v_raw, "v_proj")

        # --- Output projection ---
        out_proj = _load_tensor(readers, f"{hf_prefix}.self_attn.out_proj.weight")
        weights[f"{out_prefix}.w_o"] = _transpose_2d(out_proj, "o_proj")

        # --- MLP: split fused gating.linear_in [2*intermediate, hidden] -> gate, up ---
        gating_in = _load_tensor(readers, f"{hf_prefix}.gating.linear_in.weight")
        # gating_in is [2*intermediate, hidden]
        gate_raw = gating_in[:intermediate_size, :]  # [intermediate, hidden]
        up_raw = gating_in[intermediate_size:, :]     # [intermediate, hidden]

        weights[f"{out_prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
        weights[f"{out_prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")

        # --- Down projection: gating.linear_out [hidden, intermediate] ---
        gating_out = _load_tensor(readers, f"{hf_prefix}.gating.linear_out.weight")
        weights[f"{out_prefix}.w_down"] = _transpose_2d(gating_out, "down_proj")

    # --- Final norm (out_norm.alpha) ---
    out_norm = _load_tensor(readers, "out_norm.alpha")
    weights["temporal.final_norm"] = out_norm.flatten().astype(np.float32)

    # --- LM head: text_linear.weight [text_out_vocab, hidden] ---
    text_linear = _load_tensor(readers, "text_linear.weight")
    weights["temporal.w_out"] = _transpose_2d(text_linear, "text_linear")

    # Store attention/MLP sizes
    weights["temporal._attention_size"] = attention_size
    weights["temporal._mlp_size"] = intermediate_size


def _load_depth_weights(
    weights: WeightDict,
    readers,
    inferred: dict,
) -> None:
    """Load depth transformer weights for ALL codebooks (0 through num_codebooks-1).

    The depth transformer has per-codebook gating MLPs, per-codebook attention
    (stacked in fused in_proj_weight / out_proj.weight), per-codebook embeddings,
    and per-codebook output heads. Norms are shared across codebooks.

    Each codebook's weights are stored as 'depth_cb{cb}.*' in standard decoder
    format so we can build one TRT engine per codebook.
    """
    depth_hidden = inferred["depth_hidden"]
    depth_num_layers = inferred["depth_num_layers"]
    depth_intermediate = inferred["depth_intermediate"]
    num_codebooks = inferred["num_codebooks"]
    num_depformer_emb = inferred["num_depformer_emb"]

    max_pos = 256  # position embedding placeholder

    for cb in range(num_codebooks):
        prefix = f"depth_cb{cb}"

        # Embedding table for each per-codebook depth engine.
        #
        # Moshi depth autoregressive semantics:
        #   cb=0: input comes from depformer_in.0 @ temporal_hidden (via input_embed)
        #         -> its embedding table is unused at runtime, use depformer_emb.0 as placeholder
        #   cb>0: input is the previous codebook's generated token, looked up via
        #         depformer_emb.{cb-1}  (i.e., cb=1 uses depformer_emb.0, etc.)
        #
        # There are 15 depformer_emb entries (indices 0..14) for 16 codebooks.
        if cb == 0:
            # cb=0's embedding table is never used (input_embed overrides it),
            # but we still need a valid table for the engine to build.
            emb_idx = 0
        else:
            emb_idx = cb - 1  # cb=1 -> depformer_emb.0, ..., cb=15 -> depformer_emb.14
        emb_idx = min(emb_idx, num_depformer_emb - 1)  # clamp to valid range
        dep_emb = _load_tensor(readers, f"depformer_emb.{emb_idx}.weight")
        weights[f"{prefix}.embedding"] = dep_emb.astype(np.float32)

        # Position embedding placeholder (zeros -- Moshi depth has no positional encoding)
        weights[f"{prefix}.position_embedding"] = np.zeros(
            (max_pos, depth_hidden), dtype=np.float32)

        for i in range(depth_num_layers):
            hf_prefix = f"depformer.layers.{i}"
            out_prefix = f"{prefix}.layer.{i}"

            # --- Pre-attention norm (shared across codebooks) ---
            norm1 = _load_tensor(readers, f"{hf_prefix}.norm1.alpha")
            weights[f"{out_prefix}.input_norm"] = norm1.flatten().astype(np.float32)

            # --- Post-attention norm (shared across codebooks) ---
            norm2 = _load_tensor(readers, f"{hf_prefix}.norm2.alpha")
            weights[f"{out_prefix}.post_attn_norm"] = norm2.flatten().astype(np.float32)

            # --- Attention: extract codebook cb's Q,K,V from fused in_proj_weight ---
            # in_proj_weight is [num_codebooks * 3 * depth_hidden, depth_hidden]
            in_proj = _load_tensor(readers, f"{hf_prefix}.self_attn.in_proj_weight")
            cb_offset = cb * 3 * depth_hidden
            cb_qkv = in_proj[cb_offset:cb_offset + 3 * depth_hidden, :]
            q_raw = cb_qkv[:depth_hidden, :]
            k_raw = cb_qkv[depth_hidden:2 * depth_hidden, :]
            v_raw = cb_qkv[2 * depth_hidden:, :]

            weights[f"{out_prefix}.w_q"] = _transpose_2d(q_raw, "dep_q")
            weights[f"{out_prefix}.w_k"] = _transpose_2d(k_raw, "dep_k")
            weights[f"{out_prefix}.w_v"] = _transpose_2d(v_raw, "dep_v")

            # --- Output projection: extract codebook cb's slice ---
            out_proj = _load_tensor(readers, f"{hf_prefix}.self_attn.out_proj.weight")
            cb_o_offset = cb * depth_hidden
            cb_out = out_proj[cb_o_offset:cb_o_offset + depth_hidden, :]
            weights[f"{out_prefix}.w_o"] = _transpose_2d(cb_out, "dep_o")

            # --- MLP: codebook cb's gating ---
            gating_in = _load_tensor(
                readers, f"{hf_prefix}.gating.{cb}.linear_in.weight")
            gate_raw = gating_in[:depth_intermediate, :]
            up_raw = gating_in[depth_intermediate:, :]

            weights[f"{out_prefix}.w_gate"] = _transpose_2d(gate_raw, "dep_gate")
            weights[f"{out_prefix}.w_up"] = _transpose_2d(up_raw, "dep_up")

            gating_out = _load_tensor(
                readers, f"{hf_prefix}.gating.{cb}.linear_out.weight")
            weights[f"{out_prefix}.w_down"] = _transpose_2d(gating_out, "dep_down")

        # The official depformer applies its output head directly to the last
        # residual stream. Omitting final_norm is intentional: RMSNorm with an
        # all-ones scale would still change that stream.

        # Output head: linears.{cb}.weight [codebook_size, depth_hidden]
        lin = _load_tensor(readers, f"linears.{cb}.weight")
        weights[f"{prefix}.w_out"] = _transpose_2d(lin, "dep_lm_head")

        # Store sizes
        weights[f"{prefix}._attention_size"] = depth_hidden
        weights[f"{prefix}._mlp_size"] = depth_intermediate


def _load_embedding_weights(
    weights: WeightDict,
    readers,
    inferred: dict,
) -> None:
    """Load all per-codebook embedding and projection weights."""
    num_codebooks = inferred["num_codebooks"]
    num_depformer_emb = inferred["num_depformer_emb"]

    # Per-codebook audio embeddings for temporal transformer
    for cb in range(num_codebooks):
        key = f"emb.{cb}.weight"
        if _has_tensor(readers, key):
            weights[f"audio_emb.{cb}"] = _load_tensor(
                readers, key).astype(np.float32)

    # Per-codebook embeddings for depth transformer
    for cb in range(num_depformer_emb):
        key = f"depformer_emb.{cb}.weight"
        if _has_tensor(readers, key):
            weights[f"depformer_emb.{cb}"] = _load_tensor(
                readers, key).astype(np.float32)

    # Temporal->depth projection weights
    for cb in range(num_codebooks):
        key = f"depformer_in.{cb}.weight"
        if _has_tensor(readers, key):
            weights[f"depformer_in.{cb}"] = _load_tensor(
                readers, key).astype(np.float32)

    # Depth text embedding
    if _has_tensor(readers, "depformer_text_emb.weight"):
        weights["depformer_text_emb"] = _load_tensor(
            readers, "depformer_text_emb.weight").astype(np.float32)

    # Per-codebook output heads
    for cb in range(num_codebooks):
        key = f"linears.{cb}.weight"
        if _has_tensor(readers, key):
            weights[f"output_head.{cb}"] = _load_tensor(
                readers, key).astype(np.float32)

    # Text embedding and output for temporal
    if _has_tensor(readers, "text_emb.weight"):
        weights["text_emb"] = _load_tensor(
            readers, "text_emb.weight").astype(np.float32)

    if _has_tensor(readers, "text_linear.weight"):
        weights["text_linear"] = _load_tensor(
            readers, "text_linear.weight").astype(np.float32)


_PERSONAPLEX_MIMI_FILENAME = "tokenizer-e351c8d8-checkpoint125.safetensors"


def _translate_personaplex_mimi_weights(
    source: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map the Moshi-native PersonaPlex Mimi checkpoint to TRT Mimi names."""
    mapped: dict[str, np.ndarray] = {}

    for key, value in source.items():
        target = None
        if key.startswith("encoder.model."):
            target = key.replace("encoder.model.", "encoder.layers.", 1)
            target = target.replace(".conv.conv.", ".conv.")
        elif key.startswith("decoder.model."):
            target = key.replace("decoder.model.", "decoder.layers.", 1)
            target = target.replace(".conv.conv.", ".conv.")
            target = target.replace(".convtr.convtr.", ".conv.")
        elif key == "downsample.conv.conv.conv.weight":
            target = "downsample.conv.weight"
        elif key == "upsample.convtr.convtr.convtr.weight":
            target = "upsample.conv.weight"
        elif key.startswith("quantizer.rvq_first."):
            target = key.replace(
                "quantizer.rvq_first.",
                "quantizer.semantic_residual_vector_quantizer.", 1)
        elif key.startswith("quantizer.rvq_rest."):
            target = key.replace(
                "quantizer.rvq_rest.",
                "quantizer.acoustic_residual_vector_quantizer.", 1)

        if target is not None:
            target = target.replace(".vq.layers.", ".layers.")
            target = target.replace(
                "._codebook.embedding_sum", ".codebook.embed_sum")
            target = target.replace(
                "._codebook.cluster_usage", ".codebook.cluster_usage")
            target = target.replace(
                "._codebook._initialized", ".codebook.initialized")
            mapped[target] = value

    layer_pattern = re.compile(
        r"^(encoder|decoder)_transformer\.transformer\.layers\.(\d+)\.")
    layer_prefixes = {
        match.group(0)[:-1]
        for key in source
        if (match := layer_pattern.match(key)) is not None
    }
    for source_prefix in sorted(layer_prefixes):
        target_prefix = source_prefix.replace(
            ".transformer.layers.", ".layers.")
        direct_suffixes = {
            "norm1.weight": "input_layernorm.weight",
            "norm1.bias": "input_layernorm.bias",
            "norm2.weight": "post_attention_layernorm.weight",
            "norm2.bias": "post_attention_layernorm.bias",
            "linear1.weight": "mlp.fc1.weight",
            "linear2.weight": "mlp.fc2.weight",
            "layer_scale_1.scale": "self_attn_layer_scale.scale",
            "layer_scale_2.scale": "mlp_layer_scale.scale",
            "self_attn.out_proj.weight": "self_attn.o_proj.weight",
        }
        for source_suffix, target_suffix in direct_suffixes.items():
            mapped[f"{target_prefix}.{target_suffix}"] = source[
                f"{source_prefix}.{source_suffix}"]

        fused_qkv = source[f"{source_prefix}.self_attn.in_proj_weight"]
        q_proj, k_proj, v_proj = np.split(fused_qkv, 3, axis=0)
        mapped[f"{target_prefix}.self_attn.q_proj.weight"] = q_proj
        mapped[f"{target_prefix}.self_attn.k_proj.weight"] = k_proj
        mapped[f"{target_prefix}.self_attn.v_proj.weight"] = v_proj

    return mapped


def _load_mimi_weights(model_dir: str | Path | None = None):
    """Load the checkpoint-owned Mimi codec, with kyutai/mimi as fallback.

    Returns a dict mapping weight name -> numpy array.
    Codebook embeddings are computed as embed_sum / cluster_usage.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import json

    cfg_path = hf_hub_download("kyutai/mimi", "config.json")
    with open(cfg_path) as f:
        mimi_cfg = json.load(f)

    sf_path = None
    if model_dir is not None:
        candidate = Path(model_dir) / _PERSONAPLEX_MIMI_FILENAME
        if candidate.is_file():
            sf_path = str(candidate)
    if sf_path is None:
        sf_path = hf_hub_download("kyutai/mimi", "model.safetensors")

    mimi_weights = {}
    with safe_open(sf_path, framework="pt") as sf:
        for key in sf.keys():
            mimi_weights[key] = sf.get_tensor(key).numpy().astype(np.float32)
    if Path(sf_path).name == _PERSONAPLEX_MIMI_FILENAME:
        mimi_weights = _translate_personaplex_mimi_weights(mimi_weights)

    # Compute actual codebook embeddings from embed_sum / cluster_usage
    for prefix in ["quantizer.semantic_residual_vector_quantizer",
                    "quantizer.acoustic_residual_vector_quantizer"]:
        i = 0
        while f"{prefix}.layers.{i}.codebook.embed_sum" in mimi_weights:
            embed_sum = mimi_weights[f"{prefix}.layers.{i}.codebook.embed_sum"]
            usage = mimi_weights[f"{prefix}.layers.{i}.codebook.cluster_usage"]
            codebook = embed_sum / np.maximum(usage[:, None], 1e-8)
            mimi_weights[f"{prefix}.layers.{i}.codebook.embedding"] = codebook
            i += 1

    return mimi_weights, mimi_cfg


def _mimi_causal_pad_total(kernel_size, stride, dilation=1):
    """Compute total causal padding for MimiConv1d."""
    eff_kernel = (kernel_size - 1) * dilation + 1
    return eff_kernel - stride


def _mimi_extra_padding(input_length, kernel_size, stride, padding_total):
    """Compute extra right-padding to match HF MimiConv1d ceil-mode output.

    HF computes: n_frames = ceil((length - kernel + padding_total) / stride + 1)
    Then pads right so that exactly n_frames output frames are produced.
    Without this, strided convolutions lose the fractional last frame.
    """
    import math
    n_frames = math.ceil((input_length - kernel_size + padding_total) / stride + 1) - 1
    ideal_length = n_frames * stride + kernel_size - padding_total
    return max(0, ideal_length - input_length)


def _add_mimi_conv1d_causal(network, inp, weight, bias, out_channels,
                             kernel_size, stride=1, dilation=1,
                             dtype=np.float32):
    """Causal Conv1d matching HF MimiConv1d (left-pad + extra right-pad).

    HF MimiConv1d adds:
      - padding_total on the LEFT (causal)
      - extra_padding on the RIGHT (to ensure ceil-mode frame count)
    """
    pad_total = _mimi_causal_pad_total(kernel_size, stride, dilation)
    input_length = inp.shape[2]
    extra_pad = _mimi_extra_padding(input_length, kernel_size, stride, pad_total)

    # Apply both left (causal) and right (extra) padding
    if pad_total > 0 or extra_pad > 0:
        if extra_pad > 0:
            # Need both left and right padding
            inp = graph_ops.add_reflect_pad_1d(network, inp, pad_total, extra_pad)
        else:
            # Only left padding needed
            inp = graph_ops.add_causal_pad_1d(network, inp, pad_total)

    return graph_ops.add_conv1d(
        network, inp, weight, bias, out_channels, kernel_size,
        stride=stride, padding=0, dtype=dtype)


def _add_mimi_resblock(network, inp, dim, compress, conv1_w, conv1_b,
                        conv2_w, conv2_b, residual_kernel_size=3,
                        dtype=np.float32):
    """Mimi residual block: ELU -> Conv1d(dim, dim//compress, residual_kernel_size)
                            -> ELU -> Conv1d(dim//compress, dim, 1) + skip."""
    hidden = dim // compress
    residual = inp
    x = graph_ops.add_elu(network, inp)
    x = _add_mimi_conv1d_causal(
        network, x, conv1_w, conv1_b, hidden, residual_kernel_size,
        dtype=dtype)
    x = graph_ops.add_elu(network, x)
    x = _add_mimi_conv1d_causal(
        network, x, conv2_w, conv2_b, dim, 1, dtype=dtype)
    # Skip connection (identity shortcut, use_conv_shortcut=False)
    out = network.add_elementwise(
        residual, x, trt.ElementWiseOperation.SUM)
    return out.get_output(0)


def _add_mimi_transformer_layer(network, hidden, seq_len, hidden_size,
                                 num_heads, head_dim, intermediate_size,
                                 norm_eps, layer_weights, dtype=np.float32):
    """Single Mimi transformer layer: LayerNorm + Attn + LayerScale + Residual
                                      + LayerNorm + MLP + LayerScale + Residual.

    Uses full self-attention with RoPE (single-pass, no KV cache).
    """
    # Epsilon constant for LayerNorm
    eps_const = graph_ops.add_constant(
        network, (1, 1), np.array([[norm_eps]], dtype=dtype), dtype=dtype)

    # Pre-attention LayerNorm
    residual = hidden
    hidden = graph_ops.add_layer_norm(
        network, hidden, hidden_size,
        layer_weights["input_layernorm.weight"],
        layer_weights["input_layernorm.bias"],
        eps_const, dtype=dtype)

    # Self-attention with RoPE
    cos_table = layer_weights["_cos_table"]
    sin_table = layer_weights["_sin_table"]
    hidden = graph_ops.add_self_attention_block_with_rope(
        network, hidden,
        layer_weights["self_attn.q_proj.weight"],
        layer_weights["self_attn.k_proj.weight"],
        layer_weights["self_attn.v_proj.weight"],
        layer_weights["self_attn.o_proj.weight"],
        hidden_size, num_heads, seq_len,
        cos_table, sin_table, dtype=dtype, causal=True,
        interleaved_rope=True)

    # LayerScale: element-wise multiply by learned scale
    scale = graph_ops.add_constant(
        network, (1, hidden_size),
        layer_weights["self_attn_layer_scale.scale"].reshape(1, -1),
        dtype=dtype)
    hidden = network.add_elementwise(
        hidden, scale, trt.ElementWiseOperation.PROD).get_output(0)

    # Residual
    hidden = network.add_elementwise(
        residual, hidden, trt.ElementWiseOperation.SUM).get_output(0)

    # Post-attention LayerNorm
    residual = hidden
    hidden = graph_ops.add_layer_norm(
        network, hidden, hidden_size,
        layer_weights["post_attention_layernorm.weight"],
        layer_weights["post_attention_layernorm.bias"],
        eps_const, dtype=dtype)

    # MLP: fc1 -> GELU -> fc2
    fc1_out = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, intermediate_size,
        layer_weights["mlp.fc1.weight"], dtype=dtype)
    fc1_out = graph_ops.add_gelu_erf(network, fc1_out, dtype=dtype)
    mlp_out = graph_ops.add_matmul_rhs_constant(
        network, fc1_out, intermediate_size, hidden_size,
        layer_weights["mlp.fc2.weight"], dtype=dtype)

    # MLP LayerScale
    mlp_scale = graph_ops.add_constant(
        network, (1, hidden_size),
        layer_weights["mlp_layer_scale.scale"].reshape(1, -1),
        dtype=dtype)
    mlp_out = network.add_elementwise(
        mlp_out, mlp_scale, trt.ElementWiseOperation.PROD).get_output(0)

    # Residual
    hidden = network.add_elementwise(
        residual, mlp_out, trt.ElementWiseOperation.SUM).get_output(0)

    return hidden


def _build_mimi_rope_tables(seq_len, head_dim, base=10000.0):
    """Build RoPE cos/sin tables for Mimi transformer."""
    # Standard RoPE: each position gets cos/sin for each dimension pair
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # [seq_len, head_dim/2]
    # Paired halves pattern matching HF rotate_half: cat((freqs, freqs), dim=-1)
    cos_table = np.cos(freqs)
    sin_table = np.sin(freqs)
    # Expand to full head_dim: concatenate halves (NOT interleaved repeat)
    cos_full = np.concatenate([cos_table, cos_table], axis=1)  # [seq_len, head_dim]
    sin_full = np.concatenate([sin_table, sin_table], axis=1)
    return cos_full, sin_full


def _build_mimi_encoder_engine(
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    model_dir: str | Path | None = None,
) -> bytes | None:
    """Build Mimi encoder as a native TRT engine.

    Architecture: Conv1d encoder -> 8-layer Transformer (RoPE, GELU, LayerScale)
                  -> Downsample conv -> RVQ quantize -> codec tokens

    Input: audio_input [1, 1, num_samples] (24kHz float32)
    Output: codec_tokens [32, num_frames] (int32 codebook indices)
    """
    if trt is None or graph_ops is None:
        if verbose:
            print("[trtmc build] Skipping Mimi encoder (TRT not available)",
                  file=sys.stderr)
        return None

    print("[trtmc build] Building Mimi encoder TRT engine ...", file=sys.stderr)

    mimi_w, mimi_cfg = _load_mimi_weights(model_dir)
    work_np_dtype = np.float16 if precision == "fp16" else np.float32
    work_trt_dtype = trt.float16 if precision == "fp16" else trt.float32

    # Config
    hidden_size = mimi_cfg["hidden_size"]       # 512
    num_heads = mimi_cfg["num_attention_heads"]  # 8
    head_dim = mimi_cfg["head_dim"]              # 64
    intermediate_size = mimi_cfg["intermediate_size"]  # 2048
    num_layers = mimi_cfg["num_hidden_layers"]   # 8
    norm_eps = mimi_cfg["norm_eps"]              # 1e-5
    codebook_dim = mimi_cfg["codebook_dim"]      # 256
    codebook_size = mimi_cfg["codebook_size"]    # 2048
    rope_theta = mimi_cfg.get("rope_theta", 10000.0)
    compress = mimi_cfg.get("compress", 2)
    upsampling_ratios = mimi_cfg["upsampling_ratios"]  # [8, 6, 5, 4]
    num_filters = mimi_cfg["num_filters"]        # 64
    kernel_size = mimi_cfg["kernel_size"]        # 7
    residual_kernel_size = mimi_cfg.get("residual_kernel_size", 3)
    last_kernel_size = mimi_cfg.get("last_kernel_size", 3)

    # Input size: 100800 samples = 4.2s at 24kHz
    # Handles Recording.wav (99844 samples after 48kHz->24kHz resample, zero-padded).
    # HF Mimi uses ceil-mode causal convolutions; with the corrected extra_padding
    # this produces the same 53 frames as HF for 100800 input samples.
    num_input_samples = 100800

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(
            explicit_batch=True,
            strongly_typed=True,
        )
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Input: [1, 1, num_input_samples]
    audio_input = network.add_input(
        "audio_input", trt.float32, (1, 1, num_input_samples))

    # ===== Encoder ConvNet =====
    # Encoder structure from weights:
    # layers.0: Conv1d(1, 64, 7) -- input conv
    # layers.1: ResBlock(64, compress=2)
    # layers.3: Conv1d(64, 128, 8, stride=4) -- downsample
    # layers.4: ResBlock(128, compress=2)
    # layers.6: Conv1d(128, 256, 10, stride=5) -- downsample
    # layers.7: ResBlock(256, compress=2)
    # layers.9: Conv1d(256, 512, 12, stride=6) -- downsample
    # layers.10: ResBlock(512, compress=2)
    # layers.12: Conv1d(512, 1024, 16, stride=8) -- downsample
    # layers.14: Conv1d(1024, 512, 3) -- output conv

    # Encoder layer indices and their configs
    channels = [num_filters * (2 ** i) for i in range(len(upsampling_ratios) + 1)]
    # channels = [64, 128, 256, 512, 1024] for upsampling_ratios=[8,6,5,4], num_filters=64

    x = audio_input
    if work_trt_dtype != trt.float32:
        x = network.add_cast(x, work_trt_dtype).get_output(0)

    # Layer 0: input conv
    x = _add_mimi_conv1d_causal(
        network, x,
        mimi_w["encoder.layers.0.conv.weight"],
        mimi_w["encoder.layers.0.conv.bias"],
        channels[0], kernel_size, dtype=work_np_dtype)

    # Encoder blocks (for each upsampling ratio)
    # Layer indices: resblock at 1,4,7,10; downsample conv at 3,6,9,12
    enc_layer_map = [
        (1, 3, upsampling_ratios[3]),   # resblock(64), downsample(64->128, stride=4)
        (4, 6, upsampling_ratios[2]),   # resblock(128), downsample(128->256, stride=5)
        (7, 9, upsampling_ratios[1]),   # resblock(256), downsample(256->512, stride=6)
        (10, 12, upsampling_ratios[0]), # resblock(512), downsample(512->1024, stride=8)
    ]

    for i, (res_idx, down_idx, ratio) in enumerate(enc_layer_map):
        in_ch = channels[i]
        out_ch = channels[i + 1]

        # Residual block
        x = _add_mimi_resblock(
            network, x, in_ch, compress,
            mimi_w[f"encoder.layers.{res_idx}.block.1.conv.weight"],
            mimi_w[f"encoder.layers.{res_idx}.block.1.conv.bias"],
            mimi_w[f"encoder.layers.{res_idx}.block.3.conv.weight"],
            mimi_w[f"encoder.layers.{res_idx}.block.3.conv.bias"],
            residual_kernel_size, dtype=work_np_dtype)

        # ELU activation between resblock and downsample
        # (HF encoder has ELU at layers 2, 5, 8, 11)
        x = graph_ops.add_elu(network, x)

        # Downsample conv (stride = ratio, kernel = 2*ratio)
        ds_kernel = 2 * ratio
        x = _add_mimi_conv1d_causal(
            network, x,
            mimi_w[f"encoder.layers.{down_idx}.conv.weight"],
            mimi_w[f"encoder.layers.{down_idx}.conv.bias"],
            out_ch, ds_kernel, stride=ratio, dtype=work_np_dtype)

    # ELU before output conv (HF encoder layer 13)
    x = graph_ops.add_elu(network, x)

    # Output conv: Conv1d(1024, 512, last_kernel_size)
    x = _add_mimi_conv1d_causal(
        network, x,
        mimi_w["encoder.layers.14.conv.weight"],
        mimi_w["encoder.layers.14.conv.bias"],
        hidden_size, last_kernel_size, dtype=work_np_dtype)

    # ===== Encoder Transformer =====
    # x is [1, hidden_size, seq_len_after_conv]
    # Transformer expects [seq_len, hidden_size]
    # Transpose: [1, C, L] -> [L, C]
    enc_conv_out = x
    enc_seq_len = enc_conv_out.shape[2]

    # Reshape [1, hidden_size, enc_seq_len] -> [enc_seq_len, hidden_size]
    shuf = network.add_shuffle(enc_conv_out)
    shuf.first_transpose = trt.Permutation([0, 2, 1])  # [1, L, C]
    shuf.reshape_dims = (enc_seq_len, hidden_size)
    x = shuf.get_output(0)

    # Build RoPE tables
    cos_table, sin_table = _build_mimi_rope_tables(
        enc_seq_len, head_dim, rope_theta)
    # Tile across all heads: [seq_len, head_dim] -> [seq_len, hidden_size]
    cos_full = np.tile(cos_table, (1, num_heads)).astype(np.float32)
    sin_full = np.tile(sin_table, (1, num_heads)).astype(np.float32)

    for layer_idx in range(num_layers):
        prefix = f"encoder_transformer.layers.{layer_idx}"
        layer_w = {
            "input_layernorm.weight": mimi_w[f"{prefix}.input_layernorm.weight"],
            "input_layernorm.bias": mimi_w[f"{prefix}.input_layernorm.bias"],
            "self_attn.q_proj.weight": mimi_w[f"{prefix}.self_attn.q_proj.weight"].T.copy(),
            "self_attn.k_proj.weight": mimi_w[f"{prefix}.self_attn.k_proj.weight"].T.copy(),
            "self_attn.v_proj.weight": mimi_w[f"{prefix}.self_attn.v_proj.weight"].T.copy(),
            "self_attn.o_proj.weight": mimi_w[f"{prefix}.self_attn.o_proj.weight"].T.copy(),
            "self_attn_layer_scale.scale": mimi_w[f"{prefix}.self_attn_layer_scale.scale"],
            "post_attention_layernorm.weight": mimi_w[f"{prefix}.post_attention_layernorm.weight"],
            "post_attention_layernorm.bias": mimi_w[f"{prefix}.post_attention_layernorm.bias"],
            "mlp.fc1.weight": mimi_w[f"{prefix}.mlp.fc1.weight"].T.copy(),
            "mlp.fc2.weight": mimi_w[f"{prefix}.mlp.fc2.weight"].T.copy(),
            "mlp_layer_scale.scale": mimi_w[f"{prefix}.mlp_layer_scale.scale"],
            "_cos_table": cos_full,
            "_sin_table": sin_full,
        }
        x = _add_mimi_transformer_layer(
            network, x, enc_seq_len, hidden_size, num_heads, head_dim,
            intermediate_size, norm_eps, layer_w, dtype=work_np_dtype)

    # Transformer output: [enc_seq_len, hidden_size]
    # Back to [1, hidden_size, enc_seq_len]
    shuf2 = network.add_shuffle(x)
    shuf2.reshape_dims = (1, enc_seq_len, hidden_size)
    shuf2.second_transpose = trt.Permutation([0, 2, 1])  # [1, C, L]
    x = shuf2.get_output(0)

    # ===== Downsample =====
    # downsample.conv.weight: [512, 512, 4], stride=compress(=2)
    ds_w = mimi_w["downsample.conv.weight"]
    x = _add_mimi_conv1d_causal(
        network, x, ds_w, None, hidden_size, 4, stride=compress,
        dtype=work_np_dtype)

    ds_seq_len = x.shape[2]  # seq_len after downsample

    # ===== RVQ Quantization =====
    # input_proj: Conv1d(512, 256, 1) -- project to codebook dim
    # Then for each codebook: find nearest entry, subtract quantized

    # Semantic quantizer (1 codebook)
    sem_input_proj_w = mimi_w[
        "quantizer.semantic_residual_vector_quantizer.input_proj.weight"]
    # [256, 512, 1] -> Conv1d(512, 256, 1) with no bias

    # Project to codebook dim
    sem_projected = graph_ops.add_conv1d(
        network, x, sem_input_proj_w, None, codebook_dim, 1,
        dtype=work_np_dtype)

    # Quantize semantic codebook (codebook 0)
    sem_cb = mimi_w[
        "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embedding"]
    # [2048, 256]

    # Find nearest: argmax(x @ cb.T) which is equivalent to argmin(||x - cb||^2)
    # ||x - cb||^2 = ||x||^2 - 2*x*cb + ||cb||^2
    # argmin = argmax(2*x*cb - ||cb||^2) = argmax(x @ cb.T) when ||cb|| is uniform
    # More precisely: argmax(x @ cb.T - 0.5 * ||cb||^2)

    # sem_projected: [1, codebook_dim, ds_seq_len]
    # Reshape to [ds_seq_len, codebook_dim]
    sp_shuf = network.add_shuffle(sem_projected)
    sp_shuf.first_transpose = trt.Permutation([0, 2, 1])  # [1, L, D]
    sp_shuf.reshape_dims = (ds_seq_len, codebook_dim)
    sp_flat = sp_shuf.get_output(0)

    # Codebook norms: [1, 2048]
    cb_norms_sq = np.sum(sem_cb ** 2, axis=1, keepdims=True).T  # [1, 2048]

    # Similarity: [L, D] @ [D, 2048] = [L, 2048]
    cb_const = graph_ops.add_constant(
        network, (codebook_dim, codebook_size), sem_cb.T.copy(),
        dtype=work_np_dtype)
    sim = network.add_matrix_multiply(
        sp_flat, trt.MatrixOperation.NONE,
        cb_const, trt.MatrixOperation.NONE).get_output(0)

    # Subtract 0.5 * ||cb||^2
    half_norms = graph_ops.add_constant(
        network, (1, codebook_size), (-0.5 * cb_norms_sq),
        dtype=work_np_dtype)
    sim = network.add_elementwise(
        sim, half_norms, trt.ElementWiseOperation.SUM).get_output(0)

    # Argmax: [L, 2048] -> [L, 1]
    sem_topk = network.add_topk(sim, trt.TopKOperation.MAX, 1, 1 << 1)
    sem_indices = sem_topk.get_output(1)  # [L, 1] int32

    # Collect all codebook indices: will be [32, L]
    all_indices = [sem_indices]  # Start with semantic

    # For the residual: we need to subtract the quantized embedding
    # quantized = gather(codebook, indices)
    # But for subsequent RVQ layers, the input is the residual
    # This is complex in TRT; we do it iteratively

    # Acoustic quantizer: 31 codebooks applied to the same projected input
    acou_input_proj_w = mimi_w[
        "quantizer.acoustic_residual_vector_quantizer.input_proj.weight"]
    acou_projected = graph_ops.add_conv1d(
        network, x, acou_input_proj_w, None, codebook_dim, 1,
        dtype=work_np_dtype)

    # Reshape to [ds_seq_len, codebook_dim]
    ap_shuf = network.add_shuffle(acou_projected)
    ap_shuf.first_transpose = trt.Permutation([0, 2, 1])
    ap_shuf.reshape_dims = (ds_seq_len, codebook_dim)
    acou_residual = ap_shuf.get_output(0)

    for cb_idx in range(31):
        acou_cb = mimi_w[
            f"quantizer.acoustic_residual_vector_quantizer.layers.{cb_idx}.codebook.embedding"]

        # Similarity
        acou_cb_const = graph_ops.add_constant(
            network, (codebook_dim, codebook_size), acou_cb.T.copy(),
            dtype=work_np_dtype)
        acou_cb_norms_sq = np.sum(acou_cb ** 2, axis=1, keepdims=True).T
        acou_half_norms = graph_ops.add_constant(
            network, (1, codebook_size),
            -0.5 * acou_cb_norms_sq, dtype=work_np_dtype)

        acou_sim = network.add_matrix_multiply(
            acou_residual, trt.MatrixOperation.NONE,
            acou_cb_const, trt.MatrixOperation.NONE).get_output(0)
        acou_sim = network.add_elementwise(
            acou_sim, acou_half_norms, trt.ElementWiseOperation.SUM).get_output(0)

        # Argmax
        acou_topk = network.add_topk(acou_sim, trt.TopKOperation.MAX, 1, 1 << 1)
        acou_idx = acou_topk.get_output(1)  # [L, 1]
        all_indices.append(acou_idx)

        # Subtract quantized from residual for next codebook
        # Gather: select rows from codebook by index
        # acou_cb: [2048, 256], acou_idx: [L, 1]
        # Reshape index to [L]
        idx_flat = network.add_shuffle(acou_idx)
        idx_flat.reshape_dims = (ds_seq_len,)

        acou_cb_2d = graph_ops.add_constant(
            network, (codebook_size, codebook_dim), acou_cb.copy(),
            dtype=work_np_dtype)
        gathered = network.add_gather(
            acou_cb_2d, idx_flat.get_output(0), axis=0).get_output(0)
        # gathered: [L, 256]

        acou_residual = network.add_elementwise(
            acou_residual, gathered, trt.ElementWiseOperation.SUB).get_output(0)

    # Stack all indices: each is [L, 1], concat to [L, 32], then transpose to [32, L]
    concat = network.add_concatenation(all_indices)
    concat.axis = 1  # concat along dim 1: [L, 32]
    stacked = concat.get_output(0)

    # Transpose to [32, L]
    out_shuf = network.add_shuffle(stacked)
    out_shuf.first_transpose = trt.Permutation([1, 0])
    codec_tokens = out_shuf.get_output(0)

    # Cast to float32 for output (TRT output must be float; C++ will cast to int32)
    cast_layer = network.add_cast(codec_tokens, trt.float32)
    codec_out = cast_layer.get_output(0)

    codec_out.name = "codec_tokens"
    network.mark_output(codec_out)

    if verbose:
        print(f"[trtmc build] Mimi encoder: {num_input_samples} samples -> "
              f"{ds_seq_len} frames x 32 codebooks", file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("[trtmc build] ERROR: Mimi encoder engine build failed",
              file=sys.stderr)
        return None

    plan_bytes = bytes(plan)
    print(f"[trtmc build] Mimi encoder engine built ({len(plan_bytes) / (1024*1024):.1f} MB)",
          file=sys.stderr)
    return plan_bytes


def _build_mimi_decoder_engine(
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    num_input_codebooks: int = 0,
    num_frames: int = 53,
    model_dir: str | Path | None = None,
) -> bytes | None:
    """Build Mimi decoder as a native TRT engine.

    Architecture: Codebook embedding lookup + sum (dequantize)
                  -> Upsample conv -> 8-layer Transformer -> Conv1d decoder

    If num_input_codebooks > 0, the decoder only uses that many codebooks
    (1 semantic + (num_input_codebooks-1) acoustic).  The remaining Mimi
    acoustic codebooks are skipped so they don't inject noise from index-0
    lookups.  This is critical for PersonaPlex which generates 16 codebook
    tokens per frame while Mimi natively has 32 codebooks.

    Input: codec_tokens [num_input_codebooks, num_frames] (float32, cast from int32 indices)
    Output: audio_output [1, 1, num_output_samples]
    """
    if trt is None or graph_ops is None:
        if verbose:
            print("[trtmc build] Skipping Mimi decoder (TRT not available)",
                  file=sys.stderr)
        return None

    print("[trtmc build] Building Mimi decoder TRT engine ...", file=sys.stderr)

    mimi_w, mimi_cfg = _load_mimi_weights(model_dir)
    work_np_dtype = np.float16 if precision == "fp16" else np.float32

    # Config
    hidden_size = mimi_cfg["hidden_size"]
    num_heads = mimi_cfg["num_attention_heads"]
    head_dim = mimi_cfg["head_dim"]
    intermediate_size = mimi_cfg["intermediate_size"]
    num_layers = mimi_cfg["num_hidden_layers"]
    norm_eps = mimi_cfg["norm_eps"]
    codebook_dim = mimi_cfg["codebook_dim"]
    codebook_size = mimi_cfg["codebook_size"]
    rope_theta = mimi_cfg.get("rope_theta", 10000.0)
    compress = mimi_cfg.get("compress", 2)
    upsampling_ratios = mimi_cfg["upsampling_ratios"]
    num_filters = mimi_cfg["num_filters"]
    kernel_size = mimi_cfg["kernel_size"]
    residual_kernel_size = mimi_cfg.get("residual_kernel_size", 3)
    last_kernel_size = mimi_cfg.get("last_kernel_size", 3)

    # Fixed decoder frame capacity for this engine plan.
    # Runtime can decode up to this many frames per invocation.
    num_frames = int(max(1, num_frames))

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(
            explicit_batch=True,
            strongly_typed=True,
        )
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Input: [num_codebooks, num_frames] as float32 (indices cast from int32)
    # Use num_input_codebooks if specified (e.g., 16 for PersonaPlex), else full 32
    num_codebooks = num_input_codebooks if num_input_codebooks > 0 else 32
    num_acoustic = num_codebooks - 1  # first codebook is semantic
    codec_input = network.add_input(
        "codec_tokens", trt.float32, (num_codebooks, num_frames))

    # ===== RVQ Dequantization =====
    # For each codebook, gather the embedding vector by index, then sum
    # Semantic quantizer output_proj maps back to hidden_size after dequant

    # Cast float32 indices to int32 for gather
    cast_input = network.add_cast(codec_input, trt.int32)
    indices_int = cast_input.get_output(0)

    # Semantic codebook 0: indices_int[0, :] -> gather from codebook -> sum
    sem_cb = mimi_w[
        "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embedding"]

    # Get semantic indices: row 0 of [32, num_frames] -> [num_frames]
    sem_idx_slice = network.add_slice(
        indices_int, start=(0, 0), shape=(1, num_frames), stride=(1, 1))
    sem_idx = network.add_shuffle(sem_idx_slice.get_output(0))
    sem_idx.reshape_dims = (num_frames,)

    sem_cb_const = graph_ops.add_constant(
        network, (codebook_size, codebook_dim), sem_cb.copy(),
        dtype=work_np_dtype)
    sem_gathered = network.add_gather(
        sem_cb_const, sem_idx.get_output(0), axis=0).get_output(0)
    # [num_frames, codebook_dim]

    # Apply semantic output_proj: Conv1d(256, 512, 1) -- acts as linear
    # output_proj.weight: [512, 256, 1]
    sem_output_proj = mimi_w[
        "quantizer.semantic_residual_vector_quantizer.output_proj.weight"]
    # Reshape [num_frames, 256] -> [1, 256, num_frames] for conv1d
    sem_shuf = network.add_shuffle(sem_gathered)
    sem_shuf.first_transpose = trt.Permutation([1, 0])  # [256, num_frames]
    sem_shuf.reshape_dims = (1, codebook_dim, num_frames)
    sem_conv = graph_ops.add_conv1d(
        network, sem_shuf.get_output(0),
        sem_output_proj, None, hidden_size, 1, dtype=work_np_dtype)
    # [1, hidden_size, num_frames]

    quantized = sem_conv

    # Acoustic codebooks -- sum their dequantized outputs.
    # Only use the first num_acoustic codebooks (num_codebooks - 1).
    # For full Mimi this is 31; for PersonaPlex (16 depth tokens) this is 15.
    acou_sum = None
    for cb_idx in range(num_acoustic):
        acou_cb = mimi_w[
            f"quantizer.acoustic_residual_vector_quantizer.layers.{cb_idx}.codebook.embedding"]

        # Get indices for this codebook: row (1 + cb_idx) of [num_codebooks, num_frames]
        acou_idx_slice = network.add_slice(
            indices_int,
            start=(1 + cb_idx, 0),
            shape=(1, num_frames),
            stride=(1, 1))
        acou_idx = network.add_shuffle(acou_idx_slice.get_output(0))
        acou_idx.reshape_dims = (num_frames,)

        acou_cb_const = graph_ops.add_constant(
            network, (codebook_size, codebook_dim), acou_cb.copy(),
            dtype=work_np_dtype)
        acou_gathered = network.add_gather(
            acou_cb_const, acou_idx.get_output(0), axis=0).get_output(0)
        # [num_frames, codebook_dim]

        if acou_sum is None:
            acou_sum = acou_gathered
        else:
            acou_sum = network.add_elementwise(
                acou_sum, acou_gathered,
                trt.ElementWiseOperation.SUM).get_output(0)

    # Apply acoustic output_proj: Conv1d(256, 512, 1)
    acou_output_proj = mimi_w[
        "quantizer.acoustic_residual_vector_quantizer.output_proj.weight"]
    acou_shuf = network.add_shuffle(acou_sum)
    acou_shuf.first_transpose = trt.Permutation([1, 0])
    acou_shuf.reshape_dims = (1, codebook_dim, num_frames)
    acou_conv = graph_ops.add_conv1d(
        network, acou_shuf.get_output(0),
        acou_output_proj, None, hidden_size, 1, dtype=work_np_dtype)

    # Sum semantic + acoustic
    quantized = network.add_elementwise(
        quantized, acou_conv, trt.ElementWiseOperation.SUM).get_output(0)
    # [1, hidden_size, num_frames]

    # ===== Upsample =====
    # upsample.conv.weight: [512, 1, 4] -- ConvTranspose1d(512, 512, 4, stride=2, groups=512)
    # This is a grouped transposed convolution for upsampling by compress=2
    # Weight shape [512, 1, 4] means groups=512
    up_w = mimi_w["upsample.conv.weight"]
    # ConvTranspose1d with groups: weight is [C_in, C_out/groups, K] = [512, 1, 4]
    # For TRT deconv: need [C_in, C_out, 1, K] with groups
    # With groups=512 and C_out=512: C_out/groups=1

    upsample_seq_len = num_frames
    us_n, us_c, us_l = quantized.shape

    # Reshape to [1, 512, 1, num_frames] for 2D deconv
    us_reshape = network.add_shuffle(quantized)
    us_reshape.reshape_dims = (1, hidden_size, 1, upsample_seq_len)

    # Grouped ConvTranspose1d: [C_in, C_out/groups, 1, K]
    up_w_4d = up_w.reshape(hidden_size, 1, 1, 4)
    up_trt_w = trt.Weights(np.ascontiguousarray(up_w_4d, dtype=work_np_dtype))

    deconv = network.add_deconvolution_nd(
        us_reshape.get_output(0),
        num_output_maps=hidden_size,
        kernel_shape=(1, 4),
        kernel=up_trt_w,
        bias=trt.Weights())
    deconv.stride_nd = (1, compress)
    deconv.padding_nd = (0, 0)
    deconv.num_groups = hidden_size

    # Trim: causal ConvTranspose trims right side
    # padding_total = kernel_size - stride = 4 - 2 = 2
    # trim_right_ratio = 1.0: padding_right = ceil(2 * 1.0) = 2
    # padding_left = 2 - 2 = 0
    # So trim 0 from left, 2 from right
    deconv_out = deconv.get_output(0)
    deconv_len = deconv_out.shape[3]  # output length of deconv

    # Trim right by 2
    trimmed = network.add_slice(
        deconv_out,
        start=(0, 0, 0, 0),
        shape=(1, hidden_size, 1, deconv_len - 2),
        stride=(1, 1, 1, 1))

    # Reshape back to 3D: [1, hidden_size, new_len]
    up_out_len = deconv_len - 2
    us_back = network.add_shuffle(trimmed.get_output(0))
    us_back.reshape_dims = (1, hidden_size, up_out_len)
    x = us_back.get_output(0)

    # ===== Decoder Transformer =====
    dec_seq_len = up_out_len

    # Transpose to [dec_seq_len, hidden_size]
    dec_shuf = network.add_shuffle(x)
    dec_shuf.first_transpose = trt.Permutation([0, 2, 1])
    dec_shuf.reshape_dims = (dec_seq_len, hidden_size)
    x = dec_shuf.get_output(0)

    # Build RoPE tables for decoder transformer
    cos_table, sin_table = _build_mimi_rope_tables(
        dec_seq_len, head_dim, rope_theta)
    cos_full = np.tile(cos_table, (1, num_heads)).astype(np.float32)
    sin_full = np.tile(sin_table, (1, num_heads)).astype(np.float32)

    for layer_idx in range(num_layers):
        prefix = f"decoder_transformer.layers.{layer_idx}"
        layer_w = {
            "input_layernorm.weight": mimi_w[f"{prefix}.input_layernorm.weight"],
            "input_layernorm.bias": mimi_w[f"{prefix}.input_layernorm.bias"],
            "self_attn.q_proj.weight": mimi_w[f"{prefix}.self_attn.q_proj.weight"].T.copy(),
            "self_attn.k_proj.weight": mimi_w[f"{prefix}.self_attn.k_proj.weight"].T.copy(),
            "self_attn.v_proj.weight": mimi_w[f"{prefix}.self_attn.v_proj.weight"].T.copy(),
            "self_attn.o_proj.weight": mimi_w[f"{prefix}.self_attn.o_proj.weight"].T.copy(),
            "self_attn_layer_scale.scale": mimi_w[f"{prefix}.self_attn_layer_scale.scale"],
            "post_attention_layernorm.weight": mimi_w[f"{prefix}.post_attention_layernorm.weight"],
            "post_attention_layernorm.bias": mimi_w[f"{prefix}.post_attention_layernorm.bias"],
            "mlp.fc1.weight": mimi_w[f"{prefix}.mlp.fc1.weight"].T.copy(),
            "mlp.fc2.weight": mimi_w[f"{prefix}.mlp.fc2.weight"].T.copy(),
            "mlp_layer_scale.scale": mimi_w[f"{prefix}.mlp_layer_scale.scale"],
            "_cos_table": cos_full,
            "_sin_table": sin_full,
        }
        x = _add_mimi_transformer_layer(
            network, x, dec_seq_len, hidden_size, num_heads, head_dim,
            intermediate_size, norm_eps, layer_w, dtype=work_np_dtype)

    # Back to [1, hidden_size, dec_seq_len]
    dec_shuf2 = network.add_shuffle(x)
    dec_shuf2.reshape_dims = (1, dec_seq_len, hidden_size)
    dec_shuf2.second_transpose = trt.Permutation([0, 2, 1])
    x = dec_shuf2.get_output(0)

    # ===== Decoder ConvNet =====
    # Decoder structure (mirror of encoder):
    # layers.0: Conv1d(512, 1024, 7) -- input conv
    # layers.2: ConvTranspose1d(1024, 512, 16, stride=8) -- upsample
    # layers.3: ResBlock(512, compress=2)
    # layers.5: ConvTranspose1d(512, 256, 12, stride=6)
    # layers.6: ResBlock(256, compress=2)
    # layers.8: ConvTranspose1d(256, 128, 10, stride=5)
    # layers.9: ResBlock(128, compress=2)
    # layers.11: ConvTranspose1d(128, 64, 8, stride=4)
    # layers.12: ResBlock(64, compress=2)
    # layers.14: Conv1d(64, 1, 3) -- output conv

    channels = [num_filters * (2 ** i)
                for i in range(len(upsampling_ratios), -1, -1)]
    # [1024, 512, 256, 128, 64]

    # Input conv: Conv1d(512, 1024, 7) + ELU
    x = _add_mimi_conv1d_causal(
        network, x,
        mimi_w["decoder.layers.0.conv.weight"],
        mimi_w["decoder.layers.0.conv.bias"],
        channels[0], kernel_size, dtype=work_np_dtype)
    x = graph_ops.add_elu(network, x)  # decoder layer 1

    # Decoder blocks (reverse of encoder)
    # HF order: ConvTranspose -> ResBlock -> ELU -> ...
    dec_layer_map = [
        (2, 3, upsampling_ratios[0]),   # upsample 8x (1024->512), resblock(512)
        (5, 6, upsampling_ratios[1]),   # upsample 6x (512->256), resblock(256)
        (8, 9, upsampling_ratios[2]),   # upsample 5x (256->128), resblock(128)
        (11, 12, upsampling_ratios[3]), # upsample 4x (128->64), resblock(64)
    ]

    for i, (up_idx, res_idx, ratio) in enumerate(dec_layer_map):
        out_ch = channels[i + 1]

        # ConvTranspose1d upsample
        up_kernel = 2 * ratio
        x = graph_ops.add_conv1d_transpose(
            network, x,
            mimi_w[f"decoder.layers.{up_idx}.conv.weight"],
            mimi_w[f"decoder.layers.{up_idx}.conv.bias"],
            out_ch, up_kernel, stride=ratio, padding=0,
            dtype=work_np_dtype)

        # Trim right by ratio (causal trim: padding_total = kernel - stride = ratio)
        if ratio > 0:
            x = graph_ops.add_slice_trim_right(network, x, ratio)

        # Residual block
        x = _add_mimi_resblock(
            network, x, out_ch, compress,
            mimi_w[f"decoder.layers.{res_idx}.block.1.conv.weight"],
            mimi_w[f"decoder.layers.{res_idx}.block.1.conv.bias"],
            mimi_w[f"decoder.layers.{res_idx}.block.3.conv.weight"],
            mimi_w[f"decoder.layers.{res_idx}.block.3.conv.bias"],
            residual_kernel_size, dtype=work_np_dtype)

        # ELU after resblock (decoder layers 4, 7, 10, 13)
        x = graph_ops.add_elu(network, x)

    # Output conv: Conv1d(64, 1, last_kernel_size)
    x = _add_mimi_conv1d_causal(
        network, x,
        mimi_w["decoder.layers.14.conv.weight"],
        mimi_w["decoder.layers.14.conv.bias"],
        1, last_kernel_size, dtype=work_np_dtype)

    if x.dtype != trt.float32:
        x = network.add_cast(x, trt.float32).get_output(0)

    # Output: [1, 1, num_output_samples]
    x.name = "audio_output"
    network.mark_output(x)

    output_samples = x.shape[2]
    if verbose:
        print(f"[trtmc build] Mimi decoder: {num_frames} frames x {num_codebooks} codebooks "
              f"(1 semantic + {num_acoustic} acoustic) -> "
              f"{output_samples} samples", file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("[trtmc build] ERROR: Mimi decoder engine build failed",
              file=sys.stderr)
        return None

    plan_bytes = bytes(plan)
    print(f"[trtmc build] Mimi decoder engine built ({len(plan_bytes) / (1024*1024):.1f} MB)",
          file=sys.stderr)
    return plan_bytes


plugin = PersonaPlexPlugin()
