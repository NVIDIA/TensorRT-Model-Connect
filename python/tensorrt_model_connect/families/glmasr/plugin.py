# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-ASR family plugin.

GLM-ASR pairs a Whisper-style audio encoder with an ordinary Llama decoder. A
two-layer projector maps stacked encoder frames into the decoder embedding
space, and those embeddings replace the ``audio_token_id`` positions in the
prompt. The decoder therefore runs in embed-input mode rather than reaching the
encoder through cross-attention.

The checkpoint declares ``model_type`` "glmasr". Family discovery indexes a
family id as an exact alias and resolves exact aliases before prefixes, so this
family claims the checkpoint ahead of the ``glm`` family's "glm" prefix without
any extra discriminator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tensorrt_model_connect.config import ModelConfig

from .audio_encoder_builder import build_audio_encoder_engine
from .checkpoint_mapper import (
    WeightDict,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
    _transpose_2d,
)
from .config import (
    MERGE_FACTOR,
    audio_encoder_config,
    audio_token_id,
    projector_config,
)
from .standard_decoder_builder import build_standard_decoder_engine

_MODEL_TYPES = frozenset({"glmasr", "glm_asr"})

_DECODER = "language_model.model."
_LM_HEAD = "language_model.lm_head.weight"
_AUDIO_TOWER = "audio_tower."
_PROJECTOR = "multi_modal_projector."

# The processor extracts mel features with the Whisper front-end parameters.
_MEL_DEFAULTS = {
    "n_fft": 400,
    "hop_length": 160,
    "chunk_length": 30,
    "sampling_rate": 16000,
}


class GlmAsrPlugin:
    """Build-side owner for GLM-ASR checkpoints."""

    name = "glmasr"
    runtime_strategy = "glmasr_speech_to_text"

    # Projected audio frames replace placeholder tokens in the decoder prompt.
    embed_input = True

    def matches(self, model_type: str) -> bool:
        return (model_type or "").strip().lower() in _MODEL_TYPES

    def get_audio_config(self, config: ModelConfig) -> dict | None:
        """Mel front-end contract carried in the bundle."""
        encoder = audio_encoder_config(config.raw)
        if encoder is None:
            return None
        return {
            "mel_n_fft": _MEL_DEFAULTS["n_fft"],
            "mel_hop_length": _MEL_DEFAULTS["hop_length"],
            "mel_chunk_length": _MEL_DEFAULTS["chunk_length"],
            "mel_sampling_rate": _MEL_DEFAULTS["sampling_rate"],
            "mel_num_bins": encoder.num_mel_bins,
        }

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict | None:
        """Decoder shape and the audio placeholder token, hoisted to top level.

        The checkpoint keeps the decoder under ``text_config`` and the bundle's
        config.json is written from the unmerged checkpoint config, so the
        native runtime would otherwise find no top-level layer or head counts
        and fall back to defaults, sizing the KV cache wrongly. Publishing the
        resolved decoder fields here keeps the runtime's view correct without
        disturbing the nested ``audio_config`` the builders read.
        """
        overrides = {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "max_position_embeddings": config.max_position_embeddings,
        }
        token = audio_token_id(config.raw)
        if token is not None:
            overrides["audio_token_id"] = token
        return overrides

    def load_weights(
        self, model_dir: str, config: ModelConfig, *, precision: str = "fp32",
    ) -> WeightDict:
        """Load the Llama decoder, the audio tower, and the projector.

        The decoder maps onto the shared flat weight-key convention. The audio
        tower and projector keep their checkpoint names under ``audio.`` and
        ``projector.`` namespaces, because their builders consume the HF layout
        directly.
        """
        readers = _open_safetensors(Path(model_dir))
        target_dtype = _target_np_dtype(precision)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers

        weights = WeightDict()

        embedding = _load_tensor(readers, f"{_DECODER}embed_tokens.weight")
        if embedding.shape != (vocab, hidden):
            raise ValueError(
                f"embed_tokens shape {embedding.shape} != ({vocab}, {hidden})"
            )
        weights["embedding"] = embedding.astype(target_dtype)

        attention_size = 0
        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"{_DECODER}layers.{layer_idx}"

            weights[f"{prefix}.input_norm"] = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight",
            ).astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight",
            ).astype(np.float32)

            # Llama attention: separate projections, no biases and no QK norms.
            for engine_key, hf_key in (
                ("w_q", "q_proj"), ("w_k", "k_proj"),
                ("w_v", "v_proj"), ("w_o", "o_proj"),
            ):
                raw = _load_tensor(
                    readers, f"{hf_prefix}.self_attn.{hf_key}.weight")
                weights[f"{prefix}.{engine_key}"] = _transpose_2d(
                    raw, f"{hf_prefix}.self_attn.{hf_key}", precision)
                if engine_key == "w_q":
                    attention_size = raw.shape[0]

            for engine_key, hf_key in (
                ("w_gate", "gate_proj"), ("w_up", "up_proj"),
                ("w_down", "down_proj"),
            ):
                raw = _load_tensor(readers, f"{hf_prefix}.mlp.{hf_key}.weight")
                weights[f"{prefix}.{engine_key}"] = _transpose_2d(
                    raw, f"{hf_prefix}.mlp.{hf_key}", precision)

        weights["final_norm"] = _load_tensor(
            readers, f"{_DECODER}norm.weight",
        ).astype(np.float32)
        weights["w_out"] = _transpose_2d(
            _load_tensor(readers, _LM_HEAD), "lm_head", precision)
        weights["_attention_size"] = attention_size

        encoder = audio_encoder_config(config.raw)
        weights["_audio_encoder_cfg"] = (
            {} if encoder is None else encoder.__dict__.copy()
        )
        projector = projector_config(config.raw)
        weights["_projector_cfg"] = (
            {} if projector is None else projector.__dict__.copy()
        )

        for reader in readers:
            for key in reader.keys():
                if key.startswith(_AUDIO_TOWER):
                    weights[f"audio.{key}"] = _load_tensor([reader], key)
                elif key.startswith(_PROJECTOR):
                    weights[f"projector.{key[len(_PROJECTOR):]}"] = (
                        _load_tensor([reader], key))

        return weights

    def build_engine(
        self, config: ModelConfig, weights, max_cache_length: int,
        *, precision: str = "fp32", quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        """Build the Llama decoder in embed-input mode.

        ``text_config`` is an ordinary Llama decoder: RMSNorm, SwiGLU, full
        rotary, and no attention or MLP biases. Embed-input mode is what lets
        the runtime substitute projected audio frames for the placeholder
        tokens during prefill.
        """
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
            activation="silu",
            embed_input=True,
            verbose=verbose,
        )

    def build_extra_engines(
        self, config: ModelConfig, weights, max_cache_length: int,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> dict | None:
        """Bake the mel filterbank into the bundle for the C++ front-end.

        The checkpoint's processor is a WhisperFeatureExtractor, so the filters
        are the Slaney-normalised Whisper bank at this checkpoint's bin count.
        """
        encoder = audio_encoder_config(config.raw)
        if encoder is None:
            return None
        try:
            from transformers.audio_utils import mel_filter_bank
        except ImportError:
            print(
                "[trtmc build] Warning: transformers.audio_utils not available, "
                "skipping mel filterbank embedding",
                file=sys.stderr,
            )
            return None

        num_mel_bins = encoder.num_mel_bins
        n_fft = _MEL_DEFAULTS["n_fft"]
        sampling_rate = _MEL_DEFAULTS["sampling_rate"]
        n_freq_bins = 1 + n_fft // 2

        filters = mel_filter_bank(
            num_frequency_bins=n_freq_bins,
            num_mel_filters=num_mel_bins,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        header = np.array([n_freq_bins, num_mel_bins], dtype=np.int32)
        payload = header.tobytes() + np.ascontiguousarray(
            filters, dtype=np.float32).tobytes()

        if verbose:
            print(
                f"[trtmc build] Mel filterbank: {n_freq_bins}x{num_mel_bins} "
                f"({len(payload)} bytes)",
                file=sys.stderr,
            )
        return {"mel_filterbank": payload}

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        """Build the audio encoder and its projector as one engine."""
        if audio_encoder_config(config.raw) is None:
            return None
        return build_audio_encoder_engine(
            config, weights, precision=precision, verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        """Encoder geometry the runtime needs to place audio embeddings."""
        encoder = audio_encoder_config(config.raw)
        projector = projector_config(config.raw)
        if encoder is None or projector is None:
            return None
        frames = encoder.max_position_embeddings
        return {
            "num_mel_bins": encoder.num_mel_bins,
            "mel_length": frames * 2,
            "encoder_frames": frames,
            "encoder_layers": encoder.num_hidden_layers,
            "audio_merge_factor": MERGE_FACTOR,
            "num_audio_embeddings": frames // MERGE_FACTOR,
            "audio_embedding_size": projector.out_features,
            "has_vision_engine": True,
        }


plugin = GlmAsrPlugin()
