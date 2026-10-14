# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the four native LFM2.5 audio inference engines and tokenizer assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors import safe_open

from .config import validate_dense_lfm2_config

CHECKPOINT = "LiquidAI/LFM2.5-Audio-1.5B"
REVISION = "c362a0625dfe45aa588dce5f0ada28a7e5707628"
CONTEXT_LENGTH = 32768


@dataclass(frozen=True)
class BuildRequest:
    model_dir: Path
    task: str = "offline_speech_dialogue"
    precision: str = "bf16"
    max_sequence_length: int | None = CONTEXT_LENGTH
    backend: str = "trt"
    verbose: bool = False

    def __post_init__(self):
        if self.precision != "bf16":
            raise ValueError("LFM2.5 audio requires BF16 precision")
        if self.backend != "trt":
            raise ValueError("LFM2.5 audio builds only standard TensorRT plans")
        if self.max_sequence_length is not None and (
            type(self.max_sequence_length) is not int
            or not 1 <= self.max_sequence_length <= CONTEXT_LENGTH
        ):
            raise ValueError("max_sequence_length must be between 1 and 32768")


def load_config(model_dir):
    root = Path(model_dir)
    raw = json.loads((root / "config.json").read_text())
    if raw.get("architectures") != ["Lfm2AudioForConditionalGeneration"]:
        raise ValueError("LFM audio requires Lfm2AudioForConditionalGeneration")
    if raw.get("codebooks") != 8 or raw.get("depthformer") != {
        "layers": 6,
        "dim": 1024,
        "tie": True,
    }:
        raise ValueError("LFM audio requires the eight-codebook depth transformer")
    encoder = raw["encoder"]
    expected = {
        "feat_in": 128,
        "n_layers": 17,
        "d_model": 512,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "causal_downsampling": False,
        "self_attention_model": "rel_pos",
        "n_heads": 8,
        "conv_kernel_size": 9,
        "conv_norm_type": "batch_norm",
        "att_context_size": [-1, -1],
        "xscaling": False,
        "ff_expansion_factor": 4,
        "reduction": None,
        "reduction_position": None,
        "reduction_factor": 1,
        "conv_context_size": None,
        "untie_biases": True,
    }
    for key, value in expected.items():
        if encoder.get(key) != value:
            raise ValueError(f"unsupported LFM audio encoder {key}: {encoder.get(key)!r}")
    expected_frontend = {
        "sample_rate": 16000,
        "normalize": "per_feature",
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "features": 128,
        "n_fft": 512,
        "log": True,
        "frame_splicing": 1,
        "pad_to": 0,
        "pad_value": 0.0,
    }
    for key, value in expected_frontend.items():
        if raw["preprocessor"].get(key) != value:
            raise ValueError(f"unsupported LFM audio frontend {key}")
    if raw["lfm"].get("hidden_size") != 2048 or raw["lfm"].get("vocab_size") != 65536:
        raise ValueError("LFM2.5 audio requires its 2048-wide multimodal backbone")
    detok = json.loads((root / "audio_detokenizer/config.json").read_text())
    if detok.get("architectures") != ["Lfm25AudioTokenizer"] or detok.get("sliding_window") != 30:
        raise ValueError("LFM2.5 audio requires the custom LFM audio detokenizer")
    if detok.get("hidden_size") != 512 or detok.get("output_size") != 1282:
        raise ValueError("LFM2.5 audio requires its 512-wide spectral detokenizer")
    return raw, detok


def backbone_config(raw):
    raw = dict(raw)
    raw["architectures"] = ["Lfm2ForCausalLM"]
    raw["layer_types"] = [
        "full_attention" if layer == "sliding_attention" else layer for layer in raw["layer_types"]
    ]
    return SimpleNamespace(raw=raw)


def read_weights(path):
    with safe_open(str(path), framework="numpy") as reader:
        return {
            name: np.asarray(reader.get_tensor(name), dtype=np.float32) for name in reader.keys()
        }


def mel_filters():
    """Slaney triangular filters with area normalization, matching librosa."""

    def hz_to_mel(hz):
        hz = np.asarray(hz, np.float64)
        return np.where(
            hz >= 1000, 15 + np.log(np.maximum(hz, 1) / 1000) / (np.log(6.4) / 27), hz / (200 / 3)
        )

    mel = np.linspace(hz_to_mel(0), hz_to_mel(8000), 130)
    hz = np.where(mel >= 15, 1000 * np.exp((np.log(6.4) / 27) * (mel - 15)), mel * (200 / 3))
    fft_freq = np.linspace(0, 8000, 257)
    slopes = hz[:, None] - fft_freq
    lower = -slopes[:-2] / np.diff(hz)[:-1, None]
    upper = slopes[2:] / np.diff(hz)[1:, None]
    bank = np.maximum(0, np.minimum(lower, upper)).astype(np.float32)
    bank *= (2 / (hz[2:] - hz[:-2]))[:, None]
    return bank


def build(request, writer):
    if not isinstance(request, BuildRequest):
        raise TypeError("LFM2.5 audio requires its family-owned BuildRequest")
    if request.precision != "bf16":
        raise ValueError("LFM2.5 audio currently requires the checkpoint's BF16 precision")
    if request.task not in {
        "offline_speech_dialogue",
        "speech_transcription",
        "speech_to_speech_response",
        "text_to_speech",
        "streaming_text_to_speech",
        "audio_text_to_text",
        "text_conversation",
    }:
        raise ValueError("unsupported LFM2.5 audio task")
    from .backbone import build_lfm2_engine
    from .checkpoint_mapper import load_lfm2_weights
    from .conformer import build_encoder
    from .depthformer import build_depthformer

    root = Path(request.model_dir)
    raw, detok_raw = load_config(root)
    max_context = request.max_sequence_length or CONTEXT_LENGTH
    if (
        isinstance(max_context, bool)
        or not isinstance(max_context, int)
        or not 1 <= max_context <= CONTEXT_LENGTH
    ):
        raise ValueError("max_sequence_length must be between 1 and 32768")
    lfm_config = backbone_config(raw["lfm"])
    detok_config = backbone_config(detok_raw)
    resolved = validate_dense_lfm2_config(lfm_config)
    detok_resolved = validate_dense_lfm2_config(detok_config)
    weights = read_weights(root / "model.safetensors")
    writer.set_header(family="lfm2_audio", task=request.task, backend=request.backend)
    backbone_weights = load_lfm2_weights(str(root), lfm_config)
    writer.add_bytes(
        "backbone.plan",
        build_lfm2_engine(
            lfm_config, backbone_weights, max_context, precision="bf16", verbose=request.verbose
        ),
    )
    del backbone_weights
    writer.add_bytes(
        "encoder.plan",
        build_encoder(
            weights,
            raw["encoder"],
            max(9, max_context * 8),
            precision="bf16",
            verbose=request.verbose,
        ),
    )
    writer.add_bytes(
        "depthformer.plan",
        build_depthformer(weights, precision="bf16", verbose=request.verbose, fp32_attention=True),
    )
    detok_weights = load_lfm2_weights(
        str(root / "audio_detokenizer"), detok_config, precision="fp32", head_name="lin"
    )
    # A frame produces six spectral steps. The detokenizer's attention window
    # is 30 steps; its absolute RoPE positions continue through the utterance.
    writer.add_bytes(
        "detokenizer.plan",
        build_lfm2_engine(
            detok_config,
            detok_weights,
            max_context * 6,
            precision="fp32",
            sliding_window=30,
            verbose=request.verbose,
        ),
    )
    writer.add_bytes(
        "text_embeddings.f32", weights["lfm.embed_tokens.weight"].astype("<f4").tobytes()
    )
    writer.add_bytes(
        "audio_embeddings.f32", weights["audio_embedding.embedding.weight"].astype("<f4").tobytes()
    )
    depth_embeddings = np.stack(
        [weights[f"depth_embeddings.{i}.embedding.weight"] for i in range(8)]
    )
    writer.add_bytes("depth_embeddings.f32", depth_embeddings.astype("<f4").tobytes())
    detok = read_weights(root / "audio_detokenizer/model.safetensors")
    writer.add_bytes("detokenizer_embeddings.f32", detok["emb.emb.weight"].astype("<f4").tobytes())
    writer.add_bytes("mel_filters.f32", mel_filters().astype("<f4").tobytes())
    runtime = {
        "version": 1,
        "precision": "bf16",
        "max_context": max_context,
        "lfm": resolved.bundle_overrides(),
        "detokenizer": detok_resolved.bundle_overrides(),
        "interleaved_n_text": raw["interleaved_n_text"],
        "interleaved_n_audio": raw["interleaved_n_audio"],
        "input_sample_rate": 16000,
        "output_sample_rate": 24000,
    }
    writer.add_json("runtime.json", runtime)
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        writer.add_bytes(filename, (root / filename).read_bytes())
