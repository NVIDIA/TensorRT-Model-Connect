# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and check the original VibeVoice checkpoint configuration.

Every value the engines and the runtime depend on is read here and every
setting the graphs do not implement is refused by name, so a checkpoint that
differs from the implemented architecture fails at build time instead of
producing plausible audio from a different model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Token ids the generation loop is built around. They are checked against the
# tokenizer's added tokens at build time.
AUDIO_BOS_TOKEN = "<|vision_start|>"
AUDIO_EOS_TOKEN = "<|vision_end|>"
AUDIO_TOKEN = "<|vision_pad|>"
EOS_TOKEN = "<|endoftext|>"

# The prompt frame of the reference chat template.
SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)
VOICE_INPUT_HEADER = " Voice input:\n"
TEXT_INPUT_HEADER = " Text input:\n"
SPEECH_OUTPUT_HEADER = " Speech output:\n"

# The tokenizer is not part of the checkpoint; preprocessor_config.json names
# its source and this pinned revision is the one the family is validated with.
TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-1.5B"
TOKENIZER_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


@dataclass(frozen=True)
class SpeechTokenizerConfig:
    """Geometry of one causal convolutional speech tokenizer (encoder order)."""

    channels: int
    vae_dim: int
    num_filters: int
    ratios: tuple[int, ...]
    depths: tuple[int, ...]
    eps: float
    kernel_size: int = 7
    ffn_expansion: int = 4

    @property
    def hop_length(self) -> int:
        product = 1
        for ratio in self.ratios:
            product *= ratio
        return product

    def stage_channels(self) -> tuple[int, ...]:
        """Channels of the stem stage followed by each downsampling stage."""
        return tuple(self.num_filters * (2**index) for index in range(len(self.ratios) + 1))


@dataclass(frozen=True)
class VibeVoiceConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    max_position_embeddings: int
    acoustic: SpeechTokenizerConfig
    semantic: SpeechTokenizerConfig
    acoustic_vae_std: float
    head_layers: int
    head_intermediate_size: int
    head_eps: float
    latent_size: int
    ddpm_num_steps: int
    sample_rate: int
    target_dbfs: float
    normalize_eps: float
    frequency_embedding_size: int = 256
    diffusion_max_period: int = 10000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("vibevoice checkpoint: " + message)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"vibevoice checkpoint is missing {path.name}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"vibevoice {path.name} is not a JSON object")
    return value


def _speech_tokenizer(raw: dict, name: str, *, decoder: bool) -> SpeechTokenizerConfig:
    _require(isinstance(raw, dict), f"{name} is missing")
    fixed = {
        "causal": True,
        "channels": 1,
        "conv_bias": True,
        "conv_norm": "none",
        "corpus_normalize": 0.0,
        "disable_last_norm": True,
        "layernorm": "RMSNorm",
        "layernorm_elementwise_affine": True,
        "mixer_layer": "depthwise_conv",
        "pad_mode": "constant",
    }
    for key, expected in fixed.items():
        _require(raw.get(key) == expected, f"{name}.{key}={raw.get(key)!r} is not implemented")
    depths = tuple(int(item) for item in str(raw["encoder_depths"]).split("-"))
    encoder_ratios = tuple(int(item) for item in raw["encoder_ratios"])
    _require(len(depths) == len(encoder_ratios) + 1, f"{name} depths do not match its ratios")
    if decoder:
        _require(
            raw.get("decoder_depths") in (None, "-".join(str(d) for d in reversed(depths))),
            f"{name}.decoder_depths is not the mirrored encoder depths",
        )
        _require(
            tuple(raw.get("decoder_ratios") or ()) == encoder_ratios,
            f"{name}.decoder_ratios differs from its encoder ratios",
        )
        _require(
            int(raw.get("decoder_n_filters", 0)) == int(raw["encoder_n_filters"]),
            f"{name}.decoder_n_filters differs from its encoder filters",
        )
    return SpeechTokenizerConfig(
        channels=1,
        vae_dim=int(raw["vae_dim"]),
        num_filters=int(raw["encoder_n_filters"]),
        # The encoder applies the ratios in reverse of their listed order.
        ratios=tuple(reversed(encoder_ratios)),
        depths=depths,
        eps=float(raw["layernorm_eps"]),
    )


def load(model_dir: str | Path) -> VibeVoiceConfig:
    """Read config.json and preprocessor_config.json of the original layout."""
    root = Path(model_dir)
    raw = _read_json(root / "config.json")
    processor = _read_json(root / "preprocessor_config.json")
    _require(raw.get("model_type") == "vibevoice", "model_type is not vibevoice")
    _require(
        raw.get("architectures") == ["VibeVoiceForConditionalGeneration"],
        "architectures is not [VibeVoiceForConditionalGeneration]",
    )

    decoder = raw.get("decoder_config")
    _require(
        isinstance(decoder, dict) and decoder.get("model_type") == "qwen2",
        "decoder_config is not a qwen2 decoder",
    )
    _require(decoder.get("hidden_act") == "silu", "decoder_config.hidden_act is not silu")
    _require(decoder.get("rope_scaling") is None, "decoder_config.rope_scaling is not implemented")
    _require(
        not decoder.get("use_sliding_window", False),
        "decoder_config.use_sliding_window: sliding-window attention is not implemented",
    )
    _require(
        bool(decoder.get("tie_word_embeddings")),
        "decoder_config.tie_word_embeddings: untied LM heads are not implemented",
    )
    hidden = int(decoder["hidden_size"])
    heads = int(decoder["num_attention_heads"])
    kv_heads = int(decoder["num_key_value_heads"])
    _require(hidden % heads == 0 and heads % kv_heads == 0, "attention head geometry is invalid")

    head = raw.get("diffusion_head_config")
    _require(isinstance(head, dict), "diffusion_head_config is missing")
    for key, expected in {
        "diffusion_type": "ddpm",
        "prediction_type": "v_prediction",
        "ddpm_beta_schedule": "cosine",
    }.items():
        _require(
            head.get(key) == expected,
            f"diffusion_head_config.{key}={head.get(key)!r} is not implemented",
        )
    _require(int(head["hidden_size"]) == hidden, "diffusion head width differs from the decoder")

    acoustic = _speech_tokenizer(
        raw.get("acoustic_tokenizer_config"), "acoustic_tokenizer_config", decoder=True
    )
    semantic = _speech_tokenizer(
        raw.get("semantic_tokenizer_config"), "semantic_tokenizer_config", decoder=False
    )
    acoustic_raw = raw["acoustic_tokenizer_config"]
    semantic_raw = raw["semantic_tokenizer_config"]
    _require(
        acoustic_raw.get("std_dist_type") == "gaussian", "acoustic std_dist_type is not gaussian"
    )
    _require(semantic_raw.get("std_dist_type") == "none", "semantic std_dist_type is not none")
    _require(acoustic.ratios == semantic.ratios, "acoustic and semantic tokenizers differ in hop")
    _require(
        int(raw.get("acoustic_vae_dim", acoustic.vae_dim)) == acoustic.vae_dim,
        "acoustic_vae_dim differs from the acoustic tokenizer",
    )
    _require(
        int(raw.get("semantic_vae_dim", semantic.vae_dim)) == semantic.vae_dim,
        "semantic_vae_dim differs from the semantic tokenizer",
    )
    _require(
        int(head["latent_size"]) == acoustic.vae_dim,
        "diffusion latent differs from the acoustic VAE",
    )

    audio = processor.get("audio_processor")
    _require(isinstance(audio, dict), "preprocessor_config.json has no audio_processor")
    _require(
        int(processor.get("speech_tok_compress_ratio", 0)) == acoustic.hop_length,
        "preprocessor hop differs from the tokenizer hop",
    )
    _require(
        bool(audio.get("normalize_audio")) and bool(processor.get("db_normalize")),
        "voice prompts without loudness normalization are not implemented",
    )
    _require(
        processor.get("language_model_pretrained_name") == TOKENIZER_REPOSITORY,
        f"tokenizer source {processor.get('language_model_pretrained_name')!r} is not "
        f"{TOKENIZER_REPOSITORY}",
    )

    return VibeVoiceConfig(
        hidden_size=hidden,
        intermediate_size=int(decoder["intermediate_size"]),
        num_layers=int(decoder["num_hidden_layers"]),
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=hidden // heads,
        rms_norm_eps=float(decoder["rms_norm_eps"]),
        rope_theta=float(decoder["rope_theta"]),
        vocab_size=int(decoder["vocab_size"]),
        max_position_embeddings=int(decoder["max_position_embeddings"]),
        acoustic=acoustic,
        semantic=semantic,
        # The sigma-VAE draws its per-sample noise scale from N(0, fix_std / 0.8).
        acoustic_vae_std=float(acoustic_raw["fix_std"]) / 0.8,
        head_layers=int(head["head_layers"]),
        head_intermediate_size=int(float(head["head_ffn_ratio"]) * hidden),
        head_eps=float(head["rms_norm_eps"]),
        latent_size=int(head["latent_size"]),
        ddpm_num_steps=int(head["ddpm_num_steps"]),
        sample_rate=int(audio["sampling_rate"]),
        target_dbfs=float(audio["target_dB_FS"]),
        normalize_eps=float(audio["eps"]),
    )
