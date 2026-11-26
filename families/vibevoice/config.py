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
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


_INT32_MAX = 2**31 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_DEPTHS = re.compile(r"[1-9][0-9]*(-[1-9][0-9]*)*")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("vibevoice checkpoint: " + message)


def _unique_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"a JSON object repeats the key {key!r}")
        result[key] = value
    return result


def _no_constant(name: str):
    raise ValueError(f"vibevoice checkpoint: JSON holds the non-number {name}")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"vibevoice checkpoint is missing {path.name}: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_keys,
        parse_constant=_no_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"vibevoice {path.name} is not a JSON object")
    return value


def _integer(raw: dict, key: str, name: str, low: int = 1, high: int = _INT32_MAX) -> int:
    """A JSON integer (not a bool, a float or a string) in [low, high]."""
    value = raw.get(key)
    _require(type(value) is int, f"{name}.{key}={value!r} is not an integer")
    _require(low <= value <= high, f"{name}.{key}={value} lies outside [{low}, {high}]")
    return value


def _number(raw: dict, key: str, name: str) -> float:
    """A finite JSON number that float32 holds, read without a string or bool coercion."""
    value = raw.get(key)
    _require(type(value) in (int, float), f"{name}.{key}={value!r} is not a number")
    _require(
        math.isfinite(value) and abs(value) <= _FLOAT32_MAX,
        f"{name}.{key}={value!r} is not a finite float32",
    )
    return float(value)


def _positive(raw: dict, key: str, name: str) -> float:
    """A positive number that stays a normal float32: the engines hold it as a float32
    constant, where 1e-100 would be 0 and a subnormal may be flushed to 0."""
    value = _number(raw, key, name)
    _require(
        value >= float(np.finfo(np.float32).tiny),
        f"{name}.{key}={value!r} must be a positive normal float32",
    )
    return value


def _same(value, expected) -> bool:
    """Equal and of the same JSON type: True is not 1, 1 is not 1.0."""
    return type(value) is type(expected) and value == expected


def _positive_ints(value, name: str) -> tuple[int, ...]:
    _require(
        isinstance(value, list)
        and len(value) > 0
        and all(type(item) is int and 1 <= item <= _INT32_MAX for item in value),
        f"{name}={value!r} is not a list of positive integers",
    )
    return tuple(value)


def usable_gain(target_dbfs: float, eps: float) -> bool:
    """Mirror of the runtime's check: the voice-prompt loudness gain
    float(10^(target / 20)) / (rms + float(eps)), formed in float32 for rms from 0
    to FLT_MAX, stays finite and nonzero."""
    tiny = float(np.finfo(np.float32).tiny)
    if not math.isfinite(target_dbfs) or not tiny <= eps <= _FLOAT32_MAX:
        return False
    target = float(np.float32(target_dbfs))
    try:
        power = 10.0 ** (target / 20.0)
    except OverflowError:
        power = math.inf
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        amplitude = np.float32(power)
        eps32 = np.float32(eps)
        silence = amplitude / eps32
        loudest = np.float32(_FLOAT32_MAX) + eps32
        quietest = amplitude / loudest
    return bool(
        amplitude >= tiny
        and np.isfinite(silence)
        and silence >= tiny
        and np.isfinite(loudest)
        and quietest > 0.0
    )


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
        _require(_same(raw.get(key), expected), f"{name}.{key}={raw.get(key)!r} is not implemented")
    written = raw.get("encoder_depths")
    _require(
        isinstance(written, str) and _DEPTHS.fullmatch(written) is not None,
        f"{name}.encoder_depths={written!r} is not positive decimals joined by '-'",
    )
    depths = tuple(int(item) for item in written.split("-"))
    encoder_ratios = _positive_ints(raw.get("encoder_ratios"), f"{name}.encoder_ratios")
    _require(len(depths) == len(encoder_ratios) + 1, f"{name} depths do not match its ratios")
    filters = _integer(raw, "encoder_n_filters", name)
    if decoder:
        _require(
            raw.get("decoder_depths") is None
            or _same(raw.get("decoder_depths"), "-".join(str(d) for d in reversed(depths))),
            f"{name}.decoder_depths is not the mirrored encoder depths",
        )
        _require(
            _positive_ints(raw.get("decoder_ratios"), f"{name}.decoder_ratios") == encoder_ratios,
            f"{name}.decoder_ratios differs from its encoder ratios",
        )
        _require(
            _integer(raw, "decoder_n_filters", name) == filters,
            f"{name}.decoder_n_filters differs from its encoder filters",
        )
    return SpeechTokenizerConfig(
        channels=1,
        vae_dim=_integer(raw, "vae_dim", name),
        num_filters=filters,
        # The encoder applies the ratios in reverse of their listed order.
        ratios=tuple(reversed(encoder_ratios)),
        depths=depths,
        eps=_positive(raw, "layernorm_eps", name),
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
        decoder.get("use_sliding_window", False) is False,
        "decoder_config.use_sliding_window: sliding-window attention is not implemented",
    )
    _require(
        decoder.get("tie_word_embeddings") is True,
        "decoder_config.tie_word_embeddings: untied LM heads are not implemented",
    )
    name = "decoder_config"
    hidden = _integer(decoder, "hidden_size", name)
    heads = _integer(decoder, "num_attention_heads", name)
    kv_heads = _integer(decoder, "num_key_value_heads", name)
    _require(
        hidden % heads == 0 and heads % kv_heads == 0 and (hidden // heads) % 2 == 0,
        "attention head geometry is invalid (rotary halves need an even head width)",
    )
    rope_theta = _number(decoder, "rope_theta", name)
    # The runtime takes inverse frequencies in (0, 1]: 1 / theta ** (2i / d) with theta > 1.
    _require(rope_theta > 1.0, f"decoder_config.rope_theta={rope_theta!r} must exceed 1")

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
    _require(
        _integer(head, "hidden_size", "diffusion_head_config") == hidden,
        "diffusion head width differs from the decoder",
    )
    ffn_ratio = _positive(head, "head_ffn_ratio", "diffusion_head_config")
    head_intermediate = ffn_ratio * hidden
    _require(
        head_intermediate.is_integer() and 1 <= head_intermediate <= _INT32_MAX,
        f"diffusion_head_config.head_ffn_ratio={ffn_ratio!r} gives no integer FFN width",
    )

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
    for key, tokenizer in (("acoustic_vae_dim", acoustic), ("semantic_vae_dim", semantic)):
        _require(
            key not in raw or _integer(raw, key, "config") == tokenizer.vae_dim,
            f"{key} differs from its tokenizer",
        )
    # The voice encoder holds fix_std / 0.8 as a float32 constant.
    vae_std = _positive(acoustic_raw, "fix_std", "acoustic_tokenizer_config") / 0.8
    _require(
        vae_std <= _FLOAT32_MAX,
        f"acoustic_tokenizer_config.fix_std gives the noise scale {vae_std!r}, above float32",
    )
    latent = _integer(head, "latent_size", "diffusion_head_config")
    _require(latent == acoustic.vae_dim, "diffusion latent differs from the acoustic VAE")

    audio = processor.get("audio_processor")
    _require(isinstance(audio, dict), "preprocessor_config.json has no audio_processor")
    _require(
        _integer(processor, "speech_tok_compress_ratio", "preprocessor_config")
        == acoustic.hop_length,
        "preprocessor hop differs from the tokenizer hop",
    )
    _require(
        audio.get("normalize_audio") is True and processor.get("db_normalize") is True,
        "voice prompts without loudness normalization are not implemented",
    )
    _require(
        processor.get("language_model_pretrained_name") == TOKENIZER_REPOSITORY,
        f"tokenizer source {processor.get('language_model_pretrained_name')!r} is not "
        f"{TOKENIZER_REPOSITORY}",
    )
    target_dbfs = _number(audio, "target_dB_FS", "audio_processor")
    eps = _positive(audio, "eps", "audio_processor")
    _require(
        usable_gain(target_dbfs, eps),
        f"audio_processor target_dB_FS={target_dbfs!r} with eps={eps!r} gives no finite "
        "nonzero float32 loudness gain",
    )

    return VibeVoiceConfig(
        hidden_size=hidden,
        intermediate_size=_integer(decoder, "intermediate_size", name),
        num_layers=_integer(decoder, "num_hidden_layers", name),
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=hidden // heads,
        rms_norm_eps=_positive(decoder, "rms_norm_eps", name),
        rope_theta=rope_theta,
        vocab_size=_integer(decoder, "vocab_size", name),
        max_position_embeddings=_integer(decoder, "max_position_embeddings", name),
        acoustic=acoustic,
        semantic=semantic,
        # The sigma-VAE draws its per-sample noise scale from N(0, fix_std / 0.8).
        acoustic_vae_std=vae_std,
        head_layers=_integer(head, "head_layers", "diffusion_head_config"),
        head_intermediate_size=int(head_intermediate),
        head_eps=_positive(head, "rms_norm_eps", "diffusion_head_config"),
        latent_size=latent,
        # The runtime's DPM-Solver steps over this schedule, at least the default steps.
        ddpm_num_steps=_integer(head, "ddpm_num_steps", "diffusion_head_config", 11, 100_000),
        sample_rate=_integer(audio, "sampling_rate", "audio_processor"),
        target_dbfs=target_dbfs,
        normalize_eps=eps,
    )
