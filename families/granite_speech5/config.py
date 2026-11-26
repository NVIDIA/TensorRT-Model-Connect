# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Granite Speech 5 CTC configuration this family builds, read and checked in one place.

Values are read with the types and defaults of the reference configuration classes
(``GraniteSpeech5CTCConfig``, ``GraniteSpeech5EncoderConfig``, ``GraniteSpeech5FeatureExtractor``):
an absent key takes the reference default, a key the reference types as optional may be ``null``,
and every other value must have its JSON type (no coercion). A value the graph would not reproduce
is refused here, naming it, instead of being built into a different model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARCHITECTURE = "GraniteSpeech5ForCTC"
MODEL_TYPE = "granite_speech5_ctc"
ENCODER_TYPE = "granite_speech5_encoder"
FEATURE_EXTRACTOR = "GraniteSpeech5FeatureExtractor"
TOKENIZER_CLASS = "ParakeetTokenizer"
# torch.nn.LayerNorm and BatchNorm1d defaults of the reference encoder.
NORM_EPS = 1e-5
# The feature extractor concatenates every log-mel frame with its delta and stacks frame pairs.
DELTA_EXPANSION = 2
FRAME_STACKING = 2
# Engine profiles: tuned for this clip length; the default and the largest accepted longest clip.
OPT_SECONDS = 15
DEFAULT_MAX_SECONDS = 30
MAX_SECONDS_LIMIT = 600
# Values that size the graph or host work before the checkpoint's tensors are checked are bounded
# far above the checkpoint's own (512, 3, 16, 16384, 16000): the DFT basis holds n_fft * (n_fft + 2)
# floats, the deltas unroll one slice pair per tap, every layer adds a block and its inventory.
LIMITS = {"n_fft": 8192, "delta_win_length": 31, "num_hidden_layers": 256, "vocab_size": 1 << 20,
          "sampling_rate": 192000}


@dataclass(frozen=True)
class FrontEnd:
    """Effective GraniteSpeech5FeatureExtractor arguments."""

    sample_rate: int
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int
    delta_win_length: int
    floor_db: float

    @property
    def bins(self) -> int:
        return self.n_fft // 2 + 1

    def mel_frames(self, samples: int) -> int:
        """Log-mel frames the reference keeps for ``samples`` PCM values, before stacking."""
        return FRAME_STACKING * -(-(samples // self.hop_length) // FRAME_STACKING)

    def stacked_frames(self, samples: int) -> int:
        """Encoder input frames for ``samples`` PCM values."""
        return self.mel_frames(samples) // FRAME_STACKING

    def needs_tail_sample(self, samples: int) -> bool:
        """Whether the reference appends one zero sample so its last frame pair is complete."""
        needed = (self.mel_frames(samples) - 1) * self.hop_length + 1
        return samples < needed


@dataclass(frozen=True)
class Encoder:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    context_size: int
    max_position_embeddings: int
    conv_kernel_size: int
    conv_expansion_factor: int
    subsample_layers: tuple[int, ...]
    feed_forward_bias: bool
    num_mel_bins: int
    vocab_size: int

    @property
    def attention_width(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def conv_channels(self) -> int:
        return self.hidden_size * self.conv_expansion_factor

    @property
    def input_width(self) -> int:
        return self.num_mel_bins * DELTA_EXPANSION * FRAME_STACKING

    def self_conditioning_after(self, layer: int) -> bool:
        """The reference feeds the CTC posteriors back after the layer ``num_layers // 2 - 1``."""
        return layer + 1 == self.num_layers // 2


@dataclass(frozen=True)
class GraniteSpeech5Config:
    front_end: FrontEnd
    encoder: Encoder
    vocab_size: int
    blank_id: int

    def layer_frames(self, stacked: int) -> list[int]:
        """Frames entering each encoder layer, then the output frame count, for ``stacked`` input frames."""
        frames = [stacked]
        for layer in range(self.encoder.num_layers):
            current = frames[-1]
            frames.append(current // 2 if layer in self.encoder.subsample_layers else current)
        return frames

    def output_frames(self, samples: int) -> int:
        """CTC frames (one id each) for a clip of ``samples`` PCM values."""
        return self.layer_frames(self.front_end.stacked_frames(samples))[-1]


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"repeats the key {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    """One JSON object; a repeated key is an error, not a silent last-wins."""
    if not path.is_file():
        raise FileNotFoundError(f"granite_speech5 checkpoint is missing {path.name}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except ValueError as error:
        raise ValueError(f"granite_speech5 {path.name} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"granite_speech5 {path.name} must hold one JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"granite_speech5 does not build this checkpoint: {message}")


def _value(section: dict, key: str, default, where: str, *, nullable: bool = False):
    if key not in section:
        return default
    value = section[key]
    if value is None and not nullable:
        raise ValueError(f"granite_speech5 {where}.{key} must not be null")
    return value


def _int(section: dict, key: str, default, where: str, *, minimum: int = 1, nullable: bool = False):
    value = _value(section, key, default, where, nullable=nullable)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"granite_speech5 {where}.{key} must be a JSON integer, not {value!r}")
    if value < minimum:
        raise ValueError(f"granite_speech5 {where}.{key} must be at least {minimum}, not {value}")
    if value > np.iinfo(np.int32).max:
        raise ValueError(f"granite_speech5 {where}.{key} {value} does not fit int32")
    if value > LIMITS.get(key, value):
        raise ValueError(f"granite_speech5 builds {where}.{key} up to {LIMITS[key]}, not {value}")
    return value


def _number(section: dict, key: str, default, where: str) -> float:
    value = _value(section, key, default, where)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"granite_speech5 {where}.{key} must be a JSON number, not {value!r}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    with np.errstate(over="ignore"):
        stored = np.float32(value) if finite else np.float32(np.inf)
    if not finite or not np.isfinite(stored):
        raise ValueError(f"granite_speech5 {where}.{key} {value!r} is not a finite float32")
    return float(value)


def _bool(section: dict, key: str, default: bool, where: str) -> bool:
    value = _value(section, key, default, where)
    if not isinstance(value, bool):
        raise ValueError(f"granite_speech5 {where}.{key} must be a JSON boolean, not {value!r}")
    return value


def _string(section: dict, key: str, default, where: str) -> str:
    value = _value(section, key, default, where)
    if not isinstance(value, str):
        raise ValueError(f"granite_speech5 {where}.{key} must be a JSON string, not {value!r}")
    return value


def _encoder(section: object) -> Encoder:
    where = "encoder_config"
    if not isinstance(section, dict):
        raise ValueError(f"granite_speech5 config.json encoder_config must be a JSON object, not {section!r}")
    _require(_string(section, "model_type", ENCODER_TYPE, where) == ENCODER_TYPE,
             f"encoder model_type {section.get('model_type')!r}")
    hidden = _int(section, "hidden_size", 1024, where)
    heads = _int(section, "num_attention_heads", 8, where)
    kv_heads = _int(section, "num_key_value_heads", None, where, nullable=True)
    head_dim = _int(section, "head_dim", None, where, nullable=True)
    head_dim = hidden // heads if head_dim is None else head_dim
    _require(head_dim > 0, f"head_dim {head_dim} for hidden_size {hidden} and {heads} heads")
    # The reference projects keys and values with num_attention_heads heads either way, so only
    # num_key_value_heads == num_attention_heads runs in it.
    _require(kv_heads is None or kv_heads == heads,
             f"num_key_value_heads {kv_heads} differs from num_attention_heads {heads}")
    _require(_string(section, "hidden_act", "silu", where) == "silu",
             f"hidden_act {section.get('hidden_act')!r} (the graph builds SiLU)")
    layers = _int(section, "num_hidden_layers", 16, where)
    context = _int(section, "context_size", 128, where)
    max_positions = _int(section, "max_position_embeddings", 512, where)
    _require(context <= max_positions,
             f"context_size {context} above max_position_embeddings {max_positions} (the reference refuses it too)")
    kernel = _int(section, "conv_kernel_size", 7, where)
    # The reference pads (k - 1) // 2 on both sides; an even kernel would shorten every block's output.
    _require(kernel % 2 == 1, f"an even conv_kernel_size {kernel}")
    raw_subsample = _value(section, "subsample_layers", None, where, nullable=True)
    if raw_subsample is None:
        subsample = (0, 1)
    else:
        if not isinstance(raw_subsample, list):
            raise ValueError(f"granite_speech5 {where}.subsample_layers must be a JSON list, not {raw_subsample!r}")
        subsample = tuple(_int({"value": item}, "value", None, f"{where}.subsample_layers", minimum=0)
                          for item in raw_subsample)
    # The reference halves the output length once per listed entry but subsamples only in listed
    # layers that exist, so a repeated or out-of-range entry makes its own two counts disagree.
    _require(len(set(subsample)) == len(subsample) and all(index < layers for index in subsample),
             f"subsample_layers {list(subsample)} for {layers} layers")
    for key in ("attention_dropout", "initializer_range"):
        _number(section, key, 0.0, where)  # inference ignores them, as the reference does in eval mode
    # The reference passes activation_dropout to functional.dropout, which refuses p outside [0, 1] in eval mode too.
    dropout = _number(section, "activation_dropout", 0.0, where)
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"granite_speech5 {where}.activation_dropout {dropout} is outside [0, 1]; the reference's "
                         "dropout refuses it")
    return Encoder(
        hidden_size=hidden,
        intermediate_size=_int(section, "intermediate_size", 4096, where),
        num_layers=layers,
        num_heads=heads,
        head_dim=head_dim,
        context_size=context,
        max_position_embeddings=max_positions,
        conv_kernel_size=kernel,
        conv_expansion_factor=_int(section, "conv_expansion_factor", 2, where),
        subsample_layers=tuple(sorted(subsample)),
        feed_forward_bias=_bool(section, "attention_bias", True, where),
        num_mel_bins=_int(section, "num_mel_bins", 80, where),
        vocab_size=_int(section, "vocab_size", 16384, where),
    )


def _front_end(processor: dict) -> FrontEnd:
    section = processor.get("feature_extractor")
    where = "processor_config.feature_extractor"
    if not isinstance(section, dict):
        raise ValueError(f"granite_speech5 processor_config.json needs a feature_extractor object, not {section!r}")
    _require(section.get("feature_extractor_type") == FEATURE_EXTRACTOR,
             f"feature extractor {section.get('feature_extractor_type')!r}")
    front = FrontEnd(
        sample_rate=_int(section, "sampling_rate", 16000, where),
        n_fft=_int(section, "n_fft", 512, where),
        win_length=_int(section, "win_length", 400, where),
        hop_length=_int(section, "hop_length", 160, where),
        n_mels=_int(section, "num_mel_bins", 80, where),
        delta_win_length=_int(section, "delta_win_length", 3, where, minimum=3),
        floor_db=_number(section, "logmel_floor_db", 8.0, where),
    )
    # The graph centres frames with n_fft / 2 samples of reflect padding on each side.
    _require(front.n_fft % 2 == 0, f"an odd n_fft {front.n_fft}")
    _require(front.win_length <= front.n_fft, f"win_length {front.win_length} above n_fft {front.n_fft}")
    # A one-sample hop would let the appended tail sample add a frame the graph does not slice.
    _require(front.hop_length >= 2, f"hop_length {front.hop_length}")
    _require(front.n_mels <= front.bins, f"{front.n_mels} mel bins over {front.bins} frequency bins")
    return front


def _blank_id(top: dict, tokenizer_config: dict, pieces: dict[str, int], vocab_size: int) -> int:
    pad = _int(top, "pad_token_id", 0, "config", minimum=0, nullable=True)
    _require(pad is not None, "pad_token_id null (it is the CTC blank)")
    _require(pad < vocab_size, f"pad_token_id {pad} outside the vocabulary of {vocab_size}")
    token = tokenizer_config.get("pad_token")
    if isinstance(token, dict):
        token = token.get("content")
    _require(isinstance(token, str) and token in pieces,
             f"tokenizer pad_token {tokenizer_config.get('pad_token')!r} (the decoder drops it as the blank)")
    # The model masks and trains with pad_token_id as the blank; the tokenizer drops its own pad
    # token. One id must be both, or the transcript keeps or loses a token the model did not mean.
    _require(pieces[token] == pad, f"tokenizer pad token id {pieces[token]} differs from pad_token_id {pad}")
    return pad


def resolve(model_dir: Path, pieces: dict[str, int] | None = None) -> GraniteSpeech5Config:
    """Read config.json, processor_config.json and tokenizer_config.json; raise naming what is not built.

    ``pieces`` maps every token string to its id (from :mod:`.vocabulary`); it resolves the
    tokenizer's pad token, which is the CTC blank.
    """
    model_dir = Path(model_dir)
    top = read_json(model_dir / "config.json")
    _require(top.get("model_type") == MODEL_TYPE, f"model_type {top.get('model_type')!r}")
    _require(top.get("architectures") == [ARCHITECTURE], f"architectures {top.get('architectures')!r}")
    # The CTC head is the encoder's own output projection only while the two are tied.
    _require(_bool(top, "tie_word_embeddings", True, "config"), "an untied CTC head (tie_word_embeddings false)")
    # An absent or null encoder_config is the reference's default encoder.
    raw_encoder = _value(top, "encoder_config", None, "config", nullable=True)
    encoder = _encoder({} if raw_encoder is None else raw_encoder)
    vocab = _int(top, "vocab_size", 16384, "config")
    _require(vocab == encoder.vocab_size, f"vocab_size {vocab} differs from encoder_config.vocab_size {encoder.vocab_size}")
    front = _front_end(read_json(model_dir / "processor_config.json"))
    _require(front.n_mels == encoder.num_mel_bins,
             f"{front.n_mels} feature-extractor mel bins for an encoder of {encoder.num_mel_bins}")
    tokenizer_config = read_json(model_dir / "tokenizer_config.json")
    _require(tokenizer_config.get("tokenizer_class") == TOKENIZER_CLASS,
             f"tokenizer_class {tokenizer_config.get('tokenizer_class')!r} (the CTC decoding is ParakeetTokenizer's)")
    # Transformers' clean-up rewrites spaces before punctuation after decoding; the runtime does not.
    _require(not _bool(tokenizer_config, "clean_up_tokenization_spaces", False, "tokenizer_config"),
             "clean_up_tokenization_spaces true (the runtime decodes without the clean-up)")
    if pieces is None:
        from .vocabulary import read_pieces

        pieces = read_pieces(model_dir)
    blank = _blank_id(top, tokenizer_config, pieces, vocab)
    return GraniteSpeech5Config(front_end=front, encoder=encoder, vocab_size=vocab, blank_id=blank)


def sample_limits(cfg: GraniteSpeech5Config, max_samples: int | None = None) -> dict:
    """The accepted clip lengths in samples and the engine's waveform profile.

    The shortest clip yields one CTC frame and leaves the reflect padding a sample to mirror; the
    longest is ``max_samples`` (default ``DEFAULT_MAX_SECONDS``). The engine reads the waveform
    after the tail sample and the reflect padding the runtime adds.
    """
    fe = cfg.front_end
    halvings = len(cfg.encoder.subsample_layers)
    # stacked >= 2**halvings frames  <=>  hop-frames >= FRAME_STACKING * 2**halvings - 1
    min_samples = max((FRAME_STACKING * 2**halvings - 1) * fe.hop_length, fe.n_fft // 2 + 1)
    while cfg.output_frames(min_samples) < 1:  # guards the closed form above
        min_samples += 1
    limit = MAX_SECONDS_LIMIT * fe.sample_rate
    if max_samples is None:
        max_samples = DEFAULT_MAX_SECONDS * fe.sample_rate
    if not isinstance(max_samples, int) or isinstance(max_samples, bool):
        raise ValueError(f"granite_speech5 max_sequence_length must be an integer sample count, not {max_samples!r}")
    if not min_samples <= max_samples <= limit:
        raise ValueError(f"granite_speech5 max_sequence_length is the longest clip in samples: "
                         f"{min_samples} to {limit}, not {max_samples}")
    opt_samples = min(max(OPT_SECONDS * fe.sample_rate, min_samples), max_samples)
    return {
        "min_samples": min_samples,
        "opt_samples": opt_samples,
        "max_samples": max_samples,
        # waveform input = clip, one optional tail sample, n_fft / 2 reflected samples on each side
        "min_waveform": min_samples + fe.n_fft,
        "opt_waveform": opt_samples + fe.n_fft,
        "max_waveform": max_samples + 1 + fe.n_fft,
    }
