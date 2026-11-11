# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-ASR family builder.

The checkpoint pairs two causal sigma-VAE convolution encoders (acoustic and
semantic, 7.5 frames per second at 24 kHz) and their speech connectors with a
Qwen2 decoder. Connector rows replace the ``<|box_start|>`` speech placeholders
of a fixed chat prompt and the decoder generates the transcript as a JSON list
of ``{"Start", "End", "Speaker", "Content"}`` segments.

The bundle holds an audio-encoder engine, a dual-profile decoder engine with a
native KV cache, the tokenizer and ``runtime.json``. The checkpoint ships no
tokenizer; like the official processor, the build uses a checkpoint's own
tokenizer when it ships ``tokenizer_config.json`` and otherwise the
``Qwen/Qwen2.5-7B`` tokenizer the official demo names, at a pinned revision.
"""

from __future__ import annotations

import json
from pathlib import Path

from .audio_encoder_builder import build_audio_encoder_engine
from .checkpoint import SAMPLE_RATE, VibeVoiceAsrConfig, _open_safetensors, load_config
from .decoder_builder import KEY_BLOCK, build_decoder_engine

FAMILY = "vibevoice_asr"
TASK = "speech_transcription"

TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-7B"
TOKENIZER_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

# ``VibeVoiceASRProcessor``: system turn, then a user turn holding the speech
# placeholders and the duration, encoded without a generation prompt.
SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output in JSON format."
)
TRANSCRIPTION_KEYS = "Start time, End time, Speaker ID, Content"
SPECIAL_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "speech_start": "<|object_ref_start|>",
    "speech_end": "<|object_ref_end|>",
    "speech_pad": "<|box_start|>",
    "eos": "<|endoftext|>",
}

DEFAULT_MAX_SEQUENCE_LENGTH = 65536  # the model's single-pass budget (about 60 min of audio)
MAX_PREFILL_ROWS = 256
ENCODER_CHUNK_FRAMES = 450  # 60 s per encoder call, as the official long-audio segments
NORMALIZE_TARGET_DBFS = -25.0
NORMALIZE_EPS = 1e-6


def resolve_capacity(max_sequence_length: int | None, cfg: VibeVoiceAsrConfig) -> int:
    """KV-cache rows: prompt (system, speech, duration) plus generated tokens."""
    capacity = int(max_sequence_length or DEFAULT_MAX_SEQUENCE_LENGTH)
    limit = cfg.decoder.max_position_embeddings
    if capacity > limit:
        raise ValueError(f"{FAMILY} max_sequence_length {capacity} exceeds {limit} positions")
    if capacity < KEY_BLOCK or capacity % KEY_BLOCK:
        raise ValueError(f"{FAMILY} max_sequence_length must be a positive multiple of {KEY_BLOCK}")
    return capacity


def tokenizer_root(model_dir: Path) -> Path | None:
    """The checkpoint's own tokenizer directory, or None for the pinned Qwen2.5 tokenizer.

    ``VibeVoiceASRProcessor.from_pretrained`` uses a checkpoint's tokenizer when
    it ships ``tokenizer_config.json`` and otherwise loads Qwen2.5's.
    """
    return model_dir if (model_dir / "tokenizer_config.json").is_file() else None


def _tokenizer_files(model_dir: Path) -> dict[str, bytes]:
    root = tokenizer_root(model_dir)
    if root is None:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                TOKENIZER_REPOSITORY, revision=TOKENIZER_REVISION, allow_patterns=TOKENIZER_FILES
            )
        )
    files = {}
    for name in TOKENIZER_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"{FAMILY} tokenizer file is missing: {path}")
        files[name] = path.read_bytes()
    return files


def special_token_ids(tokenizer_json: bytes) -> dict[str, int]:
    """Ids of the prompt and stop tokens, which must be special added tokens."""
    added = {item["content"]: item for item in json.loads(tokenizer_json)["added_tokens"]}
    ids = {}
    for role, content in SPECIAL_TOKENS.items():
        item = added.get(content)
        if item is None or not item.get("special"):
            raise ValueError(f"{FAMILY} tokenizer lacks the special token {content}")
        ids[role] = int(item["id"])
    return ids


def user_suffix(duration: str, context_info: str = "") -> str:
    """User-turn text after the speech span (``VibeVoiceASRProcessor``)."""
    context = context_info.strip()
    if context:
        return (
            f"This is a {duration} seconds audio, with extra info: {context}\n\n"
            f"Please transcribe it with these keys: {TRANSCRIPTION_KEYS}"
        )
    return (
        f"This is a {duration} seconds audio, please transcribe it with these keys: "
        f"{TRANSCRIPTION_KEYS}"
    )


def tokenizer_checks(tokenizer_json: bytes) -> list[dict]:
    """Prompt pieces with the ids the reference fast tokenizer assigns them.

    The native tokenizer must reproduce every one before the runtime loads, so a
    pre-tokenizer or merge difference fails at load instead of shifting the prompt.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
    im_start, im_end = SPECIAL_TOKENS["im_start"], SPECIAL_TOKENS["im_end"]
    texts = (
        f"{im_start}system\n{SYSTEM_PROMPT}{im_end}\n",
        f"{im_start}user\n",
        f"\n{user_suffix('12.34')}{im_end}\n",
        f"\n{user_suffix('5.07', 'Hotwords: NVIDIA, TensorRT, VibeVoice')}{im_end}\n",
        f"\n{user_suffix('61.80', '会议记录：张三、李四 — café naïve')}{im_end}\n",
    )
    return [
        {"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids}
        for text in texts
    ]


def runtime_contract(cfg: VibeVoiceAsrConfig, capacity: int, tokenizer_json: bytes) -> dict:
    hop = cfg.hop_length
    history = max(cfg.acoustic.history_samples(), cfg.semantic.history_samples())
    context_frames = -(-history // hop)
    decoder = cfg.decoder
    return {
        "tensor_parallel_size": 1,
        "sample_rate": SAMPLE_RATE,
        "hop_length": hop,
        "encoder_chunk_frames": ENCODER_CHUNK_FRAMES,
        "encoder_context_frames": context_frames,
        "normalize_target_dbfs": NORMALIZE_TARGET_DBFS,
        "normalize_eps": NORMALIZE_EPS,
        "hidden_size": decoder.hidden_size,
        "num_layers": decoder.num_layers,
        "num_kv_heads": decoder.num_kv_heads,
        "head_dim": decoder.head_dim,
        "vocab_size": decoder.vocab_size,
        "max_sequence_length": capacity,
        "max_prefill_rows": min(MAX_PREFILL_ROWS, capacity),
        "system_prompt": SYSTEM_PROMPT,
        "transcription_keys": TRANSCRIPTION_KEYS,
        "token_ids": special_token_ids(tokenizer_json),
        "tokenizer_checks": tokenizer_checks(tokenizer_json),
    }


def build(request, writer) -> None:
    """Build one VibeVoice-ASR bundle."""
    if request.task != TASK:
        raise ValueError(f"{FAMILY} supports only task={TASK}")
    if request.backend != "trt":
        raise ValueError(f"{FAMILY} supports only backend=trt")
    if str(request.precision).lower() != "bf16":
        raise NotImplementedError(f"{FAMILY} builds the checkpoint's native bf16 only")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(f"{FAMILY} does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError(f"{FAMILY} does not support fp32_layers")
    if request.dynamic_kv_cache:
        raise NotImplementedError("vibevoice_asr does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError(f"{FAMILY} transcribes one request at a time")
    if int(request.tensor_parallel_size) != 1 or int(request.context_parallel_size) != 1:
        raise NotImplementedError(f"{FAMILY} runs on one GPU")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError(f"{FAMILY} takes audio, not an image size")
    if request.video_num_frames is not None:
        raise NotImplementedError(f"{FAMILY} does not support video_num_frames")
    if request.mtp_seq_len is not None:
        raise NotImplementedError(f"{FAMILY} does not support MTP speculative decoding")
    if request.graph_transform is not None:
        raise NotImplementedError(f"{FAMILY} does not support graph transforms")

    model_dir = Path(request.model_dir)
    cfg = load_config(model_dir)
    capacity = resolve_capacity(request.max_sequence_length, cfg)
    tokenizer = _tokenizer_files(model_dir)
    runtime = runtime_contract(cfg, capacity, tokenizer["tokenizer.json"])
    chunk_frames = runtime["encoder_chunk_frames"] + runtime["encoder_context_frames"]
    verbose = bool(request.verbose)

    writer.set_header(family="vibevoice_asr", task=request.task, backend=request.backend)
    weights = _open_safetensors(model_dir)
    try:
        writer.add_bytes(
            "audio_encoder.plan",
            build_audio_encoder_engine(
                cfg,
                weights,
                max_samples=chunk_frames * cfg.hop_length,
                opt_samples=225 * cfg.hop_length,
                verbose=verbose,
            ),
        )
        writer.add_bytes(
            "decoder.plan",
            build_decoder_engine(
                cfg.decoder,
                weights,
                capacity=capacity,
                max_prefill_rows=runtime["max_prefill_rows"],
                verbose=verbose,
            ),
        )
    finally:
        weights.close()
    writer.add_json("runtime.json", runtime)
    for name, data in tokenizer.items():
        writer.add_bytes(name, data)
