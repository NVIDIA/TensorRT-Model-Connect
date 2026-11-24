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
``Qwen/Qwen2.5-7B`` tokenizer the official demo names, at a pinned revision,
read from the local Hugging Face cache (the build does not download it).
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
    if max_sequence_length is not None and (
        isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int)
    ):
        raise ValueError(f"{FAMILY} max_sequence_length must be an integer, got {max_sequence_length!r}")
    capacity = DEFAULT_MAX_SEQUENCE_LENGTH if max_sequence_length is None else max_sequence_length
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


TOKENIZER_FETCH_COMMAND = (
    f"hf download {TOKENIZER_REPOSITORY} {' '.join(TOKENIZER_FILES)} "
    f"--revision {TOKENIZER_REVISION}"
)


def _cached_tokenizer() -> Path:
    """The pinned Qwen2.5 tokenizer from the local Hugging Face cache, never downloaded."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(
            snapshot_download(
                TOKENIZER_REPOSITORY,
                revision=TOKENIZER_REVISION,
                allow_patterns=TOKENIZER_FILES,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError as error:
        raise FileNotFoundError(
            f"{FAMILY} reads the {TOKENIZER_REPOSITORY} tokenizer from the local Hugging Face "
            f"cache; fetch it with: {TOKENIZER_FETCH_COMMAND}"
        ) from error


def _tokenizer_files(model_dir: Path) -> dict[str, bytes]:
    root = tokenizer_root(model_dir)
    cached = root is None
    if cached:
        root = _cached_tokenizer()
    files = {}
    for name in TOKENIZER_FILES:
        path = root / name
        if not path.is_file():
            hint = f"; fetch it with: {TOKENIZER_FETCH_COMMAND}" if cached else ""
            raise FileNotFoundError(f"{FAMILY} tokenizer file is missing: {path}{hint}")
        files[name] = path.read_bytes()
    check_tokenizer_config(files["tokenizer_config.json"])
    return files


# tokenizer_config.json switches with which transformers would encode or decode
# differently from the runtime, which reads only tokenizer.json.
TOKENIZER_CONFIG_SWITCHES = (
    "add_bos_token",
    "add_eos_token",
    "add_prefix_space",
    "clean_up_tokenization_spaces",
    "split_special_tokens",
)


def check_tokenizer_config(tokenizer_config_json: bytes) -> None:
    """Each switch must be off (false or absent), as in Qwen2.5's tokenizer_config.json."""
    raw = json.loads(tokenizer_config_json)
    if not isinstance(raw, dict):
        raise ValueError(f"{FAMILY} tokenizer_config.json must be a JSON object")
    for key in TOKENIZER_CONFIG_SWITCHES:
        value = raw.get(key)
        if value is not None and value is not False:
            raise ValueError(
                f"{FAMILY} tokenizer_config.json {key} is not supported: {json.dumps(value)}"
            )


def special_token_ids(tokenizer_json: bytes) -> dict[str, int]:
    """Ids of the prompt and stop tokens, which must be special added tokens."""
    added = {item["content"]: item for item in json.loads(tokenizer_json).get("added_tokens", [])}
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
    Each added token must already have the id the reference assigns it: the reference
    re-adds them by content, while the runtime and runtime.json take the listed ids.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
    for item in json.loads(tokenizer_json).get("added_tokens", []):
        assigned = tokenizer.token_to_id(item["content"])
        if assigned != item["id"]:
            raise ValueError(
                f"{FAMILY} tokenizer.json lists the added token {item['content']} with id "
                f"{item['id']}, but the reference tokenizer assigns it {assigned}"
            )
    im_start, im_end = SPECIAL_TOKENS["im_start"], SPECIAL_TOKENS["im_end"]
    texts = (
        f"{im_start}system\n{SYSTEM_PROMPT}{im_end}\n",
        f"{im_start}user\n",
        f"\n{user_suffix('12.34')}{im_end}\n",
        f"\n{user_suffix('5.07', 'Hotwords: NVIDIA, TensorRT, VibeVoice')}{im_end}\n",
        f"\n{user_suffix('61.80', '会议记录：张三、李四 — café naïve')}{im_end}\n",
        # NFC (e + U+0301) and \p{N} beyond ASCII: Arabic-Indic digits, a superscript.
        f"\n{user_suffix('3.20', 'Cafe\u0301 \u0661\u0662\u0663 x\u00b2')}{im_end}\n",
    )
    return [
        {"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids}
        for text in texts
    ]


def check_tokenizer_ids(tokenizer_json: bytes, vocab_size: int) -> None:
    """Every token id must be a row of the decoder's embedding table."""
    raw = json.loads(tokenizer_json)
    ids = list(raw["model"]["vocab"].values())
    ids += [item.get("id") for item in raw.get("added_tokens", [])]
    stray = [i for i in ids if type(i) is not int or not 0 <= i < vocab_size]
    if stray:
        raise ValueError(
            f"{FAMILY} tokenizer ids {stray[:5]} are outside the decoder vocabulary of {vocab_size}"
        )


# The Split pattern of Qwen2.5's tokenizer.json, the one the runtime's tokenizer implements.
QWEN_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def check_merges(vocab: dict, merges, unsupported) -> None:
    """The merge table as the reference reads it, and one the runtime's batch merge reproduces.

    One entry format for the whole table (pairs, or "left right" strings with exactly one
    space, after any "#version" header lines); each operand and the result in the
    vocabulary; and no operand that only a merge of a later rank makes, so merging every
    occurrence of the best pair at once gives the reference's one-at-a-time result.
    """
    if not isinstance(merges, list):
        raise unsupported("merge", merges)
    strings = bool(merges) and isinstance(merges[0], str)
    pairs = []
    for entry in merges:
        if strings:
            if not isinstance(entry, str):
                raise unsupported("merge", entry)
            if entry.startswith("#version"):
                continue
            parts = entry.split(" ")
            if len(parts) != 2:
                raise unsupported("merge", entry)
        elif isinstance(entry, list) and len(entry) == 2 and all(isinstance(x, str) for x in entry):
            parts = entry
        else:
            raise unsupported("merge", entry)
        if parts[0] not in vocab or parts[1] not in vocab or parts[0] + parts[1] not in vocab:
            raise unsupported("merge", entry)
        pairs.append(tuple(parts))
    rank = {pair: index for index, pair in enumerate(pairs)}
    produced: dict[str, int] = {}
    for (left, right), index in rank.items():
        produced[left + right] = max(produced.get(left + right, -1), index)
    for (left, right), index in rank.items():
        if produced.get(left, -1) >= index or produced.get(right, -1) >= index:
            raise unsupported("merge", [left, right])


def check_tokenizer_layout(tokenizer_json: bytes) -> None:
    """The bundle's tokenizer must have Qwen2.5's layout, the one the runtime reproduces.

    The runtime rejects any other at load; the build rejects it first, with the same rules.
    """
    raw = json.loads(tokenizer_json)

    def unsupported(part, value):
        return ValueError(f"{FAMILY} tokenizer.json {part} is not supported: {json.dumps(value)}")

    def flag(item, key, fallback=False):
        """A boolean option: absent or null is `fallback`, anything but a boolean is refused."""
        value = item.get(key)
        if value is None:
            return fallback
        if not isinstance(value, bool):
            raise unsupported(key, value)
        return value

    if not isinstance(raw, dict):
        raise unsupported("document", raw)

    def is_nfc(norm):
        if not isinstance(norm, dict):
            return False
        if norm.get("type") == "NFC":
            return True
        children = norm.get("normalizers")
        return norm.get("type") == "Sequence" and isinstance(children, list) and all(map(is_nfc, children))

    normalizer = raw.get("normalizer")
    if normalizer is not None and not is_nfc(normalizer):
        raise unsupported("normalizer", normalizer)
    for field in ("truncation", "padding"):
        if raw.get(field) is not None:
            raise unsupported(field, raw[field])
    post = raw.get("post_processor")
    if post is not None and not (isinstance(post, dict) and post.get("type") == "ByteLevel"):
        raise unsupported("post_processor", post)
    pre = raw.get("pre_tokenizer")
    steps = pre.get("pretokenizers") if isinstance(pre, dict) and pre.get("type") == "Sequence" else None
    split, byte_level = steps if isinstance(steps, list) and len(steps) == 2 else ({}, {})
    pattern = split.get("pattern") if isinstance(split, dict) else None
    if not (
        isinstance(split, dict)
        and isinstance(byte_level, dict)
        and split.get("type") == "Split"
        and isinstance(pattern, dict)
        and pattern.get("Regex") == QWEN_SPLIT_PATTERN
        and split.get("behavior") == "Isolated"
        and not flag(split, "invert")
        and byte_level.get("type") == "ByteLevel"
        and not flag(byte_level, "add_prefix_space", True)
        and not flag(byte_level, "use_regex", True)
    ):
        raise unsupported("pre_tokenizer", pre)
    decoder = raw.get("decoder")
    if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
        raise unsupported("decoder", decoder)
    bpe = raw.get("model")
    if not isinstance(bpe, dict) or not isinstance(bpe.get("vocab"), dict):
        raise unsupported("model", bpe if not isinstance(bpe, dict) else "without a vocab object")
    model_section = {k: v for k, v in bpe.items() if k not in ("vocab", "merges")}
    if (
        model_section.get("type", "BPE") != "BPE"
        or flag(model_section, "byte_fallback")
        or flag(model_section, "ignore_merges")
        or model_section.get("dropout") is not None
        or model_section.get("unk_token") is not None
        or model_section.get("continuing_subword_prefix") not in (None, "")
        or model_section.get("end_of_word_suffix") not in (None, "")
    ):
        raise unsupported("model", model_section)
    check_merges(bpe["vocab"], bpe.get("merges"), unsupported)
    added = raw.get("added_tokens", [])
    if not isinstance(added, list):
        raise unsupported("added_tokens", added)
    for token in added:
        if (
            not isinstance(token, dict)
            or not isinstance(token.get("content"), str)
            or type(token.get("id")) is not int
        ):
            raise ValueError(
                f"{FAMILY} tokenizer.json added token needs a string content and an integer id: "
                f"{json.dumps(token)}"
            )
        if not token["content"]:
            raise ValueError(f"{FAMILY} tokenizer.json added token needs a nonempty content: {json.dumps(token)}")
        if any(flag(token, option) for option in ("lstrip", "rstrip", "single_word", "normalized")):
            raise unsupported("added token option", token)


def runtime_contract(cfg: VibeVoiceAsrConfig, capacity: int, tokenizer_json: bytes) -> dict:
    check_tokenizer_layout(tokenizer_json)
    check_tokenizer_ids(tokenizer_json, cfg.decoder.vocab_size)
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
    for name in ("tensor_parallel_size", "context_parallel_size"):
        value = getattr(request, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{FAMILY} {name} must be an integer, got {value!r}")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
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
