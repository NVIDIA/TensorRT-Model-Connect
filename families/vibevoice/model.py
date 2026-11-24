# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a VibeVoice text-to-speech bundle from the original checkpoint layout."""

from __future__ import annotations

import collections
import json
import math
import numbers
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt

from . import config as config_module
from .codec_engines import build_feedback_engine, build_voice_encoder_engine, stream_state_size
from .head_engine import build_head_engine
from .lm_engine import build_embed_engine, build_lm_engine
from .weights import Checkpoint

FAMILY = "vibevoice"
_TASK = "text_to_speech"
_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1
_PRECISIONS = {"bf16": trt.bfloat16, "fp32": trt.float32}
# Positions per sequence when the build does not state max_sequence_length:
# room for a long prompt and about half an hour of speech at 7.5 frames/s.
DEFAULT_CAPACITY = 16384
MIN_CAPACITY = 64
PREFILL_CHUNK = 512
MAX_VOICE_SECONDS = 60
# Generation defaults of the reference (generation_config.json of the Transformers conversion).
DEFAULT_GUIDANCE_SCALE = 1.3
DEFAULT_DIFFUSION_STEPS = 10


def _resolve_tokenizer_dir(model_dir: Path) -> Path:
    """The checkpoint ships no tokenizer; its preprocessor config names the Qwen2.5 one."""
    if all((model_dir / name).is_file() for name in config_module.TOKENIZER_FILES):
        return model_dir
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            config_module.TOKENIZER_REPOSITORY,
            revision=config_module.TOKENIZER_REVISION,
            allow_patterns=config_module.TOKENIZER_FILES,
        )
    )


def _unique_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{FAMILY} tokenizer JSON repeats the key {key!r}")
        result[key] = value
    return result


def _tokenizer_document(text: bytes) -> dict:
    raw = json.loads(text, object_pairs_hook=_unique_keys)
    if not isinstance(raw, dict):
        raise ValueError(f"{FAMILY} tokenizer file must be a JSON object")
    return raw


# Qwen2.5's tokenizer_config.json switches, all off: the runtime implements none of them.
TOKENIZER_CONFIG_SWITCHES = (
    "add_bos_token",
    "add_eos_token",
    "add_prefix_space",
    "clean_up_tokenization_spaces",
    "split_special_tokens",
)


def check_tokenizer_config(tokenizer_config_json: bytes) -> None:
    """Each switch must be off (false or absent), as in Qwen2.5's tokenizer_config.json."""
    raw = _tokenizer_document(tokenizer_config_json)
    for key in TOKENIZER_CONFIG_SWITCHES:
        value = raw.get(key)
        if value is not None and value is not False:
            raise ValueError(
                f"{FAMILY} tokenizer_config.json {key} is not supported: {json.dumps(value)}"
            )


def check_tokenizer_ids(tokenizer_json: bytes, vocab_size: int) -> None:
    """Every token id must be a row of the language model's embedding table."""
    raw = _tokenizer_document(tokenizer_json)
    ids = list(raw["model"]["vocab"].values())
    # One token per vocabulary id: the reference merges by operand ids, so a second token on
    # an id would merge like the first one, which the runtime's string matching cannot follow.
    counts = collections.Counter(i for i in ids if type(i) is int)
    shared = sorted(i for i, n in counts.items() if n > 1)
    if shared:
        raise ValueError(
            f"{FAMILY} tokenizer vocabulary ids {shared[:5]} are used by more than one token"
        )
    ids += [item.get("id") for item in raw.get("added_tokens", [])]
    stray = [i for i in ids if type(i) is not int or not 0 <= i < vocab_size]
    if stray:
        raise ValueError(
            f"{FAMILY} tokenizer ids {stray[:5]} are outside the embedding table of "
            f"{vocab_size} rows"
        )
    # The ids the reference assigns, as the runtime checks them: it re-adds the added tokens
    # by content, in order, keeping an earlier added token's id, else the vocabulary id, else
    # the next new id, which counts up from the number of vocabulary entries. It gives a new
    # token that id even where a sparse vocabulary gives the row to another token, which then
    # decodes as the added token.
    vocab = raw["model"]["vocab"]
    rows = {i: token for token, i in vocab.items()}
    next_id = len(vocab)
    assigned: dict[str, int] = {}
    for item in raw.get("added_tokens", []):
        content = item.get("content")
        if content in assigned:
            expected = assigned[content]
        elif content in vocab:
            expected = vocab[content]
        else:
            expected = next_id
            next_id += 1
        if item.get("id") != expected:
            raise ValueError(
                f"{FAMILY} tokenizer.json added token {content} has id {item.get('id')}, "
                f"but the reference assigns it {expected}"
            )
        holder = rows.get(expected, content)
        if holder != content:
            raise ValueError(
                f"{FAMILY} tokenizer.json added token {content} has id {expected}, "
                f"the vocabulary row of {holder}"
            )
        assigned[content] = expected


# The Split pattern of Qwen2.5's tokenizer.json, the one the runtime's tokenizer implements.
QWEN_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def check_merges(vocab: dict, merges, unsupported) -> None:
    """The merge table as the reference reads it: one entry format for the whole table
    (pairs, or "left right" strings with exactly one space, after any "#version" header
    lines), and each operand and the result in the vocabulary."""
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


def check_tokenizer_layout(tokenizer_json: bytes) -> None:
    """The bundle's tokenizer must have Qwen2.5's layout, the one the runtime reproduces.

    The runtime rejects any other at load; the build rejects it first, with the same rules.
    """
    raw = _tokenizer_document(tokenizer_json)

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

    def is_nfc(norm):
        if not isinstance(norm, dict):
            return False
        if norm.get("type") == "NFC":
            return True
        children = norm.get("normalizers")
        return (
            norm.get("type") == "Sequence"
            and isinstance(children, list)
            and all(map(is_nfc, children))
        )

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
    steps = (
        pre.get("pretokenizers")
        if isinstance(pre, dict) and pre.get("type") == "Sequence"
        else None
    )
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
            raise ValueError(
                f"{FAMILY} tokenizer.json added token needs a nonempty content: "
                f"{json.dumps(token)}"
            )
        flag(token, "special")  # a boolean, as the runtime reads it
        if any(flag(token, option) for option in ("lstrip", "rstrip", "single_word", "normalized")):
            raise unsupported("added token option", token)


def _special_token_ids(tokenizer_json: dict) -> dict[str, int]:
    """Ids of the control tokens, which must be four distinct special added tokens."""
    added = {
        item["content"]: item["id"]
        for item in tokenizer_json.get("added_tokens", [])
        if item.get("special") is True
    }
    names = {
        "eos": config_module.EOS_TOKEN,
        "audio_bos": config_module.AUDIO_BOS_TOKEN,
        "audio_eos": config_module.AUDIO_EOS_TOKEN,
        "audio_token": config_module.AUDIO_TOKEN,
    }
    missing = [token for token in names.values() if token not in added]
    if missing:
        raise ValueError("vibevoice tokenizer lacks special tokens: " + ", ".join(missing))
    ids = {key: added[token] for key, token in names.items()}
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"{FAMILY} control tokens share an id: {ids}")
    return ids


def rope_inverse_frequencies(cfg) -> np.ndarray:
    """``1 / theta ** (arange(0, d, 2) / d)`` evaluated in float32, as the reference does."""
    exponents = np.arange(0, cfg.head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(
        cfg.head_dim
    )
    return (np.float32(1.0) / np.power(np.float32(cfg.rope_theta), exponents)).astype(np.float32)


def train_sigmas(num_train_timesteps: int) -> np.ndarray:
    """sqrt((1 - alpha_bar) / alpha_bar) of the squaredcos_cap_v2 schedule, in float32."""

    def alpha_bar(time: float) -> float:
        return math.cos((time + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [
        min(
            1 - alpha_bar((i + 1) / num_train_timesteps) / alpha_bar(i / num_train_timesteps), 0.999
        )
        for i in range(num_train_timesteps)
    ]
    alphas = np.float32(1.0) - np.asarray(betas, dtype=np.float64).astype(np.float32)
    # torch.cumprod accumulates float32 in double on the CPU; 1 - alpha_bar cancels
    # badly near t = 0, so a float32 running product would move small sigmas by 1e-4.
    cumulative = np.cumprod(alphas.astype(np.float64)).astype(np.float32)
    return np.sqrt((np.float32(1.0) - cumulative) / cumulative).astype(np.float32)


def sequence_geometry(max_sequence_length, max_position_embeddings: int) -> tuple[int, int]:
    """KV positions per sequence and the prefill chunk, which never exceeds them."""
    if max_sequence_length is not None and (
        isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, numbers.Integral)
    ):
        raise ValueError(
            f"vibevoice max_sequence_length must be an integer, not {max_sequence_length!r}"
        )
    capacity = DEFAULT_CAPACITY if max_sequence_length is None else int(max_sequence_length)
    if not MIN_CAPACITY <= capacity <= max_position_embeddings:
        raise ValueError(
            f"vibevoice max_sequence_length must lie in [{MIN_CAPACITY}, "
            f"{max_position_embeddings}], not {capacity}"
        )
    return capacity, min(PREFILL_CHUNK, capacity)


def _check_schedule(sigmas: np.ndarray, steps: int) -> None:
    """Mirror of the runtime's check_schedule (runtime/scheduler.cpp): the rounded
    "linspace" timesteps of ``steps`` must give a strictly rising log-SNR, in float32."""
    train = len(sigmas)
    if not 1 <= steps <= train:
        raise ValueError(f"{FAMILY} diffusion_steps must lie in [1, {train}]")
    stop = float(train - 1)
    spacing = stop / steps
    previous = None
    for index in range(steps, 0, -1):
        timestep = int(np.rint(stop if index == steps else index * spacing))
        sigma = np.float32(sigmas[timestep])
        alpha = np.float32(1.0) / np.sqrt(sigma * sigma + np.float32(1.0))
        level = np.log(alpha) - np.log(sigma * alpha)
        if not np.isfinite(level) or (previous is not None and not level > previous):
            raise ValueError(f"{FAMILY} diffusion_steps={steps} repeats a noise level of the schedule")
        previous = level


def runtime_contract(
    cfg, *, precision: str, capacity: int, prefill_chunk: int, special: dict[str, int]
) -> dict:
    """runtime.json, checked with the rules the runtime applies when it loads the bundle
    (runtime/runtime_config.cpp), so that every bundle that builds also loads."""
    max_voice_samples = (
        MAX_VOICE_SECONDS * cfg.sample_rate // cfg.acoustic.hop_length * cfg.acoustic.hop_length
    )
    inverse = rope_inverse_frequencies(cfg)
    sigmas = train_sigmas(cfg.ddpm_num_steps)
    state = stream_state_size(cfg)
    if not np.all((inverse > 0.0) & (inverse <= 1.0)):
        raise ValueError(f"{FAMILY} rotary inverse frequencies must lie in (0, 1]")
    # The voice encoder stores the sigma-VAE noise scale in the engine's type.
    stored = np.float32(cfg.acoustic_vae_std)
    if precision == "bf16":
        stored = np.float32(stored.astype(ml_dtypes.bfloat16))
    if not (np.isfinite(stored) and stored >= np.finfo(np.float32).tiny):
        raise ValueError(
            f"{FAMILY} noise scale {cfg.acoustic_vae_std!r} is not a normal {precision} number"
        )
    if not (np.all(np.isfinite(sigmas)) and sigmas[0] > 0.0 and np.all(np.diff(sigmas) > 0.0)):
        raise ValueError(f"{FAMILY} training sigmas must be positive and strictly increasing")
    _check_schedule(sigmas, DEFAULT_DIFFUSION_STEPS)
    if not 1 <= max_voice_samples <= _INT32_MAX or not 1 <= state <= _INT32_MAX:
        raise ValueError(f"{FAMILY} voice-prompt or stream-state size exceeds int32")
    element = 2 if precision == "bf16" else 4
    for what, factors in (
        ("KV cache", (cfg.num_layers, 2, capacity, cfg.num_kv_heads, cfg.head_dim, element)),
        ("prompt buffer", (capacity, cfg.hidden_size, element)),
        ("voice-prompt noise", (max_voice_samples // cfg.acoustic.hop_length, cfg.latent_size, 4)),
    ):
        if math.prod(factors) > _INT64_MAX:
            raise ValueError(f"{FAMILY} {what} size overflows the runtime's sizes")
    control = sorted(special.values())
    return {
        "precision": precision,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_kv_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "capacity": capacity,
        "prefill_chunk": prefill_chunk,
        "rope_inv_freq": [float(value) for value in inverse],
        "control_token_ids": control,
        "eos_token_id": special["eos"],
        "audio_bos_token_id": special["audio_bos"],
        "audio_eos_token_id": special["audio_eos"],
        "audio_token_id": special["audio_token"],
        "system_prompt": config_module.SYSTEM_PROMPT,
        "voice_input_header": config_module.VOICE_INPUT_HEADER,
        "text_input_header": config_module.TEXT_INPUT_HEADER,
        "speech_output_header": config_module.SPEECH_OUTPUT_HEADER,
        "audio_bos_token": config_module.AUDIO_BOS_TOKEN,
        "audio_eos_token": config_module.AUDIO_EOS_TOKEN,
        "audio_token": config_module.AUDIO_TOKEN,
        "sample_rate": cfg.sample_rate,
        "hop_length": cfg.acoustic.hop_length,
        "latent_size": cfg.latent_size,
        "target_dbfs": cfg.target_dbfs,
        "normalize_eps": cfg.normalize_eps,
        "max_voice_samples": max_voice_samples,
        "stream_state_size": state,
        "train_sigmas": [float(value) for value in sigmas],
        "default_guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "default_diffusion_steps": DEFAULT_DIFFUSION_STEPS,
    }


def build(request, writer) -> None:
    """Build one vibevoice bundle: five engines, the tokenizer and runtime.json."""
    if request.task != _TASK:
        raise ValueError(f"vibevoice supports only task={_TASK}")
    if request.backend != "trt":
        raise ValueError("vibevoice supports only backend=trt")
    if request.max_batch_size != 1:
        raise NotImplementedError("vibevoice generates one request at a time (max_batch_size=1)")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("vibevoice runs on one GPU without tensor or context parallelism")
    if request.dynamic_kv_cache:
        raise NotImplementedError("vibevoice does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("vibevoice takes text, not an image size")
    if request.video_num_frames is not None:
        raise NotImplementedError("vibevoice takes text, not video")
    if request.quantization not in (None, "none"):
        raise NotImplementedError("vibevoice does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("vibevoice does not support per-layer precision overrides")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("vibevoice has no MTP head")
    if request.graph_transform is not None:
        raise NotImplementedError("vibevoice does not accept graph transforms")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise NotImplementedError(
            f"vibevoice builds bf16 (the checkpoint's type) or fp32, not {precision}"
        )
    dtype = _PRECISIONS[precision]
    model_dir = Path(request.model_dir)
    cfg = config_module.load(model_dir)
    capacity, prefill_chunk = sequence_geometry(
        request.max_sequence_length, cfg.max_position_embeddings
    )
    tokenizer_dir = _resolve_tokenizer_dir(model_dir)
    tokenizer_bytes = (tokenizer_dir / "tokenizer.json").read_bytes()
    check_tokenizer_config((tokenizer_dir / "tokenizer_config.json").read_bytes())
    check_tokenizer_layout(tokenizer_bytes)
    check_tokenizer_ids(tokenizer_bytes, cfg.vocab_size)
    special = _special_token_ids(_tokenizer_document(tokenizer_bytes))
    # The checkpoint's headers bound the geometry before anything is sized from it.
    weights = Checkpoint(model_dir, cfg)
    runtime = runtime_contract(
        cfg,
        precision=precision,
        capacity=capacity,
        prefill_chunk=prefill_chunk,
        special=special,
    )
    control = runtime["control_token_ids"]
    max_voice_samples = runtime["max_voice_samples"]
    verbose = bool(request.verbose)

    writer.set_header(family="vibevoice", task=request.task, backend=request.backend)
    writer.add_bytes(
        "lm.plan",
        build_lm_engine(
            weights,
            cfg,
            dtype=dtype,
            capacity=capacity,
            prefill_chunk=prefill_chunk,
            control_token_ids=control,
            verbose=verbose,
        ),
    )
    writer.add_bytes(
        "embed.plan",
        build_embed_engine(weights, cfg, dtype=dtype, max_tokens=PREFILL_CHUNK, verbose=verbose),
    )
    writer.add_bytes(
        "diffusion_head.plan", build_head_engine(weights, cfg, dtype=dtype, verbose=verbose)
    )
    writer.add_bytes(
        "feedback.plan", build_feedback_engine(weights, cfg, dtype=dtype, verbose=verbose)
    )
    writer.add_bytes(
        "voice_encoder.plan",
        build_voice_encoder_engine(
            weights, cfg, dtype=dtype, max_samples=max_voice_samples, verbose=verbose
        ),
    )
    writer.add_bytes("tokenizer.json", tokenizer_bytes)
    writer.add_json("runtime.json", runtime)
