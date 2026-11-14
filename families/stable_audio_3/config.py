# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the Stable Audio 3 ``model_config.json`` and the T5Gemma encoder config.

The upstream ``stable-audio-3`` library builds the model from this JSON with
many defaults that live in its constructors. Every value that changes the graph
is resolved here once, and every option the family does not implement is
rejected instead of being ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

TEXT_ENCODER_DIR = "t5gemma-b-b-ul2"

# Constructor defaults of the reference modules that the checkpoint config does
# not spell out.
_RMS_NORM_EPS = 1e-5  # transformer.RMSNorm
_QK_RMS_NORM_EPS = 1e-6  # transformer.Attention(qk_norm_eps=...)
_TIMESTEP_FEATURES = 256  # DiffusionTransformer(timestep_features_dim=...)
_NUMBER_FEATURES = 256  # NumberEmbedder(dim=...)
_FOURIER_MIN_FREQ = 0.5  # ExpoFourierFeatures
_FOURIER_MAX_FREQ = 10000.0
_DECODER_FF_MULT = 3  # TransformerResamplingBlock(ff_mult=...)
_ENCODER_CHUNK_SIZE = 32  # StableAudioModel._adapt_sample_size: encoder "chunk_size" default
# StableAudioModel.generate defaults: the largest render (120 s) and the headroom
# added to the requested duration before the latent length is chosen.
DEFAULT_MAX_SAMPLES = 5292032
DURATION_PADDING_SECONDS = 6.0
# ConditionedDiffusionModelWrapper default inference schedule.
_DEFAULT_SAMPLING_SHIFT = {"type": "logsnr", "anchor_length": 2000, "anchor_logsnr": -6.2,
                           "rate": 0.0, "logsnr_end": 2.0}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"stable_audio_3 checkpoint is missing {path.name}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"stable_audio_3 does not support {message}")


def _text_encoder(model_dir: Path, conditioning: dict) -> dict:
    prompt = [entry for entry in conditioning["configs"] if entry["id"] == "prompt"]
    _require(len(prompt) == 1 and prompt[0]["type"] == "t5gemma", "a non-T5Gemma prompt conditioner")
    options = prompt[0]["config"]
    _require(options.get("model_name", "google/t5gemma-b-b-ul2") == "google/t5gemma-b-b-ul2",
             "a T5Gemma variant other than t5gemma-b-b-ul2")
    _require(options.get("subfolder") == TEXT_ENCODER_DIR,
             f"a text encoder outside the {TEXT_ENCODER_DIR}/ subfolder")
    _require(options.get("padding_mode") == "learned", "a prompt padding mode other than learned")
    _require(not options.get("project_out", False), "a projected prompt conditioner")
    raw = _read_json(model_dir / TEXT_ENCODER_DIR / "config.json")
    encoder = raw["encoder"]
    _require(raw.get("model_type") == "t5gemma", "a text encoder that is not T5Gemma")
    _require(encoder["hidden_activation"] == "gelu_pytorch_tanh", "a T5Gemma activation other than tanh GELU")
    _require(not encoder.get("attention_bias", False), "T5Gemma attention biases")
    _require(encoder["num_attention_heads"] == encoder["num_key_value_heads"], "grouped-query T5Gemma attention")
    _require(encoder.get("rope_scaling") in (None, {}), "scaled T5Gemma RoPE")
    max_length = int(options["max_length"])
    # Sliding layers only differ from global ones beyond the window, which the
    # padded prompt never reaches.
    _require(int(encoder["sliding_window"]) >= max_length, "a prompt longer than the T5Gemma sliding window")
    hidden = int(encoder["hidden_size"])
    _require(hidden == int(conditioning["cond_dim"]), "a prompt width that differs from cond_dim")
    return {
        "max_length": max_length,
        "hidden": hidden,
        "layers": int(encoder["num_hidden_layers"]),
        "heads": int(encoder["num_attention_heads"]),
        "head_dim": int(encoder["head_dim"]),
        "intermediate": int(encoder["intermediate_size"]),
        "eps": float(encoder["rms_norm_eps"]),
        "rope_theta": float(encoder["rope_theta"]),
        "query_scale": float(encoder["query_pre_attn_scalar"]) ** -0.5,
        "vocab": int(encoder["vocab_size"]),
        "pad_token_id": int(raw["pad_token_id"]),
    }


def _seconds_conditioner(conditioning: dict) -> dict:
    seconds = [entry for entry in conditioning["configs"] if entry["id"] == "seconds_total"]
    _require(len(seconds) == 1 and seconds[0]["type"] == "number", "a seconds_total conditioner of another type")
    options = seconds[0]["config"]
    _require(options.get("fourier_features_type") == "expo", "learned Fourier features for seconds_total")
    ids = {entry["id"] for entry in conditioning["configs"]}
    _require(ids == {"prompt", "seconds_total"}, f"conditioners {sorted(ids - {'prompt', 'seconds_total'})}")
    return {
        "min": float(options["min_val"]),
        "max": float(options["max_val"]),
        "features": _NUMBER_FEATURES,
        "min_freq": _FOURIER_MIN_FREQ,
        "max_freq": _FOURIER_MAX_FREQ,
    }


def _dit(diffusion: dict, cond_dim: int, latent_dim: int) -> dict:
    _require(diffusion["type"] == "dit", "a diffusion backbone other than the DiT")
    _require(diffusion.get("cross_attention_cond_ids") == ["prompt", "seconds_total"],
             "cross-attention conditioning other than [prompt, seconds_total]")
    _require(diffusion.get("global_cond_ids") == ["seconds_total"], "global conditioning other than seconds_total")
    _require(diffusion.get("local_add_cond_ids") == ["inpaint_mask", "inpaint_masked_input"],
             "local conditioning other than the inpaint mask and masked input")
    for key in ("input_concat_ids", "prepend_cond_ids", "modular_local_cond_configs"):
        _require(not diffusion.get(key), key)
    dit = diffusion["config"]
    allowed = {"io_channels", "embed_dim", "depth", "num_heads", "cond_token_dim", "global_cond_dim",
               "local_add_cond_dim", "global_cond_type", "timestep_features_type", "attn_kwargs", "norm_type",
               "norm_kwargs", "ff_kwargs", "num_memory_tokens"}
    _require(set(dit) <= allowed, f"DiT options {sorted(set(dit) - allowed)}")
    _require(dit["global_cond_type"] == "adaLN", "prepend global conditioning")
    _require(dit["timestep_features_type"] == "expo", "learned timestep features")
    _require(dit["norm_type"] == "rms_norm" and dit.get("norm_kwargs") == {"force_fp32": True},
             "DiT norms other than force-fp32 RMSNorm")
    _require(dit["attn_kwargs"] == {"qk_norm": "rms", "differential": True},
             "DiT attention other than differential attention with RMS qk-norm")
    _require(set(dit.get("ff_kwargs", {})) <= {"mult"}, "DiT feed-forward options other than mult")
    _require(int(dit["io_channels"]) == latent_dim, "a DiT width that differs from the latent width")
    _require(int(dit["cond_token_dim"]) == cond_dim and int(dit["global_cond_dim"]) == cond_dim,
             "conditioning widths that differ from cond_dim")
    _require(int(dit["local_add_cond_dim"]) == latent_dim + 1, "a local conditioning width other than latent+1")
    embed = int(dit["embed_dim"])
    heads = int(dit["num_heads"])
    _require(embed % heads == 0, "an embed_dim not divisible by num_heads")
    head_dim = embed // heads
    return {
        "io_channels": latent_dim,
        "embed": embed,
        "depth": int(dit["depth"]),
        "heads": heads,
        "head_dim": head_dim,
        # ContinuousTransformer(rotary_pos_emb=True): RotaryEmbedding(max(dim_heads // 2, 32)).
        "rope_dim": max(head_dim // 2, 32),
        "cond_dim": cond_dim,
        "local_dim": int(dit["local_add_cond_dim"]),
        "ff_inner": int(embed * float(dit.get("ff_kwargs", {}).get("mult", 4))),
        "memory_tokens": int(dit.get("num_memory_tokens", 0)),
        "norm_eps": _RMS_NORM_EPS,
        "qk_norm_eps": _QK_RMS_NORM_EPS,
        "timestep_features": _TIMESTEP_FEATURES,
        "fourier_min_freq": _FOURIER_MIN_FREQ,
        "fourier_max_freq": _FOURIER_MAX_FREQ,
    }


def _decoder(autoencoder: dict, latent_dim: int, audio_channels: int) -> dict:
    patched = autoencoder["pretransform"]
    _require(patched["type"] == "patched", "an autoencoder pretransform other than patching")
    patch = patched["config"]
    _require(set(patch) <= {"patch_size", "channels"}, f"patched pretransform options {sorted(patch)}")
    _require(int(patch["channels"]) == audio_channels, "a patch channel count that differs from the audio channels")
    bottleneck = autoencoder["bottleneck"]
    _require(bottleneck["type"] == "softnorm", "a bottleneck other than softnorm")
    _require(int(bottleneck["config"].get("noise_augment_dim", 0)) == 0, "bottleneck noise augmentation")
    _require(bool(bottleneck["config"].get("auto_scale")), "a softnorm bottleneck without running_std")
    decoder = autoencoder["decoder"]
    _require(decoder["type"] == "taae_v2", "a decoder other than taae_v2 (SAME)")
    options = decoder["config"]
    # Keys the reference passes through **kwargs without reading them, plus ``mask_noise``: noise the official
    # decoder adds to its new tokens at inference, which the family leaves out on purpose (deterministic decoder).
    ignored = {"use_snake", "use_dilated_conv", "checkpointing", "conv_bias", "mapping_style",
               "enable_inner_layer_dropout", "use_flash", "mask_noise"}
    allowed = ignored | {"out_channels", "channels", "c_mults", "strides", "latent_dim", "transformer_depths",
                         "sinusoidal_blocks", "conformer", "layer_scale", "differential", "dim_heads",
                         "sliding_window", "variable_stride"}
    _require(set(options) <= allowed, f"decoder options {sorted(set(options) - allowed)}")
    _require(len(options["c_mults"]) == 1 and len(options["strides"]) == 1, "a multi-stage SAME decoder")
    _require(not options.get("conformer", False) and not options.get("layer_scale", False),
             "conformer or layer-scale decoder blocks")
    _require(bool(options.get("differential", True)), "non-differential decoder attention")
    _require(bool(options.get("variable_stride", False)), "a decoder without variable stride")
    _require(options.get("sliding_window") == [1, 1], "a decoder window other than one latent on each side")
    _require(int(options["latent_dim"]) == latent_dim, "a decoder latent width that differs from the latents")
    width = int(options["channels"]) * int(options["c_mults"][0])
    stride = int(options["strides"][0])
    out_channels = int(options["out_channels"])
    patch_size = int(patch["patch_size"])
    _require(out_channels == audio_channels * patch_size, "decoder outputs that do not fill the audio patches")
    dim_heads = int(options["dim_heads"])
    _require(width % dim_heads == 0, "a decoder width not divisible by dim_heads")
    depth = int(options["transformer_depths"][0])
    sinusoidal = int(options.get("sinusoidal_blocks", [0])[0])
    return {
        "latent_dim": latent_dim,
        "width": width,
        "stride": stride,
        "depth": depth,
        "heads": width // dim_heads,
        "head_dim": dim_heads,
        # TransformerBlock(add_rope=True): RotaryEmbedding(dim_heads // 2).
        "rope_dim": dim_heads // 2,
        "ff_inner": width * _DECODER_FF_MULT,
        # TransformerResamplingBlock: block i uses sin(pi x) when depth - i < sinusoidal_blocks.
        "sinusoidal": [(depth - index) < sinusoidal for index in range(depth)],
        # One latent token plus `stride` new tokens per latent; the band reaches one
        # latent group on each side.
        "group": stride + 1,
        "window": stride + 1,
        "out_channels": out_channels,
        "patch_size": patch_size,
        "audio_channels": audio_channels,
    }


def _sampling_shift(diffusion: dict) -> dict:
    options = diffusion.get("sampling_distribution_shift_options")
    if options is None:
        return dict(_DEFAULT_SAMPLING_SHIFT)
    shift_type = options.get("type", "full")
    _require(shift_type == "logsnr", f"sampling distribution shift {shift_type!r}")
    _require(set(options) <= {"type", "anchor_length", "anchor_logsnr", "rate", "logsnr_end"},
             f"LogSNR shift options {sorted(options)}")
    shift = {"type": "logsnr", "anchor_length": 2000, "anchor_logsnr": -6.2, "rate": 1.0, "logsnr_end": 2.0}
    shift.update({key: value for key, value in options.items() if key != "type"})
    return shift


def resolve(model_dir: str | Path) -> dict:
    """Return the family's normalized description of one Stable Audio 3 checkpoint."""
    root = Path(model_dir)
    raw = _read_json(root / "model_config.json")
    _require(raw.get("model_type") == "diffusion_cond_inpaint", f"model_type {raw.get('model_type')!r}")
    model = raw["model"]
    autoencoder_block = model["pretransform"]
    _require(autoencoder_block["type"] == "autoencoder", "a pretransform other than the SAME autoencoder")
    autoencoder = autoencoder_block["config"]
    latent_dim = int(autoencoder["latent_dim"])
    audio_channels = int(raw["audio_channels"])
    _require(int(autoencoder["io_channels"]) == audio_channels, "autoencoder io channels that differ from the audio")
    _require(int(model["io_channels"]) == latent_dim, "a model io width that differs from the latent width")
    conditioning = model["conditioning"]
    cond_dim = int(conditioning["cond_dim"])
    _require(not conditioning.get("default_keys") and not conditioning.get("pre_encoded_keys"),
             "conditioner default or pre-encoded keys")
    diffusion = model["diffusion"]
    objective = diffusion.get("diffusion_objective", "v")
    # Post-trained checkpoints. Base (rectified_flow) checkpoints are sampled with classifier-free guidance and
    # negative prompts, which the family does not implement.
    _require(objective == "rf_denoiser", f"diffusion objective {objective!r} (base checkpoints need guidance)")
    decoder = _decoder(autoencoder, latent_dim, audio_channels)
    downsampling = int(autoencoder["downsampling_ratio"])
    _require(downsampling == decoder["stride"] * decoder["patch_size"],
             "a downsampling ratio that differs from stride x patch size")
    encoder_options = autoencoder["encoder"]["config"]
    latent_align = int(encoder_options.get("chunk_size", _ENCODER_CHUNK_SIZE)) // int(encoder_options["strides"][0])
    sample_rate = int(raw["sample_rate"])
    return {
        "sample_rate": sample_rate,
        "audio_channels": audio_channels,
        "latent_dim": latent_dim,
        "downsampling_ratio": downsampling,
        "latent_align": latent_align,
        "objective": objective,
        "sampling_shift": _sampling_shift(diffusion),
        "text": _text_encoder(root, conditioning),
        "seconds": _seconds_conditioner(conditioning),
        "dit": _dit(diffusion, cond_dim, latent_dim),
        "decoder": decoder,
        "headroom_latents": int(DURATION_PADDING_SECONDS * sample_rate / downsampling),
    }


def max_latents(cfg: dict) -> int:
    """Largest latent length of one render: ``StableAudioModel.generate`` caps the
    render at its ``sample_size`` default whatever duration is requested."""
    return DEFAULT_MAX_SAMPLES // cfg["downsampling_ratio"]
