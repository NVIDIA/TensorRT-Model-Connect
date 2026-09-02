# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned pure TensorRT build for NVIDIA NemotronLabs VoiceChat 11B."""

from __future__ import annotations

import gc
import hashlib
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensorrt_model_connect import trt_compat

from ...bundle_writer import BundleInfo, BundleSection, write_bundle
from ...build_timing import (
    new_build_timing,
    timed_build_phase,
    timed_trt_compile,
    write_build_timing,
)
from ...parallel_config import normalize_parallel_config
from .config import ModelConfig


name = "nemotron_voicechat"
runtime_strategy = "nemotron_voicechat_full_duplex"
VOICECHAT_MODEL_ID = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
VOICECHAT_REVISION = "359ada7b1c60851e40ff08065f9b0340244f27e0"
VOICECHAT_WEIGHT_SHA256 = "d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b"
TEXT_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
TEXT_MODEL_REVISION = "6533e8de2c68e4536bf7c411d7a3ce5734111476"
VOICECHAT_ASSET_SHA256 = {
    "config.json": "2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5",
    "rnnt_tokenizer/tokenizer.model": (
        "07d4e5a63840a53ab2d4d106d2874768143fb3fbdd47938b3910d2da05bfb0a9"
    ),
    "rnnt_tokenizer/tokenizer_config.json": (
        "10e393422195708d9d131f66dda09430dc472d3f7d36c3e4fd0a5135bf4809b0"
    ),
    "rnnt_tokenizer/vocab.json": (
        "35051f795cec22352180d3a3d2a1383ffbe0d968cd14049ad6da254492f3678e"
    ),
    "LICENSE": "c55ed9a3dd7c5df14a2496d8ff0f6b1941f807af1769b09611012c4ec56960a0",
}
TEXT_ASSET_SHA256 = {
    "tokenizer.json": "3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f",
    "tokenizer_config.json": "a3e4d48a0f8b4ce6fd746464199299cbdefe6ee202e34048940da2d9838a9aa6",
    "special_tokens_map.json": ("2a4d2e7403546286e5d75f5b6b3c197490be67fb1e2118e5c60ad5c26e6668b1"),
}

_TEXT_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

_THINKER_CONFIG = {
    "model_type": "nemotron_voicechat",
    "architectures": ["NemotronVoiceChatForConditionalGeneration"],
    "vocab_size": 131072,
    "hidden_size": 4480,
    "intermediate_size": 15680,
    "num_hidden_layers": 56,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1.0e-5,
    "max_position_embeddings": 131072,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 12,
    "hybrid_override_pattern": ("M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"),
    "mamba_num_heads": 128,
    "mamba_head_dim": 80,
    "n_groups": 8,
    "ssm_state_size": 128,
    "conv_kernel": 4,
}


def _voicechat_sections(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_config = raw.get("model")
    if not isinstance(model_config, dict):
        return {}, {}
    stt = model_config.get("stt")
    speech = model_config.get("speech_generation")
    stt_model = stt.get("model") if isinstance(stt, dict) else None
    speech_model = speech.get("model") if isinstance(speech, dict) else None
    return (
        stt_model if isinstance(stt_model, dict) else {},
        speech_model if isinstance(speech_model, dict) else {},
    )


def matches(config: object) -> bool:
    """Recognize the distinctive nested NeMo VoiceChat checkpoint config."""
    raw = getattr(config, "raw", config)
    if not isinstance(raw, dict):
        return False
    stt, speech = _voicechat_sections(raw)
    perception = stt.get("perception") if stt else None
    tts_config = speech.get("tts_config") if speech else None
    codec_config = speech.get("codec_config") if speech else None
    return (
        isinstance(perception, dict)
        and isinstance(tts_config, dict)
        and isinstance(codec_config, dict)
        and str(stt.get("pretrained_llm", "")) == TEXT_MODEL_ID
        and int(codec_config.get("num_quantizers", 0)) == 31
        and int(codec_config.get("codebook_size", 0)) == 1024
    )


def _thinker_config(model_path: Path, precision: str) -> ModelConfig:
    config = ModelConfig.from_json(json.dumps(_THINKER_CONFIG))
    config.raw.update(_THINKER_CONFIG)
    config.raw.update(
        {
            "_model_dir": str(model_path),
            "_fp32_layers": [],
            "_parallel_build_enabled": False,
            "_resolved_build_precision": precision,
        }
    )
    return config


def _validate_options(options: dict[str, object]) -> tuple[str, int, int]:
    precision = str(options.get("precision") or "fp32").lower()
    if precision != "fp32":
        raise ValueError(
            "VoiceChat currently requires fp32 for token-for-token parity with the public reference"
        )
    if options.get("quantize") or options.get("quant_scales"):
        raise ValueError("VoiceChat native builds do not support quantization yet")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("VoiceChat owns fixed recurrent/KV session state")
    if options.get("rtx"):
        raise ValueError("VoiceChat requires the TensorRT backend, not TensorRT-RTX")
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.enabled:
        raise NotImplementedError("VoiceChat native builds currently require one GPU")
    requested_batch = options.get("max_batch_size")
    max_batch_size = 1 if requested_batch is None else int(requested_batch)
    if max_batch_size != 1:
        raise ValueError("VoiceChat full-duplex sessions support batch size 1")
    requested_cache = options.get("max_cache_length")
    max_cache_length = 8192 if requested_cache is None else int(requested_cache)
    if max_cache_length < 512 or max_cache_length > 131072:
        raise ValueError("VoiceChat max_cache_length must be in [512, 131072]")
    family_options = dict(options.get("family_build_options") or {})
    mel_length = int(family_options.pop("mel_length", 3000))
    if family_options:
        raise ValueError(f"Unknown VoiceChat build options: {sorted(family_options)}")
    if mel_length < 17 or mel_length > 12000:
        raise ValueError("VoiceChat mel_length must be in [17, 12000]")
    return precision, max_cache_length, mel_length


def _resolve_text_assets() -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=TEXT_MODEL_ID,
            revision=TEXT_MODEL_REVISION,
            allow_patterns=list(_TEXT_ASSETS),
        )
    )
    _verify_asset_set(snapshot, TEXT_ASSET_SHA256, label="Nemotron text")
    return snapshot


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_asset_set(root: Path, expected: dict[str, str], *, label: str) -> None:
    for relative, expected_digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"{label} asset is missing: {path}")
        digest = _file_sha256(path)
        if digest != expected_digest:
            raise ValueError(
                f"{label} asset SHA-256 mismatch for {relative}: "
                f"expected {expected_digest}, got {digest}"
            )


def _verify_exact_checkpoint(model_path: Path) -> None:
    weights_path = model_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"VoiceChat checkpoint weights are missing: {weights_path}")

    # Hugging Face snapshots link to a content-addressed blob whose filename is
    # the LFS SHA-256. Arbitrary/local files are streamed once before we attach
    # the pinned public revision to bundle provenance.
    resolved = weights_path.resolve()
    digest = resolved.name.lower() if weights_path.is_symlink() else ""
    if resolved.parent.name != "blobs" or digest != VOICECHAT_WEIGHT_SHA256:
        digest = _file_sha256(weights_path)
    if digest != VOICECHAT_WEIGHT_SHA256:
        raise ValueError(
            "VoiceChat model.safetensors SHA-256 mismatch: "
            f"expected {VOICECHAT_WEIGHT_SHA256}, got {digest}"
        )


def _verify_exact_model_assets(model_path: Path) -> None:
    _verify_exact_checkpoint(model_path)
    _verify_asset_set(model_path, VOICECHAT_ASSET_SHA256, label="VoiceChat")


def _gpu_name() -> str:
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError:
            return ""
    try:
        success = cudart.cudaError_t.cudaSuccess if hasattr(cudart, "cudaError_t") else 0
        status, device = cudart.cudaGetDevice()
        if status != success:
            return ""
        status, properties = cudart.cudaGetDeviceProperties(device)
        if status != success:
            return ""
        device_name = properties.name
        if isinstance(device_name, bytes):
            return device_name.decode("utf-8", errors="replace").rstrip("\x00")
        return str(device_name).rstrip("\x00")
    except Exception:
        return ""


def _runtime_config(
    *,
    thinker: ModelConfig,
    stt: dict[str, Any],
    speech: dict[str, Any],
    precision: str,
    max_cache_length: int,
    mel_length: int,
) -> dict[str, Any]:
    from . import native_core

    layer_types = native_core._parse_layer_types(str(_THINKER_CONFIG["hybrid_override_pattern"]))
    perception = stt["perception"]
    encoder = perception["encoder"]
    tts = speech["tts_config"]
    tts_backbone = tts["backbone_config"]
    codec = speech["codec_config"]
    d_inner = int(_THINKER_CONFIG["mamba_num_heads"]) * int(_THINKER_CONFIG["mamba_head_dim"])
    conv_dim = d_inner + 2 * int(_THINKER_CONFIG["n_groups"]) * int(
        _THINKER_CONFIG["ssm_state_size"]
    )
    return {
        "model_type": name,
        "architectures": ["NemotronVoiceChatForConditionalGeneration"],
        "runtime_strategy": runtime_strategy,
        "engine_backend": "trt",
        "precision": precision,
        "vocab_size": thinker.vocab_size,
        "hidden_size": thinker.hidden_size,
        "num_hidden_layers": thinker.num_hidden_layers,
        "num_attention_heads": thinker.num_attention_heads,
        "num_key_value_heads": thinker.num_key_value_heads,
        "head_dim": thinker.head_dim,
        "max_cache_length": max_cache_length,
        "layer_types": layer_types,
        "num_mamba_layers": layer_types.count("mamba2"),
        "num_attention_layers": layer_types.count("attention"),
        "d_inner": d_inner,
        "conv_dim": conv_dim,
        "mamba_d_state": int(_THINKER_CONFIG["ssm_state_size"]),
        "mamba_d_conv": int(_THINKER_CONFIG["conv_kernel"]),
        "mamba_nheads": int(_THINKER_CONFIG["mamba_num_heads"]),
        "mamba_head_dim": int(_THINKER_CONFIG["mamba_head_dim"]),
        "n_groups": int(_THINKER_CONFIG["n_groups"]),
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 12,
        "voicechat_frame_length_ms": 80,
        "input_sample_rate": 16000,
        "output_sample_rate": 22050,
        "voicechat_input_samples_per_frame": 1280,
        "mel_n_fft": 512,
        "mel_win_length": 400,
        "mel_hop_length": 160,
        "mel_num_bins": int(perception["preprocessor"]["features"]),
        "mel_preemphasis": float(perception["preprocessor"].get("preemph", 0.97)),
        "mel_length": mel_length,
        "perception_hidden_size": int(encoder["d_model"]),
        "perception_num_layers": int(encoder["n_layers"]),
        "perception_num_heads": int(encoder["n_heads"]),
        "perception_att_context_left": int(encoder["att_context_size"][0]),
        "perception_att_context_right": int(encoder["att_context_size"][1]),
        "perception_streaming_cache_left": int(encoder["att_context_size"][0]),
        "perception_streaming_time_cache": 8,
        "perception_streaming_pre_encode_cache": 9,
        "perception_streaming_drop_pre_encoded": 2,
        "voicechat_duplex_text_weight": float(stt.get("duplex_text_channel_weight", 1.0)),
        "voicechat_duplex_audio_weight": float(stt.get("duplex_user_channel_weight", 1.0)),
        "voicechat_duplex_function_weight": float(stt.get("duplex_function_channel_weight", 2.0)),
        "rnnt_pred_hidden_size": 640,
        "rnnt_pred_num_layers": 2,
        "rnnt_vocab_size": 1024,
        "rnnt_blank_id": 1024,
        "rnnt_max_symbols_per_step": 10,
        "voicechat_rnnt_eou_frames": 10,
        "voicechat_rnnt_bou_frames": 3,
        "voicechat_rnnt_min_speech_frames": 3,
        "voicechat_rnnt_min_speech_frames_first_turn": 2,
        "voicechat_function_max_call_tokens": 512,
        "voicechat_function_max_response_tokens": 1024,
        "voicechat_function_max_async_steps": 2048,
        "voicechat_function_tool_timeout_ms": 15000,
        "voicechat_function_on_hold_min_pad_frames": 17,
        "tts_hidden_size": int(tts_backbone["hidden_size"]),
        "tts_num_layers": int(tts_backbone["num_hidden_layers"]),
        "tts_num_heads": int(tts_backbone["num_attention_heads"]),
        "tts_num_key_value_heads": int(tts_backbone["num_key_value_heads"]),
        "tts_head_dim": int(tts_backbone["head_dim"]),
        "tts_max_cache_length": min(
            max_cache_length, int(tts_backbone.get("sliding_window", 7500))
        ),
        "tts_num_quantizers": int(tts["num_quantizers"]),
        "tts_codebook_size": int(tts["codebook_size"]),
        "tts_guidance_scale": float(speech.get("inference_guidance_scale", 0.2)),
        "tts_top_p": float(speech.get("inference_top_p_or_k", 0.95)),
        "tts_noise_scale": float(speech.get("inference_noise_scale", 0.001)),
        "tts_num_refinement_steps": 8,
        "tts_mog_num_predictions": int(tts["mog_head_config"]["num_predictions"]),
        "codec_num_quantizers": int(codec["num_quantizers"]),
        "codec_codebook_size": int(codec["codebook_size"]),
        "codec_latent_size": int(codec["latent_size"]),
        "codec_wav_to_token_ratio": int(codec["wav_to_token_ratio"]),
        "voicechat_max_response_frames": 256,
        "voicechat_tts_text_token_ratio_cap": 16,
        "voicechat_tts_text_token_ratio_min_tokens": 5,
        "voicechat_max_pending_input_ms": 30000,
        "voicechat_max_pending_events": 4096,
        "voicechat_stream_tick_ms": 80,
        "voicechat_model_id": VOICECHAT_MODEL_ID,
        "voicechat_model_revision": VOICECHAT_REVISION,
        "voicechat_text_model_id": TEXT_MODEL_ID,
        "voicechat_text_model_revision": TEXT_MODEL_REVISION,
        "decoder_engine_layout": "single",
        "tensor_parallel_mode": "single",
        "tensor_parallel_size": 1,
    }


def build(model_dir: str, output_path: str, **options: object) -> None:
    """Build every VoiceChat component and package one native TRT bundle."""
    from . import native_core

    model_path = Path(model_dir)
    precision, max_cache_length, mel_length = _validate_options(options)
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if not matches(raw):
        raise ValueError(f"Not a {VOICECHAT_MODEL_ID} checkpoint: {model_path}")
    _verify_exact_model_assets(model_path)
    stt, speech = _voicechat_sections(raw)
    thinker = _thinker_config(model_path, precision)
    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)
    verbose = bool(options.get("verbose"))
    sections: list[BundleSection] = []

    with timed_build_phase(timing, "weights_loading_thinker_s"):
        thinker_weights = native_core.VoiceChatThinkerBuilder().load_weights(
            str(model_path), thinker
        )
    with timed_trt_compile(timing, "thinker"):
        thinker_plan = native_core.build_thinker_engine(
            thinker,
            thinker_weights,
            max_cache_length,
            verbose=verbose,
        )
    sections.append(BundleSection("engine_plan", thinker_plan))
    del thinker_weights, thinker_plan
    gc.collect()

    with timed_build_phase(timing, "weights_loading_perception_s"):
        perception_weights = native_core.load_perception_weights(str(model_path), stt)
    from . import streaming_perception

    with timed_trt_compile(timing, "perception_stream_first"):
        perception_stream_first = streaming_perception._build_streaming_encoder(
            perception_weights,
            0,
            first_step=True,
            verbose=verbose,
        )
    with timed_trt_compile(timing, "perception_stream"):
        perception_stream = streaming_perception._build_streaming_encoder(
            perception_weights,
            0,
            first_step=False,
            verbose=verbose,
        )
    sections.extend(
        [
            BundleSection("perception_stream_first_plan", perception_stream_first),
            BundleSection("perception_stream_plan", perception_stream),
        ]
    )
    sections.append(
        BundleSection(
            "mel_filterbank",
            struct.pack("<ii", 257, 128)
            + perception_weights["mel_filterbank"].astype("<f4", copy=False).tobytes(),
        )
    )
    sections.append(
        BundleSection(
            "mel_window", perception_weights["mel_window"].astype("<f4", copy=False).tobytes()
        )
    )
    del perception_weights, perception_stream_first, perception_stream
    gc.collect()

    with timed_build_phase(timing, "weights_loading_rnnt_s"):
        rnnt_weights = native_core.load_rnnt_weights(str(model_path))
    with timed_trt_compile(timing, "rnnt_predictor"):
        rnnt_predictor = native_core.build_rnnt_predictor(rnnt_weights, verbose=verbose)
    with timed_trt_compile(timing, "rnnt_joint"):
        rnnt_joint = native_core.build_rnnt_joint(rnnt_weights, verbose=verbose)
    sections.extend(
        [
            BundleSection("rnnt_predictor_plan", rnnt_predictor),
            BundleSection("rnnt_joint_plan", rnnt_joint),
        ]
    )
    del rnnt_weights, rnnt_predictor, rnnt_joint
    gc.collect()

    from .native_tts import build_tts_sections

    tts_max_cache_length = min(
        max_cache_length,
        int(speech["tts_config"]["backbone_config"].get("sliding_window", 7500)),
    )
    with timed_trt_compile(timing, "tts"):
        sections.extend(
            BundleSection(section_name, payload)
            for section_name, payload in build_tts_sections(
                str(model_path),
                raw,
                max_cache_length=tts_max_cache_length,
                verbose=verbose,
            )
        )

    from .native_codec import CODEC_ENGINE_SECTION, build_codec_engine_from_checkpoint

    with timed_trt_compile(timing, "codec"):
        codec_plan = build_codec_engine_from_checkpoint(
            str(model_path),
            verbose=verbose,
        )
    sections.append(BundleSection(CODEC_ENGINE_SECTION, codec_plan))
    del codec_plan
    gc.collect()

    text_assets = _resolve_text_assets()
    for filename in _TEXT_ASSETS:
        sections.append(BundleSection(filename, (text_assets / filename).read_bytes()))
    for source, section_name in (
        (model_path / "rnnt_tokenizer/tokenizer.model", "rnnt_tokenizer.model"),
        (model_path / "rnnt_tokenizer/tokenizer_config.json", "rnnt_tokenizer_config.json"),
        (model_path / "rnnt_tokenizer/vocab.json", "rnnt_vocab.json"),
        (model_path / "LICENSE", "model_license.txt"),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"VoiceChat bundle asset is missing: {source}")
        sections.append(BundleSection(section_name, source.read_bytes()))

    trt_version = trt_compat.tensorrt_version() or "unknown"
    trt_abi = trt_compat.tensorrt_abi(trt_version)
    runtime_config = _runtime_config(
        thinker=thinker,
        stt=stt,
        speech=speech,
        precision=precision,
        max_cache_length=max_cache_length,
        mel_length=mel_length,
    )
    runtime_config["trt_version"] = trt_version
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    sections.append(
        BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
    )
    sections.append(
        BundleSection(
            "provenance.json",
            json.dumps(
                {
                    "model_id": VOICECHAT_MODEL_ID,
                    "model_revision": VOICECHAT_REVISION,
                    "model_safetensors_sha256": VOICECHAT_WEIGHT_SHA256,
                    "model_asset_sha256": VOICECHAT_ASSET_SHA256,
                    "text_model_id": TEXT_MODEL_ID,
                    "text_model_revision": TEXT_MODEL_REVISION,
                    "text_asset_sha256": TEXT_ASSET_SHA256,
                },
                indent=2,
            ).encode("utf-8"),
        )
    )
    info = BundleInfo(
        model_id=VOICECHAT_MODEL_ID,
        model_type=name,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=_gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=thinker.vocab_size,
        hidden_size=thinker.hidden_size,
        num_layers=thinker.num_hidden_layers,
        num_attention_heads=thinker.num_attention_heads,
        num_key_value_heads=thinker.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        tokenizer_add_special_tokens=False,
    )
    with timed_build_phase(timing, "bundle_write_s"):
        write_bundle(output_path, info, sections)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
    print(f"[trtmc build] VoiceChat bundle saved: {output_path}", file=sys.stderr)
