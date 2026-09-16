# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned pure TensorRT build for NVIDIA NemotronLabs VoiceChat 11B."""

from __future__ import annotations

import gc
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ModelConfig


VOICECHAT_MODEL_ID = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
TEXT_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
TEXT_MODEL_REVISION = "6533e8de2c68e4536bf7c411d7a3ce5734111476"
_TEXT_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

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


def _normalize_quantization(value: str | None) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none"}:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"int8", "int8_sq"}:
        return "int8_sq"
    raise ValueError(
        "VoiceChat experimental quantization supports only int8/int8_sq"
    )


def _thinker_quantized_weight_names(weights: dict[str, Any]) -> list[str]:
    """Return every static thinker GEMM selected by the W8A8 path."""
    names: list[str] = []
    for layer_idx, layer_type in enumerate(weights["_layer_types"]):
        prefix = f"layer.{layer_idx}"
        if layer_type == "mamba2":
            names.extend((f"{prefix}.mamba_in_proj", f"{prefix}.mamba_out_proj"))
        elif layer_type == "mlp":
            names.extend((f"{prefix}.w_up", f"{prefix}.w_down"))
        elif layer_type == "attention":
            names.extend(f"{prefix}.{suffix}" for suffix in ("w_q", "w_k", "w_v", "w_o"))
    # The language head remains FP16; the sibling function head uses W8A8.
    names.append("w_function_head")
    return names


def _build_thinker_quant_context(
    weights: dict[str, Any],
    *,
    graph_ops_module: Any | None = None,
):
    """Derive the family-owned runtime-absmax W8A8 Thinker context."""
    from .quantization import VoiceChatQuantContext

    if graph_ops_module is None:
        from . import graph_ops as graph_ops_module

    return VoiceChatQuantContext.from_weights(
        weights,
        _thinker_quantized_weight_names(weights),
        graph_ops_module,
    )


def _resolve_text_assets() -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=TEXT_MODEL_ID,
            revision=TEXT_MODEL_REVISION,
            allow_patterns=list(_TEXT_ASSETS),
        )
    )
    missing = [relative for relative in _TEXT_ASSETS if not (snapshot / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "VoiceChat text asset snapshot is missing required files: " + ", ".join(missing)
        )
    return snapshot


def _runtime_config(
    *,
    thinker: ModelConfig,
    stt: dict[str, Any],
    speech: dict[str, Any],
    precision: str,
    quantization: str | None,
    tts_linear_precision: str,
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
    runtime = {
        "model_type": "nemotron_voicechat",
        "architectures": ["NemotronVoiceChatForConditionalGeneration"],
        "runtime_strategy": "nemotron_voicechat_full_duplex",
        "engine_backend": "trt",
        "precision": precision,
        "vocab_size": thinker.vocab_size,
        "hidden_size": thinker.hidden_size,
        "num_hidden_layers": thinker.num_hidden_layers,
        "num_attention_heads": thinker.num_attention_heads,
        "num_key_value_heads": thinker.num_key_value_heads,
        "head_dim": thinker.head_dim,
        "max_cache_length": max_cache_length,
        "num_mamba_layers": layer_types.count("mamba2"),
        "num_attention_layers": layer_types.count("attention"),
        "d_inner": d_inner,
        "conv_dim": conv_dim,
        "mamba_d_state": int(_THINKER_CONFIG["ssm_state_size"]),
        "mamba_d_conv": int(_THINKER_CONFIG["conv_kernel"]),
        "mamba_nheads": int(_THINKER_CONFIG["mamba_num_heads"]),
        "mamba_head_dim": int(_THINKER_CONFIG["mamba_head_dim"]),
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 12,
        "input_sample_rate": 16000,
        "output_sample_rate": 22050,
        "input_samples_per_frame": 1280,
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
        "rnnt_pred_hidden_size": 640,
        "rnnt_pred_num_layers": 2,
        "rnnt_vocab_size": 1024,
        "rnnt_blank_id": 1024,
        "rnnt_max_symbols_per_step": 10,
        "rnnt_eou_frames": 10,
        "rnnt_bou_frames": 3,
        "rnnt_min_speech_frames": 3,
        "rnnt_min_speech_frames_first_turn": 2,
        "function_max_call_tokens": 512,
        "function_max_response_tokens": 1024,
        "function_max_async_steps": 2048,
        "function_tool_timeout_ms": 15000,
        "function_on_hold_min_pad_frames": 17,
        "tts_hidden_size": int(tts_backbone["hidden_size"]),
        "tts_num_layers": int(tts_backbone["num_hidden_layers"]),
        "tts_num_heads": int(tts_backbone["num_attention_heads"]),
        "tts_num_key_value_heads": int(tts_backbone["num_key_value_heads"]),
        "tts_head_dim": int(tts_backbone["head_dim"]),
        "tts_kv_width": int(tts_backbone["hidden_size"]),
        "tts_max_cache_length": min(
            max_cache_length, int(tts_backbone.get("sliding_window", 7500))
        ),
        "tts_sliding_window_pattern": int(tts_backbone.get("sliding_window_pattern", 6)),
        "tts_max_position_embeddings": int(
            tts_backbone.get("max_position_embeddings", 131072)
        ),
        "tts_linear_precision": tts_linear_precision,
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
        "max_response_frames": 256,
        "tts_text_token_ratio_cap": 16,
        "tts_text_token_ratio_min_tokens": 5,
        "max_pending_input_ms": 30000,
        "max_pending_events": 4096,
        "stream_tick_ms": 80,
        "context_rollover_soft_frames": 1125,
        "context_rollover_hard_frames": 1375,
        "context_memory_max_tokens": 96,
        "voicechat_text_model_id": TEXT_MODEL_ID,
        "voicechat_text_model_revision": TEXT_MODEL_REVISION,
        "default_system_prompt": (
            "You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. "
            "Answer in a spoken, conversational style rather than a written one. Do not repeat "
            "the same sentence over and over again. Start the conversation by greeting the user."
        ),
        "tokenizer_add_special_tokens": False,
        "tokenizer_prefix_ids": [],
        "tokenizer_suffix_ids": [],
    }
    if quantization:
        runtime.update(
            {
                "quantization": {
                    "format": quantization,
                    "scheme": "w8a8",
                    "scale_source": "runtime_absmax",
                    "scope": "thinker_static_gemms_except_lm_head",
                },
                "quantization_format": quantization,
                "quantization_scheme": "w8a8",
                "quantization_scope": "thinker_static_gemms_except_lm_head",
                "quantization_scale_source": "runtime_absmax",
                "quantization_experimental": True,
                "thinker_embedding_precision": "fp16",
                "thinker_lm_head_precision": "fp16",
            }
        )
    return runtime


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build every VoiceChat component and package one native TRT bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("nemotron_voicechat does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("nemotron_voicechat does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("nemotron_voicechat does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("nemotron_voicechat does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("nemotron_voicechat does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "speech_session":
        raise ValueError("nemotron_voicechat supports only task=speech_session")
    if request.precision != "fp32":
        raise ValueError("Nemotron VoiceChat requires precision=fp32")
    if request.backend != "trt":
        raise ValueError("Nemotron VoiceChat requires the TensorRT backend")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("Nemotron VoiceChat requires tensor_parallel_size=1")
    quantization = _normalize_quantization(request.quantization)
    if request.fp32_layers:
        raise NotImplementedError("Nemotron VoiceChat does not support fp32_layers")

    model_path = Path(request.model_dir)
    precision = request.precision
    max_cache_length = int(request.max_sequence_length or 8192)
    if max_cache_length < 512 or max_cache_length > 131072:
        raise ValueError("Nemotron VoiceChat max_sequence_length must be in [512, 131072]")
    mel_length = 3000
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if not matches(raw):
        raise ValueError(f"Not a {VOICECHAT_MODEL_ID} checkpoint: {model_path}")
    # Resolve the pinned tokenizer snapshot before starting any long engine
    # compilation, and reuse the same files for the TTS character tables.
    text_assets = _resolve_text_assets()
    stt, speech = _voicechat_sections(raw)
    thinker = _thinker_config(model_path, precision)
    tts_linear_precision = "fp16" if quantization else "fp32"
    verbose = request.verbose
    writer.set_header(family="nemotron_voicechat", task=request.task, backend=request.backend)

    from . import native_core

    thinker_weights = native_core.VoiceChatThinkerBuilder().load_weights(str(model_path), thinker)
    thinker_quant_ctx = (
        _build_thinker_quant_context(thinker_weights) if quantization else None
    )
    thinker_plan = native_core.build_thinker_engine(
        thinker,
        thinker_weights,
        max_cache_length,
        verbose=verbose,
        quant_ctx=thinker_quant_ctx,
    )
    writer.add_bytes("engine.plan", thinker_plan)
    del thinker_weights, thinker_plan
    gc.collect()

    perception_weights = native_core.load_perception_weights(str(model_path), stt)
    from . import streaming_perception

    perception_stream_first = streaming_perception._build_streaming_encoder(
        perception_weights,
        0,
        first_step=True,
        verbose=verbose,
    )
    perception_stream = streaming_perception._build_streaming_encoder(
        perception_weights,
        0,
        first_step=False,
        verbose=verbose,
    )
    writer.add_bytes("perception.first.plan", perception_stream_first)
    writer.add_bytes("perception.plan", perception_stream)
    writer.add_bytes(
        "mel_filterbank",
        struct.pack("<ii", 257, 128)
        + perception_weights["mel_filterbank"].astype("<f4", copy=False).tobytes(),
    )
    writer.add_bytes(
        "mel.window",
        perception_weights["mel_window"].astype("<f4", copy=False).tobytes(),
    )
    del perception_weights, perception_stream_first, perception_stream
    gc.collect()

    rnnt_weights = native_core.load_rnnt_weights(str(model_path))
    rnnt_predictor = native_core.build_rnnt_predictor(rnnt_weights, verbose=verbose)
    rnnt_joint = native_core.build_rnnt_joint(rnnt_weights, verbose=verbose)
    writer.add_bytes("rnnt.predictor.plan", rnnt_predictor)
    writer.add_bytes("rnnt.joint.plan", rnnt_joint)
    del rnnt_weights, rnnt_predictor, rnnt_joint
    gc.collect()

    from .native_tts import build_tts_sections

    tts_max_cache_length = min(
        max_cache_length,
        int(speech["tts_config"]["backbone_config"].get("sliding_window", 7500)),
    )
    for section_name, payload in build_tts_sections(
        str(model_path),
        raw,
        tokenizer_dir=text_assets,
        max_cache_length=tts_max_cache_length,
        linear_precision=tts_linear_precision,
        verbose=verbose,
    ):
        writer.add_bytes(section_name, payload)

    from .native_codec import build_codec_engine_from_checkpoint

    codec_plan = build_codec_engine_from_checkpoint(
        str(model_path),
        verbose=verbose,
    )
    writer.add_bytes("codec.plan", codec_plan)
    del codec_plan
    gc.collect()

    for filename in _TEXT_ASSETS:
        writer.add_bytes(filename, (text_assets / filename).read_bytes())
    rnnt_vocab_path = model_path / "rnnt_tokenizer/vocab.json"
    rnnt_vocab = json.loads(rnnt_vocab_path.read_text(encoding="utf-8"))
    if isinstance(rnnt_vocab, dict):
        vocabulary: list[str | None] = [None] * (
            max(int(index) for index in rnnt_vocab.values()) + 1
        )
        for token, index in rnnt_vocab.items():
            vocabulary[int(index)] = str(token)
        if any(token is None for token in vocabulary):
            raise ValueError("VoiceChat RNNT vocabulary IDs must be contiguous")
        rnnt_vocab = vocabulary
    if not isinstance(rnnt_vocab, list) or not all(isinstance(token, str) for token in rnnt_vocab):
        raise ValueError("VoiceChat RNNT vocabulary must be a JSON token list or token-to-ID map")
    writer.add_json("rnnt.vocab.json", rnnt_vocab)

    runtime_config = _runtime_config(
        thinker=thinker,
        stt=stt,
        speech=speech,
        precision=precision,
        quantization=quantization,
        tts_linear_precision=tts_linear_precision,
        max_cache_length=max_cache_length,
        mel_length=mel_length,
    )
    writer.add_json("runtime.json", runtime_config)
    writer.add_json(
        "provenance.json",
        {
            "text_model_id": TEXT_MODEL_ID,
            "text_model_revision": TEXT_MODEL_REVISION,
            "quantization": quantization or "none",
            "quantization_scale_source": (
                "runtime_absmax" if quantization else None
            ),
            "tts_linear_precision": tts_linear_precision,
        },
    )
