# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a family-owned Cohere Conformer encoder and autoregressive decoder."""

from pathlib import Path
import json
import tensorrt as trt
from .graph import Graph
from .weights import Weights


def _engine(weights, config, kind, max_tokens, verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 1
    settings.clear_flag(trt.BuilderFlag.TF32)
    settings.max_aux_streams = 0
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    graph = Graph(network, weights)
    if kind == "encoder":
        features = network.add_input("features", trt.float32, (1, 128, -1))
        profile.set_shape("features", (1, 128, 2), (1, 128, 800), (1, 128, 3500))
        output = graph.encoder(features, config["encoder"], 3500)
        output.name = "encoded"
    else:
        tokens = network.add_input("tokens", trt.int32, (1, -1))
        encoded = network.add_input("encoded", trt.float32, (1, -1, 1280))
        profile.set_shape("tokens", (1, 1), (1, min(32, max_tokens)), (1, max_tokens))
        profile.set_shape("encoded", (1, 1, 1280), (1, 100, 1280), (1, 438, 1280))
        output = graph.decoder(tokens, encoded, config["transf_decoder"]["config_dict"])
        output.name = "log_probs"
    network.mark_output(output)
    settings.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError(f"Cohere ASR {kind} engine build failed")
    return bytes(plan)


def build(request, writer):
    if request.task != "speech_transcription":
        raise ValueError("cohere_asr supports only speech_transcription")
    if request.backend != "trt":
        raise ValueError("cohere_asr supports only the trt backend")
    if request.precision != "fp32":
        raise ValueError("cohere_asr currently supports precision=fp32")
    if request.dynamic_kv_cache:
        raise NotImplementedError("cohere_asr does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise ValueError("cohere_asr supports one audio request")
    if request.tensor_parallel_size != 1:
        raise ValueError("cohere_asr does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise ValueError("cohere_asr does not support context parallelism")
    if request.image_height is not None or request.image_width is not None:
        raise ValueError("cohere_asr has no image dimensions")
    if request.video_num_frames is not None:
        raise ValueError("cohere_asr has no video input")
    if request.quantization not in (None, "none"):
        raise ValueError("cohere_asr does not support quantization")
    if request.fp32_layers:
        raise ValueError("cohere_asr has no mixed-precision layer selection")
    if request.mtp_seq_len is not None:
        raise ValueError("cohere_asr has no speculative decoding")
    max_tokens = request.max_sequence_length or 256
    if not 16 <= max_tokens <= 1024:
        raise ValueError("cohere_asr requires max_sequence_length between 16 and 1024")
    directory = Path(request.model_dir)
    config = json.loads((directory / "config.json").read_text())
    encoder = config.get("encoder", {})
    decoder = config.get("transf_decoder", {}).get("config_dict", {})
    if (
        config.get("model_type") != "cohere_asr"
        or encoder.get("d_model") != 1280
        or encoder.get("n_layers") != 48
        or encoder.get("conv_kernel_size") != 9
        or encoder.get("subsampling_factor") != 8
        or decoder.get("hidden_size") != 1024
        or decoder.get("num_layers") != 8
        or config.get("supported_languages") != ["en", "ar"]
    ):
        raise ValueError(
            "cohere_asr requires the published 48-layer Arabic/English Conformer checkpoint"
        )
    frontend = config.get("preprocessor", {})
    if (
        config.get("sample_rate") != 16000
        or config.get("max_audio_clip_s") != 35
        or encoder.get("feat_in") != 128
        or encoder.get("subsampling") != "dw_striding"
        or encoder.get("causal_downsampling", False)
        or encoder.get("self_attention_model") != "rel_pos"
        or encoder.get("conv_norm_type") != "batch_norm"
        or encoder.get("xscaling", False)
        or decoder.get("hidden_act") != "relu"
        or decoder.get("learn_positional_encodings", False)
        or not decoder.get("pre_ln")
        or config.get("head", {}).get("num_classes") != 16384
        or any(
            frontend.get(k) != value
            for k, value in {
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": 128,
                "n_fft": 512,
                "log": True,
                "frame_splicing": 1,
                "dither": 1e-5,
                "pad_to": 0,
            }.items()
        )
    ):
        raise ValueError("cohere_asr checkpoint preprocessing or topology is unsupported")
    weights = Weights(directory)
    writer.set_header(family="cohere_asr", task=request.task, backend=request.backend)
    writer.add_bytes(
        "encoder.plan", _engine(weights, config, "encoder", max_tokens, request.verbose)
    )
    writer.add_bytes(
        "decoder.plan", _engine(weights, config, "decoder", max_tokens, request.verbose)
    )
    writer.add_bytes("mel_filters.f32", weights["preprocessor.featurizer.fb"].tobytes())
    writer.add_bytes("window.f32", weights["preprocessor.featurizer.window"].tobytes())
    tokenizer = json.loads((directory / "tokenizer.json").read_text())
    vocabulary = tokenizer["model"]["vocab"]
    pieces = (
        [entry[0] for entry in vocabulary]
        if isinstance(vocabulary, list)
        else [None] * len(vocabulary)
    )
    if isinstance(vocabulary, dict):
        for text, index in vocabulary.items():
            pieces[index] = text
    special = json.loads((directory / "tokenizer_config.json").read_text())["added_tokens_decoder"]
    for index, entry in special.items():
        pieces[int(index)] = entry["content"]
    writer.add_json(
        "tokens.json",
        {"pieces": pieces, "special_ids": [int(i) for i, e in special.items() if e.get("special")]},
    )
    ids = {text: index for index, text in enumerate(pieces)}
    runtime = {
        "sample_rate": 16000,
        "max_samples": 560000,
        "max_tokens": max_tokens,
        "eos_id": 3,
        "encoder_hidden": 1280,
        "arabic_prompt": [
            ids[t]
            for t in [
                "<|startofcontext|>",
                "<|startoftranscript|>",
                "<|emo:undefined|>",
                "<|ar|>",
                "<|ar|>",
                "<|pnc|>",
                "<|noitn|>",
                "<|notimestamp|>",
                "<|nodiarize|>",
            ]
        ],
        "english_prompt": [
            ids[t]
            for t in [
                "<|startofcontext|>",
                "<|startoftranscript|>",
                "<|emo:undefined|>",
                "<|en|>",
                "<|en|>",
                "<|pnc|>",
                "<|noitn|>",
                "<|notimestamp|>",
                "<|nodiarize|>",
            ]
        ],
    }
    writer.add_json("runtime.json", runtime)
