# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the family-owned Phonon transformer, flow head and audio decoder."""

from pathlib import Path
import io
import json


def _validate_architecture(config):
    expected = {
        "model_type": "gradium_phonon_tts",
        "hidden_size": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 2048,
        "latent_dim": 32,
        "audio_prompt_dim": 512,
        "hidden_act": "gelu",
        "attention_bias": False,
        "layer_norm_eps": 1e-5,
        "vocab_size": 4001,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"unsupported Phonon architecture: {name}={config.get(name)!r}")
    flow = config["flow_head_config"]
    for name, value in {
        "flow_type": "lsd",
        "hidden_size": 512,
        "num_hidden_layers": 6,
        "latent_dim": 32,
        "condition_dim": 512,
        "hidden_act": "silu",
        "layer_norm_eps": 1e-6,
        "rms_norm_eps": 1e-5,
    }.items():
        if flow.get(name) != value:
            raise ValueError(f"unsupported Phonon flow head: {name}")
    mimi = config["mimi"]
    expected_codec = {
        "channels": 1,
        "sample_rate": 24000,
        "frame_rate": 12.5,
        "dimension": 512,
        "quantizer_dimension": 32,
        "quantizer_output_dimension": 512,
        "n_filters": 64,
        "n_residual_layers": 1,
        "ratios": [6, 5, 4],
        "kernel_size": 7,
        "last_kernel_size": 3,
        "residual_kernel_size": 3,
        "compress": 2,
        "transformer_d_model": 512,
        "transformer_num_heads": 8,
        "transformer_num_layers": 2,
        "transformer_context": 250,
        "transformer_max_period": 10000.0,
        "gelu": "erf",
    }
    for name, value in expected_codec.items():
        if mimi.get(name) != value:
            raise ValueError(f"unsupported Phonon audio decoder: {name}")
    if config.get("rope_parameters") != {"rope_type": "default", "rope_theta": 10000.0}:
        raise ValueError("unsupported Phonon rotary embeddings")
    if config.get("flow_lm", {}).get("time_rms_norm") != "rms":
        raise ValueError("unsupported Phonon timestep normalization")


def _engine(module, inputs, names, outputs, dynamic_axes, shapes, verbose):
    import torch
    import tensorrt as trt

    data = io.BytesIO()
    torch.onnx.export(
        module.eval(),
        inputs,
        data,
        input_names=names,
        output_names=outputs,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(data.getvalue()):
        raise RuntimeError(
            "Phonon ONNX import failed: "
            + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    for name, bounds in shapes.items():
        profile.set_shape(name, *bounds)
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Phonon TensorRT engine build failed")
    return bytes(plan)


def build(request, writer):
    """Produce native FP32 plans and default-voice/tokenizer assets."""
    if request.task != "text_to_speech" or request.backend != "trt":
        raise ValueError("Phonon supports text_to_speech with the trt backend only")
    if request.precision != "fp32":
        raise ValueError("Phonon currently supports fp32 only")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError("Phonon requires single-device execution")
    if request.max_batch_size != 1:
        raise ValueError("Phonon supports batch size 1 only")
    if request.quantization not in (None, "none") or request.fp32_layers:
        raise ValueError("Phonon does not support quantization or per-layer precision overrides")
    if request.dynamic_kv_cache:
        raise NotImplementedError("phonon does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise ValueError("Phonon does not support speculative decoding")
    if any(
        v is not None for v in (request.image_height, request.image_width, request.video_num_frames)
    ):
        raise ValueError("Phonon does not accept image or video dimensions")
    root = Path(request.model_dir)
    config = json.loads((root / "config.json").read_text())
    _validate_architecture(config)
    import torch
    from safetensors.torch import load_file
    from .graphs import Backbone, Codec, Flow

    torch.set_num_threads(4)
    weights = load_file(root / "model.safetensors")
    weights = {name: value.float() for name, value in weights.items()}
    maximum = request.max_sequence_length or 256
    if maximum > config["max_position_embeddings"]:
        raise ValueError("Phonon sequence length exceeds checkpoint position limit")
    voice = load_file(root / "default-voice.safetensors")["speaker_wavs"][0].T.float()
    if voice.ndim != 2 or voice.shape[1] != 512 or maximum <= voice.shape[0] + 2:
        raise ValueError("Phonon sequence budget must leave room for default voice, text and audio")
    # Project static conditioning at build time, preserving the checkpoint's exact equations.
    p = "flow_lm.condition_provider.conditioners"
    speakers = weights[p + ".num_speakers.embed.weight"][config["num_speakers"]]
    condition = speakers @ weights[p + ".num_speakers.output_proj.weight"].T
    half = config["duration_delta_embedding_dim"] // 2
    phases = (
        config["duration_delta_scale"]
        * config["duration_delta"]
        / (config["duration_delta_max_period"] ** (torch.arange(half).float() / (half - 1)))
    )
    condition += (
        torch.cat((phases.cos(), phases.sin()))
        @ weights[p + ".duration_delta.output_proj.weight"].T
    )
    projection = weights["flow_lm.input_linear.weight"]
    bos = weights["flow_lm.bos_emb"].reshape(32) @ projection.T + condition
    voice = voice @ weights[p + ".speaker_wavs.output_proj.weight"].T
    writer.set_header(family="phonon", task=request.task, backend=request.backend)
    for name, value in {
        "text.embed": weights[p + ".transcript_in_segment.embed.weight"],
        "voice.embed": voice,
        "bos.embed": bos,
        "latent.proj": projection,
        "condition.embed": condition,
    }.items():
        writer.add_bytes(name, value.contiguous().numpy().tobytes())
    writer.add_bytes("tokenizer.json", (root / "tokenizer.json").read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": maximum,
            "voice_frames": len(voice),
            "sample_rate": 24000,
            "samples_per_frame": 1920,
            "temperature": 0.3,
            "eos_threshold": 4.0,
            "tensor_parallel_size": 1,
        },
    )
    writer.add_bytes(
        "backbone.plan",
        _engine(
            Backbone(weights),
            (torch.zeros(1, 8, 512),),
            ["embeddings"],
            ["condition", "eos"],
            {"embeddings": {1: "sequence"}},
            {"embeddings": ((1, 1, 512), (1, min(160, maximum), 512), (1, maximum, 512))},
            request.verbose,
        ),
    )
    writer.add_bytes(
        "flow.plan",
        _engine(
            Flow(weights),
            (torch.zeros(1, 32), torch.zeros(1, 512)),
            ["noise", "condition"],
            ["latent"],
            {},
            {"noise": ((1, 32),) * 3, "condition": ((1, 512),) * 3},
            request.verbose,
        ),
    )
    writer.add_bytes(
        "codec.plan",
        _engine(
            Codec(weights),
            (torch.zeros(1, 2, 32),),
            ["latents"],
            ["pcm"],
            {"latents": {1: "frames"}, "pcm": {2: "samples"}},
            {"latents": ((1, 1, 32), (1, 32, 32), (1, maximum, 32))},
            request.verbose,
        ),
    )
