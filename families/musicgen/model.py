# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5 + delayed-codebook MusicGen + non-causal 32 kHz EnCodec, in TensorRT.

The initial FP32 path recomputes a bounded causal sequence at each step. It
intentionally exposes no cache or parallelism claim. All neural computation,
including both codec LSTMs, runs in the bundle's TensorRT engines.
"""

import json
from pathlib import Path

import numpy as np

from .support import describe


TEXT_LENGTH = 64


def relative_buckets(length, buckets=32, distance=128):
    relative = np.arange(length)[None, :] - np.arange(length)[:, None]
    half = buckets // 2
    result = (relative > 0).astype(np.int32) * half
    absolute = np.abs(relative)
    exact = half // 2
    large = exact + (
        np.log(np.maximum(absolute, exact) / exact) / np.log(distance / exact) * (half - exact)
    ).astype(np.int32)
    return result + np.where(absolute < exact, absolute, np.minimum(large, half - 1))


def build_text(state, config, verbose):
    import tensorrt as trt
    from .graph import Graph

    g = Graph(verbose)
    n = g.net
    c = config["text_encoder"]
    heads = c["num_heads"]
    ids = n.add_input("input_ids", trt.int32, (1, TEXT_LENGTH))
    mask = n.add_input("attention_mask", trt.float32, (1, 1, 1, TEXT_LENGTH))
    x = n.add_gather(g.const(state["text_encoder.shared.weight"]), ids, 0).get_output(0)
    bucket = relative_buckets(
        TEXT_LENGTH, c["relative_attention_num_buckets"], c["relative_attention_max_distance"]
    )
    table = state[
        "text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ]
    bias = table[bucket].transpose(2, 0, 1)[None]
    mask = g.op(mask, g.const(bias))
    for i in range(c["num_layers"]):
        p = f"text_encoder.encoder.block.{i}"
        y = g.norm(x, state[p + ".layer.0.layer_norm.weight"], epsilon=c["layer_norm_epsilon"])
        q, k, v = [
            g.linear(y, state[p + f".layer.0.SelfAttention.{key}.weight"])
            for key in ("q", "k", "v")
        ]
        y = g.attention(q, k, v, heads, mask, 1.0)
        x = g.op(x, g.linear(y, state[p + ".layer.0.SelfAttention.o.weight"]))
        y = g.norm(x, state[p + ".layer.1.layer_norm.weight"], epsilon=c["layer_norm_epsilon"])
        y = g.linear(y, state[p + ".layer.1.DenseReluDense.wi.weight"])
        y = n.add_activation(y, trt.ActivationType.RELU).get_output(0)
        x = g.op(x, g.linear(y, state[p + ".layer.1.DenseReluDense.wo.weight"]))
    x = g.norm(
        x, state["text_encoder.encoder.final_layer_norm.weight"], epsilon=c["layer_norm_epsilon"]
    )
    x = g.linear(x, state["enc_to_dec_proj.weight"], state["enc_to_dec_proj.bias"])
    return g.finish(x, "encoder_hidden_states")


def build_decoder(state, config, length, verbose):
    import tensorrt as trt
    from .graph import Graph

    g = Graph(verbose)
    n = g.net
    c = config["decoder"]
    h, heads, codebooks = c["hidden_size"], c["num_attention_heads"], c["num_codebooks"]
    ids = n.add_input("audio_ids", trt.int32, (2, codebooks, length))
    encoder = n.add_input("encoder_hidden_states", trt.float32, (2, TEXT_LENGTH, h))
    encoder_mask = n.add_input("encoder_mask", trt.float32, (2, 1, 1, TEXT_LENGTH))
    position = n.add_input("position", trt.int32, (1,))
    root = "decoder.model.decoder"
    embeddings = []
    for i in range(codebooks):
        tokens = n.add_slice(ids, (0, i, 0), (2, 1, length), (1, 1, 1)).get_output(0)
        tokens = g.reshape(tokens, (2, length))
        embeddings.append(
            n.add_gather(g.const(state[root + f".embed_tokens.{i}.weight"]), tokens, 0).get_output(
                0
            )
        )
    x = embeddings[0]
    for y in embeddings[1:]:
        x = g.op(x, y)
    # Matches MusicgenSinusoidalPositionalEmbedding, including its cos/sin order.
    frequency = np.exp(
        np.arange(h // 2, dtype=np.float32) * np.float32(-np.log(10000) / (h // 2 - 1))
    )
    phase = np.arange(length, dtype=np.float32)[:, None] * frequency[None, :]
    x = g.op(x, g.const(np.concatenate((np.cos(phase), np.sin(phase)), axis=1)[None]))
    causal = np.where(
        np.arange(length)[None, :] > np.arange(length)[:, None], np.finfo(np.float32).min, 0.0
    ).astype(np.float32)[None, None]
    mask = g.const(causal)
    for i in range(c["num_hidden_layers"]):
        p = root + f".layers.{i}"
        for name, norm_name, source, attn_mask in (
            ("self_attn", "self_attn_layer_norm", None, mask),
            ("encoder_attn", "encoder_attn_layer_norm", encoder, encoder_mask),
        ):
            y = g.norm(
                x, state[p + f".{norm_name}.weight"], state[p + f".{norm_name}.bias"], epsilon=1e-5
            )
            q = g.linear(y, state[p + f".{name}.q_proj.weight"])
            source = y if source is None else source
            k, v = [g.linear(source, state[p + f".{name}.{key}_proj.weight"]) for key in ("k", "v")]
            y = g.attention(q, k, v, heads, attn_mask, (h // heads) ** -0.5)
            x = g.op(x, g.linear(y, state[p + f".{name}.out_proj.weight"]))
        y = g.norm(
            x,
            state[p + ".final_layer_norm.weight"],
            state[p + ".final_layer_norm.bias"],
            epsilon=1e-5,
        )
        y = g.linear(y, state[p + ".fc1.weight"])
        y = n.add_activation(y, trt.ActivationType.GELU_ERF).get_output(0)
        x = g.op(x, g.linear(y, state[p + ".fc2.weight"]))
    x = g.norm(
        x, state[root + ".layer_norm.weight"], state[root + ".layer_norm.bias"], epsilon=1e-5
    )
    x = n.add_gather(x, position, 1).get_output(0)
    logits = [g.linear(x, state[f"decoder.lm_heads.{i}.weight"]) for i in range(codebooks)]
    concat = n.add_concatenation(logits)
    concat.axis = 1
    return g.finish(concat.get_output(0), "logits")


def build_codec(state, config, frames, verbose):
    import tensorrt as trt
    from .graph import Graph

    g = Graph(verbose)
    n = g.net
    c = config["audio_encoder"]
    codes = n.add_input("audio_codes", trt.int32, (4, frames))
    parts = []
    for i in range(4):
        ids = n.add_slice(codes, (i, 0), (1, frames), (1, 1)).get_output(0)
        table = state[f"audio_encoder.quantizer.layers.{i}.codebook.embed"]
        parts.append(n.add_gather(g.const(table), ids, 0).get_output(0))
    x = parts[0]
    for part in parts[1:]:
        x = g.op(x, part)
    x = g.reshape(x, (1, frames, c["hidden_size"]), (0, 2, 1))

    def conv(x, key, stride=1, transpose=False):
        key = "audio_encoder.decoder." + key + ".conv"
        v, scale = state[key + ".weight_v"], state[key + ".weight_g"]
        weight = v * (scale / np.sqrt(np.sum(v * v, axis=(1, 2), keepdims=True)))
        kernel = weight.shape[-1]
        padding = kernel - stride
        right, left = padding // 2, padding - padding // 2
        if not transpose and padding:
            x = g.pad(x, left, right)
        x = g.conv(x, weight, state[key + ".bias"], stride, transpose)
        if transpose:
            x = n.add_slice(
                x, (0, 0, left), (1, x.shape[1], x.shape[2] - padding), (1, 1, 1)
            ).get_output(0)
        return x

    def elu(x):
        layer = n.add_activation(x, trt.ActivationType.ELU)
        layer.alpha = 1.0
        return layer.get_output(0)

    x = conv(x, "layers.0")
    channels = x.shape[1]
    y = g.reshape(x, (1, channels, frames), (0, 2, 1))
    residual = y
    for i in range(c["num_lstm_layers"]):
        p = "audio_encoder.decoder.layers.1.lstm"
        y = g.lstm(
            y,
            *[
                state[p + f".{key}_l{i}"]
                for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
            ],
        )
    x = g.reshape(g.op(y, residual), (1, frames, channels), (0, 2, 1))
    for stage, ratio in enumerate(c["upsampling_ratios"]):
        index = 3 + stage * 3
        x = conv(elu(x), f"layers.{index}", ratio, True)
        y = conv(elu(x), f"layers.{index + 1}.block.1")
        y = conv(elu(y), f"layers.{index + 1}.block.3")
        x = g.op(x, y)
    x = conv(elu(x), "layers.15")
    return g.finish(x, "waveform")


def validate_request(
    *,
    backend,
    context_parallel_size,
    dynamic_kv_cache,
    fp32_layers,
    image_height,
    image_width,
    max_batch_size,
    max_sequence_length,
    mtp_seq_len,
    precision,
    quantization,
    task,
    tensor_parallel_size,
    video_num_frames,
):
    if task != "text_to_audio" or backend != "trt":
        raise ValueError("MusicGen requires task=text_to_audio and backend=trt")
    if precision != "fp32":
        raise ValueError("MusicGen currently supports fp32 only")
    if tensor_parallel_size != 1 or context_parallel_size != 1:
        raise ValueError("MusicGen currently supports one GPU only")
    if dynamic_kv_cache:
        raise NotImplementedError("musicgen does not support dynamic_kv_cache")
    if max_batch_size != 1:
        raise ValueError("MusicGen supports batch size one with a bounded causal sequence")
    if quantization not in (None, "none") or fp32_layers or mtp_seq_len is not None:
        raise ValueError("MusicGen does not support quantization, mixed precision, or MTP")
    if any(value is not None for value in (image_height, image_width, video_num_frames)):
        raise ValueError("MusicGen does not accept image or video build options")
    length = max_sequence_length or 256
    if not 11 <= length <= 1504:
        raise ValueError("MusicGen max_sequence_length must be in [11, 1504]")
    return length


def build(request, writer):
    from tensorrt_model_connect.model_support import ModelMetadata
    import torch
    from transformers import AutoTokenizer

    length = validate_request(
        backend=request.backend,
        context_parallel_size=request.context_parallel_size,
        dynamic_kv_cache=request.dynamic_kv_cache,
        fp32_layers=request.fp32_layers,
        image_height=request.image_height,
        image_width=request.image_width,
        max_batch_size=request.max_batch_size,
        max_sequence_length=request.max_sequence_length,
        mtp_seq_len=request.mtp_seq_len,
        precision=request.precision,
        quantization=request.quantization,
        task=request.task,
        tensor_parallel_size=request.tensor_parallel_size,
        video_num_frames=request.video_num_frames,
    )
    root = Path(request.model_dir)
    config = json.loads((root / "config.json").read_text())
    if describe(ModelMetadata(config, {})) is None:
        raise ValueError("MusicGen requires the mono facebook/musicgen-medium architecture")
    a, t, d = (config[key] for key in ("audio_encoder", "text_encoder", "decoder"))
    if (
        a["use_causal_conv"]
        or a["use_conv_shortcut"]
        or a["normalize"]
        or a["upsampling_ratios"] != [8, 5, 4, 4]
        or a["num_residual_layers"] != 1
        or a["num_filters"] != 64
        or a["norm_type"] != "weight_norm"
        or t["feed_forward_proj"] != "relu"
        or d["activation_function"] != "gelu"
        or d["scale_embedding"]
    ):
        raise ValueError("MusicGen checkpoint topology differs from the supported architecture")
    state = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    state = {key: tensor.float().numpy() for key, tensor in state.items()}
    writer.set_header(family="musicgen", task=request.task, backend=request.backend)
    for name, function, args in (
        ("text.plan", build_text, ()),
        ("decoder.plan", build_decoder, (length,)),
        ("codec.plan", build_codec, (length - 4,)),
    ):
        print(f"[musicgen] Building {name}", flush=True)
        writer.add_bytes(name, function(state, config, *args, request.verbose))
    writer.add_json(
        "runtime.json",
        {
            "sequence_length": length,
            "audio_frames": length - 4,
            "text_length": TEXT_LENGTH,
            "hidden_size": d["hidden_size"],
            "codebooks": 4,
            "vocab_size": 2048,
            "pad_token_id": 2048,
            "sample_rate": 32000,
            "samples_per_frame": 640,
        },
    )
    # Serialize the backend actually used by the pinned reference library.
    # Transformers 5.2's T5 tokenizer reconstructs it from the checkpoint's
    # vocabulary, and its normalization can differ from the legacy JSON file.
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        writer.add_bytes(name, (root / name).read_bytes())
