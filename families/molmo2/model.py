# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Molmo2-8B: SigLIP features, masked attention pooling, and a Qwen3-derived decoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


class Weights:
    def __init__(self, root, precision="fp32"):
        self.root = Path(root)
        self.precision = precision
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

    def __call__(self, name):
        with safe_open(self.root / self.index[name], framework="numpy") as source:
            value = source.get_tensor(name).astype(np.float32)
            if self.precision == "bf16":
                import ml_dtypes

                value = value.astype(ml_dtypes.bfloat16).astype(np.float32)
            return value


def validate_request(
    config,
    *,
    backend,
    context_parallel_size,
    dynamic_kv_cache,
    fp32_layers,
    image_height,
    image_width,
    max_batch_size,
    mtp_seq_len,
    precision,
    quantization,
    task,
    tensor_parallel_size,
    video_num_frames,
):
    if config.get("model_type") != "molmo2" or config.get("architectures") != [
        "Molmo2ForConditionalGeneration"
    ]:
        raise ValueError("molmo2 requires Molmo2ForConditionalGeneration metadata")
    text = config["text_config"]
    if (
        text.get("norm_after")
        or text.get("qk_norm_type") != "qwen3"
        or not text.get("use_qk_norm")
        or text.get("qkv_bias")
        or text.get("rope_scaling")
        or text.get("rope_scaling_layers")
        or text.get("hidden_act", "silu") != "silu"
    ):
        raise ValueError("unsupported Molmo2 decoder topology")
    if task != "vision_language_generation" or backend != "trt":
        raise ValueError("molmo2 requires task=vision_language_generation and backend=trt")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("molmo2 supports fp32 or bf16 compute")
    if dynamic_kv_cache:
        raise NotImplementedError("molmo2 does not support dynamic_kv_cache")
    if (
        max_batch_size != 1
        or tensor_parallel_size != 1
        or context_parallel_size != 1
        or fp32_layers
        or mtp_seq_len is not None
        or quantization not in {None, "none"}
        or image_height is not None
        or image_width is not None
        or video_num_frames is not None
    ):
        raise ValueError("molmo2 supports unquantized single-device, single-image requests")
    if config["vit_config"]["image_default_input_size"] != [378, 378]:
        raise ValueError("molmo2 requires the checkpoint's 378x378 image crops")


def validate_image_contract(root, config):
    """Reject checkpoint variants that differ from native crop and token semantics."""
    expected_tokens = {
        "image_start_token_id": 151936,
        "image_end_token_id": 151937,
        "image_patch_id": 151938,
        "image_col_id": 151939,
        "low_res_image_start_token_id": 151940,
    }
    if any(config.get(key) != value for key, value in expected_tokens.items()):
        raise ValueError("unsupported Molmo2 image token IDs")
    vision, adapter = config["vit_config"], config["adapter_config"]
    if (
        vision["image_patch_size"] != 14
        or vision["image_num_pos"] != 729
        or vision["hidden_act"] != "gelu_pytorch_tanh"
        or adapter["vit_layers"] != [-3, -9]
        or not adapter["pooling_attention_mask"]
        or adapter["hidden_act"] != "silu"
        or adapter["hidden_size"] != vision["hidden_size"]
        or adapter["head_dim"] != vision["head_dim"]
        or adapter["num_attention_heads"] != vision["num_attention_heads"]
        or adapter["num_key_value_heads"] != vision["num_attention_heads"]
        or vision["num_key_value_heads"] != vision["num_attention_heads"]
        or adapter["text_hidden_size"] != config["text_config"]["hidden_size"]
        or not vision["float32_attention"]
        or not adapter["float32_attention"]
        or config.get("quantization_config")
        or config["text_config"].get("quantization_config")
    ):
        raise ValueError("unsupported Molmo2 vision or prepared-weight topology")
    processor = json.loads((root / "preprocessor_config.json").read_text())
    expected = {
        "size": {"height": 378, "width": 378},
        "patch_size": 14,
        "pooling_size": [2, 2],
        "max_crops": 8,
        "overlap_margins": [4, 4],
        "resample": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "do_convert_rgb": True,
    }
    if any(processor.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported Molmo2 image preprocessing configuration")
    framing = json.loads((root / "processor_config.json").read_text())
    if (
        framing.get("image_use_col_tokens") is not True
        or framing.get("use_single_crop_col_tokens") is not False
        or framing.get("use_single_crop_start_token") is not True
    ):
        raise ValueError("unsupported Molmo2 image prompt framing")


def build_decoder(config, weights, precision, maximum, verbose=False):
    from .graph import Graph
    import tensorrt as trt

    g = Graph(precision, verbose)
    t = config["text_config"]
    hidden, heads, kv, dim = (
        t[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim")
    )
    x = g.input(
        "embeddings", trt.float32, (1, hidden), (min(512, maximum), hidden), (maximum, hidden)
    )
    x = g.cast(x, g.dtype)
    positions = g.input("positions", trt.int32, (1,), (min(512, maximum),), (maximum,))
    mask = g.input(
        "attention_mask",
        trt.float32,
        (1, 1, 2),
        (1, min(512, maximum), min(512, maximum) + 1),
        (1, maximum, 2 * maximum),
    )
    frequencies = 1.0 / (t["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    angles = np.arange(maximum, dtype=np.float32)[:, None] * frequencies[None, :]
    angles = np.concatenate([angles, angles], axis=-1)
    cos = g.reshape(g.gather(g.constant(np.cos(angles)), positions, 0), (-1, 1, dim))
    sin = g.reshape(g.gather(g.constant(np.sin(angles)), positions, 0), (-1, 1, dim))
    # The upstream casts its RoPE tables to the decoder dtype before rotation.
    cos, sin = g.cast(cos, g.dtype), g.cast(sin, g.dtype)
    rotate_indices = np.concatenate([np.arange(dim // 2, dim), np.arange(dim // 2)]).astype(
        np.int32
    )
    signs = g.constant(
        np.concatenate([-np.ones(dim // 2), np.ones(dim // 2)])
        .astype(np.float32)
        .reshape(1, 1, dim),
        g.dtype,
    )

    def rotary(y):
        rotated = g.mul(g.gather(y, g.constant(rotate_indices), 2), signs)
        return g.add(g.mul(y, cos), g.mul(rotated, sin))

    def expand(y):
        y = g.reshape(y, (-1, kv, dim), (1, 0, 2))
        if heads != kv:
            y = g.reshape(y, (kv, 1, -1, dim))
            y = g.add(y, g.constant(np.zeros((1, heads // kv, 1, 1), np.float32), g.dtype))
            y = g.reshape(y, (heads, -1, dim))
        return y

    for i in range(t["num_hidden_layers"]):
        p = f"model.transformer.blocks.{i}."
        norm = g.rms(x, weights(p + "attn_norm.weight"), t["layer_norm_eps"])
        fused = weights(p + "self_attn.att_proj.weight")
        q_w, k_w, v_w = np.split(fused, [heads * dim, (heads + kv) * dim])
        q = g.reshape(g.linear(norm, q_w), (-1, heads, dim))
        k = g.reshape(g.linear(norm, k_w), (-1, kv, dim))
        v = g.linear(norm, v_w)
        q = rotary(g.rms(q, weights(p + "self_attn.q_norm.weight"), t["layer_norm_eps"]))
        k = g.reshape(
            rotary(g.rms(k, weights(p + "self_attn.k_norm.weight"), t["layer_norm_eps"])),
            (-1, kv * dim),
        )
        g.output(k, f"new_k_{i}")
        g.output(v, f"new_v_{i}")
        past_k = g.input(f"past_k_{i}", g.dtype, (1, kv * dim), (1, kv * dim), (maximum, kv * dim))
        past_v = g.input(f"past_v_{i}", g.dtype, (1, kv * dim), (1, kv * dim), (maximum, kv * dim))
        k = expand(g.concat([past_k, k], 0))
        v = expand(g.concat([past_v, v], 0))
        q = g.transpose(q, (1, 0, 2))
        attended = g.attention(q, k, v, mask)
        attended = g.reshape(g.transpose(attended, (1, 0, 2)), (-1, heads * dim))
        x = g.add(x, g.linear(attended, weights(p + "self_attn.attn_out.weight")))
        norm = g.rms(x, weights(p + "ff_norm.weight"), t["layer_norm_eps"])
        # Molmo2 stores [up, gate], unlike Qwen's usual gate/up ordering.
        up_w, gate_w = np.split(weights(p + "mlp.ff_proj.weight"), 2)
        ff = g.mul(g.silu(g.linear(norm, gate_w)), g.linear(norm, up_w))
        x = g.add(x, g.linear(ff, weights(p + "mlp.ff_out.weight")))
    x = g.rms(x, weights("model.transformer.ln_f.weight"), t["layer_norm_eps"])
    shape = g.cast(g.net.add_shape(x).get_output(0), trt.int32)
    length = g.gather(shape, g.constant(np.array([0], np.int32)), 0)
    last = g.ew(length, g.constant(np.array([1], np.int32)), trt.ElementWiseOperation.SUB)
    x = g.gather(x, last, 0)
    g.output(g.linear(x, weights("lm_head.weight")), "logits", trt.float32)
    return g.finish()


def build_vision(config, weights, precision, verbose=False):
    from .graph import Graph
    import tensorrt as trt

    g = Graph(precision, verbose)
    v, a = config["vit_config"], config["adapter_config"]
    hidden, dim, heads = v["hidden_size"], v["head_dim"], v["num_attention_heads"]
    x = g.input("pixels", trt.float32, (2, 729, 588), (3, 729, 588), (9, 729, 588))
    x = g.cast(x, g.dtype)
    pooling = g.input("pooling", trt.int32, (392, 4), (588, 4), (2000, 4))
    p = "model.vision_backbone."

    def linear(y, name, bias=True):
        return g.linear(
            y, weights(p + name + ".weight"), weights(p + name + ".bias") if bias else None
        )

    def attention(query, values, name, nq, nk, mask=None):
        q = g.reshape(linear(query, name + ".wq"), (-1, nq, heads, dim), (0, 2, 1, 3))
        k = g.reshape(linear(values, name + ".wk"), (-1, nk, heads, dim), (0, 2, 1, 3))
        val = g.reshape(linear(values, name + ".wv"), (-1, nk, heads, dim), (0, 2, 1, 3))
        y = g.attention(q, k, val, mask)
        y = g.reshape(g.transpose(y, (0, 2, 1, 3)), (-1, nq, hidden))
        return linear(y, name + ".wo")

    x = linear(x, "image_vit.patch_embedding")
    x = g.add(x, g.constant(weights(p + "image_vit.positional_embedding")[None], g.dtype))
    selected = [
        (layer + v["num_hidden_layers"]) % v["num_hidden_layers"] for layer in a["vit_layers"]
    ]
    features = {}
    for i in range(max(selected) + 1):
        name = f"image_vit.transformer.resblocks.{i}"
        y = g.norm(
            x,
            weights(p + name + ".attention_norm.weight"),
            weights(p + name + ".attention_norm.bias"),
            v["layer_norm_eps"],
        )
        x = g.add(x, attention(y, y, name + ".attention", 729, 729))
        y = g.norm(
            x,
            weights(p + name + ".ffn_norm.weight"),
            weights(p + name + ".ffn_norm.bias"),
            v["layer_norm_eps"],
        )
        x = g.add(
            x, linear(g.gelu(linear(y, name + ".feed_forward.w1")), name + ".feed_forward.w2")
        )
        if i in selected:
            features[i] = x
    x = g.reshape(g.concat([features[i] for i in selected], 2), (-1, hidden * len(selected)))
    zero = g.constant(np.zeros((1, 1), np.int32))
    valid = g.ew(
        pooling, g.constant(np.full((1, 1), -1, np.int32)), trt.ElementWiseOperation.GREATER
    )
    indices = g.ew(pooling, zero, trt.ElementWiseOperation.MAX)
    to_pool = g.gather(x, indices, 0)
    validity = g.reshape(g.cast(valid, g.dtype), (-1, 4, 1))
    to_pool = g.mul(to_pool, validity)
    total = g.reduce(to_pool, trt.ReduceOperation.SUM, 1)
    denominator = g.reduce(validity, trt.ReduceOperation.SUM, 1)
    query = g.ew(total, denominator, trt.ElementWiseOperation.DIV)
    mask = g.mul(
        g.ew(
            g.cast(valid, trt.float32),
            g.constant(np.ones((1, 1), np.float32)),
            trt.ElementWiseOperation.SUB,
        ),
        g.scalar(1e30, 2),
    )
    mask = g.reshape(mask, (-1, 1, 1, 4))
    pooled = attention(query, to_pool, "image_pooling_2d", 1, 4, mask)
    y = g.mul(
        g.silu(linear(pooled, "image_projector.w1", False)),
        linear(pooled, "image_projector.w3", False),
    )
    y = g.reshape(linear(y, "image_projector.w2", False), (-1, a["text_hidden_size"]))
    g.output(y, "features", trt.float32)
    return g.finish()


def build(request, writer):
    root = Path(request.model_dir)
    config = json.loads((root / "config.json").read_text())
    validate_request(
        config,
        backend=request.backend,
        context_parallel_size=request.context_parallel_size,
        dynamic_kv_cache=request.dynamic_kv_cache,
        fp32_layers=request.fp32_layers,
        image_height=request.image_height,
        image_width=request.image_width,
        max_batch_size=request.max_batch_size,
        mtp_seq_len=request.mtp_seq_len,
        precision=request.precision,
        quantization=request.quantization,
        task=request.task,
        tensor_parallel_size=request.tensor_parallel_size,
        video_num_frames=request.video_num_frames,
    )
    validate_image_contract(root, config)
    maximum = request.max_sequence_length or 2048
    if not 512 <= maximum <= config["text_config"]["max_position_embeddings"]:
        raise ValueError("molmo2 max_sequence_length must be between 512 and the checkpoint limit")
    weights = Weights(root, request.precision)
    writer.set_header(family="molmo2", task=request.task, backend=request.backend)
    writer.add_bytes(
        "vision.plan", build_vision(config, weights, request.precision, request.verbose)
    )
    writer.add_bytes(
        "engine.plan", build_decoder(config, weights, request.precision, maximum, request.verbose)
    )
    embedding = np.concatenate(
        [weights("model.transformer.wte.embedding"), weights("model.transformer.wte.new_embedding")]
    )
    writer.add_bytes("embedding.f32", embedding.tobytes())
    text = config["text_config"]
    generation = json.loads((root / "generation_config.json").read_text())
    writer.add_json(
        "runtime.json",
        {
            "num_layers": text["num_hidden_layers"],
            "hidden_size": text["hidden_size"],
            "kv_dim": text["num_key_value_heads"] * text["head_dim"],
            "precision": request.precision,
            "max_sequence_length": maximum,
            "embedding_rows": len(embedding),
            "vocab_size": text["vocab_size"],
            "bos_token_id": generation["bos_token_id"],
            "eos_token_id": generation["eos_token_id"],
            "max_crops": 8,
        },
    )
    writer.add_bytes("tokenizer.json", (root / "tokenizer.json").read_bytes())
