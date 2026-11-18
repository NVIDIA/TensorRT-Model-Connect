# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text-only Qwen2.5 Omni Thinker with family-owned TensorRT graphs and fixed KV storage."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt


def load_weights(directory):
    from safetensors import safe_open

    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {
        key: shard
        for key, shard in index.items()
        if key.startswith("thinker.model.") or key == "thinker.lm_head.weight"
    }
    weights = {}
    for shard in sorted(set(selected.values())):
        with safe_open(str(directory / shard), framework="pt", device="cpu") as handle:
            for key in selected:
                if selected[key] == shard:
                    weights[key.removeprefix("thinker.")] = handle.get_tensor(key).float().numpy()
    return weights


def build_engine(weights, cfg, length, stream, verbose=False):
    import torch
    from transformers import Qwen2_5OmniTextConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniRotaryEmbedding

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    width, heads, kv, dim = (
        cfg[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim")
    )

    def const(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(value.shape, value).get_output(0)

    def op(a, b, kind):
        return net.add_elementwise(a, b, kind).get_output(0)

    def add(a, b):
        return op(a, b, trt.ElementWiseOperation.SUM)

    def mul(a, b):
        return op(a, b, trt.ElementWiseOperation.PROD)

    def reshape(x, shape):
        layer = net.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def linear(x, key):
        result = net.add_matrix_multiply(
            x,
            trt.MatrixOperation.NONE,
            const(weights[key + ".weight"]),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        if key + ".bias" in weights:
            result = add(result, const(weights[key + ".bias"].reshape(1, -1)))
        return result

    def norm(x, key):
        axis = len(x.shape) - 1
        average = net.add_reduce(mul(x, x), trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
        ones = (1,) * len(x.shape)
        root = net.add_unary(
            add(average, const(np.full(ones, cfg["rms_norm_eps"]))), trt.UnaryOperation.SQRT
        ).get_output(0)
        gamma_shape = (1,) * axis + (int(x.shape[-1]),)
        return mul(
            op(x, root, trt.ElementWiseOperation.DIV),
            const(weights[key + ".weight"].reshape(gamma_shape)),
        )

    def silu(x):
        return mul(x, net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0))

    token = net.add_input("input_ids", trt.int32, (1,))
    position = net.add_input("position", trt.int32, (1,))
    indices = net.add_constant((length,), np.arange(length, dtype=np.int32)).get_output(0)
    current = op(indices, position, trt.ElementWiseOperation.EQUAL)
    write = reshape(current, (1, 1, length, 1))
    future = op(indices, position, trt.ElementWiseOperation.GREATER)
    mask = reshape(
        mul(net.add_cast(future, trt.float32).get_output(0), const(np.array([-1e9]))),
        (1, 1, 1, 1, length),
    )
    hidden = net.add_gather(const(weights["model.embed_tokens.weight"]), token, 0).get_output(0)
    rotary = Qwen2_5OmniRotaryEmbedding(Qwen2_5OmniTextConfig(**cfg))
    with torch.no_grad():
        cosine, sine = rotary(torch.empty(1, length, width), torch.arange(length)[None, :])
    cosine, sine = tuple(
        reshape(net.add_gather(const(x.numpy()[0]), position, 0).get_output(0), (1, 1, 1, dim))
        for x in (cosine, sine)
    )
    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        normal = norm(hidden, prefix + ".input_layernorm")
        query = reshape(linear(normal, prefix + ".self_attn.q_proj"), (1, heads, 1, dim))
        key = reshape(linear(normal, prefix + ".self_attn.k_proj"), (1, kv, 1, dim))
        value = reshape(linear(normal, prefix + ".self_attn.v_proj"), (1, kv, 1, dim))

        def rope(x, nheads):
            first = net.add_slice(
                x, (0, 0, 0, 0), (1, nheads, 1, dim // 2), (1, 1, 1, 1)
            ).get_output(0)
            second = net.add_slice(
                x, (0, 0, 0, dim // 2), (1, nheads, 1, dim // 2), (1, 1, 1, 1)
            ).get_output(0)
            second = net.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
            concat = net.add_concatenation([second, first])
            concat.axis = 3
            return add(mul(x, cosine), mul(concat.get_output(0), sine))

        query = rope(query, heads)
        key = rope(key, kv)
        old_k = net.add_input(f"cache_k_{layer}", trt.float32, (1, kv, length, dim))
        old_v = net.add_input(f"cache_v_{layer}", trt.float32, (1, kv, length, dim))
        key = net.add_select(write, key, old_k).get_output(0)
        value = net.add_select(write, value, old_v).get_output(0)
        for tensor, name in ((key, f"present_k_{layer}"), (value, f"present_v_{layer}")):
            tensor.name = name
            net.mark_output(tensor)
        query = reshape(query, (1, kv, heads // kv, 1, dim))
        grouped_key = reshape(key, (1, kv, 1, length, dim))
        grouped_value = reshape(value, (1, kv, 1, length, dim))
        scores = net.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, grouped_key, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = add(mul(scores, const(np.full((1, 1, 1, 1, 1), dim**-0.5))), mask)
        probability = net.add_softmax(scores)
        probability.axes = 16
        attended = net.add_matrix_multiply(
            probability.get_output(0),
            trt.MatrixOperation.NONE,
            grouped_value,
            trt.MatrixOperation.NONE,
        ).get_output(0)
        hidden = add(
            hidden, linear(reshape(attended, (1, heads * dim)), prefix + ".self_attn.o_proj")
        )
        normal = norm(hidden, prefix + ".post_attention_layernorm")
        inner = mul(
            silu(linear(normal, prefix + ".mlp.gate_proj")), linear(normal, prefix + ".mlp.up_proj")
        )
        hidden = add(hidden, linear(inner, prefix + ".mlp.down_proj"))
    logits = linear(norm(hidden, "model.norm"), "lm_head")
    logits.name = "logits"
    net.mark_output(logits)
    if not builder.build_serialized_network_to_stream(net, options, PlanWriter(stream)):
        raise RuntimeError("Qwen2.5 Omni Thinker engine build failed")


class PlanWriter(trt.IStreamWriter):
    def __init__(self, stream):
        trt.IStreamWriter.__init__(self)
        self.stream = stream

    def write(self, data):
        return self.stream.write(data)


def build(request, writer):
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen2_5_omni does not support dynamic_kv_cache")
    if request.task != "text_continuation" or request.backend != "trt":
        raise ValueError("Qwen2_5Omni requires text_continuation with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.image_height is not None
        or request.image_width is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("Qwen2_5Omni supports one fp32 sequence with a fixed KV profile")
    directory = Path(request.model_dir)
    from transformers import Qwen2_5OmniConfig

    root_config = Qwen2_5OmniConfig.from_pretrained(directory)
    cfg = root_config.thinker_config.text_config.to_dict()
    cfg["head_dim"] = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    cfg["eos_token_id"] = root_config.thinker_config.eos_token_id
    length = request.max_sequence_length or 128
    if (
        cfg["hidden_act"] != "silu"
        or cfg["use_sliding_window"]
        or cfg["tie_word_embeddings"]
        or cfg["rope_parameters"]["rope_type"] != "default"
    ):
        raise ValueError("Unsupported Qwen2.5 Omni Thinker text decoder configuration")
    if not 2 <= length <= min(1024, cfg["max_position_embeddings"]):
        raise ValueError("Qwen2.5 Omni Thinker fixed KV profile must be in [2, 1024]")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    writer.set_header(family="qwen2_5_omni", task=request.task, backend=request.backend)
    with writer.open_section("model.plan") as section:
        build_engine(load_weights(directory), cfg, length, section, request.verbose)
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=cfg["hidden_size"],
            layers=cfg["num_hidden_layers"],
            kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            vocab_size=cfg["vocab_size"],
            eos_id=cfg["eos_token_id"],
        ),
    )
