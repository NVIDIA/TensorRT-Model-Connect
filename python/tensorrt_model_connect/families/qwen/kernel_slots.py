# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct external-kernel slots owned by the Qwen family."""

from ...kernel_slots import ArgumentSpec, KernelSlot, TensorSpec


def _decode_attention_instances(config: object) -> tuple[str, ...]:
    return tuple(
        f"decoder.layers.{index}.decode_attention"
        for index in range(int(getattr(config, "num_hidden_layers")))
    )


def _validate_decode_attention_build(arguments: object) -> None:
    precision = getattr(arguments, "precision", None)
    if precision not in (None, "bf16"):
        raise ValueError("qwen.decode_attention@1 requires --precision bf16")
    if getattr(arguments, "decoder_engine_layout", "split") != "split":
        raise ValueError("qwen.decode_attention@1 requires split decoder engines")
    if int(getattr(arguments, "tensor_parallel_size", 1) or 1) != 1:
        raise ValueError("qwen.decode_attention@1 does not support tensor parallel builds")
    if getattr(arguments, "triattention_stats", None) is not None:
        raise ValueError("qwen.decode_attention@1 cannot be combined with TriAttention")
    if getattr(arguments, "quantize", None) is not None:
        raise ValueError("qwen.decode_attention@1 does not support quantized builds")


DECODE_ATTENTION = KernelSlot(
    id="qwen.decode_attention@1",
    description=(
        "Single-token post-RoPE grouped-query attention over native Qwen KV cache "
        "exposed as 64-token pages."
    ),
    inputs=(
        TensorSpec("query", "bfloat16", ("num_query_heads", "head_dim")),
        TensorSpec(
            "key",
            "bfloat16",
            (1, "num_kv_heads", "kv_capacity", "head_dim"),
        ),
        TensorSpec(
            "value",
            "bfloat16",
            (1, "num_kv_heads", "kv_capacity", "head_dim"),
        ),
        TensorSpec("key_value_lengths", "int32", (1,)),
        TensorSpec("page_offsets", "int32", (1,)),
        TensorSpec("page_table", "int32", ("num_pages",)),
    ),
    outputs=(TensorSpec("context", "bfloat16", "same_as_input_0"),),
    workspace_bytes=0,
    model_arguments=(ArgumentSpec("softmax_scale", "float32"),),
    instances=_decode_attention_instances,
    validate_build=_validate_decode_attention_build,
)


SLOTS = (DECODE_ATTENTION,)
