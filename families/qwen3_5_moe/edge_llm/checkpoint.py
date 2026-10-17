# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family source-layout admission using safetensors headers, never weight conversion."""

from pathlib import Path

from .precision import object_file


def projection_layouts(weight_format: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Describe precision-bearing checkpoint fields for the two exact catalog rows."""
    result = {}

    def projection(name, output, input_size, kind):
        if kind == "int4":
            fields = {
                "qweight": ("I32", (input_size // 8, output)),
                "qzeros": ("I32", (input_size // 128, output // 8)),
                "g_idx": ("I32", (input_size,)),
                "scales": ("F16", (input_size // 128, output)),
            }
        elif kind == "nvfp4":
            fields = {
                "weight": ("U8", (output, input_size // 2)),
                "weight_scale": ("F8_E4M3", (output, input_size // 16)),
                "weight_scale_2": ("F32", ()),
                "input_scale": ("F32", ()),
            }
        elif kind == "fp8":
            fields = {
                "weight": ("F8_E4M3", (output, input_size)),
                "weight_scale": ("F32", ()),
                "input_scale": ("F32", ()),
            }
        else:
            fields = {"weight": ("BF16", (output, input_size))}
        result.update({name + "." + key: value for key, value in fields.items()})

    for layer in range(40):
        prefix = f"model.language_model.layers.{layer}."
        for expert in range(256):
            for name, output, input_size in (
                ("gate_proj", 512, 2048),
                ("up_proj", 512, 2048),
                ("down_proj", 2048, 512),
            ):
                projection(
                    f"{prefix}mlp.experts.{expert}.{name}", output, input_size, weight_format
                )
        for name, output, input_size in (
            ("gate_proj", 512, 2048),
            ("up_proj", 512, 2048),
            ("down_proj", 2048, 512),
        ):
            projection(
                prefix + "mlp.shared_expert." + name,
                output,
                input_size,
                "nvfp4" if weight_format == "nvfp4" else "original",
            )
        attention = (
            (
                ("self_attn.q_proj", 8192, 2048),
                ("self_attn.k_proj", 512, 2048),
                ("self_attn.v_proj", 512, 2048),
                ("self_attn.o_proj", 2048, 4096),
            )
            if (layer + 1) % 4 == 0
            else (
                ("linear_attn.in_proj_qkv", 8192, 2048),
                ("linear_attn.in_proj_z", 4096, 2048),
                ("linear_attn.out_proj", 2048, 4096),
            )
        )
        for name, output, input_size in attention:
            projection(
                prefix + name, output, input_size, "fp8" if weight_format == "nvfp4" else "original"
            )
    projection("lm_head", 248320, 2048, "nvfp4" if weight_format == "nvfp4" else "original")
    return result


def validate_layout(tensors: dict, weight_format: str) -> None:
    """Reject source/header precision conflicts; upstream validates tensor values."""
    expected = projection_layouts(weight_format)
    for name, description in expected.items():
        if tensors.get(name) != description:
            raise ValueError(f"Checkpoint precision/shape differs for {name}")
    visual = [name for name in tensors if name.startswith("model.visual.")]
    if not visual:
        raise ValueError("Complete checkpoint requires original visual tensors")
    for name, (dtype, shape) in tensors.items():
        if name in expected or name.startswith("mtp."):
            continue
        if dtype not in {"BF16", "F16", "F32"} or name.endswith(
            (
                ".qweight",
                ".qzeros",
                ".g_idx",
                ".scales",
                ".weight_scale",
                ".weight_scale_2",
                ".input_scale",
            )
        ):
            raise ValueError(f"Unexpected packed tensor outside declared precision modules: {name}")
        if not all(type(size) is int and size > 0 for size in shape):
            raise ValueError(f"Invalid original tensor shape: {name}")


def validate_checkpoint(root: Path, weight_format: str) -> dict:
    """Check every index mapping against actual shard headers, with contained paths.

    Safetensors validates container offsets and shapes without materializing weights.
    GPTQ group-index values and packing remain the pinned Edge builder's authority.
    """
    from safetensors import safe_open

    index = object_file(root / "model.safetensors.index.json").get("weight_map")
    if (
        not isinstance(index, dict)
        or not index
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in index.items())
    ):
        raise ValueError("A complete safetensors index is required")
    tensors = {}
    for shard in sorted(set(index.values())):
        relative = Path(shard)
        path = root / relative
        if (
            relative.name != shard
            or relative.suffix != ".safetensors"
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError("Checkpoint index shard must be an existing contained root file")
        with safe_open(path, framework="numpy") as source:
            for name in source.keys():
                if name in tensors or index.get(name) != shard:
                    raise ValueError("Checkpoint index/header mapping or duplicate tensor conflict")
                tensor = source.get_slice(name)
                tensors[name] = (tensor.get_dtype(), tuple(tensor.get_shape()))
    if set(tensors) != set(index):
        raise ValueError("Checkpoint index references missing tensors")
    validate_layout(tensors, weight_format)
    return {
        "tensor_count": len(tensors),
        "shard_count": len(set(index.values())),
        "layout": weight_format,
    }
