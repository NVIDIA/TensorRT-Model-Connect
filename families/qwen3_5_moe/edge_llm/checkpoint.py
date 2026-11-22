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


def original_layouts(text: dict) -> dict:
    """Describe original fused experts and unquantized projections from the validated config."""
    hidden, width = text["hidden_size"], text["moe_intermediate_size"]
    experts, shared = text["num_experts"], text["shared_expert_intermediate_size"]
    q = text["num_attention_heads"] * text["head_dim"]
    kv = text["num_key_value_heads"] * text["head_dim"]
    k = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    v = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    result = {"lm_head.weight": ("BF16", (text["vocab_size"], hidden))}
    for layer, kind in enumerate(text["layer_types"]):
        prefix = f"model.language_model.layers.{layer}."
        shapes = {
            "mlp.experts.gate_up_proj": (experts, 2 * width, hidden),
            "mlp.experts.down_proj": (experts, hidden, width),
            "mlp.shared_expert.gate_proj.weight": (shared, hidden),
            "mlp.shared_expert.up_proj.weight": (shared, hidden),
            "mlp.shared_expert.down_proj.weight": (hidden, shared),
        }
        if kind == "full_attention":
            shapes.update({"self_attn.q_proj.weight": (2 * q, hidden),
                           "self_attn.k_proj.weight": (kv, hidden),
                           "self_attn.v_proj.weight": (kv, hidden),
                           "self_attn.o_proj.weight": (hidden, q)})
        else:
            shapes.update({"linear_attn.in_proj_qkv.weight": (2 * k + v, hidden),
                           "linear_attn.in_proj_z.weight": (v, hidden),
                           "linear_attn.out_proj.weight": (hidden, v)})
        result.update({prefix + name: ("BF16", dims) for name, dims in shapes.items()})
    return result


def validate_layout(tensors: dict, weight_format: str, *, text: dict | None = None) -> None:
    """Reject source/header precision conflicts; upstream validates tensor values."""
    routed_only = weight_format == "nvfp4" and text is not None and text["num_hidden_layers"] == 48
    if weight_format == "none" or routed_only:
        if text is None:
            raise ValueError("Original projection layouts require a text config")
        expected = original_layouts(text)
    else:
        expected = projection_layouts(weight_format)
    if routed_only:
        h, w = text["hidden_size"], text["moe_intermediate_size"]
        for i in range(text["num_hidden_layers"]):
            prefix = f"model.language_model.layers.{i}.mlp.experts."
            del expected[prefix + "gate_up_proj"], expected[prefix + "down_proj"]
            for e in range(text["num_experts"]):
                for name, n, k in (("gate_proj", w, h), ("up_proj", w, h), ("down_proj", h, w)):
                    module = f"{prefix}{e}.{name}."
                    expected.update({module + "weight": ("U8", (n, k // 2)),
                                     module + "weight_scale": ("F8_E4M3", (n, k // 16)),
                                     module + "weight_scale_2": ("F32", ()),
                                     module + "input_scale": ("F32", ())})
        if any(k.startswith("model.language_model.") and k.endswith((".k_proj.k_scale", ".v_proj.v_scale"))
               for k in tensors):
            raise ValueError("Unexpected quantized text KV in the routed-only 122B source")

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


def _shard_path(root: Path, shard: str, allow_snapshot_blobs: bool) -> Path:
    """Allow flat shard names; an HF snapshot may reference only its own repository blobs."""
    relative = Path(shard)
    if relative.name != shard or relative.suffix != ".safetensors":
        raise ValueError("Checkpoint index shard must be an existing contained root file")
    path = root / relative
    if path.is_symlink():
        snapshot = root.resolve()
        repo = snapshot.parent.parent
        blobs = repo / "blobs"
        target = path.resolve()
        if (not allow_snapshot_blobs or snapshot.parent.name != "snapshots"
                or not repo.name.startswith("models--") or blobs.is_symlink()
                or target.parent != blobs):
            raise ValueError("Checkpoint index shard must be an existing contained root file")
        path = target
    if not path.is_file():
        raise ValueError("Checkpoint index shard must be an existing contained root file")
    return path


def validate_checkpoint(root: Path, weight_format: str, *, allow_snapshot_blobs: bool = False) -> dict:
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
        path = _shard_path(root, shard, allow_snapshot_blobs)
        with safe_open(path, framework="numpy") as source:
            for name in source.keys():
                if name in tensors or index.get(name) != shard:
                    raise ValueError("Checkpoint index/header mapping or duplicate tensor conflict")
                tensor = source.get_slice(name)
                tensors[name] = (tensor.get_dtype(), tuple(tensor.get_shape()))
    if set(tensors) != set(index):
        raise ValueError("Checkpoint index references missing tensors")
    if weight_format == "none" or (weight_format == "nvfp4" and (root / "config.json").is_file()):
        validate_layout(tensors, weight_format, text=object_file(root / "config.json")["text_config"])
    else:
        validate_layout(tensors, weight_format)
    return {
        "tensor_count": len(tensors),
        "shard_count": len(set(index.values())),
        "layout": weight_format,
        **({"text_kv_dtype": "fp8" if any(
            k.startswith("model.language_model.") and k.endswith(".k_proj.k_scale") for k in tensors
        ) else "fp16"} if weight_format == "nvfp4" else {}),
    }
