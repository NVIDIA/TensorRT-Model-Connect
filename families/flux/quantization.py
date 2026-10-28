# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev NVFP4 checkpoint reader and TensorRT graph helper.

black-forest-labs/FLUX.2-dev-NVFP4 holds the whole transformer in the original
(non-diffusers) key layout in one safetensors file:

  * NVFP4 projections carry ``weight`` (packed E2M1, uint8 ``[out, in/2]``),
    ``weight_scale`` (per-16 e4m3 ``[out, in/16]``), ``weight_scale_2`` (fp32
    scalar) and ``input_scale`` (fp32 scalar, activation amax / (6 x 448)).
  * Every other tensor (modulation, embedders, norms, and the projections the
    export left in BF16) is a plain BF16 tensor.

The packed bytes and scales are reused bit-exact. No re-quantization happens at
build time. The checkpoint is converted to the diffusers names the builder
already consumes: fused ``qkv`` is split by rows (the three parts share the
scalar scales) and ``adaLN_modulation`` is swapped from (shift, scale) to
(scale, shift).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes  # noqa: F401  (registers bfloat16 for NumPy)
import numpy as np

NVFP4_BLOCK_SIZE = 16
NVFP4_CHECKPOINT_NAMES = ("flux2-dev-nvfp4.safetensors", "flux2-dev-nvfp4-mixed.safetensors")

_BLOCK_STEMS = {
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
}
_QKV_STEMS = {
    "img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}
_SINGLE_STEMS = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}
_GLOBAL_STEMS = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
    "final_layer.adaLN_modulation.1": "norm_out.linear",
}
_KEY_RE = re.compile(
    r"^(?P<stem>.+)\.(?P<leaf>weight|bias|scale|input_scale|weight_scale|weight_scale_2)$"
)
_DOUBLE_RE = re.compile(r"^double_blocks\.(\d+)\.(.+)$")
_SINGLE_RE = re.compile(r"^single_blocks\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class NVFP4Projection:
    """One checkpoint projection: packed FP4 weight plus its scales."""

    packed: np.ndarray  # uint8 [out, in // 2], low nibble first
    block_scale: np.ndarray  # uint8 [out, in // 16], raw e4m3 bytes
    global_scale: float  # weight_scale_2
    input_scale: float  # activation global scale
    out_features: int
    in_features: int


def unswizzle_block_scales(flat: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Undo the cuBLAS 128x4 tile swizzle of block scales; returns ``[rows, cols]``."""
    if rows % 128 or cols % 4:
        raise ValueError(
            f"swizzled block scales need rows % 128 == 0 and cols % 4 == 0, got {rows}x{cols}"
        )
    tiles = flat.reshape(-1, 32, 4, 4).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(rows // 128, cols // 4, 128, 4).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(tiles.reshape(rows, cols))


def swap_nibbles(packed: np.ndarray) -> np.ndarray:
    """High-nibble-first pairs to low-nibble-first pairs."""
    return np.ascontiguousarray(((packed & 0x0F) << 4) | (packed >> 4))


def find_nvfp4_checkpoint(model_dir: str | Path) -> Path:
    """Locate the FLUX.2 NVFP4 file next to the BF16 repo files; refuse ambiguity."""
    found = [Path(model_dir) / name for name in NVFP4_CHECKPOINT_NAMES]
    found = [path for path in found if path.is_file()]
    if not found:
        raise FileNotFoundError(
            "FLUX.2 NVFP4 needs one of "
            f"{', '.join(NVFP4_CHECKPOINT_NAMES)} from black-forest-labs/FLUX.2-dev-NVFP4 "
            f"inside {model_dir}"
        )
    if len(found) > 1:
        raise ValueError(f"Both NVFP4 files exist in {model_dir}; keep exactly one")
    return found[0]


def _split_bfl_key(key: str) -> tuple[str, str]:
    """Return (module stem, leaf) for a checkpoint key; unknown layouts raise."""
    match = _KEY_RE.match(key)
    if match is None:
        raise ValueError(f"unexpected FLUX.2 NVFP4 checkpoint key: {key}")
    return match["stem"], match["leaf"]


def _diffusers_names(stem: str) -> tuple[list[str], bool]:
    """Map a BFL module stem to diffusers stems. Second value is True for a fused qkv."""
    double = _DOUBLE_RE.match(stem)
    if double is not None:
        index, rest = double.groups()
        prefix = f"transformer_blocks.{index}."
        if rest in _QKV_STEMS:
            return [prefix + name for name in _QKV_STEMS[rest]], True
        if rest in _BLOCK_STEMS:
            return [prefix + _BLOCK_STEMS[rest]], False
    single = _SINGLE_RE.match(stem)
    if single is not None:
        index, rest = single.groups()
        if rest in _SINGLE_STEMS:
            return [f"single_transformer_blocks.{index}." + _SINGLE_STEMS[rest]], False
    if stem in _GLOBAL_STEMS:
        return [_GLOBAL_STEMS[stem]], False
    raise ValueError(f"unexpected FLUX.2 NVFP4 checkpoint module: {stem}")


def read_nvfp4_checkpoint(
    path: str | Path,
) -> tuple[dict[str, NVFP4Projection], dict[str, np.ndarray]]:
    """Read the checkpoint into (quantized projections, plain float32 tensors).

    Both dicts use diffusers names without the ``.weight`` suffix for projections
    (``transformer_blocks.0.ff.linear_in``) and with the suffix for plain tensors
    (``transformer_blocks.0.attn.norm_q.weight``). Plain 2-D weights stay in the
    checkpoint ``[out, in]`` layout; the caller transposes them like the BF16 loader.
    """
    import torch
    from safetensors import safe_open

    projections: dict[str, NVFP4Projection] = {}
    plain: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="pt") as reader:
        keys = set(reader.keys())
        for key in sorted(keys):
            stem, leaf = _split_bfl_key(key)
            names, fused = _diffusers_names(stem)
            if leaf in ("weight_scale", "weight_scale_2", "input_scale"):
                continue  # consumed with the packed weight
            if leaf == "weight" and f"{stem}.weight_scale" in keys:
                packed = reader.get_tensor(key).numpy()
                block_scale = reader.get_tensor(f"{stem}.weight_scale").view(torch.uint8).numpy()
                global_scale = float(reader.get_tensor(f"{stem}.weight_scale_2").float())
                input_scale = float(reader.get_tensor(f"{stem}.input_scale").float())
                if packed.dtype != np.uint8 or packed.ndim != 2:
                    raise ValueError(f"{key}: expected packed uint8 [out, in/2]")
                out_features, half_in = packed.shape
                in_features = half_in * 2
                if block_scale.size != out_features * (in_features // NVFP4_BLOCK_SIZE):
                    raise ValueError(f"{stem}.weight_scale has shape {block_scale.shape}")
                block_scale = unswizzle_block_scales(
                    block_scale.reshape(-1), out_features, in_features // NVFP4_BLOCK_SIZE
                )
                packed = swap_nibbles(packed)
                if not (
                    np.isfinite(global_scale)
                    and global_scale > 0
                    and np.isfinite(input_scale)
                    and input_scale > 0
                ):
                    raise ValueError(f"{stem}: scales must be finite and positive")
                parts = 3 if fused else 1
                if out_features % parts:
                    raise ValueError(f"{key}: rows not divisible by {parts}")
                rows = out_features // parts
                for index, name in enumerate(names):
                    sl = slice(index * rows, (index + 1) * rows)
                    projections[name] = NVFP4Projection(
                        packed=np.ascontiguousarray(packed[sl]),
                        block_scale=np.ascontiguousarray(block_scale[sl]),
                        global_scale=global_scale,
                        input_scale=input_scale,
                        out_features=rows,
                        in_features=in_features,
                    )
                continue
            tensor = reader.get_tensor(key).float().numpy()
            if leaf == "scale":
                leaf = "weight"
            if fused:
                parts = np.split(tensor, 3, axis=0)
            else:
                parts = [tensor]
            if stem == "final_layer.adaLN_modulation.1" and leaf == "weight":
                shift, scale = np.split(tensor, 2, axis=0)
                parts = [np.concatenate([scale, shift], axis=0)]
            for name, part in zip(names, parts):
                plain[f"{name}.{leaf}"] = np.ascontiguousarray(part)
    return projections, plain


def add_nvfp4_matmul(network, activation, layer: NVFP4Projection, keep_alive: list, constant):
    """activation [seq, in] -> [seq, out] through dynamic-quantized FP4 GEMM.

    ``constant(value)`` must add a fp32 ``(1,)`` constant and return its tensor.
    """
    import tensorrt as trt

    output_dtype = activation.dtype
    n, k = layer.out_features, layer.in_features
    keep_alive.append(layer.packed)
    keep_alive.append(layer.block_scale)

    w_fp4 = trt.Weights(trt.DataType.FP4, layer.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp4).get_output(0)
    w_bs = trt.Weights(trt.DataType.FP8, layer.block_scale.ctypes.data, layer.block_scale.size)
    w_bs_const = network.add_constant((n, k // NVFP4_BLOCK_SIZE), w_bs).get_output(0)

    w_gs = constant(layer.global_scale)
    w_effective_scale = network.add_dequantize(w_bs_const, w_gs, trt.float32)
    w_dequant = network.add_dequantize(w_const, w_effective_scale.get_output(0), trt.float32)
    w_dequant.axis = 1
    w_compute = network.add_cast(w_dequant.get_output(0), output_dtype).get_output(0)

    act_gs = constant(layer.input_scale)
    a_dynq = network.add_dynamic_quantize(
        activation, 1, NVFP4_BLOCK_SIZE, trt.DataType.FP4, trt.DataType.FP8
    )
    a_dynq.set_input(1, act_gs)
    reconstruction_scale = network.add_cast(act_gs, output_dtype).get_output(0)
    a_effective_scale = network.add_dequantize(
        a_dynq.get_output(1), reconstruction_scale, output_dtype
    )
    a_dequant = network.add_dequantize(
        a_dynq.get_output(0), a_effective_scale.get_output(0), output_dtype
    )
    a_dequant.axis = 1

    output = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_compute,
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    return (
        output
        if output.dtype == output_dtype
        else network.add_cast(output, output_dtype).get_output(0)
    )


def load_flux2_dit_weights_nvfp4(path: str | Path):
    """Build the builder's weight dict from the NVFP4 file.

    Plain 2-D weights are transposed to ``[in, out]`` float32 like
    ``load_flux2_dit_weights``. Quantized projections appear only as
    ``<prefix>.nvfp4`` entries; there is no ``<prefix>.weight`` for them, so a
    projection that is not in the checkpoint's quantized set can never fall back
    to zeros or to a missing tensor silently.
    """
    from .checkpoint_mapper import WeightDict

    projections, plain = read_nvfp4_checkpoint(path)
    weights = WeightDict()
    for name, value in plain.items():
        if name.endswith(".weight") and value.ndim == 2:
            value = np.ascontiguousarray(value.T)
        weights[name] = value
    for mod_key in (
        "double_stream_modulation_img",
        "double_stream_modulation_txt",
        "single_stream_modulation",
    ):
        weights[mod_key] = weights.pop(f"{mod_key}.linear.weight")
    for name, layer in projections.items():
        weights[f"{name}.nvfp4"] = layer
    return weights
