# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1:1 port of standard_checkpoint_mapper.cpp + tensor_math.cpp to Python.

Loads HF safetensors and maps keys to the flat weight dict expected by
standard_decoder_builder.py. All projections are transposed from HF
[out, in] layout to [in, out] for TRT matmul.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Register bfloat16 dtype with numpy (needed for safetensors without torch).
try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

from safetensors import safe_open


def _target_np_dtype(precision: str) -> np.dtype:
    """Map precision string to numpy dtype for weight storage."""
    if precision in ("fp16", "bf16"):
        return np.float16
    return np.float32


def _transpose_2d(arr: np.ndarray, name: str, precision: str = "fp32") -> np.ndarray:
    """Transpose [rows, cols] -> [cols, rows] in C-contiguous target dtype."""
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=_target_np_dtype(precision))


class WeightDict(dict):
    """A dict mapping logical weight names to flat float32 arrays.

    Keys follow the convention used by standard_decoder_builder.py:
      - embedding: [vocab, hidden]
      - layer.{i}.input_norm: [hidden]
      - layer.{i}.w_q: [hidden, attention_size]
      - layer.{i}.w_k: [hidden, kv_attention_size]
      - layer.{i}.w_v: [hidden, kv_attention_size]
      - layer.{i}.q_bias: [attention_size]       (optional)
      - layer.{i}.k_bias: [kv_attention_size]    (optional)
      - layer.{i}.v_bias: [kv_attention_size]    (optional)
      - layer.{i}.q_norm: [attention_size]        (optional)
      - layer.{i}.k_norm: [kv_attention_size]     (optional)
      - layer.{i}.w_o: [attention_size, hidden]
      - layer.{i}.post_attn_norm: [hidden]
      - layer.{i}.w_gate: [hidden, mlp_size]
      - layer.{i}.w_up: [hidden, mlp_size]
      - layer.{i}.w_down: [mlp_size, hidden]
      - final_norm: [hidden]
      - w_out: [hidden, vocab]
    """


# ---------------------------------------------------------------------------
# Safetensors I/O helpers
# ---------------------------------------------------------------------------


def _detect_framework() -> str:
    """Use 'torch' if available (handles BF16 natively), else 'numpy'."""
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        return "numpy"


class _TorchBinReader:
    """Adapter that wraps a pytorch .bin state dict with the safetensors reader
    interface (keys() / get_tensor())."""

    def __init__(self, path: Path):
        import torch

        self._state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self) -> list[str]:
        return list(self._state.keys())

    def get_tensor(self, name: str):
        return self._state[name]


class _ReaderCollection(list):
    """Reader list with a cached tensor-name -> reader lookup table."""

    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {}
            for reader in readers:
                for key in reader.keys():
                    tensor_map[key] = reader
        self.tensor_map = tensor_map


def _open_safetensors(model_dir: Path) -> list:
    """Open all safetensor shards (or pytorch .bin) in a model directory."""
    fw = _detect_framework()
    single = model_dir / "model.safetensors"
    if single.exists():
        return _ReaderCollection([safe_open(str(single), framework=fw)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        import json

        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework=fw) for shard in shard_files
        }
        tensor_map = {name: readers_by_file[shard] for name, shard in weight_map.items()}
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map=tensor_map,
        )

    # Diffusers format: diffusion_pytorch_model.safetensors
    diff_single = model_dir / "diffusion_pytorch_model.safetensors"
    if diff_single.exists():
        return _ReaderCollection([safe_open(str(diff_single), framework=fw)])

    diff_index = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if diff_index.exists():
        import json

        index = json.loads(diff_index.read_text())
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework=fw) for shard in shard_files
        }
        tensor_map = {name: readers_by_file[shard] for name, shard in weight_map.items()}
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map=tensor_map,
        )

    # Fallback: pytorch_model.bin (older HF models)
    bin_single = model_dir / "pytorch_model.bin"
    if bin_single.exists():
        return _ReaderCollection([_TorchBinReader(bin_single)])

    raise FileNotFoundError(
        f"No model.safetensors, index.json, or pytorch_model.bin in {model_dir}"
    )


def _has_tensor(readers: list, name: str) -> bool:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        return name in tensor_map
    for r in readers:
        if name in r.keys():
            return True
    return False


def _to_numpy_fp32(t) -> np.ndarray:
    """Convert a safetensors/torch tensor to numpy float32 with minimal copies."""
    if hasattr(t, "numpy"):
        dtype = getattr(t, "dtype", None)
        if str(dtype) == "torch.float32":
            return t.numpy()
        return t.float().numpy()

    dtype_str = str(t.dtype)
    if t.dtype == np.uint16 or dtype_str == "bfloat16":
        t = t.view(np.uint16).astype(np.uint32) << 16
        return t.view(np.float32)
    if dtype_str == "float16":
        return t.astype(np.float32)
    return np.asarray(t, dtype=np.float32)



# Qwen3.8 FP8 checkpoints store the large projections as float8_e4m3 together
# with a companion "<name>_scale_inv" tensor holding one bf16 scale per
# weight_block_size block (128x128 for Qwen3.8-27B-FP8). Converting the raw
# float8 values without applying those scales silently yields wrong weights, so
# the two are always resolved together.
_FP8_DTYPE_NAMES = frozenset({
    "torch.float8_e4m3fn", "float8_e4m3fn", "float8_e4m3",
    "torch.float8_e5m2", "float8_e5m2",
})
_SCALE_INV_SUFFIX = "_scale_inv"


def _is_fp8_tensor(t) -> bool:
    return str(getattr(t, "dtype", "")) in _FP8_DTYPE_NAMES


def _apply_block_scales(values: np.ndarray, scale_inv: np.ndarray) -> np.ndarray:
    """Scale a dequantized FP8 weight in place by its per-block scales.

    ``scale_inv`` carries one entry per block; block extents are derived from the
    two shapes so the loader does not need to read weight_block_size, and a
    trailing partial block is handled by clipping. The multiply is done block by
    block rather than by expanding the scales, because an expanded scale array
    would be as large as the weight itself.
    """
    if values.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            f"FP8 block dequantization expects 2-D weight and scales, got "
            f"{values.ndim}-D and {scale_inv.ndim}-D")
    rows, cols = values.shape
    s_rows, s_cols = scale_inv.shape
    block_r = -(-rows // s_rows)
    block_c = -(-cols // s_cols)
    for i in range(s_rows):
        r0, r1 = i * block_r, min((i + 1) * block_r, rows)
        if r0 >= r1:
            break
        for j in range(s_cols):
            c0, c1 = j * block_c, min((j + 1) * block_c, cols)
            if c0 >= c1:
                break
            values[r0:r1, c0:c1] *= scale_inv[i, j]
    return values


def _load_tensor(readers: list, name: str) -> np.ndarray:
    raw = _get_raw_tensor(readers, name)
    values = _to_numpy_fp32(raw)
    if not _is_fp8_tensor(raw):
        return values

    scale_name = name + _SCALE_INV_SUFFIX
    if not _has_tensor(readers, scale_name):
        # Refuse to return unscaled float8 values: they look like a plausible
        # weight tensor and would corrupt the engine silently.
        raise KeyError(
            f"FP8 tensor {name!r} has no companion {scale_name!r}; "
            "cannot dequantize")
    scale_inv = _to_numpy_fp32(_get_raw_tensor(readers, scale_name))
    return _apply_block_scales(values, scale_inv)


def _get_raw_tensor(readers: list, name: str):
    """Fetch a tensor from the shard collection without dtype conversion."""
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        reader = tensor_map.get(name)
        if reader is None:
            raise KeyError(f"Tensor not found: {name}")
        return reader.get_tensor(name)
    for r in readers:
        if name in r.keys():
            return r.get_tensor(name)
    raise KeyError(f"Tensor not found: {name}")
