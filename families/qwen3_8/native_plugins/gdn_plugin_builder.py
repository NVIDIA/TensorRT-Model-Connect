# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time helpers for the cuTile Gated DeltaNet decode plugin.

  ensure_plugin_loaded()      build libtrtmc_qwen3_8_native_plugin.so, dlopen it
  compile_cubins(...)         compile the four cuTile stage kernels for one shape/dtype
  add_gdn_decode_plugin()     add the decode plugin layer to a TensorRT network
  add_gdn_prefill_plugin()    add the static-shape prefill plugin layer to a TensorRT network

Decode (T = 1) and prefill (static T, with valid_len for a padded last chunk) are separate plugins.

Environment (all optional):
  TRTMC_GDN_CONV_BLOCK / TRTMC_GDN_BLOCK_V / TRTMC_GDN_STREAM_V   decode kernel tile sizes
  TRTMC_GDN_PREFILL_BLOCK_V                                       prefill delta-rule V tile
  TRTMC_GDN_PREFILL_INTER=auto|cuda|cutile                        prefill state-chain kernel: CUDA (NVRTC,
                                                                  validated on sm_120) or cuTile; auto picks CUDA where validated
"""

from __future__ import annotations

import ctypes
import importlib.util
import io
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_PLUGIN_PATH: Path | None = None
_PLUGIN_HANDLE = None
_CUBIN_CACHE: dict[tuple, dict[str, bytes]] = {}

PLUGIN_NAME = "Qwen38GdnDecodeTileGym"
_CONV_KERNEL = 4
_NORM_EPS = 1e-6
_STAGE_SYMBOLS = {"conv": "gdn_conv_silu", "gate": "gdn_gate", "gdn": "gdn_delta",
                  "norm": "gdn_norm"}


def _tensorrt_library() -> Path:
    env = os.environ.get("TRT_LIB_DIR")
    if env and (Path(env) / "libnvinfer.so.11").is_file():
        return Path(env) / "libnvinfer.so.11"
    spec = importlib.util.find_spec("tensorrt_libs")
    for location in (spec.submodule_search_locations if spec else None) or ():
        library = Path(location) / "libnvinfer.so.11"
        if library.is_file():
            return library
    raise RuntimeError("libnvinfer.so.11 not found in the active Python installation")


def _tensorrt_include_dir() -> Path:
    env = os.environ.get("TRT_ROOT")
    candidates = [Path(env) / "include"] if env else []
    candidates += [Path("/usr/include/x86_64-linux-gnu"), Path("/usr/include/aarch64-linux-gnu"),
                   Path("/usr/include"), Path("/opt/tensorrt/include")]
    for candidate in candidates:
        if (candidate / "NvInferRuntime.h").is_file():
            return candidate
    raise RuntimeError("NvInferRuntime.h not found; set TRT_ROOT to a TensorRT install")


def _cuda_include_dir() -> Path:
    candidates = [Path(os.environ["CUDA_HOME"]) / "include"] if os.environ.get("CUDA_HOME") else []
    candidates.append(Path("/usr/local/cuda/include"))
    spec = importlib.util.find_spec("nvidia.cu13")
    for location in (spec.submodule_search_locations if spec else None) or ():
        candidates.append(Path(location) / "include")
    for candidate in candidates:
        if (candidate / "cuda.h").is_file():
            return candidate
    raise RuntimeError("cuda.h not found; set CUDA_HOME to a CUDA toolkit")


def prefill_plugin_unavailable_reason() -> str | None:
    """None if the cuTile GDN prefill plugin can be built and loaded here, else why not."""
    import importlib.util

    try:
        if importlib.util.find_spec("cuda.tile") is None:
            return "the cuda-tile package is not installed"
    except ModuleNotFoundError:
        return "the cuda-tile package is not installed"
    try:
        ensure_plugin_loaded()
    except Exception as error:  # build tools, TensorRT or CUDA headers missing, ...
        first = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
        return f"the plugin library could not be built or loaded ({first})"
    return None


def ensure_plugin_loaded(*, verbose: bool = False) -> Path:
    """Build the plugin library once per process and load it for graph construction."""
    global _PLUGIN_PATH, _PLUGIN_HANDLE
    if _PLUGIN_PATH is not None:
        return _PLUGIN_PATH
    source_dir = Path(__file__).parent
    build_dir = Path(tempfile.mkdtemp(prefix="qwen3-8-gdn-plugin-"))
    configure = [
        "cmake", "-S", str(source_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
        f"-DQWEN3_8_TRT_INCLUDE_DIR={_tensorrt_include_dir()}",
        f"-DQWEN3_8_TRT_LIBRARY={_tensorrt_library()}",
        f"-DQWEN3_8_CUDA_INCLUDE_DIR={_cuda_include_dir()}",
    ]
    build = ["cmake", "--build", str(build_dir), "-j4"]
    kwargs = {} if verbose else {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                                 "text": True}
    try:
        subprocess.run(configure, check=True, **kwargs)
        subprocess.run(build, check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Qwen3.8 native plugin build failed\n{error.stdout or ''}") from error
    path = (build_dir / "libtrtmc_qwen3_8_native_plugin.so").resolve()
    if not path.is_file():
        raise RuntimeError(f"Qwen3.8 native plugin build did not produce {path}")
    _PLUGIN_HANDLE = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _PLUGIN_PATH = path
    return path


def _array_constraint(ct, constraint_cls, dtype, shape, strides=None):
    """Array argument with constant shape and strides and 16-byte alignment assumptions.

    A `None` in `shape` is a runtime extent (the strides that depend on it become runtime too).
    The alignment and divisibility hints let the tile compiler use vectorized accesses, as
    `ct.launch` does for real tensors. TensorRT aligns tensor and workspace buffers to 256
    bytes, and every sliced view used here starts at a multiple of 16 bytes.
    """
    if strides is None:  # row-major contiguous
        rev, step = [], 1
        for n in reversed(shape):
            rev.append(step)
            step = None if (step is None or n is None) else step * n
        strides = tuple(reversed(rev))
    elem_bytes = 2 if dtype in (ct.bfloat16, ct.float16) else 4
    vec = 16 // elem_bytes
    return constraint_cls(
        dtype, len(shape), index_dtype=ct.int32, stride_constant=tuple(strides),
        shape_constant=tuple(shape), alias_groups=[], stride_lower_bound_incl=0,
        may_alias_internally=False, base_addr_divisible_by=16,
        stride_divisible_by=tuple(1 if st is None else math.gcd(int(st), vec) for st in strides))


def block_sizes() -> tuple[int, int, int]:
    """(conv block, delta-rule V block, V stream tile); overridable for tuning."""
    return (int(os.environ.get("TRTMC_GDN_CONV_BLOCK", "256")),
            int(os.environ.get("TRTMC_GDN_BLOCK_V", "32")),
            int(os.environ.get("TRTMC_GDN_STREAM_V", "16")))


def prefill_block_v() -> int:
    """V tile of the prefill inter-chunk kernel (64 was fastest on RTX 5090)."""
    return int(os.environ.get("TRTMC_GDN_PREFILL_BLOCK_V", "64"))


def prefill_chunk() -> int:
    """Tokens per chunk of the chunked delta rule (32 was fastest on RTX 5090)."""
    return int(os.environ.get("TRTMC_GDN_PREFILL_CHUNK", "32"))


def compile_cubins(*, num_heads: int, num_kv_heads: int, head_dim: int, bf16: bool,
                   gpu_code: str, hidden_size: int = 0) -> dict[str, bytes]:
    """Compile the four stage kernels (cutile/gdn_tilegym.py) for fixed shapes.

    A nonzero `hidden_size` selects the gate kernel that also computes the a and b projections.
    """
    conv_block, block_v, stream_v = block_sizes()
    key = (num_heads, num_kv_heads, head_dim, bf16, gpu_code, conv_block, block_v, stream_v,
           hidden_size)
    if key in _CUBIN_CACHE:
        return _CUBIN_CACHE[key]

    import cuda.tile as ct
    from cuda.tile.compilation import (ArrayConstraint, CallingConvention, KernelSignature,
                                       export_kernel)

    from ..cutile import gdn_tilegym as K

    H, D, NK = num_heads, head_dim, num_kv_heads
    DI = H * D
    CD = 2 * NK * D + DI
    act = ct.bfloat16 if bf16 else ct.float16
    f32 = ct.float32

    def arr(dtype, shape):
        return _array_constraint(ct, ArrayConstraint, dtype, shape)

    def pow2(n):
        return 1 << (n - 1).bit_length()

    stages = {
        "conv": (K.conv_silu_kernel, [
            arr(act, (CD,)), arr(act, (CD, _CONV_KERNEL)), arr(f32, (CD, _CONV_KERNEL)),
            arr(f32, (CD,)), arr(act, (CD,)), arr(act, (CD, _CONV_KERNEL)), conv_block]),
        "gate": (K.gate_kernel, [
            arr(act, (H,)), arr(act, (H,)), arr(f32, (H,)), arr(f32, (H,)), arr(act, (H,)),
            arr(f32, (H,)), pow2(H)]) if not hidden_size else (K.ab_gate_kernel, [
            arr(act, (hidden_size,)), arr(f32, (2 * H, hidden_size)), arr(f32, (H,)),
            arr(f32, (H,)), arr(act, (H,)), arr(f32, (H,)), H, hidden_size,
            min(1024, pow2(hidden_size))]),
        "gdn": (K.gdn_kernel, [
            arr(act, (1, 1, NK, D)), arr(act, (1, 1, NK, D)), arr(act, (1, 1, H, D)),
            arr(f32, (1, 1, H)), arr(act, (1, 1, H)), arr(act, (1, 1, H, D)),
            arr(f32, (1, H, D, D)), arr(f32, (1, H, D, D)), float(D) ** -0.5, pow2(D),
            block_v, stream_v]),
        "norm": (K.norm_gated_kernel, [
            arr(act, (H, D)), arr(act, (H, D)), arr(f32, (H, D)), arr(act, (H, D)),
            _NORM_EPS, D, pow2(D)]),
    }
    cubins = {}
    for name, (kernel, parameters) in stages.items():
        signature = KernelSignature(parameters, CallingConvention.cutile_python_v2(),
                                    symbol=_STAGE_SYMBOLS[name])
        buffer = io.BytesIO()
        export_kernel(kernel, [signature], buffer, gpu_code=gpu_code, output_format="cubin")
        cubins[name] = buffer.getvalue()
    _CUBIN_CACHE[key] = cubins
    return cubins


def _block_threads(cubin: bytes) -> int:
    """Block size the tile compiler chose for the kernel in `cubin`.

    The cubin stores it as a one-byte attribute (0x4c) counted in groups of 128 threads; a
    kernel without the attribute uses one group. Launching with any other block size fails.
    """
    import struct

    shoff, = struct.unpack_from("<Q", cubin, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", cubin, 0x3A)
    sections = [struct.unpack_from("<IIQQQQ", cubin, shoff + i * shentsize) for i in range(shnum)]
    str_off = sections[shstrndx][4]
    for name_off, _, _, _, off, size in sections:
        start = str_off + name_off
        if not cubin[start:cubin.index(b"\0", start)].startswith(b".nv.info."):
            continue
        pos = off
        while pos < off + size:
            fmt, attr = cubin[pos], cubin[pos + 1]
            pos += 2
            if fmt == 4:  # EIFMT_SVAL: 2-byte length, then the payload
                length, = struct.unpack_from("<H", cubin, pos)
                pos += 2 + length
            elif fmt == 3:  # EIFMT_HVAL: 2-byte value
                pos += 2
            elif fmt == 2:  # EIFMT_BVAL: 1-byte value
                if attr == 0x4c:
                    return 128 * cubin[pos]
                pos += 1
            elif fmt != 1:  # EIFMT_NVAL has no payload; anything else is unknown
                break
    return 128


def _gpu_code() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def add_gdn_decode_plugin(network, trt, *, inputs, num_heads: int, num_kv_heads: int,
                          head_dim: int, bf16: bool, hidden_size: int = 0):
    """Add the Gated DeltaNet decode plugin layer (steps 3-10 of the GDN block).

    `inputs` must be the 11 tensors listed in gdn_tilegym_plugin.cpp. A nonzero `hidden_size`
    makes the plugin compute the a and b projections too: inputs 2 and 3 are then the
    normalized hidden row [1, hidden_size] and the fp32 weight [2 * num_heads, hidden_size]
    (a rows, then b rows).
    Returns the layer; outputs are (gated out, present_conv, present_ssm).
    """
    ensure_plugin_loaded()
    conv_dim = 2 * num_kv_heads * head_dim + num_heads * head_dim
    cubins = compile_cubins(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                            bf16=bf16, gpu_code=_gpu_code(), hidden_size=hidden_size)
    creator = trt.get_plugin_registry().get_creator(PLUGIN_NAME, "1", "")
    if creator is None:
        raise RuntimeError(f"TensorRT plugin creator {PLUGIN_NAME} is not registered")
    conv_block, block_v, _ = block_sizes()

    def int_field(name, value):
        return trt.PluginField(name, np.array([value], dtype=np.int32), trt.PluginFieldType.INT32)

    fields = [int_field("num_heads", num_heads), int_field("num_kv_heads", num_kv_heads),
              int_field("head_dim", head_dim), int_field("conv_dim", conv_dim),
              int_field("act_bytes", 2), int_field("conv_block", conv_block),
              int_field("block_v", block_v), int_field("fuse_ab", int(hidden_size > 0)),
              int_field("hidden_size", hidden_size)]
    fields.append(trt.PluginField(
        "threads", np.array([_block_threads(blob) for blob in cubins.values()], dtype=np.int32),
        trt.PluginFieldType.INT32))
    for name, blob in cubins.items():
        fields.append(trt.PluginField(f"cubin_{name}", np.frombuffer(blob, dtype=np.int8),
                                      trt.PluginFieldType.INT8))
    plugin = creator.create_plugin(PLUGIN_NAME, trt.PluginFieldCollection(fields),
                                   trt.TensorRTPhase.BUILD)
    if plugin is None:
        raise RuntimeError(f"TensorRT failed to create plugin {PLUGIN_NAME}")
    return network.add_plugin_v3(list(inputs), [], plugin)


PREFILL_PLUGIN_NAME = "Qwen38GdnPrefillTileGym"
_PREFILL_SYMBOLS = {"conv": "gdnp_conv", "conv_state": "gdnp_conv_state", "gate": "gdnp_gate",
                    "intra": "gdnp_intra", "inter": "gdnp_inter", "norm": "gdnp_norm"}
_PREFILL_CUBIN_CACHE: dict[tuple, "_Cubins"] = {}
_CUDA_INTER_ARCHS = {"sm_120"}  # GPUs the CUDA state-chain kernel has been validated on
# Dynamic shared memory of gdn_chain_inter.cu: two stages of
# KQ (64 x 136 fp16) + K_w^T (128 x 40 fp16) + QK_mask (32 x 40 fp16) + V_corr (32 x 68 fp32) + g (32 fp32).
_CHAIN_INTER_SMEM = 2 * (64 * 136 * 2 + 128 * 40 * 2 + 32 * 40 * 2 + 32 * 68 * 4 + 32 * 4)


class _Cubins(dict):
    """Stage name -> cubin. `inter_cuda` tells that the inter stage is the CUDA kernel."""

    inter_cuda = False


def _inter_cuda_mode() -> str:
    mode = os.environ.get("TRTMC_GDN_PREFILL_INTER", "auto")
    if mode not in ("auto", "cuda", "cutile"):
        raise ValueError("TRTMC_GDN_PREFILL_INTER must be auto, cuda or cutile")
    return mode


def _inter_cuda_unsupported(K, gpu_code: str, chunk: int, head_dim: int) -> str | None:
    if gpu_code not in _CUDA_INTER_ARCHS:
        return f"the CUDA chain kernel is not validated on {gpu_code}"
    if chunk != 32 or head_dim != 128:
        return "the CUDA chain kernel needs chunk 32 and head_dim 128"
    if K._MM_DTYPE != K.ct.float16:
        return "the CUDA chain kernel needs fp16 matmul operands"
    return None


def _compile_chain_inter(gpu_code: str, bf16: bool) -> bytes:
    """gdn_chain_inter.cu -> cubin for `gpu_code` with NVRTC."""
    from cuda.bindings import nvrtc

    source = (Path(__file__).parent / "gdn_chain_inter.cu").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(source.encode(), b"gdn_chain_inter.cu", 0, [], [])
    options = [f"--gpu-architecture={gpu_code}".encode(), b"-std=c++17",
               f"-DOUT_BF16={int(bf16)}".encode()]
    result, = nvrtc.nvrtcCompileProgram(prog, len(options), options)
    if int(result) != 0:
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC failed to compile gdn_chain_inter.cu: {log.decode(errors='replace')}")
    _, size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    nvrtc.nvrtcDestroyProgram(prog)
    return cubin


def compile_prefill_cubins(*, num_heads: int, num_kv_heads: int, head_dim: int, seq_len: int,
                           bf16: bool, gpu_code: str, dynamic: bool = False) -> _Cubins:
    """Compile the six prefill stage kernels (cutile/gdn_prefill.py). The inter stage is the CUDA
    kernel gdn_chain_inter.cu (NVRTC) when TRTMC_GDN_PREFILL_INTER allows it, else a cuTile kernel.

    Static: shapes are constants for `seq_len` tokens. Dynamic: the token count is a runtime
    extent (any 1 <= T <= seq_len; the kernels read it from the array shapes)."""
    from ..cutile import gdn_prefill as K

    conv_block, conv_rows = K.CONV_BLOCK, K.CONV_ROWS
    block_v = prefill_block_v()
    chunk = prefill_chunk()
    key = (num_heads, num_kv_heads, head_dim, seq_len, bf16, gpu_code, conv_block, conv_rows,
           block_v, chunk, dynamic, _inter_cuda_mode())
    if key in _PREFILL_CUBIN_CACHE:
        return _PREFILL_CUBIN_CACHE[key]

    import cuda.tile as ct
    from cuda.tile.compilation import (ArrayConstraint, CallingConvention, KernelSignature,
                                       export_kernel)

    H, D, NK, T = num_heads, head_dim, num_kv_heads, seq_len
    CD = 2 * NK * D + H * D
    act, f32, i32 = (ct.bfloat16 if bf16 else ct.float16), ct.float32, ct.int32
    mode = _inter_cuda_mode()
    use_cuda = False
    if mode != "cutile":
        reason = _inter_cuda_unsupported(K, gpu_code, chunk, head_dim)
        chain_cubin = None
        if reason is None:
            try:
                chain_cubin = _compile_chain_inter(gpu_code, bf16)
            except Exception as error:  # NVRTC missing or failing
                reason = str(error).splitlines()[0] if str(error) else type(error).__name__
        if reason is not None and mode == "cuda":
            raise RuntimeError(f"TRTMC_GDN_PREFILL_INTER=cuda is not available: {reason}")
        use_cuda = reason is None
    if T % chunk and not dynamic:
        raise ValueError(f"seq_len {T} must be a multiple of the prefill chunk {chunk}")
    NC = None if dynamic else T // chunk
    if dynamic:
        T = None  # runtime extent

    def arr(dtype, shape, strides=None):
        return _array_constraint(ct, ArrayConstraint, dtype, shape, strides)

    def pow2(n):
        return 1 << (n - 1).bit_length()

    hoist = ct.float16 if use_cuda else f32  # intra -> inter operands that only feed matmuls
    TH = None if T is None else T * H
    qkv_rows = (None if T is None else T * CD, CD, D, 1)  # q, k, v: strided views of the [T, CD] conv output
    stages = {
        "conv": (K.conv_kernel, [
            arr(act, (T, CD)), arr(act, (CD, _CONV_KERNEL)), arr(f32, (CD, _CONV_KERNEL)),
            arr(f32, (CD,)), arr(act, (T, CD)), conv_block, conv_rows]),
        "conv_state": (K.conv_state_kernel, [
            arr(act, (T, CD)), arr(act, (CD, _CONV_KERNEL)), arr(i32, (1,)),
            arr(act, (CD, _CONV_KERNEL)), conv_block]),
        "gate": (K.gate_kernel, [
            arr(act, (T, H)), arr(act, (T, H)), arr(f32, (H,)), arr(f32, (H,)), arr(i32, (1,)),
            arr(act, (T, H)), arr(f32, (T, H)), pow2(H)]),
        "intra": (K.intra_cuda_kernel if use_cuda else K.intra_kernel, [
            arr(act, (1, T, NK, D), qkv_rows), arr(act, (1, T, NK, D), qkv_rows),
            arr(act, (1, T, H, D), qkv_rows), arr(f32, (1, T, H)), arr(act, (1, T, H)),
            arr(hoist, (H, NC, 2 * chunk, D)), arr(hoist, (H, NC, D, chunk) if use_cuda else (H, NC, chunk, D)),
            arr(hoist, (H, NC, chunk, chunk)), arr(f32, (H, NC, chunk, D)), arr(f32, (H, NC, chunk)),
            float(D) ** -0.5, chunk, pow2(D)]),
        "inter": (None if use_cuda else K.inter_kernel, [
            arr(f32, (H, NC, 2 * chunk, D)), arr(f32, (H, NC, chunk, D)),
            arr(f32, (H, NC, chunk, chunk)), arr(f32, (H, NC, chunk, D)), arr(f32, (H, NC, chunk)),
            arr(act, (1, T, H, D)), arr(f32, (1, H, D, D)), arr(f32, (1, H, D, D)),
            chunk, pow2(D), block_v]),
        "norm": (K.norm_kernel, [
            arr(act, (TH, D)), arr(act, (TH, D)), arr(f32, (H, D)), arr(i32, (1,)),
            arr(act, (TH, D)), _NORM_EPS, D, pow2(D), H]),
    }
    cubins = _Cubins()
    cubins.inter_cuda = use_cuda
    for name, (kernel, parameters) in stages.items():
        if name == "inter" and use_cuda:
            cubins[name] = chain_cubin
            continue
        signature = KernelSignature(parameters, CallingConvention.cutile_python_v2(),
                                    symbol=_PREFILL_SYMBOLS[name])
        buffer = io.BytesIO()
        export_kernel(kernel, [signature], buffer, gpu_code=gpu_code, output_format="cubin")
        cubins[name] = buffer.getvalue()
    _PREFILL_CUBIN_CACHE[key] = cubins
    return cubins


def add_gdn_prefill_plugin(network, trt, *, inputs, num_heads: int, num_kv_heads: int,
                           head_dim: int, seq_len: int, bf16: bool, dynamic: bool = False):
    """Add the prefill plugin layer (steps 3-10).

    Static (default): exactly `seq_len` rows per call. `dynamic=True`: the row count T of the
    inputs is a runtime extent, 1 <= T <= `seq_len` (the TensorRT optimization profile maximum).

    `inputs` must be the 12 tensors listed in gdn_prefill_plugin.cpp (the last one is the int32
    `valid_len`, 1 <= valid_len <= seq_len). Returns the layer; outputs are
    (gated out [seq_len, d_inner], present_conv, present_ssm). Rows of `out` at or after
    `valid_len` are zero.
    """
    ensure_plugin_loaded()
    conv_dim = 2 * num_kv_heads * head_dim + num_heads * head_dim
    cubins = compile_prefill_cubins(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                    head_dim=head_dim, seq_len=seq_len, bf16=bf16,
                                    gpu_code=_gpu_code(), dynamic=dynamic)
    creator = trt.get_plugin_registry().get_creator(PREFILL_PLUGIN_NAME, "1", "")
    if creator is None:
        raise RuntimeError(f"TensorRT plugin creator {PREFILL_PLUGIN_NAME} is not registered")
    from ..cutile import gdn_prefill as K

    conv_block, block_v, chunk = K.CONV_BLOCK, prefill_block_v(), prefill_chunk()

    def int_field(name, value):
        return trt.PluginField(name, np.array([value], dtype=np.int32), trt.PluginFieldType.INT32)

    fields = [int_field("num_heads", num_heads), int_field("num_kv_heads", num_kv_heads),
              int_field("head_dim", head_dim), int_field("conv_dim", conv_dim),
              int_field("seq_len", seq_len), int_field("act_bytes", 2),
              int_field("conv_block", conv_block), int_field("block_v", block_v),
              int_field("chunk", chunk), int_field("dynamic", int(dynamic)),
              int_field("conv_rows", K.CONV_ROWS),
              int_field("inter_cuda", int(cubins.inter_cuda)),
              int_field("inter_smem", _CHAIN_INTER_SMEM if cubins.inter_cuda else 0)]
    fields.append(trt.PluginField(
        "threads", np.array([_block_threads(blob) for blob in cubins.values()], dtype=np.int32),
        trt.PluginFieldType.INT32))
    for name, blob in cubins.items():
        fields.append(trt.PluginField(f"cubin_{name}", np.frombuffer(blob, dtype=np.int8),
                                      trt.PluginFieldType.INT8))
    plugin = creator.create_plugin(PREFILL_PLUGIN_NAME, trt.PluginFieldCollection(fields),
                                   trt.TensorRTPhase.BUILD)
    if plugin is None:
        raise RuntimeError(f"TensorRT failed to create plugin {PREFILL_PLUGIN_NAME}")
    return network.add_plugin_v3(list(inputs), [], plugin)
