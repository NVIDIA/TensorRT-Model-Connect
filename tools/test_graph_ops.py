#!/usr/bin/env python3
"""Unit tests for every graph_ops function against HuggingFace references.

Tests every parameter combination of:
  - compute_alibi_slopes: power-of-2 (8,16) and non-power-of-2 (6,12)
  - make_rope_table: interleaved × partial_rotary_factor × cosine
  - make_rotate_half_matrix: interleaved × partial_rotary_factor
  - add_rms_norm: vs torch manual RMSNorm
  - add_rms_norm_per_head: vs torch per-head RMSNorm
  - add_layer_norm: vs torch.nn.LayerNorm
  - add_gelu_new: vs HF NewGELUActivation (tanh approx)
  - add_activation(silu): vs torch.nn.SiLU
  - add_activation(relu): vs torch.nn.ReLU
  - add_apply_rope (rotated-half): vs HF LLaMA rotate_half
  - add_apply_rope (interleaved): vs CodeGen rotate_every_two

Run inside the container:
    python3 tools/test_graph_ops.py
"""

from __future__ import annotations

import math
import sys

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn

# cuda-python bindings
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart  # type: ignore[no-redef]

sys.path.insert(0, "python")
from tensorrt_model_connect import graph_ops


# ---------------------------------------------------------------
# Helpers: build a tiny TRT engine, run it, return numpy output
# ---------------------------------------------------------------

def _check(status):
    if hasattr(cudart, "cudaError_t"):
        ok = cudart.cudaError_t.cudaSuccess
    else:
        ok = 0
    if status != ok:
        raise RuntimeError(f"CUDA error: {status}")


def _run_trt_graph(build_fn, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build a TRT engine from build_fn, feed inputs, return outputs."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    config.clear_flag(trt.BuilderFlag.TF32)

    # Create input tensors
    trt_inputs = {}
    for name, arr in inputs.items():
        dt = trt.float32 if arr.dtype == np.float32 else trt.int32
        t = network.add_input(name, dt, tuple(arr.shape))
        trt_inputs[name] = t

    # Let build_fn add ops and return output dict {name: ITensor}
    trt_outputs = build_fn(network, trt_inputs)

    for name, tensor in trt_outputs.items():
        tensor.name = name
        network.mark_output(tensor)
        tensor.dtype = trt.float32

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT build failed")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()

    err, stream = cudart.cudaStreamCreate()
    _check(err)

    device_bufs = {}
    host_out = {}
    for i in range(engine.num_io_tensors):
        tname = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(tname))
        nbytes = int(np.prod(shape)) * 4
        err, ptr = cudart.cudaMallocAsync(nbytes, stream)
        _check(err)
        device_bufs[tname] = ptr
        mode = engine.get_tensor_mode(tname)
        if mode == trt.TensorIOMode.INPUT:
            arr = inputs[tname]
            cudart.cudaMemcpyAsync(ptr, arr.ctypes.data, nbytes,
                                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        else:
            host_out[tname] = np.zeros(shape, dtype=np.float32)
        ctx.set_tensor_address(tname, ptr)

    ctx.execute_async_v3(stream)

    for name, arr in host_out.items():
        cudart.cudaMemcpyAsync(arr.ctypes.data, device_bufs[name], arr.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaStreamSynchronize(stream)
    for ptr in device_bufs.values():
        cudart.cudaFreeAsync(ptr, stream)
    cudart.cudaStreamDestroy(stream)

    return host_out


# ---------------------------------------------------------------
# 1. compute_alibi_slopes — vs HF build_alibi_tensor slopes
# ---------------------------------------------------------------

def _hf_alibi_slopes(num_heads: int) -> np.ndarray:
    """Reference: HF BloomModel.build_alibi_tensor slope computation."""
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        extra_powers = torch.arange(1, 1 + 2 * (num_heads - closest_power_of_2), 2,
                                    dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)])
    return slopes.numpy()


def test_alibi_slopes():
    for n in [1, 2, 4, 8, 16, 32, 6, 12, 3, 5, 7]:
        ours = graph_ops.compute_alibi_slopes(n)
        ref = _hf_alibi_slopes(n)
        assert ours.shape == ref.shape == (n,), f"n={n}: shape {ours.shape} vs {ref.shape}"
        assert np.allclose(ours, ref, atol=1e-7), \
            f"n={n}: max diff {np.abs(ours - ref).max():.2e}"
    print("  PASS  compute_alibi_slopes (11 head counts)")


# ---------------------------------------------------------------
# 2. make_rope_table — vs HF reference implementations
# ---------------------------------------------------------------

def _hf_rope_table_llama(max_len, hidden, num_heads, theta, cosine):
    """Standard LLaMA-style RoPE: pairs (d, d+half_rotary)."""
    head_dim = hidden // num_heads
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    table = np.full((max_len, hidden), 1.0 if cosine else 0.0, dtype=np.float32)
    for pos in range(max_len):
        for h in range(num_heads):
            for i in range(half):
                angle = pos * inv_freq[i]
                val = float(np.cos(angle) if cosine else np.sin(angle))
                table[pos, h * head_dim + i] = val
                table[pos, h * head_dim + half + i] = val
    return table


def _hf_rope_table_interleaved(max_len, hidden, num_heads, theta, cosine,
                                partial_factor=1.0):
    """CodeGen/GPT-J style: repeat_interleave, pairs (d, d+1)."""
    head_dim = hidden // num_heads
    rotary_ndims = int(head_dim * partial_factor)
    half = rotary_ndims // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, rotary_ndims, 2, dtype=np.float64) / rotary_ndims))
    table = np.full((max_len, hidden), 1.0 if cosine else 0.0, dtype=np.float32)
    for pos in range(max_len):
        for h in range(num_heads):
            for i in range(half):
                angle = pos * inv_freq[i]
                val = float(np.cos(angle) if cosine else np.sin(angle))
                # repeat_interleave: dim 2*i and 2*i+1 share freq i
                table[pos, h * head_dim + 2 * i] = val
                table[pos, h * head_dim + 2 * i + 1] = val
    return table


def test_rope_table():
    params = dict(max_cache_length=16, hidden_size=64, num_attention_heads=4,
                  rope_theta=10000.0)
    # Standard rotated-half, full RoPE
    for cosine in [True, False]:
        ours = graph_ops.make_rope_table(**params, cosine=cosine)
        ref = _hf_rope_table_llama(16, 64, 4, 10000.0, cosine)
        assert np.allclose(ours, ref, atol=1e-6), \
            f"rotated-half cos={cosine}: max diff {np.abs(ours - ref).max():.2e}"

    # Interleaved, full RoPE
    for cosine in [True, False]:
        ours = graph_ops.make_rope_table(**params, cosine=cosine, interleaved=True)
        ref = _hf_rope_table_interleaved(16, 64, 4, 10000.0, cosine)
        assert np.allclose(ours, ref, atol=1e-6), \
            f"interleaved cos={cosine}: max diff {np.abs(ours - ref).max():.2e}"

    # Partial rotary (factor=0.5), standard
    ours = graph_ops.make_rope_table(
        16, 64, 4, 10000.0, True, partial_rotary_factor=0.5)
    head_dim = 16
    # Non-rotary dims (last 8 per head) should be 1.0 (cosine default)
    for h in range(4):
        assert np.all(ours[:, h * head_dim + 8 : h * head_dim + 16] == 1.0), \
            "Partial rotary: non-rotary dims not 1.0"

    # Partial rotary (factor=0.5), interleaved
    ours_i = graph_ops.make_rope_table(
        16, 64, 4, 10000.0, True, partial_rotary_factor=0.5, interleaved=True)
    ref_i = _hf_rope_table_interleaved(16, 64, 4, 10000.0, True, 0.5)
    assert np.allclose(ours_i, ref_i, atol=1e-6), \
        f"interleaved partial: max diff {np.abs(ours_i - ref_i).max():.2e}"

    # Partial rotary (factor=0.25 — StableLM style)
    ours_025 = graph_ops.make_rope_table(
        8, 128, 2, 10000.0, True, partial_rotary_factor=0.25)
    hd = 64
    rotary_nd = 16
    for h in range(2):
        # Non-rotary dims should be 1.0
        assert np.all(ours_025[:, h * hd + rotary_nd : h * hd + hd] == 1.0)

    print("  PASS  make_rope_table (6 combinations: rotated/interleaved × cos/sin × partial)")


# ---------------------------------------------------------------
# 3. make_rotate_half_matrix — vs HF rotate_half / rotate_every_two
# ---------------------------------------------------------------

def _hf_rotate_half(x: np.ndarray) -> np.ndarray:
    """HF LLaMA rotate_half: swap first/second halves with sign flip."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def _hf_rotate_every_two(x: np.ndarray) -> np.ndarray:
    """HF CodeGen rotate_every_two: pair adjacent dims."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    stacked = np.stack([-x2, x1], axis=-1)
    return stacked.reshape(x.shape)


def test_rotate_half_matrix():
    rng = np.random.RandomState(42)

    # Standard rotated-half, full RoPE
    for hidden, heads in [(64, 4), (128, 8), (768, 12)]:
        mat = graph_ops.make_rotate_half_matrix(hidden, heads)
        x = rng.randn(1, hidden).astype(np.float32)
        ours = x @ mat
        # Reference: per-head rotate_half
        hd = hidden // heads
        ref_parts = []
        for h in range(heads):
            ref_parts.append(_hf_rotate_half(x[:, h*hd:(h+1)*hd]))
        ref = np.concatenate(ref_parts, axis=-1)
        assert np.allclose(ours, ref, atol=1e-6), \
            f"rotated-half h={hidden} n={heads}: max diff {np.abs(ours - ref).max():.2e}"

    # Interleaved (CodeGen style), full RoPE
    for hidden, heads in [(64, 4), (128, 8)]:
        mat = graph_ops.make_rotate_half_matrix(hidden, heads, interleaved=True)
        x = rng.randn(1, hidden).astype(np.float32)
        ours = x @ mat
        hd = hidden // heads
        ref_parts = []
        for h in range(heads):
            ref_parts.append(_hf_rotate_every_two(x[:, h*hd:(h+1)*hd]))
        ref = np.concatenate(ref_parts, axis=-1)
        assert np.allclose(ours, ref, atol=1e-6), \
            f"interleaved h={hidden} n={heads}: max diff {np.abs(ours - ref).max():.2e}"

    # Partial rotary (factor=0.5), standard
    mat_p = graph_ops.make_rotate_half_matrix(64, 4, partial_rotary_factor=0.5)
    x = rng.randn(1, 64).astype(np.float32)
    ours_p = x @ mat_p
    hd = 16
    rotary = 8
    for h in range(4):
        base = h * hd
        # Rotary dims: should match rotate_half on first 8 dims
        ref_rot = _hf_rotate_half(x[:, base:base+rotary])
        assert np.allclose(ours_p[:, base:base+rotary], ref_rot, atol=1e-6)
        # Non-rotary dims: should be identity (pass-through)
        assert np.allclose(ours_p[:, base+rotary:base+hd], x[:, base+rotary:base+hd], atol=1e-6)

    # Partial rotary (factor=0.5), interleaved
    mat_pi = graph_ops.make_rotate_half_matrix(64, 4, partial_rotary_factor=0.5,
                                                interleaved=True)
    ours_pi = x @ mat_pi
    for h in range(4):
        base = h * hd
        ref_rot_i = _hf_rotate_every_two(x[:, base:base+rotary])
        assert np.allclose(ours_pi[:, base:base+rotary], ref_rot_i, atol=1e-6)
        assert np.allclose(ours_pi[:, base+rotary:base+hd], x[:, base+rotary:base+hd], atol=1e-6)

    print("  PASS  make_rotate_half_matrix (7 combinations: rotated/interleaved × sizes × partial)")


# ---------------------------------------------------------------
# 4. add_rms_norm — vs torch reference
# ---------------------------------------------------------------

def test_rms_norm():
    rng = np.random.RandomState(42)
    for hidden in [64, 768]:
        x_np = rng.randn(1, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp):
            eps_t = graph_ops.add_constant(net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_ops.add_rms_norm(net, inp["x"], hidden, gamma_np, eps_t)
            return {"out": out}

        result = _run_trt_graph(build, {"x": x_np})
        trt_out = result["out"]

        # Reference: manual RMSNorm
        x_t = torch.tensor(x_np)
        rms = torch.sqrt((x_t ** 2).mean(dim=-1, keepdim=True) + eps)
        ref = (x_t / rms * torch.tensor(gamma_np)).numpy()

        assert np.allclose(trt_out, ref, atol=1e-5), \
            f"rms_norm h={hidden}: max diff {np.abs(trt_out - ref).max():.2e}"

    print("  PASS  add_rms_norm (2 hidden sizes)")


# ---------------------------------------------------------------
# 5. add_rms_norm_per_head — vs torch per-head reference
# ---------------------------------------------------------------

def test_rms_norm_per_head():
    rng = np.random.RandomState(42)
    for num_heads, head_dim in [(4, 16), (12, 64)]:
        hidden = num_heads * head_dim
        x_np = rng.randn(1, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp, nh=num_heads, hd=head_dim):
            eps_t = graph_ops.add_constant(net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_ops.add_rms_norm_per_head(net, inp["x"], nh, hd, gamma_np, eps_t)
            return {"out": out}

        result = _run_trt_graph(build, {"x": x_np})
        trt_out = result["out"].flatten()

        # Reference: per-head RMSNorm
        x_t = torch.tensor(x_np).reshape(num_heads, head_dim)
        rms = torch.sqrt((x_t ** 2).mean(dim=-1, keepdim=True) + eps)
        g = torch.tensor(gamma_np).reshape(num_heads, head_dim)
        ref = (x_t / rms * g).reshape(1, hidden).numpy().flatten()

        assert np.allclose(trt_out, ref, atol=1e-5), \
            f"rms_norm_per_head nh={num_heads}: max diff {np.abs(trt_out - ref).max():.2e}"

    print("  PASS  add_rms_norm_per_head (2 head configs)")


# ---------------------------------------------------------------
# 6. add_layer_norm — vs torch.nn.LayerNorm
# ---------------------------------------------------------------

def test_layer_norm():
    rng = np.random.RandomState(42)
    for hidden in [64, 768]:
        x_np = rng.randn(1, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        beta_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp, h=hidden):
            eps_t = graph_ops.add_constant(net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_ops.add_layer_norm(net, inp["x"], h, gamma_np, beta_np, eps_t)
            return {"out": out}

        result = _run_trt_graph(build, {"x": x_np})
        trt_out = result["out"]

        # Reference: torch.nn.LayerNorm
        ln = nn.LayerNorm(hidden, eps=eps)
        with torch.no_grad():
            ln.weight.copy_(torch.tensor(gamma_np))
            ln.bias.copy_(torch.tensor(beta_np))
            ref = ln(torch.tensor(x_np)).numpy()

        assert np.allclose(trt_out, ref, atol=1e-5), \
            f"layer_norm h={hidden}: max diff {np.abs(trt_out - ref).max():.2e}"

    print("  PASS  add_layer_norm (2 hidden sizes)")


# ---------------------------------------------------------------
# 7. add_gelu_new — vs HF NewGELUActivation
# ---------------------------------------------------------------

def test_gelu_new():
    rng = np.random.RandomState(42)
    x_np = rng.randn(1, 128).astype(np.float32)

    def build(net, inp):
        out = graph_ops.add_gelu_new(net, inp["x"])
        return {"out": out}

    result = _run_trt_graph(build, {"x": x_np})
    trt_out = result["out"]

    # Reference: HF NewGELUActivation (tanh approximation)
    x_t = torch.tensor(x_np)
    ref = (0.5 * x_t * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x_t + 0.044715 * x_t ** 3)))).numpy()

    assert np.allclose(trt_out, ref, atol=1e-5), \
        f"gelu_new: max diff {np.abs(trt_out - ref).max():.2e}"

    print("  PASS  add_gelu_new")


# ---------------------------------------------------------------
# 8. add_activation — silu, relu
# ---------------------------------------------------------------

def test_activations():
    rng = np.random.RandomState(42)
    x_np = rng.randn(1, 128).astype(np.float32)

    for act_name, torch_fn in [("silu", nn.SiLU()), ("relu", nn.ReLU()),
                                ("gelu_new", None), ("gelu", None)]:
        def build(net, inp, an=act_name):
            out = graph_ops.add_activation(net, inp["x"], an)
            return {"out": out}

        result = _run_trt_graph(build, {"x": x_np})
        trt_out = result["out"]

        x_t = torch.tensor(x_np)
        if torch_fn is not None:
            ref = torch_fn(x_t).numpy()
        else:
            # gelu_new / gelu use tanh approx
            ref = (0.5 * x_t * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x_t + 0.044715 * x_t ** 3)))).numpy()

        assert np.allclose(trt_out, ref, atol=1e-5), \
            f"activation {act_name}: max diff {np.abs(trt_out - ref).max():.2e}"

    print("  PASS  add_activation (silu, relu, gelu_new, gelu)")


# ---------------------------------------------------------------
# 9. add_apply_rope — rotated-half (LLaMA) and interleaved (CodeGen)
# ---------------------------------------------------------------

def _hf_apply_rope_llama(x, cos, sin, head_dim):
    """LLaMA-style: rotate_half pairs (d, d+half)."""
    half = head_dim // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


def _hf_apply_rope_codegen(x, cos_interleaved, sin_interleaved):
    """CodeGen-style: rotate_every_two pairs (d, d+1)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = np.stack([-x2, x1], axis=-1).reshape(x.shape)
    return x * cos_interleaved + rotated * sin_interleaved


def test_apply_rope():
    rng = np.random.RandomState(42)
    hidden = 64
    heads = 4
    head_dim = 16
    max_len = 8

    for interleaved in [False, True]:
        for pos in [0, 3, 7]:
            x_np = rng.randn(1, hidden).astype(np.float32)
            cos_tbl = graph_ops.make_rope_table(
                max_len, hidden, heads, 10000.0, True, interleaved=interleaved)
            sin_tbl = graph_ops.make_rope_table(
                max_len, hidden, heads, 10000.0, False, interleaved=interleaved)
            rot_mat = graph_ops.make_rotate_half_matrix(
                hidden, heads, interleaved=interleaved)

            def build(net, inp, ct=cos_tbl, st=sin_tbl, rm=rot_mat):
                cos_t = graph_ops.add_constant(net, ct.shape, ct)
                sin_t = graph_ops.add_constant(net, st.shape, st)
                rot_t = graph_ops.add_constant(net, rm.shape, rm)
                out = graph_ops.add_apply_rope(net, inp["x"], inp["pos"], cos_t, sin_t, rot_t)
                return {"out": out}

            pos_np = np.array([pos], dtype=np.int32)
            result = _run_trt_graph(build, {"x": x_np, "pos": pos_np})
            trt_out = result["out"].flatten()

            # Reference: per-head application
            cos_row = cos_tbl[pos]  # [hidden]
            sin_row = sin_tbl[pos]  # [hidden]
            ref_parts = []
            for h in range(heads):
                s = h * head_dim
                e = s + head_dim
                xh = x_np[0, s:e]
                ch = cos_row[s:e]
                sh = sin_row[s:e]
                if interleaved:
                    rh = _hf_apply_rope_codegen(xh, ch, sh)
                else:
                    rh = _hf_apply_rope_llama(xh, ch, sh, head_dim)
                ref_parts.append(rh)
            ref = np.concatenate(ref_parts)

            assert np.allclose(trt_out, ref, atol=1e-5), \
                f"rope interleaved={interleaved} pos={pos}: " \
                f"max diff {np.abs(trt_out - ref).max():.2e}"

    # Partial rotary (factor=0.5), standard
    pf = 0.5
    cos_tbl_p = graph_ops.make_rope_table(max_len, hidden, heads, 10000.0, True,
                                           partial_rotary_factor=pf)
    sin_tbl_p = graph_ops.make_rope_table(max_len, hidden, heads, 10000.0, False,
                                           partial_rotary_factor=pf)
    rot_mat_p = graph_ops.make_rotate_half_matrix(hidden, heads, partial_rotary_factor=pf)
    x_np = rng.randn(1, hidden).astype(np.float32)

    def build_p(net, inp):
        cos_t = graph_ops.add_constant(net, cos_tbl_p.shape, cos_tbl_p)
        sin_t = graph_ops.add_constant(net, sin_tbl_p.shape, sin_tbl_p)
        rot_t = graph_ops.add_constant(net, rot_mat_p.shape, rot_mat_p)
        out = graph_ops.add_apply_rope(net, inp["x"], inp["pos"], cos_t, sin_t, rot_t)
        return {"out": out}

    pos_np = np.array([3], dtype=np.int32)
    result = _run_trt_graph(build_p, {"x": x_np, "pos": pos_np})
    trt_out = result["out"].flatten()

    # Reference: partial rotary — first 8 dims get RoPE, last 8 are identity
    rotary_nd = int(head_dim * pf)
    cos_row_p = cos_tbl_p[3]
    sin_row_p = sin_tbl_p[3]
    ref_parts = []
    for h in range(heads):
        s = h * head_dim
        xh = x_np[0, s:s+head_dim]
        ch = cos_row_p[s:s+head_dim]
        sh = sin_row_p[s:s+head_dim]
        # RoPE applied to first rotary_nd dims
        xr = _hf_apply_rope_llama(xh[:rotary_nd], ch[:rotary_nd], sh[:rotary_nd], rotary_nd)
        ref_parts.append(np.concatenate([xr, xh[rotary_nd:]]))
    ref = np.concatenate(ref_parts)
    assert np.allclose(trt_out, ref, atol=1e-5), \
        f"rope partial=0.5: max diff {np.abs(trt_out - ref).max():.2e}"

    print("  PASS  add_apply_rope (7 combinations: rotated/interleaved × positions × partial)")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("graph_ops unit tests vs HuggingFace references")
    print("=" * 55)

    # Pure numpy tests (no TRT needed)
    test_alibi_slopes()
    test_rope_table()
    test_rotate_half_matrix()

    # TRT graph op tests
    test_rms_norm()
    test_rms_norm_per_head()
    test_layer_norm()
    test_gelu_new()
    test_activations()
    test_apply_rope()

    print("=" * 55)
    print("ALL PASS")


if __name__ == "__main__":
    main()
