#!/usr/bin/env python3
"""Focused graph_ops checks against Hugging Face references.

Tests retained family-owned operations for:
  - compute_alibi_slopes: power-of-2 (8,16) and non-power-of-2 (6,12)
  - add_rms_norm: vs torch manual RMSNorm
  - add_rms_norm_per_head: vs torch per-head RMSNorm
  - add_layer_norm: vs torch.nn.LayerNorm
  - add_gelu_new: vs HF NewGELUActivation (tanh approx)
  - add_activation(silu): vs torch.nn.SiLU
  - add_activation(relu): vs torch.nn.ReLU

Run inside the container:
    python3 tools/test_graph_ops.py
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

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


class _OwnedGraphOps:
    """Resolve each operation from a family that actually retains it."""

    def __init__(self):
        families = Path("python") / "tensorrt_model_connect" / "families"
        self._module_names = [
            f"tensorrt_model_connect.families.{family_dir.name}.model.model"
            for family_dir in sorted(families.iterdir())
            if (family_dir / "plugin.py").is_file()
            and (family_dir / "model" / "model.py").is_file()
        ]

    def __getattr__(self, name: str):
        for module_name in self._module_names:
            module = importlib.import_module(module_name)
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(f"No family-owned graph_ops.py defines {name}")


graph_ops = _OwnedGraphOps()


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
    """Reference implementation for ALiBi slope computation."""
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
# Main
# ---------------------------------------------------------------

def main():
    print("graph_ops unit tests vs HuggingFace references")
    print("=" * 55)

    # Pure NumPy plus TensorRT graph checks.
    test_alibi_slopes()
    test_rms_norm()
    test_rms_norm_per_head()
    test_layer_norm()
    test_gelu_new()
    test_activations()
    print("=" * 55)
    print("ALL PASS")


if __name__ == "__main__":
    main()
