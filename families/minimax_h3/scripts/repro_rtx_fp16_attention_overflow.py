# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro: TRT-RTX fused FP16 IAttention returns inf for large sum(p*v).

One head, 38247 keys, head_dim 128. All logits are ~30 (near-uniform softmax, so
every key has weight ~1/N) and one V column has mean ~100. The mathematically
exact output is bounded by max|V| (~400), far inside FP16 range, but the fused
FP16 kernel returns +-inf for that column on TensorRT-RTX 1.7.1.134 / 1.8.1.68
(sm_120): the *unnormalized* sum_j p_j v_j (~N * 100 = 3.8e6) apparently exceeds
the FP16 range before the 1/sum(p) normalization. Scaling V by an exact power of two
(and the result back in FP32) makes the same kernel finite.

    python repro_rtx_fp16_attention_overflow.py        # needs a CUDA GPU + tensorrt_rtx
"""

import numpy as np
from cuda.bindings import runtime as cudart

try:
    import tensorrt_rtx as trt
except ImportError:  # pragma: no cover
    try:
        import tensorrt_rtx_bindings as trt
    except ImportError:
        import tensorrt as trt

N, D = 38247, 128


def _ck(result):
    error = result[0] if isinstance(result, tuple) else result
    assert int(error) == 0, f"CUDA error {error}"
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


def run(q, k, v, v_scale):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    shape = q.shape
    qi, ki, vi = (net.add_input(n, trt.float16, shape) for n in ("q", "k", "v"))

    def const(value, dtype):
        arr = np.full((1, 1, 1, 1), value, dtype=np.float32)
        return net.add_cast(net.add_constant(arr.shape, trt.Weights(arr)).get_output(0), dtype).get_output(0)

    vv = vi
    if v_scale != 1.0:
        vv = net.add_elementwise(vi, const(1.0 / v_scale, trt.float16), trt.ElementWiseOperation.PROD).get_output(0)
    attention = net.add_attention(qi, ki, vv, trt.AttentionNormalizationOp.SOFTMAX, False)
    attention.decomposable = False
    out = attention.get_output(0)
    if v_scale != 1.0:
        wide = net.add_cast(out, trt.float32).get_output(0)
        wide = net.add_elementwise(wide, const(v_scale, trt.float32), trt.ElementWiseOperation.PROD).get_output(0)
        out = net.add_cast(wide, trt.float16).get_output(0)
    out.name = "out"
    net.mark_output(out)
    plan = builder.build_serialized_network(net, builder.create_builder_config())
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()
    stream = _ck(cudart.cudaStreamCreate())
    for name, array in (("q", q), ("k", k), ("v", v)):
        ptr = _ck(cudart.cudaMalloc(array.nbytes))
        _ck(cudart.cudaMemcpy(ptr, array.ctypes.data, array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        ctx.set_tensor_address(name, int(ptr))
    host = np.empty(shape, np.float16)
    ptr = _ck(cudart.cudaMalloc(host.nbytes))
    ctx.set_tensor_address("out", int(ptr))
    assert ctx.execute_async_v3(int(stream))
    _ck(cudart.cudaStreamSynchronize(stream))
    _ck(cudart.cudaMemcpy(host.ctypes.data, ptr, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    return host


def main():
    rng = np.random.default_rng(0)
    e = np.ones(D, np.float32) / np.sqrt(D)
    q = (1.0 * e + 0.05 * rng.standard_normal((1, 1, N, D))).astype(np.float16)
    k = (30.0 * e + 0.5 * rng.standard_normal((1, 1, N, D))).astype(np.float16)
    v = rng.standard_normal((1, 1, N, D)).astype(np.float32)
    v[..., 52] = 100 + 100 * rng.standard_normal((1, 1, N))
    v = v.astype(np.float16)
    for scale in (1.0, 256.0):
        out = run(q, k, v, scale)
        print(f"V scale {scale:g}: non-finite outputs = {int((~np.isfinite(out)).sum())} of {out.size}")
    print("expected: 0 non-finite for both (|true output| <= max|V|); the unscaled run is non-zero on affected stacks")


if __name__ == "__main__":
    main()
