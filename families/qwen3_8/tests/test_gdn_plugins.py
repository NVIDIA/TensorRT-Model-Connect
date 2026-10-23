# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-plugin TensorRT engines for the Qwen3.8 GDN decode and prefill plugins vs a torch reference.

Needs a CUDA GPU, TensorRT, cuda.tile (cuTile) and a C++ compiler with cmake; skipped otherwise.
The reference runs the Gated DeltaNet core token by token in fp32.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
pytest.importorskip("cuda.tile")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

from families.qwen3_8.native_plugins.gdn_plugin_builder import (  # noqa: E402
    add_gdn_decode_plugin,
    add_gdn_prefill_plugin,
)

H, NKV, D, DC = 48, 16, 128, 4
GPG = H // NKV
QK = NKV * D
DI = H * D
CD = 2 * QK + DI
SEQ = 128
DEV = "cuda"


def reference_step(qkv, z, a, b, conv, w, bias, A, dt_bias, gamma, S, dtype):
    """One token of the GDN core (conv1d step, gates, delta rule, gated RMSNorm) in fp32."""
    f = torch.float32
    new_conv = torch.cat([conv[:, 1:], qkv.T.to(f)], 1)
    y = torch.nn.functional.silu((new_conv * w).sum(1) + bias)
    q, k, v = y[:QK].view(NKV, D), y[QK:2 * QK].view(NKV, D), y[2 * QK:].view(H, D)
    q = torch.nn.functional.normalize(q, dim=1, eps=1e-6).repeat_interleave(GPG, 0) * D ** -0.5
    k = torch.nn.functional.normalize(k, dim=1, eps=1e-6).repeat_interleave(GPG, 0)
    g = A * torch.nn.functional.softplus(a.view(-1).to(f) + dt_bias)
    beta = torch.sigmoid(b.view(-1).to(f))
    S = S * g.exp().view(H, 1, 1)
    kv = torch.einsum("hkv,hk->hv", S, k)
    S = S + torch.einsum("hk,hv->hkv", k, (v - kv) * beta.view(H, 1))
    o = torch.einsum("hkv,hk->hv", S, q).to(dtype).to(f)
    o = o * torch.rsqrt(o.pow(2).mean(1, keepdim=True) + 1e-6) * gamma.view(H, D)
    o = o * torch.nn.functional.silu(z.view(H, D).to(f))
    return o.to(dtype).view(1, DI), new_conv, S


def make_params(seed=5):
    g = torch.Generator().manual_seed(seed)
    return dict(
        w=(torch.randn(CD, DC, generator=g) * 0.5).to(DEV),
        bias=(torch.randn(CD, generator=g) * 0.1).to(DEV),
        A=(-torch.rand(H, generator=g) * 4).to(DEV),
        dt_bias=torch.randn(H, generator=g).to(DEV),
        gamma=(torch.rand(D, generator=g) + 0.5).to(DEV).repeat(H),  # per head-dim weight
    )


def add_constants(net, p):
    def const(arr):
        return net.add_constant(
            arr.shape, trt.Weights(np.ascontiguousarray(arr.cpu().numpy()))).get_output(0)
    return [const(p["w"]), const(p["bias"]), const(p["A"]), const(p["dt_bias"]), const(p["gamma"])]


HID = 5120  # hidden size of the a and b projections


def build_engine(kind, bf16, p, fuse_ab=False):
    tdt = trt.bfloat16 if bf16 else trt.float16
    rows = 1 if kind == "decode" else SEQ
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    ins = [net.add_input("qkv", tdt, (rows, CD)), net.add_input("z", tdt, (rows, DI))]
    if fuse_ab:  # normalized hidden row and the a|b projection weight
        ins += [net.add_input("x", tdt, (rows, HID)),
                net.add_input("w_ab", trt.float32, (2 * H, HID))]
    else:
        ins += [net.add_input("a", tdt, (rows, H)), net.add_input("b", tdt, (rows, H))]
    ins.append(net.add_input("conv_in", tdt, (CD, DC)))
    ins += add_constants(net, p)
    ins.append(net.add_input("ssm_in", trt.float32, (H, D, D)))
    if kind == "decode":
        layer = add_gdn_decode_plugin(net, trt, inputs=ins, num_heads=H, num_kv_heads=NKV,
                                      head_dim=D, bf16=bf16,
                                      hidden_size=HID if fuse_ab else 0)
    else:
        ins.append(net.add_input("valid_len", trt.int32, (1,)))
        layer = add_gdn_prefill_plugin(net, trt, inputs=ins, num_heads=H, num_kv_heads=NKV,
                                       head_dim=D, seq_len=SEQ, bf16=bf16)
    for i, name in enumerate(["out", "conv_out", "ssm_out"]):
        out = layer.get_output(i)
        out.name = name
        net.mark_output(out)
    blob = builder.build_serialized_network(net, builder.create_builder_config())
    assert blob is not None, "engine build failed"
    return trt.Runtime(logger).deserialize_cuda_engine(blob).create_execution_context()


def run_engine(ctx, tensors):
    for name, t in tensors.items():
        ctx.set_tensor_address(name, t.data_ptr())
    assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()


def tolerances(bf16, absmax):
    # bf16 keeps about 3 significant digits; the reference is fp32
    return (0.01 * absmax + 0.03, 0.02) if bf16 else (0.002 * absmax + 0.01, 0.003)


def err(x, y):
    return (x.float() - y.float()).abs().max().item()


@pytest.mark.parametrize("fuse_ab", [False, True], ids=["ab_inputs", "fused_ab"])
@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp16"])
def test_decode_plugin(bf16, fuse_ab):
    dt = torch.bfloat16 if bf16 else torch.float16
    p = make_params()
    ctx = build_engine("decode", bf16, p, fuse_ab)
    torch.manual_seed(1)
    qkv, z = torch.randn(1, CD, device=DEV, dtype=dt), torch.randn(1, DI, device=DEV, dtype=dt)
    a, b = torch.randn(1, H, device=DEV, dtype=dt), torch.randn(1, H, device=DEV, dtype=dt)
    conv0 = torch.randn(CD, DC, device=DEV).to(dt)
    S0 = torch.randn(H, D, D, device=DEV) * 0.1
    out = torch.empty(1, DI, device=DEV, dtype=dt)
    conv_o, ssm_o = torch.empty(CD, DC, device=DEV, dtype=dt), torch.empty_like(S0)
    if fuse_ab:  # a and b come from x @ w_ab^T, rounded to the activation dtype
        x = torch.randn(1, HID, device=DEV, dtype=dt)
        w_ab = torch.randn(2 * H, HID, device=DEV) * 0.05
        ab = (x.float() @ w_ab.T).to(dt)
        a, b = ab[:, :H].contiguous(), ab[:, H:].contiguous()
        inputs = dict(qkv=qkv, z=z, x=x, w_ab=w_ab)
    else:
        inputs = dict(qkv=qkv, z=z, a=a, b=b)
    run_engine(ctx, dict(**inputs, conv_in=conv0, ssm_in=S0, out=out, conv_out=conv_o,
                         ssm_out=ssm_o))
    r_out, r_conv, r_S = reference_step(qkv, z, a, b, conv0.float(), p["w"], p["bias"], p["A"],
                                        p["dt_bias"], p["gamma"], S0, dt)
    out_tol, ssm_tol = tolerances(bf16, r_out.float().abs().max().item())
    assert err(out, r_out) <= out_tol
    assert err(conv_o, r_conv) == 0
    assert err(ssm_o, r_S) <= ssm_tol


def reference_tokens(qkv, z, a, b, conv, S, p, count, dt):
    rows = []
    for t in range(count):
        o, conv, S = reference_step(qkv[t:t + 1], z[t:t + 1], a[t:t + 1], b[t:t + 1], conv, p["w"],
                                    p["bias"], p["A"], p["dt_bias"], p["gamma"], S, dt)
        rows.append(o)
    return torch.cat(rows, 0), conv, S


@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp16"])
@pytest.mark.parametrize("valid_len", [128, 100, 64, 5, 1])
def test_prefill_plugin_padding(bf16, valid_len):
    """A chunk with `valid_len` real tokens: valid rows match, padding rows are zero, and the
    states reflect only the real tokens."""
    dt = torch.bfloat16 if bf16 else torch.float16
    p = make_params()
    ctx = build_engine("prefill", bf16, p)
    torch.manual_seed(valid_len)
    qkv, z = torch.randn(SEQ, CD, device=DEV, dtype=dt), torch.randn(SEQ, DI, device=DEV, dtype=dt)
    a, b = torch.randn(SEQ, H, device=DEV, dtype=dt), torch.randn(SEQ, H, device=DEV, dtype=dt)
    conv0 = torch.randn(CD, DC, device=DEV).to(dt)
    S0 = torch.randn(H, D, D, device=DEV) * 0.1
    vl = torch.tensor([valid_len], device=DEV, dtype=torch.int32)
    out = torch.full((SEQ, DI), 7.0, device=DEV, dtype=dt)
    conv_o, ssm_o = torch.empty(CD, DC, device=DEV, dtype=dt), torch.empty_like(S0)
    run_engine(ctx, dict(qkv=qkv, z=z, a=a, b=b, conv_in=conv0, ssm_in=S0, valid_len=vl, out=out,
                         conv_out=conv_o, ssm_out=ssm_o))
    r_out, r_conv, r_S = reference_tokens(qkv, z, a, b, conv0.float(), S0.clone(), p, valid_len, dt)
    out_tol, ssm_tol = tolerances(bf16, r_out.float().abs().max().item())
    assert err(out[:valid_len], r_out) <= out_tol
    assert (out[valid_len:] == 0).all()
    assert err(conv_o, r_conv) == 0
    assert err(ssm_o, r_S) <= ssm_tol


@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp16"])
def test_prefill_plugin_chunked_prompt(bf16):
    """A 300-token prompt as 128 + 128 + 44 (padded) with the states carried between calls."""
    dt = torch.bfloat16 if bf16 else torch.float16
    total = 300
    p = make_params()
    ctx = build_engine("prefill", bf16, p)
    torch.manual_seed(77)
    qkv, z = torch.randn(total, CD, device=DEV, dtype=dt), torch.randn(total, DI, device=DEV, dtype=dt)
    a, b = torch.randn(total, H, device=DEV, dtype=dt), torch.randn(total, H, device=DEV, dtype=dt)
    conv = torch.randn(CD, DC, device=DEV).to(dt)
    S = torch.randn(H, D, D, device=DEV) * 0.1
    r_out, r_conv, r_S = reference_tokens(qkv, z, a, b, conv.float(), S.clone(), p, total, dt)
    outs = []
    for start in range(0, total, SEQ):
        n = min(SEQ, total - start)

        def chunk(x):
            pad = torch.zeros(SEQ - n, x.shape[1], device=DEV, dtype=dt)
            return torch.cat([x[start:start + n], pad], 0)
        out = torch.empty(SEQ, DI, device=DEV, dtype=dt)
        conv_o, ssm_o = torch.empty(CD, DC, device=DEV, dtype=dt), torch.empty_like(S)
        run_engine(ctx, dict(qkv=chunk(qkv), z=chunk(z), a=chunk(a), b=chunk(b), conv_in=conv,
                             ssm_in=S, valid_len=torch.tensor([n], device=DEV, dtype=torch.int32),
                             out=out, conv_out=conv_o, ssm_out=ssm_o))
        outs.append(out[:n].clone())
        conv, S = conv_o, ssm_o
    out_tol, ssm_tol = tolerances(bf16, r_out.float().abs().max().item())
    assert err(torch.cat(outs, 0), r_out) <= out_tol
    assert err(conv, r_conv) == 0
    assert err(S, r_S) <= ssm_tol
