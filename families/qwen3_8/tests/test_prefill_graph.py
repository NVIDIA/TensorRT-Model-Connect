# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-token prefill layers must reproduce the single-token layers they replace.

Each GPU test builds a tiny one-layer engine twice -- the existing single-token
layer driven token by token, and the batched prefill layer driven once -- and
compares outputs and persistent state, including a partial last chunk
(``valid_len < chunk_len``) and a chunk longer than the delta rule's inner block.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")

HID, H, HK, D, DCONV, MLP = 64, 4, 2, 16, 4, 96
D_INNER = H * D
QK = HK * D
CONV_DIM = 2 * QK + D_INNER
ATT_HEADS, KV_HEADS, ATT_D, ROT = 4, 2, 16, 8
ATT_SIZE = ATT_HEADS * ATT_D
KV_SIZE = KV_HEADS * ATT_D


def _need_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")


def _rnd(rng, *shape, scale=0.1):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _deltanet_weights(rng, prefix):
    return {
        f"{prefix}.input_norm": 1 + _rnd(rng, HID),
        f"{prefix}.deltanet_in_proj_qkv": _rnd(rng, HID, CONV_DIM, scale=0.3),
        f"{prefix}.deltanet_z_proj": _rnd(rng, HID, D_INNER, scale=0.3),
        f"{prefix}.deltanet_a_proj": _rnd(rng, HID, H, scale=0.3),
        f"{prefix}.deltanet_b_proj": _rnd(rng, HID, H, scale=0.3),
        f"{prefix}.A": -np.exp(rng.uniform(-1.5, 0.5, H)).astype(np.float32),
        f"{prefix}.dt_bias": _rnd(rng, H, scale=0.5),
        f"{prefix}.conv1d_weight": _rnd(rng, CONV_DIM, DCONV, scale=0.5),
        f"{prefix}.conv1d_bias": _rnd(rng, CONV_DIM),
        f"{prefix}.deltanet_norm": 1 + _rnd(rng, D_INNER),
        f"{prefix}.deltanet_out_proj": _rnd(rng, D_INNER, HID, scale=0.3),
        f"{prefix}.post_attn_norm": 1 + _rnd(rng, HID),
        f"{prefix}.w_gate": _rnd(rng, HID, MLP, scale=0.3),
        f"{prefix}.w_up": _rnd(rng, HID, MLP, scale=0.3),
        f"{prefix}.w_down": _rnd(rng, MLP, HID, scale=0.3),
    }


def _attention_weights(rng, prefix):
    return {
        f"{prefix}.input_norm": 1 + _rnd(rng, HID),
        f"{prefix}.w_q": _rnd(rng, HID, ATT_SIZE, scale=0.3),
        f"{prefix}.w_gate_attn": _rnd(rng, HID, ATT_SIZE, scale=0.3),
        f"{prefix}.w_k": _rnd(rng, HID, KV_SIZE, scale=0.3),
        f"{prefix}.w_v": _rnd(rng, HID, KV_SIZE, scale=0.3),
        f"{prefix}.w_o": _rnd(rng, ATT_SIZE, HID, scale=0.3),
        f"{prefix}.q_norm": 1 + _rnd(rng, ATT_D),
        f"{prefix}.k_norm": 1 + _rnd(rng, ATT_D),
        f"{prefix}.post_attn_norm": 1 + _rnd(rng, HID),
        f"{prefix}.w_gate": _rnd(rng, HID, MLP, scale=0.3),
        f"{prefix}.w_up": _rnd(rng, HID, MLP, scale=0.3),
        f"{prefix}.w_down": _rnd(rng, MLP, HID, scale=0.3),
    }


class _Net:
    """One tiny strongly-typed network plus the helpers to run it."""

    def __init__(self, precision: str) -> None:
        self.fp32 = precision == "fp32"
        self.np_dtype = np.float32 if self.fp32 else np.float16
        self.trt_dtype = trt.float32 if self.fp32 else trt.float16
        self.torch_dtype = torch.float32 if self.fp32 else torch.float16
        self.builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        self.config = self.builder.create_builder_config()
        self.config.builder_optimization_level = 1
        # The delta rule's fp32 tiles must not run in TF32 (see build_engine_prefill).
        self.config.clear_flag(trt.BuilderFlag.TF32)

    def cast_in(self, tensor):
        if self.fp32:
            return tensor
        return self.network.add_cast(tensor, self.trt_dtype).get_output(0)

    def mark(self, **tensors) -> None:
        for name, tensor in tensors.items():
            if tensor.dtype != trt.float32:
                tensor = self.network.add_cast(tensor, trt.float32).get_output(0)
            tensor.name = name
            self.network.mark_output(tensor)

    def build(self) -> bytes:
        plan = self.builder.build_serialized_network(self.network, self.config)
        assert plan is not None
        return bytes(plan)


def _run(plan: bytes, feeds: dict, outputs: list[str]) -> dict:
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    keep, result = [], {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            tensor = feeds[name].contiguous().cuda()
        else:
            dtype = {trt.float32: torch.float32, trt.float16: torch.float16,
                     trt.int32: torch.int32}[engine.get_tensor_dtype(name)]
            tensor = torch.zeros(tuple(context.get_tensor_shape(name)), dtype=dtype, device="cuda")
            result[name] = tensor
        keep.append(tensor)
        context.set_tensor_address(name, tensor.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return {name: result[name].float().cpu() for name in outputs}


def _assert_close(actual, expected, tolerance, what) -> None:
    error = (actual - expected).abs().max().item()
    scale = max(expected.abs().max().item(), 1e-3)
    assert error <= tolerance * scale, f"{what}: max|err|={error:.3e} vs max|ref|={scale:.3e}"


@pytest.mark.gpu
@pytest.mark.parametrize("precision,chunk_len,tolerance",
                         [("fp32", 16, 1e-5), ("fp32", 128, 1e-5), ("fp16", 128, 3e-2)])
def test_deltanet_prefill_matches_single_token(precision, chunk_len, tolerance) -> None:
    _need_gpu()
    from families.qwen3_8 import engine_builder, graph_ops, prefill_graph

    rng = np.random.default_rng(0)
    weights = _deltanet_weights(rng, "L")

    single = _Net(precision)
    x = single.network.add_input("x", single.trt_dtype, (1, HID))
    conv = single.network.add_input("conv_state", trt.float32, (CONV_DIM, DCONV))
    ssm = single.network.add_input("ssm_state", trt.float32, (H, D, D))
    eps = graph_ops.add_constant(single.network, (1, 1), np.array([1e-6]), dtype=single.np_dtype)
    out = engine_builder._add_deltanet_layer(
        network=single.network, hidden=x, conv_state_in=single.cast_in(conv), ssm_state_in=ssm,
        eps_tensor=eps, weights=weights, prefix="L", hidden_size=HID, d_inner=D_INNER,
        d_conv=DCONV, conv_dim=CONV_DIM, num_heads=H, num_kv_heads=HK, head_dim=D,
        mlp_size=MLP, dtype=single.np_dtype)
    single.mark(hidden=out["hidden"], present_conv=out["present_conv"],
                present_ssm=out["present_ssm"])
    single_plan = single.build()

    batched = _Net(precision)
    x = batched.network.add_input("x", batched.trt_dtype, (chunk_len, HID))
    conv = batched.network.add_input("conv_state", trt.float32, (CONV_DIM, DCONV))
    ssm = batched.network.add_input("ssm_state", trt.float32, (H, D, D))
    valid_len = batched.network.add_input("valid_len", trt.int32, (1,))
    eps = graph_ops.add_constant(batched.network, (1, 1), np.array([1e-6]), dtype=batched.np_dtype)
    out = prefill_graph.add_deltanet_layer_prefill(
        network=batched.network, hidden=x, conv_state_in=batched.cast_in(conv), ssm_state_in=ssm,
        eps_tensor=eps,
        valid_rows=prefill_graph.add_valid_row_mask(batched.network, valid_len, chunk_len),
        conv_gather_index=prefill_graph.add_conv_state_gather_index(
            batched.network, valid_len, DCONV),
        weights=weights, prefix="L", chunk_len=chunk_len, block_len=min(chunk_len, 64), hidden_size=HID,
        d_inner=D_INNER, d_conv=DCONV, conv_dim=CONV_DIM, num_heads=H, num_kv_heads=HK,
        head_dim=D, mlp_size=MLP, dtype=batched.np_dtype)
    batched.mark(hidden=out["hidden"], present_conv=out["present_conv"],
                 present_ssm=out["present_ssm"])
    batched_plan = batched.build()

    names = ["hidden", "present_conv", "present_ssm"]
    for valid in (chunk_len, chunk_len - 3, chunk_len // 2 + 1, 1):
        xs = torch.from_numpy(_rnd(rng, chunk_len, HID, scale=1.0)).to(single.torch_dtype)
        conv0 = torch.from_numpy(_rnd(rng, CONV_DIM, DCONV, scale=0.5))
        ssm0 = torch.from_numpy(_rnd(rng, H, D, D, scale=0.3))
        conv_t, ssm_t, rows = conv0.clone(), ssm0.clone(), []
        for t in range(valid):
            step = _run(single_plan, {"x": xs[t:t + 1], "conv_state": conv_t, "ssm_state": ssm_t},
                        names)
            rows.append(step["hidden"])
            conv_t, ssm_t = step["present_conv"], step["present_ssm"]
        got = _run(batched_plan, {"x": xs, "conv_state": conv0, "ssm_state": ssm0,
                                  "valid_len": torch.tensor([valid], dtype=torch.int32)}, names)
        _assert_close(got["hidden"][:valid], torch.cat(rows), tolerance, f"hidden valid={valid}")
        _assert_close(got["present_conv"], conv_t, tolerance, f"conv valid={valid}")
        _assert_close(got["present_ssm"], ssm_t, tolerance, f"ssm valid={valid}")


@pytest.mark.gpu
@pytest.mark.parametrize("precision,tolerance", [("fp32", 1e-5), ("fp16", 3e-2)])
def test_attention_prefill_matches_single_token(precision, tolerance) -> None:
    _need_gpu()
    from families.qwen3_8 import engine_builder, graph_ops, prefill_graph

    chunk_len, cached, max_cache = 16, 5, 24
    rng = np.random.default_rng(1)
    weights = _attention_weights(rng, "L")
    cos = graph_ops.make_rope_table_half_dim(
        max_cache + chunk_len, ATT_D, 10000.0, cosine=True, partial_rotary_factor=ROT / ATT_D)
    sin = graph_ops.make_rope_table_half_dim(
        max_cache + chunk_len, ATT_D, 10000.0, cosine=False, partial_rotary_factor=ROT / ATT_D)

    def build(rows: int, single_token: bool) -> bytes:
        net = _Net(precision)
        n = net.network
        x = n.add_input("x", net.trt_dtype, (rows, HID))
        pos = n.add_input("pos", trt.int32, (rows,))
        mask_width = max_cache + (1 if single_token else rows)
        mask = n.add_input("mask", trt.float32, (rows, max_cache if single_token else mask_width))
        cache_k = n.add_input("cache_k", net.trt_dtype, (max_cache, KV_SIZE))
        cache_v = n.add_input("cache_v", net.trt_dtype, (max_cache, KV_SIZE))
        eps = graph_ops.add_constant(n, (1, 1), np.array([1e-6]), dtype=net.np_dtype)
        cos_t = graph_ops.add_constant(n, cos.shape, cos, dtype=net.np_dtype)
        sin_t = graph_ops.add_constant(n, sin.shape, sin, dtype=net.np_dtype)
        mask = net.cast_in(mask)
        common = dict(
            network=n, hidden=x, cache_k=cache_k, cache_v=cache_v, eps_tensor=eps,
            cos_half_tensor=cos_t, sin_half_tensor=sin_t, weights=weights, prefix="L",
            hidden_size=HID, attn_size=ATT_SIZE, kv_attention_size=KV_SIZE,
            num_heads=ATT_HEADS, num_kv_heads=KV_HEADS, head_dim=ATT_D,
            rotary_embedding_dim=ROT, max_cache_length=max_cache, mlp_size=MLP,
            dtype=net.np_dtype)
        if single_token:
            self_col = graph_ops.add_constant(n, (1, 1), np.zeros((1, 1)), dtype=net.np_dtype)
            full = n.add_concatenation([mask, self_col])
            full.axis = 1
            out = engine_builder._add_full_attention_layer(
                attention_mask=full.get_output(0), position_id=pos, **common)
        else:
            out = prefill_graph.add_full_attention_layer_prefill(
                attention_mask=mask, position_ids=pos, chunk_len=rows, **common)
        net.mark(hidden=out["hidden"], present_k=out["present_k"], present_v=out["present_v"])
        return net.build()

    single_plan, batched_plan = build(1, True), build(chunk_len, False)
    names = ["hidden", "present_k", "present_v"]
    dtype = torch.float32 if precision == "fp32" else torch.float16
    xs = torch.from_numpy(_rnd(rng, chunk_len, HID, scale=1.0)).to(dtype)
    base_k = torch.from_numpy(_rnd(rng, max_cache, KV_SIZE, scale=1.0)).to(dtype)
    base_v = torch.from_numpy(_rnd(rng, max_cache, KV_SIZE, scale=1.0)).to(dtype)

    cache_k, cache_v, rows, ks, vs = base_k.clone(), base_v.clone(), [], [], []
    for t in range(chunk_len):
        mask = torch.full((1, max_cache), -1e4)
        mask[:, :cached + t] = 0
        step = _run(single_plan, {"x": xs[t:t + 1], "pos": torch.tensor([cached + t], dtype=torch.int32),
                                  "mask": mask, "cache_k": cache_k, "cache_v": cache_v}, names)
        rows.append(step["hidden"])
        ks.append(step["present_k"])
        vs.append(step["present_v"])
        cache_k[cached + t] = step["present_k"][0].to(dtype)
        cache_v[cached + t] = step["present_v"][0].to(dtype)

    mask = torch.full((chunk_len, max_cache + chunk_len), -1e4)
    mask[:, :cached] = 0
    for i in range(chunk_len):
        mask[i, max_cache:max_cache + i + 1] = 0
    got = _run(batched_plan, {"x": xs, "pos": torch.arange(cached, cached + chunk_len, dtype=torch.int32),
                              "mask": mask, "cache_k": base_k, "cache_v": base_v}, names)
    _assert_close(got["hidden"], torch.cat(rows), tolerance, "hidden")
    _assert_close(got["present_k"], torch.cat(ks), tolerance, "present_k")
    _assert_close(got["present_v"], torch.cat(vs), tolerance, "present_v")


@pytest.mark.gpu
@pytest.mark.parametrize("precision,tolerance", [("fp32", 1e-5), ("fp16", 3e-2)])
@pytest.mark.parametrize("cached", [0, 5, 13])
@pytest.mark.parametrize("rows_dynamic", [True, False], ids=["dynamic-rows", "fixed-rows"])
def test_attention_prefill_dynamic_cache_matches_full_cache(
    precision, tolerance, cached, rows_dynamic
) -> None:
    """Attention over only the filled cache rows (dynamic past length) = attention over the whole
    cache buffer with the unfilled rows masked, with a dynamic or a fixed token count."""
    _need_gpu()
    from families.qwen3_8 import graph_ops, prefill_graph

    rows, max_cache = 16, 24
    rng = np.random.default_rng(2)
    weights = _attention_weights(rng, "L")
    cos = graph_ops.make_rope_table_half_dim(
        max_cache + rows, ATT_D, 10000.0, cosine=True, partial_rotary_factor=ROT / ATT_D)
    sin = graph_ops.make_rope_table_half_dim(
        max_cache + rows, ATT_D, 10000.0, cosine=False, partial_rotary_factor=ROT / ATT_D)

    def build(dynamic: bool) -> bytes:
        net = _Net(precision)
        n = net.network
        row_dim = -1 if dynamic and rows_dynamic else rows
        x = n.add_input("x", net.trt_dtype, (row_dim, HID))
        pos = n.add_input("pos", trt.int32, (row_dim,))
        mask = n.add_input(
            "mask", trt.float32,
            (-1, -1) if dynamic and rows_dynamic else (rows, -1 if dynamic else max_cache + rows))
        past = -1 if dynamic else max_cache
        cache_k = n.add_input("cache_k", net.trt_dtype, (past, KV_SIZE))
        cache_v = n.add_input("cache_v", net.trt_dtype, (past, KV_SIZE))
        if dynamic:
            profile = net.builder.create_optimization_profile()
            if rows_dynamic:
                profile.set_shape("x", (1, HID), (rows, HID), (rows, HID))
                profile.set_shape("pos", (1,), (rows,), (rows,))
            profile.set_shape("mask", (1, 1) if rows_dynamic else (rows, rows),
                              (rows, 5 + rows), (rows, max_cache + rows))
            for name in ("cache_k", "cache_v"):
                profile.set_shape(name, (0, KV_SIZE), (5, KV_SIZE), (max_cache, KV_SIZE))
            net.config.add_optimization_profile(profile)
        eps = graph_ops.add_constant(n, (1, 1), np.array([1e-6]), dtype=net.np_dtype)
        cos_t = graph_ops.add_constant(n, cos.shape, cos, dtype=net.np_dtype)
        sin_t = graph_ops.add_constant(n, sin.shape, sin, dtype=net.np_dtype)
        out = prefill_graph.add_full_attention_layer_prefill(
            network=n, hidden=x, cache_k=cache_k, cache_v=cache_v, attention_mask=net.cast_in(mask),
            position_ids=pos, chunk_len=None if dynamic and rows_dynamic else rows, eps_tensor=eps,
            cos_half_tensor=cos_t, sin_half_tensor=sin_t, weights=weights, prefix="L",
            hidden_size=HID, attn_size=ATT_SIZE, kv_attention_size=KV_SIZE, num_heads=ATT_HEADS,
            num_kv_heads=KV_HEADS, head_dim=ATT_D, rotary_embedding_dim=ROT,
            max_cache_length=max_cache, mlp_size=MLP, dtype=net.np_dtype, dynamic_cache=dynamic)
        net.mark(hidden=out["hidden"], present_k=out["present_k"], present_v=out["present_v"])
        return net.build()

    def run_dynamic(plan: bytes, feeds: dict) -> dict:
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        ctx = engine.create_execution_context()
        shapes = {"x": (rows, HID), "pos": (rows,), "mask": (rows, cached + rows),
                  "cache_k": (cached, KV_SIZE), "cache_v": (cached, KV_SIZE)}
        for name, shape in shapes.items():
            assert ctx.set_input_shape(name, shape)
        keep, result = [], {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                tensor = feeds[name].contiguous().cuda()  # caches: the full buffer, first rows bound
            else:
                dtype = {trt.float32: torch.float32, trt.float16: torch.float16}[
                    engine.get_tensor_dtype(name)]
                tensor = torch.zeros(tuple(ctx.get_tensor_shape(name)), dtype=dtype, device="cuda")
                result[name] = tensor
            keep.append(tensor)
            ctx.set_tensor_address(name, tensor.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return {name: tensor.float().cpu() for name, tensor in result.items()}

    dtype = torch.float32 if precision == "fp32" else torch.float16
    xs = torch.from_numpy(_rnd(rng, rows, HID, scale=1.0)).to(dtype)
    base_k = torch.from_numpy(_rnd(rng, max_cache, KV_SIZE, scale=1.0)).to(dtype)
    base_v = torch.from_numpy(_rnd(rng, max_cache, KV_SIZE, scale=1.0)).to(dtype)
    pos = torch.arange(cached, cached + rows, dtype=torch.int32)

    full_mask = torch.full((rows, max_cache + rows), -1e4)
    full_mask[:, :cached] = 0
    for i in range(rows):
        full_mask[i, max_cache:max_cache + i + 1] = 0
    names = ["hidden", "present_k", "present_v"]
    full = _run(build(False), {"x": xs, "pos": pos, "mask": full_mask,
                               "cache_k": base_k, "cache_v": base_v}, names)

    dyn_mask = torch.full((rows, cached + rows), -1e4)
    dyn_mask[:, :cached] = 0
    for i in range(rows):
        dyn_mask[i, cached:cached + i + 1] = 0
    got = run_dynamic(build(True), {"x": xs, "pos": pos, "mask": dyn_mask,
                                    "cache_k": base_k, "cache_v": base_v})
    for name in names:
        _assert_close(got[name], full[name], tolerance, f"{name} (cached={cached})")


def _run_in_place(plan: bytes, feeds: dict, outputs: list[str], alias: dict) -> tuple[dict, dict]:
    """Like `_run`, but each output in `alias` shares memory with the named input (the KV cache
    update writes in place). Returns (outputs, the input tensors after the run)."""
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    tensors, result = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            tensors[name] = feeds[name].contiguous().cuda()
            context.set_tensor_address(name, tensors[name].data_ptr())
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            if name in alias:
                context.set_tensor_address(name, tensors[alias[name]].data_ptr())
            else:
                dtype = {trt.float32: torch.float32, trt.float16: torch.float16}[
                    engine.get_tensor_dtype(name)]
                result[name] = torch.zeros(tuple(context.get_tensor_shape(name)), dtype=dtype,
                                           device="cuda")
                context.set_tensor_address(name, result[name].data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return ({n: result[n].float().cpu() for n in outputs},
            {n: t.float().cpu() for n, t in tensors.items()})


@pytest.mark.gpu
@pytest.mark.parametrize("precision,tolerance", [("fp32", 1e-4), ("fp16", 3e-2)])
@pytest.mark.parametrize("cached", [0, 5, 13])
@pytest.mark.parametrize("rows", [1, 16])
def test_attention_native_kv_cache_matches_concat(precision, tolerance, cached, rows) -> None:
    """KV cache update in place + attention over `kv_len` rows = concat + masked attention over the
    whole cache buffer. `rows=1` is the decode layer, `rows=16` the prefill layer."""
    _need_gpu()
    from families.qwen3_8 import engine_builder, graph_ops, prefill_graph

    max_cache, pad = 24, 16
    rng = np.random.default_rng(3)
    weights = _attention_weights(rng, "L")
    cos = graph_ops.make_rope_table_half_dim(
        max_cache + pad, ATT_D, 10000.0, cosine=True, partial_rotary_factor=ROT / ATT_D)
    sin = graph_ops.make_rope_table_half_dim(
        max_cache + pad, ATT_D, 10000.0, cosine=False, partial_rotary_factor=ROT / ATT_D)

    def common(net, x, ck, cv, pos, mask, kv_update):
        n = net.network
        eps = graph_ops.add_constant(n, (1, 1), np.array([1e-6]), dtype=net.np_dtype)
        return dict(
            network=n, hidden=x, cache_k=ck, cache_v=cv, attention_mask=mask,
            eps_tensor=eps,
            cos_half_tensor=graph_ops.add_constant(n, cos.shape, cos, dtype=net.np_dtype),
            sin_half_tensor=graph_ops.add_constant(n, sin.shape, sin, dtype=net.np_dtype),
            weights=weights, prefix="L", hidden_size=HID, attn_size=ATT_SIZE,
            kv_attention_size=KV_SIZE, num_heads=ATT_HEADS, num_kv_heads=KV_HEADS, head_dim=ATT_D,
            rotary_embedding_dim=ROT, max_cache_length=max_cache, mlp_size=MLP,
            dtype=net.np_dtype, kv_update=kv_update)

    def layer(net, pos, kwargs):
        if rows == 1:
            return engine_builder._add_full_attention_layer(position_id=pos, **kwargs)
        return prefill_graph.add_full_attention_layer_prefill(
            position_ids=pos, chunk_len=rows, **kwargs)

    def build_concat() -> bytes:
        net = _Net(precision)
        n = net.network
        x = n.add_input("x", net.trt_dtype, (rows, HID))
        pos = n.add_input("pos", trt.int32, (rows,))
        mask = n.add_input("mask", trt.float32, (rows, max_cache + rows) if rows > 1
                           else (1, max_cache + 1))
        ck = n.add_input("cache_k", net.trt_dtype, (max_cache, KV_SIZE))
        cv = n.add_input("cache_v", net.trt_dtype, (max_cache, KV_SIZE))
        out = layer(net, pos, common(net, x, ck, cv, pos, net.cast_in(mask), None))
        net.mark(hidden=out["hidden"], present_k=out["present_k"], present_v=out["present_v"])
        return net.build()

    def build_native() -> bytes:
        net = _Net(precision)
        n = net.network
        x = n.add_input("x", net.trt_dtype, (rows, HID))
        pos = n.add_input("pos", trt.int32, (rows,))
        write_idx = n.add_input("write_idx", trt.int32, (1,))
        kv_len = n.add_input("kv_len", trt.int32, (1,))
        shape = (1, KV_HEADS, max_cache + pad, ATT_D)
        ck = n.add_input("cache_k", net.trt_dtype, shape)
        cv = n.add_input("cache_v", net.trt_dtype, shape)
        out = layer(net, pos, common(net, x, ck, cv, pos, None, (write_idx, kv_len)))
        net.mark(hidden=out["hidden"])
        for name, tensor in (("cache_k_out", out["present_k"]), ("cache_v_out", out["present_v"])):
            tensor.name = name
            n.mark_output(tensor)
        return net.build()

    dtype = torch.float32 if precision == "fp32" else torch.float16
    xs = torch.from_numpy(_rnd(rng, rows, HID, scale=1.0)).to(dtype)
    base_k = torch.from_numpy(_rnd(rng, max_cache, KV_SIZE, scale=1.0)).to(dtype)
    base_v = torch.from_numpy(_rnd(rng, max_cache, KV_SIZE, scale=1.0)).to(dtype)
    pos = torch.arange(cached, cached + rows, dtype=torch.int32)

    mask = torch.full((rows, max_cache + rows), -1e4)
    mask[:, :cached] = 0
    for i in range(rows):
        mask[i, max_cache:max_cache + i + 1] = 0
    if rows == 1:
        mask = mask.reshape(1, max_cache + 1)
    names = ["hidden", "present_k", "present_v"]
    ref = _run(build_concat(), {"x": xs, "pos": pos, "mask": mask, "cache_k": base_k,
                                "cache_v": base_v}, names)

    def heads_major(rows_kv):  # [S, Hkv * D] -> [1, Hkv, S, D], padded to max_cache + pad rows
        out = torch.zeros(1, KV_HEADS, max_cache + pad, ATT_D, dtype=dtype)
        out[0, :, :rows_kv.shape[0]] = rows_kv.reshape(-1, KV_HEADS, ATT_D).permute(1, 0, 2)
        return out

    cache_k, cache_v = heads_major(base_k), heads_major(base_v)
    got, after = _run_in_place(
        build_native(),
        {"x": xs, "pos": pos, "write_idx": torch.tensor([cached], dtype=torch.int32),
         "kv_len": torch.tensor([cached + rows], dtype=torch.int32),
         "cache_k": cache_k, "cache_v": cache_v},
        ["hidden"], {"cache_k_out": "cache_k", "cache_v_out": "cache_v"})
    _assert_close(got["hidden"], ref["hidden"], tolerance, f"hidden (cached={cached})")
    for name, key in (("present_k", "cache_k"), ("present_v", "cache_v")):
        new_rows = after[key][0, :, cached:cached + rows].permute(1, 0, 2).reshape(rows, KV_SIZE)
        _assert_close(new_rows, ref[name], tolerance, f"{name} written (cached={cached})")
    # rows outside the written range are untouched
    for key, base in (("cache_k", base_k), ("cache_v", base_v)):
        expected = heads_major(base).float()
        keep = torch.ones(max_cache + pad, dtype=torch.bool)
        keep[cached:cached + rows] = False
        assert torch.equal(after[key][0][:, keep], expected[0][:, keep]), f"{key} rows outside the write changed"
