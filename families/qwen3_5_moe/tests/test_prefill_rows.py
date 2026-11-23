# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-MoE prefill rows against the one-token graphs: dense W4A16 experts, per-row MRoPE, the split rule."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from families.qwen3_5_moe import moe, native_model, vision  # noqa: E402
from families.qwen3_5_moe.quantization import gptq_projection  # noqa: E402

HIDDEN, WIDTH, EXPERTS, TOP_K, ROWS = 256, 128, 8, 2, 6


def _gptq(rng, k: int, n: int):
    qweight = rng.integers(0, 2**32, size=(k // 8, n), dtype=np.uint64).astype(np.uint32).view(np.int32)
    qzeros = np.full((k // 128, n // 8), 0x88888888, np.uint32).view(np.int32)
    scales = (rng.uniform(0.5, 1.5, (k // 128, n)) / 64).astype(np.float16)
    return gptq_projection(qweight, qzeros, scales, (np.arange(k) // 128).astype(np.int32))


def _experts(rng) -> moe.MoEWeights:
    experts = tuple(
        moe.ExpertWeights(_gptq(rng, HIDDEN, WIDTH), _gptq(rng, HIDDEN, WIDTH), _gptq(rng, WIDTH, HIDDEN))
        for _ in range(EXPERTS)
    )
    shared = moe.ExpertWeights(*(
        (rng.standard_normal(shape) * 0.05).astype(np.float32)
        for shape in ((HIDDEN, WIDTH), (HIDDEN, WIDTH), (WIDTH, HIDDEN))
    ))
    return moe.MoEWeights(
        router=rng.standard_normal((HIDDEN, EXPERTS)).astype(np.float32),
        experts=experts,
        shared=shared,
        shared_gate=(rng.standard_normal((HIDDEN, 1)) * 0.05).astype(np.float32),
        top_k=TOP_K,
    )


def _run(build, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Build ``build(network, tensors) -> output`` over runtime-row inputs and run it once."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    tensors = {}
    for name, value in inputs.items():
        dtype = {np.float16: trt.float16, np.float32: trt.float32, np.int32: trt.int32}[value.dtype.type]
        tensors[name] = network.add_input(name, dtype, (-1, *value.shape[1:]))
        profile.set_shape(name, (1, *value.shape[1:]), value.shape, value.shape)
    config.add_optimization_profile(profile)
    output = build(network, tensors)
    if output.dtype != trt.float32:
        output = network.add_cast(output, trt.float32).get_output(0)
    output.name = "output"
    network.mark_output(output)
    engine = trt.Runtime(logger).deserialize_cuda_engine(builder.build_serialized_network(network, config))
    context = engine.create_execution_context()
    bound = {}
    for name, value in inputs.items():
        bound[name] = torch.from_numpy(value).cuda()
        context.set_input_shape(name, value.shape)
        context.set_tensor_address(name, bound[name].data_ptr())
    result = torch.empty(tuple(context.get_tensor_shape("output")), dtype=torch.float32, device="cuda")
    context.set_tensor_address("output", result.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return result.cpu().numpy()


@pytest.mark.gpu
@pytest.mark.trt
def test_dense_w4a16_experts_match_the_per_token_selected_experts():
    """Every row runs every expert, and only the routed ones contribute, as on the one-token path."""
    rng = np.random.default_rng(0)
    weights = _experts(rng)
    rows = (rng.standard_normal((ROWS, HIDDEN)) * 0.5).astype(np.float16)
    keep: list = []
    dense = _run(lambda network, x: moe.add_dense_moe_rows(network, x["rows"], weights, keep), {"rows": rows})
    selected = _run(lambda network, x: moe.add_sparse_moe(network, x["rows"], weights), {"rows": rows})
    assert keep, "the stacked experts must outlive the network"
    assert np.abs(dense - selected).max() <= 1e-2 * np.abs(selected).max()


def test_only_gptq_experts_take_the_dense_prefill_path():
    rng = np.random.default_rng(1)
    weights = _experts(rng)
    assert moe.dense_rows_supported(weights)
    assert native_model._dense_prefill_experts({"layer.0.moe": weights, "embedding": np.zeros(2)})
    original = moe.ExpertWeights(*(np.zeros(shape, np.float32) for shape in ((HIDDEN, WIDTH),) * 2 + ((WIDTH, HIDDEN),)))
    mixed = moe.MoEWeights(weights.router, (original,) + weights.experts[1:], weights.shared, weights.shared_gate, TOP_K)
    assert not moe.dense_rows_supported(mixed)
    assert not native_model._dense_prefill_experts({"layer.0.moe": weights, "layer.1.moe": mixed})
    assert not native_model._dense_prefill_experts({"embedding": np.zeros(2)})


@pytest.mark.gpu
@pytest.mark.trt
def test_per_row_mrope_tables_equal_the_one_token_tables():
    rng = np.random.default_rng(2)
    half, window = 8, 40
    rope = {"mrope_section": [2, 3, 3], "mrope_interleaved": True}
    cosine = rng.standard_normal((window, half)).astype(np.float32)
    sine = rng.standard_normal((window, half)).astype(np.float32)
    coordinates = rng.integers(0, window, size=(ROWS, 3)).astype(np.int32)

    def constants(network):
        return (network.add_constant(cosine.shape, cosine).get_output(0),
                network.add_constant(sine.shape, sine).get_output(0))

    def rows(network, x):
        cos, sin = vision.add_mrope_prompt_rows(network, *constants(network), x["coordinates"], rope, half,
                                                np.float32)
        return network.add_concatenation([cos, sin]).get_output(0)

    got = _run(rows, {"coordinates": coordinates})
    for row in range(ROWS):

        def one(network, x):
            coordinate = network.add_shuffle(x["coordinate"])
            coordinate.reshape_dims = (3,)
            cos, sin = vision.add_mrope_rows(network, *constants(network), coordinate.get_output(0), rope, half,
                                             np.float32)
            return network.add_concatenation([cos, sin]).get_output(0)

        want = _run(one, {"coordinate": coordinates[row:row + 1]})
        np.testing.assert_array_equal(got[[row, ROWS + row]], want)


def test_the_prefill_engine_is_built_only_when_a_second_weight_copy_fits():
    gib = 1024**3
    assert native_model._two_engines_fit(20 * gib, 121 * gib)  # 35B-A3B GPTQ on a 128 GB unified-memory GPU
    assert not native_model._two_engines_fit(67 * gib, 121 * gib)  # the BF16 original: decode engine alone
    assert not native_model._two_engines_fit(2 * gib, None)  # no device to ask: keep the single-engine layout


@pytest.mark.parametrize(
    ("gpu_gib", "options", "wanted"),
    [
        (121, {}, True),
        (30, {}, False),
        (None, {}, False),
        (121, {"require_mtp": True}, False),
        (121, {"has_draft": True}, False),
        (121, {"tensor_parallel_size": 2}, False),
    ],
)
def test_the_split_decision_uses_the_mapped_shards_and_the_gpu_memory(monkeypatch, gpu_gib, options, wanted):
    gib = 1024**3
    readers = [SimpleNamespace(mapping=_Sized(12 * gib)), SimpleNamespace(mapping=_Sized(8 * gib))]
    weights = {"layer.0.moe": _experts(np.random.default_rng(3))}
    monkeypatch.setattr(native_model, "_gpu_memory_bytes", lambda: None if gpu_gib is None else gpu_gib * gib)
    assert native_model._checkpoint_bytes(readers) == 20 * gib
    arguments = {"tensor_parallel_size": 1, "require_mtp": False, "has_draft": False, **options}
    assert native_model._wants_prefill_engine(weights, readers, **arguments) is wanted


class _Sized:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class _RecordingWriter:
    def __init__(self):
        self.sections: dict[str, object] = {}

    def add_bytes(self, name: str, data) -> None:
        self.sections[name] = data


def test_a_failed_prefill_build_keeps_the_single_engine_bundle():
    writer = _RecordingWriter()
    assert native_model._stage_prefill_engine(writer, lambda: b"plan")
    assert writer.sections == {"prefill.plan": b"plan"}

    def failed():
        raise RuntimeError("TensorRT engine build failed")

    writer = _RecordingWriter()
    assert not native_model._stage_prefill_engine(writer, failed)
    assert writer.sections == {}
