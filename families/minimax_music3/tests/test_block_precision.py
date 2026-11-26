# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DiT and depth decoder blocks follow the build precision (TensorRT networks).

Under ``precision="bf16"`` the blocks run in bf16; under ``"fp32"`` the graphs are
float32 as before. Inputs, outputs, stems and heads are float32 in both.
"""

from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")

dit = importlib.import_module("families.minimax_music3.dit")
dib = importlib.import_module("families.minimax_music3.dit_builder")
dd = importlib.import_module("families.minimax_music3.depth_decoder")
ddb = importlib.import_module("families.minimax_music3.depth_decoder_builder")
model = importlib.import_module("families.minimax_music3.model")
engines = importlib.import_module("families.minimax_music3.engines")

LATENT_LENGTH = 16


def _gpu_trt():
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("a TensorRT network needs a GPU")
    return trt


def _network(trt):
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    return builder, builder.create_network(), builder.create_builder_config()


def _layers(network):
    return [network.get_layer(i) for i in range(network.num_layers)]


def _dit_weights(layers: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    def tensor(*shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    w = {
        "preprocess_conv.weight": tensor(dit.CONCAT_CHANNELS, dit.CONCAT_CHANNELS, 1),
        "postprocess_conv.weight": tensor(dit.IN_CHANNELS, dit.IN_CHANNELS, 1),
        "proj_in.weight": tensor(dit.INNER_DIM, dit.CONCAT_CHANNELS),
        "proj_out.weight": tensor(dit.IN_CHANNELS, dit.INNER_DIM),
        "time_proj.weight": tensor(dib.FOURIER_HALF_DIM, 1, scale=1.0),
        "time_embed.linear_1.weight": tensor(dit.INNER_DIM, 2 * dib.FOURIER_HALF_DIM),
        "time_embed.linear_1.bias": tensor(dit.INNER_DIM),
        "time_embed.linear_2.weight": tensor(dit.INNER_DIM, dit.INNER_DIM),
        "time_embed.linear_2.bias": tensor(dit.INNER_DIM),
    }
    for layer in range(layers):
        p = f"transformer_blocks.{layer}"
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            w[f"{p}.attn.{name}.weight"] = tensor(dit.INNER_DIM, dit.INNER_DIM)
        for norm in ("norm1", "norm2"):
            w[f"{p}.{norm}.weight"] = 1.0 + tensor(dit.INNER_DIM)
            w[f"{p}.{norm}.bias"] = tensor(dit.INNER_DIM)
        w[f"{p}.ff_in.weight"] = tensor(2 * dit.FF_INNER_DIM, dit.INNER_DIM)
        w[f"{p}.ff_in.bias"] = tensor(2 * dit.FF_INNER_DIM)
        w[f"{p}.ff_out.weight"] = tensor(dit.INNER_DIM, dit.FF_INNER_DIM)
        w[f"{p}.ff_out.bias"] = tensor(dit.INNER_DIM)
    return w


def _dit_network(trt, weights, dtype, layers=1):
    builder, network, config = _network(trt)
    latents = network.add_input(dib.LATENTS_NAME, trt.float32, (1, dit.IN_CHANNELS, LATENT_LENGTH))
    condition = network.add_input(
        dib.CONDITION_NAME, trt.float32, (1, LATENT_LENGTH, dit.CONDITION_DIM)
    )
    timestep = network.add_input(dib.TIMESTEP_SCALAR_NAME, trt.float32, (1, 1, 1))
    embedded = dib.add_timestep_embedding(network, trt, timestep, weights=weights)
    out = dib.add_dit(
        network,
        trt,
        latents,
        condition,
        embedded,
        latent_length=LATENT_LENGTH,
        weights=weights,
        num_layers=layers,
        dtype=dtype,
    )
    out.name = dib.OUTPUT_NAME
    network.mark_output(out)
    return builder, network, config, out


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_dit_block_matmuls_run_in_the_build_precision(precision: str) -> None:
    trt = _gpu_trt()
    dtype = trt.bfloat16 if precision == "bf16" else None
    _, network, _, out = _dit_network(trt, _dit_weights(1), dtype)
    matmuls = [layer for layer in _layers(network) if layer.type == trt.LayerType.MATRIX_MULTIPLY]
    kinds = [layer.get_output(0).dtype for layer in matmuls]

    # Two timestep-embedding linears and proj_in, six projections and two attention
    # products in the block, then proj_out.
    assert len(matmuls) == 3 + 8 + 1
    block = trt.bfloat16 if precision == "bf16" else trt.float32
    assert kinds[:3] == [trt.float32] * 3
    assert kinds[3:-1] == [block] * 8
    assert kinds[-1] == trt.float32
    assert out.dtype == trt.float32
    convolutions = [layer for layer in _layers(network) if layer.type == trt.LayerType.CONVOLUTION]
    assert [layer.get_output(0).dtype for layer in convolutions] == [trt.float32] * 2


@pytest.mark.gpu
def test_dit_bf16_block_matches_float32() -> None:
    trt = _gpu_trt()
    import torch

    weights = _dit_weights(1)
    rng = np.random.default_rng(1)
    feeds = {
        dib.LATENTS_NAME: rng.standard_normal((1, dit.IN_CHANNELS, LATENT_LENGTH)).astype(
            np.float32
        ),
        dib.CONDITION_NAME: (
            0.5 * rng.standard_normal((1, LATENT_LENGTH, dit.CONDITION_DIM))
        ).astype(np.float32),
        dib.TIMESTEP_SCALAR_NAME: np.full((1, 1, 1), 0.4, dtype=np.float32),
    }
    outputs = {}
    for name, dtype in (("fp32", None), ("bf16", trt.bfloat16)):
        builder, network, config, _ = _dit_network(trt, weights, dtype)
        dib.configure(config, trt)
        plan = builder.build_serialized_network(network, config)
        assert plan is not None
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        buffers = {key: torch.from_numpy(value).cuda() for key, value in feeds.items()}
        buffers[dib.OUTPUT_NAME] = torch.empty((1, dit.IN_CHANNELS, LATENT_LENGTH), device="cuda")
        for key, value in buffers.items():
            context.set_tensor_address(key, value.data_ptr())
        stream = torch.cuda.Stream()
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        outputs[name] = buffers[dib.OUTPUT_NAME].double().cpu()

    error = float((outputs["bf16"] - outputs["fp32"]).norm() / outputs["fp32"].norm())
    assert torch.isfinite(outputs["bf16"]).all()
    # bf16 rounding is visible (the block really runs in bf16) but small.
    assert 1e-5 < error < 2e-2


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_depth_decoder_matmuls_run_in_the_build_precision(precision: str) -> None:
    trt = _gpu_trt()
    dtype = trt.bfloat16 if precision == "bf16" else None
    weights = {
        "pos_embedding.weight": np.zeros(
            (dd.MAX_POSITION_EMBEDDINGS, dd.HIDDEN_SIZE), dtype=np.float32
        ),
        "norm.weight": np.ones(dd.HIDDEN_SIZE, dtype=np.float32),
    }
    for layer in range(dd.NUM_LAYERS):
        p = f"layers.{layer}"
        for name in ("to_q", "to_k", "to_v", "to_out"):
            weights[f"{p}.attn.{name}.weight"] = np.zeros(
                (dd.HIDDEN_SIZE, dd.HIDDEN_SIZE), dtype=np.float32
            )
        for name in ("gate_proj", "up_proj"):
            weights[f"{p}.{name}.weight"] = np.zeros(
                (dd.INTERMEDIATE_SIZE, dd.HIDDEN_SIZE), dtype=np.float32
            )
        weights[f"{p}.down_proj.weight"] = np.zeros(
            (dd.HIDDEN_SIZE, dd.INTERMEDIATE_SIZE), dtype=np.float32
        )
        for norm in ("input_layernorm", "post_attention_layernorm"):
            weights[f"{p}.{norm}.weight"] = np.ones(dd.HIDDEN_SIZE, dtype=np.float32)
    _, network, _ = _network(trt)
    source = network.add_input(ddb.INPUT_NAME, trt.float32, (1, 8, dd.HIDDEN_SIZE))
    out = ddb.add_depth_decoder(network, trt, source, steps=8, weights=weights, dtype=dtype)
    matmuls = [layer for layer in _layers(network) if layer.type == trt.LayerType.MATRIX_MULTIPLY]

    block = trt.bfloat16 if precision == "bf16" else trt.float32
    assert len(matmuls) == 9 * dd.NUM_LAYERS
    assert {layer.get_output(0).dtype for layer in matmuls} == {block}
    assert out.dtype == trt.float32


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_build_one_passes_the_block_precision(monkeypatch, precision: str) -> None:
    """``_build_one`` gives the DiT and the depth decoder bf16 blocks only under bf16."""

    trt = _gpu_trt()
    seen = {}

    def fake_dit(
        network,
        trt_,
        latents,
        condition,
        embedded,
        *,
        latent_length,
        weights,
        dtype=None,
        quant=None,
    ):
        seen["dit"] = dtype
        assert quant is None
        # Every input reaches the output, so the stand-in network builds.
        pooled = network.add_reduce(condition, trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
        turned = network.add_shuffle(pooled)
        turned.first_transpose = (0, 2, 1)
        total = network.add_elementwise(latents, turned.get_output(0), trt.ElementWiseOperation.SUM)
        return network.add_elementwise(
            total.get_output(0), embedded, trt.ElementWiseOperation.SUM
        ).get_output(0)

    def fake_decoder(network, trt_, inputs_embeds, *, steps, weights, dtype=None, quant=None):
        seen["depth"] = dtype
        assert quant is None
        return network.add_identity(inputs_embeds).get_output(0)

    depth_engine = importlib.import_module("families.minimax_music3.depth_decoder_engine")
    monkeypatch.setattr(dib, "add_dit", fake_dit)
    monkeypatch.setattr(
        dib, "add_timestep_embedding", lambda network, trt_, timestep, *, weights: timestep
    )
    monkeypatch.setattr(ddb, "add_depth_decoder", fake_decoder)
    monkeypatch.setattr(
        depth_engine,
        "add_input_sequence",
        lambda network, trt_, hidden, codes, **_: network.add_identity(hidden).get_output(0),
    )
    monkeypatch.setattr(
        depth_engine,
        "add_output_heads",
        lambda network, trt_, hidden, **_: network.add_identity(hidden).get_output(0),
    )
    monkeypatch.setattr(
        depth_engine,
        "add_hidden_output",
        lambda network, trt_, hidden: network.add_identity(hidden).get_output(0),
    )
    monkeypatch.setattr(
        depth_engine,
        "add_frame_embedding",
        lambda network, trt_, codes, **_: network.add_identity(codes).get_output(0),
    )
    weights = {
        engines.DIT_ENGINE: {},
        engines.DEPTH_DECODER_ENGINE: {},
        engines.LANGUAGE_MODEL_ENGINE: {"model.embed_tokens.weight": None},
    }
    model._build_one(engines.DIT_ENGINE, weights, latent_length=4, precision=precision)
    model._build_one(engines.DEPTH_DECODER_ENGINE, weights, precision=precision)

    expected = trt.bfloat16 if precision == "bf16" else None
    assert seen == {"dit": expected, "depth": expected}
