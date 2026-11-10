# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prefill and decode engines attend over the bound cache prefix, not the whole allocation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.phi4_multimodal import default_decoder  # noqa: E402
from families.phi4_multimodal import model as family_model  # noqa: E402
from families.phi4_multimodal.config import ModelConfig  # noqa: E402

from .test_family_contract import _rand, _write_config, _write_safetensors  # noqa: E402

MAX_CACHE = 64
MASKED = -1.0e4
VOCAB, HIDDEN, HEADS, KV_HEADS, MLP = 64, 32, 4, 2, 64


def _write_checkpoint(path: Path) -> ModelConfig:
    head_dim = HIDDEN // HEADS
    _write_config(
        path,
        {
            "model_type": "phi4mm",
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "num_hidden_layers": 2,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "intermediate_size": MLP,
            "max_position_embeddings": 4 * MAX_CACHE,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
        },
    )
    tensors = {"model.embed_tokens.weight": _rand(VOCAB, HIDDEN) * 0.1}
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        qkv = HEADS * head_dim + 2 * KV_HEADS * head_dim
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": np.ones(HIDDEN, np.float32),
                f"{prefix}.post_attention_layernorm.weight": np.ones(HIDDEN, np.float32),
                f"{prefix}.self_attn.qkv_proj.base_layer.weight": _rand(qkv, HIDDEN) * 0.1,
                f"{prefix}.self_attn.o_proj.base_layer.weight": _rand(HIDDEN, HIDDEN) * 0.1,
                f"{prefix}.mlp.gate_up_proj.base_layer.weight": _rand(2 * MLP, HIDDEN) * 0.1,
                f"{prefix}.mlp.down_proj.base_layer.weight": _rand(HIDDEN, MLP) * 0.1,
            }
        )
    tensors["model.norm.weight"] = np.ones(HIDDEN, np.float32)
    tensors["lm_head.weight"] = _rand(VOCAB, HIDDEN) * 0.1
    _write_safetensors(path, tensors)
    return ModelConfig.from_dir(path)


def _run(engine, cache_rows: int, caches: dict, tokens: list[int], position: int) -> dict:
    """One prefill launch binding the first ``cache_rows`` rows of each cache."""
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream().cuda_stream
    rows = len(tokens)
    mask = torch.full((rows, cache_rows + rows), MASKED)
    mask[:, :position] = 0.0
    for row in range(rows):
        mask[row, cache_rows : cache_rows + row + 1] = 0.0
    inputs = {
        "token_id": torch.tensor(tokens, dtype=torch.int32),
        "position_id": torch.arange(position, position + rows, dtype=torch.int32),
        "attention_mask": mask,
        "input_embed": torch.zeros(rows, HIDDEN),
        "use_input_embed": torch.zeros(rows, 1),
    }
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    bound = {}
    for name in names:
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        if name in caches:
            value = caches[name][:cache_rows]
        else:
            value = inputs[name].cuda()
            if -1 not in tuple(engine.get_tensor_shape(name)):  # the decode engine's static inputs
                value = value.reshape(tuple(engine.get_tensor_shape(name)))
        bound[name] = value
        if -1 in tuple(engine.get_tensor_shape(name)):
            assert context.set_input_shape(name, tuple(value.shape)), name
        context.set_tensor_address(name, value.data_ptr())
    outputs = {}
    for name in names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            outputs[name] = torch.empty(
                tuple(context.get_tensor_shape(name)), dtype=torch.float32, device="cuda"
            )
            context.set_tensor_address(name, outputs[name].data_ptr())
    assert context.execute_async_v3(stream)
    torch.cuda.synchronize()
    return {name: value.cpu().numpy() for name, value in outputs.items()}


def _engine(tmp_path: Path, role: str):
    config = _write_checkpoint(tmp_path)
    config.raw["_model_dir"] = str(tmp_path)
    config.raw["_decoder_engine_role"] = role
    instance = family_model._Phi4MultimodalModel()
    weights = instance.load_weights(str(tmp_path), config)
    return trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
        instance.build_engine(config, weights, MAX_CACHE, precision="fp32")
    )


def _random_caches(engine) -> dict:
    caches = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if name.startswith("cache_"):
            width = engine.get_tensor_shape(name)[1]
            # Rows beyond the prefix hold garbage that the mask must hide.
            caches[name] = torch.randn(MAX_CACHE, width, device="cuda")
    return caches


@pytest.mark.gpu
@pytest.mark.trt
def test_prefill_over_the_cached_prefix_matches_the_whole_cache(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    engine = _engine(tmp_path, "prefill")
    assert tuple(engine.get_tensor_shape("cache_k_0"))[0] == -1
    assert [tuple(s) for s in engine.get_tensor_profile_shape("cache_k_0", 0)] == [
        (1, engine.get_tensor_shape("cache_k_0")[1]),
        (1, engine.get_tensor_shape("cache_k_0")[1]),
        (MAX_CACHE, engine.get_tensor_shape("cache_k_0")[1]),
    ]

    caches = _random_caches(engine)

    rng = np.random.default_rng(0)
    for position, rows in ((0, 1), (5, 16)):
        tokens = rng.integers(0, VOCAB, 7).tolist()
        whole = _run(engine, MAX_CACHE, caches, tokens, position)
        prefix = _run(engine, rows, caches, tokens, position)
        for name, value in whole.items():
            np.testing.assert_allclose(prefix[name], value, rtol=1e-4, atol=1e-4, err_msg=name)


@pytest.mark.gpu
@pytest.mark.trt
def test_decode_over_the_cached_prefix_matches_the_whole_cache(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    engine = _engine(tmp_path, "decode")
    width = engine.get_tensor_shape("cache_k_0")[1]
    opt_rows = min(MAX_CACHE, default_decoder.DECODE_OPT_CACHE_ROWS)
    assert tuple(engine.get_tensor_shape("cache_k_0"))[0] == -1
    assert [tuple(s) for s in engine.get_tensor_profile_shape("cache_k_0", 0)] == [
        (1, width),
        (opt_rows, width),
        (MAX_CACHE, width),
    ]
    caches = _random_caches(engine)

    rng = np.random.default_rng(1)
    for position, rows in ((0, 1), (5, 16), (40, 48), (MAX_CACHE - 1, MAX_CACHE)):
        token = rng.integers(0, VOCAB, 1).tolist()
        whole = _run(engine, MAX_CACHE, caches, token, position)
        prefix = _run(engine, rows, caches, token, position)
        for name, value in whole.items():
            np.testing.assert_allclose(prefix[name], value, rtol=1e-4, atol=1e-4, err_msg=name)
