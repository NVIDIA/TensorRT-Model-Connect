# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 prefill engine: chunked prefill against the one-token engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from .. import deltanet_rows, model  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from .test_quantization import _write_checkpoint  # noqa: E402

MAX_CACHE = 32
MASKED = -1.0e4


class _Stepper:
    """Runs one engine context with host-managed KV caches and recurrent state."""

    def __init__(self, engine, profile: int, state: dict) -> None:
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        if engine.num_optimization_profiles > 1:
            assert self.context.set_optimization_profile_async(profile, self.stream)
        self.state = state

    def run(self, tokens: list[int], position: int) -> np.ndarray:
        rows = len(tokens)
        width = MAX_CACHE + rows
        mask = torch.full((rows, width), MASKED)
        mask[:, :position] = 0.0
        for row in range(rows):
            mask[row, MAX_CACHE : MAX_CACHE + row + 1] = 0.0
        inputs = {
            "token_id": torch.tensor(tokens, dtype=torch.int32),
            "position_id": torch.arange(position, position + rows, dtype=torch.int32),
            "attention_mask": mask,
        }
        bound = {}
        for name, value in inputs.items():
            value = value.cuda()
            bound[name] = value
            if -1 in tuple(self.engine.get_tensor_shape(name)):
                assert self.context.set_input_shape(name, tuple(value.shape))
            self.context.set_tensor_address(name, value.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name in self.state:
                    self.context.set_tensor_address(name, self.state[name].data_ptr())
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = (
                torch.float16
                if self.engine.get_tensor_dtype(name) == trt.float16
                else torch.float32
            )
            outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
            self.context.set_tensor_address(name, outputs[name].data_ptr())
        assert self.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()
        for name, value in outputs.items():
            if name.startswith(("present_conv", "present_ssm")):
                target = self.state[
                    name.replace("present_conv", "conv_state").replace("present_ssm", "ssm_state")
                ]
                target.copy_(value.reshape(target.shape))
            elif name.startswith("present_"):
                cache = self.state[name.replace("present_", "cache_")]
                cache[position : position + rows] = value
        return outputs["logits"][-1].float().cpu().numpy()


def _state(engine) -> dict:
    state = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if name.startswith(("cache_", "conv_state", "ssm_state")):
            dtype = torch.float16 if engine.get_tensor_dtype(name) == trt.float16 else torch.float32
            state[name] = torch.zeros(
                tuple(engine.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
    return state


def _engine(plan: bytes):
    return trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)


@pytest.mark.gpu
@pytest.mark.trt
def test_chunked_prefill_matches_one_token_engine(tmp_path: Path) -> None:
    """Prefill chunks leave the logits, KV rows and recurrent state of token-by-token steps."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _write_checkpoint(tmp_path, quantized=False)
    config = ModelConfig.from_dir(tmp_path)
    instance = model._Qwen35Model()
    weights = instance.load_weights(str(tmp_path), config)
    single = _engine(instance.build_engine(config, weights, MAX_CACHE, precision="fp32"))
    prefill_engine = _engine(
        instance.build_engine(
            config, weights, MAX_CACHE, precision="fp32", prefill_rows=model.PREFILL_MAX_TOKENS
        )
    )
    assert prefill_engine.num_optimization_profiles == 1
    assert [tuple(s) for s in prefill_engine.get_tensor_profile_shape("token_id", 0)] == [
        (1,),
        (min(model.PREFILL_OPT_TOKENS, MAX_CACHE),),
        (MAX_CACHE,),
    ]

    prompt, next_token = [3, 17, 5, 9, 30, 2, 11, 21, 7, 14, 26], 8
    want_state = _state(single)
    stepper = _Stepper(single, 0, want_state)
    want = [stepper.run([token], position) for position, token in enumerate(prompt)]
    want.append(stepper.run([next_token], len(prompt)))

    # Prefill chunks on the prefill engine, then decode on the one-token engine.
    got_state = _state(single)
    prefill = _Stepper(prefill_engine, 0, got_state)
    decode = _Stepper(single, 0, got_state)
    got = [
        prefill.run(prompt[:7], 0),
        prefill.run(prompt[7:], 7),
        decode.run([next_token], len(prompt)),
    ]

    # FP32 engines still run matmuls in TF32, and one row and several rows take
    # different kernels, so compare at a fraction of each tensor's scale.
    errors = {}
    for step, (logits, reference) in enumerate(zip(got, (want[6], want[10], want[11]))):
        assert np.argmax(logits) == np.argmax(reference), step
        errors[f"logits[{step}]"] = np.abs(logits - reference).max() / np.abs(reference).max()
    for name, reference in want_state.items():
        reference = reference.float().cpu().numpy()
        value = got_state[name].float().cpu().numpy()
        if name.startswith("cache_"):
            reference, value = reference[: len(prompt) + 1], value[: len(prompt) + 1]
        errors[name] = np.abs(value - reference).max() / max(np.abs(reference).max(), 1e-6)
    assert max(errors.values()) < 5e-3, errors


@pytest.mark.trt
def test_prefill_engine_rejects_media_and_speculative_builds(tmp_path: Path) -> None:
    """Media, MTP hidden-state and draft-feature engines keep one row per launch."""
    _write_checkpoint(tmp_path, quantized=False)
    config = ModelConfig.from_dir(tmp_path)
    instance = model._Qwen35Model()
    weights = instance.load_weights(str(tmp_path), config)
    for options in (
        {"media_inputs": True},
        {"export_hidden_state": True},
        {"feature_layers": (0,)},
    ):
        with pytest.raises(NotImplementedError, match="prefill engines"):
            instance.build_engine(config, weights, MAX_CACHE, prefill_rows=16, **options)
    with pytest.raises(ValueError, match="prefill_rows"):
        instance.build_engine(config, weights, MAX_CACHE, prefill_rows=model.PREFILL_MAX_TOKENS + 1)


@pytest.mark.gpu
@pytest.mark.trt
def test_triangular_solve_matches_numpy_on_nearly_parallel_keys() -> None:
    """The batched block inverse stays exact where a Neumann series over 64 rows breaks down."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    heads, size = 2, deltanet_rows.SOLVE_ROWS
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    lower = network.add_input("lower", trt.float32, (heads, size, size))
    inverse = deltanet_rows._inverse_unit_lower(network, lower, heads, size)
    inverse.name = "inverse"
    network.mark_output(inverse)
    config = builder.create_builder_config()
    # Check the algorithm in full FP32; engines may run these matmuls in TF32.
    config.clear_flag(trt.BuilderFlag.TF32)
    engine = trt.Runtime(logger).deserialize_cuda_engine(
        builder.build_serialized_network(network, config)
    )
    context = engine.create_execution_context()
    rng = np.random.default_rng(0)
    # beta ~ 1 and nearly identical keys: strictly lower entries close to one.
    strict = np.tril(rng.uniform(0.9, 1.0, (heads, size, size)), -1)
    matrix = (np.eye(size) + strict).astype(np.float32)
    source = torch.from_numpy(matrix).cuda()
    result = torch.empty_like(source)
    context.set_tensor_address("lower", source.data_ptr())
    context.set_tensor_address("inverse", result.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    want = np.linalg.inv(matrix.astype(np.float64))
    got = result.cpu().numpy()
    assert np.abs(got - want).max() <= 1e-4 * np.abs(want).max()


def test_the_prefill_engine_is_built_only_when_a_second_weight_copy_fits() -> None:
    gib = 1024**3
    assert model._two_engines_fit(18 * gib, 121 * gib)  # 9B FP16 on a 128 GB unified-memory GPU
    assert not model._two_engines_fit(54 * gib, 121 * gib)  # 27B FP16: the decode engine alone
    assert model._two_engines_fit(54 * gib, 275 * gib)
    assert model._two_engines_fit(54 * gib, None)  # no device to ask: keep the prefill engine
    weights = {"w": np.zeros((4, 8), dtype=np.float16), "_rows": 4}
    assert model._host_weight_bytes(weights) == 64
