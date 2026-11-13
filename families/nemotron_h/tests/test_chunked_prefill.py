# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H prefill engine: chunked prefill against the one-token engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from .. import model  # noqa: E402
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
    _write_checkpoint(tmp_path, fmt=None)
    config = ModelConfig.from_dir(tmp_path)
    instance = model._NemotronHModel()

    def build(**options):
        weights = instance.load_weights(str(tmp_path), config)
        return _engine(
            instance.build_engine(config, weights, MAX_CACHE, precision="fp32", **options)
        )

    single = build()
    prefill_engine = build(prefill_rows=model.PREFILL_MAX_TOKENS)
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
def test_prefill_engine_rejects_paired_draft_features(tmp_path: Path) -> None:
    """Engines that export paired-draft features keep one row per launch."""
    _write_checkpoint(tmp_path, fmt=None)
    config = ModelConfig.from_dir(tmp_path)
    instance = model._NemotronHModel()
    weights = instance.load_weights(str(tmp_path), config)
    with pytest.raises(NotImplementedError, match="prefill engines"):
        instance.build_engine(
            config, weights, MAX_CACHE, prefill_rows=16, target_feature_layers=(0,)
        )
    with pytest.raises(ValueError, match="prefill_rows"):
        instance.build_engine(config, weights, MAX_CACHE, prefill_rows=model.PREFILL_MAX_TOKENS + 1)


@pytest.mark.gpu
@pytest.mark.trt
def test_chunked_ssd_state_stays_finite_at_large_decays() -> None:
    """At |dt A| in the millions the chunk state matches the recurrence instead of overflowing."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from .. import ssd_rows

    heads, rows, head_dim, state_dim = 2, ssd_rows.MAX_ROWS, 4, 3
    rng = np.random.default_rng(3)
    x = rng.standard_normal((heads, rows, head_dim)).astype(np.float32)
    b = rng.standard_normal((heads, rows, state_dim)).astype(np.float32)
    c = rng.standard_normal((heads, rows, state_dim)).astype(np.float32)
    dt = rng.uniform(1.0, 30.0, (heads, rows)).astype(np.float32)
    a = np.array([-0.5, -5.0e4], np.float32)  # the second head decays by ~1e6 per row
    d = np.ones(heads, np.float32)
    state = rng.standard_normal((heads, head_dim, state_dim)).astype(np.float32)

    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    inputs = {
        name: network.add_input(name, trt.float32, value.shape)
        for name, value in (("x", x), ("b", b), ("c", c), ("dt", dt), ("state", state))
    }
    y, final = ssd_rows.add_chunked_ssd(
        network, x=inputs["x"], b=inputs["b"], c=inputs["c"], dt=inputs["dt"], a=a, d=d,
        state=inputs["state"], num_heads=heads,
    )
    # gamma_S is read from the cumulative sum: a separate reduction sums in another order
    # and can exceed gamma_i, which overflows exp(gamma_S - gamma_i) on some tactics.
    layers = [network.get_layer(i).type for i in range(network.num_layers)]
    assert trt.LayerType.REDUCE not in layers
    y.name, final.name = "y", "final"
    network.mark_output(y)
    network.mark_output(final)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
        builder.build_serialized_network(network, config)
    )
    context = engine.create_execution_context()
    device = {name: torch.from_numpy(value).cuda() for name, value in
              (("x", x), ("b", b), ("c", c), ("dt", dt), ("state", state))}
    out = {
        name: torch.empty(tuple(engine.get_tensor_shape(name)), device="cuda")
        for name in ("y", "final")
    }
    for name, value in {**device, **out}.items():
        context.set_tensor_address(name, value.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    want = state.astype(np.float64)
    for t in range(rows):  # the one-token recurrence
        decay = np.exp(dt[:, t] * a)[:, None, None]
        want = decay * want + (dt[:, t, None, None] * x[:, t, :, None]) * b[:, t, None, :]
    got = out["final"].cpu().numpy()
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
