# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-ASR split decoder: one prefill-engine launch with audio rows against token-by-token steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.numpy")

from families.glmasr import model  # noqa: E402

from .test_quantization import D, _write_checkpoint  # noqa: E402

MAX_CACHE = 32
MASKED = -1.0e4


class _Stepper:
    """One engine context with host-managed KV caches."""

    def __init__(self, engine, profile: int, caches: dict) -> None:
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        if engine.num_optimization_profiles > 1:
            assert self.context.set_optimization_profile_async(profile, self.stream)
        self.caches = caches

    def run(self, tokens, embeds, flags, position: int) -> np.ndarray:
        rows = len(tokens)
        mask = torch.full((rows, MAX_CACHE + rows), MASKED)
        mask[:, :position] = 0.0
        for row in range(rows):
            mask[row, MAX_CACHE : MAX_CACHE + row + 1] = 0.0
        inputs = {
            "token_id": torch.tensor(tokens, dtype=torch.int32),
            "position_id": torch.arange(position, position + rows, dtype=torch.int32),
            "attention_mask": mask,
            "input_embed": torch.from_numpy(np.asarray(embeds, np.float32)),
            "use_input_embed": torch.tensor(flags, dtype=torch.float32),
        }
        bound = {}
        for name, value in inputs.items():
            bound[name] = value.cuda()
            if -1 in tuple(self.engine.get_tensor_shape(name)):
                assert self.context.set_input_shape(name, tuple(value.shape))
            self.context.set_tensor_address(name, bound[name].data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name in self.caches:
                    self.context.set_tensor_address(name, self.caches[name].data_ptr())
                continue
            dtype = (
                torch.float16
                if self.engine.get_tensor_dtype(name) == trt.float16
                else torch.float32
            )
            outputs[name] = torch.empty(
                tuple(self.context.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
            self.context.set_tensor_address(name, outputs[name].data_ptr())
        assert self.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()
        for name, value in outputs.items():
            if name.startswith("present_"):
                cache = self.caches[name.replace("present_", "cache_")]
                cache[position : position + rows] = value.reshape(rows, -1)
        return outputs["logits"].reshape(-1, outputs["logits"].shape[-1])[-1].float().cpu().numpy()


def _caches(engine) -> dict:
    caches = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if name.startswith("cache_"):
            dtype = torch.float16 if engine.get_tensor_dtype(name) == trt.float16 else torch.float32
            caches[name] = torch.zeros(
                tuple(engine.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
    return caches


@pytest.mark.gpu
@pytest.mark.trt
def test_prefill_with_audio_rows_matches_token_steps(tmp_path: Path) -> None:
    """One prefill launch over text and audio rows leaves the logits and KV rows of single steps."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = _write_checkpoint(tmp_path)
    config.raw["_model_dir"] = str(tmp_path)
    config.raw["_fp32_layers"] = ()
    plugin = model.GlmAsrPlugin()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    def build(role: str):
        config.raw["_decoder_engine_role"] = role
        weights = plugin.load_weights(str(tmp_path), config, precision="fp32")
        return runtime.deserialize_cuda_engine(
            plugin.build_engine(config, weights, MAX_CACHE, precision="fp32")
        )

    single, prefill = build("single"), build("prefill")
    assert single.num_optimization_profiles == 1
    assert prefill.num_optimization_profiles == 1
    # Without a role the decoder keeps the one-token graph that decodes split bundles.
    config.raw.pop("_decoder_engine_role")
    weights = plugin.load_weights(str(tmp_path), config, precision="fp32")
    default = runtime.deserialize_cuda_engine(
        plugin.build_engine(config, weights, MAX_CACHE, precision="fp32")
    )
    assert tuple(default.get_tensor_shape("token_id")) == (1,)

    rng = np.random.default_rng(1)
    tokens = [3, 17, 5, 9, 30, 2, 11, 21, 7]
    flags = [0, 0, 1, 1, 1, 1, 0, 0, 0]  # rows 2..5 carry projected audio frames
    embeds = rng.standard_normal((len(tokens), D)).astype(np.float32) * 0.5
    next_token = 8

    want_caches = _caches(single)
    stepper = _Stepper(single, 0, want_caches)
    for position, token in enumerate(tokens):
        want = stepper.run([token], embeds[position : position + 1], [flags[position]], position)
    want_next = stepper.run([next_token], np.zeros((1, D)), [0], len(tokens))

    got_caches = _caches(prefill)
    got = _Stepper(prefill, 0, got_caches).run(tokens, embeds, flags, 0)
    got_next = _Stepper(single, 0, got_caches).run([next_token], np.zeros((1, D)), [0], len(tokens))

    errors = {}
    for label, value, reference in (("prefill", got, want), ("decode", got_next, want_next)):
        assert np.argmax(value) == np.argmax(reference), label
        errors[label] = np.abs(value - reference).max() / np.abs(reference).max()
    for name, reference in want_caches.items():
        reference = reference.float().cpu().numpy()[: len(tokens) + 1]
        value = got_caches[name].float().cpu().numpy()[: len(tokens) + 1]
        errors[name] = np.abs(value - reference).max() / max(np.abs(reference).max(), 1e-6)
    # FP32 engines run matmuls in TF32; one row and several rows take different kernels.
    assert max(errors.values()) < 5e-3, errors
