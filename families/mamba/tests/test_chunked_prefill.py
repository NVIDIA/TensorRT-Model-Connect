# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba prefill engine: chunked selective scan against the one-token engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from .. import model, scan_rows  # noqa: E402
from ..config import ModelConfig  # noqa: E402

VOCAB, HIDDEN, LAYERS, STATE, KERNEL, DT_RANK = 48, 64, 2, 16, 4, 8
INNER = 2 * HIDDEN


def _write_checkpoint(path: Path) -> ModelConfig:
    rng = np.random.default_rng(0)

    def w(*shape, scale=0.1):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    tensors = {
        "backbone.embeddings.weight": w(VOCAB, HIDDEN, scale=1.0),
        "backbone.norm_f.weight": np.ones(HIDDEN, np.float32),
        "lm_head.weight": w(VOCAB, HIDDEN),
    }
    for layer in range(LAYERS):
        base = f"backbone.layers.{layer}"
        tensors.update(
            {
                f"{base}.norm.weight": np.ones(HIDDEN, np.float32),
                f"{base}.mixer.in_proj.weight": w(2 * INNER, HIDDEN),
                f"{base}.mixer.conv1d.weight": w(INNER, 1, KERNEL, scale=0.5),
                f"{base}.mixer.conv1d.bias": w(INNER),
                f"{base}.mixer.x_proj.weight": w(DT_RANK + 2 * STATE, INNER),
                f"{base}.mixer.dt_proj.weight": w(INNER, DT_RANK),
                f"{base}.mixer.dt_proj.bias": w(INNER, scale=1.0),
                # Decays from slow (A ~ -0.1) to fast (A ~ -16), as in real checkpoints.
                f"{base}.mixer.A_log": np.log(
                    np.tile(np.linspace(0.1, 16.0, STATE), (INNER, 1))
                ).astype(np.float32),
                f"{base}.mixer.D": np.ones(INNER, np.float32),
                f"{base}.mixer.out_proj.weight": w(HIDDEN, INNER),
            }
        )
    path.mkdir(parents=True, exist_ok=True)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mamba",
                "architectures": ["MambaForCausalLM"],
                "vocab_size": VOCAB,
                "hidden_size": HIDDEN,
                "intermediate_size": INNER,
                "num_hidden_layers": LAYERS,
                "state_size": STATE,
                "conv_kernel": KERNEL,
                "time_step_rank": DT_RANK,
                "expand": 2,
                "layer_norm_epsilon": 1e-5,
            }
        )
    )
    return ModelConfig.from_dir(path)


def _run(engine, state: dict, tokens: list[int]) -> np.ndarray:
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream().cuda_stream
    token = torch.tensor(tokens, dtype=torch.int32).cuda()
    if -1 in tuple(engine.get_tensor_shape("token_id")):
        assert context.set_input_shape("token_id", (len(tokens),))
    context.set_tensor_address("token_id", token.data_ptr())
    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if name in state:
                context.set_tensor_address(name, state[name].data_ptr())
            continue
        outputs[name] = torch.empty(
            tuple(context.get_tensor_shape(name)), dtype=torch.float32, device="cuda"
        )
        context.set_tensor_address(name, outputs[name].data_ptr())
    assert context.execute_async_v3(stream)
    torch.cuda.synchronize()
    for name, value in outputs.items():
        if name.startswith("present_"):
            state[
                name.replace("present_conv", "conv_state").replace("present_ssm", "ssm_state")
            ].copy_(value)
    return outputs["logits"][-1].cpu().numpy()


def _state(engine) -> dict:
    return {
        engine.get_tensor_name(i): torch.zeros(
            tuple(engine.get_tensor_shape(engine.get_tensor_name(i))),
            dtype=torch.float32,
            device="cuda",
        )
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_name(i).startswith(("conv_state", "ssm_state"))
    }


@pytest.mark.gpu
@pytest.mark.trt
def test_prefill_engine_matches_one_token_engine(tmp_path: Path) -> None:
    """Prefill chunks leave the logits and recurrent state of token-by-token steps."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = _write_checkpoint(tmp_path)
    instance = model._MambaModel()
    weights = instance.load_weights(str(tmp_path), config)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    decoder = runtime.deserialize_cuda_engine(instance.build_engine(config, weights, 64))
    prefill = runtime.deserialize_cuda_engine(
        instance.build_engine(config, weights, 64, prefill_rows=scan_rows.MAX_ROWS)
    )
    assert [tuple(s) for s in prefill.get_tensor_profile_shape("token_id", 0)] == [
        (1,),
        (min(model.PREFILL_OPT_TOKENS, scan_rows.MAX_ROWS),),
        (scan_rows.MAX_ROWS,),
    ]

    rng = np.random.default_rng(1)
    prompt = rng.integers(0, VOCAB, 75).tolist()  # two chunks: 64 + 11 rows
    want_state = _state(decoder)
    for token in prompt:
        want = _run(decoder, want_state, [token])
    got_state = _state(decoder)
    got = None
    for start in range(0, len(prompt), scan_rows.MAX_ROWS):
        got = _run(prefill, got_state, prompt[start : start + scan_rows.MAX_ROWS])

    errors = {"logits": np.abs(got - want).max() / np.abs(want).max()}
    for name, reference in want_state.items():
        reference = reference.cpu().numpy()
        errors[name] = np.abs(got_state[name].cpu().numpy() - reference).max() / max(
            np.abs(reference).max(), 1e-6
        )
    assert np.argmax(got) == np.argmax(want)
    # FP32 engines run matmuls in TF32; one row and several rows take different kernels.
    assert max(errors.values()) < 5e-3, errors


@pytest.mark.trt
def test_prefill_engine_rejects_unsupported_builds(tmp_path: Path) -> None:
    """Prefill engines are single-GPU, bounded by the scan size and without debug outputs."""
    config = _write_checkpoint(tmp_path)
    instance = model._MambaModel()
    weights = instance.load_weights(str(tmp_path), config)
    with pytest.raises(ValueError, match="prefill_rows"):
        instance.build_engine(config, weights, 64, prefill_rows=scan_rows.MAX_ROWS + 1)
    with pytest.raises(NotImplementedError, match="prefill"):
        instance.build_engine(config, weights, 64, prefill_rows=8, debug_layer_outputs=True)
