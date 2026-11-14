# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV prefill engine: multi-row chunks against token-by-token steps of the one-token engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
if not hasattr(trt, "Builder"):  # another test module put a stand-in module in sys.modules
    pytest.skip("needs TensorRT", allow_module_level=True)

from .. import model, wkv_rows  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from .test_quantization import HIDDEN, LAYERS, VOCAB, _write_tiny_rwkv  # noqa: E402

_STATES = ("attn", "ff", "num", "den", "max")


def _wide_mixing(seed: int) -> dict:
    """Decays and bonuses over the range of trained checkpoints, and keys large enough to need the shift."""
    rng = np.random.default_rng(seed)
    extra = {}
    for layer in range(LAYERS):
        prefix = f"rwkv.blocks.{layer}.attention"
        extra[f"{prefix}.time_decay"] = rng.uniform(-5.0, 2.0, HIDDEN).astype(np.float32)
        extra[f"{prefix}.time_first"] = rng.uniform(-2.0, 3.0, HIDDEN).astype(np.float32)
        extra[f"{prefix}.key.weight"] = (rng.standard_normal((HIDDEN, HIDDEN)) * 0.5).astype(
            np.float32
        )
    return extra


def _state(engine) -> dict:
    return {
        engine.get_tensor_name(i): torch.zeros(
            tuple(engine.get_tensor_shape(engine.get_tensor_name(i))),
            dtype=torch.float32,
            device="cuda",
        )
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_name(i).split("_state")[0] in _STATES
        and "_state_" in engine.get_tensor_name(i)
    }


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
            kind, layer = name[len("present_") :].rsplit("_", 1)
            state[f"{kind}_state_{layer}"].copy_(value)
    return outputs["logits"][-1].cpu().numpy()


@pytest.mark.gpu
@pytest.mark.trt
def test_prefill_engine_matches_one_token_engine(tmp_path: Path) -> None:
    """Prefill chunks leave the logits and recurrent state of token-by-token steps."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _write_tiny_rwkv(tmp_path, seed=3, extra=_wide_mixing(4))
    config = ModelConfig.from_dir(tmp_path)
    instance = model._RwkvModel()
    weights = instance.load_weights(str(tmp_path), config)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    decoder = runtime.deserialize_cuda_engine(instance.build_engine(config, weights, 128))
    prefill = runtime.deserialize_cuda_engine(
        instance.build_engine(config, weights, 128, prefill_rows=wkv_rows.MAX_ROWS)
    )
    assert [tuple(s) for s in prefill.get_tensor_profile_shape("token_id", 0)] == [
        (1,),
        (model.PREFILL_OPT_TOKENS,),
        (wkv_rows.MAX_ROWS,),
    ]

    rng = np.random.default_rng(1)
    prompt = rng.integers(0, VOCAB, 75).tolist()  # two chunks: 64 + 11 rows
    following = rng.integers(0, VOCAB, 3).tolist()
    want_state = _state(decoder)
    want = [_run(decoder, want_state, [token]) for token in prompt][-1:]
    got_state = _state(decoder)
    got = None
    for start in range(0, len(prompt), wkv_rows.MAX_ROWS):
        got = _run(prefill, got_state, prompt[start : start + wkv_rows.MAX_ROWS])
    got = [got]
    # Decode on the one-token engine from both states.
    for token in following:
        want.append(_run(decoder, want_state, [token]))
        got.append(_run(decoder, got_state, [token]))

    def relative(value, reference):
        return float(np.abs(value - reference).max() / max(np.abs(reference).max(), 1e-6))

    errors = {}
    for step, (value, reference) in enumerate(zip(got, want)):
        assert np.argmax(value) == np.argmax(reference), step
        errors[f"logits[{step}]"] = relative(value, reference)
    for layer in range(LAYERS):
        for kind in ("attn", "ff"):
            name = f"{kind}_state_{layer}"
            errors[name] = relative(got_state[name].cpu().numpy(), want_state[name].cpu().numpy())
        # num and den may carry another scale (max); their ratio may not.
        ratio = {
            key: (state[f"num_state_{layer}"] / state[f"den_state_{layer}"]).cpu().numpy()
            for key, state in (("got", got_state), ("want", want_state))
        }
        errors[f"num/den_{layer}"] = relative(ratio["got"], ratio["want"])
    # FP32 engines run matmuls in TF32; one row and several rows take different kernels.
    assert max(errors.values()) < 5e-3, errors


@pytest.mark.trt
def test_prefill_engine_rejects_unsupported_builds(tmp_path: Path) -> None:
    """Prefill engines are single-GPU, unquantized, bounded by the distance table, without debug outputs."""
    _write_tiny_rwkv(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    instance = model._RwkvModel()
    weights = instance.load_weights(str(tmp_path), config)
    with pytest.raises(ValueError, match="prefill_rows"):
        instance.build_engine(config, weights, 128, prefill_rows=wkv_rows.MAX_ROWS + 1)
    with pytest.raises(NotImplementedError, match="prefill"):
        instance.build_engine(config, weights, 128, prefill_rows=8, debug_layer_outputs=True)
