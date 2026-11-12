# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine numerics against the reference: cross-attention states and teacher-forced logits.

The family's two engines are built in this process from the case checkpoint
and fed the reference's own pixel values. The encoder engine's
``cross_key``/``cross_value`` are compared with every decoder layer's
cross-attention projections of the reference encoder output; the decoder
engine is then stepped through the reference's greedy tokens, writing its
key/value rows into the cache as the runtime does, and its logits at every
position are compared with the reference's teacher-forced logits.
"""

from __future__ import annotations

import math

from .common import CASES, Reference, asset, model_dir, parametrize_cases

# Measured on the five line images the case started with (B300, TensorRT 11.4,
# Transformers 5.2.0 on the same GPU with TF32 off); the sixth, a tall canvas
# added later, lands inside the correct band (6.2e-05 and 4.7e-05):
#
# ==========================  ======================  ==================
# engines                     cross K/V (relative)    logits (absolute)
# ==========================  ======================  ==================
# correct                     2.9e-05 to 2.0e-04      3.2e-05 to 1.5e-04
# reference on the CPU        4.4e-05 to 4.0e-04      2.8e-05 to 9.9e-05
# decoder: tanh GELU          (unchanged)             2.5e-03 to 3.3e-03
# decoder: layer norm 1e-12   (unchanged)             1.1e-03 to 3.0e-03
# encoder: layer norm 1e-5    1.8e-04 to 6.8e-04      1.7e-04 to 7.6e-04
# encoder: tanh GELU          5.9e-02 to 2.0e-01      5.0e-02 to 1.6e-01
# ==========================  ======================  ==================
#
# Every one of these defects still produces the reference's greedy tokens on
# every image, so the token-exact E2E cannot see them; these gates can. The
# engines are as close to the GPU reference as the reference's own CPU run is.
# The cross-attention gate is relative to the largest reference value and is a
# structural check (5x the correct worst case); the logit gate sits 3.4x above
# the correct worst case and below every decoder defect and the worst encoder
# layer-norm case.
_CROSS_TOLERANCE = {"fp32": 1e-3}
_LOGIT_TOLERANCE = {"fp32": 5e-4}


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        parametrize_cases(metafunc, "case_name")
    if "fp32_case_name" in metafunc.fixturenames:
        parametrize_cases(metafunc, "fp32_case_name",
                          keep=lambda manifest: manifest["precision"] == "fp32")


class _Engines:
    """The family's encoder and decoder plans, run directly with TensorRT."""

    def __init__(self, torch, encoder_plan: bytes, decoder_plan: bytes) -> None:
        import tensorrt as trt

        self.torch = torch
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.encoder = runtime.deserialize_cuda_engine(encoder_plan)
        self.decoder = runtime.deserialize_cuda_engine(decoder_plan)
        self.dtypes = {trt.float32: torch.float32, trt.int32: torch.int32}
        # Cache rows, [layers, heads, capacity, head_dim]: the runtime's default token limit.
        self.capacity = int(self.decoder.get_tensor_shape("cache_key")[2])

    def _tensor(self, engine, name: str):
        shape = tuple(engine.get_tensor_shape(name))
        return self.torch.zeros(shape, dtype=self.dtypes[engine.get_tensor_dtype(name)],
                                device="cuda")

    def _run(self, engine, context, tensors: dict) -> None:
        for name, tensor in tensors.items():
            assert tensor.is_contiguous()
            context.set_tensor_address(name, tensor.data_ptr())
        stream = self.torch.cuda.current_stream()
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    def encode(self, pixel_values):
        context = self.encoder.create_execution_context()
        tensors = {name: self._tensor(self.encoder, name) for name in ("cross_key", "cross_value")}
        tensors["pixel_values"] = pixel_values.to(self.torch.float32).contiguous()
        self._run(self.encoder, context, tensors)
        return tensors["cross_key"], tensors["cross_value"]

    def teacher_forced_logits(self, cross_key, cross_value, tokens: list[int]):
        context = self.decoder.create_execution_context()
        tensors = {name: self._tensor(self.decoder, name)
                   for name in ("token_id", "position", "cache_key", "cache_value", "new_key",
                                "new_value", "logits")}
        tensors["cross_key"], tensors["cross_value"] = cross_key, cross_value
        rows = []
        for position, token in enumerate(tokens):
            tensors["token_id"].fill_(token)
            tensors["position"].fill_(position)
            self._run(self.decoder, context, tensors)
            rows.append(tensors["logits"][0].clone())
            # The runtime writes the token's rows into the cache at its position.
            tensors["cache_key"][:, :, position] = tensors["new_key"][:, :, 0]
            tensors["cache_value"][:, :, position] = tensors["new_value"][:, :, 0]
        return self.torch.stack(rows)


def _reference_cross(reference: Reference, pixel_values):
    """Every decoder layer's cross-attention key/value projection of the encoder output."""
    torch = reference.torch
    with torch.no_grad():
        states = reference.model.encoder(pixel_values=pixel_values).last_hidden_state[0]
        keys, values = [], []
        for layer in reference.model.decoder.model.decoder.layers:
            heads = layer.encoder_attn.num_heads
            for projection, out in ((layer.encoder_attn.k_proj, keys),
                                    (layer.encoder_attn.v_proj, values)):
                rows = projection(states)
                out.append(rows.view(rows.shape[0], heads, -1).permute(1, 0, 2))
    return torch.stack(keys), torch.stack(values)


def _reference_logits(reference: Reference, pixel_values, tokens: list[int]):
    torch = reference.torch
    with torch.no_grad():
        decoder_input_ids = torch.tensor([tokens], device="cuda")
        output = reference.model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids,
                                 use_cache=False)
    return output.logits[0].float()


def _measure(engines: _Engines, reference: Reference, image) -> dict:
    """Largest absolute differences from the reference on one line image."""
    pixel_values = reference.pixel_values(image)
    cross_key, cross_value = engines.encode(pixel_values)
    expected_key, expected_value = _reference_cross(reference, pixel_values)
    # Bounded like the runtime: every fed token needs a cache row.
    generated = reference.generate(image, max_new_tokens=engines.capacity)
    tokens = [reference.model.generation_config.decoder_start_token_id, *generated[:-1]]
    logits = engines.teacher_forced_logits(cross_key, cross_value, tokens)
    expected_logits = _reference_logits(reference, pixel_values, tokens)
    def relative(got, want):
        return float((got.float() - want).abs().max()) / float(want.abs().max())

    return {
        "cross": max(relative(cross_key, expected_key), relative(cross_value, expected_value)),
        "logits": float((logits.float() - expected_logits).abs().max()),
        "steps": len(tokens),
        "argmax": bool(logits.argmax(-1).tolist() == [*generated]),
    }


def _engines(reference: Reference, checkpoint, manifest: dict, patch=None) -> _Engines:
    from .. import config as config_module, model

    cfg = config_module.resolve(checkpoint)
    weights = model._Weights(model._open_checkpoint(checkpoint))
    assert manifest["precision"] == "fp32"
    capacity = int(manifest.get("max_sequence_length") or cfg.decoder.max_positions)
    encoder = model.build_encoder_engine(weights, cfg)
    if patch is not None:
        patch()
    decoder = model.build_decoder_engine(weights, cfg, capacity)
    return _Engines(reference.torch, encoder, decoder)


def test_engines_match_the_reference(case_name: str) -> None:
    _, manifest, case = CASES[case_name]
    checkpoint = model_dir(manifest["hf_id"], manifest.get("hf_revision"))
    reference = Reference(checkpoint)
    engines = _engines(reference, checkpoint, manifest)
    precision = manifest["precision"]
    for raw in case["test_images"]:
        measured = _measure(engines, reference, asset(raw))
        print(f"trocr {precision} {raw}: cross relative {measured['cross']:.3e}, logits maxabs "
              f"{measured['logits']:.3e} over {measured['steps']} steps")
        assert measured["cross"] <= _CROSS_TOLERANCE[precision], raw
        assert measured["logits"] <= _LOGIT_TOLERANCE[precision], raw
        assert measured["argmax"], f"{raw}: teacher-forced argmax differs from the reference"


def _tanh_gelu(network, inp, dtype=None):
    """The defect under test: the tanh approximation in place of the exact GELU."""
    import numpy as np
    import tensorrt as trt

    from .. import graph

    dtype = np.float32 if dtype is None else dtype
    rank = len(inp.shape)

    def op(left, right, kind):
        return network.add_elementwise(left, right, kind).get_output(0)

    def constant(value):
        return graph.add_scalar(network, rank, value, dtype)

    prod, total = trt.ElementWiseOperation.PROD, trt.ElementWiseOperation.SUM
    cube = op(op(inp, inp, prod), inp, prod)
    inner = op(constant(math.sqrt(2.0 / math.pi)),
               op(inp, op(constant(0.044715), cube, prod), total), prod)
    tanh = network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
    return op(op(constant(0.5), inp, prod), op(constant(1.0), tanh, total), prod)


def test_the_logit_gate_rejects_a_tanh_gelu_decoder(fp32_case_name: str, monkeypatch) -> None:
    """The weakest decoder defect checked: the gate must see it on the case images."""
    from .. import graph

    _, manifest, case = CASES[fp32_case_name]
    checkpoint = model_dir(manifest["hf_id"], manifest.get("hf_revision"))
    reference = Reference(checkpoint)
    engines = _engines(reference, checkpoint, manifest,
                       patch=lambda: monkeypatch.setattr(graph, "add_gelu", _tanh_gelu))
    least = min(_measure(engines, reference, asset(raw))["logits"]
                for raw in case["test_images"])
    print(f"trocr fp32 tanh-GELU decoder: smallest logits maxabs over the images {least:.3e}")
    # Every image sees it, not only the most sensitive one.
    assert least > 2 * _LOGIT_TOLERANCE["fp32"]
